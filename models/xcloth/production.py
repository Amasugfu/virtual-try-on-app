import numpy as np
import torch
from torch import nn
import cv2
import open3d as o3d

import tempfile

from .settings.model_settings import (
    xClothSettings,
    DEFAULT_XCLOTH_SETTINGS,
    CameraSettings,
    DEFAULT_CAMERA_SETTINGS,
)
from .components.encoder import Encoder
from .components.decoder import DepthDecoder, NormDecoder, RGBDecoder
from .components.reconstruct import GarmentModel3D

from .train.preprocessing import process_poses, render_front
from .components.utils import compute_pixsep, pose_smpl, create_o3d_mesh, create_o3d_pcd, o3d_to_skinned_glb
from .components.const import SMPL_COMMON_SCALE


class XCloth(nn.Module):
    def __init__(self, settings: xClothSettings = DEFAULT_XCLOTH_SETTINGS) -> None:
        super().__init__()
        self._settings = settings
        self._encoder = Encoder(settings)
        self._parallel_decoders = nn.ModuleDict(
            {
                "Depth": DepthDecoder(settings),
                "Norm": NormDecoder(settings),
                "RGB": RGBDecoder(settings),
            }
        )

    @property
    def n_peelmaps(self):
        return self._settings.n_peelmaps

    def forward(self, x_img: torch.Tensor, x_smpl: torch.Tensor = None):
        if x_smpl is None:
            x_smpl = self.get_smpl_prior(x_img)

        if len(x_img.shape) == 3:  # C x H x W
            x_img = x_img.unsqueeze(0)  # 1 x C x H x W

        if len(x_smpl.shape) == 3:
            x_smpl = x_smpl.unsqueeze(0)

        x = self._encoder(x_img, x_smpl)
        y = {name: decoder(x) for name, decoder in self._parallel_decoders.items()}
        y["Img"] = x_img
        return y

    def reconstruct3d(
        self, result, batch=1, mask: np.ndarray | None = None, path: str | None = None
    ):
        mesh = GarmentModel3D.from_tensor_dict(result, batch)

        mesh[0].mask = mask != 0

        mesh[0].backproject(thres=0.4, depth_offset=0.5, denoise_dist=0.01)
        m = mesh[0].to_static_mesh(
            path=path,
            smooth_iter=10,
            smooth_lambda=0.7,
            # sampler_mesh=lowpoly,
            sampler_depth=7,
            sampler_dilation=0.4,
            file_extension="glb",
        )
        return m

    def save(self, path, n=None, loss_hist=None):
        torch.save(
            {
                "epoch": n,
                "loss_hist": loss_hist,
                "state": self.state_dict(),
            },
            path,
        )

    def load(self, path):
        chkpt = torch.load(path)
        self.load_state_dict(chkpt["state"])
        return chkpt["epoch"], chkpt["loss_hist"]

    @property
    def dims(self):
        return self._settings.input_h, self._settings.input_w


class Pipeline:
    # train data scale
    # this is used for fitting the smpl model into the 512 x 512 frame
    SMPL_REFERENCE_SCALE = SMPL_COMMON_SCALE

    def __init__(
        self,
        smpl_root="models/smpl",
        model: XCloth = XCloth(),
        camera_settings: CameraSettings = DEFAULT_CAMERA_SETTINGS,
    ) -> None:
        self.smpl_root = smpl_root
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(device=self._device)
        self.camera_settings = camera_settings

    def compute_smpl_peelmaps(self, pose):
        """compute smpl priors from the given pose

        Parameters
        ----------
        x_pose : ndarray
            first 3 elements are camera position, followed by 4 euler angle value for
            z-axis of left shoulder, right shoulder, left hip and right hip
        """
        verts, joints, _, _, faces = pose_smpl(pose, return_faces=True)

        src_dict = {
            "trans": np.zeros(3),
            "scale": self.SMPL_REFERENCE_SCALE,
            "pose": pose.flatten(),
        }

        fov = (
            (self.camera_settings.fov, self.camera_settings.fov)
            if isinstance(self.camera_settings.fov, float)
            else self.camera_settings.fov
        )
        
        #############################################################
        ### debug block
        return 0, joints[0].detach().cpu().numpy()
        #############################################################

        pm_depth = process_poses(
            src_dict,
            self.smpl_root,
            fov=fov,
            z=self.camera_settings.z,
            smpl_vert=(verts - joints[0]).cpu(),
            smpl_face=faces,
        )
        return pm_depth, joints[0].detach().cpu().numpy()

    def transform_image(
        self, img, center, trans, input_scale, corner1, corner2
    ):
        size_h, size_w = self._model.dims
        size = np.array([size_w, size_h])
        sep = compute_pixsep(size, self.camera_settings.fov, self.camera_settings.z)

        def __trans_helper(c):
            c = ((c - trans) / input_scale - center) * self.SMPL_REFERENCE_SCALE
            c = size // 2 + np.round(c[:-1] / sep)
            return c.astype(int)

        corner1 = __trans_helper(corner1)
        corner2 = __trans_helper(corner2)

        img_size = np.abs(corner2 - corner1 + 1).astype(int)
        img = np.moveaxis(cv2.resize(img, img_size), -1, 0)
        padded = np.zeros([3, size_w, size_h])

        y = corner1[1]
        x = corner1[0]

        padded[:, y : y + img_size[1], x : x + img_size[0]] = img
        return padded

    def reconstruct(self, x_img, x_smpl, center):
        #############################################################
        ### debug block
        with open("debug_results/result.glb", "rb") as glb:
            v = np.load("debug_results/v.npy")
            f = np.load("debug_results/f.npy")
            c = np.load("debug_results/c.npy")
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(v),
                triangles=o3d.utility.Vector3iVector(f)
            )
            mesh.vertex_colors = o3d.utility.Vector3dVector(c)
            return glb.read(), mesh
        #############################################################
        
        x_img = torch.as_tensor(x_img, dtype=torch.float32).to(device=self._device)
        x_smpl = (
            torch.as_tensor(np.stack(x_smpl), dtype=torch.float32).to(
                device=self._device
            )
            + 0.5
        )

        tmp_dir = tempfile.TemporaryDirectory()
        with torch.no_grad():
            result = self._model(x_img, x_smpl)
            mesh = self._model.reconstruct3d(
                result=result,
                mask=np.all(x_img.cpu().numpy(), axis=0),
            )
        
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) / self.SMPL_REFERENCE_SCALE + center)
            
        o3d_to_skinned_glb(mesh, export_folder=tmp_dir.name, armature_path=None)

        with open(f"{tmp_dir.name}/result.glb", "rb") as f:
            return f.read()

    def __call__(self, x_img, x_pose, smpl_pose):       
        pm_depth, center = self.compute_smpl_peelmaps(smpl_pose)
        x_img = self.transform_image(
            x_img,
            center,
            x_pose[3:6],
            x_pose[6],
            x_pose[7:10],
            x_pose[10:13],
        )
        gltf, mesh = self.reconstruct(x_img, pm_depth, center)

        return gltf, mesh
