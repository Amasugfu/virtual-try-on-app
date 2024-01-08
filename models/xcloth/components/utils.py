import torch
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d

import copy, os
from dataclasses import dataclass
from typing import Tuple, Any, List


def create_idx_mat(size):
    return np.indices(size).swapaxes(0, 2).reshape((-1, 2))


def create_norm_coords(size, idx_mat=None):
    """
    transform pixel coordinates [0, 0] [0, 1] ... [n, m] to [-n/2, -m/2] ... [n/2, m/2]

    i.e. transform the origin from the upper corner to the center of image
    """
    assert size[0] % 2 == 0 and size[1] % 2 == 0
    if idx_mat is None:
        idx_mat = create_idx_mat(size)
    else:
        idx_mat = copy.deepcopy(idx_mat)

    idx_mat[:, 0] -= size[0] // 2
    idx_mat[:, 1] -= size[1] // 2
    # idx_mat[:, 1] *= -1
    return idx_mat


def transform_coords_norm2real(
        size: int|Tuple[int, int], 
        coords: Any, 
        z: float, 
        fov: float
):
    """
    transform pixel coordinates `[[-pix_x/2, -pix_y/2], ..., [0, 0], ..., [pix_x/2, pix_y/2]]` to 3d world (x, y) coordinates.

    @param: size: image dimension
    @param: coords: pixel coordinates matrix of dimension of N x 2
    @param: z: camera z position
    @param: fov: camera fov in radian

    @return: (real world 3D coordinates in matrix of dimension of N x 2, real world distance per pixel)
    """
    real_coords = np.zeros_like(coords, dtype=np.float64)
    def __compute_pixsep(__s, __fov):
        """
        compute real distance between each pixel.
        """
        a = z*np.tan(__fov / 2)
        return __s / a / 2

    if type(fov) == tuple:
        assert len(size) == len(fov)
        sep = [__compute_pixsep(s, f) for s, f in zip(size, fov)]
    else:
        sep = __compute_pixsep(size[0], fov)
        sep = [sep for _ in range(len(size))]

    for dim in range(len(coords.shape)):
        real_coords[:, dim] = coords[:, dim] / sep[dim]

    return real_coords, sep


# reconstruct from depth
def reconstruct_from_depth(pm_depth, thres, z, fov):
    """
    reconstruct point cloud from depth peelmaps

    @param: pm_depth: P x H x W
    @param: thres: mask threshold
    @param: z: camera z position
    @param: fov: camera fov in radian
    @param: mask: boolean matrix of dimension H x W

    @return: (iterable of point cloud of each peeled layer, row id, col id)
    """
    img_size = pm_depth.shape[1:]
    pix_coords = create_idx_mat(img_size)
    row, col = pix_coords[:, 1], pix_coords[:, 0]
    
    pix_coords_norm = create_norm_coords(img_size, idx_mat=pix_coords)
    real_coords, sep = transform_coords_norm2real(img_size, pix_coords_norm, z, fov)
    
    for depth in pm_depth:
        # flatten depth peelmaps --> N x 1
        f_d = depth[row, col].reshape(-1, 1)
        thres_mask = (np.abs(depth) > thres)[row, col]
        # reconstruct original (x, y) coordinates
        p = real_coords * (1 - f_d/z)
        p = np.hstack([p, f_d])

        yield p, thres_mask, row, col


def partial_mesh(pcds):
    return pcds


def poisson_surface_reconsturction(pcd, depth):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh


def refine_geometry(pcds, depth):
    mesh, _ = poisson_surface_reconsturction(np.sum(pcds), depth)
    return mesh


def map_texture(meshes):
    return meshes


@dataclass
class CameraSettings:
    z: float = 1
    fov: float|Tuple[float, float] = np.pi/3


@dataclass
class GarmentModel3D:
    """
    pm_depth: P x 1 x H x W
    pm_norm: P x 3 x H x W
    pm_rgb: P x 3 x H x W
    mask: H x W
    """
    img: Any
    pm_depth: Any
    # pm_seg: Any
    pm_norm: Any
    pm_rgb: Any
    mask: Any = None
    camera_settings: CameraSettings = CameraSettings()
    
    pcds: List[o3d.geometry.PointCloud]|None = None

    @classmethod
    def from_tensor_dict(cls, t_dict, B, **kwargs):
        return [
            cls(
                img=(tmp := t_dict["Img"][i].cpu().numpy()),
                pm_depth=t_dict["Depth"][i].cpu().numpy(), 
                pm_norm=t_dict["Norm"][i].cpu().numpy(), 
                pm_rgb=np.concatenate([np.expand_dims(tmp, 0), t_dict["RGB"][i].cpu().numpy()], axis=0),
                **kwargs
            )
            for i in range(B)
        ]

    def to_obj(self, depth=9, path: str|None = None):
        """
        parse to an OBJ 3D model.

        @param: depth: poisson surface reconstruction depth

        @return: triange mesh with texture
        """
        if self.pcds is None: self.reconstruct()

        mesh = refine_geometry(self.pcds, depth=depth)
        mesh = map_texture(mesh)

        if path is not None:
            o3d.io.write_triangle_mesh(path, mesh)

        return mesh

    def to_img(self):
        """
        parse each peelmaps to PNG images.
        """
        pass

    def reconstruct(self, thres: float = 5e-5, path: str|None = None):
        """
        reconstruct the model as a 3d meshes using the peelmaps.
        texture is also generated.

        @params: thres: depth below this value will be removed
        @params: path: if passed, it will save each peelamps in the specific location, which is a directory

        @return: point cloud of each peeled layer
        """
        pcds = []

        for i, (pcd, thres_mask, row_id, col_id) in enumerate(reconstruct_from_depth(self.pm_depth.squeeze(), thres, self.camera_settings.z, self.camera_settings.fov)):
            norm = np.moveaxis(self.pm_norm[i], 0, -1)[row_id, col_id]
            rgb = np.moveaxis(self.pm_rgb[i], 0, -1)[row_id, col_id]
            
            if self.mask is not None:
                thres_mask = thres_mask & self.mask[row_id, col_id]

            reconstructed_pcd = o3d.geometry.PointCloud()
            reconstructed_pcd.points = o3d.utility.Vector3dVector(pcd[thres_mask])
            reconstructed_pcd.normals = o3d.utility.Vector3dVector(norm[thres_mask])
            reconstructed_pcd.colors = o3d.utility.Vector3dVector(rgb[thres_mask])
            
            pcds.append(reconstructed_pcd)

        self.pcds = pcds
        if path is not None:
            self.save_pcd(path)

    def save_pcd(self, path: str):
        if path is not None and not os.path.isdir(path):
            os.makedirs(path)

        if self.pcds is not None:
            for i, pcd in enumerate(self.pcds):
                o3d.io.write_point_cloud(f"{path}/layer_{i}.ply", pcd)
    

            