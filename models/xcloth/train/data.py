
from dataclasses import dataclass
from typing import List, Tuple, Any
from trimesh import Trimesh

from .preprocessing import process_garments, process_poses
from ..settings.model_settings import DEFAULT_XCLOTH_SETTINGS

import os, glob
import pickle
import torch
import numpy as np


@dataclass
class MeshData:
    path: str
    mesh: Trimesh
    img: Any
    peelmap_depth: List[Any]
    peelmap_norm: List[Any]
    peelmap_rgb: List[Any]


class DataLoader:
    def __init__(self, settings=DEFAULT_XCLOTH_SETTINGS) -> None:
        self.__registered_pose = {}
        self.__registered_mesh = {}
        self.__settings = settings
        self.__TARGET_FUNC_MAP = {
            "input": self.__process_poses,
            "truth": self.__process_garments,
        }

    @property
    def stats(self):
        return {
            "registered pose": len(self.__registered_pose),
            "registered mesh": len(self.__registered_mesh),
            "common keys": self.__registered_pose.keys() & self.__registered_mesh.keys(),
            "extra pose": self.__registered_pose.keys() - self.__registered_mesh.keys(),
            "extra mesh": self.__registered_mesh.keys() - self.__registered_pose.keys(),
        }
    
    @property
    def pose(self):
        return self.__registered_pose
    
    @property
    def mesh(self):
        return self.__registered_mesh
    
    def __getitem__(self, i):
        return self.__registered_pose[i], self.__registered_mesh[i]

    def __process_poses(self, in_dir, out_dir, whitelist, no_replace, verbose, log_file, smpl_path):
        for garment_id in os.listdir(in_dir):
            for pose_file in os.listdir(os.path.join(in_dir, garment_id)):
                # skip files that has been processed
                name = pose_file[:-4]
                if (no_replace and name in no_replace) or (whitelist and name not in whitelist):
                    if verbose: print(f"skipped - {name}")
                    continue

                if verbose: print(f"processing - {name}")

                # pose_file = 1-1.pkl, 1-2.pkl, etc...
                # process pose                
                data = process_poses(
                    os.path.join(in_dir, garment_id, pose_file),
                    smpl_path,
                    (self.__settings.input_h, self.__settings.input_w),
                    max_hits=self.__settings.n_peelmaps
                )

                # save processed result
                self.__registered_pose[name] = data
                self.save(data, out_dir, name, verbose=verbose, log_file=log_file)

    def __process_garments(self, in_dir, out_dir, whitelist, no_replace, verbose, log_file, **kwargs):
        for garment_id in os.listdir(in_dir):
            # skip files that has been processed
            if (no_replace and garment_id in no_replace) or (whitelist and garment_id not in whitelist): 
                if verbose: print(f"skipped - {garment_id}")
                continue

            # find the obj file
            for file in os.listdir(os.path.join(in_dir, garment_id)):
                if file == "model_cleaned.obj":
                    if verbose: print(f"processing - {garment_id}")

                    # process model
                    data = process_garments(
                        os.path.join(in_dir, garment_id, file),
                        (self.__settings.input_h, self.__settings.input_w),
                        max_hits=self.__settings.n_peelmaps
                    )

                    # save processed result
                    self.__registered_mesh[garment_id] = data
                    self.save(data, out_dir, garment_id, verbose=verbose, log_file=log_file)

                    break

    def process_data(self, in_dir, out_dir, whitelist=None, verbose=False, log_file=None, no_replace=True, target="truth", **kwargs):
        if log_file is not None:
            log_file = open(os.path.join(out_dir, log_file), "a")

        if no_replace:
            no_replace = set(f[:-4] for f in os.listdir(out_dir))

        if whitelist is not None:
            with open(whitelist, "r") as f:
                whitelist = set(s.strip()[:-4] for s in f.readlines())

        self.__TARGET_FUNC_MAP[target](in_dir, out_dir, whitelist, no_replace, verbose, log_file, **kwargs)

        if log_file is not None:
            log_file.close()

    def save(self, data, dir_path, name, verbose=False, log_file=None):
        with open(os.path.join(dir_path, name + ".pkl"), "wb") as file:
            pickle.dump(data, file)

        # print progress
        if verbose:
            print(f"saved - {name}")

        # log
        if log_file is not None:
            log_file.write(f"{name}\n")
            log_file.flush()

    def load_all(self, root_dir, pose_dir="pose", mesh_dir="mesh"):
        self.load_dir(root_dir, pose_dir, self.__registered_pose)
        self.load_dir(root_dir, mesh_dir, self.__registered_mesh)

    @staticmethod
    def load_dir(root_dir, sub_dir, target):
        """
        load pose peelmaps, which is a list of dict of H x W matrix

        the data should be store as root_dir/sub_dir/name.pkl
        """
        for filename in glob.iglob(f"{root_dir}/{sub_dir}/*.pkl"):
            with open(filename, "rb") as file:
                data = pickle.load(file)
                target[filename.replace('\\', '/').split('/')[-1][:-4]] = data

    def make_tensors(self, batch):
        """
        tranform the data into tensors of dimension N x B x P x C x H x W

        only data with both pose and mesh will be transformed
        
        N: total number of data
        B: batch size
        P: number of peeled layers

        @return: pose, depth, norm, rgb
        """
        keys = self.__registered_pose.keys() & self.__registered_mesh.keys()

        pose = np.stack([np.stack(self.__registered_pose[i]) for i in keys])
        pose = torch.from_numpy(pose).reshape(-1, batch, *pose.shape[1:])
        
        mesh = [
            (
                np.moveaxis((m := self.__registered_mesh[i]).img, -1, 0),
                np.stack(m.peelmap_depth), 
                np.stack(m.peelmap_norm), 
                np.stack(m.peelmap_rgb)
            ) 
                for i in keys
        ]
        mesh = zip(*mesh)
        mesh = [torch.from_numpy((tmp := np.stack(i))).reshape(-1, batch, *tmp.shape[1:]) for i in mesh]

        return pose, *mesh

    def make_Xy(self, pose, img, depth, norm, rgb, scale_rgb=True, cuda=True, dtype=torch.float32):
        """
        @param: pose: N x B x P x H x W
        @param: img: N x B x 3 x H x W
        @param: depth: N x B x P x H x W
        @param: norm: N x B x P x 3 x H x W
        @param: rgb: N x B x P x 3 x H x W

        @return: (X, y)

        X: N x B x (3 + P) x H x W
        y: N x B x P x (1 + 3 + 3) x H x W
        """
        if scale_rgb:
            img = img / 255
            rgb = rgb / 255

        identity = lambda x: x.to(dtype=dtype).cuda() if cuda else x.to(dtype=dtype)

        X = identity(torch.concatenate([img, pose], dim=2))
        rgb = rgb[:, :, 1:]

        return X, {
            "Depth": identity(depth).unsqueeze(dim=3),
            "Norm": identity(norm),
            "RGB": identity(rgb)
        }
