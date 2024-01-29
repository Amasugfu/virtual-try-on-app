
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Set
from trimesh import Trimesh

from .preprocessing import process_garments, process_poses
from ..settings.model_settings import DEFAULT_XCLOTH_SETTINGS

import os, glob
import pickle
import torch
from torch.utils.data import Dataset

import numpy as np


def load_dir(root_dir, sub_dir, target, mask=None, excld=True):
    """
    load pose peelmaps, which is a list of dict of H x W matrix

    the data should be store as root_dir/sub_dir/name.pkl

    @param: target: storage dictionary
    @param: mask: white/blacklist
    @param: exclud: `True` if the mask is a blacklist and `False` if the mask is whitelist
    """
    for filename in glob.iglob(f"{root_dir}/{sub_dir}/*-1.pkl"):
        name = filename.replace('\\', '/').split('/')[-1][:-4]

        if mask is not None:
            if excld:
                if name in mask: continue
            elif name not in mask: continue

        target[name] = filename


@dataclass
class MeshData:
    path: str
    mesh: Trimesh
    img: Any
    peelmap_depth: List[Any]
    peelmap_norm: List[Any]
    peelmap_rgb: List[Any]


class MeshDataSet(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 pose_dir: str = "pose", 
                 mesh_dir: str = "mesh",
                 mask: Set[str]|None = None, 
                 excld: bool = True,
                 scale_rgb: bool = True, 
                 dtype=torch.float32, 
                 depth_offset: float = 0.,):
        """
        @param: root_dir: path to the root directory.
        @param: pose_dir: name of the directory storing processed pose pickles.
        @param: mesh_dir: name of the directory storing processed mesh pickles.
        @param: mask: white/blacklist
        @param: exclud: `True` if the mask is a blacklist and `False` if the mask is whitelist

        @param: scale_rgb: if `True`, transform rgb to range of [0, 1]
        """
        self.__root_dir = root_dir
        self.__pose_dir = pose_dir
        self.__mesh_dir = mesh_dir
        self.__mask = mask
        self.__excld = excld
        self.__scale_rgb = scale_rgb
        self.__dtype = dtype
        self.__depth_offset = depth_offset

        self.__registered_pose = {}
        self.__registered_mesh = {}
        self.reload_all()
        self.__common_keys = list(self.__registered_pose.keys() & self.__registered_mesh.keys())

    def __len__(self):
        return len(self.__common_keys)
    
    def __getitem__(self, index) -> Any:
        X, y = self.make_Xy(self.__common_keys[index])
        return X, y["Depth"], y["Norm"], y["RGB"]

    def reload_all(self):
        load_dir(self.__root_dir, self.__pose_dir, self.__registered_pose, self.__mask, self.__excld)
        load_dir(self.__root_dir, self.__mesh_dir, self.__registered_mesh, self.__mask, self.__excld)

    @property
    def stats(self):
        return {
            "registered pose": len(self.__registered_pose),
            "registered mesh": len(self.__registered_mesh),
            "common keys": self.__registered_pose.keys() & self.__registered_mesh.keys(),
            "extra pose": self.__registered_pose.keys() - self.__registered_mesh.keys(),
            "extra mesh": self.__registered_mesh.keys() - self.__registered_pose.keys(),
        }

    def make_Xy(self, name):
        """
        tranform the data into tensors which can be fed directly to the model

        only data with both pose and mesh will be transformed

        X: (3 + P) x H x W
        y: Dict[
            depth[P x 1 x H x W], 
            norm[P x 3 x H x W], 
            rgb[P x 3 x H x W]
            ]

        Notes: 
        N: total number of data
        P: number of peeled layers
        """
        with open(self.__registered_pose[name], "rb") as f:
            p = pickle.load(f)
        with open(self.__registered_mesh[name], "rb") as f:
            m = pickle.load(f)

        # make input
        pose = torch.tensor(np.stack(p))

        pose[pose != 0] += self.__depth_offset
        
        img = np.moveaxis(m.img, -1, 0)
        if self.__scale_rgb: img = img / 255

        X = torch.concatenate([torch.from_numpy(img), pose], dim=0).to(dtype=self.__dtype)
        
        # make truth
        def __make_depth(__m):
            __depth = torch.tensor(np.stack(__m.peelmap_depth)).to(dtype=self.__dtype)
            __depth[__depth != 0] += self.__depth_offset
            return __depth.unsqueeze(dim=1)

        def __make_norm(__m):
            __norm = torch.tensor(np.stack(__m.peelmap_norm)).to(dtype=self.__dtype)
            return __norm

        def __make_rgb(__m):
            __rgb = torch.tensor(np.stack(__m.peelmap_rgb)).to(dtype=self.__dtype)[1:]
            if self.__scale_rgb: __rgb = __rgb / 255
            return __rgb

        return X, {
                "Depth": __make_depth(m), 
                "Norm": __make_norm(m), 
                "RGB": __make_rgb(m)
            } 


class DataProccessor:
    def __init__(self, settings=DEFAULT_XCLOTH_SETTINGS) -> None:
        self.__settings = settings
        self.__TARGET_FUNC_MAP = {
            "input": self.__process_poses,
            "truth": self.__process_garments,
        }

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

"""
deprecated
"""
    # def load_all(self, root_dir, pose_dir="pose", mesh_dir="mesh", **kwargs):
    #     self.load_dir(root_dir, pose_dir, self.__registered_pose, **kwargs)
    #     self.load_dir(root_dir, mesh_dir, self.__registered_mesh, **kwargs)

    # @staticmethod
    # def load_dir(root_dir, sub_dir, target, mask=None, excld=True):
    #     """
    #     load pose peelmaps, which is a list of dict of H x W matrix

    #     the data should be store as root_dir/sub_dir/name.pkl
    #     """
    #     for filename in glob.iglob(f"{root_dir}/{sub_dir}/*.pkl"):
    #         name = filename.replace('\\', '/').split('/')[-1][:-4]

    #         if mask is not None:
    #             if excld and name in mask: continue
    #             if not excld and name not in mask: continue

    #         with open(filename, "rb") as file:
    #             data = pickle.load(file)
    #             target[name] = data

    # def make_tensors(self, batch):
    #     """
    #     tranform the data into tensors of dimension N x B x P x C x H x W

    #     only data with both pose and mesh will be transformed
        
    #     N: total number of data
    #     B: batch size
    #     P: number of peeled layers

    #     @return: pose, depth, norm, rgb
    #     """
    #     keys = self.__registered_pose.keys() & self.__registered_mesh.keys()

    #     pose = np.stack([np.stack(self.__registered_pose[i]) for i in keys])
    #     pose = torch.from_numpy(pose).reshape(-1, batch, *pose.shape[1:])
        
    #     mesh = [
    #         (
    #             np.moveaxis((m := self.__registered_mesh[i]).img, -1, 0),
    #             np.stack(m.peelmap_depth), 
    #             np.stack(m.peelmap_norm), 
    #             np.stack(m.peelmap_rgb)
    #         ) 
    #             for i in keys
    #     ]
    #     mesh = zip(*mesh)
    #     mesh = [torch.from_numpy((tmp := np.stack(i))).reshape(-1, batch, *tmp.shape[1:]) for i in mesh]

    #     return pose, *mesh

    # def make_Xy(self, pose, img, depth, norm, rgb, 
    #             scale_rgb=True, 
    #             cuda=True, 
    #             dtype=torch.float32, 
    #             depth_offset=0.):
    #     """
    #     @param: pose: N x B x P x H x W
    #     @param: img: N x B x 3 x H x W
    #     @param: depth: N x B x P x H x W
    #     @param: norm: N x B x P x 3 x H x W
    #     @param: rgb: N x B x P x 3 x H x W

    #     @return: (X, y)

    #     X: N x B x (3 + P) x H x W
    #     y: N x B x P x (1 + 3 + 3) x H x W
    #     """
    #     if scale_rgb:
    #         img = img / 255
    #         rgb = rgb / 255

    #     identity = lambda x: x.to(dtype=dtype).cuda() if cuda else x.to(dtype=dtype)

    #     pose[pose != 0] += depth_offset
    #     X = identity(torch.concatenate([img, pose], dim=2))

    #     depth[depth != 0] += depth_offset
    #     return X, {
    #         "Depth": identity(depth).unsqueeze(dim=3),
    #         "Norm": identity(norm),
    #         "RGB": identity(rgb)[:, :, 1:]
    #     }
