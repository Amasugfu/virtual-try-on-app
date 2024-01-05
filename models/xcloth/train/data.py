
from dataclasses import dataclass
from typing import List, Tuple, Any
from trimesh import Trimesh

from .preprocessing import process_garments, process_poses
from ..settings.model_settings import DEFAULT_XCLOTH_SETTINGS

import os, glob
import pickle

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
        self.__registered_input = {}
        self.__registered_truth = {}
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
                self.__registered_input[name] = data
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
                    self.__registered_truth[garment_id] = data
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

    def load_all(self, dir_path):
        for filename in glob.glob("*.pkl"):
            with open(os.path.join(dir_path, filename), "rb") as file:
                self.__registered_truth[filename.split("/")[-1][:-4]] = pickle.load(file)

    def __getitem__(self, i):
        return self.__registered_truth[i]
    
    