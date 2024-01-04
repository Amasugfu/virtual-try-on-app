
from dataclasses import dataclass
from typing import List, Tuple, Any
from trimesh import Trimesh

from .preprocessing import process_model
from ..settings.model_settings import DEFAULT_XCLOTH_SETTINGS

import os
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
        self._register = {}
        self._settings = settings

    def load_n_process(self, in_dir):
        for name in os.listdir(in_dir):
            for file in os.listdir(os.path.join(in_dir, name)):
                if file == "model_cleaned.obj":
                    p = os.path.join(in_dir, name, file)
                    data = process_model(
                        p,
                        (self._settings.input_h, self._settings.input_w),
                        max_hits=self._settings.n_peelmaps
                    )
                    self._register[name] = data

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self._register, file)

    def load(self, path):
        with open(path, "rb") as file:
            self._register = pickle.load(file)

    def __getitem__(self, i):
        return self._register[i]
    
    