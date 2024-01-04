import numpy as np
import pandas as pd
import trimesh

from ..settings.model_settings import DEFAULT_XCLOTH_SETTINGS

from typing import Tuple

import pyrender
from PIL import Image

import pickle

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import torch


def project_rays(mesh, 
                 grid_dim: Tuple[int, int] = (DEFAULT_XCLOTH_SETTINGS.input_h, DEFAULT_XCLOTH_SETTINGS.input_w), 
                 fov: Tuple[float, float] = (60.0, 60.0), 
                 z: float = 1.0,
                 max_hits: int = DEFAULT_XCLOTH_SETTINGS.n_peelmaps):
    """
    project rays to the model the get the intersection infomation
    
    @return: raw peelmaps
    """
    grid = np.indices(grid_dim).swapaxes(0, 2).reshape((-1, 2))
    grid_dim = np.array(grid_dim) // 2
    grid[:, 0] -= grid_dim[0]
    grid[:, 1] -= grid_dim[1]
    
    sep = np.tan(np.deg2rad(fov) / 2) / grid_dim      # separation per pixel in real coords
    directions = np.concatenate([grid * sep, np.full((grid.shape[0], 1), -z)], axis=1)
    origins = np.zeros(directions.shape)
    origins[:, -1] = z

    # compute ray intersection
    face_ids, ray_ids, locations = mesh.ray.intersects_id(origins, directions, multiple_hits=True, max_hits=max_hits, return_locations=True)

    # sort intersection by depth
    argsort = np.argsort(locations[:, -1])[::-1]
    face_ids = face_ids[argsort]
    locations = locations[argsort]
    ray_ids = ray_ids[argsort]

    counts = pd.Series(ray_ids)
    counts = counts.groupby(counts).cumcount().to_numpy()

    # i = layer; peelmaps[i] = (world_coords, pixel_coords)
    peelmaps = [(locations[counts == i], ray_ids[counts == i], face_ids[counts == i]) for i in range(max_hits)]
    return peelmaps


def make_depth_peelmap(world_coords, 
                       row, col, 
                       dim: Tuple[int, int]):
    depth = np.zeros(dim)
    depth[row, col] = world_coords[:, -1]
    return depth


def make_rgb_peelmap(face_ids, world_coords, mesh, 
                      row, col, 
                      dim: Tuple[int, int]):
    faces = mesh.faces[face_ids]     # indices of the 3 vertices of the face
    uv = mesh.visual.uv[faces]       # 2d image coordinates of the 3 vertices
    bary = trimesh.triangles.points_to_barycentric(mesh.vertices[faces], world_coords)
    uv_hit = bary.reshape(-1, 1, 3) @ uv
    rgba = mesh.visual.material.to_color(uv_hit.reshape(-1, 2))
    rgba_img = np.zeros((*dim, 4), dtype=np.uint8)
    rgba_img[row, col] = rgba.astype(np.uint8)
    
    rgb_img = Image.fromarray(rgba_img)
    rgb_img = np.array(rgb_img.convert("RGB"))

    return np.moveaxis(rgb_img, -1, 0)


def make_normal_peelmap(face_ids, mesh, 
                        row, col, 
                        dim: Tuple[int, int]):
    normals = mesh.face_normals[face_ids]
    normal_img = np.zeros((*dim, 3))
    normal_img[row, col] = normals
    return np.moveaxis(normal_img, -1, 0)


def make_peelmaps(peelmaps,
                  mesh,
                  dim: Tuple[int, int] = (DEFAULT_XCLOTH_SETTINGS.input_h, DEFAULT_XCLOTH_SETTINGS.input_w)):
    pm_depth = []
    pm_rgb = []
    pm_normals = []

    for world_coords, pixel_coords, face_ids in peelmaps:
        row = pixel_coords // dim[0]
        col = pixel_coords % dim[1]

        pm_depth.append(make_depth_peelmap(world_coords, row, col, dim))
        pm_rgb.append(make_rgb_peelmap(face_ids, world_coords, mesh, row, col, dim))
        pm_normals.append(make_normal_peelmap(face_ids, mesh, row, col, dim))

    return pm_depth, pm_normals, pm_rgb


def render_front(mesh):
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
    scene.add(pr_mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1)
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0 ,1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    r = pyrender.OffscreenRenderer(512, 512)
    rgb, _ = r.render(scene)
    return rgb[::-1]


def process_garments(
    path: str,
    grid_dim: Tuple[int, int] = (DEFAULT_XCLOTH_SETTINGS.input_h, DEFAULT_XCLOTH_SETTINGS.input_w), 
    fov: Tuple[float, float] = (60.0, 60.0), 
    z: float = 1.0,
    max_hits: int = DEFAULT_XCLOTH_SETTINGS.n_peelmaps
):
    """
    process the obj model into [depth, rgba, normals] peelmap representation

    @param: path: the path to the obj model
    
    @return: mesh, rgb front img, depth peelmap, normal peelmap, rgb peelmap
    """

    mesh = trimesh.load(path)
    img = render_front(mesh)
    peelmaps = project_rays(mesh, grid_dim, fov, z, max_hits)
    d, n, r = make_peelmaps(peelmaps, mesh, grid_dim)

    from .data import MeshData

    return MeshData(
        path=path,
        mesh=mesh, 
        img=img, 
        peelmap_depth=d,
        peelmap_norm=n,
        peelmap_rgb=r)


def process_poses(
    path: str,
    grid_dim: Tuple[int, int] = (DEFAULT_XCLOTH_SETTINGS.input_h, DEFAULT_XCLOTH_SETTINGS.input_w), 
    fov: Tuple[float, float] = (60.0, 60.0), 
    z: float = 1.0,
    max_hits: int = DEFAULT_XCLOTH_SETTINGS.n_peelmaps
):
    """
    process the pose into peelmap representation

    @param: path: the path to the obj model
    
    @return: depth peelmaps
    """
    with open(path, "rb") as f:
        src_dict = pickle.load(f)

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='male',
        model_root='../../smpl/',
    )

    pose = src_dict['pose']
    trans = src_dict['trans']
    scale = src_dict['scale']

    fin_pose= torch.FloatTensor(pose).unsqueeze(0)
    # fin_pose = fin_pose.cuda()

    fin_shape = torch.zeros((1,10)).float()
    fin_shape = fin_shape.cuda()

    ret_verts, _ = smpl_layer(fin_pose, fin_shape)

    ret_verts = ret_verts.detach().cpu().numpy()[0]

    trans_verts = ret_verts * scale + trans

    mesh = trimesh.Trimesh(vertices=trans_verts, faces=smpl_layer.th_faces.detach().cpu().numpy())
    peelmaps = project_rays(mesh, grid_dim, fov, z, max_hits)
    pm_depth = []

    for world_coords, pixel_coords, _ in enumerate(peelmaps):
        row = pixel_coords // grid_dim[0]
        col = pixel_coords % grid_dim[1]

        pm_depth.append(make_depth_peelmap(world_coords, row, col, grid_dim))
    
    return pm_depth