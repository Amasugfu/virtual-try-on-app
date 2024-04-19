import bpy
import numpy as np
import torch
from sklearn.preprocessing import normalize
import open3d as o3d

from trimesh.triangles import points_to_barycentric

from typing import Tuple
    
def compute_closest_position(src_mesh, tgt_mesh):
    t_src = o3d.t.geometry.TriangleMesh.from_legacy(src_mesh)
    queries = np.asarray(tgt_mesh.vertices)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_src)

    result = scene.compute_closest_points(queries)
    positions = result["points"]
    distances = np.linalg.norm(positions.numpy() - queries)
    face_ids = result["primitive_ids"].numpy()
    face_normals = result["primitive_normals"]
    
    uv = result["primitive_uvs"].numpy()
    barycentric_weights = np.zeros_like(queries)
    barycentric_weights[:, 1] = uv[:, 0]
    barycentric_weights[:, 2] = uv[:, 1]
    barycentric_weights[:, 0] = 1 - barycentric_weights.sum(axis=-1)
    
    return positions, distances, barycentric_weights, face_ids, face_normals


def compute_angle(normals, directions):
    directions = normalize(directions, axis=1)
    angles = np.arccos(np.abs(np.dot(normals, directions))) # the angle can at most deviate by 90 degrees
    return angles


def compute_threshold_distance(src_mesh, ratio):
    d = src_mesh.get_max_bound() - src_mesh.get_min_bound()
    return d * ratio


def find_match(src_mesh, tgt_mesh, thres_bb_ratio=0.05, thres_angle=35, radian=False):
    if not radian:
        thres_angle = np.deg2rad(thres_angle)

    position, distance, barycentric_weights, face_ids, face_normals = compute_closest_position(src_mesh, tgt_mesh)
    angles = compute_angle(face_normals, position - np.asarray(src_mesh.vetices))
    bbd = compute_threshold_distance(tgt_mesh, thres_bb_ratio)

    filter_ = (distance < bbd) & (angles < thres_angle)
    return filter_, barycentric_weights[filter_], face_ids[filter_]


def copy_weights(src_mesh, src_weights, tgt_weights, match, bary_weights, face_ids):
    faces = np.asarray(src_mesh.triangles)[face_ids]
    weights = np.take_along_axis(src_weights, faces, axis=0)
    tgt_weights[match] = weights * bary_weights