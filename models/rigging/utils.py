import open3d as o3d
import numpy as np  
from sklearn.preprocessing import normalize

import tempfile
import os

from ..xcloth.components.utils import pose_smpl, o3d_to_skinned_glb

def compute_closest_position(src_mesh, tgt_mesh):
    t_src = o3d.t.geometry.TriangleMesh.from_legacy(src_mesh)
    queries = np.asarray(tgt_mesh.vertices, dtype=np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_src)

    result = scene.compute_closest_points(queries)
    positions = result["points"].numpy()
    distances = np.linalg.norm(positions - queries, axis=-1)
    face_ids = result["primitive_ids"].numpy()
    face_normals = result["primitive_normals"].numpy()

    uv = result["primitive_uvs"].numpy()
    barycentric_weights = np.zeros_like(queries)
    barycentric_weights[:, 1] = uv[:, 0]
    barycentric_weights[:, 2] = uv[:, 1]
    barycentric_weights[:, 0] = 1 - barycentric_weights.sum(axis=-1)

    return positions, distances, barycentric_weights, face_ids, face_normals


def composite_bary_weights(src_mesh, src, bary_weights, face_ids):
    faces = np.asarray(src_mesh.triangles)[face_ids]
    weights = src[faces.flatten()].reshape((*faces.shape, -1))
    bary_weights = normalize(bary_weights, axis=1)
    return (np.expand_dims(bary_weights, axis=-1) * weights).sum(axis=1)


def paint_mesh_to_glb(mesh, pose):
    smpl_mesh, _, _, weights = pose_smpl(pose, return_mesh=True, return_weights=True)
    from .weight_transfer_robust import transfer_weights
    try:
        tgt_weights = transfer_weights(
            smpl_mesh, mesh, weights.cpu().numpy()
        )
    except:
        tgt_weights = transfer_weights(
            smpl_mesh, mesh, weights.cpu().numpy(), copy_all=True
        )
        
    tmp_dir = tempfile.TemporaryDirectory()
    out_name = "result_rigged.glb"
    o3d_to_skinned_glb(mesh, export_folder=tmp_dir.name, weights=tgt_weights, output_name=out_name)
    
    with open(os.path.join(tmp_dir.name, out_name), "rb") as f:
        return f.read()