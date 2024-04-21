import torch
import numpy as np
import open3d as o3d

from ..xcloth.components.utils import pose_smpl
from .utils import compute_closest_position, composite_bary_weights

def lbs(W, T, V, pose_offsets=None, inverse=False):
    # if inverse_rotation:
    #     view = T.view(-1, 4, 4)
    #     rotation = view[:, :3, :3]
    #     # translation = view[:, :3, -1]
    #     rotation = torch.linalg.inv(rotation)
    #     view[:, :3, :3] = rotation
    #     # T[:, :3, -1] = translation
    if pose_offsets is not None:
        V = V + pose_offsets
        
    if inverse:
        T = torch.inverse(T.view(-1, 4, 4))
    
    V_homo = torch.concat([V, torch.ones((V.shape[0], 1), device=V.device)], dim=-1).unsqueeze(dim=-1)
    T = (W @ T.reshape(-1, 16)).view(-1, 4, 4)
    V_homo = T @ V_homo
    return V_homo[:, :3, 0]


def compute_V_norm(W, T, V, pose_offsets=0):
    return lbs(W, T, V) - pose_offsets


def interpolate_pose_offsets(src_mesh, tgt_mesh, pose_offsets):
    _, _, bary_weights, face_ids, _ = compute_closest_position(src_mesh, tgt_mesh)
    return composite_bary_weights(src_mesh, pose_offsets, bary_weights, face_ids)
    
    
def restore_T_pose(mesh, weights, pose, device="cuda"):
    W = torch.as_tensor(weights, device=device)
    V = torch.as_tensor(np.asarray(mesh.vertices), device=device)
    
    mesh, T, pose_offsets = pose_smpl(pose, return_faces=True)
    pose_offsets = interpolate_pose_offsets()
    
    V = compute_V_norm(W, V, T, pose_offsets)
    m = o3d.geometry.TriangleMesh(mesh)
    m.vertices = o3d.utility.Vector3dVector(V.detach().cpu().numpy())
    return m