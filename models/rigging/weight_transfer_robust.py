import numpy as np
import torch
from pytorch3d.ops import cot_laplacian
from scipy import sparse
import scipy.sparse.linalg as splinalg
from sklearn.preprocessing import normalize
import open3d as o3d

import multiprocessing
from multiprocessing import current_process

from .utils import compute_closest_position, composite_bary_weights

if current_process().name == 'MainProcess':
    import bpy


def compute_angle(normals, directions):
    normals = normalize(normals, axis=1)
    directions = normalize(directions, axis=1)
    dot = np.sum(normals * directions, axis=-1)
    angles = np.arccos(
        np.abs(np.clip(dot, -1, 1))
    )  # the angle can at most deviate by 90 degrees
    return angles


def compute_threshold_distance(src_mesh, ratio):
    d = src_mesh.get_max_bound() - src_mesh.get_min_bound()
    return np.linalg.norm(d) * ratio


def find_match(src_mesh, tgt_mesh, thres_bb_ratio=0.05, thres_angle=35, radian=False):
    if not radian:
        thres_angle = np.deg2rad(thres_angle)

    positions, distances, barycentric_weights, face_ids, face_normals = (
        compute_closest_position(src_mesh, tgt_mesh)
    )
    angles = compute_angle(
        face_normals, positions - np.asarray(tgt_mesh.vertices, dtype=np.float32)
    )
    bbd = compute_threshold_distance(tgt_mesh, thres_bb_ratio)

    filter_ = (distances <= bbd) & (angles <= thres_angle)
    return filter_, barycentric_weights, face_ids


def copy_weights(src_mesh, src_weights, tgt_weights, match, bary_weights, face_ids):
    w = composite_bary_weights(src_mesh, src_weights, bary_weights, face_ids)
    tgt_weights[match] = w[match]


def get_edges(triangle):
    for i in range(3):
        yield (triangle[i], triangle[(i + 1) % 3])


def compute_opposite_contangent(edge, triangle, vertices):
    i, j = edge
    k = np.setdiff1d(triangle, edge, assume_unique=True).item()  # opposite vertex
    ki = vertices[i] - vertices[k]
    kj = vertices[j] - vertices[k]

    unit_vec = normalize([ki, kj], axis=1)
    a = np.arccos(np.clip(np.dot(unit_vec[0], unit_vec[1]), -1, 1))
    return 1 / np.tan(a)


def compute_LM_task(vertices, faces):
    # code reference to: http://mobile.rodolphe-vaillant.fr/entry/101/definition-laplacian-matrix-for-triangle-meshes
    L = sparse.lil_matrix((vertices.shape[0], vertices.shape[0]), dtype=np.float64)
    M_inv_d = np.zeros(vertices.shape[0])
    
    for face in faces:    
        position = vertices[face].reshape(3, 3)
        area = np.cross(
            position[1] - position[0],
            position[2] - position[0]
        )
        area = np.linalg.norm(area) / 6
        
        for e in get_edges(face):
            w = compute_opposite_contangent(e, face, vertices)
            i, j = e
            L[i, j] += w
            L[j, i] += w
            L[i, i] -= w
            L[j, j] -= w
            
            M_inv_d[i] = area          
        
    return L, M_inv_d

def compute_LM(mesh, n_workers=None):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
        
    partition = np.split(faces, n_workers, axis=0)
    args = [(vertices, p) for p in partition]
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.starmap(compute_LM_task, args)

    results_T = list(zip(*results))
    L = np.sum(results_T[0])
    M_inv = sparse.diags(np.reciprocal(np.sum(results_T[1], axis=0)))

    return L, M_inv


def sparse_cholesky(A): 
    # The input matrix A must be a sparse symmetric positive-definite.
    # code reference from: https://gist.github.com/omitakahiro/c49e5168d04438c5b20c921b928f1f5d
    LU = splinalg.splu(A, diag_pivot_thresh=0) # sparse LU decomposition
    return LU.L.dot( sparse.diags(LU.U.diagonal()**0.5) )


def index_sparse(A, row, col):
    A = torch.index_select(A, 0, row)
    A = torch.index_select(A, 1, col)
    return A


def weights_inpainting(tgt_mesh, tgt_weights, match, no_match, device):
    t_vertices = torch.as_tensor(np.asarray(tgt_mesh.vertices), dtype=torch.float32, device=device)
    t_faces = torch.as_tensor(np.asarray(tgt_mesh.triangles), dtype=int, device=device)
    match = torch.as_tensor(match)
    no_match = torch.as_tensor(no_match)
    
    L, M_inv = cot_laplacian(t_vertices, t_faces)
    L = L.cpu()
    M_inv = torch.sparse.spdiags(M_inv.squeeze().cpu(), torch.tensor([0]), (M_inv.shape[0], M_inv.shape[0]))
    
    Q = -L + L @ M_inv @ L
    Q_UU = index_sparse(Q, no_match, no_match).to_dense()
    Q_UI = index_sparse(Q, no_match, match)
    
    W_I = torch.as_tensor(tgt_weights[match, :], dtype=torch.float32)
    filter_ = torch.any(W_I != 0, dim=0)
    W_I = W_I[:, filter_]
    B = Q_UI @ W_I
    
    W_U = torch.linalg.solve(Q_UU.to(device=device), B.to(device=device))
    return W_U.cpu().numpy(), filter_.numpy()
    
    # for bone_id in range(tgt_weights.shape[-1]):
    #     w_I = W_I[:, bone_id]
    #     if np.all(w_I == 0):
    #         continue
        
    #     # solve for Ax = b, where A = Q_UU, x = w_u, b = -Q_UI @ w_I
    #     b = -Q_UI @ w_I
    #     y = splinalg.spsolve(LL, b)       
    #     tgt_weights[no_match, bone_id] = splinalg.spsolve(LL.T, y)
        

def transfer_weights(
    src_mesh,
    tgt_mesh,
    src_weights,
    thres_bb_ratio=0.05,
    thres_angle=35,
    radian=False,
    return_match=False,
    copy_all=False,
    # n_workers=None,
    device="cuda"
):
    filter_, bary_weights, face_ids = find_match(
        src_mesh, tgt_mesh, thres_bb_ratio, thres_angle, radian
    )
    
    if copy_all:
        filter_[~filter_] = True

    tgt_weights = np.zeros((filter_.shape[0], src_weights.shape[-1]))
    copy_weights(src_mesh, src_weights, tgt_weights, filter_, bary_weights, face_ids)
    
    match = np.where(filter_)[0]
    no_match = np.where(~filter_)[0]
        
    # L, M_inv = compute_LM(tgt_mesh, n_workers)    
    if not copy_all:
        W_U, filter_ = weights_inpainting(tgt_mesh, tgt_weights, match, no_match, device)
        tgt_weights[np.ix_(no_match, filter_)] = W_U
        tgt_weights = normalize(np.clip(tgt_weights, 0, None), axis=1)
        tgt_weights[np.all(tgt_weights == 0, axis=-1), 0] = 1.

    if return_match:
        return tgt_weights, match, no_match
    else:
        return tgt_weights
