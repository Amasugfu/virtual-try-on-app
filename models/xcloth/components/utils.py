import numpy as np
import open3d as o3d
from scipy.ndimage import generic_filter
import torch

import copy
from typing import Tuple, Any

from ...smplx_.lbs import lbs
from ...smplx_.body_models import SMPL
from .const import JOINTS_MAP

import os
import subprocess


def create_idx_mat(size):
    """
    @return: row x col x yx-coords
    """
    return np.moveaxis(np.indices(size), 0, -1)


def transform_coords_pix2norm(size, idx_mat=None):
    """
    transform pixel coordinates `[0, 0] [0, 1] ... [n, m]` to `[-n/2, -m/2] ... [n/2, m/2]`

    i.e. transform the origin from the upper corner to the center of image
    """
    if idx_mat is None:
        idx_mat = create_idx_mat(size)
    else:
        idx_mat = copy.deepcopy(idx_mat)

    idx_mat[:, :, 0] -= size[0] // 2
    idx_mat[:, :, 1] -= size[1] // 2
    # idx_mat[:, 1] *= -1
    return idx_mat


def compute_pixsep(s, fov, z):
    """
    compute real distance between each pixel.
    """
    a = z*np.tan(fov / 2)
    return 2*a / s


def transform_coords_norm2real(
        size: int|Tuple[int, int], 
        coords: Any, 
        z: float, 
        fov: float
):
    """
    transform pixel coordinates `[-pix_y/2, -pix_x/2], ..., [0, 0], ..., [pix_y/2, pix_x/2]` to 3d world (x, y) coordinates.

    @param: size: image dimension
    @param: coords: pixel coordinates matrix of dimension of N x 2
    @param: z: camera z position
    @param: fov: camera fov in radian

    @return: (real world 3D coordinates in matrix of dimension of N x 2, real world distance per pixel)
    """
    real_coords = np.zeros_like(coords, dtype=np.float64)

    if type(fov) == tuple:
        assert len(size) == len(fov)
        sep = [compute_pixsep(s, f, z) for s, f in zip(size, fov)]
    else:
        sep = compute_pixsep(size[0], fov, z)
        sep = [sep for _ in range(len(size))]

    for dim in range(coords.shape[-1]):
        real_coords[:, :, dim] = coords[:, :, dim] * sep[dim]

    # swap (y, x) to (x, y)
    real_coords[:, :, [0, 1]] = real_coords[:, :, [1, 0]]

    return real_coords, sep


def compute_distance_from_mesh(mesh, points):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    return scene.compute_distance(points)


def compute_closest_point(mesh, points):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    return scene.compute_closest_points(points)


def create_distance_filter(points, filter_obj, u_dist=None, l_dist=None):
    """
    @param: filter_obj: the triangular mesh to compute the distance from
    @param: u_dist: the max distance away from the `filter_obj`
    @param: l_dist: the min distance away from the `filter_obj`

    @return: a filter masking only the vertices that is within the distance from the `filter_obj`
    """
    distance = compute_distance_from_mesh(filter_obj, points)

    f = True
    
    if u_dist is not None: f = (distance <= u_dist).logical_and(f)
    if l_dist is not None: f = (distance >= l_dist).logical_and(f)

    return f


def filter_o3d_pcd(pcd, _filter):
    if type(_filter) != np.ndarray: 
        _filter = _filter.numpy()
        
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[_filter])
    if len(pcd.normals) > 0: 
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[_filter])
    if len(pcd.colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[_filter])


def find_border(mat):
    """
    find the border of the binary mat
    
    @param: mat: the 2d matrix

    @return: a 2d matrix where the border is marked `True`
    """
    mask = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]).astype(bool).flatten()

    def __replace_non_border(x):
        return 0 if np.all(x[mask]) else x[4]
    
    return generic_filter(mat, __replace_non_border, 3)


def create_o3d_mesh(pcd, faces, face_normals=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = pcd.points
    mesh.vertex_normals = pcd.normals
    mesh.vertex_colors = pcd.colors
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if face_normals is not None:
        mesh.triangle_normals = o3d.utility.Vector3dVector(face_normals)
    return mesh


def create_o3d_pcd(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd
        

def psr(pcd, depth):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh   

            
def pose_smpl(pose, gender="male", return_T_only=False, return_mesh=False, return_faces=False, return_weights=False, device="cuda"):
    """pose the smpl model

    Parameters
    ----------
    pose : np.ndarray
        the pose array for the the smplx package
    gender : str, optional
        smplx settings, by default "male"
    return_T_only : bool, optional
        return the transformation matrix, by default False
    return_mesh : bool, optional
        return the open3d mesh, by default False
    return_faces : bool, optional
        return the face in vertex indices, by default False
    return_weights : bool, optional
        return the smpl skinning weights, by default False
    device : str, optional
        y default "cuda"

    Returns
    -------
    Optional[open3d triagnle mesh | (vertices, joints)], transformation matrix, pose offsets, Optional[weights], Optional[face]
    """
    smpl = SMPL(model_path="models/smpl", gender=gender).to(device=device)
    with torch.no_grad():
        fin_pose= torch.FloatTensor(pose).unsqueeze(0).to(device=device)
        smpl_output, T, pose_offsets = smpl(
            global_orient=fin_pose[:, :3], 
            body_pose=fin_pose[:, 3:]
        )
        
        ret = T, pose_offsets
        if return_T_only:
            return ret

        ret_verts = smpl_output.vertices
        ret_joints = smpl_output.joints

        trans_verts = ret_verts.squeeze() #* scale + trans
        trans_joints = ret_joints.squeeze() #* scale + trans
        
        if return_weights:
            ret = *ret, smpl.lbs_weights
            
        if return_mesh:
            smpl_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(trans_verts.detach().cpu()),
                triangles=o3d.utility.Vector3iVector(smpl.faces),
            )
            return smpl_mesh, *ret
        
        ret = (trans_verts, trans_joints, *ret)
        if return_faces: 
            ret = (*ret, smpl.faces.squeeze())
            
        return ret
        

def o3d_to_skinned_glb(mesh, export_folder, weights=None, armature_path="models/data/test_data/assets/smpl_male_blend2.glb", output_name="result.glb"):
    """convert open3d mesh to glb by calling o3d_to_glb.py

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        the input mesh
    export_folder : str
        output folder
    weights : np.ndarray, optional
        the weights of each vertices on each joints, by default None
    armature_path : str, optional
        the glb file storing the armature/skeleton, by default "models/data/test_data/assets/smpl_male_blend2.glb"
    output_name : str, optional
        name of the output file, by default "result.glb"
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    
    pv = os.path.join(export_folder, "v.npy")
    pf = os.path.join(export_folder, "f.npy")
    
    np.save(pv, vertices)
    np.save(pf, faces)
    
    cmds = [
        "python", 
        "o3d_to_glb.py",
        "-v", pv, 
        "-f", pf, 
        "-o", os.path.join(export_folder, output_name)
    ]
    
    if colors.size > 0:
        pc = os.path.join(export_folder, "c.npy")
        np.save(pc, colors)
        cmds.append("-c")
        cmds.append(pc)
        
    if armature_path is not None:
        p = os.path.abspath(armature_path)
        cmds.append("-a")
        cmds.append(p)
        
    if weights is not None:
        pw = os.path.join(export_folder, "w.npy")
        np.save(pw, weights)
        cmds.append("-w")
        cmds.append(pw)

    subprocess.run(cmds)