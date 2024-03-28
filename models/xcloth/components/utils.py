import numpy as np
import open3d as o3d
from scipy.ndimage import generic_filter

import copy
from typing import Tuple, Any


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


def create_distance_filter(points, filter_obj, u_dist=None, l_dist=None):
    """
    @param: filter_obj: the triangular mesh to compute the distance from
    @param: dist: the max distance away from the `filter_obj`

    @return: a filter masking only the vertices that is within the distance from the `filter_obj`
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh=o3d.t.geometry.TriangleMesh.from_legacy(filter_obj))

    distance = scene.compute_distance(points)

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


def create_o3d_mesh(pcd, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = pcd.points
    mesh.vertex_normals = pcd.normals
    # mesh.vertex_colors = pcd.colors
    mesh.triangles = o3d.utility.Vector3iVector(faces)
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

            