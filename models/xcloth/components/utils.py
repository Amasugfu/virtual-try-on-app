import torch
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d

import copy, os
from dataclasses import dataclass
from typing import Tuple, Any, List

from itertools import product, combinations
from collections import deque

def create_idx_mat(size):
    """
    @return: row x col x yx-coords
    """
    return np.moveaxis(np.indices(size), 0, -1)


def create_norm_coords(size, idx_mat=None):
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
    def __compute_pixsep(__s, __fov):
        """
        compute real distance between each pixel.
        """
        a = z*np.tan(__fov / 2)
        return __s / a / 2

    if type(fov) == tuple:
        assert len(size) == len(fov)
        sep = [__compute_pixsep(s, f) for s, f in zip(size, fov)]
    else:
        sep = __compute_pixsep(size[0], fov)
        sep = [sep] * len(size)

    for dim in range(coords.shape[-1]):
        real_coords[:, :, dim] = coords[:, :, dim] / sep[dim]

    # swap (y, x) to (x, y)
    real_coords[:, :, [0, 1]] = real_coords[:, :, [1, 0]]

    return real_coords, sep


def reconstruct_from_depth(pm_depth, thres, z, fov, depth_offset=0.):
    """
    reconstruct point cloud from depth peelmaps

    @param: pm_depth: P x H x W
    @param: thres: mask threshold
    @param: z: camera z position
    @param: fov: camera fov in radian
    @param: mask: boolean matrix of dimension H x W

    @return: (iterable of point cloud of each peeled layer, row id, col id)
    """
    img_size = pm_depth.shape[1:]
    pix_coords = create_idx_mat(img_size)
    
    pix_coords_norm = create_norm_coords(img_size, idx_mat=pix_coords)
    real_coords, sep = transform_coords_norm2real(img_size, pix_coords_norm, z, fov)
    
    for depth in pm_depth:
        thres_mask = depth > thres
        # reconstruct original (x, y) coordinates
        f_d = np.expand_dims(depth - depth_offset, -1)
        p = real_coords * (1 - f_d/z)
        p = np.concatenate([p, f_d], axis=-1)

        yield p, thres_mask


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
    _filter = _filter.numpy()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[_filter])
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[_filter])
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[_filter])


def filter_pcd(pcds, _filters):
    for pcd, f in zip(pcds, _filters):
        filter_o3d_pcd(pcd, f)


def partial_mesh(pcd, mask, thres=0.025):
    """
    given grid:

    [[0, 1],
    
     [2, 3]]

    add [0, 1, 2] and [0, 2, 3] as faces if all elements are `True` in the mask
    """
        
    faces = []
    idx = np.empty_like(mask, dtype=int)
    idx[mask] = np.arange(mask.sum())
    
    def __getitem(src, tup):
        return src[tup[0], tup[1]]
    
    def __within_thres(i):
        dist = [np.linalg.norm(pcd[a] - pcd[b]) for a, b in combinations(i, 2)]
        return max(dist) <= thres

    for row, col in zip(*np.where(mask)):
        op = tuple(product((1, 0), repeat=2))

        pool = [(row, col)]
        for r_off, c_off in op[:-1]:
            tmp = row + r_off, col + c_off
            pool.append(tmp)

        valid = [__getitem(mask, i) for i in pool]
        indices = [__getitem(idx, i) for i in pool]

        if not valid[1]: continue

        if valid[-2] and __within_thres(tmp := indices[:-1]): 
            faces.append(tmp)
 
        if valid[-1] and __within_thres(tmp := [*indices[:2], indices[-1]]): 
            faces.append(tmp)

    return faces


def create_mesh(pcd, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = pcd.points
    mesh.vertex_normals = pcd.normals
    # mesh.vertex_colors = pcd.colors
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh
        

def psr(pcd, depth):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh


def refine_geometry(meshes, refine_depth, sampler_mesh, dilate_factor):
    mesh = np.sum(meshes)

    # sample points on PSR
    pcd = sampler_mesh.sample_points_poisson_disk(
        int(np.asarray(mesh.vertices).size * dilate_factor)
    )
    psr_filter = create_distance_filter(
        np.asarray(pcd.points, dtype=np.float32), 
        mesh, 
        l_dist=0.025,
        u_dist=0.1
    )
    filter_o3d_pcd(pcd, psr_filter.logical_not())

    o3d.visualization.draw_geometries([pcd])

    return mesh


def map_texture(meshes):
    return meshes


@dataclass
class CameraSettings:
    z: float = 1
    fov: float|Tuple[float, float] = np.pi/3


@dataclass
class GarmentModel3D:
    """
    pm_depth: P x 1 x H x W
    pm_norm: P x 3 x H x W
    pm_rgb: P x 3 x H x W
    mask: H x W
    """
    img: Any
    pm_depth: Any
    # pm_seg: Any
    pm_norm: Any
    pm_rgb: Any
    mask: Any = None
    camera_settings: CameraSettings = CameraSettings()
    
    pcd_masks: List[Any]|None = None
    pcds: List[o3d.geometry.PointCloud]|None = None

    @classmethod
    def from_tensor_dict(cls, t_dict, B, **kwargs):
        return [
            cls.from_tensors(
                img=t_dict["Img"][i],
                pm_depth=t_dict["Depth"][i], 
                pm_norm=t_dict["Norm"][i], 
                pm_rgb=t_dict["RGB"][i],
                **kwargs
            )
            for i in range(B)
        ]
    
    @classmethod
    def from_tensors(cls, img, pm_depth, pm_norm, pm_rgb, **kwargs):
        img = img.cpu().numpy()
        pm_depth = pm_depth.cpu().numpy()
        pm_norm = pm_norm.cpu().numpy()
        pm_rgb = np.concatenate([np.expand_dims(img, 0), pm_rgb.cpu().numpy()], axis=0)

        return cls(
            img=img,
            pm_depth=pm_depth, 
            pm_norm=pm_norm, 
            pm_rgb=pm_rgb,
            **kwargs
        )
    
    def denoise(self, dist: float = 0.01, path: str|None = None):
        """
        remove outliners using low-poly reconstruction.
        """
        if self.pcds is None: return

        # filter pcd
        filter_obj = psr(np.sum(self.pcds), depth=5)
        pt_filters = [create_distance_filter(pcd, filter_obj, u_dist=dist) for pcd in self.np_pcds]
        filter_pcd(self.pcds, pt_filters)

        # update mask
        for mask, f in zip(self.pcd_masks, pt_filters):
            mask[mask] = f

        if path is not None:
            o3d.io.write_triangle_mesh(os.path.join(path, "low_poly.obj"), filter_obj)

        return filter_obj

    def to_obj(self, 
               refine_depth: int = 9, 
               face_dist: float = 0.025, 
               path: str|None = None,
               smooth_iter: int = 1,
               lambda_filter: float = 0.5,
               mu: float = -0.53,
               sampler_mesh: o3d.geometry.TriangleMesh|None = None,
               sampler_dilation: float = 0.5):
        """
        parse to an OBJ 3D model.

        @param: mode: `"poisson" | "ballpivoting"`
        @param: depth: poisson surface reconstruction depth
        @param: dist: the max distance the vertices are away from the low-poly reconstruction. used for filtering out outliners

        @return: triange mesh with texture
        """
        if self.pcds is None: return

        # partial mesh 
        faces = [partial_mesh(pcd, mask, face_dist) for pcd, mask in zip(self.np_pcds, self.pcd_masks)]
        meshes = [create_mesh(pcd, faces_) for pcd, faces_ in zip(self.pcds, faces)]

        # refine geometry
        mesh = refine_geometry(
            meshes, 
            refine_depth, 
            sampler_mesh if sampler_mesh is not None else psr(np.sum(self.pcds), depth=5),
            sampler_dilation
        )
        mesh = mesh.filter_smooth_taubin(smooth_iter, lambda_filter, mu)

        # texture inpainting
        # mesh = map_texture(mesh)

        if path is not None:
            o3d.io.write_triangle_mesh(path, mesh)

        return mesh

    def to_img(self):
        """
        parse each peelmaps to PNG images.
        """
        pass

    def backproject(self, 
                    thres: float = 5e-5, 
                    depth_offset: float = 0., 
                    denoise_dist: float|None = 0.01, 
                    path: str|None = None):
        """
        reconstruct the model as a 3d meshes using the peelmaps.
        texture is also generated.

        @params: thres: depth below this value will be removed
        @params: path: if passed, it will save each peelamps in the specific location, which is a directory

        @return: point cloud of each peeled layer
        """
        pcds = []
        pcd_masks = []

        for i, (pcd, thres_mask) in enumerate(reconstruct_from_depth(self.pm_depth.squeeze(), thres, self.camera_settings.z, self.camera_settings.fov, depth_offset=depth_offset)):
            norm = np.moveaxis(self.pm_norm[i], 0, -1)
            rgb = np.moveaxis(self.pm_rgb[i], 0, -1)
            
            if self.mask is not None:
                thres_mask = thres_mask & self.mask

            # if there is no i-th intersection, there must be no (i+1)-th intersection
            if thres_mask.sum() == 0:
                break

            reconstructed_pcd = o3d.geometry.PointCloud()
            reconstructed_pcd.points = o3d.utility.Vector3dVector(pcd[thres_mask])
            reconstructed_pcd.normals = o3d.utility.Vector3dVector(norm[thres_mask])
            reconstructed_pcd.colors = o3d.utility.Vector3dVector(rgb[thres_mask])

            pcds.append(reconstructed_pcd)
            pcd_masks.append(thres_mask)

        self.pcds = pcds
        self.pcd_masks = pcd_masks

        # denoise
        lowpoly = self.denoise(denoise_dist, path) if denoise_dist is not None else None

        self.save_pcd(path)

        return lowpoly

    def save_pcd(self, path: str):
        if path is None: return

        if not os.path.isdir(path):
            os.makedirs(path)

        if self.pcds is not None:
            for i, pcd in enumerate(self.pcds):
                o3d.io.write_point_cloud(f"{path}/layer_{i}.ply", pcd)

    @property
    def np_pcds(self):
        if self.pcds is None: return None
        return [np.asarray(t.points, dtype=np.float32) for t in self.pcds]
    

            