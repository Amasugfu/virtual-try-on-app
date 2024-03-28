import numpy as np
import open3d as o3d

import os
from dataclasses import dataclass
from typing import Tuple, Any, List

from itertools import product, combinations

from .utils import (
    create_idx_mat, transform_coords_pix2norm, transform_coords_norm2real, psr, compute_pixsep,
    create_o3d_mesh, create_o3d_pcd, create_distance_filter, filter_o3d_pcd, find_border
)


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
    
    pix_coords_norm = transform_coords_pix2norm(img_size, idx_mat=pix_coords)
    real_coords, sep = transform_coords_norm2real(img_size, pix_coords_norm, z, fov)
    
    for depth in pm_depth:
        thres_mask = depth > thres
        # reconstruct original (x, y) coordinates
        f_d = np.expand_dims(depth - depth_offset, -1)
        p = real_coords * (1 - f_d/z)
        p = np.concatenate([p, f_d], axis=-1)

        yield p, thres_mask


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
        # return whether the points are within a distance to form a face
        dist = [np.linalg.norm(pcd[a] - pcd[b]) for a, b in combinations(i, 2)]
        return max(dist) <= thres

    op = tuple(product((1, 0), repeat=2))
    # for all points that are considered a vertex, create a face if the condition is met
    for row, col in zip(*np.where(mask)):
        # get the pixel coordinates on the right, bottom, and bottom right
        pool = [(row, col)]
        for r_off, c_off in op[:-1]:
            tmp = row + r_off, col + c_off
            pool.append(tmp)

        valid = [__getitem(mask, i) for i in pool] # check if all points are considered as vertices
        indices = [__getitem(idx, i) for i in pool] 

        # if bottom right is not considered as a vertex, no face is created
        if not valid[1]: continue

        if valid[-2] and __within_thres(tmp := indices[:-1]): 
            faces.append(tmp)
 
        if valid[-1] and __within_thres(tmp := [*indices[:2], indices[-1]]): 
            faces.append(tmp)

    faces = np.asarray(faces)

    return faces


def refine_geometry(meshes: o3d.geometry.TriangleMesh,
                    borderline: np.ndarray,
                    max_dist: float, 
                    min_dist: float, 
                    sampler_mesh: o3d.geometry.TriangleMesh, 
                    dilate_factor: float, 
                    merge_eps: float = 1e-2):
    mesh = np.sum(meshes)
    borderline = create_o3d_pcd(borderline)

    # sample points on PSR
    # pcd = sampler_mesh.sample_points_poisson_disk(
    #     int(np.asarray(mesh.vertices).size * dilate_factor)
    # )

    vertices = np.asarray(sampler_mesh.vertices)
    sampler_pcd = create_o3d_pcd(vertices)

    distance = np.asarray(sampler_pcd.compute_point_cloud_distance(borderline))
    vertex_filter = (distance <= max_dist) & (distance >= min_dist) & (vertices[:, 2] >= mesh.get_min_bound()[2])
    filter_o3d_pcd(sampler_pcd, vertex_filter)

    vertex_ids = np.where(vertex_filter)[0]
    faces = np.asarray(sampler_mesh.triangles)
    face_filter = np.any(np.isin(faces, vertex_ids), axis=-1)
    
    sampler_mesh.triangles = o3d.utility.Vector3iVector(faces[face_filter])
    sampler_mesh.remove_unreferenced_vertices()

    # psr_filter = create_distance_filter(
    #     np.asarray(pcd.points, dtype=np.float32), 
    #     mesh, 
    #     l_dist=0.025,
    #     u_dist=0.1
    # )
    # filter_o3d_pcd(pcd, psr_filter.logical_not())

    mesh = sampler_mesh + mesh
    # exit()

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
    
    pcds: List[o3d.geometry.PointCloud]|None = None
    pcd_masks: List[np.ndarray]|None = None
    borders: List[np.ndarray]|None = None

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
        for pcd, f in zip(self.pcds, pt_filters):
            filter_o3d_pcd(pcd, f)

        # update mask
        for mask, f in zip(self.pcd_masks, pt_filters):
            mask[mask] = f

        if path is not None:
            o3d.io.write_triangle_mesh(os.path.join(path, "low_poly.obj"), filter_obj)

        return filter_obj

    def to_obj(self, 
               face_dist: float = 0.025, 
               path: str|None = None,
               smooth_iter: int = 1,
               lambda_filter: float = 0.5,
               mu: float = -0.53,
               refine_dist: float = 0.01,
               refine_dist_min: float = 0.001,
               sampler_depth: int = 5, 
               sampler_mesh: o3d.geometry.TriangleMesh|None = None,
               sampler_dilation: float = 0.5):
        """
        parse to an OBJ 3D model.

        @param: mode: `"poisson" | "ballpivoting"`
        @param: refine_depth: poisson surface reconstruction depth
        @param: dist: the max distance the vertices are away from the low-poly reconstruction. used for filtering out outliners

        @return: triange mesh with texture
        """
        if self.pcds is None: return

        # partial mesh 
        faces = [partial_mesh(pcd, mask, face_dist) for pcd, mask in zip(self.np_pcds, self.pcd_masks)]
        meshes = [create_o3d_mesh(pcd, faces_) for pcd, faces_ in zip(self.pcds, faces)]

        # compute borders coordinates
        borderline = [
            np.asarray(self.pcds[i].points, dtype=np.float32)[self.borders[i][self.pcd_masks[i]]] 
                for i in range(len(self.pcds))
        ]
        borderline = np.concatenate(borderline)

        # refine geometry
        mesh = refine_geometry(
            meshes, 
            borderline, 
            refine_dist,
            refine_dist_min,
            sampler_mesh if sampler_mesh is not None else psr(np.sum(self.pcds), depth=sampler_depth),
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

            pcds.append(create_o3d_pcd(pcd[thres_mask], normals=norm[thres_mask], colors=rgb[thres_mask]))
            pcd_masks.append(thres_mask)

        self.pcds = pcds
        self.pcd_masks = pcd_masks

        # denoise
        lowpoly = self.denoise(denoise_dist, path) if denoise_dist is not None else None

        # find borders
        self.borders = [find_border(m) for m in self.pcd_masks]

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