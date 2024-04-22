import numpy as np
import open3d as o3d
from scipy.interpolate import NearestNDInterpolator

import os
from dataclasses import dataclass
from typing import Tuple, Any, List
# from confmap.confmap import BFF

from itertools import product, combinations

from .utils import (
    create_idx_mat,
    transform_coords_pix2norm,
    transform_coords_norm2real,
    psr,
    create_o3d_mesh,
    create_o3d_pcd,
    create_distance_filter,
    filter_o3d_pcd,
    find_border,
)

from ..settings.model_settings import CameraSettings, DEFAULT_CAMERA_SETTINGS

def reconstruct_from_depth(pm_depth, thres, z, fov, depth_offset=0.0):
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
        p = real_coords * (1 - f_d / z)
        p = np.concatenate([p, f_d], axis=-1)

        yield p, thres_mask


def partial_mesh(pcd, mask, thres=0.025):
    """
    given grid:

    [[0, 1],

     [2, 3]]

    add [0, 1, 3] and [0, 2, 3] as faces if all elements are `True` in the mask
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
        # get the pixel coordinates on the bottom right, right, bottom
        pool = [(row, col)]
        for r_off, c_off in op[:-1]:
            tmp = row + r_off, col + c_off
            if tmp[0] < mask.shape[0] and tmp[1] < mask.shape[1]:
                pool.append(tmp)

        if len(pool) < 4: 
            continue

        valid = [
            __getitem(mask, i) for i in pool
        ]  # check if all points are considered as vertices
        indices = [__getitem(idx, i) for i in pool]

        # if bottom right is not considered as a vertex, no face is created
        if not valid[1]:
            continue

        if valid[-2] and __within_thres(tmp := [indices[0], indices[2], indices[1]]):
            faces.append(tmp)

        if valid[-1] and __within_thres(tmp := [indices[0], indices[3], indices[1]]):
            faces.append(tmp)

    faces = np.asarray(faces)

    return faces


def refine_geometry(
    meshes: o3d.geometry.TriangleMesh,
    borderline: np.ndarray,
    max_dist: float,
    min_dist: float,
    sampler_mesh: o3d.geometry.TriangleMesh,
    dilate_factor: float,
    merge_eps: float = 1e-2,
):
    # combine partial meshes
    mesh = np.sum(meshes)
        
    borderline = create_o3d_pcd(borderline)

    # sample points on PSR
    # pcd = sampler_mesh.sample_points_poisson_disk(
    #     int(np.asarray(mesh.vertices).size * dilate_factor)
    # )

    vertices = np.asarray(sampler_mesh.vertices)
    sampler_pcd = create_o3d_pcd(
        vertices, normals=np.asarray(sampler_mesh.vertex_normals)
    )

    distance = np.asarray(sampler_pcd.compute_point_cloud_distance(borderline))
    vertex_filter = (
        (distance <= max_dist)
        # & (distance >= min_dist)
        & (vertices[:, 2] >= mesh.get_min_bound()[2])
    )
    filter_o3d_pcd(sampler_pcd, vertex_filter)

    vertex_ids = np.where(vertex_filter)[0]
    faces = np.asarray(sampler_mesh.triangles)
    face_mask = np.all(np.isin(faces, vertex_ids, invert=True), axis=-1)

    sampler_mesh.remove_triangles_by_mask(face_mask)
    sampler_mesh.remove_unreferenced_vertices()

    # o3d.visualization.draw_geometries([mesh, borderline])

    return sampler_mesh


def map_texture(meshes, refined, rgb_map):
    # auto seam estimation
    vertices = [np.asarray(mesh.vertices) for mesh in meshes]
    labels = [np.full(v.shape[0], i) for i, v in enumerate(vertices)]
    interp = NearestNDInterpolator(
        np.vstack(vertices),
        np.hstack(labels)
    )
    
    queries = np.asarray(refined.vertices)
    result = interp(queries).astype(int)
    
    # c = np.zeros_like(queries)
    # c[result == 1, 1] = 1.
    # c[result == 0, 0] = 1.
    # p = create_o3d_pcd(queries, colors=c)
    
    faces = np.asarray(refined.triangles)
    b_result = result[faces]
    boundaries = faces[(tmp := np.any(b_result != b_result[:, 0].reshape(-1, 1), axis=-1))] # it is a boundary face if it contains more than 1 unique peelmap id
    b_result = b_result[tmp]
    
    # duplicate boundary faces to corresponding peelmap
    b_faces = { i: [] for i in np.unique(result)}
    for i, r in enumerate(b_result):
        uniq = np.unique(r)
        for j in uniq:
            b_faces[j].append(boundaries[i])
    
    for i, mesh in enumerate(meshes):
        remove_ids = np.where(result != i)[0]
        b_ids = np.unique(np.stack(b_faces[i]).flatten())
        remove_ids = np.setdiff1d(remove_ids, np.intersect1d(remove_ids, b_ids, assume_unique=True), assume_unique=True)
                
        partition = o3d.geometry.TriangleMesh(refined)
        partition.remove_vertices_by_index(remove_ids)
        
        tmp = partition + mesh
        tmp.remove_unreferenced_vertices()
        
        # o3d.visualization.draw_geometries([tmp])
        
        # parameterization
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(tmp)
        uv = t_mesh.compute_uvatlas()
        
        pass
    
    return meshes


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
    camera_settings: CameraSettings = DEFAULT_CAMERA_SETTINGS

    pcds: List[o3d.geometry.PointCloud] | None = None
    pcd_masks: List[np.ndarray] | None = None
    borders: List[np.ndarray] | None = None

    @classmethod
    def from_tensor_dict(cls, t_dict, B, **kwargs):
        return [
            cls.from_tensors(
                img=t_dict["Img"][i],
                pm_depth=t_dict["Depth"][i],
                pm_norm=t_dict["Norm"][i],
                pm_rgb=t_dict["RGB"][i],
                **kwargs,
            )
            for i in range(B)
        ]

    @classmethod
    def from_tensors(cls, img, pm_depth, pm_norm, pm_rgb, **kwargs):
        img = img.cpu().numpy()
        pm_depth = pm_depth.cpu().numpy()
        pm_norm = pm_norm.cpu().numpy()
        pm_rgb = np.concatenate([np.expand_dims(img, 0), pm_rgb.cpu().numpy()], axis=0)

        return cls(img=img, pm_depth=pm_depth, pm_norm=pm_norm, pm_rgb=pm_rgb, **kwargs)

    def denoise(self, dist: float = 0.01, path: str | None = None):
        """
        remove outliners using low-poly reconstruction.
        """
        if self.pcds is None:
            return

        # filter pcd
        filter_obj = psr(np.sum(self.pcds), depth=5)
        pt_filters = [
            create_distance_filter(pcd, filter_obj, u_dist=dist) for pcd in self.np_pcds
        ]
        for pcd, f in zip(self.pcds, pt_filters):
            filter_o3d_pcd(pcd, f)

        # update mask
        for mask, f in zip(self.pcd_masks, pt_filters):
            mask[mask] = f.numpy()

        if path is not None:
            o3d.io.write_triangle_mesh(os.path.join(path, "low_poly.obj"), filter_obj)

        return filter_obj

    def to_static_mesh(
        self,
        path: str | None = None,
        face_dist: float = 0.025,
        smooth_iter: int = 10,
        smooth_lambda: float = 0.5,
        smooth_mu: float = -0.53,
        refine_dist_max: float = 0.01,
        refine_dist_min: float = 0.001,
        sampler_depth: int = 5,
        sampler_mesh: o3d.geometry.TriangleMesh | None = None,
        sampler_dilation: float = 0.5,
        file_extension: str = "gltf",
    ) -> o3d.geometry.TriangleMesh:
        """export the backprojected peelmaps to a static mesh with texture
        
        the reconstruction process is as follows:
        1. reconstruct partial faces 
        2. refine mesh by sampling faces to fill the missing faces
        3. generate texture atlas
        
        Parameters
        ----------
        path : str | None, optional
            location to save the exported mesh, by default None
        face_dist : float, optional
            max distance between the vertices that can form a mesh, by default 0.025
        smooth_iter : int, optional
            smooth iteration, by default 10
        smooth_lambda : float, optional
            smoothing parameter, by default 0.5
        smooth_mu : float, optional
            smoothing parameter, by default -0.53
        refine_dist_max : float, optional
            the max distance a psr vertice can deviate from the partially reconstructed mesh, by default 0.01
        refine_dist_min : float, optional
            the min distance a psr vertice can deviate from the partially reconstructed mesh, by default 0.001
        sampler_depth : int, optional
            the depth of psr for generating a sampler mesh for sampling missing faces, will be ignored if sampler_mesh is specified, by default 5
        sampler_mesh : o3d.geometry.TriangleMesh | None, optional
            the mesh to sample missing faces from, by default None
        sampler_dilation : float, optional
            _description_, by default 0.5
        file_extension : str, optional
            file format to save, by default "gltf"

        Returns
        -------
        open3d.geometry.TriangleMesh
            the resulting mesh
        """
        if self.pcds is None:
            return

        # partial mesh
        faces = [
            partial_mesh(pcd, mask, face_dist)
            for pcd, mask in zip(self.np_pcds, self.pcd_masks)
        ]
        meshes = [
            create_o3d_mesh(pcd, faces_)
            for pcd, faces_ in zip(self.pcds, faces)
        ]

        # compute borders coordinates
        borderline = [
            np.asarray(self.pcds[i].points, dtype=np.float32)[
                self.borders[i][self.pcd_masks[i]]
            ]
            for i in range(len(self.pcds))
        ]
        borderline = np.concatenate(borderline)

        # refine geometry
        refined = refine_geometry(
            meshes,
            borderline,
            refine_dist_max,
            refine_dist_min,
            sampler_mesh if sampler_mesh is not None else psr(np.sum(self.pcds), depth=sampler_depth),
            sampler_dilation,
        )
        
        # mesh = map_texture(meshes, refined, self.pm_rgb)
        mesh = np.sum(meshes) + refined
        
        mesh = mesh.filter_smooth_taubin(smooth_iter, smooth_lambda, smooth_mu)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh.orient_triangles()
            
        # o3d.visualization.draw_geometries([mesh])

        # texture inpainting
        # mesh = map_texture(mesh)

        if path is not None:
            o3d.io.write_triangle_mesh(f"{path}.{file_extension}", mesh)

        return mesh

    def to_img(self):
        """
        parse each peelmaps to PNG images.
        """
        pass

    def backproject(
        self,
        thres: float = 5e-5,
        depth_offset: float = 0.0,
        denoise_dist: float | None = 0.01,
        path: str | None = None,
    ):
        """
        reconstruct the model as a 3d meshes using the peelmaps.
        texture is also generated.

        @params: thres: depth below this value will be removed
        @params: path: if passed, it will save each peelamps in the specific location, which is a directory

        @return: point cloud of each peeled layer
        """
        pcds = []
        pcd_masks = []

        for i, (pcd, thres_mask) in enumerate(
            reconstruct_from_depth(
                self.pm_depth.squeeze(),
                thres,
                self.camera_settings.z,
                self.camera_settings.fov,
                depth_offset=depth_offset,
            )
        ):
            norm = np.moveaxis(self.pm_norm[i], 0, -1)
            rgb = np.moveaxis(self.pm_rgb[i], 0, -1)

            if self.mask is not None:
                thres_mask = thres_mask & self.mask

            # if there is no i-th intersection, there must be no (i+1)-th intersection
            if thres_mask.sum() == 0:
                break

            pcds.append(
                create_o3d_pcd(
                    pcd[thres_mask], normals=norm[thres_mask], colors=rgb[thres_mask]
                )
            )
            pcd_masks.append(thres_mask)

        self.pcds = pcds
        self.pcd_masks = pcd_masks
        
        self.save_pcd(f"{path}/raw")

        # denoise
        lowpoly = self.denoise(denoise_dist, path) if denoise_dist is not None else None

        # find borders
        self.borders = [find_border(m) for m in self.pcd_masks]

        self.save_pcd(path)

        return lowpoly

    def save_pcd(self, path: str):
        if path is None:
            return

        if not os.path.isdir(path):
            os.makedirs(path)

        if self.pcds is not None:
            for i, pcd in enumerate(self.pcds):
                o3d.io.write_point_cloud(f"{path}/layer_{i}.ply", pcd)

    @property
    def np_pcds(self):
        if self.pcds is None:
            return None
        return [np.asarray(t.points, dtype=np.float32) for t in self.pcds]
