import numpy as np
import open3d as o3d
from scipy.ndimage import generic_filter
import torch

import copy
from typing import Tuple, Any

from ...smplx_.lbs import lbs
from ...smplx_.body_models import SMPL
from .const import JOINTS_MAP

import bpy
import bmesh


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
    @param: dist: the max distance away from the `filter_obj`

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

            
def pose_smpl(pose, gender="male", return_T_only=False, return_mesh=False, return_faces=False, device="cuda"):
    smpl = SMPL(model_path="models/smpl", gender=gender)
    with torch.no_grad():
        fin_pose= torch.FloatTensor(pose).unsqueeze(0).to(device=device)
        smpl_output, T, pose_offsets = smpl(
            global_orient=fin_pose[:, :3], 
            body_pose=fin_pose[:, 3:]
        )
        
        if return_T_only:
            return T, pose_offsets

        ret_verts = smpl_output.vertices
        ret_joints = smpl_output.joints

        trans_verts = ret_verts.squeeze() #* scale + trans
        trans_joints = ret_joints.squeeze() #* scale + trans
        
        if return_mesh:
            smpl_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(trans_verts.detach().cpu()),
                triangles=o3d.utility.Vector3iVector(smpl.faces.detach().cpu()),
            )
            return smpl_mesh, T, pose_offsets
        
        ret = (trans_verts, trans_joints, T, pose_offsets)
        if return_faces: 
            return *ret, smpl.faces
        else:
            return ret
    
    
def o3d_to_skinned_glb(mesh, weights, export_path, armature_path="models/data/test_data/assets/smpl_male_blend2.glb"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=armature_path)

    # bpy.data.objects["SMPL-mesh-male"].select_set(True)
    obj = bpy.context.scene.objects["SMPL-mesh-male"]
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # bmesh.ops.delete(bm, geom=bm.verts)
    vertices = np.array(mesh.vertices)
    vertices[:, -1] *= -1
    vertices[:, [1, -1]] = vertices[:, [-1, 1]]
    
    faces = np.asarray(mesh.triangles)
    
    colors = np.asarray(mesh.vertex_colors)

    # add vertices and faces
    bm_verts = []
    bm_verts_id = []
    for v in vertices:
        vert = bm.verts.new(v)
        vert.index = len(bm.verts)
        bm_verts_id.append(vert.index)
        bm_verts.append(vert)
        
    for f in faces:
        bm.faces.new([
            bm_verts[f[0]],
            bm_verts[f[1]],
            bm_verts[f[2]]
        ])
        
    bm.to_mesh(obj.data)  
    bm.free()

    # add sknning weights
    for g in obj.vertex_groups:
        i = JOINTS_MAP[g.name]
        if i == -1:
            continue
        w = weights[:, i]
        for i, v_id in enumerate(bm_verts_id):
            obj.vertex_groups[g.name].add([v_id], w[i], "REPLACE")

    # add color / texture    
    color_layer_name = "vertex_colors"
    obj.data.vertex_colors.new(name=color_layer_name)
    color_layer = obj.data.vertex_colors[color_layer_name]

    inverse_id_map = { v: i for i , v in enumerate(bm_verts_id)}
    for poly in obj.data.polygons:
        for vert_i_poly, vert_i_mesh in enumerate(poly.vertices):  
            if vert_i_mesh in inverse_id_map.keys():
                vert_i_loop = poly.loop_indices[vert_i_poly]
                color_layer.data[vert_i_loop].color = (*colors[inverse_id_map[vert_i_mesh]], 1.)
        
    # reference from: https://stackoverflow.com/questions/67854896/how-do-i-set-the-base-colour-of-a-material-to-equal-vertex-colours-in-blender-2

    material_name = "material0"
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    obj.data.materials.append(material)

    # Get node tree from the material
    nodes = material.node_tree.nodes
    principled_bsdf_node = nodes.get("Principled BSDF")

    # Get Vertex Color Node, create it if it does not exist in the current node tree
    vertex_color_node = nodes.new(type="ShaderNodeVertexColor")

    # Set the vertex_color layer we created at the beginning as input
    vertex_color_node.layer_name = color_layer_name

    # Link Vertex Color Node "Color" output to Principled BSDF Node "Base Color" input
    links = material.node_tree.links
    links.new(vertex_color_node.outputs[0], principled_bsdf_node.inputs[0])

    obj.data.update()

    bpy.ops.export_scene.gltf(filepath=export_path)