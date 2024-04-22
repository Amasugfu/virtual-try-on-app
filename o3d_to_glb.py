import numpy as np
import open3d as o3d
import bpy
import bmesh

import os
import argparse

JOINTS_MAP = {
    "root": -1,
    "Pelvis": 0,
    "L_Hip": 1,
    "L_Knee": 4,
    "L_Ankle": 7,
    "L_Foot": 10,
    "R_Hip": 2,
    "R_Knee": 5,
    "R_Ankle": 8,
    "R_Foot": 11,
    "Spine1": 3,
    "Spine2": 6,
    "Spine3": 9,
    "Neck": 12,
    "Head": 15,
    "L_Collar": 13,
    "L_Shoulder": 16,
    "L_Elbow": 18,
    "L_Wrist": 20,
    "L_Hand": 22,
    "R_Collar": 14,
    "R_Shoulder": 17,
    "R_Elbow": 19,
    "R_Wrist": 21,
    "R_Hand": 23,
}


def o3d_to_skinned_glb(vertices, faces, export_path, colors, weights=None, armature_path="models/data/test_data/assets/smpl_male_blend2.glb"):
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if armature_path is None:
            m = bpy.data.meshes.new("NewMesh") 
            obj = bpy.data.objects.new("NewObject", m) 
            bpy.context.collection.objects.link(obj) 
        else:            
            abs_path = os.path.abspath(armature_path)
            bpy.ops.import_scene.gltf(filepath=abs_path)
            obj = bpy.context.scene.objects["SMPL-mesh-male"]
            
        bpy.context.view_layer.objects.active = obj

        bm = bmesh.new()
        bm.from_mesh(obj.data)

        bmesh.ops.delete(bm, geom=bm.verts)
        vertices[:, -1] *= -1
        vertices[:, [1, -1]] = vertices[:, [-1, 1]]

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
        if weights is not None:
            for g in obj.vertex_groups:
                i = JOINTS_MAP[g.name]
                if i == -1:
                    continue
                w = weights[:, i]
                for i, v_id in enumerate(bm_verts_id):
                    obj.vertex_groups[g.name].add([v_id], w[i], "REPLACE")

        # add color / texture    
        if colors is not None:
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

        bpy.ops.export_scene.gltf(filepath=export_path, use_visible=armature_path is None)
    except Exception as e:
        print(e)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", required=True)
    parser.add_argument("-f", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-c")
    parser.add_argument("-w")
    parser.add_argument("-a", default=None)
    args = parser.parse_args()
    
    vertices = np.load(args.v)
    faces = np.load(args.f)
    colors = np.load(args.c) if args.c else None
    weights = np.load(args.w) if args.w else None
    
    o3d_to_skinned_glb(vertices, faces, args.o, colors, weights, args.a)