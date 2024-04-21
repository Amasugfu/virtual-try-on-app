if __name__ == "__main__":

    import os, sys
    m_path = os.path.abspath(os.path.join("..\.."))
    if m_path not in sys.path:
        sys.path.append(m_path)
        sys.path.append(os.getcwd())
        
    # from models.rigging.weight_transfer import main
    from models.rigging.weight_transfer_robust import transfer_weights

    import open3d as o3d
    import numpy as np
    import torch

    # %%
    from models.smplx_.body_models import SMPL

    smpl = SMPL("models/smpl", gender="male")
    src_weights = smpl.lbs_weights

    # %%
    import pickle
    import torch

    src_dict_path = "models/xcloth/no_git_test_data/16-14/16-14.pkl"

    with open(src_dict_path, "rb") as f:
        src_dict = pickle.load(f)

    pose = src_dict['pose']
    trans = src_dict['trans']
    scale = src_dict['scale']

    with torch.no_grad():
        fin_pose= torch.FloatTensor(pose).unsqueeze(0)
        smpl_output, T, _ = smpl(
            global_orient=fin_pose[:, :3], 
            body_pose=fin_pose[:, 3:]
        )

        ret_verts = smpl_output.vertices
        ret_joints = smpl_output.joints

        trans_verts = ret_verts.squeeze() #* scale + trans
        trans_joints = ret_joints.squeeze() #* scale + trans

    smpl_mesh = o3d.geometry.TriangleMesh()
    smpl_mesh.vertices = o3d.utility.Vector3dVector(trans_verts)
    smpl_mesh.triangles = o3d.utility.Vector3iVector(smpl.faces.astype(int))
    smpl_mesh.compute_vertex_normals()

    joints = trans_joints[:24]
    joints_pt = o3d.geometry.PointCloud()
    joints_pt.points = o3d.utility.Vector3dVector(joints)
    joints_pt.colors = o3d.utility.Vector3dVector(np.repeat([[0, 0, 1]], repeats=joints.shape[0], axis=0))

    # %%
    # glb_path = "C:/Users/User/Downloads/smpl_male_blend2.glb"
    # glb = o3d.io.read_triangle_mesh(glb_path)

    # o3d.visualization.draw_geometries([glb, o3d.geometry.LineSet.create_from_triangle_mesh(smpl_mesh)])

    # %%
    # source_mesh_path = "../no_git_test_data/1-1/smpl1.obj"
    target_mesh_path = "models/xcloth/no_git_test_data/16-14/model_cleaned.obj"

    # src_mesh = o3d.io.read_triangle_mesh(source_mesh_path)
    tgt_mesh = o3d.io.read_triangle_mesh(target_mesh_path, True)
    # tgt_mesh.compute_vertex_normals()
    # align with smpl
    tgt_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tgt_mesh.vertices) / scale - trans + smpl_output.joints[:, 0].cpu().numpy())

    # o3d.visualization.draw_geometries([o3d.geometry.LineSet.create_from_triangle_mesh(smpl_mesh), joints_pt, tgt_mesh])

    # %%
    # tgt_weights, v_match, v_no_match = main(smpl_mesh, tgt_mesh, src_weights.cpu().numpy(), threshold_distance=0.5/scale)
    tgt_weights, v_match, v_no_match = transfer_weights(smpl_mesh, tgt_mesh, src_weights.cpu().numpy(), return_match=True, n_workers=8)

    # %%
    tgt_weights.shape

    # %%
    colors = np.zeros((tgt_weights.shape[0], 3))
    colors[v_match] = np.array([0., .5, 0.])
    colors[v_no_match] = np.array([.5, 0., 0])
    tgt_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    tgt_mesh.triangle_uvs = o3d.utility.Vector2dVector()

    # %%
    o3d.visualization.draw_geometries([joints_pt, tgt_mesh])

    # %%
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    from trimesh.triangles import points_to_barycentric

    from typing import Tuple

    def comupte_transform(
        target: torch.Tensor, 
        vertices: torch.Tensor,
        faces: torch.Tensor, 
        T: torch.Tensor,
        device: torch.device = torch.device("cuda")
    ) -> torch.Tensor:
        """compute tranformation matrix for each vertex in the target mesh 
        by finding the closest point on the reference faces.

        Parameters
        ----------
        target : torch.Tensor
            the vertices to compute the transformation for.
            [n, 3]
        vertices : torch.Tensor
            the vertices in 3d position of the reference mesh
            [m, 3]
        faces : torch.Tensor
            the faces in vertex indeice of the reference mesh
            [f, 3]
        T: torch,.Tensor
            the transformation matices of each vertex
            [m, 4, 4]

        Returns
        -------
        torch.Tensor
            the computed tranformation matrix of shape [n, 4, 4]
        """

        face_positions = vertices[faces]
        _, face_ids, _ = point_to_mesh_distance(
            target.to(device=device).unsqueeze(0),
            face_positions.to(device=device).unsqueeze(0))
        face_ids = face_ids.squeeze().to(device=target.device)
        
        bary_coords = points_to_barycentric(face_positions[face_ids], target)
        ref_T = T[faces[face_ids]]
        target_T = (bary_coords * ref_T).sum(axis=1)
        
        return target_T
        

    # %%
    # tgt_T = comupte_transform(
    #     torch.as_tensor(np.asarray(tgt_mesh.vertices)),
    #     torch.as_tensor(np.asarray(smpl_mesh.vertices)),
    #     torch.as_tensor(np.asarray(smpl_mesh.triangles, dtype=int)),
    #     T)

    # %%
    def lbs(W, T, V, inverse=False):
        V_homo = torch.concat([V, torch.ones((V.shape[0], 1), device=V.device)], dim=-1).unsqueeze(dim=-1)
        T = (W @ T).view(-1, 4, 4)
        if inverse:
            T = torch.linalg.inv(T)
        V_homo = T @ V_homo
        return V_homo[:, :3, 0]

    # %%
    from models.xcloth.components.utils import create_o3d_pcd

    device = "cuda"
    W = torch.as_tensor(tgt_weights, device=device).to(dtype=torch.float32)
    T0 = T.to(device=device)
    T1 = original_T.cuda()
    V = torch.as_tensor(np.asarray(tgt_mesh.vertices), device=device).to(dtype=torch.float32)
    V_norm = lbs(W, T0, V, inverse=True)

    W_smpl = src_weights.cuda()
    V_smpl = torch.as_tensor(np.asarray(smpl_mesh.vertices), device=device).to(dtype=torch.float32)
    v_smpl_norm = lbs(W_smpl, T0, V_smpl, inverse=True)

    pcd1 = create_o3d_pcd(lbs(W, T1, V_norm).detach().cpu())
    pcd2 = create_o3d_pcd(v_smpl_norm.cpu().numpy(), colors=np.ones(V_smpl.shape)/2)

    o3d.visualization.draw_geometries([pcd1, pcd2])

    # %%
    o3d.visualization.draw_geometries([pcd2])

    # %%
    tgt_mesh.vertices = pcd1.points

    # %%
    o3d.visualization.draw_geometries([pcd2, tgt_mesh])


