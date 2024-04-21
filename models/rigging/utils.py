import open3d as o3d
import numpy as np  
from sklearn.preprocessing import normalize

def compute_closest_position(src_mesh, tgt_mesh):
    t_src = o3d.t.geometry.TriangleMesh.from_legacy(src_mesh)
    queries = np.asarray(tgt_mesh.vertices, dtype=np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_src)

    result = scene.compute_closest_points(queries)
    positions = result["points"].numpy()
    distances = np.linalg.norm(positions - queries, axis=-1)
    face_ids = result["primitive_ids"].numpy()
    face_normals = result["primitive_normals"].numpy()

    uv = result["primitive_uvs"].numpy()
    barycentric_weights = np.zeros_like(queries)
    barycentric_weights[:, 1] = uv[:, 0]
    barycentric_weights[:, 2] = uv[:, 1]
    barycentric_weights[:, 0] = 1 - barycentric_weights.sum(axis=-1)

    return positions, distances, barycentric_weights, face_ids, face_normals


def composite_bary_weights(src_mesh, src, bary_weights, face_ids):
    faces = np.asarray(src_mesh.triangles)[face_ids]
    weights = src[faces.flatten()].reshape((*faces.shape, -1))
    bary_weights = normalize(bary_weights, axis=1)
    return (np.expand_dims(bary_weights, axis=-1) * weights).sum(axis=1)