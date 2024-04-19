"""
https://github.com/rin-23/RobustSkinWeightsTransferCode
"""
import sys
import time

import numpy as np
# import numpy.linalg as nplinalg
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.spatial import cKDTree
from pypardiso import spsolve

import open3d as o3d
from xcloth.components.utils import compute_closest_point

if sys.version_info > (3, 0):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import (
            Optional,  # noqa: F401
            Dict,  # noqa: F401
            List,  # noqa: F401
            Tuple,  # noqa: F401
            Pattern,  # noqa: F401
            Callable,  # noqa: F401
            Any,  # noqa: F401
            Text,  # noqa: F401
            Generator,  # noqa: F401
            Union  # noqa: F401
        )


from logging import (
    getLogger,
    INFO,  # noqa: F401
)
logger = getLogger(__name__)
logger.setLevel(INFO)

########################################################################################################################

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} の実行時間: {end_time - start_time} 秒")
        return result
    return wrapper


@timeit
def create_vertex_data_array(mesh):
    # type: (o3d.geometry.TriangleMesh) -> np.ndarray
    """Create a structured numpy array containing vertex index, position, and normal."""

    vertex_data = np.zeros(
            len(mesh.vertices),
            dtype=[
                ("index", np.int64),
                ("position", np.float64, 3),
                ("normal", np.float64, 3),
                ("face_index", np.int64),
                ("weights", np.float64, 3)
            ])
    
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    for i in range(vertices.shape[0]):
        position = vertices[i]  # type: ignore
        normal = normals[i]  # type: ignore
        vertex_data[i] = (
            i,
            position,
            normal,
            -1,
            np.zeros(3)
        )

    return vertex_data


@timeit
def get_closest_points(source_mesh, target_vertex_data):
    # type: (om.MFnMesh, np.ndarray) -> np.ndarray
    """get closest points and return a structured numpy array similar to target_vertex_data."""

    closest_points_data = np.zeros(target_vertex_data.shape, dtype=target_vertex_data.dtype)
    num_vertices = target_vertex_data.shape[0]

    closest_points = compute_closest_point(source_mesh, np.asarray([data["position"] for data in target_vertex_data], dtype=np.float32))
    vertex_normals = np.asarray(source_mesh.vertex_normals)
    faces = np.asarray(source_mesh.triangles)

    for i in range(num_vertices):       
        # Get closest point on source mesh
        pos = closest_points["points"][i].numpy()
        face_index = closest_points["primitive_ids"][i].item()

        u, v = closest_points["primitive_uvs"][i].numpy()
        weights = np.array([1 - u - v, u, v])
        
        norm = weights.reshape(3, -1) * vertex_normals[faces[face_index]]
        norm = norm.sum(axis=0)

        # Store target vertex index, closest point position, and closest point normal
        closest_points_data[i] = (
            target_vertex_data[i]["index"],
            pos,
            norm,
            face_index,
            weights
        )
        
    return closest_points_data


@timeit
def get_closest_points_by_kdtree(source_mesh, target_vertex_data):
    # type: (om.MFnMesh, np.ndarray) -> np.ndarray
    """get closest points and return a structured numpy array similar to target_vertex_data."""

    source_vertex_data = create_vertex_data_array(source_mesh)
    B_positions = np.array([vertex["position"] for vertex in source_vertex_data])
    A_positions = np.array([vertex["position"] for vertex in target_vertex_data])

    tree = cKDTree(B_positions)
    _, indices = tree.query(A_positions)

    nearest_in_B_for_A = source_vertex_data[indices]
    return nearest_in_B_for_A


@timeit
def filter_high_confidence_matches(target_vertex_data, closest_points_data, max_distance, max_angle):
    # type: (np.ndarray, np.ndarray, float, float) -> List[int]
    """filter high confidence matches using structured arrays."""

    target_positions = target_vertex_data["position"]
    target_normals = target_vertex_data["normal"]
    source_positions = closest_points_data["position"]
    source_normals = closest_points_data["normal"]

    # Calculate distances (vectorized)
    distances = np.linalg.norm(source_positions - target_positions, axis=1)

    # Calculate angles between normals (vectorized)
    cos_angles = np.einsum("ij,ij->i", source_normals, target_normals)
    cos_angles /= np.linalg.norm(source_normals, axis=1) * np.linalg.norm(target_normals, axis=1)
    cos_angles = np.abs(cos_angles)  # Consider opposite normals by taking absolute value
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi

    # Apply thresholds (vectorized)
    high_confidence_indices = np.where((distances <= max_distance) & (angles <= max_angle))[0]

    return high_confidence_indices.tolist()


@timeit
def copy_weights_for_confident_matches(source_mesh, source_weights, confident_vertex_indices, closest_points_data):
    # type: (om.MFnMesh, om.MFnMesh, List[int], np.ndarray) -> Dict[int, np.ndarray]
    """copy weights for confident matches."""

    # source_skin_cluster_name = get_skincluster(source_mesh.name())
    # source_skin_cluster = as_mfn_skin_cluster(source_skin_cluster_name)
    # deformer_bones = cmds.skinCluster(source_skin_cluster_name, query=True, influence=True)

    # target_skin_cluster_name = get_or_create_skincluster(target_mesh.name(), deformer_bones)
    # target_skin_cluster = as_mfn_skin_cluster(target_skin_cluster_name)

    known_weights = {}  # type: Dict[int, np.ndarray]
    faces = np.asarray(source_mesh.triangles)

    # copy weights
    for i in confident_vertex_indices:
        src_face_index = closest_points_data[i]["face_index"]

        if src_face_index < 0:
            continue

        weights = closest_points_data[i]["weights"].reshape(3, -1) * source_weights[faces[src_face_index]]
        weights = weights.sum(axis=0)

        if len(weights) <= 0:
            continue

        known_weights[i] = np.array(weights)

    return known_weights


def add_laplacian_entry_in_place(L, tri_positions, tri_indices):
    # type: (sp.lil_matrix, np.ndarray, np.ndarray) -> None
    """add laplacian entry.

    CAUTION: L is modified in-place.
    """

    i1 = tri_indices[0]
    i2 = tri_indices[1]
    i3 = tri_indices[2]

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]

    # calculate cotangent
    cotan1 = compute_cotangent(v2, v1, v3)
    cotan2 = compute_cotangent(v1, v2, v3)

    # update laplacian matrix
    L[i1, i2] += cotan1  # type: ignore
    L[i2, i1] += cotan1  # type: ignore
    L[i1, i1] -= cotan1  # type: ignore
    L[i2, i2] -= cotan1  # type: ignore

    L[i2, i3] += cotan2  # type: ignore
    L[i3, i2] += cotan2  # type: ignore
    L[i2, i2] -= cotan2  # type: ignore
    L[i3, i3] -= cotan2  # type: ignore


def add_area_in_place(areas, tri_positions, tri_indices):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> None
    """add area.

    CAUTION: areas is modified in-place.
    """

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    for idx in tri_indices:
        areas[idx] += area


def compute_laplacian_and_mass_matrix(mesh):
    # type: (om.MFnMesh) -> Tuple[sp.csr_array, sp.dia_matrix]
    """compute laplacian matrix from mesh.

    treat area as mass matrix.
    """

    # initialize sparse laplacian matrix
    n_vertices = len(mesh.vertices)
    L = sp.lil_matrix((n_vertices, n_vertices))
    areas = np.zeros(n_vertices)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    for face in faces:
        tri_positions = vertices[face]
        add_laplacian_entry_in_place(L, tri_positions, face)
        add_area_in_place(areas, tri_positions, face)

    L_csr = L.tocsr()
    M_csr = sp.diags(areas)

    return L_csr, M_csr


def compute_cotangent(v1, v2, v3):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> float
    """compute cotangent from three points."""

    edeg1 = v2 - v1
    edeg2 = v3 - v1

    norm1 = np.cross(edeg1, edeg2)
    area = np.linalg.norm(norm1)

    cotan = edeg1.dot(edeg2) / area

    return cotan


def __do_inpainting(mesh, known_weights):
    # type: (om.MFnMesh, Dict[int, np.ndarray]) -> np.ndarray

    L, M = compute_laplacian_and_mass_matrix(mesh)
    Q = -L + L @ sp.diags(np.reciprocal(M.diagonal())) @ L

    S_match = np.array(list(known_weights.keys()))
    S_nomatch = np.array(list(set(range(len(mesh.vertices))) - set(S_match)))

    Q_UU = sp.csr_matrix(Q[np.ix_(S_nomatch, S_nomatch)])
    Q_UI = sp.csr_matrix(Q[np.ix_(S_nomatch, S_match)])

    num_vertices = len(mesh.vertices)
    num_bones = len(next(iter(known_weights.values())))

    W = np.zeros((num_vertices, num_bones))
    for i, weights in known_weights.items():
        W[i] = weights

    W_I = W[S_match, :]
    W_U = W[S_nomatch, :]

    for bone_idx in range(num_bones):
        b = -Q_UI @ W_I[:, bone_idx]
        try:
            W_U[:, bone_idx] = splinalg.lsqr(Q_UU, b)[0]
        except:
            pass

    W[S_nomatch, :] = W_U

    # apply constraints,

    # each element is between 0 and 1
    W = np.clip(W, 0.0, 1.0)

    # normalize each row to sum to 1
    W = W / W.sum(axis=1, keepdims=True) + 1e-8

    return W


def compute_weights_for_remaining_vertices(target_mesh, known_weights):
    # type: (om.MFnMesh, Dict[int, np.ndarray]) -> np.ndarray
    """compute weights for remaining vertices."""
   
    try:
        optimized = __do_inpainting(target_mesh, known_weights)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error: {}".format(e))
        raise

    return optimized


def calculate_threshold_distance(mesh, threadhold_ratio=0.05):
    # type: (o3d.geometry.TriangleMesh, float) -> float
    """Returns 𝑑𝑏𝑜𝑥 * 0.05

    𝑑𝑏𝑜𝑥 is the target mesh bounding box diagonal length.
    """

    bbox_min = mesh.get_min_bound()
    bbox_max = mesh.get_max_bound()
    bbox_diag = bbox_max - bbox_min
    bbox_diag_length = np.linalg.norm(bbox_diag)

    threshold_distance = bbox_diag_length * threadhold_ratio

    return threshold_distance


def segregate_vertices_by_confidence(src_mesh, dst_mesh, threshold_distance=0.05, threshold_angle=25.0, use_kdtree=False):
    # type: (o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh, float, float, bool) -> Tuple[List[int], List[int]]

    threshold_distance = calculate_threshold_distance(dst_mesh, threshold_distance)
    target_vertex_data = create_vertex_data_array(dst_mesh)

    if use_kdtree:
        closest_points_data = get_closest_points_by_kdtree(src_mesh, target_vertex_data)
    else:
        closest_points_data = get_closest_points(src_mesh, target_vertex_data)

    confident_vertex_indices = filter_high_confidence_matches(target_vertex_data, closest_points_data, threshold_distance, threshold_angle)
    unconvinced_vertex_indices = list(set(range(len(dst_mesh.vertices))) - set(confident_vertex_indices))

    return confident_vertex_indices, unconvinced_vertex_indices


def main(source_mesh, target_mesh, source_weights, threshold_distance=0.05, threshold_angle=25.0):
    # setup
    tmp = segregate_vertices_by_confidence(source_mesh, target_mesh, threshold_distance=threshold_distance, threshold_angle=threshold_angle)
    target_vertex_data = create_vertex_data_array(target_mesh)

    # confidence
    confident_vertex_indices = tmp[0]
    unconvinced_vertex_indices = tmp[1]

    closest_points_data = get_closest_points(source_mesh, target_vertex_data)
    known_weights = copy_weights_for_confident_matches(source_mesh, source_weights, confident_vertex_indices, closest_points_data)

    # inpainting
    optimized_weights = compute_weights_for_remaining_vertices(target_mesh, known_weights)
    return optimized_weights, confident_vertex_indices, unconvinced_vertex_indices
    # apply_weight_inpainting(target_mesh, optimized_weights, unconvinced_vertex_indices)


if __name__ == "__main__":
    main()