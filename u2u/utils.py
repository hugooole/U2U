# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os
import pathlib
from typing import Optional, Tuple

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.linalg import polar as sci_polar
from scipy.spatial.transform import Rotation
from tetgen import TetGen
from uipc import Transform, view
from uipc.geometry import SimplicialComplex
from uipc.geometry import linemesh as linemesh_fn
from uipc.geometry import pointcloud as pointcloud_fn
from uipc.geometry import tetmesh as tetmesh_fn
from uipc.geometry import trimesh as trimesh_fn

from u2u.pose import Pose

from .mesh_factory import Mesh


class AssetDir:
    this_file = pathlib.Path(os.path.dirname(__file__)).resolve()
    _output_path = pathlib.Path(this_file / "../output/").resolve()
    _assets_path = pathlib.Path(this_file / "../assets/").resolve()
    _tetmesh_path = _assets_path / "tetmesh"
    _trimesh_path = _assets_path / "trimesh"
    _urdf_path = _assets_path / "urdf"
    _usd_path = _assets_path / "usd"
    _houdini_path = _assets_path / "houdini"
    _config_path = pathlib.Path(this_file / "../configs/").resolve()
    _docs_path = pathlib.Path(this_file / "../docs/").resolve()

    @staticmethod
    def asset_path():
        # 返回资源目录的路径
        return str(AssetDir._assets_path)

    @staticmethod
    def tetmesh_path():
        return str(AssetDir._tetmesh_path)

    @staticmethod
    def trimesh_path():
        return str(AssetDir._trimesh_path)

    @staticmethod
    def urdf_path():
        return str(AssetDir._urdf_path)

    @staticmethod
    def usd_path():
        return str(AssetDir._usd_path)

    @staticmethod
    def docs_path():
        return str(AssetDir._docs_path)

    @staticmethod
    def output_path(file):
        file_dir = pathlib.Path(file).absolute()
        this_python_root = AssetDir.this_file.parent
        # get the relative path from the python root to the file
        relative_path = file_dir.relative_to(this_python_root)
        # construct the output path
        output_dir = AssetDir._output_path / relative_path / ""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return str(output_dir)

    @staticmethod
    def houdini_path():
        return str(AssetDir._houdini_path)

    @staticmethod
    def config_path():
        return str(AssetDir._config_path)

    @staticmethod
    def folder(file):
        return pathlib.Path(file).absolute().parent


def angular_velocity_to_rotation_matrix_dot(
    angular_velocity: np.ndarray, rotation_matrix: np.ndarray, degrees: bool = False
) -> np.ndarray:
    """
    Convert an angular velocity vector into the derivative of rotation matrix
    R^dot = Omega @ R, where Omega is the skew-symmetric matrix of the angular velocity vector

    Args:
        angular_velocity (np.ndarray): shape is [3]
        rotation_matrix (np.ndarray): shape is [3, 3]
        degrees (bool, optional): whether the angular velocity is in degrees. Defaults to False
    Returns:
        derivative of rotation matrix: np.ndarray: shape is [3, 3]
    """

    assert angular_velocity.shape == (3,), "angular_velocity must be a vector of length 3"
    assert rotation_matrix.shape == (3, 3), "rotation_matrix must be a 3x3 matrix"

    if degrees:
        angular_velocity = np.deg2rad(angular_velocity)

    # Get the skew-symmetric matrix of the angular velocity vector
    omega = np.array(
        [
            [0, -angular_velocity[2], angular_velocity[1]],
            [angular_velocity[2], 0, -angular_velocity[0]],
            [-angular_velocity[1], angular_velocity[0], 0],
        ],
        dtype=np.float64,
    )

    return omega @ rotation_matrix.astype(np.float64)


def generate_tetrahedral_grid(points, faces):
    """
    Generate a tetrahedral grid from the provided points and faces using PyVista and TetGen.

    This function takes a set of points and faces, creates a PolyData object,
    triangulates it, and then applies TetGen library to generate a tetrahedral
    grid. Tetrahedral grids are essential for finite element analysis, spatial
    meshing, and various computational geometry applications.

    :param points: Array of XYZ coordinates representing the points in the mesh.
        The points define the vertices of the geometry used to create the
        PolyData object.
    :type points: numpy.ndarray
    :param faces: Array representing the face connectivity. This defines
        how vertices are used together to form the faces of the geometry.
        which format must be [n_faces, ind0, ind1, ind2, ..., ind_n, ...]
    :type faces: numpy.ndarray

    :return: Output tetrahedral grid generated after processing via TetGen.
        Contains volumetric elements suitable for computational modeling.
    :rtype: pyvista.UnstructuredGrid
    """
    import pyvista as pv

    # Create a PyVista PolyData object
    pv_mesh = pv.PolyData(points, faces)
    tri_mesh = pv_mesh.triangulate()
    tet = TetGen(tri_mesh)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    grid = tet.grid
    return grid


def get_transform(prim) -> np.ndarray:
    xformable = UsdGeom.Xformable(prim)
    gf_mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    gf_mat_np = np.array(gf_mat.GetTranspose(), dtype=np.float64).reshape(4, 4)
    return gf_mat_np


def get_position_and_orientation(prim, unit) -> Pose:
    xformable = UsdGeom.Xformable(prim)
    gf_mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = gf_mat.ExtractTranslation()
    rotation_mat = gf_mat.ExtractRotation()
    orientation = Gf.Quatf(rotation_mat.GetQuat())
    p = np.asarray(translation * unit)
    q = np.asarray(
        [
            orientation.GetReal(),
            orientation.GetImaginary()[0],
            orientation.GetImaginary()[1],
            orientation.GetImaginary()[2],
        ]
    )
    return Pose(p=p, q=q)


def transform_and_scale_points(points_np, t: Transform, meters_per_unit: float = 1.0):
    points_np = t.apply_to(points_np)
    points_np *= meters_per_unit
    return points_np


def get_mass_density(prim, default_value: float):
    mass_density = default_value
    if prim.HasAPI(UsdPhysics.MassAPI):
        usd_mass = UsdPhysics.MassAPI(prim)
        mass_density = usd_mass.GetDensityAttr().Get() or default_value
    return mass_density


def create_simplicial_complex(
    mesh: Mesh, pose: Optional[np.ndarray] = None, tethedralize: bool = False
) -> SimplicialComplex:
    """
    Create a simplicial complex from the provided mesh data.

    This function generates a simplicial complex using the vertices and faces of
    the given mesh. It supports optional tetrahedralization, which creates a
    tetrahedral grid-based representation of the mesh data. If a pose matrix is
    provided, it is applied as a transformation to the mesh.

    :param mesh:
        The input mesh object containing points and faces for generating
        the simplicial complex.
    :param pose:
        A 4x4 transformation matrix in numpy.ndarray format,
        applied as a pose to the mesh. Defaults to None.
    :param tethedralize:
        A boolean flag indicating whether to tetrahedralize the mesh.
        If True, the output mesh will be tetrahedralized. Defaults to False.
    :return:
        A simplicial complex generated from the input mesh, with the
        specified transformations and configurations applied.
    """
    # Create a new simplicial complex from mesh vertices and faces
    # ensuring correct data types for numerical stability

    # 3D
    sc: SimplicialComplex = None
    if tethedralize:
        if mesh.faces.ndim == 2:
            num_faces = mesh.faces.shape[-1]
            faces_count = np.full((mesh.faces.shape[0],), num_faces)
            mesh.faces = np.hstack((faces_count.reshape(-1, 1), mesh.faces)).flatten()

        grid = generate_tetrahedral_grid(mesh.points, mesh.faces)
        cells = grid.cells.reshape(-1, 5)[:, 1:]
        sc = tetmesh_fn(np.array(grid.points, dtype=np.float64), cells)
    else:
        # Codim 1D : pointcloud
        if mesh.faces is None:
            sc = pointcloud_fn(mesh.points.astype(np.float64))
        elif mesh.faces.shape[1] == 2:
            sc = linemesh_fn(mesh.points.astype(np.float64), mesh.faces.astype(np.int32))
        elif mesh.faces.shape[1] == 3:
            sc = trimesh_fn(mesh.points.astype(np.float64), mesh.faces.astype(np.int32))
        else:
            assert False, f"Unsupported mesh face indices count: {mesh.faces.shape[1]}"

    if pose is not None:
        # Convert the input pose to a Transform object with float64 precision
        pose_transform = Transform(pose.astype(np.float64))

        # Apply the transformation matrix to the mesh's transform buffer
        view(sc.transforms())[:] = pose_transform.matrix()

    return sc


def extract_rot_and_scale_from_transform(
    transform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rotation, scale = sci_polar(transform[:3, :3], side="right")
    scale_mat = np.eye(4, dtype=np.float64)
    scale_mat[:3, :3] = scale
    return rotation, scale_mat


def gf_quat_to_rotation(gf_quat: Gf.Quatf) -> Rotation:
    real_part: float = gf_quat.GetReal()
    imag_part: Gf.Vec3f = gf_quat.GetImaginary()
    return Rotation.from_quat([imag_part[0], imag_part[1], imag_part[2], real_part])


def orthogonalize_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Orthogonalize a 3x3 rotation matrix using SVD (Polar Decomposition).
    Ensures det(R) = +1 (proper rotation, not reflection).

    This fixes numerical errors from physics simulations that can produce
    slightly non-orthogonal matrices, causing quaternion conversion failures.
    """
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt

    # Fix determinant to +1 if it's -1 (reflection)
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt

    return R_ortho
