# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from abc import ABC, abstractmethod

import numpy as np
import trimesh
from loguru import logger
from pxr import UsdGeom

from .usd_utils import get_prim_type_name


def _axis_transform(axis: str) -> np.ndarray | None:
    """Return a 4x4 transform that reorients trimesh geometry (along Z) to the given USD axis.

    Returns None when no rotation is needed (axis="Z").
    """
    axis = axis.upper()
    if axis == "Z":
        return None
    t = np.eye(4, dtype=np.float64)
    if axis == "X":
        # Z→X: rotate 90° around Y
        t[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    elif axis == "Y":
        # Z→Y: rotate -90° around X
        t[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    return t


def get_points_and_faces(prim):
    """
    Extracts points and face definitions of a USD (Universal Scene Description) geometry
    as a PyVista-compatible format.

    The method converts the given USD mesh primitive into corresponding points and
    face information that can be used in PyVista for further processing or visualization.
    The face data structure is transformed from USD's format to PyVista's expected format
    by including a count of vertices per face followed by the vertex indices.

    :param prim: USD geometry primitive representing the mesh.
    :type prim: UsdPrim
    :return:
        A tuple containing:
        - A NumPy array of 3D points defining the vertices of the mesh.
        - A NumPy array defining the faces in PyVista-compatible format.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    mesh_api = UsdGeom.Mesh(prim)
    points = np.array(mesh_api.GetPointsAttr().Get())
    face_vertex_counts = list(mesh_api.GetFaceVertexCountsAttr().Get())
    face_vertex_indices = list(mesh_api.GetFaceVertexIndicesAttr().Get())
    # Convert to PyVista format
    faces = []
    i = 0
    for count in face_vertex_counts:
        face = [count] + face_vertex_indices[i : i + count]
        faces.extend(face)
        i += count
    faces = np.array(faces)
    return points, faces


def repair_mesh(pv_mesh, prim_path: str = ""):
    """Repair a non-closed mesh using a two-stage strategy.

    Stage 1: PyVista fill_holes (fast, handles simple cases).
    Stage 2: pymeshfix deep repair (handles complex topology issues).

    Args:
        pv_mesh: A PyVista PolyData mesh to repair.
        prim_path: USD prim path for logging context.

    Returns:
        A repaired PyVista PolyData mesh (new object, original is not modified).
    """
    import pyvista as pv

    # Stage 1: Try PyVista fill_holes first (fast path)
    fill_holes_result = pv_mesh.fill_holes(1000.0)
    if fill_holes_result.is_manifold and fill_holes_result.n_open_edges == 0:
        logger.debug(f"Mesh {prim_path} repaired successfully with PyVista fill_holes.")
        return fill_holes_result

    # Stage 2: Use pymeshfix for deep repair
    try:
        import pymeshfix
    except ImportError:
        logger.warning(
            f"Mesh {prim_path} could not be fully repaired with fill_holes and pymeshfix is not installed. "
            "Install it with: uv add pymeshfix"
        )
        return fill_holes_result

    try:
        logger.debug(f"Mesh {prim_path} fill_holes insufficient, attempting pymeshfix deep repair...")
        points = np.array(pv_mesh.points)
        faces_raw = np.array(pv_mesh.faces)
        tri_faces = faces_raw.reshape(-1, 4)[:, 1:4]

        fixer = pymeshfix.MeshFix(points, tri_faces, verbose=False)
        fixer.repair()

        repaired_points = np.array(fixer.points)
        repaired_faces_tri = np.array(fixer.faces)

        if repaired_points.shape[0] == 0 or repaired_faces_tri.shape[0] == 0:
            logger.warning(f"Mesh {prim_path} pymeshfix returned empty mesh, falling back to fill_holes result.")
            return fill_holes_result

        n_faces = repaired_faces_tri.shape[0]
        pv_faces = np.column_stack([np.full(n_faces, 3, dtype=np.int32), repaired_faces_tri]).flatten()
        repaired = pv.PolyData(repaired_points, pv_faces)

        if repaired.is_manifold and repaired.n_open_edges == 0:
            logger.debug(f"Mesh {prim_path} repaired successfully with pymeshfix.")
            return repaired

        logger.warning(
            f"Mesh {prim_path} still has issues after pymeshfix "
            f"(manifold={repaired.is_manifold}, open_edges={repaired.n_open_edges}), "
            "using best available result."
        )
        return repaired
    except Exception as e:
        logger.warning(f"Mesh {prim_path} pymeshfix repair failed: {e}. Using fill_holes result.")
        return fill_holes_result


class Mesh(ABC):
    """Base class for all mesh types.

    This class defines the common interface and functionality for all mesh types.
    Subclasses must implement the _create_trimesh method to create a trimesh object
    specific to their shape.
    """

    def __init__(self, prim):
        """Initialize a mesh from a USD primitive.

        Args:
            prim: USD primitive representing the mesh

        Attributes:
            _api: USD geometry API (set by subclasses)
            _points: Vertex coordinates array
            _faces: Face indices array
            _approx_info: Mesh approximation metadata (None if not approximated)
        """
        self._api = None  # Will be set by subclasses
        self._points = None
        self._faces = None
        self._approx_info = None

    def _apply_approximation(self, prim, config: dict):
        """Apply mesh approximation to this mesh.

        Replaces the mesh geometry with an approximation (e.g., convex hull).
        The approximation result is stored in _approx_info.

        Args:
            prim: USD primitive (for logging context)
            config: Approximation configuration dict with:
                - method: One of "convex_hull", "bounding_box", "bounding_sphere",
                         "coacd", "quadratic"
                - params: Optional method-specific parameters (e.g., coacd threshold)
        """
        method = config.get("method")
        params = config.get("params")
        prim_path = str(prim.GetPath())
        self._approx_info = approximate_mesh(mesh=self, prim_path=prim_path, method=method, params=params)

    @property
    def approx_info(self) -> dict | None:
        """Get mesh approximation metadata.

        Returns:
            dict with approximation details if approximation was applied:
                - method: Approximation method used
                - prim_path: USD prim path
                - orig_verts/faces: Original mesh counts
                - new_verts/faces: Approximated mesh counts
                - Additional method-specific keys (e.g., threshold, parts)
            None if no approximation was applied.
        """
        return self._approx_info

    @abstractmethod
    def _create_trimesh(self):
        """Create a trimesh object for this mesh type.

        Returns:
            A trimesh object representing the mesh
        """
        pass

    def _initialize_points_and_faces(self, trimesh_obj):
        """Initialize points and faces from a trimesh object.

        Args:
            trimesh_obj: A trimesh object
        """
        self._points = trimesh_obj.vertices.astype(np.float64)
        self._faces = trimesh_obj.faces.astype(np.int32)

    @property
    def points(self) -> np.ndarray:
        """Get the mesh vertices.

        Returns:
            numpy.ndarray: Array of vertex coordinates
        """
        return self._points

    @points.setter
    def points(self, value: np.ndarray):
        """Set the mesh vertices.

        Args:
            value (numpy.ndarray): Array of vertex coordinates
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Points must be a numpy array")
        self._points = value.astype(np.float64)

    @property
    def faces(self) -> np.ndarray:
        """Get the mesh faces.

        Returns:
            numpy.ndarray: Array of face indices
        """
        return self._faces

    @faces.setter
    def faces(self, value: np.ndarray):
        """Set the mesh faces.

        Args:
            value (numpy.ndarray): Array of face indices
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Faces must be a numpy array")
        self._faces = value.astype(np.int32)


class Cube(Mesh):
    """Mesh representation of a USD Cube primitive."""

    def __init__(self, prim: UsdGeom.Cube):
        """Initialize a cube mesh from a USD Cube primitive.

        Args:
            prim: USD Cube primitive
        """
        super().__init__(prim)
        self._api = UsdGeom.Cube(prim)
        self._size = self._api.GetSizeAttr().Get() or 1.0
        trimesh_obj = self._create_trimesh()
        self._initialize_points_and_faces(trimesh_obj)

    def _create_trimesh(self):
        """Create a box trimesh.

        Returns:
            A trimesh box with the specified size
        """
        return trimesh.creation.box(extents=(self._size, self._size, self._size))


class Sphere(Mesh):
    """Mesh representation of a USD Sphere primitive."""

    def __init__(self, prim: UsdGeom.Sphere):
        """Initialize a sphere mesh from a USD Sphere primitive.

        Args:
            prim: USD Sphere primitive
        """
        super().__init__(prim)
        self._api = UsdGeom.Sphere(prim)
        self._radius = self._api.GetRadiusAttr().Get() or 1.0
        trimesh_obj = self._create_trimesh()
        self._initialize_points_and_faces(trimesh_obj)

    def _create_trimesh(self):
        """Create a sphere trimesh.

        Returns:
            A trimesh icosphere with the specified radius
        """
        return trimesh.creation.icosphere(subdivisions=3, radius=self._radius)


class Cylinder(Mesh):
    """Mesh representation of a USD Cylinder primitive."""

    def __init__(self, prim: UsdGeom.Cylinder):
        super().__init__(prim)
        self._api = UsdGeom.Cylinder(prim)
        self._radius = self._api.GetRadiusAttr().Get() or 1.0
        self._height = self._api.GetHeightAttr().Get() or 1.0
        self._axis = str(self._api.GetAxisAttr().Get() or "Y")
        trimesh_obj = self._create_trimesh()
        self._initialize_points_and_faces(trimesh_obj)

    def _create_trimesh(self):
        return trimesh.creation.cylinder(
            radius=self._radius, height=self._height, transform=_axis_transform(self._axis)
        )


class Cone(Mesh):
    """Mesh representation of a USD Cone primitive."""

    def __init__(self, prim: UsdGeom.Cone):
        super().__init__(prim)
        self._api = UsdGeom.Cone(prim)
        self._radius = self._api.GetRadiusAttr().Get() or 1.0
        self._height = self._api.GetHeightAttr().Get() or 1.0
        self._axis = str(self._api.GetAxisAttr().Get() or "Y")
        trimesh_obj = self._create_trimesh()
        self._initialize_points_and_faces(trimesh_obj)

    def _create_trimesh(self):
        return trimesh.creation.cone(radius=self._radius, height=self._height, transform=_axis_transform(self._axis))


class Capsule(Mesh):
    """Mesh representation of a USD Capsule primitive."""

    def __init__(self, prim: UsdGeom.Capsule):
        super().__init__(prim)
        self._api = UsdGeom.Capsule(prim)
        self._radius = self._api.GetRadiusAttr().Get() or 1.0
        self._height = self._api.GetHeightAttr().Get() or 1.0
        self._axis = str(self._api.GetAxisAttr().Get() or "Y")
        trimesh_obj = self._create_trimesh()
        self._initialize_points_and_faces(trimesh_obj)

    def _create_trimesh(self):
        return trimesh.creation.capsule(radius=self._radius, height=self._height, transform=_axis_transform(self._axis))


class BasisCurves(Mesh):
    """Mesh representation of a USD BasisCurves primitive.

    This class handles the conversion of BasisCurves to a mesh format.
    """

    def __init__(self, prim: UsdGeom.BasisCurves):
        """Initialize a mesh from a USD BasisCurves primitive.

        Args:
            prim: USD BasisCurves primitive
        """
        super().__init__(prim)
        self._api = UsdGeom.BasisCurves(prim)
        ps = self._api.GetPointsAttr().Get()
        self._points = np.array(ps, dtype=np.float64).reshape(-1, 3)
        v_counts = self._api.GetCurveVertexCountsAttr().Get()
        offset = 0
        edges = []
        for vc in v_counts:
            offset += vc
            segments = self.divide_into_segments(vc)
            edges.extend(segments)
        if self._api.GetWrapAttr().Get() == "periodic":
            edges.append([offset - 1, 0])  # Connect last point to first for periodic curves
        edges = np.array(edges, dtype=np.int32)
        self._faces = edges.reshape(-1, 2)

    def divide_into_segments(self, count) -> np.ndarray:
        # e.g.
        # input  4
        # output [0 1, 1 2, 2 3]
        segments = []
        assert count >= 2, "Curve must have at least two points to form segments."
        for i in range(count - 1):
            segments.append([i, i + 1])
        assert segments[-1][1] == count - 1, f"Last segment end {segments[-1][1]} != {count - 1}"
        return np.array(segments)

    def _create_trimesh(self):
        pass


class Points(Mesh):
    """Mesh representation of a USD Points primitive.

    This class handles the conversion of Points to a mesh format.
    """

    def __init__(self, prim: UsdGeom.Points):
        """Initialize a mesh from a USD Points primitive.

        Args:
            prim: USD Points primitive
        """
        super().__init__(prim)
        self._api = UsdGeom.Points(prim)
        ps = self._api.GetPointsAttr().Get()
        assert len(ps) > 0, f"Points{prim.GetPath()} must have at least one point"

        ps = np.array(ps, dtype=np.float64)
        self._points = ps.reshape(-1, 3)  # Ensure points are in Nx3 format
        self._faces = None  # Points do not have faces

    def _create_trimesh(self):
        pass


class GeomMesh(Mesh):
    """Mesh representation of a USD Mesh primitive.

    This class handles arbitrary mesh geometry from USD.
    """

    def __init__(
        self,
        prim: UsdGeom.Mesh,
        need_closed: bool = True,
        need_triangulate: bool = True,
        approximation_config: dict | None = None,
    ):
        """Initialize a mesh from a USD Mesh primitive.

        Args:
            prim: USD Mesh primitive
            need_closed: Whether to ensure mesh is watertight (ignored if approximation_config provided)
            need_triangulate: Whether to triangulate non-triangle meshes
            approximation_config: Optional dict with 'method' and 'params' keys for mesh approximation
        """
        super().__init__(prim)
        self._api = UsdGeom.Mesh(prim)
        self.need_closed = need_closed
        self.need_triangulate = need_triangulate
        self.approximation_config = approximation_config
        self._process_mesh(prim)

    def _process_mesh(
        self,
        prim,
    ):
        """Process the mesh data using pyvista.

        This method extracts points and faces from the USD mesh and ensures
        the mesh is properly triangulated and watertight (closed).

        If approximation_config is provided, mesh approximation is applied
        instead of mesh repair.

        Args:
            prim: USD Mesh primitive
        """
        import pyvista as pv

        points_np, faces_np = get_points_and_faces(prim)
        pv_mesh = pv.PolyData(points_np, faces_np)

        if not pv_mesh.is_all_triangles and not self.need_triangulate:
            self._points = np.array(pv_mesh.points, dtype=np.float64)
            faces_raw = np.array(pv_mesh.faces, dtype=np.int32)
            self._faces = faces_raw.reshape(-1, 5)  # Each face: [4, v0, v1, v2, v3]
            self._faces = self._faces[:, 1:5]  # Keep only the vertex indices
            # check orientation
            if self._api.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
                # v0 v1 v2 v3 -> v0 v3 v2 v1
                self._faces[:, [1, 3]] = self._faces[:, [3, 1]]
            # Apply approximation if configured
            if self.approximation_config:
                self._apply_approximation(prim, self.approximation_config)
            return

        # Triangulate if needed
        if not pv_mesh.is_all_triangles and self.need_triangulate:
            logger.warning(f"Mesh {prim.GetPath()} is not triangulated, triangulating now.")
            pv_mesh = pv_mesh.triangulate()

        # If approximation is configured, skip mesh repair and apply approximation
        if self.approximation_config:
            # Store mesh data without repair
            self._points = np.array(pv_mesh.points, dtype=np.float64)
            faces_raw = np.array(pv_mesh.faces, dtype=np.int32)
            faces_reshaped = faces_raw.reshape(-1, 4)  # Each face: [3, v0, v1, v2]
            self._faces = faces_reshaped[:, 1:4]  # Only keep v0, v1, v2

            # check orientation
            if self._api.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
                self._faces[:, [1, 2]] = self._faces[:, [2, 1]]  # flip face orientation

            # Apply approximation
            self._apply_approximation(prim, self.approximation_config)
            return

        # Check if mesh is closed (only when no approximation)
        if self.need_closed:
            if not pv_mesh.is_manifold or pv_mesh.n_open_edges > 0:
                logger.warning(f"Mesh {prim.GetPath()} is not closed, attempting to repair it.")
                pv_mesh = repair_mesh(pv_mesh, prim_path=str(prim.GetPath()))

        # Store mesh data
        self._points = np.array(pv_mesh.points, dtype=np.float64)
        faces_raw = np.array(pv_mesh.faces, dtype=np.int32)
        faces_reshaped = faces_raw.reshape(-1, 4)  # Each face: [3, v0, v1, v2]
        self._faces = faces_reshaped[:, 1:4]  # Only keep v0, v1, v2

        # check orientation
        if self._api.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            self._faces[:, [1, 2]] = self._faces[:, [2, 1]]  # flip face orientation

    def _create_trimesh(self):
        """Not used for GeomMesh as we process the mesh directly.

        This method is implemented to satisfy the abstract method requirement
        but is not used in this subclass.
        """
        pass


def approximate_mesh(mesh: Mesh, prim_path: str, method: str, params: float | None = None) -> dict:
    """Apply mesh approximation by replacing the mesh points and faces in-place.

    Ported from Newton's ``ModelBuilder.approximate_meshes()`` logic.

    Args:
        mesh: Mesh object whose ``points``/``faces`` will be replaced.
        prim_path: USD prim path (used for logging only).
        method: One of ``"convex_hull"``, ``"bounding_box"``, ``"bounding_sphere"``,
            ``"coacd"``, ``"quadratic"``.
        params: Optional method-specific parameters:
            - For "coacd": concavity threshold (default 0.05)
            - For other methods: ignored

    Returns:
        A dict with approximation details (method, orig/new counts, and
        method-specific keys such as ``threshold`` and ``parts`` for coacd).
    """
    info: dict = {"method": method, "prim_path": prim_path}
    points = mesh.points
    faces = mesh.faces
    if points is None or faces is None:
        return info

    orig_verts = points.shape[0]
    orig_faces = faces.shape[0]
    info["orig_verts"] = orig_verts
    info["orig_faces"] = orig_faces
    logger.debug(f"[{prim_path}] Before approximation ({method}): verts={orig_verts}, faces={orig_faces}")

    if method == "convex_hull":
        try:
            hull = trimesh.Trimesh(points, faces).convex_hull
            mesh.points = hull.vertices.astype(np.float64)
            mesh.faces = hull.faces.astype(np.int32)
            logger.debug(f"Applied convex_hull approximation to {prim_path}")
        except Exception as e:
            logger.warning(f"convex_hull failed for {prim_path}: {e}, falling back to bounding_box")
            return approximate_mesh(mesh, prim_path, "bounding_box")

    elif method == "bounding_box":
        try:
            obb = trimesh.Trimesh(points, faces).bounding_box_oriented
            mesh.points = obb.vertices.astype(np.float64)
            mesh.faces = obb.faces.astype(np.int32)
            logger.debug(f"Applied bounding_box approximation to {prim_path}")
        except Exception as e:
            logger.warning(f"bounding_box failed for {prim_path}: {e}")

    elif method == "bounding_sphere":
        try:
            tmesh = trimesh.Trimesh(points, faces)
            center = tmesh.bounding_sphere.primitive.center
            radius = tmesh.bounding_sphere.primitive.radius
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            sphere.apply_translation(center)
            mesh.points = sphere.vertices.astype(np.float64)
            mesh.faces = sphere.faces.astype(np.int32)
            logger.debug(f"Applied bounding_sphere approximation to {prim_path}")
        except Exception as e:
            logger.warning(f"bounding_sphere failed for {prim_path}: {e}, falling back to bounding_box")
            return approximate_mesh(mesh, prim_path, "bounding_box")

    elif method == "coacd":
        try:
            import coacd as _coacd

            _coacd.set_log_level("off")
            cmesh = _coacd.Mesh(points, faces)
            coacd_settings = {
                "threshold": 1.0,
                "mcts_nodes": 20,
                "mcts_iterations": 5,
                "mcts_max_depth": 1,
                "merge": False,
                "max_convex_hull": 5,
            }
            parts = _coacd.run_coacd(cmesh, **coacd_settings)
            if parts:
                # Merge all convex parts into a single mesh
                all_verts = []
                all_faces = []
                vertex_offset = 0
                for part_verts, part_faces in parts:
                    v = np.asarray(part_verts, dtype=np.float64)
                    f = np.asarray(part_faces, dtype=np.int32)
                    all_verts.append(v)
                    all_faces.append(f + vertex_offset)
                    vertex_offset += v.shape[0]
                mesh.points = np.concatenate(all_verts, axis=0)
                mesh.faces = np.concatenate(all_faces, axis=0)
                info["threshold"] = coacd_settings["threshold"]
                info["parts"] = len(parts)
                logger.debug(f"Applied coacd approximation ({len(parts)} parts, merged) to {prim_path}")
            else:
                logger.warning(f"coacd returned empty result for {prim_path}, falling back to convex_hull")
                approximate_mesh(mesh, prim_path, "convex_hull")
        except ImportError:
            logger.warning(f"coacd not installed, falling back to convex_hull for {prim_path}")
            approximate_mesh(mesh, prim_path, "convex_hull")
        except Exception as e:
            logger.warning(f"coacd failed for {prim_path}: {e}, falling back to convex_hull")
            approximate_mesh(mesh, prim_path, "convex_hull")

    elif method == "quadratic":
        try:
            simplified = trimesh.Trimesh(points, faces).simplify_quadric_decimation(face_count=len(faces) // 4)
            mesh.points = simplified.vertices.astype(np.float64)
            mesh.faces = simplified.faces.astype(np.int32)
            logger.debug(f"Applied quadratic simplification to {prim_path}")
        except Exception as e:
            logger.warning(f"quadratic simplification failed for {prim_path}: {e}")

    else:
        logger.warning(f"Unknown approximation method '{method}' for {prim_path}, skipping")
        return info

    new_verts = mesh.points.shape[0] if mesh.points is not None else 0
    new_faces = mesh.faces.shape[0] if mesh.faces is not None else 0
    info["new_verts"] = new_verts
    info["new_faces"] = new_faces
    logger.debug(f"[{prim_path}] After approximation ({method}): verts={new_verts}, faces={new_faces}")
    return info


class MeshFactory:
    """Factory class for creating mesh objects from USD primitives.

    This class provides a static method to create the appropriate mesh object
    based on the type of USD primitive.
    """

    @staticmethod
    def get_mesh(
        prim,
        need_closed: bool = True,
        need_triangulate: bool = True,
        approx_config: dict[str, str] | None = None,
    ) -> Mesh:
        """Create a mesh object from a USD primitive.

        If ``approx_config`` is provided, mesh approximation is applied to
        GeomMesh types, replacing the mesh geometry with the requested
        approximation (e.g., convex hull, bounding box).

        Args:
            prim: USD primitive.
            need_closed: Whether the mesh must be watertight (ignored for GeomMesh
                if approx_config is provided).
            need_triangulate: Whether to triangulate non-triangle meshes.
            approx_config: Optional dict with mesh approximation configuration:
                - method: One of "convex_hull", "bounding_box", "bounding_sphere",
                    "coacd", "quadratic"
                - params: Optional method-specific parameters (e.g., coacd threshold)

        Returns:
            Mesh: An instance of a Mesh subclass appropriate for the primitive type.

        Raises:
            ValueError: If the primitive type is not supported.
        """

        prim_name = get_prim_type_name(prim)
        if prim_name == "Mesh":
            return GeomMesh(prim, need_closed, need_triangulate, approx_config)
        elif prim_name == "Cube":
            return Cube(prim)
        elif prim_name == "Sphere":
            return Sphere(prim)
        elif prim_name == "Cylinder":
            return Cylinder(prim)
        elif prim_name == "Cone":
            return Cone(prim)
        elif prim_name == "Capsule":
            return Capsule(prim)
        elif prim_name == "BasisCurves":
            return BasisCurves(prim)
        elif prim_name == "Points":
            return Points(prim)
        else:
            raise ValueError(f"Unknown primitive type: {prim_name} of {prim.GetPath()}")
