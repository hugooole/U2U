# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import numpy as np
from loguru import logger
from pxr import Usd, UsdGeom, UsdPhysics
from uipc import Transform, builtin, view
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import (
    SimplicialComplex,
    halfplane,
    label_surface,
    merge,
)
from uipc.geometry import (
    trimesh as trimesh_fn,
)

from ..mesh_factory import MeshFactory
from ..usd_utils import get_prim_name, get_prim_type_name
from ..utils import (
    angular_velocity_to_rotation_matrix_dot,
    create_simplicial_complex,
    extract_rot_and_scale_from_transform,
    get_mass_density,
    get_transform,
)
from .base import SceneBuilderBase


class RigidBodyBuilder(SceneBuilderBase):
    """Scene builder for rigid (affine) bodies."""

    def __init__(self, scene):
        """Initialize the rigid body simulator.

        Args:
            scene: The simulation scene
        """
        super().__init__(scene)
        self.env_elem = self.scene.env_elem
        self._rigid_body = []
        self.ground_name = None
        self.mesh_approximation_map: dict[str, dict] = {}

    def parse_usd(self, stage: Usd.Stage, exclude_paths: set[str] = set()) -> None:
        """Parse the USD stage to identify rigid body prims.

        Args:
            stage: The USD stage to parse
        """
        articulation_root_path: str = ""

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim_path in exclude_paths:
                continue
            if any(prim_path.startswith(path) for path in exclude_paths):
                continue

            # Skip invisible
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                visibility = imageable.ComputeVisibility(Usd.TimeCode.Default())
                if visibility == UsdGeom.Tokens.invisible:
                    continue

            # Skip articulations and its childrens
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_root_path = str(prim.GetPath())
                continue

            if articulation_root_path and str(prim.GetPath()).startswith(articulation_root_path):
                continue

            # Skip meshes with no points
            if prim.IsA(UsdGeom.Mesh):
                usdMesh = UsdGeom.Mesh(prim)
                attr = usdMesh.GetPointsAttr().Get()
                if attr is None or len(attr) == 0:
                    continue

            if get_prim_type_name(prim) == "Scope" and prim.HasAPI(UsdPhysics.CollisionAPI):
                self._rigid_body.extend(prim.GetChildren())

            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                self._rigid_body.append(prim)

    def build(self, num_instances: int = 1, env_offsets: list | None = None) -> None:
        """Build the rigid body simulation objects.

        Args:
            num_instances: Number of environment instances (>1 enables multi-instance mode).
            env_offsets: List of N 4x4 offset transforms when num_instances > 1.
        """
        if len(self._rigid_body) == 0:
            logger.info("No RigidBodyAPI or CollisionAPI prims found, skipping AffineBody construction.")
            return

        logger.debug(f"rigid bodies to build: {[str(prim.GetPath()) for prim in self._rigid_body]}")

        abd = AffineBodyConstitution()
        multi = num_instances > 1
        N = num_instances

        for prim in self._rigid_body:
            if prim.IsA(UsdGeom.Gprim):
                if prim.IsA(UsdGeom.Plane):
                    extent = prim.GetAttribute("extent").Get()
                    extent = np.array(extent, dtype=np.float64)
                    transform = get_transform(prim)

                    center = (extent[0] + extent[1]) / 2.0 + transform[:3, 3]
                    up_dir = prim.GetAttribute("axis").Get()
                    if not up_dir or up_dir == UsdGeom.Tokens.z:
                        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    elif up_dir == UsdGeom.Tokens.y:
                        normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    elif up_dir == UsdGeom.Tokens.x:
                        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    else:
                        logger.warning(f"Unknown up direction {up_dir} for plane {prim.GetPath()}")
                        raise ValueError(f"Unknown up direction {up_dir}")

                    g = halfplane(center, normal)
                    simp = self.scene.objects().create(get_prim_name(prim.GetPath()))
                    self.ground_name = get_prim_name(prim.GetPath())
                    simp.geometries().create(g)
                    logger.debug(f"Adding plane {prim.GetPath()} to scene.")
                    continue

                prim_path_str = str(prim.GetPath())
                approx_config = None
                for ancestor_path in self.mesh_approximation_map.keys():
                    if prim_path_str.startswith(ancestor_path):
                        approx_config = self.mesh_approximation_map[ancestor_path]
                        break
                mesh_prim = MeshFactory.get_mesh(prim, approx_config=approx_config)
                approx_info = getattr(mesh_prim, "approx_info", None)

                # Get Prim Transform
                gf_mat_np = get_transform(prim)

                # Extract scale from transform
                rotation, scale_mat = extract_rot_and_scale_from_transform(gf_mat_np)
                scale_trans = Transform(scale_mat)

                # Apply scale
                mesh_prim.points = scale_trans.apply_to(mesh_prim.points)

                # Convert unit
                mesh_prim.points *= self.metersPerUnit
                gf_mat_np[:3, 3] *= self.metersPerUnit
                gf_mat_np[:3, :3] = rotation

                # Create SimplicialComplex and apply AffineBodyConstitution
                mesh = create_simplicial_complex(mesh_prim, gf_mat_np)

                if multi:
                    mesh.instances().resize(N)
                    base_transform = mesh.transforms().view()[0].copy()
                    for inst_i in range(N):
                        view(mesh.transforms())[inst_i] = env_offsets[inst_i] @ base_transform

                mass_density = get_mass_density(prim, default_value=1000.0)

                abd.apply_to(
                    sc=mesh,
                    kappa=100 * 1e6,  # resistance coefficient (MPa)
                    mass_density=mass_density,
                )

                # Fix the static objects for collision detection
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    view(mesh.instances().find(builtin.is_fixed))[:] = 1
                    self.env_elem.apply_to(mesh)

                label_surface(mesh)

                # Setting linear and angular velocity for the moving objects
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_api = UsdPhysics.RigidBodyAPI(prim)
                    velocity = np.array(rigid_api.GetVelocityAttr().Get(), np.float64)
                    angular_velocity = rigid_api.GetAngularVelocityAttr().Get()
                    rotation_matrix = angular_velocity_to_rotation_matrix_dot(
                        np.array(angular_velocity).astype(np.float64),
                        mesh.transforms().view()[0][:3, :3],
                        degrees=True,
                    )
                    velocity = velocity * self.metersPerUnit
                    velocity_mat = np.zeros((4, 4), dtype=np.float64)
                    velocity_mat[:3, 3] = velocity
                    velocity_mat[:3, :3] = rotation_matrix
                    view(mesh.instances().find(builtin.velocity))[:] = velocity_mat

                # Create geometry in the scene
                simp = self.scene.objects().create(get_prim_name(prim.GetPath()))

                # Record geometry slots
                geo_slot, rest_slot = simp.geometries().create(mesh)
                self._geometry[str(prim.GetPath())] = {
                    "prim": prim,
                    "geo_slot": geo_slot,
                    "rest_slot": rest_slot,
                    "type": "rigid_body" if prim.HasAPI(UsdPhysics.RigidBodyAPI) else "collider",
                    "approx_info": approx_info,
                }
            elif get_prim_type_name(prim) == "Xform":
                logger.info(f"Processing Xform prim {prim.GetPath()}")

                def get_geometry(
                    prim_list: list[Usd.Prim] | Usd.Prim,
                    link_prim: Usd.Prim | None = None,
                ) -> SimplicialComplex:
                    if isinstance(prim_list, Usd.Prim):
                        prim_list = [prim_list]
                    assert len(prim_list) >= 1, "No prim found in the prim list"
                    link_meshes = []
                    for prim in prim_list:
                        prim_path_str = str(prim.GetPath())
                        approx_config = None
                        for ancestor_path in self.mesh_approximation_map.keys():
                            if prim_path_str.startswith(ancestor_path):
                                approx_config = self.mesh_approximation_map[ancestor_path]
                                break
                        mesh_prim = MeshFactory.get_mesh(prim, approx_config=approx_config)
                        gf_mat_np = get_transform(prim)
                        mesh_prim.points *= self.metersPerUnit
                        mesh_prim.points = Transform(gf_mat_np).apply_to(mesh_prim.points)
                        simp: SimplicialComplex = trimesh_fn(
                            mesh_prim.points.astype(np.float64),
                            mesh_prim.faces.astype(np.int32),
                        )
                        link_meshes.append(simp)
                    mesh = merge(link_meshes)
                    transform_l2w = get_transform(link_prim)
                    transform_w2l = Transform(np.linalg.inv(transform_l2w))
                    transform_w2l.apply_to(view(mesh.positions()))
                    view(mesh.transforms())[:] = transform_l2w
                    return mesh

                # Check for "collisions" child (same as articulation.py extract_link_geo)
                search_prim = prim
                if "collisions" in prim.GetChildrenNames():
                    search_prim = prim.GetChild("collisions")

                mesh_prim_list = []
                if search_prim.IsInstance():
                    stage = search_prim.GetStage()
                    proto = search_prim.GetPrototype()
                    for child in Usd.PrimRange(proto):
                        path = str(child.GetPath())
                        if "visuals" in path:
                            continue
                        if not child.IsA(UsdGeom.Gprim):
                            continue
                        inst_path = child.GetPath().ReplacePrefix(proto.GetPath(), search_prim.GetPath())
                        inst_child = stage.GetPrimAtPath(inst_path)
                        if inst_child and inst_child.IsValid():
                            mesh_prim_list.append(inst_child)
                else:
                    for child_prim in Usd.PrimRange(search_prim, Usd.TraverseInstanceProxies()):
                        path = str(child_prim.GetPath())
                        if "visuals" in path:
                            continue
                        if not child_prim.IsA(UsdGeom.Gprim):
                            continue
                        mesh_prim_list.append(child_prim)

                mesh = get_geometry(mesh_prim_list, prim)

                # if is_trimesh_closed(mesh):
                #     raise RuntimeError(f"mesh {prim.GetPath()} is not closed")
                print(f"mesh {prim.GetPath()} is closed")

                if multi:
                    mesh.instances().resize(N)
                    base_transform = mesh.transforms().view()[0].copy()
                    for inst_i in range(N):
                        view(mesh.transforms())[inst_i] = env_offsets[inst_i] @ base_transform

                mass_density = get_mass_density(prim, default_value=1000.0)
                abd.apply_to(
                    sc=mesh,
                    kappa=100 * 1e6,  # resistance coefficient (MPa)
                    mass_density=mass_density,
                )

                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    view(mesh.instances().find(builtin.is_fixed))[:] = 1
                    self.env_elem.apply_to(mesh)

                label_surface(mesh)

                simp = self.scene.objects().create(get_prim_name(prim.GetPath()))

                # Record geometry slots
                geo_slot, rest_slot = simp.geometries().create(mesh)
                self._geometry[str(prim.GetPath())] = {
                    "prim": prim,
                    "geo_slot": geo_slot,
                    "rest_slot": rest_slot,
                    "type": "rigid_body" if prim.HasAPI(UsdPhysics.RigidBodyAPI) else "collider",
                }
