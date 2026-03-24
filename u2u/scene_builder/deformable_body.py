# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from loguru import logger
from pxr import Usd, UsdGeom, UsdPhysics
from uipc import Transform, view
from uipc.constitution import ElasticModuli, StableNeoHookean
from uipc.geometry import (
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
)
from uipc.unit import kPa

from ..mesh_factory import MeshFactory
from ..utils import (
    create_simplicial_complex,
    get_mass_density,
    get_transform,
    transform_and_scale_points,
)
from .base import SceneBuilderBase


class DeformableBuilder(SceneBuilderBase):
    """Scene builder for deformable bodies."""

    def __init__(self, scene):
        """Initialize the deformable body simulator.

        Args:
            scene: The simulation scene
        """
        super().__init__(scene)
        self._deformable_bodies = set()

    def parse_usd(self, stage: Usd.Stage, exclude_paths: set[str] = set()) -> None:
        """Parse the USD stage to identify deformable body prims.

        Args:
            stage: The USD stage to parse
        """
        for prim in stage.Traverse():
            if str(prim.GetPath()) in exclude_paths:
                continue
            # Skip invisible
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                visibility = imageable.ComputeVisibility(Usd.TimeCode.Default())
                if visibility == UsdGeom.Tokens.invisible:
                    continue

            # Skip meshes with no points
            if prim.IsA(UsdGeom.Mesh):
                usdMesh = UsdGeom.Mesh(prim)
                attr = usdMesh.GetPointsAttr().Get()
                if attr is None or len(attr) == 0:
                    continue

            if "DeformableBodyAPI" in prim.GetAppliedSchemas():
                # check if it is a rigid body
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    logger.warning(
                        "PhysicsSchema.DeformableBodyAPI cannot be applied to a primitive with UsdPhysics.RigidBodyAPI"
                    )
                    raise ValueError
                # check if it is a UsdGeom.Mesh
                if not prim.IsA(UsdGeom.Mesh):
                    logger.error(
                        f"PhysicsSchema.DeformableBodyAPI must be applied to a UsdGeom.Mesh: {prim.GetPath()} not a UsdGeom.Mesh"
                    )
                    raise ValueError
                self._deformable_bodies.add(prim)

    def build(self, num_instances: int = 1, env_offsets: list | None = None) -> None:
        """Build the deformable body simulation objects."""
        if len(self._deformable_bodies) == 0:
            logger.info("No DeformableBodyAPI prims found, skipping DeformableBody construction.")
            return

        logger.info(f"deformable bodies to build: {[str(prim.GetPath()) for prim in self._deformable_bodies]}")

        snk = StableNeoHookean()
        for prim in self._deformable_bodies:
            mesh_prim = MeshFactory.get_mesh(prim, need_closed=False)
            t = Transform(get_transform(prim))
            mesh_prim.points = transform_and_scale_points(mesh_prim.points, t, meters_per_unit=self.metersPerUnit)

            soft_mesh = create_simplicial_complex(mesh_prim, tethedralize=True)

            if num_instances > 1:
                soft_mesh.instances().resize(num_instances)
                base_transform = soft_mesh.transforms().view()[0].copy()
                for inst_i in range(num_instances):
                    view(soft_mesh.transforms())[inst_i] = env_offsets[inst_i] @ base_transform

            label_surface(soft_mesh)
            label_triangle_orient(soft_mesh)
            soft_mesh = flip_inward_triangles(soft_mesh)

            # deform_api = DeformableBodyAPI(prim)
            # youngsModulus = deform_api.GetYoungsModulusAttr().Get()
            # poissonRatio = deform_api.GetPossionRatioAttr().Get()
            youngsModulus = prim.GetAttribute("deform:youngsModulus").Get()
            poissonRatio = prim.GetAttribute("deform:possionRatio").Get()
            moduli = ElasticModuli.youngs_poisson(youngsModulus * kPa, poissonRatio)
            mass_density = get_mass_density(prim, 3e3)
            snk.apply_to(soft_mesh, moduli, mass_density=mass_density)

            simp = self.scene.objects().create(str(prim.GetPath()))
            geo_slot, rest_slot = simp.geometries().create(soft_mesh)
            self._geometry[str(prim.GetPath())] = {
                "prim": prim,
                "geo_slot": geo_slot,
                "rest_slot": rest_slot,
                "transform": t,
                "type": "deformable_body",
            }
