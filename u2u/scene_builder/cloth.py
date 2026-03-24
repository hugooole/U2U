# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from loguru import logger
from pxr import Usd, UsdGeom
from uipc import Transform, view
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, NeoHookeanShell
from uipc.geometry import label_surface
from uipc.unit import kPa

from ..mesh_factory import MeshFactory
from ..utils import (
    create_simplicial_complex,
    get_mass_density,
    get_transform,
    transform_and_scale_points,
)
from .base import SceneBuilderBase


class ClothBuilder(SceneBuilderBase):
    """Scene builder for cloth bodies."""

    def __init__(self, scene):
        """Initialize the cloth simulator.

        Args:
            scene: The simulation scene
        """
        super().__init__(scene)
        self._cloth_prims = set()

    def parse_usd(self, stage: Usd.Stage, exclude_paths: set[str] = set()) -> None:
        """Parse the USD stage to identify cloth prims.

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

            # if prim.HasAPI(ClothPhysicsAPI):
            #     self._cloth_prims.append(prim)

            if "ClothPhysicsAPI" in prim.GetAppliedSchemas():
                self._cloth_prims.add(prim)

    def build(self, num_instances: int = 1, env_offsets: list | None = None) -> None:
        """Build the cloth simulation objects."""
        if len(self._cloth_prims) == 0:
            logger.info("No ClothAPI prims found, skipping cloth construction.")
            return

        nhs = NeoHookeanShell()
        dsb = DiscreteShellBending()
        for prim in self._cloth_prims:
            # Get cloth mesh
            mesh_prim = MeshFactory.get_mesh(prim, need_closed=False)

            # Preprocessing mesh before applying cloth
            t = Transform(get_transform(prim))
            mesh_prim.points = transform_and_scale_points(mesh_prim.points, t, self.metersPerUnit)

            # Create trimesh and apply cloth constitutions
            mesh = create_simplicial_complex(mesh_prim)

            if num_instances > 1:
                mesh.instances().resize(num_instances)
                base_transform = mesh.transforms().view()[0].copy()
                for inst_i in range(num_instances):
                    view(mesh.transforms())[inst_i] = env_offsets[inst_i] @ base_transform

            # Set mass or density
            mass_density = get_mass_density(prim, default_value=100.0)

            # Apply contact model
            label_surface(mesh)

            # Apply NeoHookeanShell and DiscreteShellBending
            # cloth_api = ClothPhysicsAPI(prim)
            # youngsModulus = cloth_api.GetYoungsModulusAttr().Get()
            # poissonRatio = cloth_api.GetPossionRatioAttr().Get()
            # thickness = cloth_api.GetThicknessAttr().Get()
            # blendingStiffness = cloth_api.GetBlendingStiffnessAttr().Get()
            youngsModulus = prim.GetAttribute("physics:youngsModulus").Get()
            poissonRatio = prim.GetAttribute("physics:possionRatio").Get()
            thickness = prim.GetAttribute("physics:thickness").Get()
            blendingStiffness = prim.GetAttribute("physics:blendingStiffness").Get()
            moduli = ElasticModuli2D.youngs_poisson(youngsModulus * kPa, poissonRatio)
            nhs.apply_to(mesh, moduli, mass_density=mass_density, thickness=thickness)
            dsb.apply_to(mesh, blendingStiffness)

            # Create Object
            simp = self.scene.objects().create(f"{prim.GetPath()}")
            geo_slot, rest_slot = simp.geometries().create(mesh)
            self._geometry[str(prim.GetPath())] = {
                "prim": prim,
                "geo_slot": geo_slot,
                "rest_slot": rest_slot,
                "transform": t,
                "type": "cloth",
            }
