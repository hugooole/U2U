# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import re
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from loguru import logger
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from uipc import view

from u2u.env import Env
from u2u.env_manager import EnvManager
from u2u.scene import Scene
from u2u.scene_builder import (
    Articulation,
    ArticulationBuilder,
    ClothBuilder,
    DeformableBuilder,
    FixedJoint,
    PrismaticJoint,
    RevoluteJoint,
    RigidBodyBuilder,
)
from u2u.usd_utils import get_float, has_attribute, read_usd
from u2u.utils import get_transform, gf_quat_to_rotation


class UsdParserConfig(TypedDict, total=False):
    """Configuration for USD parsing behavior.

    Attributes:
        root_path: Root path to start parsing from (default: "/")
        ignore_paths: List of prim paths to exclude from parsing
        skip_mesh_approximation: Skip automatic mesh approximation (default: False)
        approx_method: Default approximation method to use if no physics:approximation attribute is found (default: "convexdecomposition")
        multi_env: Enable multi-environment instance mode (default: False)
        env_scope_path: Scope path containing env_N prims (default: "/World/envs")
    """

    root_path: str
    ignore_paths: list[str]
    skip_mesh_approximation: bool
    approx_method: str
    multi_env: bool
    env_scope_path: str


class UsdParser:
    def __init__(self, scene: Scene, stage: Usd.Stage | str):
        self.scene = scene
        if isinstance(stage, str):
            stage = read_usd(stage)
        self.stage = stage
        self.scene.meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.scene_config = self.scene.config()
        self._setup_gravity()

        # init scene_builder
        self.articulation_builder = ArticulationBuilder(self.scene)
        self.rigidbody_builder = RigidBodyBuilder(self.scene)
        self.cloth_builder = ClothBuilder(self.scene)
        self.deformable_builder = DeformableBuilder(self.scene)

    def _parse_units(self):
        """Parse the units from the stage."""
        self.mass_unit = 1.0  # kg/m^3
        self.linear_unit = 1.0  # m
        try:
            if UsdPhysics.StageHasAuthoredKilogramsPerUnit(self.stage):
                self.mass_unit = UsdPhysics.GetStageKilogramsPerUnit(self.stage)
        except Exception as e:
            logger.warning(f"Failed to get mass unit: {e}")
        try:
            if UsdGeom.StageHasAuthoredMetersPerUnit(self.stage):
                self.linear_unit = UsdGeom.GetStageMetersPerUnit(self.stage)
        except Exception as e:
            logger.warning(f"Failed to get linear unit: {e}")

        self.articulation_builder.metersPerUnit = self.linear_unit
        self.rigidbody_builder.metersPerUnit = self.linear_unit

    def _setup_gravity(self):
        """Set up gravity based on the stage's up axis."""
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        gravity_view = view(self.scene_config.find("gravity"))
        self.scene.up_axis = up_axis
        if not up_axis:
            gravity_view[:] = [[0.0], [0.0], [-9.8]]
            logger.info("No up axis found, assuming z up.")
        elif up_axis == UsdGeom.Tokens.z:
            gravity_view[:] = [[0.0], [0.0], [-9.8]]
        elif up_axis == UsdGeom.Tokens.y:
            gravity_view[:] = [[0.0], [-9.8], [0.0]]
        elif up_axis == UsdGeom.Tokens.x:
            gravity_view[:] = [[-9.8], [0.0], [0.0]]
        else:
            raise ValueError(f"Unsupported up axis {up_axis}.")

    def parse_usd(
        self,
        root_path: str = "/",
        ignore_paths: list[str] | None = None,
        skip_mesh_approximation: bool = False,
        approx_method: Literal[
            "convexdecomposition", "convexhull", "boundingsphere", "boundingcube", "meshsimplification"
        ]
        | None = None,
    ) -> None:
        self._parse_units()

        if ignore_paths is None:
            ignore_paths = []

        non_regex_ignore_paths = [path for path in ignore_paths if ".*" not in path]

        ret_dict = UsdPhysics.LoadUsdPhysicsFromRange(self.stage, [root_path], excludePaths=non_regex_ignore_paths)

        # Mapping from USD physics:approximation values to remeshing methods.
        # Ported from Newton's parse_usd (newton/_src/utils/import_usd.py).

        approximation_to_remeshing_method = {
            "convexdecomposition": "coacd",
            "convexhull": "convex_hull",
            "boundingsphere": "bounding_sphere",
            "boundingcube": "bounding_box",
            "meshsimplification": "quadratic",
        }

        # prim_path -> {"method": str, "coacd_threshold": float | None}
        self.mesh_approximation_map: dict[str, dict] = {}

        if not skip_mesh_approximation:
            for prim in self.stage.Traverse():
                prim_path = str(prim.GetPath())
                if any(re.match(p, prim_path) for p in ignore_paths):
                    continue

                if not prim.HasAPI(UsdPhysics.MeshCollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue

                if approx_method is not None:
                    approximation = approx_method
                else:
                    approx_attr = prim.GetAttribute("physics:approximation")
                    if not approx_attr or not approx_attr.HasAuthoredValue():
                        # use coacd by default
                        approximation = "convexdecomposition"
                    else:
                        approximation = approx_attr.Get()

                remeshing_method = approximation_to_remeshing_method.get(approximation.lower())
                if remeshing_method is None:
                    logger.warning(
                        f"Unknown physics:approximation '{approximation}' on shape at '{prim_path}', skipping."
                    )
                    continue

                # Read optional per-prim coacd threshold
                coacd_threshold = None
                threshold_attr = prim.GetAttribute("rbs:coacd_threshold")
                if threshold_attr and threshold_attr.HasAuthoredValue():
                    coacd_threshold = float(threshold_attr.Get())

                self.mesh_approximation_map[prim_path] = {
                    "method": remeshing_method,
                    "params": coacd_threshold,
                }

                if prim_path in self.mesh_approximation_map:
                    approx = self.mesh_approximation_map[prim_path]

                    logger.debug(
                        f"Mesh approximation '{approx['method']}' requested for {prim_path}"
                        f" (coacd_threshold={approx.get('params', 'None')})"
                    )

        self.rigidbody_builder.mesh_approximation_map = self.mesh_approximation_map
        self.articulation_builder.mesh_approximation_map = self.mesh_approximation_map

        ignored_body_paths = set()
        body_specs = {}
        body_density = {}
        material_specs = {}
        joint_descriptions = {}
        default_shape_density = 1000.0  # kg/m^3
        art_body_keys = set()

        # Parsing physics materials from the stage
        for sdf_path, desc in data_for_key(ret_dict, UsdPhysics.ObjectType.RigidBodyMaterial):
            if warn_invalid_desc(sdf_path, desc):
                continue
            material_specs[str(sdf_path)] = PhysicsMaterial(
                staticFriction=desc.staticFriction,
                dynamicFriction=desc.dynamicFriction,
                restitution=desc.restitution,
                # TODO: if desc.density is 0, then we should look for mass somewhere
                density=desc.density if desc.density > 0.0 else default_shape_density,
            )

        # Parsing rigid body
        if UsdPhysics.ObjectType.RigidBody in ret_dict:
            prim_paths, rigid_body_descs = ret_dict[UsdPhysics.ObjectType.RigidBody]
            for prim_path, rigid_body_desc in zip(prim_paths, rigid_body_descs, strict=False):
                if warn_invalid_desc(prim_path, rigid_body_desc):
                    continue
                body_path = str(prim_path)
                if any(re.match(p, body_path) for p in ignore_paths):
                    ignored_body_paths.add(body_path)
                    continue
                body_specs[body_path] = rigid_body_desc
                body_density[body_path] = default_shape_density
                prim = self.stage.GetPrimAtPath(prim_path)
                # Marking for deprecation --->
                if prim.HasRelationship("material:binding:physics"):
                    other_paths = prim.GetRelationship("material:binding:physics").GetTargets()
                    if len(other_paths) > 0:
                        material = material_specs[str(other_paths[0])]
                        if material.density > 0.0:
                            body_density[body_path] = material.density

                if prim.HasAPI(UsdPhysics.MassAPI):
                    if has_attribute(prim, "physics:density"):
                        d = get_float(prim, "physics:density")
                        density = d * self.mass_unit  # / (linear_unit**3)
                        body_density[body_path] = density
                # <--- Marking for deprecation

        # Parsing articulation
        if UsdPhysics.ObjectType.Articulation in ret_dict:
            for key, value in ret_dict.items():
                if key in {
                    UsdPhysics.ObjectType.FixedJoint,
                    UsdPhysics.ObjectType.RevoluteJoint,
                    UsdPhysics.ObjectType.PrismaticJoint,
                    UsdPhysics.ObjectType.SphericalJoint,
                    UsdPhysics.ObjectType.D6Joint,
                    UsdPhysics.ObjectType.DistanceJoint,
                }:
                    paths, joint_specs = value
                    for path, joint_spec in zip(paths, joint_specs, strict=False):
                        joint_descriptions[str(path)] = joint_spec

            paths, articulation_descs = ret_dict[UsdPhysics.ObjectType.Articulation]

            for path, desc in zip(paths, articulation_descs, strict=False):
                if warn_invalid_desc(path, desc):
                    continue
                articulation_path = str(path)
                if any(re.match(p, articulation_path) for p in ignore_paths):
                    continue
                robot_prim = self.stage.GetPrimAtPath(path)
                robot_name = articulation_path
                logger.debug(f"Parsing articulation: {robot_name}")
                self.articulation_builder._robot_transforms[robot_name] = get_transform(robot_prim)
                if robot_name not in self.articulation_builder._articulation_bodies:
                    self.articulation_builder._articulation_bodies[robot_name] = []
                    # Try to get root_is_fixed from USD prim custom attribute
                    rbs_root_is_fixed_attr = robot_prim.GetAttribute("rbs:root_is_fixed")
                    root_is_fixed = True  # default value
                    if rbs_root_is_fixed_attr and rbs_root_is_fixed_attr.IsValid():
                        root_is_fixed_value = rbs_root_is_fixed_attr.Get()
                        if root_is_fixed_value is not None:
                            root_is_fixed = bool(root_is_fixed_value)
                            logger.debug(f"Found `rbs:root_is_fixed` attribute on {robot_name}: {root_is_fixed}")

                    robot = Articulation(
                        name=robot_name,
                        robot_prim=self.stage.GetPrimAtPath(robot_name),
                        is_root_fixed=root_is_fixed,
                    )
                    self.articulation_builder.robot_dict[robot_name] = robot

                for body in desc.articulatedBodies:
                    key = str(body)
                    if key in ignored_body_paths:
                        continue

                    if body == Sdf.Path.emptyPath:
                        continue

                    if key in body_specs:
                        # Delete body spec from body_specs which is already parsed as articulated body
                        del body_specs[key]

                    art_body_keys.add(key)

                for joint in desc.articulatedJoints:
                    joint_key = str(joint)
                    robot.joint_names.append(joint_key)
                    joint_desc = joint_descriptions[joint_key]
                    key = joint_desc.type
                    joint_path = str(joint_desc.primPath)
                    joint_prim = self.stage.GetPrimAtPath(joint_path)

                    if key == UsdPhysics.ObjectType.RevoluteJoint:
                        joint_api = UsdPhysics.RevoluteJoint(joint_prim)
                        joint = RevoluteJoint(
                            prim=joint_prim,
                            body0=self.stage.GetPrimAtPath(joint_api.GetBody0Rel().GetTargets()[0]),
                            body1=self.stage.GetPrimAtPath(joint_api.GetBody1Rel().GetTargets()[0]),
                            axis=str(joint_api.GetAxisAttr().Get()),
                            lower_limit=float(joint_api.GetLowerLimitAttr().Get()),
                            upper_limit=float(joint_api.GetUpperLimitAttr().Get()),
                            local_pos0=np.array(joint_api.GetLocalPos0Attr().Get()) * self.linear_unit,
                            local_orient0=gf_quat_to_rotation(joint_api.GetLocalRot0Attr().Get()),
                            local_pos1=np.array(joint_api.GetLocalPos1Attr().Get()) * self.linear_unit,
                            local_orient1=gf_quat_to_rotation(joint_api.GetLocalRot1Attr().Get()),
                        )
                    elif key == UsdPhysics.ObjectType.PrismaticJoint:
                        joint_api = UsdPhysics.PrismaticJoint(joint_prim)
                        joint = PrismaticJoint(
                            prim=joint_prim,
                            body0=self.stage.GetPrimAtPath(joint_api.GetBody0Rel().GetTargets()[0]),
                            body1=self.stage.GetPrimAtPath(joint_api.GetBody1Rel().GetTargets()[0]),
                            axis=str(joint_api.GetAxisAttr().Get()),
                            lower_limit=float(joint_api.GetLowerLimitAttr().Get()),
                            upper_limit=float(joint_api.GetUpperLimitAttr().Get()),
                            local_pos0=np.array(joint_api.GetLocalPos0Attr().Get()) * self.linear_unit,
                            local_orient0=gf_quat_to_rotation(joint_api.GetLocalRot0Attr().Get()),
                            local_pos1=np.array(joint_api.GetLocalPos1Attr().Get()) * self.linear_unit,
                            local_orient1=gf_quat_to_rotation(joint_api.GetLocalRot1Attr().Get()),
                        )
                    elif key == UsdPhysics.ObjectType.FixedJoint:
                        joint_api = UsdPhysics.FixedJoint(joint_prim)
                        body0_targets = joint_api.GetBody0Rel().GetTargets()
                        body1_targets = joint_api.GetBody1Rel().GetTargets()

                        joint = FixedJoint(
                            prim=joint_prim,
                            body0=self.stage.GetPrimAtPath(body0_targets[0]) if body0_targets else None,
                            body1=self.stage.GetPrimAtPath(body1_targets[0]) if body1_targets else None,
                            local_pos0=np.array(joint_api.GetLocalPos0Attr().Get()) * self.linear_unit,
                            local_orient0=gf_quat_to_rotation(joint_api.GetLocalRot0Attr().Get()),
                            local_pos1=np.array(joint_api.GetLocalPos1Attr().Get()) * self.linear_unit,
                            local_orient1=gf_quat_to_rotation(joint_api.GetLocalRot1Attr().Get()),
                        )
                    else:
                        raise ValueError(f"Unsupported joint type: {key}, joint path: {joint_path}")

                    self.articulation_builder._articulation_bodies[robot_name].append(joint)

                # Find the root link from joint topology: a body that appears as body0
                # but never as body1 in any joint. Store these root links separately
                # so that build() processes their geometry before the joint loop.
                joint_list = self.articulation_builder._articulation_bodies[robot_name]
                body0_paths = {str(j.body0.GetPath()) for j in joint_list if j.body0 is not None}
                body1_paths = {str(j.body1.GetPath()) for j in joint_list if j.body1 is not None}
                root_link_paths = body0_paths - body1_paths
                if root_link_paths:
                    root_link_prims = [self.stage.GetPrimAtPath(p) for p in root_link_paths]
                    self.articulation_builder._root_links[robot_name] = root_link_prims
                    # If no explicit rbs:root_is_fixed attribute, default to floating base
                    # since the USD doesn't define a root joint (typical for mobile robots).
                    attr = robot_prim.GetAttribute("rbs:root_is_fixed")
                    if not (attr and attr.IsValid() and attr.Get() is not None):
                        robot.is_root_fixed = False
                        logger.debug(
                            f"No root joint found for {robot_name}, defaulting to floating base (is_root_fixed=False)"
                        )
                    for p in root_link_paths:
                        logger.debug(f"Detected root link {p} in {robot_name}")

        # insert remaining bodies that were not part of any articulation so far
        self.rigidbody_builder.parse_usd(self.stage, exclude_paths=set(art_body_keys) | set(non_regex_ignore_paths))

        # Parsing deformable body
        # Temperal schema------------------>
        self.deformable_builder.parse_usd(self.stage, exclude_paths=set(art_body_keys) | set(non_regex_ignore_paths))
        # <--------------------------

        # Parsing cloth body
        # Temperal schema------------------>
        self.cloth_builder.parse_usd(self.stage, exclude_paths=set(art_body_keys) | set(non_regex_ignore_paths))
        # <--------------------------

        # Summary
        n_robots = len(self.articulation_builder.robot_dict)
        n_approx = len(self.mesh_approximation_map)
        if n_robots > 0 or n_approx > 0:
            logger.info(
                f"USD parsed: {n_robots} articulation(s), "
                f"{len(art_body_keys)} articulated bodies, "
                f"{n_approx} mesh approximation(s)"
            )

    def parse_and_build_scene(
        self,
        root_path: str = "/",
        ignore_paths: list[str] | None = None,
        skip_mesh_approximation: bool = False,
        approx_method: str = "convexdecomposition",
        multi_env: bool = False,
        env_scope_path: str = "/World/envs",
    ) -> Scene:
        if multi_env:
            return self._parse_and_build_multi_env(
                root_path=root_path,
                ignore_paths=ignore_paths,
                skip_mesh_approximation=skip_mesh_approximation,
                approx_method=approx_method,
                env_scope_path=env_scope_path,
            )

        self.parse_usd(
            root_path=root_path,
            ignore_paths=ignore_paths,
            skip_mesh_approximation=skip_mesh_approximation,
            approx_method=approx_method,
        )
        self._build_and_collect()
        return self.scene

    def _build_and_collect(
        self,
        num_instances: int = 1,
        env_offsets: list[np.ndarray] | None = None,
    ) -> None:
        """Build all scene objects and collect geometry into scene dicts."""
        self.articulation_builder.build(self.stage, num_instances, env_offsets)
        self.rigidbody_builder.build(num_instances, env_offsets)
        self.deformable_builder.build(num_instances, env_offsets)
        self.cloth_builder.build(num_instances, env_offsets)
        self.scene.geometry_dict.update(self.rigidbody_builder.get_geometry())
        self.scene.geometry_dict.update(self.articulation_builder.get_geometry())
        self.scene.geometry_dict.update(self.deformable_builder.get_geometry())
        self.scene.geometry_dict.update(self.cloth_builder.get_geometry())
        self.scene.ground_name = self.rigidbody_builder.ground_name
        self.scene.robot_dict = dict(self.articulation_builder.robot_dict)
        for robot in self.scene.robot_dict.values():
            self.scene.geometry_dict.update(robot.link_geometry)

    def _parse_and_build_multi_env(
        self,
        root_path: str,
        ignore_paths: list[str] | None,
        skip_mesh_approximation: bool,
        approx_method: str,
        env_scope_path: str,
    ) -> Scene:
        """Parse and build scene using native multi-instance mode.

        Only env_0 (template) is parsed; clone envs are excluded.
        Builders receive num_instances and env_offsets to create
        shared meshes with per-instance transforms.
        """
        env_mgr = EnvManager(self.stage, env_scope_path)
        envs = env_mgr.detect_envs()
        num_instances = env_mgr.num_envs
        env_offsets = env_mgr.compute_offsets()

        # Build ignore list: skip all clone envs (env_1..env_N-1)
        clone_ignore = [e.env_path + "/.*" for e in env_mgr.clone_envs]
        # Also exact-match the clone env Xform prims themselves
        clone_ignore += [e.env_path for e in env_mgr.clone_envs]
        combined_ignore = clone_ignore + (ignore_paths or [])

        logger.info(f"Multi-env mode: {num_instances} instances, parsing only {env_mgr.template_env.env_path}")

        # 1. Parse only the template env
        self.parse_usd(
            root_path=root_path,
            ignore_paths=combined_ignore,
            skip_mesh_approximation=skip_mesh_approximation,
            approx_method=approx_method,
        )

        # 2. Build with multi-instance
        self._build_and_collect(num_instances, env_offsets)

        # 3. Build per-env data and per-instance geometry_dict entries
        self._build_multi_env(envs, env_offsets, env_mgr)

        return self.scene

    def _build_multi_env(
        self,
        envs: list,
        env_offsets: list,
        env_mgr: EnvManager,
    ) -> None:
        """Create per-env Env objects sharing a single Articulation instance.

        In the new design, all environments share the same Articulation object
        which manages state arrays with shape (N, J) for all N instances.
        Each Env tracks its instance_id to index into these arrays.
        """
        template_env = envs[0]
        num_instances = len(envs)

        self.scene.num_envs = num_instances

        for env_info in envs:
            i = env_info.env_id
            env_robots: dict[str, Articulation] = {}

            for robot_name, template_robot in list(self.articulation_builder.robot_dict.items()):
                # All instances share the same Articulation object
                env_robots[robot_name] = template_robot
                self.scene.robot_dict[robot_name] = template_robot

            # Build per-env geometry_dict entries for multi-instance transform writeback
            env_geo: dict[str, dict] = {}
            for prim_path, geo_info in list(self.scene.geometry_dict.items()):
                if i == 0:
                    # Add instance_id to template entries
                    geo_info.setdefault("instance_id", 0)
                else:
                    target_path = EnvManager.remap_path(prim_path, template_env.env_path, env_info.env_path)
                    target_prim = self.stage.GetPrimAtPath(target_path)
                    if target_prim and target_prim.IsValid():
                        clone_entry = {
                            "prim": target_prim,
                            "geo_slot": geo_info["geo_slot"],
                            "type": geo_info["type"],
                            "instance_id": i,
                        }
                        if "robot_name" in geo_info:
                            clone_entry["robot_name"] = geo_info[
                                "robot_name"
                            ]  # all instances share the same Articulation
                        self.scene.geometry_dict[target_path] = clone_entry
                        env_geo[target_path] = clone_entry

            env = Env(
                env_id=i,
                env_path=env_info.env_path,
                instance_id=i,
                offset_transform=env_offsets[i],
                robot_dict=env_robots,
                geometry_dict=env_geo if i > 0 else {},
            )
            self.scene.env_dict[i] = env


@dataclass
class PhysicsMaterial:
    staticFriction: float = 0.5
    dynamicFriction: float = 0.5
    restitution: float = 0.0
    density: float = 1000.0


def warn_invalid_desc(path, descriptor) -> bool:
    if not descriptor.isValid:
        logger.warning(f'Invalid {type(descriptor).__name__} descriptor for prim at path "{path}".')
        return True
    return False


def data_for_key(physics_utils_results, key):
    if key not in physics_utils_results:
        return

    yield from zip(*physics_utils_results[key], strict=False)
