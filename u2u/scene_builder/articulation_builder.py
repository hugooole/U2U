# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
import uipc.builtin as uipc_builtin
from loguru import logger
from pxr import Usd, UsdGeom
from uipc import Transform, view
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyDrivingPrismaticJoint,
    AffineBodyDrivingRevoluteJoint,
    AffineBodyFixedJoint,
    AffineBodyPrismaticJoint,
    AffineBodyPrismaticJointExternalForce,
    AffineBodyRevoluteJoint,
    AffineBodyRevoluteJointExternalForce,
    SoftTransformConstraint,
)
from uipc.geometry import SimplicialComplex, is_trimesh_closed, label_surface, linemesh, merge
from uipc.geometry import trimesh as trimesh_fn

from u2u.pose import Pose
from u2u.usd_utils import get_prim_name, get_prim_type_name

from ..mesh_factory import MeshFactory
from ..utils import (
    extract_rot_and_scale_from_transform,
    get_position_and_orientation,
    get_transform,
)
from .articulation import Articulation
from .base import SceneBuilderBase
from .joint_types import (
    FixedJoint,
    PhysicsJoint,
    PrismaticJoint,
    RevoluteJoint,
    is_active_joint,
)


class ArticulationBuilder(SceneBuilderBase):
    """Scene builder for articulation bodies."""

    def __init__(self, scene):
        """Initialize the articulation simulator.

        Args:
            scene: The simulation scene
        """
        super().__init__(scene)
        self.robot_dict: Dict[str, Articulation] = {}  # map robot name to Articulation class
        self._articulation_bodies: Dict[str, List[PhysicsJoint]] = {}
        self._robot_transforms: Dict[str, np.ndarray] = {}
        self._root_links: Dict[str, List[Usd.Prim]] = {}
        self.mesh_approximation_map: dict[str, dict] = {}

    def get_robot(self, name: str) -> Optional[Articulation]:
        """Get the Articulation object for the given robot name."""

        return self.robot_dict.get(name, None)

    def build(self, stage: Usd.Stage, num_instances: int = 1, env_offsets: list[np.ndarray] | None = None) -> None:
        """Build the articulation simulation objects.

        Args:
            stage: The USD stage.
            num_instances: Number of environment instances (>1 enables multi-instance mode).
            env_offsets: List of N 4x4 offset transforms (one per instance, relative to env_0).
                         Required when num_instances > 1.
        """
        if len(self._articulation_bodies) == 0:
            logger.info("No articulation bodies found, skipping articulation construction.")
            return

        multi = num_instances > 1
        if multi and (env_offsets is None or len(env_offsets) != num_instances):
            raise ValueError("env_offsets must be provided and match num_instances")

        N = num_instances
        if multi:
            env_offsets_arr = np.stack(env_offsets)  # (N, 4, 4)
        else:
            env_offsets_arr = np.eye(4, dtype=np.float64)[np.newaxis]  # (1, 4, 4)

        def get_geometry(
            prim_list: list[Usd.Prim] | Usd.Prim, link_prim: Usd.Prim
        ) -> tuple[SimplicialComplex, np.ndarray]:
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
                mesh_prim.points = Transform(gf_mat_np).apply_to(mesh_prim.points)  # model to world
                mesh_prim.points *= self.metersPerUnit  # convert to meters
                simp: SimplicialComplex = trimesh_fn(
                    mesh_prim.points.astype(np.float64),
                    mesh_prim.faces.astype(np.int32),
                )
                link_meshes.append(simp)
            mesh = merge(link_meshes)
            transform_l2w = get_transform(link_prim)
            rotation, _ = extract_rot_and_scale_from_transform(transform_l2w)
            transform_l2w[:3, :3] = rotation
            transform_l2w[:3, 3] *= self.metersPerUnit  # convert to meters
            transform_w2l = Transform(np.linalg.inv(transform_l2w))
            transform_w2l.apply_to(view(mesh.positions()))
            if multi:
                mesh.instances().resize(N)
                logger.debug(f"mesh instances resized to {N}")
                base_transform = transform_l2w.copy()
                view(mesh.transforms())[:] = env_offsets_arr @ base_transform
            else:
                view(mesh.transforms())[:] = transform_l2w
            return mesh, transform_l2w

        abd = AffineBodyConstitution()
        i = 0
        for robot_name, joint_list in self._articulation_bodies.items():
            i += 1
            robot = self.robot_dict[robot_name]
            setattr(self, f"robot_{i}", robot)
            self.robot_dict[robot_name] = robot

            # Sort joints in topological order
            try:
                sorted_indices = topological_sort(joint_list)
                # Process joints in topological order
                sorted_joint_list = [joint_list[idx] for idx in sorted_indices]
                # Store joint names in topological order
                robot.joint_names = [get_prim_name(joint.prim.GetPath()) for joint in sorted_joint_list]
                robot.active_joints = [
                    get_prim_name(joint.prim.GetPath()) for joint in sorted_joint_list if is_active_joint(joint)
                ]
                robot.joint_path_map = {
                    get_prim_name(joint.prim.GetPath()): str(joint.prim.GetPath())
                    for joint in sorted_joint_list
                    if is_active_joint(joint)
                }
                logger.debug(f"Joints sorted in topological order: {robot.joint_names}")
            except ValueError as e:
                logger.error(f"Topological sort failed: {e}. Processing joints in original order.")
                sorted_joint_list = joint_list

            def extract_link_geo(link_prim: Usd.Prim) -> tuple[SimplicialComplex, np.ndarray]:
                """Extract geometry from a link prim (Xform or Gprim).

                Returns:
                    Tuple of (mesh, base_transform_l2w).
                """
                prim = link_prim
                if get_prim_type_name(prim) == "Xform":
                    if "collisions" in prim.GetChildrenNames():
                        prim = prim.GetChild("collisions")

                    mesh_prim_list = []
                    if prim.IsInstance():
                        proto = prim.GetPrototype()
                        for child in Usd.PrimRange(proto):
                            path = str(child.GetPath())
                            if "visuals" in path:
                                continue
                            if not child.IsA(UsdGeom.Gprim):
                                continue
                            inst_path = child.GetPath().ReplacePrefix(proto.GetPath(), prim.GetPath())
                            inst_child = stage.GetPrimAtPath(inst_path)
                            if inst_child and inst_child.IsValid():
                                mesh_prim_list.append(inst_child)
                    else:
                        for child_prim in Usd.PrimRange(prim):
                            path = str(child_prim.GetPath())
                            if "visuals" in path:
                                continue
                            if not child_prim.IsA(UsdGeom.Gprim):
                                continue
                            mesh_prim_list.append(child_prim)
                    return get_geometry(mesh_prim_list, prim)

                elif prim.IsA(UsdGeom.Gprim):
                    return get_geometry(prim, prim)
                else:
                    raise ValueError(f"Unsupported prim type {get_prim_type_name(prim)}")

            # Process root links before joints so their geometry is available for joint references
            for root_link_prim in self._root_links.get(robot_name, []):
                if not root_link_prim.IsValid():
                    continue
                root_link_path = str(root_link_prim.GetPath())
                root_link_name = get_prim_name(root_link_path)
                link_geo, _base_t = extract_link_geo(root_link_prim)

                if link_geo.dim() != 2 or not is_trimesh_closed(link_geo):
                    raise RuntimeError(
                        f"Root link {root_link_path}'s mesh is not closed after automatic repair "
                        f"(dim={link_geo.dim()}). "
                        "Please check the source mesh for severe topology issues "
                        "(e.g., missing faces, self-intersections) and repair it manually in a mesh editor."
                    )

                self.scene.robo_elem.apply_to(link_geo)
                abd.apply_to(sc=link_geo, kappa=100 * 1e6, mass_density=1000.0)
                label_surface(link_geo)

                root_transform = link_geo.transforms().view()[0]
                robot.root_pose = Pose.from_transformation_matrix(root_transform)

                link_obj = self.scene.objects().create(root_link_name)
                link_slot, _ = link_obj.geometries().create(link_geo)

                robot.link_geometry[root_link_path] = {
                    "prim": root_link_prim,
                    "geo_slot": link_slot,
                    "type": "rigid_body",
                    "robot_name": robot_name,
                }
                logger.debug(f"Processed root link {root_link_name} for {robot_name}")

            # Process joints in the determined order
            for joint in sorted_joint_list:
                body1_prim: Usd.Prim = joint.body1
                if not body1_prim.IsValid():
                    continue
                link1_geo, _base_t = extract_link_geo(body1_prim)

                if link1_geo.dim() != 2 or not is_trimesh_closed(link1_geo):
                    raise RuntimeError(
                        f"Link {joint.body1.GetPath()}'s mesh is not closed after automatic repair "
                        f"(dim={link1_geo.dim()}). "
                        "Please check the source mesh for severe topology issues "
                        "(e.g., missing faces, self-intersections) and repair it manually in a mesh editor."
                    )

                self.scene.robo_elem.apply_to(link1_geo)
                abd.apply_to(sc=link1_geo, kappa=100 * 1e6, mass_density=1000.0)
                label_surface(link1_geo)

                is_floating_joint = False
                if isinstance(joint, FixedJoint) and joint.body0 is None:
                    if not robot.is_root_fixed:
                        robot.is_root_constrained = True
                        logger.debug(f"apply soft transform constraint to {str(joint.body1.GetPath())}")
                        stc = SoftTransformConstraint()
                        stc.apply_to(
                            link1_geo,
                            np.array([
                                1000.0,  # strength ratio of translation constraint
                                1000.0,  # strength ratio of rotation constraint
                            ]),
                        )
                        robot_transform = self._robot_transforms.get(robot_name, None)
                        assert robot_transform is not None, f"root_to_robot_transform for {robot_name} not found."
                        root_transform = link1_geo.transforms().view()[0]
                        robot.root_pose = Pose.from_transformation_matrix(root_transform)
                        robot.root_instruct_pose = robot.root_pose
                        robot.root_to_robot_transform = np.linalg.inv(robot_transform) @ root_transform
                        is_floating_joint = True
                    elif robot.is_root_fixed:
                        robot.is_root_constrained = True
                        root_transform = link1_geo.transforms().view()[0]
                        robot.root_pose = Pose.from_transformation_matrix(root_transform)
                        logger.debug(f"set is_fixed to 1 for {get_prim_name(joint.body1.GetPath())}")
                        view(link1_geo.instances().find(uipc_builtin.is_fixed))[:] = 1

                link_obj = self.scene.objects().create(get_prim_name(joint.body1.GetPath()))
                link_slot, _ = link_obj.geometries().create(link1_geo)

                if is_floating_joint:
                    self.animator.insert(
                        link_obj,
                        lambda info, rn=robot_name: self.get_robot(rn).floating_joint_anim(info),
                    )

                robot.link_geometry[str(joint.body1.GetPath())] = {
                    "prim": joint.body1,
                    "geo_slot": link_slot,
                    "type": "rigid_body",
                    "robot_name": robot_name,
                }

                if isinstance(joint, FixedJoint) and joint.body0 is not None:
                    abfj = AffineBodyFixedJoint()
                    body0_pose = get_position_and_orientation(joint.body0, self.metersPerUnit)
                    point0 = body0_pose.p
                    body1_pose = get_position_and_orientation(joint.body1, self.metersPerUnit)
                    point1 = body1_pose.p
                    body1_slot = robot.link_geometry[str(joint.body1.GetPath())]["geo_slot"]
                    body0_slot = robot.link_geometry[str(joint.body0.GetPath())]["geo_slot"]
                    joint_name_str = get_prim_name(joint.prim.GetPath())
                    joint_path_str = str(joint.prim.GetPath())

                    _N = N if multi else 1
                    v0s = (env_offsets_arr @ np.append(point0, 1.0))[:, :3]  # (N, 3) or (1, 3)
                    v1s = (env_offsets_arr @ np.append(point1, 1.0))[:, :3]
                    all_vs = np.empty((_N * 2, 3), dtype=np.float32)
                    all_vs[0::2] = v0s
                    all_vs[1::2] = v1s
                    all_es = np.stack([np.arange(0, _N * 2, 2), np.arange(1, _N * 2, 2)], axis=1).astype(np.int32)
                    jm = linemesh(all_vs, all_es)
                    abfj.apply_to(
                        jm,
                        [body1_slot] * _N,
                        list(range(_N)),
                        [body0_slot] * _N,
                        list(range(_N)),
                        [100.0] * _N,
                    )
                    jobj = self.scene.objects().create(joint_name_str)
                    jslot, _ = jobj.geometries().create(jm)
                    robot.joint_geometry[joint_path_str] = {
                        "prim": joint.prim,
                        "geo_slot": jslot,
                        "body0": str(joint.body0.GetPath()),
                        "body1": str(joint.body1.GetPath()),
                        "type": "fixed_joint",
                        "robot_name": robot_name,
                    }

                elif isinstance(joint, RevoluteJoint):
                    abrj = AffineBodyRevoluteJoint()
                    abdrj = AffineBodyDrivingRevoluteJoint()
                    abdrjef = AffineBodyRevoluteJointExternalForce()
                    assert joint.body0 is not None, f"Joint {joint.prim.GetPath()} has no body0"
                    joint_rel_pose = np.eye(4, dtype=np.float64)
                    body0_pose = get_position_and_orientation(joint.body0, self.metersPerUnit)
                    joint_rel_pose[:3, 3] = joint.local_pos0
                    joint_rel_pose[:3, :3] = joint.local_orient0.as_matrix()
                    joint_pose = body0_pose.to_transformation_matrix() @ joint_rel_pose
                    point0 = joint_pose[:3, 3]
                    if joint.axis == "X":
                        point1 = point0 + joint_pose[:3, 0]
                    elif joint.axis == "Y":
                        point1 = point0 + joint_pose[:3, 1]
                    elif joint.axis == "Z":
                        point1 = point0 + joint_pose[:3, 2]
                    else:
                        raise ValueError(f"joint {joint.prim} 's axis {joint.axis} is not supported.")
                    rbs_angle = joint.prim.GetAttribute("rbs:angle").Get()
                    angle_value = 0.0
                    if rbs_angle is not None:
                        angle_value = rbs_angle
                        logger.debug(f"joint {joint.prim.GetPath()} find `rbs:angle` value: {angle_value}")

                    body1_slot = robot.link_geometry[str(joint.body1.GetPath())]["geo_slot"]
                    body0_slot = robot.link_geometry[str(joint.body0.GetPath())]["geo_slot"]
                    joint_name_str = get_prim_name(joint.prim.GetPath())
                    joint_path_str = str(joint.prim.GetPath())

                    _N = N if multi else 1
                    v0s = (env_offsets_arr @ np.append(point0, 1.0))[:, :3]  # (N, 3) or (1, 3)
                    v1s = (env_offsets_arr @ np.append(point1, 1.0))[:, :3]
                    all_vs = np.empty((_N * 2, 3), dtype=np.float32)
                    all_vs[0::2] = v0s
                    all_vs[1::2] = v1s
                    all_es = np.stack([np.arange(0, _N * 2, 2), np.arange(1, _N * 2, 2)], axis=1).astype(np.int32)
                    jm = linemesh(all_vs, all_es)
                    abrj.apply_to(
                        jm,
                        [body0_slot] * _N,
                        list[int](range(_N)),
                        [body1_slot] * _N,
                        list(range(_N)),
                        [1000.0] * _N,
                    )
                    abdrj.apply_to(jm, [1000.0] * _N)
                    abdrjef.apply_to(jm, [0.0] * _N)
                    view(jm.edges().find("init_angle"))[:] = angle_value
                    view(jm.edges().find("aim_angle"))[:] = angle_value
                    view(jm.edges().find("angle"))[:] = angle_value
                    jobj = self.scene.objects().create(joint_name_str)
                    jslot, _ = jobj.geometries().create(jm)
                    self.animator.insert(
                        jobj,
                        lambda info, rn=robot_name: self.get_robot(rn).revolute_joint_anim(info),
                    )
                    robot.joint_geometry[joint_path_str] = {
                        "prim": joint.prim,
                        "geo_slot": jslot,
                        "body0": str(joint.body0.GetPath()),
                        "body1": str(joint.body1.GetPath()),
                        "type": "revolute_joint",
                        "robot_name": robot_name,
                    }
                    joint_name = joint_name_str
                    robot._joint_limits_raw[joint_name] = (joint.lower_limit, joint.upper_limit)

                elif isinstance(joint, PrismaticJoint):
                    abpj = AffineBodyPrismaticJoint()
                    abdpj = AffineBodyDrivingPrismaticJoint()
                    abpjef = AffineBodyPrismaticJointExternalForce()
                    assert joint.body0 is not None, f"Joint {joint.prim.GetPath()} has no body0"
                    joint_rel_pose = np.eye(4, dtype=np.float64)
                    body0_pose = get_position_and_orientation(joint.body0, self.metersPerUnit)
                    joint_rel_pose[:3, 3] = joint.local_pos0
                    joint_rel_pose[:3, :3] = joint.local_orient0.as_matrix()
                    joint_pose = body0_pose.to_transformation_matrix() @ joint_rel_pose
                    point0 = joint_pose[:3, 3]
                    if joint.axis == "X":
                        point1 = point0 - joint_pose[:3, 0]
                    elif joint.axis == "Y":
                        point1 = point0 - joint_pose[:3, 1]
                    elif joint.axis == "Z":
                        point1 = point0 - joint_pose[:3, 2]
                    else:
                        raise ValueError(f"joint {joint.prim} 's axis {joint.axis} is not supported.")

                    body1_slot = robot.link_geometry[str(joint.body1.GetPath())]["geo_slot"]
                    body0_slot = robot.link_geometry[str(joint.body0.GetPath())]["geo_slot"]
                    joint_name_str = get_prim_name(joint.prim.GetPath())
                    joint_path_str = str(joint.prim.GetPath())

                    _N = N if multi else 1
                    v0s = (env_offsets_arr @ np.append(point0, 1.0))[:, :3]  # (N, 3) or (1, 3)
                    v1s = (env_offsets_arr @ np.append(point1, 1.0))[:, :3]
                    all_vs = np.empty((_N * 2, 3), dtype=np.float32)
                    all_vs[0::2] = v0s
                    all_vs[1::2] = v1s
                    all_es = np.stack([np.arange(0, _N * 2, 2), np.arange(1, _N * 2, 2)], axis=1).astype(np.int32)
                    jm = linemesh(all_vs, all_es)
                    abpj.apply_to(
                        jm,
                        [body0_slot] * _N,
                        list(range(_N)),
                        [body1_slot] * _N,
                        list(range(_N)),
                        [100.0] * _N,
                    )
                    abdpj.apply_to(jm, [100.0] * _N)
                    abpjef.apply_to(jm, [0.0] * _N)
                    jobj = self.scene.objects().create(joint_name_str)
                    jslot, _ = jobj.geometries().create(jm)
                    self.animator.insert(
                        jobj,
                        lambda info, rn=robot_name: self.get_robot(rn).prismatic_joint_anim(info),
                    )
                    robot.joint_geometry[joint_path_str] = {
                        "prim": joint.prim,
                        "geo_slot": jslot,
                        "body0": str(joint.body0.GetPath()),
                        "body1": str(joint.body1.GetPath()),
                        "type": "prismatic_joint",
                        "robot_name": robot_name,
                    }
                    joint_name = joint_name_str
                    robot._joint_limits_raw[joint_name] = (joint.lower_limit, joint.upper_limit)
                else:
                    pass

            # After building the articulation
            robot.after_build(num_instances)


def topological_sort(joints: List[PhysicsJoint], use_dfs: bool = True) -> List[int]:
    """
    Topological sort of a list of PhysicsJoint objects.
    Args:
        joints (List[PhysicsJoint]): A list of PhysicsJoint objects.
        use_dfs (bool): If True, use depth-first search for topological sorting.
            If False, use Kahn's algorithm. Default is True.
    Returns:
        List[int]: A list of joint indices in topological order.
    """
    incoming = defaultdict(set)
    outgoing = defaultdict(set)
    nodes = set()
    # Create the graph representation
    for joint_id, joint in enumerate(joints):
        parent = str(joint.body0.GetPath()) if joint.body0 is not None else "None"
        child = str(joint.body1.GetPath())
        if not joint.body1.IsValid():
            continue
        if len(incoming[child]) == 1:
            raise ValueError(f"Multiple joints lead to body {child}")
        incoming[child].add((joint_id, parent))
        outgoing[parent].add((joint_id, child))
        nodes.add(parent)
        nodes.add(child)
    # Find root nodes (nodes with no incoming edges)
    roots = nodes - set(incoming.keys())
    if len(roots) == 0:
        raise ValueError("No root found in the joint graph.")
    joint_order = []
    visited = set()
    if use_dfs:
        # Depth-first search
        def visit(node):
            visited.add(node)
            # Sort by joint ID to retain original order if topological order is not unique
            outs = sorted(outgoing[node], key=lambda x: x[0])
            for joint_id, child in outs:
                if child in visited:
                    raise ValueError(f"Joint graph contains a cycle at body {child}")
                joint_order.append(joint_id)
                visit(child)

        roots = sorted(roots)
        for root in roots:
            visit(root)
    else:
        # Breadth-first search (Kahn's algorithm)
        queue = deque(sorted(roots))
        while queue:
            node = queue.popleft()
            visited.add(node)
            outs = sorted(outgoing[node], key=lambda x: x[0])
            for joint_id, child in outs:
                if child in visited:
                    raise ValueError(f"Joint graph contains a cycle at body {child}")
                joint_order.append(joint_id)
                queue.append(child)
    return joint_order
