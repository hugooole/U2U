# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from typing import List, Optional

import numpy as np
from pxr import Usd
from uipc import Animation, view
from uipc import builtin as uipc_builtin
from uipc.geometry import (
    GeometrySlot,
    SimplicialComplex,
    SimplicialComplexSlot,
)

from u2u.pose import Pose

from .joint_types import (
    GeometryInfo,
    JointControlMode,
)


class Articulation:
    """Represents an articulation configuration that can be built by ArticulationBuilder.

    This class stores all the necessary information to build an articulation, including
    link geometries, joint geometries, joint positions, velocities, and limits.

    State arrays are allocated by setup_state(num_instances):
    - (N, J) shape: joint_position, joint_velocity, joint_effort, joint_instruct_*,
                    joint_is_constrained, joint_is_force_constrained, control_mode
    - (J,) shape:   joint_lower/upper_limits, joint_effort_lower/upper_limits
    - (N, 4, 4):    root_pose, root_instruct_pose
    - (4, 4):       root_to_robot_transform (shared across instances)
    """

    def __init__(
        self,
        name: str,
        robot_prim: Usd.Prim,
        is_root_fixed: bool = True,
        is_root_constrained: bool = False,
    ):
        self.name = name
        self.robot_prim = robot_prim
        self.is_root_fixed = is_root_fixed
        self.is_root_constrained = is_root_constrained
        self.joint_geometry: dict[str, GeometryInfo] = {}
        self.link_geometry: dict[str, GeometryInfo] = {}
        self.joint_path_map: dict[str, str] = {}  # short name -> full USD path

        # State arrays (initialized by setup_state)
        self.joint_position: np.ndarray = None  # (N, J)
        self.joint_velocity: np.ndarray = None  # (N, J)
        self.joint_effort: np.ndarray = None  # (N, J)
        self.joint_instruct_velocity: np.ndarray = None  # (N, J)
        self.joint_instruct_position: np.ndarray = None  # (N, J)
        self.joint_instruct_effort: np.ndarray = None  # (N, J)
        self.joint_lower_limits: np.ndarray = None  # (J,)
        self.joint_upper_limits: np.ndarray = None  # (J,)
        self.joint_effort_lower_limits: np.ndarray = None  # (J,)
        self.joint_effort_upper_limits: np.ndarray = None  # (J,)

        self.joint_is_constrained: np.ndarray = None  # (N, J)
        self.joint_is_force_constrained: np.ndarray = None  # (N, J)
        self.control_mode: np.ndarray = None  # (N, J) int8
        self.root_pose: Pose = None  # Updated by animation callbacks
        self.root_instruct_pose: Pose = None  # (N, 4, 4) when multi-instance
        self.root_to_robot_transform: np.ndarray = None  # (4, 4)

        # Temporary storage for limits during build (before setup_state)
        self._joint_limits_raw: dict[str, tuple[Optional[float], Optional[float]]] = {}
        self._joint_effort_limits_raw: dict[str, tuple[Optional[float], Optional[float]]] = {}

        self.articulation_root_prim = None
        self.joint_names: list[str] = []
        self.active_joints: list[str] = []
        self.num_instances: int = 1

    def setup_state(self, num_instances: int = 1) -> None:
        """Allocate state arrays for num_instances.

        Called after joint_names are populated to initialize (N, J) arrays
        and (J,) limit arrays from temporary _joint_limits_raw storage.
        """
        self.num_instances = num_instances
        J = len(self.active_joints)

        # Allocate (N, J) state arrays
        self.joint_position = np.zeros((num_instances, J), dtype=np.float32)
        self.joint_velocity = np.zeros((num_instances, J), dtype=np.float32)
        self.joint_effort = np.zeros((num_instances, J), dtype=np.float32)
        self.joint_instruct_velocity = np.zeros((num_instances, J), dtype=np.float32)
        self.joint_instruct_position = np.zeros((num_instances, J), dtype=np.float32)
        self.joint_instruct_effort = np.zeros((num_instances, J), dtype=np.float32)
        self.joint_is_constrained = np.zeros((num_instances, J), dtype=bool)
        self.joint_is_force_constrained = np.zeros((num_instances, J), dtype=bool)
        self.control_mode = np.full((num_instances, J), JointControlMode.NONE, dtype=np.int8)
        self.root_is_constrained = np.zeros(num_instances, dtype=bool)

        # Allocate (J,) limit arrays with NaN = unlimited
        self.joint_lower_limits = np.full(J, np.nan, dtype=np.float32)
        self.joint_upper_limits = np.full(J, np.nan, dtype=np.float32)
        self.joint_effort_lower_limits = np.full(J, np.nan, dtype=np.float32)
        self.joint_effort_upper_limits = np.full(J, np.nan, dtype=np.float32)

        # Fill limits from temporary dictionaries
        for j, joint_name in enumerate(self.active_joints):
            if joint_name in self._joint_limits_raw:
                lower, upper = self._joint_limits_raw[joint_name]
                if lower is not None:
                    self.joint_lower_limits[j] = lower
                if upper is not None:
                    self.joint_upper_limits[j] = upper
            if joint_name in self._joint_effort_limits_raw:
                lower, upper = self._joint_effort_limits_raw[joint_name]
                if lower is not None:
                    self.joint_effort_lower_limits[j] = lower
                if upper is not None:
                    self.joint_effort_upper_limits[j] = upper

        # Initialize pose arrays
        self.root_pose = Pose()
        if num_instances == 1:
            self.root_instruct_pose = Pose()
        else:
            self.root_instruct_pose = np.tile(np.eye(4, dtype=np.float64), (num_instances, 1, 1))

        if self.root_to_robot_transform is None:
            self.root_to_robot_transform = np.eye(4, dtype=np.float64)

    def _rows(self, instance_ids: Optional[List[int]]) -> slice | np.ndarray:
        """Convert instance_ids to row indexer for state arrays.

        Args:
            instance_ids: None = all instances, list = specific instances

        Returns:
            slice(None) for all, ndarray of indices for subset
        """
        if instance_ids is None:
            return slice(None)
        return np.array(instance_ids, dtype=np.int32)

    def _get_joint_idx(self, name: str) -> int:
        """Get joint index by name (short name or full USD path)."""
        # Accept both short name and full USD path
        short_name = name.split("/")[-1] if "/" in name else name
        return self.active_joints.index(short_name)

    def set_control_mode(self, name: str, mode: JointControlMode, instance_ids: Optional[List[int]] = None) -> None:
        """Set the control mode for joints.

        Note: You typically don't need to call this manually. Control mode is automatically
        set when you call set_joint_position() or set_joint_velocity().

        Args:
            name: The name of the joint
            mode: The control mode to set (POSITION, VELOCITY, or NONE)
            instance_ids: None = all instances, list = specific instances
        """
        j = self._get_joint_idx(name)
        rows = self._rows(instance_ids)
        self.control_mode[rows, j] = mode

    def set_root_poses(self, poses: np.ndarray[np.float64]) -> None:
        """Set the pose of the root link with shape (num_instances, 4, 4).

        Args:
            poses: The poses to set with shape (num_instances, 4, 4).
        """
        if self.num_instances == 1:
            assert poses.shape == (4, 4), "For single instance, poses must have shape (4, 4)"
            self.root_instruct_pose = Pose.from_transformation_matrix(poses)
        else:
            assert poses.shape == (self.num_instances, 4, 4), "poses must have shape (num_instances, 4, 4)"
            self.root_instruct_pose = poses.copy()
        self.root_is_constrained[:] = True

    def set_root_fixed(self, fixed: bool) -> None:
        """Set whether the root link is fixed.

        Args:
            fixed: Whether the root link is fixed
        """
        self.is_root_fixed = fixed

    def set_joint_position(
        self,
        name: str,
        positions: np.ndarray[np.float32],
        degree: bool = False,
        instance_ids: Optional[List[int]] = None,
    ) -> None:
        """Set joint position and automatically configure position control.

        This method automatically:
        1. Sets the joint position target
        2. Enables position constraint for the joint
        3. Sets control mode to POSITION
        4. Disables force constraint if it was enabled

        Args:
            name: The name of the joint
            positions: Target position(s). Either (N,) for all instances or scalar for one
            degree: If True, convert from degrees to radians (default: False)
            instance_ids: None = all instances, list = specific instances
        """
        positions = np.atleast_1d(np.asarray(positions, dtype=np.float32))
        if degree:
            positions = np.deg2rad(positions)
        j = self._get_joint_idx(name)
        rows = self._rows(instance_ids)

        # Broadcast scalar to all rows if needed
        if positions.size == 1:
            self.joint_instruct_position[rows, j] = positions[0]
        else:
            assert positions.shape[0] == (self.num_instances if instance_ids is None else len(instance_ids))
            self.joint_instruct_position[rows, j] = positions

        self.set_control_mode(name, JointControlMode.POSITION, instance_ids)
        self.joint_is_force_constrained[rows, j] = False
        self.joint_is_constrained[rows, j] = True

    def set_joint_positions(
        self, names: list[str], positions: np.ndarray, degree: bool = False, instance_ids: Optional[List[int]] = None
    ) -> None:
        """Set multiple joint positions at once.

        Args:
            names: Joint names
            positions: (N, M) or (M,) array where M = len(names)
            degree: If True, convert from degrees to radians
            instance_ids: None = all instances, list = specific instances
        """
        positions = np.asarray(positions, dtype=np.float32)
        if degree:
            positions = np.deg2rad(positions)
        j_indices = np.array([self._get_joint_idx(name) for name in names])
        rows = self._rows(instance_ids)

        if positions.ndim == 1:
            # Broadcast single row to all instances
            self.joint_instruct_position[rows][:, j_indices] = positions
        else:
            self.joint_instruct_position[rows][:, j_indices] = positions

        for name in names:
            self.set_control_mode(name, JointControlMode.POSITION, instance_ids)
            j = self._get_joint_idx(name)
            self.joint_is_force_constrained[rows, j] = False
            self.joint_is_constrained[rows, j] = True

    def set_joint_velocity(
        self,
        name: str,
        velocities: np.ndarray[np.float32],
        degree: bool = False,
        instance_ids: Optional[List[int]] = None,
    ) -> None:
        """Set joint velocity and automatically configure velocity control.

        This method automatically:
        1. Sets the joint velocity target
        2. Enables position constraint for the joint
        3. Sets control mode to VELOCITY
        4. Disables force constraint if it was enabled

        Args:
            name: The name of the joint
            velocities: Target velocity/velocities. Either (N,) or scalar
            degree: If True, convert from degrees to radians (default: False)
            instance_ids: None = all instances, list = specific instances
        """
        velocities = np.atleast_1d(np.asarray(velocities, dtype=np.float32))
        if degree:
            velocities = np.deg2rad(velocities)
        j = self._get_joint_idx(name)
        rows = self._rows(instance_ids)

        if velocities.size == 1:
            self.joint_instruct_velocity[rows, j] = velocities[0]
        else:
            self.joint_instruct_velocity[rows, j] = velocities

        self.set_control_mode(name, JointControlMode.VELOCITY, instance_ids)
        self.joint_is_force_constrained[rows, j] = False
        self.joint_is_constrained[rows, j] = True

    def set_joint_velocities(
        self, names: list[str], velocities: np.ndarray, degree: bool = False, instance_ids: Optional[List[int]] = None
    ) -> None:
        """Set multiple joint velocities at once.

        Args:
            names: Joint names
            velocities: (N, M) or (M,) array where M = len(names)
            degree: If True, convert from degrees to radians
            instance_ids: None = all instances, list = specific instances
        """
        velocities = np.asarray(velocities, dtype=np.float32)
        if degree:
            velocities = np.deg2rad(velocities)
        j_indices = np.array([self._get_joint_idx(name) for name in names])
        rows = self._rows(instance_ids)

        if velocities.ndim == 1:
            self.joint_instruct_velocity[rows][:, j_indices] = velocities
        else:
            self.joint_instruct_velocity[rows][:, j_indices] = velocities

        for name in names:
            self.set_control_mode(name, JointControlMode.VELOCITY, instance_ids)
            j = self._get_joint_idx(name)
            self.joint_is_force_constrained[rows, j] = False
            self.joint_is_constrained[rows, j] = True

    def set_joint_effort(
        self, name: str, efforts: np.ndarray[np.float32], instance_ids: Optional[List[int]] = None
    ) -> None:
        """Set joint effort/force and automatically configure force constraint.

        This method automatically:
        1. Sets the joint effort value
        2. Disables position constraint if it was enabled
        3. Enables force constraint for the joint

        Args:
            name: The name of the joint
            efforts: Effort value(s). Either (N,) or scalar
            instance_ids: None = all instances, list = specific instances
        """
        efforts = np.atleast_1d(np.asarray(efforts, dtype=np.float32))
        j = self._get_joint_idx(name)
        rows = self._rows(instance_ids)

        if efforts.size == 1:
            self.joint_instruct_effort[rows, j] = efforts[0]
        else:
            self.joint_instruct_effort[rows, j] = efforts

        self.set_control_mode(name, JointControlMode.NONE, instance_ids)
        self.joint_is_constrained[rows, j] = False
        self.joint_is_force_constrained[rows, j] = True

    def set_joint_efforts(
        self, names: list[str], efforts: np.ndarray, instance_ids: Optional[List[int]] = None
    ) -> None:
        """Set multiple joint efforts at once.

        Args:
            names: Joint names
            efforts: (N, M) or (M,) array where M = len(names)
            instance_ids: None = all instances, list = specific instances
        """
        efforts = np.asarray(efforts, dtype=np.float32)
        j_indices = np.array([self._get_joint_idx(name) for name in names])
        rows = self._rows(instance_ids)

        self.joint_instruct_effort[rows][:, j_indices] = efforts

        for name in names:
            self.set_control_mode(name, JointControlMode.NONE, instance_ids)
            j = self._get_joint_idx(name)
            self.joint_is_constrained[rows, j] = False
            self.joint_is_force_constrained[rows, j] = True

    def set_joint_constrained(self, name: str, constrained: bool, instance_ids: Optional[List[int]] = None) -> None:
        """Set position constraint for a joint.

        Args:
            name: The name of the joint
            constrained: Whether to enable position constraint
            instance_ids: None = all instances, list = specific instances
        """
        j = self._get_joint_idx(name)
        rows = self._rows(instance_ids)
        self.joint_is_constrained[rows, j] = constrained

    def set_joint_constrained_force(
        self, name: str, constrained: bool, instance_ids: Optional[List[int]] = None
    ) -> None:
        """Set force constraint for a joint.

        Args:
            name: The name of the joint
            constrained: Whether to enable force constraint
            instance_ids: None = all instances, list = specific instances
        """
        j = self._get_joint_idx(name)
        rows = self._rows(instance_ids)
        self.joint_is_force_constrained[rows, j] = constrained

    def get_joint_position(self, name: str) -> np.ndarray:
        """Get joint position for all instances.

        Args:
            name: The name of the joint

        Returns:
            (N,) array of positions for all instances
        """
        path = self.joint_path_map.get(name, name)
        geo_slot: SimplicialComplexSlot = self.joint_geometry[path]["geo_slot"]
        geo: SimplicialComplex = geo_slot.geometry()
        if self.joint_geometry[path]["type"] == "revolute_joint":
            return view(geo.edges().find("angle"))[:]
        elif self.joint_geometry[path]["type"] == "prismatic_joint":
            return view(geo.edges().find("distance"))[:]
        else:
            raise ValueError(f"unknown joint type {self.joint_geometry[path]['type']}")

    def get_joint_positions(self, names: list[str]) -> np.ndarray:
        """Get positions for multiple joints.

        Returns:
            (N, M) array where N = num_instances, M = len(names)
        """
        positions = []
        for name in names:
            positions.append(self.get_joint_position(name))
        return np.column_stack(positions)

    def get_joint_velocity(self, name: str) -> np.ndarray:
        """Get joint velocity for all instances.

        Args:
            name: The name of the joint

        Returns:
            (N,) array of velocities for all instances
        """
        j = self._get_joint_idx(name)
        return self.joint_velocity[:, j]

    def get_joint_velocities(self, names: list[str]) -> np.ndarray:
        """Get velocities for multiple joints.

        Returns:
            (N, M) array where N = num_instances, M = len(names)
        """
        velocities = []
        for name in names:
            velocities.append(self.get_joint_velocity(name))
        return np.column_stack(velocities)

    def floating_joint_anim(self, info: Animation.UpdateInfo):  # animation function
        """Vectorized animation callback for floating joint (root).

        Updates all instances in a single call.
        """
        geo_slots: list[GeometrySlot] = info.geo_slots()
        geo: SimplicialComplex = geo_slots[0].geometry()

        # Update root poses from geometry (all instances)
        root_transforms = view(geo.transforms())[:]
        if self.num_instances == 1:
            self.root_pose = Pose.from_transformation_matrix(root_transforms[0])
        else:
            self.root_pose = root_transforms.copy()

        # Set is_constrained flag per instance
        is_constrained_attr = geo.instances().find(uipc_builtin.is_constrained)
        view(is_constrained_attr)[:] = self.root_is_constrained.astype(int)

        # Set aim_transform for all instances
        aim_transform = geo.instances().find(uipc_builtin.aim_transform)
        if self.num_instances == 1:
            if isinstance(self.root_instruct_pose, Pose):
                view(aim_transform)[0] = self.root_instruct_pose.to_transformation_matrix()
        else:
            view(aim_transform)[:] = self.root_instruct_pose

    def revolute_joint_anim(self, info: Animation.UpdateInfo):
        """Vectorized animation callback for revolute joint.

        Updates all instances in a single call.
        """
        joint_obj = info.object()
        joint_name = joint_obj.name()
        geo_slots: list[GeometrySlot] = info.geo_slots()
        geo: SimplicialComplex = geo_slots[0].geometry()

        # Get joint index
        j = self._get_joint_idx(joint_name)

        # Read current angles (all instances)
        curr_angles = view(geo.edges().find("angle"))[:]
        is_constrained_attr = geo.edges().find(uipc_builtin.is_constrained)
        is_force_constrained_attr = geo.edges().find(uipc_builtin.is_force_constrained)

        # Set constraint flags (broadcast from first row if needed)
        view(is_constrained_attr)[:] = self.joint_is_constrained[:, j]
        view(is_force_constrained_attr)[:] = self.joint_is_force_constrained[:, j]

        # Update velocity and position
        if info.frame() > 1:
            self.joint_velocity[:, j] = (curr_angles - self.joint_position[:, j]) / info.dt()
        self.joint_position[:, j] = curr_angles

        # Pre-fetch geometry attributes (avoids C++ find() calls inside conditionals)
        aim_effort_attr = geo.edges().find("effort")
        aim_angle_attr = geo.edges().find("aim_angle")

        # Compute all masks once — no repeated AND or isin
        force_mask = self.control_mode[:, j] == JointControlMode.NONE
        vel_mask = self.control_mode[:, j] == JointControlMode.VELOCITY
        pos_mask = self.control_mode[:, j] == JointControlMode.POSITION
        pos_vel_mask = vel_mask | pos_mask
        force_constrained_mask = force_mask & self.joint_is_force_constrained[:, j]

        # Force/effort control — direct boolean indexing, no np.where
        if np.any(force_constrained_mask):
            view(aim_effort_attr)[force_constrained_mask] = self.joint_instruct_effort[force_constrained_mask, j]

        # Position/velocity control
        if np.any(pos_vel_mask):
            target_angles = curr_angles.copy()
            target_angles[vel_mask] = curr_angles[vel_mask] + self.joint_instruct_velocity[vel_mask, j] * info.dt()
            target_angles[pos_mask] = self.joint_instruct_position[pos_mask, j]

            # Apply joint limits
            lower = self.joint_lower_limits[j]
            upper = self.joint_upper_limits[j]
            if not np.isnan(lower):
                target_angles[pos_vel_mask & (target_angles < lower)] = lower
            if not np.isnan(upper):
                target_angles[pos_vel_mask & (target_angles > upper)] = upper

            # Write target angles — direct boolean indexing, no np.where
            view(aim_angle_attr)[pos_vel_mask] = target_angles[pos_vel_mask]

    def prismatic_joint_anim(self, info: Animation.UpdateInfo):
        """Vectorized animation callback for prismatic joint.

        Updates all instances in a single call.
        """
        joint_obj = info.object()
        joint_name = joint_obj.name()
        geo_slots: list[GeometrySlot] = info.geo_slots()
        geo: SimplicialComplex = geo_slots[0].geometry()

        # Get joint index
        j = self._get_joint_idx(joint_name)

        # Read current state (all instances)
        curr_distances = view(geo.edges().find("distance"))[:]
        init_distance = view(geo.edges().find("init_distance"))[0]
        curr_efforts = view(geo.edges().find("effort"))[:]
        is_constrained_attr = geo.edges().find(uipc_builtin.is_constrained)
        is_force_constrained_attr = geo.edges().find(uipc_builtin.is_force_constrained)

        # Set constraint flags
        view(is_constrained_attr)[:] = self.joint_is_constrained[:, j]
        view(is_force_constrained_attr)[:] = self.joint_is_force_constrained[:, j]

        # Update velocity and position
        if info.frame() > 1:
            self.joint_velocity[:, j] = (curr_distances - self.joint_position[:, j]) / info.dt()
        self.joint_position[:, j] = curr_distances
        self.joint_effort[:, j] = curr_efforts

        # Pre-fetch geometry attributes (avoids C++ find() calls inside conditionals)
        effort_attr = geo.edges().find("effort")
        aim_distance_attr = geo.edges().find("aim_distance")

        # Compute all masks once — no repeated AND or isin
        force_mask = self.control_mode[:, j] == JointControlMode.NONE
        vel_mask = self.control_mode[:, j] == JointControlMode.VELOCITY
        pos_mask = self.control_mode[:, j] == JointControlMode.POSITION
        pos_vel_mask = vel_mask | pos_mask
        force_constrained_mask = force_mask & self.joint_is_force_constrained[:, j]

        # Force/effort control — direct boolean indexing, np.clip replaces max/min chain
        if np.any(force_constrained_mask):
            lower_eff = self.joint_effort_lower_limits[j]
            upper_eff = self.joint_effort_upper_limits[j]
            instruct = self.joint_instruct_effort[force_constrained_mask, j]
            has_lower = not np.isnan(lower_eff)
            has_upper = not np.isnan(upper_eff)
            if has_lower or has_upper:
                target_efforts = np.clip(
                    instruct,
                    a_min=lower_eff if has_lower else None,
                    a_max=upper_eff if has_upper else None,
                )
            else:
                target_efforts = instruct
            view(effort_attr)[force_constrained_mask] = target_efforts

        # Position/velocity control
        if np.any(pos_vel_mask):
            target_distances = curr_distances.copy()
            target_distances[vel_mask] = (
                curr_distances[vel_mask] + self.joint_instruct_velocity[vel_mask, j] * info.dt()
            )
            target_distances[pos_mask] = self.joint_instruct_position[pos_mask, j]

            # Apply joint limits (curr_distance + init_distance should be within limits)
            lower = self.joint_lower_limits[j]
            upper = self.joint_upper_limits[j]
            combined_distances = target_distances + init_distance
            if not np.isnan(lower):
                # Clamp: combined_distance >= lower => target_distance >= lower - init_distance
                target_distances[pos_vel_mask & (combined_distances < lower)] = lower - init_distance
            if not np.isnan(upper):
                # Clamp: combined_distance <= upper => target_distance <= upper - init_distance
                target_distances[pos_vel_mask & (combined_distances > upper)] = upper - init_distance

            # Write target distances — direct boolean indexing, no np.where
            view(aim_distance_attr)[pos_vel_mask] = target_distances[pos_vel_mask]

    def after_build(self, num_instances: int = 1) -> None:
        """Finalize articulation setup after building.

        Called by ArticulationBuilder.build() with the number of instances.
        """
        self.setup_state(num_instances)
