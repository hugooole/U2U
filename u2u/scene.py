# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from typing import TYPE_CHECKING, Any, Iterator
from weakref import proxy

import loguru
import numpy as np
import uipc
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
from scipy.spatial.transform import Rotation as R
from uipc import Matrix4x4, Vector3, view
from uipc import Scene as UScene
from uipc import builtin as uipc_builtin
from uipc.core import AffineBodyStateAccessorFeature, FiniteElementStateAccessorFeature
from uipc.geometry import SimplicialComplex, SimplicialComplexSlot
from uipc.unit import GPa

from u2u.pose import Pose
from u2u.scene_builder import Articulation
from u2u.usd_utils import (
    set_or_add_orient_translate_with_time,
    set_or_add_transform_with_time,
)
from u2u.utils import orthogonalize_rotation_matrix

if TYPE_CHECKING:
    from u2u.env import Env


@wp.kernel(enable_backward=False)
def set_affine_body_state_kernel(
    abd_body_backend_offset: wp.array(dtype=wp.uint32, ndim=1),  # type: ignore
    abd_body_transforms: wp.array(dtype=wp.mat44d, ndim=1),  # type: ignore
    abd_body_velocities_: wp.array(dtype=wp.mat44d, ndim=1),  # type: ignore
    output_transforms_: wp.array(dtype=wp.mat44d, ndim=1),  # type: ignore
    output_velocities_: wp.array(dtype=wp.mat44d, ndim=1),  # type: ignore
):
    idx = wp.tid()
    backend_idx = abd_body_backend_offset[idx]
    output_transforms_[backend_idx] = abd_body_transforms[idx]
    output_velocities_[backend_idx] = abd_body_velocities_[idx]


if TYPE_CHECKING:
    from u2u.world import World


class Scene(UScene):
    def __init__(self, config: dict):
        super().__init__(config)
        self.meters_per_unit = 1.0
        self.env_elem = self.contact_tabular().create("env_elem")
        self.robo_elem = self.contact_tabular().create("robot_elem")
        self.default_friction_rate = 1.0
        self.default_resistance = 1.0 * GPa
        self.contact_tabular().insert(
            self.env_elem,
            self.env_elem,
            self.default_friction_rate,
            self.default_resistance,
            False,
        )
        self.contact_tabular().insert(
            self.robo_elem,
            self.robo_elem,
            self.default_friction_rate,
            self.default_resistance,
            False,
        )

        # scene geo record
        self.geometry_dict: dict[str, dict] = {}
        self.robot_dict: dict[str, Articulation] | None = None

        # Multi-env support
        self.env_dict: dict[int, "Env"] = {}
        self.num_envs: int = 1

        self.ground_name = "Ground"
        self.ground_prim = None

        self.up_axis = None

        # World proxy (set by World.init())
        self._world_proxy = None

        self._abd_state_accessor: AffineBodyStateAccessorFeature | None = None
        self._abd_state_geo: SimplicialComplex | None = None
        self._fem_state_accessor: FiniteElementStateAccessorFeature | None = None
        self._fem_state_geo: SimplicialComplex | None = None

    def _set_world(self, world: "World") -> None:
        """Set the World proxy for this scene.

        This is called internally by World.init() to establish a
        back-reference from Scene to World using a weak proxy.
        This avoids circular references while allowing transparent access.

        Args:
            world: The World instance managing this scene.

        Raises:
            RuntimeError: If world is already set.
        """
        if self._world_proxy is not None:
            raise RuntimeError("World has already been set for this scene.")
        self._world_proxy = proxy(world)

    @property
    def world(self) -> "World":
        """Get the World instance managing this scene.

        Returns:
            The World instance (via weak proxy).

        Raises:
            RuntimeError: If World has not been initialized yet.
            ReferenceError: If World was deleted (should not happen in normal use).
        """
        if self._world_proxy is None:
            raise RuntimeError("World has not been initialized. Call world.init(scene) before accessing scene.world.")
        # Accessing proxy automatically validates World still exists
        # Returns the World transparently (no need to call ())
        return self._world_proxy

    @property
    def affine_body_state_accessor(self) -> AffineBodyStateAccessorFeature:
        if self._abd_state_accessor is None:
            self._abd_state_accessor = self.world.features().find(AffineBodyStateAccessorFeature)
            self._abd_state_geo = self._abd_state_accessor.create_geometry()
            self._abd_state_geo.instances().create(uipc_builtin.transform, Matrix4x4.Zero())
            self._abd_state_geo.instances().create(uipc_builtin.velocity, Matrix4x4.Zero())
        return self._abd_state_accessor

    @property
    def finite_element_state_accessor(self) -> FiniteElementStateAccessorFeature:
        if self._fem_state_accessor is None:
            self._fem_state_accessor = self.world.features().find(FiniteElementStateAccessorFeature)
            self._fem_state_geo = self._fem_state_accessor.create_geometry()
            self._fem_state_geo.instances().create(uipc_builtin.position, Vector3.Zero())
        return self._fem_state_accessor

    def reset_affine_body_state(
        self,
        backend_abd_body_offset_: np.ndarray,
        abd_body_transforms_: np.ndarray,
        abd_body_velocity_mats_: np.ndarray | None = None,
    ) -> None:
        """Reset affine body transforms using GPU-accelerated Warp kernels.

        This method efficiently updates the transforms of affine bodies in the physics simulation
        by mapping backend body offsets to their corresponding transformation matrices.

        Args:
            backend_abd_body_offset_: A 1D numpy array of uint32 indices mapping front-end body IDs
                to backend body IDs. Shape: (num_bodies,)
            abd_body_transforms_: A 2D numpy array of 4x4 transformation matrices stored in the backend.
                Shape: (num_bodies, 4, 4) - must match length of backend_abd_body_offset_
            abd_body_velocity_mats_: A 2D numpy array of 4x4 velocity matrices stored in the backend.
                Shape: (num_bodies, 4, 4) - must match length of backend_abd_body_offset_
        Raises:
            ValueError: If input arrays have incompatible shapes or dtypes.
            RuntimeError: If World has not been initialized.

        Example:
            >>> # Assuming 3 bodies with backend indices [0, 1, 2]
            >>> offsets = np.array([0, 1, 2], dtype=np.uint32)
            >>> transforms = np.zeros((3, 4, 4))  # Must have same length as offsets
            >>> velocity_mats = np.zeros((3, 4, 4))  # Must have same length as offsets
            >>> transforms[0] = np.eye(4)
            >>> transforms[1] = np.eye(4) * 2
            >>> transforms[2] = np.eye(4) * 3
            >>> velocity_mats[0] = np.eye(4)
            >>> velocity_mats[1] = np.eye(4) * 2
            >>> velocity_mats[2] = np.eye(4) * 3
            >>> scene.reset_affine_body_state(offsets, transforms, velocity_mats)
        """
        # Input validation
        if backend_abd_body_offset_.ndim != 1:
            raise ValueError(f"backend_abd_body_offset_ must be 1D, got shape {backend_abd_body_offset_.shape}")
        if abd_body_transforms_.ndim != 3 or abd_body_transforms_.shape[1:] != (4, 4):
            raise ValueError(f"abd_body_transforms_ must be shape (N, 4, 4), got {abd_body_transforms_.shape}")
        if backend_abd_body_offset_.dtype != np.uint32:
            raise ValueError(f"backend_abd_body_offset_ must be uint32, got {backend_abd_body_offset_.dtype}")
        if len(backend_abd_body_offset_) != len(abd_body_transforms_):
            raise ValueError(
                f"Length mismatch: backend_abd_body_offset_ has {len(backend_abd_body_offset_)} elements "
                f"but abd_body_transforms_ has {len(abd_body_transforms_)} elements. They must be equal."
            )
        if len(backend_abd_body_offset_) == 0:
            return  # No bodies to update

        if len(backend_abd_body_offset_) != len(abd_body_transforms_):
            raise ValueError("Length of backend_abd_body_offset_ must match length of abd_body_transforms_")

        if abd_body_velocity_mats_ is not None:
            if abd_body_velocity_mats_.ndim != 3 or abd_body_velocity_mats_.shape[1:] != (4, 4):
                raise ValueError(
                    f"abd_body_velocity_mats_ must be shape (N, 4, 4), got {abd_body_velocity_mats_.shape}"
                )
            if len(backend_abd_body_offset_) != len(abd_body_velocity_mats_):
                raise ValueError(
                    f"Length mismatch: backend_abd_body_offset_ has {len(backend_abd_body_offset_)} elements "
                    f"but abd_body_velocity_mats_ has {len(abd_body_velocity_mats_)} elements. They must be equal."
                )
        if abd_body_velocity_mats_ is None:
            abd_body_velocity_mats_ = np.zeros((len(backend_abd_body_offset_), 4, 4))

        num_bodies = len(backend_abd_body_offset_)

        # Get state data from backend
        self.affine_body_state_accessor.copy_to(self._abd_state_geo)
        transform_view = view(self._abd_state_geo.transforms())
        velocity_view = view(self._abd_state_geo.instances().find(uipc_builtin.velocity))
        transform_array = wp.from_numpy(transform_view)
        velocity_array = wp.from_numpy(velocity_view)
        backend_abd_body_offset_wp = wp.from_numpy(backend_abd_body_offset_)
        abd_body_transforms_wp = wp.from_numpy(abd_body_transforms_)
        abd_body_velocity_mats_wp = wp.from_numpy(abd_body_velocity_mats_)

        # Launch GPU kernel to update transforms
        wp.launch(
            set_affine_body_state_kernel,
            dim=num_bodies,
            inputs=[
                backend_abd_body_offset_wp,
                abd_body_transforms_wp,
                abd_body_velocity_mats_wp,
                transform_array,
                velocity_array,
            ],
        )

        # Copy modified state back to backend
        transform_view[:] = transform_array.numpy()
        velocity_view[:] = velocity_array.numpy()
        self.affine_body_state_accessor.copy_from(self._abd_state_geo)

    def reset_joint_position(
        self,
        robot_name: str,
        joint_positions: dict[str, float],
        velocities: dict[str, float] | None = None,
    ) -> None:
        """Reset joint positions and recompute link transforms via forward kinematics.

        This method directly sets the state of joints and propagates transforms through
        the kinematic tree. Unlike set_joint_position() which sets control targets,
        this method bypasses the control loop and immediately applies the new state.

        Args:
            robot_name: Full path to the robot (e.g., "/cartPole")
            joint_positions: Dictionary mapping joint names to desired positions.
                - For revolute joints: angles in radians
                - For prismatic joints: distances in meters
                - Fixed joints are ignored
            velocities: Optional dictionary mapping joint names to angular/linear velocities.
                If None, velocities are set to zero.

        Raises:
            RuntimeError: If robot_name not found in scene
            ValueError: If joint_positions contains unknown joint names
            ValueError: If velocities keys don't match joint_positions

        Example:
            >>> scene.reset_joint_position(
            ...     robot_name="/cartPole",
            ...     joint_positions={
            ...         "/cartPole/railCartJoint": 1.5,
            ...         "/cartPole/cartPoleJoint": np.pi / 6,
            ...         "/cartPole/polePoleJoint": 0.0,
            ...     }
            ... )

        Notes:
            - This method recomputes forward kinematics manually
            - Link transforms are updated immediately via reset_affine_body_state
            - Use this for initialization or state reset scenarios
            - For control during simulation, use set_joint_position() instead
        """
        # 1. Validate inputs
        robot = self._validate_reset_joint_inputs(robot_name, joint_positions, velocities)

        # 2. Build kinematic tree structure
        tree_info = self._build_kinematic_tree(robot)

        # 3. Compute forward kinematics
        link_transforms = self._compute_forward_kinematics(robot, tree_info, joint_positions)

        # 4. Collect backend offsets and transforms for batch update
        backend_offsets, transforms, velocity_mats = self._collect_link_states(robot, link_transforms, velocities)

        # 5. Set joint_constrained to False for all joints
        for joint_path in robot.active_joints:
            robot.set_joint_constrained(joint_path, False)

        # 6. Batch update all links transforms to backend via reset_affine_body_state
        if len(backend_offsets) > 0:
            self.reset_affine_body_state(backend_offsets, transforms, velocity_mats)

    def _validate_reset_joint_inputs(
        self,
        robot_name: str,
        joint_positions: dict[str, float],
        velocities: dict[str, float] | None,
    ) -> Articulation:
        """Validate inputs for reset_joint_position."""
        if robot_name not in self.robot_dict:
            raise RuntimeError(f"Robot {robot_name} not found in scene")

        robot = self.robot_dict[robot_name]

        # Check all joint names are valid
        unknown_joints = set(joint_positions.keys()) - set(robot.joint_geometry.keys())
        if unknown_joints:
            raise ValueError(f"Unknown joints: {unknown_joints}")

        # Check velocities if provided
        if velocities is not None:
            if set(velocities.keys()) != set(joint_positions.keys()):
                raise ValueError("Velocities must have the same joint names as joint_positions")

        return robot

    def _build_kinematic_tree(self, robot: Articulation) -> dict:
        """Build kinematic tree structure from joint geometry.

        Returns:
            dict with keys:
                - "adjacency": dict mapping parent link path to list of (child link path, joint path)
                - "root_link": root link path (has no parent)
                - "joint_info": dict mapping joint path to joint information
        """
        from collections import defaultdict

        adjacency = defaultdict(list)
        # Determine root by finding link that is never a body1
        all_body1s = {robot.joint_geometry[j]["body1"] for j in robot.joint_geometry}
        root_link = None
        for joint_path in robot.joint_geometry:
            body0 = robot.joint_geometry[joint_path]["body0"]
            if body0 not in all_body1s:
                root_link = body0
                break

        # Build adjacency from ALL joints
        for joint_path in robot.joint_geometry:  # Not just active_joints
            joint_data = robot.joint_geometry[joint_path]
            adjacency[joint_data["body0"]].append((joint_data["body1"], joint_path))

        return {"adjacency": adjacency, "root_link": root_link, "joint_info": robot.joint_geometry}

    def _compute_forward_kinematics(
        self,
        robot: Articulation,
        tree_info: dict,
        joint_positions: dict[str, float],
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics to get link world transforms."""
        from collections import deque

        adjacency = tree_info["adjacency"]
        root_link = tree_info["root_link"]
        joint_info = tree_info["joint_info"]

        link_transforms = {}

        # Start with root transform
        root_transform = robot.root_pose.to_transformation_matrix()
        if root_link is not None:
            link_transforms[root_link] = root_transform

        # BFS traversal to ensure parent is computed before children
        queue = deque([(root_link, root_transform)])

        while queue:
            parent_link, parent_transform = queue.popleft()

            if parent_link not in adjacency:
                continue

            for child_link, joint_path in adjacency[parent_link]:
                # Compute joint transform
                joint_transform = self._compute_joint_transform(
                    joint_info[joint_path], joint_positions.get(joint_path, 0.0)
                )

                # Compute child world transform
                child_transform = parent_transform @ joint_transform
                link_transforms[child_link] = child_transform

                queue.append((child_link, child_transform))

        return link_transforms

    def _compute_joint_transform(self, joint_info: dict, position: float) -> np.ndarray:
        """Compute 4x4 transformation matrix for a joint at given position."""
        from pxr import UsdPhysics

        joint_type = joint_info["type"]
        joint_prim = joint_info["prim"]

        physics_joint = UsdPhysics.Joint(joint_prim)

        # Get local poses
        local_pos0 = np.array(physics_joint.GetLocalPos0Attr().Get() or [0, 0, 0])
        local_rot0 = physics_joint.GetLocalRot0Attr().Get()
        if local_rot0 is None:
            local_rot0_mat = np.eye(3)
        else:
            # Quatf: GetReal() -> w, GetImaginary() -> (x, y, z)
            w = local_rot0.GetReal()
            img = local_rot0.GetImaginary()
            local_rot0_mat = R.from_quat([img[0], img[1], img[2], w]).as_matrix()

        local_pos1 = np.array(physics_joint.GetLocalPos1Attr().Get() or [0, 0, 0])
        local_rot1 = physics_joint.GetLocalRot1Attr().Get()
        if local_rot1 is None:
            local_rot1_mat = np.eye(3)
        else:
            w = local_rot1.GetReal()
            img = local_rot1.GetImaginary()
            local_rot1_mat = R.from_quat([img[0], img[1], img[2], w]).as_matrix()

        # Build parent-to-joint transform
        parent_to_joint = np.eye(4)
        parent_to_joint[:3, :3] = local_rot0_mat
        parent_to_joint[:3, 3] = local_pos0

        # Build joint-to-child transform (inverse of child's local pose)
        joint_to_child = np.eye(4)
        joint_to_child[:3, :3] = local_rot1_mat.T
        joint_to_child[:3, 3] = -local_rot1_mat.T @ local_pos1

        if joint_type == "revolute_joint":
            revolute_joint = UsdPhysics.RevoluteJoint(joint_prim)
            axis_attr = revolute_joint.GetAxisAttr().Get()
            axis = axis_attr.lower() if axis_attr else "x"

            if axis == "x":
                axis_vec = np.array([1, 0, 0])
            elif axis == "y":
                axis_vec = np.array([0, 1, 0])
            elif axis == "z":
                axis_vec = np.array([0, 0, 1])
            else:
                raise ValueError(f"Unknown axis: {axis}")

            rotation = R.from_rotvec(position * axis_vec)
            joint_rotation = np.eye(4)
            joint_rotation[:3, :3] = rotation.as_matrix()

            return parent_to_joint @ joint_rotation @ joint_to_child

        elif joint_type == "prismatic_joint":
            prismatic_joint = UsdPhysics.PrismaticJoint(joint_prim)
            axis_attr = prismatic_joint.GetAxisAttr().Get()
            axis = axis_attr.lower() if axis_attr else "x"

            if axis == "x":
                axis_vec = np.array([1, 0, 0])
            elif axis == "y":
                axis_vec = np.array([0, 1, 0])
            elif axis == "z":
                axis_vec = np.array([0, 0, 1])
            else:
                raise ValueError(f"Unknown axis: {axis}")

            translation = np.eye(4)
            translation[:3, 3] = position * axis_vec

            return parent_to_joint @ translation @ joint_to_child

        elif joint_type == "fixed_joint":
            return parent_to_joint @ joint_to_child

        else:
            raise ValueError(f"Unknown joint type: {joint_type}")

    def _collect_link_states(
        self,
        robot: Articulation,
        link_transforms: dict[str, np.ndarray],
        velocities: dict[str, float] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Collect backend offsets and transforms for all links.

        Returns:
            Tuple of (backend_offsets, transforms, velocity_mats)
        """
        backend_offsets = []
        transforms = []

        for link_path in sorted(link_transforms.keys()):
            if link_path not in robot.link_geometry:
                continue

            link_geo = robot.link_geometry[link_path]["geo_slot"].geometry()
            backend_offset = link_geo.meta().find(uipc_builtin.backend_abd_body_offset).view()[0]
            backend_offsets.append(backend_offset)
            transforms.append(link_transforms[link_path])

        backend_offsets_array = np.array(backend_offsets, dtype=np.uint32)
        transforms_array = np.array(transforms)

        # Handle velocities (set to zero if not provided)
        velocity_mats_array = None
        if velocities is not None:
            # TODO: Implement velocity matrix construction from joint velocities
            # For now, set to zero
            velocity_mats_array = np.zeros_like(transforms_array)
        else:
            velocity_mats_array = None

        return backend_offsets_array, transforms_array, velocity_mats_array

    def get_geometry(self, name: str) -> SimplicialComplex:
        if name in self.geometry_dict:
            return self.geometry_dict[name]["geo_slot"].geometry()
        else:
            raise RuntimeError(f"Geometry {name} not found.")

    def get_mass(self, name: str) -> float:
        """Get the mass of the geometry."""
        geo = self.get_geometry(name)
        return geo.instances().find(uipc_builtin.total_mass).view()[0]

    def get_inertia_matrix_com(self, name: str) -> np.ndarray:
        """Get the inertia matrix of the geometry at the center of mass."""
        geo = self.get_geometry(name)
        return geo.instances().find(uipc_builtin.inertia_tensor).view()[0]

    def get_robot(self, name: str) -> Articulation:
        if name in self.robot_dict:
            return self.robot_dict[name]
        else:
            raise RuntimeError(f"Robot {name} not found.")

    def get_env(self, env_id: int) -> "Env":
        """Get an environment instance by ID.

        Args:
            env_id: The environment index (0-based).

        Returns:
            The Env object for the given ID.

        Raises:
            RuntimeError: If env_id is not found or multi-env is not enabled.
        """
        if env_id not in self.env_dict:
            raise RuntimeError(f"Env {env_id} not found. Available: {list(self.env_dict.keys())}")
        return self.env_dict[env_id]

    def _update_robot_transforms(self, robot: Articulation, frame_idx: int | None = None) -> dict[str, Any]:
        if frame_idx is None:
            frame_idx = Usd.TimeCode.Default()

        root_pose: Pose = robot.root_pose
        root_trans = root_pose.to_transformation_matrix()
        root_to_robot_transform = robot.root_to_robot_transform

        robot_transform = root_trans @ np.linalg.inv(root_to_robot_transform)
        prim = robot.robot_prim
        prim = prim if prim.IsA(UsdGeom.Xformable) else prim.GetParent()
        xformable = UsdGeom.Xformable(prim)
        set_or_add_transform_with_time(xformable, Gf.Matrix4d(robot_transform.T), frame_idx)
        return {
            "prim_path": robot.name,
            "transform": robot_transform,
            "timestamp": frame_idx,
        }

    def _update_geo_poses(self, geo_info: dict, frame_idx: int | None = None) -> dict[str, Any]:
        if frame_idx is None:
            frame_idx = Usd.TimeCode.Default()

        prim = geo_info["prim"]
        geo_slot: SimplicialComplexSlot = geo_info["geo_slot"]
        geo = geo_slot.geometry()
        instance_id = geo_info.get("instance_id", 0)
        trans = geo.transforms().view()[instance_id]
        m = trans.copy()

        robot_name = geo_info.get("robot_name", None)
        if robot_name is not None:
            root_pose: Pose = self.robot_dict[robot_name].root_pose
            root_trans = root_pose.to_transformation_matrix()
            root_to_robot_transform = self.robot_dict[robot_name].root_to_robot_transform
            m = root_to_robot_transform @ np.linalg.inv(root_trans) @ m

        m[0:3, 3] /= self.meters_per_unit  # Convert to original unit
        # set the transform to the prim
        rotation = np.array(m[:3, :3], dtype=np.float32)
        rotation = orthogonalize_rotation_matrix(rotation)  # Ensure valid rotation matrix
        quat = R.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        gf_quatf = Gf.Quatf(*quat)
        xformable = UsdGeom.Xformable(prim)
        set_or_add_orient_translate_with_time(xformable, gf_quatf, Gf.Vec3f(*m[:3, 3]), frame_idx)

        return {
            "prim_path": str(prim.GetPath()),
            "transform": m,
            "timestamp": frame_idx,
        }

    def _update_geo_points(self, geo_info: dict, frame_idx: int | None = None) -> dict[str, Any]:
        if frame_idx is None:
            frame_idx = Usd.TimeCode.Default()
        prim = geo_info["prim"]
        geo_slot = geo_info["geo_slot"]
        t = geo_info["transform"]
        transform = np.array(t.matrix())
        transform_inv = np.linalg.inv(transform)
        geo_slot: SimplicialComplexSlot = geo_slot
        geo = geo_slot.geometry()
        vert_positions = geo.vertices().find(uipc_builtin.position).view()[..., 0]  # np.ndarray with shape (n, 3)
        vert_positions /= self.meters_per_unit
        homogeneous_vert_positions = np.hstack([vert_positions, np.ones((vert_positions.shape[0], 1))])
        vert_positions = (transform_inv @ homogeneous_vert_positions.transpose()).transpose()[:, :3]
        vec3f_array = Vt.Vec3fArray([Gf.Vec3f(*v) for v in vert_positions])
        prim.GetAttribute("points").Set(vec3f_array, frame_idx)
        return {
            "prim_path": str(prim.GetPath()),
            "points": vert_positions,
            "timestamp": frame_idx,
        }

    def _update_link_poses(self, robot: Articulation, frame_idx: int | None = None):
        pass

    def _update_joint_angles(self, robot: Articulation, frame_idx: int | None = None):
        if frame_idx is None:
            frame_idx = Usd.TimeCode.Default()
        for joint in robot.joint_geometry.values():
            geo_slot = joint["geo_slot"]
            if geo_slot is None:
                continue
            geo: uipc.geometry.SimplicialComplex = geo_slot.geometry()
            prim: Usd.Prim = joint["prim"]
            joint_angle = prim.GetAttribute("rbs:angle")
            angle_value = geo.edges().find("angle").view()[0]
            print(f"Updating joint {prim.GetPath()} angle to {angle_value} at frame {frame_idx}")
            if joint_angle is None:
                # add a default angle attribute
                joint_angle = prim.CreateAttribute("rbs:angle", Sdf.ValueTypeNames.Float, custom=True)
            joint_angle.Set(angle_value, frame_idx)

    def write_animation_to_stage(self, frame_idx: int):
        if len(self.robot_dict) != 0:
            for robot in self.robot_dict.values():
                self._update_robot_transforms(robot, frame_idx)

        for geo_info in self.geometry_dict.values():
            geo_type = geo_info["type"]
            if geo_type == "rigid_body":
                self._update_geo_poses(geo_info, frame_idx)
            elif geo_type == "collider":
                # Static colliders don't need transform updates
                continue
            else:
                self._update_geo_points(geo_info, frame_idx)

        # for robot in self.robot_dict.values():
        #     self._update_joint_angles(robot, frame_idx)

    def animation_iterator(self, frame_idx: int) -> Iterator[dict[str, Any]]:
        """
        Generates animation frames by updating transforms and points for robots and geometry
        objects based on the given frame index. This function iterates over all robots and
        geometry objects, applies their corresponding updates, and yields updated information
        for each.

        Args:
            frame_idx (int): The index of the animation frame to generate.

        Yields:
            Iterator[dict[str, Any]]: An iterator over updated robot and geometry objects'
            transforms or points for the specified frame index.
        """
        loguru.logger.debug(f"Generating animation for frame {frame_idx}")
        if len(self.robot_dict) != 0:
            loguru.logger.debug(f"Updating robot {self.robot_dict} transforms for frame {frame_idx}")
            for robot in self.robot_dict.values():
                yield self._update_robot_transforms(robot, frame_idx)

        for geo_info in self.geometry_dict.values():
            geo_type = geo_info["type"]
            if geo_type == "rigid_body":
                loguru.logger.debug(f"Updating rigid body {geo_info} transform for frame {frame_idx}")
                yield self._update_geo_poses(geo_info, frame_idx)
            elif geo_type == "collider":
                continue
            else:
                loguru.logger.debug(f"Updating point cloud {geo_info} points for frame {frame_idx}")
                yield self._update_geo_points(geo_info, frame_idx)

    def write_new_init_stage(self):
        self.write_animation_to_stage(Usd.TimeCode.Default())
