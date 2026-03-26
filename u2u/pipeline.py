# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import atexit
import os.path as osp
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polyscope as ps
import uipc.geometry as uipc_geometry
from loguru import logger
from polyscope import imgui
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation
from uipc import Engine, Logger, SceneIO, Transform
from uipc import builtin as uipc_builtin
from uipc.gui import SceneGUI

from u2u import read_usd
from u2u.mesh_factory import MeshFactory
from u2u.pose import Pose
from u2u.scene import Scene
from u2u.scene_builder import Articulation
from u2u.task_queue import Task, TaskQueue
from u2u.usd_parser import UsdParser, UsdParserConfig
from u2u.usd_utils import save_usd
from u2u.utils import (
    create_simplicial_complex,
    get_transform,
    transform_and_scale_points,
)
from u2u.world import World

if TYPE_CHECKING:
    from u2u.env import Env


class PipelineBase(ABC):
    def __init__(
        self,
        workdir: str,
        usd_path_or_stage: str | Usd.Stage,
        output_usd_path: str = None,
        logger_level: Logger.Level = Logger.Warn,
        use_warp: bool = False,
    ):
        Logger.set_level(logger_level)

        if use_warp:
            import warp as wp

            wp.init()

        self.workdir = workdir
        self.stage = read_usd(usd_path_or_stage) if isinstance(usd_path_or_stage, str) else usd_path_or_stage
        if output_usd_path is None:
            self.output_usd_path = osp.join(self.workdir, "output.usd")
            logger.info(f"Output USD path not provided. Using default: {self.output_usd_path}")
        else:
            self.output_usd_path = output_usd_path

        self.engine = Engine(backend_name="cuda", workspace=self.workdir)
        self.world = World(self.engine)
        self.config = self.setup_config()
        parser_config = self.setup_usd_parser_config()
        self.scene: Scene = UsdParser(Scene(self.config), self.stage).parse_and_build_scene(**parser_config)
        print(f"self.scene.config: {self.config}")
        self.save_scene_mode: Literal["init", "anim"] = "anim"
        self.robot: Articulation | None = None
        self._set_up_axis()
        self.scene_io = SceneIO(self.scene)
        self.world_inited = False
        self.scene_gui: SceneGUI = SceneGUI(self.scene, "split")

        self.setup_contact_tabular()
        self.user_build_scene()

        self.world.init(self.scene)
        # self.world.dump()
        self.world_inited = True

        self.after_world_init()

        # Initialize polyscope
        ps.init()
        atexit.register(self._cleanup_polyscope)
        # Register ground name
        if self.scene.ground_name is not None:
            self.scene_gui.register(self.scene.ground_name)
        else:
            self.scene_gui.register()
        # Register user callback
        ps.set_user_callback(lambda: self.ps_callback())

        # GUI Data
        self.is_running = False
        self.selected_robot_index = 0  # Track the currently selected robot index
        self._selected_instance_id: int = 0
        self._frame_count = 0
        self._screenshot_path: str | None = None

        # Initialize task queue
        self.task_queue = TaskQueue()

    def set_robot(self, name: str | None = None):
        self.robot = self.scene.get_robot(name)
        # Update selected_robot_index to match the selected robot
        robot_names = list(self.scene.robot_dict.keys())
        if name in robot_names:
            self.selected_robot_index = robot_names.index(name)

    @property
    def num_envs(self) -> int:
        """Number of environment instances (1 if multi-env is not enabled)."""
        return self.scene.num_envs

    def get_env(self, env_id: int) -> "Env":
        """Get an environment instance by ID."""
        return self.scene.get_env(env_id)

    def get_env_robot(self, env_id: int, robot_name: str) -> Articulation:
        """Get a robot from a specific environment.

        Args:
            env_id: Environment index.
            robot_name: Robot name (relative to template env, e.g. "Robot").

        Returns:
            The Articulation instance for the specified env + robot.
        """
        env = self.get_env(env_id)
        for name, robot in env.robot_dict.items():
            if name.endswith(robot_name) or name == robot_name:
                return robot
        raise RuntimeError(f"Robot '{robot_name}' not found in env {env_id}. Available: {list(env.robot_dict.keys())}")

    @abstractmethod
    def setup_config(self) -> dict[str, Any]:
        pass

    def setup_contact_tabular(self) -> None:
        pass

    def setup_usd_parser_config(self) -> "UsdParserConfig":
        """Configure USD parsing behavior.

        Subclasses can override this method to customize USD parsing parameters.

        Returns:
            Dictionary containing parsing configuration. All fields are optional:
            - root_path: Root path to start parsing from (default: "/")
            - ignore_paths: List of prim paths to exclude from parsing
            - skip_mesh_approximation: Skip automatic mesh approximation (default: False)

        Example:
            >>> def setup_usd_parser_config(self) -> UsdParserConfig:
            ...     return {
            ...         "root_path": "/World",
            ...         "ignore_paths": ["/World/Debug.*"],
            ...         "skip_mesh_approximation": True,
            ...         "approx_method": "convexdecomposition",
            ...     }
        """
        return {
            "root_path": "/",
            "ignore_paths": [],
            "skip_mesh_approximation": True,
            "approx_method": "none",
        }

    def user_build_scene(self) -> None:
        pass

    def after_world_init(self) -> None:
        pass

    def user_define_gui(self) -> None:
        """User-defined GUI elements to be added to the Polyscope interface.

        This method can be overridden by subclasses to add custom GUI elements.
        By default, it does nothing.
        """
        pass

    def _set_up_axis(self):
        # Set up Polyscope axis according to the simulation's up_axis
        if self.scene.up_axis == UsdGeom.Tokens.z:
            ps.set_up_dir("z_up")
        elif self.scene.up_axis == UsdGeom.Tokens.y:
            ps.set_up_dir("y_up")
        elif self.scene.up_axis == UsdGeom.Tokens.x:
            ps.set_up_dir("x_up")

    def _create_joint_slider(self, joint_name, position, lower_limit, upper_limit):
        """Create a slider for a joint with its name and limits.

        Args:
            joint_name: Name of the joint
            position: Current joint position
            lower_limit: Lower limit
            upper_limit: Upper limit

        Returns:
            tuple: (changed, new_position)
        """
        # Display the joint name above the slider
        imgui.Text(f"{joint_name.split('/')[-1]}")

        # Display a lower limit on the left side
        imgui.Text(f"{lower_limit:.1f}")
        imgui.SameLine()

        # Use an empty label for the slider since the name is displayed above
        changed, new_position = imgui.DragFloat(
            "##" + joint_name.split("/")[-1],
            position,
            0.1,
            lower_limit,
            upper_limit,
            "%.1f",
        )

        # Display the upper limit on the right side
        imgui.SameLine()
        imgui.Text(f"{upper_limit:.1f}")

        return changed, new_position

    def _handle_joint_sliders(self):
        """Handle joint position sliders."""
        # Create sliders for each joint
        if self.robot is None:
            return None

        imgui.Spacing()
        is_open = imgui.CollapsingHeader("Joint State Control")
        if is_open:
            for joint_name in self.robot.active_joints:
                # Convert radians to degrees for better user experience
                inst = self._selected_instance_id
                joint_type = self.robot.joint_geometry[self.robot.joint_path_map[joint_name]]["type"]
                if joint_type == "revolute_joint":
                    position = float(np.rad2deg(self.robot.get_joint_position(name=joint_name)[inst]))
                elif joint_type == "prismatic_joint":
                    position = float(self.robot.get_joint_position(name=joint_name)[inst])
                else:
                    logger.error(f"unknown joint type {joint_type}")
                    continue

                # Get joint limits
                j = self.robot._get_joint_idx(joint_name)
                lower_raw = self.robot.joint_lower_limits[j]
                upper_raw = self.robot.joint_upper_limits[j]
                lower_limit = float(lower_raw) if not np.isnan(lower_raw) else -180.0
                upper_limit = float(upper_raw) if not np.isnan(upper_raw) else 180.0

                # Create a slider for this joint
                changed_this_joint, new_position = self._create_joint_slider(
                    joint_name, position, lower_limit, upper_limit
                )

                # If the slider was changed, update our stored position
                if changed_this_joint:
                    # Update the actual joint position in the simulation
                    if joint_type == "revolute_joint":
                        self.robot.set_joint_position(joint_name, new_position * np.pi / 180.0, instance_ids=[inst])
                    elif joint_type == "prismatic_joint":
                        self.robot.set_joint_position(joint_name, new_position, instance_ids=[inst])
                    else:
                        logger.error(f"unknown joint type {joint_type}")
                        continue

        return None

    def _handle_floating_joint_controls(self):
        if self.robot is None or self.robot.is_root_fixed or not self.robot.is_root_constrained:
            return None
        imgui.Spacing()
        is_open = imgui.CollapsingHeader("Floating Joint Controls")
        if is_open:
            imgui.Spacing()
            inst = self._selected_instance_id
            multi = self.robot.num_instances > 1

            if multi:
                pose = Pose.from_transformation_matrix(self.robot.root_instruct_pose[inst])
            else:
                pose = self.robot.root_instruct_pose

            imgui.Text("Position")
            any_changed = False
            position = pose.p
            for i, axis in enumerate(["X", "Y", "Z"]):
                changed, value = imgui.DragFloat(
                    f"##pos_{axis}",
                    position[i],
                    0.001,  # speed
                    -10.0,  # min value
                    10.0,  # max value
                    f"{axis}: %.3f",
                )
                if changed:
                    any_changed = True
                    position[i] = value

            imgui.Spacing()
            imgui.Text("Rotation")
            rot = Rotation.from_quat(pose.q.squeeze(), scalar_first=True)
            rpy = rot.as_euler("xyz", degrees=True)
            for i, axis in enumerate(["roll", "pitch", "yaw"]):
                changed, value = imgui.DragFloat(f"##{axis}", rpy[i], 0.2, -180.0, 180.0, f"{axis}: %.3f")
                if changed:
                    any_changed = True
                    rpy[i] = value

            if any_changed:
                rot = Rotation.from_euler("xyz", rpy, degrees=True)
                if multi:
                    pose.set_p(position)
                    pose.set_q(rot.as_quat(scalar_first=True))
                    self.robot.root_instruct_pose[inst] = pose.to_transformation_matrix()
                else:
                    self.robot.root_instruct_pose.set_p(position)
                    self.robot.root_instruct_pose.set_q(rot.as_quat(scalar_first=True))

    def _handle_task_queue_controls(self):
        """Handle task queue control buttons and display task queue information."""
        # Create a collapsible header for task queue controls
        imgui.Spacing()
        is_open = imgui.CollapsingHeader("Task Queue Controls")

        # Only show controls if the header is open
        if is_open:
            # Add some spacing for better appearance
            imgui.Spacing()

            # Set button dimensions
            button_width = 120
            button_height = 30
            spacing_between_buttons = 40

            # Set a small left margin for better appearance
            imgui.SetCursorPosX(10)

            # Pause/Resume button
            pause_label = "Resume Queue" if self.task_queue.paused else "Pause Queue"
            if imgui.Button(label=pause_label, size=(button_width, button_height)):
                if self.task_queue.paused:
                    self.task_queue.resume()
                else:
                    self.task_queue.pause()

            # Place the Clear button on the same line with spacing
            imgui.SameLine(0, spacing_between_buttons)

            # Clear button
            if imgui.Button(label="Clear Queue", size=(button_width, button_height)):
                self.task_queue.clear()

            # Add some spacing
            imgui.Spacing()

            # Display task queue status
            imgui.Text(f"Queue Status: {'Paused' if self.task_queue.paused else 'Running'}")
            imgui.Text(f"Total tasks: {self.task_queue.get_task_count()}")

            # Display running tasks
            running_tasks = self.task_queue.get_running_tasks()
            if running_tasks:
                imgui.Text("Running tasks:")
                for task in running_tasks:
                    imgui.BulletText(f"{task.name} (Priority: {task.priority})")

            # Display pending tasks
            pending_tasks = self.task_queue.get_pending_tasks()
            if pending_tasks:
                imgui.Text("Pending tasks:")
                for task in pending_tasks:
                    dependencies = [dep.name for dep in task.dependencies]
                    dep_text = f" - Dependencies: {', '.join(dependencies)}" if dependencies else ""
                    imgui.BulletText(f"{task.name} (Priority: {task.priority}){dep_text}")

            # Add some spacing for better appearance
            imgui.Spacing()

        return

    def get_target_pose(self, name):
        from scipy.spatial.transform import Rotation as R

        geo = self.scene.get_geometry(name)
        transform: np.ndarray = geo.instances().find(uipc_builtin.transform).view()[0]

        # Extract position from transform matrix
        target_position = transform[:3, 3]

        # Extract rotation matrix and convert to quaternion
        rotation_matrix = transform[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        target_quaternion = rotation.as_quat()
        # Convert to w,x,y,z format from x,y,z,w format
        target_quaternion = np.roll(target_quaternion, 1)

        return target_position, target_quaternion

    def add_task(self, task: Task) -> Task:
        return self.task_queue.add_task(task)

    def ps_callback(self):
        if imgui.Button(label="run & stop", size=(100, 30)):
            self.is_running = not self.is_running
            logger.debug(f"is_running: {self.is_running}")
        imgui.Spacing()

        # Robot selection dropdown
        robot_names = list(self.scene.robot_dict.keys())
        if robot_names:  # Only show the dropdown if there are robots
            changed, self.selected_robot_index = imgui.Combo("Choose Robot", self.selected_robot_index, robot_names)
            self.set_robot(robot_names[self.selected_robot_index])

        if self.robot is not None and self.robot.num_instances > 1:
            robot_label = robot_names[self.selected_robot_index].split("/")[-1] if robot_names else "Robot"
            _changed, new_id = imgui.InputInt(f"{robot_label} Instance", self._selected_instance_id)
            self._selected_instance_id = max(0, min(new_id, self.robot.num_instances - 1))

        self._handle_floating_joint_controls()
        self._handle_joint_sliders()
        self._handle_task_queue_controls()
        self.user_define_gui()

        if self.is_running:
            if self.save_scene_mode == "anim":
                self.scene.write_animation_to_stage(self.world.frame())
            self.task_queue.update()
            self.world.advance()
            self.world.retrieve()
            self.scene_gui.update()

            if self.task_queue.is_finished():
                imgui.Separator()
                imgui.TextColored((0, 1, 0, 1), "All Tasks Finished!")

        self._frame_count += 1
        if self._screenshot_path is not None:
            ps.screenshot(self._screenshot_path, transparent_bg=False)
            logger.info(f"Screenshot saved to: {self._screenshot_path}")
            self._screenshot_path = None

    def import_soft_mesh_prim(self, soft_mesh_prim: Usd.Prim):
        path = str(soft_mesh_prim.GetPath())

        soft_mesh = MeshFactory.get_mesh(soft_mesh_prim, need_closed=False)

        objs = self.scene.objects().find(path)
        if len(objs) != 0:
            return
        # Create a new mesh object in the scene
        obj = self.scene.objects().create(path)

        # Preprocessing mesh before applying cloth
        t = Transform(get_transform(soft_mesh_prim))
        soft_mesh.points = transform_and_scale_points(soft_mesh.points, t, self.scene.meters_per_unit)

        # Create trimesh and apply cloth constitutions
        mesh = create_simplicial_complex(soft_mesh)

        uipc_geometry.label_surface(mesh)

        geo_slot, rest_slot = obj.geometries().create(mesh)

        self.scene.geometry_dict[path] = {
            "prim": soft_mesh_prim,
            "geo_slot": geo_slot,
            "rest_slot": rest_slot,
            "transform": t,
            "type": "soft",
        }

        return geo_slot, rest_slot

    def _cleanup_polyscope(self):
        """Shutdown polyscope to release X11 connection."""
        if getattr(self, "_polyscope_cleaned_up", False):
            return
        self._polyscope_cleaned_up = True
        try:
            ps.shutdown(allow_mid_frame_shutdown=True)
        except Exception:
            pass

    def run(self, save_on_finish=True):
        # Show the viewer
        try:
            ps.show()
        finally:
            self._cleanup_polyscope()
        if save_on_finish:
            self.save_usd()

    def screenshot(self, path: str | None = None) -> str:
        """Take a screenshot on the next frame render.

        Args:
            path: Output file path. Defaults to <workdir>/screenshot.png.

        Returns:
            The path where the screenshot will be saved.
        """
        if path is None:
            path = osp.join(self.workdir, "screenshot.png")
        self._screenshot_path = path
        logger.info(f"Screenshot scheduled, will be saved to: {path}")
        return path

    def save_usd(self):
        logger.info("Saving usd animation ...")
        if self.save_scene_mode == "init":
            self.scene.write_new_init_stage()
        save_usd(self.stage, self.output_usd_path)
