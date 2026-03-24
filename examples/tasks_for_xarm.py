# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from typing import List, Optional

import numpy as np
from loguru import logger
from mplib import Pose

from u2u.scene_builder.articulation import JointControlMode
from u2u.task_queue import Task


class MoveToTargetTask(Task):
    """Task to move the robot arm to the target position."""

    def __init__(
        self,
        runner,
        target_pose,
        target_quat,
        offset,
        control_mode: JointControlMode = JointControlMode.POSITION,
        name: str = "MoveToTarget",
        priority: int = 0,
    ):
        """Initialize the move to target task.

        Args:
            runner: The example runner instance
            name: Optional name for the task
            priority: Priority of the task
        """
        super().__init__(name=name, priority=priority)
        self.runner = runner
        self.scene = runner.scene
        self.planner = runner.planner
        self.control_mode = control_mode
        target_position, target_quaternion = target_pose, target_quat
        # Create rotation matrix from quaternion
        from scipy.spatial.transform import Rotation as R

        rotation_matrix = R.from_quat(np.roll(target_quaternion, -1)).as_matrix()

        # compute target pose and quat
        target_transform = np.eye(4)
        target_transform[:3, :3] = rotation_matrix
        target_transform[:3, 3] = target_position
        transformed_offset = target_transform @ offset
        self.target_position = transformed_offset[:3]
        self.target_quaternion = target_quaternion

        self.path = None
        self.path_vel = None
        self.current_step = 0
        self.is_planning_done = False

    def update(self) -> Optional[List[Task]]:
        """Update the robot position.

        Returns:
            None: This task doesn't create new tasks.
        """
        # Plan the path if not already done
        if not self.is_planning_done:
            try:
                # Create a Pose object from the target position and quaternion
                goal_pose = Pose(self.target_position, self.target_quaternion)

                # Get the current joint positions for all joints
                current_qpos = np.array(
                    [float(self.runner.robot.get_joint_position(joint)[0]) for joint in self.runner.robot.active_joints]
                )

                # Create a mask to only use the first 6 joints
                mask: list[bool] = [False] * 6 + [True] * (len(self.runner.robot.active_joints) - 6)

                # Plan a path to the target pose
                result = self.planner.plan_pose(
                    goal_pose=goal_pose,
                    current_qpos=current_qpos,
                    mask=mask,
                    time_step=0.01,
                )

                # Check if planning failed (result is a dict with status key)
                if result["status"] != "Success":
                    logger.error(f"{self.name} planning failed: {result['status']}")
                    self.fail()
                    return None

                self.path = result["position"]
                self.path_vel = result["velocity"]
                self.is_planning_done = True
                logger.info(f"{self.name} planning completed with {len(self.path)} waypoints")
            except Exception as e:
                logger.error(f"{self.name} planning failed: {e}")
                self.fail()
                return None

        # Execute the planned path
        if self.current_step < len(self.path):
            # Get the joint positions for the current step
            joint_positions = self.path[self.current_step]
            joint_velocities = self.path_vel[self.current_step]

            # Set the joint positions or velocities in the simulation
            # Control mode and constraints are automatically set
            for i, joint_name in enumerate(self.runner.robot.active_joints[:6]):  # Only the arm joints, not the gripper
                if self.control_mode == JointControlMode.POSITION:
                    self.runner.robot.set_joint_position(joint_name, joint_positions[i])
                elif self.control_mode == JointControlMode.VELOCITY:
                    self.runner.robot.set_joint_velocity(joint_name, joint_velocities[i])
                else:
                    logger.error(f"Invalid control mode: {self.control_mode}")
                    pass

            self.current_step += 1
        else:
            logger.info(f"{self.name} task finished")
            self.complete()

        return None


class CloseGripper(Task):
    """Task to close the gripper until it reaches a certain angle."""

    def __init__(self, runner, name: str = "CloseGripper", priority: int = 0):
        """Initialize the close gripper task.

        Args:
            runner: The example runner instance
            name: Optional name for the task
            priority: Priority of the task
        """
        super().__init__(name=name, priority=priority)
        self.runner = runner
        self.scene = runner.scene
        self.target_angle = np.deg2rad(30)

    def update(self) -> Optional[List[Task]]:
        """Update the gripper position.

        Returns:
            None: This task doesn't create new tasks.
        """
        # Update the current gripper angle
        gripper_joint_angle = float(self.runner.robot.get_joint_position(self.runner.robot.active_joints[7])[0])

        # Set the gripper velocity to close
        self.runner.set_gripper(0.5)

        # Check if we've reached the target angle
        if gripper_joint_angle >= self.target_angle:
            logger.info(f"{self.name} task finished")
            self.runner.set_gripper(0.0)
            self.complete()

        return None


class OpenGripper(Task):
    """Task to open the gripper until it reaches a certain angle."""

    def __init__(self, runner, name: str = "OpenGripper", priority: int = 0):
        """Initialize the open gripper task.

        Args:
            runner: The example runner instance
            name: Optional name for the task
            priority: Priority of the task
        """
        super().__init__(name=name, priority=priority)
        self.runner = runner
        self.scene = runner.scene
        self.gripper_joint_angle = float(runner.robot.get_joint_position(runner.robot.active_joints[7])[0])
        self.target_angle = np.deg2rad(0)  # Fully open

    def update(self) -> Optional[List[Task]]:
        """Update the gripper position.

        Returns:
            None: This task doesn't create new tasks.
        """
        # Update the current gripper angle
        self.gripper_joint_angle = float(self.runner.robot.get_joint_position(self.runner.robot.active_joints[7])[0])

        # Set the gripper velocity to open
        self.runner.set_gripper(-0.5)

        # Check if we've reached the target angle
        if self.gripper_joint_angle <= self.target_angle:
            logger.info(f"{self.name} task finished")
            self.runner.set_gripper(0.0)
            self.complete()

        return None
