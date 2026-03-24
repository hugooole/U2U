# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp
from typing import Any, List, Optional

import numpy as np
from loguru import logger

from u2u import AssetDir
from u2u.pipeline import PipelineBase
from u2u.pose import Pose
from u2u.scene import Scene
from u2u.scene_builder.articulation import Articulation
from u2u.task_queue import Task


class TwoHandsExample(PipelineBase):
    def __init__(self, workdir: str, usd_path: str):
        super().__init__(workdir, usd_path)
        self.robot = self.scene.get_robot("/World/orcahand_left")
        self.setup_joint_control_mode()

    def setup_contact_tabular(self):
        pass

    def user_build_scene(self):
        pass

    def setup_config(self) -> dict[str, Any]:
        config = Scene.default_config()
        config["contact"]["enable"] = False
        config["extras"]["root_is_fixed"] = False
        print(config)
        return config

    def setup_joint_control_mode(self):
        # set all joints to constrained
        for joint in self.robot.active_joints:
            self.robot.set_joint_constrained(joint, True)


class ControlHandTask(Task):
    def __init__(self, name, runner, robot, joint_name, priority=10):
        super().__init__(name, priority)
        self.runner = runner
        self.joint_name = joint_name
        self.runner.set_robot(robot)
        self.robot: Articulation = self.runner.robot
        self.joint_names = self.robot.active_joints

    def update(self) -> Optional[List["Task"]]:
        self.robot.set_joint_velocity(self.joint_name, 0.5)
        curr_position = self.robot.get_joint_position(self.joint_names)
        if curr_position >= self.target_angle:
            logger.info(f"Task {self.name} finished.")
            self.robot.set_joint_velocity(self.joint_name, 0.0)
            self.complete()
        return None


class ControlRootJointTask(Task):
    def __init__(self, name, runner, priority=10):
        super().__init__(name, priority)
        self.runner: PipelineBase = runner
        self.robot: Articulation = self.runner.scene.get_robot("/World/orcahand_left")

    def update(self) -> Optional[List["Task"]]:
        frame_idx = self.runner.world.frame()
        p = [0.0, 0.0, np.sin(np.pi * frame_idx / 1000.0)]
        pose = Pose(p=np.array(p), q=np.array([1.0, 0.0, 0.0, 0.0]))
        self.robot.set_root_poses(pose.to_transformation_matrix())
        if frame_idx > 1000:
            self.complete()
        return None


def main():
    runner = TwoHandsExample(
        workdir=AssetDir.output_path(__file__),
        usd_path=osp.join(AssetDir.usd_path(), "orca_hand_left.usda"),
    )

    runner.add_task(ControlRootJointTask("control_root", runner))

    runner.run()


if __name__ == "__main__":
    main()
