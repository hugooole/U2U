# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp

import numpy as np
from mplib import Planner
from pxr import Usd
from tasks_for_xarm import CloseGripper, MoveToTargetTask, OpenGripper
from uipc import Logger
from uipc.unit import GPa

from u2u import AssetDir
from u2u.pipeline import PipelineBase
from u2u.scene import Scene


class PickAndPlaceExample(PipelineBase):
    def __init__(self, workdir: str, usd_path_or_stage: str | Usd.Stage, logger_level: Logger.Level = Logger.Debug):
        super().__init__(workdir, usd_path_or_stage, logger_level=logger_level)
        self._setup_planner()

    def setup_config(self):
        config = Scene.default_config()
        config["dt"] = 1.0 / 100.0
        config["newton"]["velocity_tol"] = 0.001
        config["newton"]["transratio_tol"] = 0.01
        config["contact"]["d_hat"] = 0.001
        config["contact"]["enable"] = True
        config["contact"]["friction"]["enable"] = True
        config["sanity_check"]["enable"] = False
        return config

    def setup_contact_tabular(self):
        self.set_robot("/World/xarm6_with_gripper")
        # Set up contact tabular
        self._contact_tabular = self.scene.contact_tabular()
        self.default_friction_rate = 0.2
        self.default_resistance = 1.0 * GPa
        self._contact_tabular.default_model(self.default_friction_rate, self.default_resistance)

        # Create contact elements
        table_elem = self._contact_tabular.create("table")
        cube_elem = self._contact_tabular.create("cube")
        gripper_elem = self._contact_tabular.create("gripper")
        self._contact_tabular.insert(table_elem, cube_elem, 0.8, 1.0 * GPa)
        self._contact_tabular.insert(cube_elem, cube_elem, 0.8, 1.0 * GPa)
        self._contact_tabular.insert(gripper_elem, cube_elem, 0.8, 1.0 * GPa)
        table_elem.apply_to(self.scene.get_geometry("/World/table"))
        cube_elem.apply_to(self.scene.get_geometry("/World/target"))
        cube_elem.apply_to(self.scene.get_geometry("/World/target_01"))
        cube_elem.apply_to(self.scene.get_geometry("/World/target_02"))
        gripper_elem.apply_to(self.scene.get_geometry("/World/xarm6_with_gripper/left_finger"))
        gripper_elem.apply_to(self.scene.get_geometry("/World/xarm6_with_gripper/right_finger"))

    def user_build_scene(self) -> None:
        pass

    def _setup_planner(self):
        self.planner = Planner(
            urdf=osp.join(AssetDir.urdf_path(), "xarm/xarm6_with_gripper.urdf"),
            move_group="link6",
            srdf=osp.join(AssetDir.urdf_path(), "xarm/xarm6_with_gripper.srdf"),
        )

    def set_gripper(self, vel: float):
        # Set velocity for gripper joints
        # Control mode and constraints are automatically set by set_joint_velocity
        for joint in self.robot.active_joints[6:]:
            self.robot.set_joint_velocity(joint, vel)


def main():
    runner = PickAndPlaceExample(
        workdir=AssetDir.output_path(__file__),
        usd_path_or_stage=osp.join(AssetDir.usd_path(), "motion_gen_demo.usd"),
        logger_level=Logger.Debug,
    )
    target_pose, target_quat = runner.get_target_pose("/World/target_01")
    move_task1 = runner.add_task(
        MoveToTargetTask(
            runner,
            target_pose,
            target_quat,
            offset=np.array([-0.2, 0.0, -0.172, 1.0]),
            priority=15,
            name="step_1_move1",
        )
    )

    move_task2 = runner.add_task(
        MoveToTargetTask(
            runner,
            target_pose,
            target_quat,
            offset=np.array([0.0, 0.0, -0.172, 1.0]),
            priority=14,
            name="step_2_move2",
        )
    )
    move_task2.add_dependency(move_task1)

    close_task3 = runner.add_task(CloseGripper(runner, priority=10, name="step_3_close_gripper"))
    close_task3.add_dependency(move_task2)

    move_task4 = runner.add_task(
        MoveToTargetTask(
            runner,
            target_pose,
            target_quat,
            offset=np.array([0.0, 0.0, -0.25, 1.0]),
            priority=4,
            name="step_4_move4",
        )
    )
    move_task4.add_dependency(close_task3)

    target_pose2, target_quat2 = runner.get_target_pose("/World/target_02")
    move_task5 = runner.task_queue.add_task(
        MoveToTargetTask(
            runner,
            target_pose2,
            target_quat2,
            offset=np.array([0.0, 0.0, (-0.172 - 0.08), 1.0]),
            priority=3,
            name="step_5_move5",
        )
    )
    move_task5.add_dependency(move_task4)

    # 4. Open the gripper after waiting
    open_task6 = runner.add_task(OpenGripper(runner, priority=1, name="step_6_open_gripper"))
    open_task6.add_dependency(move_task5)

    runner.run()


if __name__ == "__main__":
    main()
