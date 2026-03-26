# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp
from typing import Any

import numpy as np
from loguru import logger
from pxr import Usd
from uipc import Logger

from u2u.controllers import PDController
from u2u.pipeline import PipelineBase
from u2u.scene import Scene
from u2u.task_queue import Task
from u2u.utils import AssetDir

"""
Cartpole PD Control Demo

Demonstrates optimized PD controller for cartpole stabilization.
The cart is first positioned at 2.0m using position control, then PD force control
brings it back to origin.

PD Parameters (optimized for fastest convergence):
- Kp = 250.0, Kd = 38.0 → damping ratio ζ ≈ 1.2 (slightly overdamped)
- Expected convergence: ~39 frames (0.39s)
- Key formula: Kd = ζ × 2√(Kp × m), where ζ=1.0-1.2 is optimal

See cartpole_analysis.py for detailed parameter tuning theory.
"""

RAIL_CART_JOINT = "/cartPole/railCartJoint"
CART_POLE_JOINT = "/cartPole/cartPoleJoint"
POLE_POLE_JOINT = "/cartPole/polePoleJoint"

INITIAL_CONFIG = {
    RAIL_CART_JOINT: 2.0,
    CART_POLE_JOINT: 0.0,
    POLE_POLE_JOINT: 0.0,
}
SETTLE_FRAMES = 100


class CartpolePipeline(PipelineBase):
    def __init__(self, workdir: str, usd_path_or_stage: str | Usd.Stage, logger_level: Logger.Level = Logger.Info):
        super().__init__(workdir, usd_path_or_stage, logger_level=logger_level)

    def setup_config(self) -> dict[str, Any]:
        config = Scene.default_config()
        config["dt"] = 0.01
        return config

    def setup_contact_tabular(self) -> None:
        pass

    def user_build_scene(self) -> None:
        pass


class ControlCartpoleTask(Task):
    """先用 set_joint_position 将小车置于非零，再用 PD 力控回零。"""

    def __init__(self, name: str, runner: PipelineBase, priority: int = 10):
        super().__init__(name, priority)
        self.runner = runner
        self.robot = runner.scene.get_robot("/cartPole")
        self.cart_pd = PDController(kp=250.0, kd=146.0, force_or_torque_limit=500.0)
        self._switched_to_force = False
        self.runner.scene.reset_joint_position(robot_name="/cartPole", joint_positions=INITIAL_CONFIG)
        self.robot.set_joint_constrained(CART_POLE_JOINT, True)
        self.robot.set_joint_constrained(POLE_POLE_JOINT, True)
        self.rail_mass = self.runner.scene.get_mass("/cartPole/rail")
        self.cart_mass = self.runner.scene.get_mass("/cartPole/cart")
        self.pole1_mass = self.runner.scene.get_mass("/cartPole/pole1")
        self.pole2_mass = self.runner.scene.get_mass("/cartPole/pole2")

        self.cart_pole_mass = self.cart_mass + self.pole1_mass + self.pole2_mass
        logger.info(f"cart_pole_mass: {self.cart_pole_mass}")

        self.rail_inertia_matrix_com = self.runner.scene.get_inertia_matrix_com("/cartPole/rail")
        self.cart_inertia_matrix_com = self.runner.scene.get_inertia_matrix_com("/cartPole/cart")
        self.pole1_inertia_matrix_com = self.runner.scene.get_inertia_matrix_com("/cartPole/pole1")
        self.pole2_inertia_matrix_com = self.runner.scene.get_inertia_matrix_com("/cartPole/pole2")

        logger.info(
            f"rail_mass: {self.rail_mass},\n cart_mass: {self.cart_mass},\n pole1_mass: {self.pole1_mass},\n pole2_mass: {self.pole2_mass}"
        )
        logger.info(
            f"rail_inertia_matrix_com: \n{self.rail_inertia_matrix_com},\n cart_inertia_matrix_com: \n{self.cart_inertia_matrix_com},\n pole1_inertia_matrix_com: \n{self.pole1_inertia_matrix_com},\n pole2_inertia_matrix_com: \n{self.pole2_inertia_matrix_com}"
        )

    def update(self) -> None:
        frame = self.runner.world.frame()
        if not self._switched_to_force and frame >= SETTLE_FRAMES:
            self._switched_to_force = True
        if self._switched_to_force:
            cart_pos = float(self.robot.get_joint_position(RAIL_CART_JOINT))
            cart_vel = float(self.robot.get_joint_velocity(RAIL_CART_JOINT))
            force = self.cart_pd.compute(0.0, cart_pos, cart_vel)
            self.robot.set_joint_effort(RAIL_CART_JOINT, np.array([force]))


def main():
    workdir = AssetDir.output_path(__file__)
    cartpole_pipeline = CartpolePipeline(
        workdir=workdir, usd_path_or_stage=osp.join(AssetDir.usd_path(), "cartpole.usda"), logger_level=Logger.Critical
    )
    cartpole_pipeline.add_task(ControlCartpoleTask(name="control_cartpole", runner=cartpole_pipeline))
    cartpole_pipeline.run()


if __name__ == "__main__":
    main()
