# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import math
import os.path as osp
from typing import Any

from loguru import logger
from polyscope import imgui
from pxr import Usd
from uipc import Logger

from u2u.pipeline import PipelineBase
from u2u.scene import Scene
from u2u.utils import AssetDir

"""
Joint Position Reset Demo

Demonstrates the difference between:
1. set_joint_position() - Sets control target (physics-based control)
2. reset_joint_position() - Directly resets state (kinematic reset)

This example uses the cartpole and shows how to:
- Reset all joints to a specific configuration mid-simulation
- Continue simulation from the reset state
- Use reset for initialization or debugging

Key Concepts:
- reset_joint_position() computes forward kinematics manually
- All link transforms are updated immediately
- Useful for initialization, state reset, or debugging specific configurations
"""

# Constants
RAIL_CART_JOINT = "/cartPole/railCartJoint"
CART_POLE_JOINT = "/cartPole/cartPoleJoint"
POLE_POLE_JOINT = "/cartPole/polePoleJoint"

# Configuration presets
UPRIGHT_CONFIG = {
    RAIL_CART_JOINT: 0.0,
    CART_POLE_JOINT: 0.0,
    POLE_POLE_JOINT: 0.0,
}

TILTED_CONFIG = {
    RAIL_CART_JOINT: 1.0,
    CART_POLE_JOINT: math.pi / 4,  # 45 degrees
    POLE_POLE_JOINT: -math.pi / 6,  # -30 degrees
}

EXTREME_CONFIG = {
    RAIL_CART_JOINT: -1.5,
    CART_POLE_JOINT: math.pi / 2,  # 90 degrees
    POLE_POLE_JOINT: math.pi / 3,  # 60 degrees
}


class ResetJointPositionDemo(PipelineBase):
    def __init__(self, workdir: str, usd_path_or_stage: str | Usd.Stage, logger_level: Logger.Level = Logger.Warn):
        super().__init__(workdir, usd_path_or_stage, logger_level=logger_level, use_warp=True)
        self.current_config = "upright"

    def setup_config(self) -> dict[str, Any]:
        config = Scene.default_config()
        return config

    def setup_contact_tabular(self) -> None:
        pass

    def user_build_scene(self) -> None:
        pass

    def after_world_init(self):
        """Called after world initialization to set up initial state."""
        logger.info("Demo initialized. Use GUI buttons to reset joint positions.")
        logger.info("Watch how the robot immediately jumps to the new configuration.")

    def user_define_gui(self):
        """Add custom GUI controls for resetting joint positions."""
        imgui.Text(f"Current Frame: {self.world.frame()}")
        imgui.Text(f"Current Config: {self.current_config}")
        imgui.Separator()

        imgui.Text("Reset to Preset Configuration:")

        if imgui.Button("Reset to Upright"):
            logger.info("Resetting to upright configuration")
            self.scene.reset_joint_position(
                robot_name="/cartPole",
                joint_positions=UPRIGHT_CONFIG,
                velocities=None,  # Zero velocities
            )
            self.current_config = "upright"

        if imgui.Button("Reset to Tilted (45°/-30°)"):
            logger.info("Resetting to tilted configuration")
            self.scene.reset_joint_position(
                robot_name="/cartPole",
                joint_positions=TILTED_CONFIG,
                velocities=None,
            )
            self.current_config = "tilted"

        if imgui.Button("Reset to Extreme (90°/60°)"):
            logger.info("Resetting to extreme configuration")
            self.scene.reset_joint_position(
                robot_name="/cartPole",
                joint_positions=EXTREME_CONFIG,
                velocities=None,
            )
            self.current_config = "extreme"

        imgui.Separator()
        imgui.Text("Custom Configuration:")

        # Get robot and current joint positions
        robot = self.scene.get_robot("/cartPole")
        cart_pos = float(robot.get_joint_position(RAIL_CART_JOINT))
        pole1_pos = float(robot.get_joint_position(CART_POLE_JOINT))
        pole2_pos = float(robot.get_joint_position(POLE_POLE_JOINT))

        # Display current positions
        imgui.Text(f"Cart: {cart_pos:.3f} m")
        imgui.Text(f"Pole 1: {math.degrees(pole1_pos):.1f}°")
        imgui.Text(f"Pole 2: {math.degrees(pole2_pos):.1f}°")

        # TODO: Add sliders for custom configuration
        # This would allow interactive testing of different joint positions


def main():
    workdir = AssetDir.output_path(__file__)
    demo = ResetJointPositionDemo(
        workdir=workdir, usd_path_or_stage=osp.join(AssetDir.usd_path(), "cartpole.usda"), logger_level=Logger.Info
    )

    logger.info("=" * 60)
    logger.info("Joint Position Reset Demo")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This demo shows how to use scene.reset_joint_position()")
    logger.info("")
    logger.info("Key Differences:")
    logger.info("  • set_joint_position():   Sets control target (smooth motion)")
    logger.info("  • reset_joint_position(): Instant state reset (teleport)")
    logger.info("")
    logger.info("Use Cases for reset_joint_position():")
    logger.info("  1. Initialize robot to specific configuration before simulation")
    logger.info("  2. Reset simulation state without recreating world")
    logger.info("  3. Debug specific joint configurations")
    logger.info("  4. Implement episode resets in RL environments")
    logger.info("")
    logger.info("Try the GUI buttons to reset the cartpole to different configs!")
    logger.info("=" * 60)

    demo.run()


if __name__ == "__main__":
    main()
