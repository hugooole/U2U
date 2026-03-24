# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp

import numpy as np
from loguru import logger
from polyscope import imgui
from pxr import Usd
from uipc import Logger
from uipc import builtin as uipc_builtin

from u2u.pipeline import PipelineBase
from u2u.scene import Scene
from u2u.usd_utils import get_or_create_collision_api, get_or_create_rigid_body_api, read_usd
from u2u.utils import AssetDir


class ABDStateResetDemo(PipelineBase):
    def __init__(self, workdir: str, usd_path_or_stage: str | Usd.Stage, logger_level: Logger.Level = Logger.Warn):
        super().__init__(workdir, usd_path_or_stage, logger_level=logger_level, use_warp=True)

    def setup_config(self):
        config = Scene.default_config()
        config["newton"]["semi_implicit"] = True
        config["contact"]["friction"]["enable"] = False
        return config

    def after_world_init(self):
        self.cube = self.scene.get_geometry("/World/Cube")
        self.cube_initial_transform = self.cube.transforms().view()
        self.cube_initial_velocity_mats = self.cube.instances().find(uipc_builtin.velocity).view()
        self.cube_backend_offset = self.cube.meta().find(uipc_builtin.backend_abd_body_offset).view().astype(np.uint32)

    def user_define_gui(self):
        if imgui.Button("Reset Cube State"):
            self.scene.reset_affine_body_state(
                backend_abd_body_offset_=self.cube_backend_offset,
                abd_body_transforms_=self.cube_initial_transform,
                abd_body_velocity_mats_=self.cube_initial_velocity_mats,
            )


def main():
    Logger.set_level(Logger.Warn)
    workdir = AssetDir.output_path(__file__)
    usd_path = AssetDir.usd_path()
    stage = read_usd(osp.join(usd_path, "HelloWorld.usda"))
    logger.info(f"Loaded USD stage from {usd_path}")

    cubeRigidBodyAPI = get_or_create_rigid_body_api(stage.GetPrimAtPath("/World/Cube"))
    cubeRigidBodyAPI.CreateAngularVelocityAttr().Set(value=(270, 0.0, 0.0))
    get_or_create_collision_api(stage.GetPrimAtPath("/World/Cube"))

    demo = ABDStateResetDemo(workdir, stage)
    demo.run()


if __name__ == "__main__":
    main()
