# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp

from loguru import logger
from pxr import Usd
from uipc import Logger

from u2u import (
    AssetDir,
    get_or_create_collision_api,
    get_or_create_rigid_body_api,
    read_usd,
)
from u2u.pipeline import PipelineBase
from u2u.scene import Scene


class HelloWorldDemo(PipelineBase):
    def __init__(self, workdir: str, usd_path_or_stage: str | Usd.Stage, logger_level: Logger.Level = Logger.Info):
        super().__init__(workdir, usd_path_or_stage, logger_level=logger_level)

    def setup_config(self):
        return Scene.default_config()


def main():
    Logger.set_level(Logger.Warn)
    workdir = AssetDir.output_path(__file__)
    usd_path = AssetDir.usd_path()
    stage = read_usd(osp.join(usd_path, "HelloWorld.usda"))
    logger.info(f"Loaded USD stage from {usd_path}")

    cubeRigidBodyAPI = get_or_create_rigid_body_api(stage.GetPrimAtPath("/World/Cube"))
    cubeRigidBodyAPI.CreateAngularVelocityAttr().Set(value=(270, 0.0, 0.0))
    get_or_create_collision_api(stage.GetPrimAtPath("/World/Cube"))

    demo = HelloWorldDemo(workdir, stage)
    demo.run()


if __name__ == "__main__":
    main()
