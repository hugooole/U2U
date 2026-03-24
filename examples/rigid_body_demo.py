# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp

from pxr import Usd
from uipc import Logger

from u2u import AssetDir
from u2u.pipeline import PipelineBase
from u2u.scene import Scene


class RigidBodyDemo(PipelineBase):
    def __init__(self, workdir: str, usd_path_or_stage: str | Usd.Stage):
        super().__init__(workdir, usd_path_or_stage)

    def setup_config(self):
        return Scene.default_config()

    def setup_contact_tabular(self):
        pass

    def user_build_scene(self):
        pass


def main():
    Logger.set_level(Logger.Warn)
    workdir = AssetDir.output_path(__file__)
    usd_path = AssetDir.usd_path()
    demo = RigidBodyDemo(workdir, osp.join(usd_path, "AnalyticCone.usda"))
    demo.run()


if __name__ == "__main__":
    main()
