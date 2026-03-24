# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp
import sys

from loguru import logger
from uipc import Logger

from u2u.pipeline import PipelineBase
from u2u.scene import Scene
from u2u.utils import AssetDir


class MultiInstancesPipeline(PipelineBase):
    def __init__(self, workdir: str, usd_path: str):
        super().__init__(workdir, usd_path, logger_level=Logger.Error)

    def setup_config(self):
        config = Scene.default_config()
        config["sanity_check"] = False
        config["contact"]["enable"] = False
        config["contact"]["d_hat"] = 0.001
        config["newton"]["semi_implicit"] = True
        config["collision_detection"]["method"] = "info_stackless_bvh"
        return config

    def setup_usd_parser_config(self):
        return {
            "root_path": "/",
            "ignore_paths": [],
            "skip_mesh_approximation": False,
            "approx_method": "convexdecomposition",
            "multi_env": True,
            "env_scope_path": "/World/envs",
        }


def main():
    logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
    usd_path = osp.join(AssetDir.usd_path(), "cartpole_128.usda")
    workdir = AssetDir.output_path(__file__)
    pipeline = MultiInstancesPipeline(workdir, usd_path)
    print(f"Number of environments: {pipeline.num_envs}")
    print(f"Number of robots: {len(pipeline.scene.robot_dict)}")
    pipeline.screenshot("/tmp/sim_screenshot.png")
    pipeline.run()


if __name__ == "__main__":
    main()
