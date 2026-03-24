# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp

from pxr import Usd
from pxr.PhysicsSchema import ClothPhysicsAPI
from pxr.UsdPhysics import MassAPI
from uipc import Logger

from u2u import AssetDir, read_usd
from u2u.pipeline import PipelineBase
from u2u.scene import Scene


class ClothDemo(PipelineBase):
    def __init__(self, workdir: str, usd_path_or_stage: str | Usd.Stage, logger_level: Logger.Level = Logger.Info):
        stage = read_usd(usd_path_or_stage)
        cloth_prim = stage.GetPrimAtPath("/World/Cloth")
        if not cloth_prim:
            raise Exception("No cloth found in the USD file.")
        cloth_api = ClothPhysicsAPI.Apply(cloth_prim)
        cloth_api.CreatePossionRatioAttr().Set(0.499)
        cloth_api.CreateYoungsModulusAttr().Set(10.0)  # Set Young's modulus to 10 kPa
        cloth_api.CreateThicknessAttr().Set(0.001)
        cloth_api.CreateBlendingStiffnessAttr().Set(10.0)
        mass_api = MassAPI(cloth_prim)
        mass_api.CreateDensityAttr().Set(0.7e2)
        super().__init__(workdir, stage, logger_level=logger_level)

    def setup_config(self):
        config = Scene.default_config()
        config["contact"]["enable"] = True
        config["contact"]["friction"]["enable"] = True
        return config

    def setup_contact_tabular(self):
        pass

    def user_build_scene(self):
        pass


def main():
    usd_path = AssetDir.usd_path()
    cloth_demo = ClothDemo(
        workdir=AssetDir.output_path(__file__),
        usd_path_or_stage=osp.join(usd_path, "HelloCloth.usda"),
    )
    cloth_demo.run()


if __name__ == "__main__":
    main()
