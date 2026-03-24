# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp

from pxr import Usd
from pxr.PhysicsSchema import DeformableBodyAPI
from pxr.UsdPhysics import MassAPI
from uipc import Logger

from u2u import AssetDir, read_usd
from u2u.pipeline import PipelineBase
from u2u.scene import Scene


class DeformableBodyDemo(PipelineBase):
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
    stage = read_usd(osp.join(usd_path, "HelloDeformable.usda"))
    soft_prim = stage.GetPrimAtPath("/World/deformableBody")
    if not soft_prim:
        raise Exception("No cloth found in the USD file.")
    deform_api = DeformableBodyAPI.Apply(soft_prim)
    deform_api.CreatePossionRatioAttr().Set(0.49)
    deform_api.CreateYoungsModulusAttr().Set(10.0)  # Set Young's modulus to 10 kPa
    mass_api = MassAPI(soft_prim)
    mass_api.CreateDensityAttr().Set(3e3)

    demo = DeformableBodyDemo(workdir, stage)
    demo.run()


if __name__ == "__main__":
    main()
