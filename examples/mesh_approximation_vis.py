# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
"""Visualize any USD asset with mesh approximation via Polyscope.

Loads a USD file through the standard :class:`PipelineBase` pipeline, which
automatically detects the ``physics:approximation`` attribute on mesh prims
and applies the corresponding approximation.

After the scene is built, the example checks whether each mesh is watertight
(closed), prints approximation details (method, threshold, parts count), and
exports the approximated meshes as OBJ and USD files.

Outputs saved to ``output/examples/mesh_approximation_vis.py/``:
    - ``output.usd``               — simulation output USD (saved on window close)
    - ``approx_<prim_name>.obj``   — approximated mesh in OBJ format
    - ``approx_<prim_name>.usd``   — approximated mesh as a new USD stage

Usage::

    uv run examples/mesh_approximation_vis.py
"""

import os.path as osp
import sys

import numpy as np
import trimesh
from loguru import logger
from pxr import Usd
from uipc import Logger

from u2u import AssetDir
from u2u.pipeline import PipelineBase
from u2u.scene import Scene
from u2u.usd_parser import UsdParserConfig


class MeshApproximationDemo(PipelineBase):
    def __init__(self, workdir: str, usd_path_or_stage: str | Usd.Stage):
        super().__init__(workdir, usd_path_or_stage, logger_level=Logger.Warn)

    def setup_config(self):
        return Scene.default_config()

    def setup_contact_tabular(self):
        pass

    def user_build_scene(self):
        pass

    def setup_usd_parser_config(self):
        return UsdParserConfig(
            root_path="/",
            ignore_paths=[],
            skip_mesh_approximation=False,
            approx_method="convexdecomposition",
        )

    def after_world_init(self):
        self._check_and_export_meshes()

    def _check_and_export_meshes(self):
        """Check watertight status and export approximated meshes for every geometry."""
        logger.info("=" * 60)
        logger.info("Mesh closure check after approximation")
        logger.info("=" * 60)

        for path, geo_info in self.scene.geometry_dict.items():
            geo_slot = geo_info.get("geo_slot")
            if geo_slot is None:
                continue

            geo = geo_slot.geometry()
            positions = np.array(geo.positions().view())[..., 0]  # (N, 3)
            tri_topo = geo.triangles().topo().view()
            triangles = np.array(tri_topo)[..., 0]  # (F, 3)

            if positions.shape[0] == 0 or triangles.shape[0] == 0:
                logger.info(f"  {path:50s}  (no triangle mesh)")
                continue

            tm = trimesh.Trimesh(vertices=positions, faces=triangles, process=False)
            is_closed = bool(tm.is_watertight)
            logger.info(
                f"  {path:50s}  verts={positions.shape[0]:6d}  faces={triangles.shape[0]:6d}  closed={is_closed}"
            )

            # Print approximation details if available
            approx_info = geo_info.get("approx_info")
            if approx_info:
                method = approx_info.get("method", "unknown")
                orig_v = approx_info.get("orig_verts", "?")
                orig_f = approx_info.get("orig_faces", "?")
                new_v = approx_info.get("new_verts", "?")
                new_f = approx_info.get("new_faces", "?")
                logger.info(f"    approx method    : {method}")
                logger.info(f"    before / after   : verts {orig_v} -> {new_v},  faces {orig_f} -> {new_f}")
                if "threshold" in approx_info:
                    logger.info(f"    coacd threshold  : {approx_info['threshold']}")
                if "parts" in approx_info:
                    logger.info(f"    coacd parts      : {approx_info['parts']}")

        logger.info("=" * 60)


DEFAULT_USD = "box_jy_phy.usd"


def main():
    Logger.set_level(Logger.Warn)

    usd_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USD
    # If not an absolute path, look under assets/usd/
    if not osp.isabs(usd_file) and not osp.exists(usd_file):
        usd_file = osp.join(AssetDir.usd_path(), usd_file)

    workdir = AssetDir.output_path(__file__)
    demo = MeshApproximationDemo(workdir=workdir, usd_path_or_stage=usd_file)
    demo.run()


if __name__ == "__main__":
    main()
