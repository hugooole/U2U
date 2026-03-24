# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from u2u.scene_builder import Articulation


@dataclass
class EnvInfo:
    """Metadata about a single environment instance detected from the USD stage."""

    env_id: int
    env_path: str  # e.g. "/World/envs/env_0"
    transform: np.ndarray  # 4x4 absolute Xform of this env


@dataclass
class Env:
    """Runtime representation of a single environment instance.

    Each Env maps to one instance index in the multi-instance geometry and
    holds per-env robot / geometry references.
    """

    env_id: int
    env_path: str
    instance_id: int  # index into mesh.instances()
    offset_transform: np.ndarray  # 4x4 relative to template env (env_0)
    robot_dict: dict[str, Articulation] = field(default_factory=dict)
    geometry_dict: dict[str, dict] = field(default_factory=dict)
