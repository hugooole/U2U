# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from __future__ import annotations

import re

import numpy as np
from loguru import logger
from pxr import Usd

from u2u.env import EnvInfo
from u2u.utils import get_transform


class EnvManager:
    """Detect and manage parallel environment instances in a USD stage.

    Environments are expected to live under a common scope (e.g. ``/World/envs``)
    with names following the pattern ``env_<N>``.  The first environment
    (``env_0``) is treated as the *template*; all others are *clones* whose
    structure is identical but whose root transform differs.
    """

    ENV_PATTERN = re.compile(r"env_(\d+)$")

    def __init__(self, stage: Usd.Stage, env_scope_path: str = "/World/envs"):
        self.stage = stage
        self.env_scope_path = env_scope_path
        self._envs: list[EnvInfo] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_envs(self) -> list[EnvInfo]:
        """Scan *env_scope_path* for ``env_<N>`` children and return sorted list."""
        if self._envs is not None:
            return self._envs

        scope_prim = self.stage.GetPrimAtPath(self.env_scope_path)
        if not scope_prim or not scope_prim.IsValid():
            raise ValueError(f"Env scope prim not found at '{self.env_scope_path}'")

        envs: list[EnvInfo] = []
        for child in scope_prim.GetChildren():
            name = child.GetName()
            m = self.ENV_PATTERN.match(name)
            if m is None:
                continue
            env_id = int(m.group(1))
            env_path = str(child.GetPath())
            transform = get_transform(child)
            envs.append(EnvInfo(env_id=env_id, env_path=env_path, transform=transform))

        envs.sort(key=lambda e: e.env_id)

        if len(envs) == 0:
            raise ValueError(f"No env_N prims found under '{self.env_scope_path}'")

        logger.info(f"Detected {len(envs)} environments under {self.env_scope_path}")
        self._envs = envs
        return envs

    @property
    def template_env(self) -> EnvInfo:
        """The first environment (env_0) used as the structural template."""
        return self.detect_envs()[0]

    @property
    def clone_envs(self) -> list[EnvInfo]:
        """All environments except the template."""
        return self.detect_envs()[1:]

    @property
    def num_envs(self) -> int:
        return len(self.detect_envs())

    def compute_offsets(self) -> list[np.ndarray]:
        """Return per-env 4x4 offset transforms relative to the template (env_0).

        offset[0] is always identity.  For env_i:
            world_transform_i = offset[i] @ world_transform_0
        so  offset[i] = T_i @ inv(T_0).
        """
        envs = self.detect_envs()
        t0_inv = np.linalg.inv(envs[0].transform)
        offsets: list[np.ndarray] = []
        for env in envs:
            offsets.append(env.transform @ t0_inv)
        return offsets

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def remap_path(path: str, source_env_path: str, target_env_path: str) -> str:
        """Replace the env prefix in *path* from source to target.

        >>> EnvManager.remap_path("/World/envs/env_0/Robot", "/World/envs/env_0", "/World/envs/env_5")
        '/World/envs/env_5/Robot'
        """
        if path.startswith(source_env_path):
            return target_env_path + path[len(source_env_path) :]
        return path
