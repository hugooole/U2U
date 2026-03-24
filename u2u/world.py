# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from uipc import Engine as UEngine
from uipc import World as UWorld

from u2u.scene import Scene


class World(UWorld):
    def __init__(self, engine: UEngine) -> None:
        super().__init__(engine)
        self._scene: Scene | None = None

    def init(self, scene: Scene) -> None:
        if self._scene is not None:
            raise RuntimeError("World has already been initialized with a scene.")
        if not isinstance(scene, Scene):
            raise TypeError("Expected scene to be an instance of u2u.scene.Scene.")

        super().init(scene)
        self._scene = scene

        # Establish back-reference via weak proxy (no circular reference)
        scene._set_world(self)

    @property
    def scene(self) -> Scene:
        if self._scene is None:
            raise RuntimeError("Scene has not been initialized.")
        return self._scene
