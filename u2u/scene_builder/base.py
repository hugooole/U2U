# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from u2u.scene import Scene


class SceneBuilderBase(ABC):
    """
    Base class for building scenes in a simulation environment.

    Provides an abstract interface to parse a USD stage and construct the necessary
    simulation objects using the parsed scene data. This class serves as a template
    for implementing specialized scene builders.

    :ivar scene: The simulation scene to be used by the builder.
    :type scene: Scene
    """

    def __init__(self, scene: "Scene"):
        """Initialize the simulator.

        Args:
            scene: The simulation scene
        """
        self.scene = scene
        self.metersPerUnit = self.scene.meters_per_unit
        self._geometry = {}

    @property
    def animator(self):
        return self.scene.animator()

    @abstractmethod
    def build(self) -> None:
        """Build the simulation objects."""
        pass

    def get_geometry(self) -> Dict[str, Any]:
        """Get the geometry dictionary.

        Returns:
            The geometry dictionary
        """
        return self._geometry
