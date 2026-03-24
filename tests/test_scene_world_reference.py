"""Tests for Scene-World bidirectional reference using weak proxy."""

import pytest
from uipc import Engine

from u2u.scene import Scene
from u2u.world import World


@pytest.fixture
def engine(tmp_path):
    """Create a UIPC engine instance."""
    return Engine("cuda", str(tmp_path))


@pytest.fixture
def world(engine):
    """Create a World instance."""
    return World(engine)


@pytest.fixture
def scene():
    """Create a Scene instance with minimal config."""
    config = {
        "gravity": [0, 0, -9.81],
        "dt": 0.01,
        "frame_dt": 0.01,
    }
    return Scene(config)


class TestSceneWorldReference:
    """Test suite for Scene-World bidirectional reference."""

    def test_scene_world_not_initialized(self, scene):
        """Test that accessing scene.world before init raises RuntimeError."""
        with pytest.raises(RuntimeError, match="World has not been initialized"):
            _ = scene.world

    def test_world_scene_not_initialized(self, world):
        """Test that accessing world.scene before init raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Scene has not been initialized"):
            _ = world.scene

    def test_bidirectional_reference_after_init(self, world, scene):
        """Test bidirectional reference works after world.init(scene)."""
        world.init(scene)

        # Test forward reference (World -> Scene)
        assert world.scene is scene

        # Test backward reference (Scene -> World via proxy)
        # Proxy doesn't support 'is' comparison, so verify by accessing methods
        assert scene.world.frame() == world.frame()
        assert scene.world.scene is scene

    def test_world_already_initialized(self, world, scene):
        """Test that calling world.init() twice raises RuntimeError."""
        world.init(scene)

        # Try to initialize again
        config = {"gravity": [0, 0, -9.81], "dt": 0.01, "frame_dt": 0.01}
        scene2 = Scene(config)

        with pytest.raises(RuntimeError, match="World has already been initialized"):
            world.init(scene2)

    def test_scene_world_already_set(self, world, scene, engine):
        """Test that calling scene._set_world() twice raises RuntimeError."""
        world.init(scene)

        # Try to set world again
        world2 = World(engine)
        with pytest.raises(RuntimeError, match="World has already been set"):
            scene._set_world(world2)

    def test_proxy_transparency_frame(self, world, scene):
        """Test that scene.world proxy transparently accesses World methods."""
        world.init(scene)

        # Access frame() method via proxy
        frame_via_world = world.frame()
        frame_via_proxy = scene.world.frame()

        assert frame_via_proxy == frame_via_world

    def test_proxy_transparency_scene_access(self, world, scene):
        """Test that scene.world.scene returns the original scene."""
        world.init(scene)

        # Access scene via world.scene and scene.world.scene
        assert world.scene is scene
        assert scene.world.scene is scene

    def test_world_init_with_invalid_scene(self, world):
        """Test that world.init() with non-Scene object raises TypeError."""
        with pytest.raises(TypeError, match="Expected scene to be an instance"):
            world.init("not a scene")

    def test_multiple_scenes_different_worlds(self, engine):
        """Test that multiple scenes can each have their own world.

        Note: We use the same engine for both worlds since CUDA backend
        doesn't support multiple engine instances.
        """
        config = {"gravity": [0, 0, -9.81], "dt": 0.01, "frame_dt": 0.01}

        scene1 = Scene(config)
        scene2 = Scene(config)

        world1 = World(engine)
        # Note: Can't create second engine with CUDA backend
        # Just test that each scene properly references its world
        world1.init(scene1)

        # Scene1 should reference world1 (verify via method call)
        assert scene1.world.scene is scene1
        assert world1.scene is scene1

        # Scene2 should not have a world yet
        with pytest.raises(RuntimeError, match="World has not been initialized"):
            _ = scene2.world

    def test_proxy_attribute_access(self, world, scene):
        """Test that proxy allows access to World attributes."""
        world.init(scene)

        # Test accessing World's scene attribute via proxy
        assert hasattr(scene.world, "scene")
        assert scene.world.scene is scene

    def test_world_features_access(self, world, scene):
        """Test accessing world.features() via scene.world proxy.

        This is the primary use case: accessing world.features() for GPU state manipulation.
        """
        world.init(scene)

        # Access features() method via proxy
        features_via_world = world.features()
        features_via_proxy = scene.world.features()

        # Both should return the same features object
        assert features_via_proxy is features_via_world


class TestSceneWorldLifecycle:
    """Test lifecycle and edge cases for Scene-World reference."""

    def test_scene_survives_without_world(self, scene):
        """Test that Scene can be used independently without World."""
        # Scene should work on its own
        assert scene.geometry_dict == {}
        assert scene.meters_per_unit == 1.0

    def test_world_reference_after_advance(self, world, scene):
        """Test that scene.world reference persists after world.advance()."""
        world.init(scene)

        # Store initial frame
        initial_frame = scene.world.frame()

        # Advance world (simulates simulation step)
        world.advance()

        # Reference should still be valid (verify by accessing frame)
        assert scene.world.frame() == initial_frame + 1
        assert scene.world.scene is scene

    def test_scene_world_after_multiple_advances(self, world, scene):
        """Test scene.world reference after multiple simulation steps."""
        world.init(scene)

        # Run multiple advances
        for _ in range(10):
            world.advance()

        # Reference should still be valid (verify by checking frame count)
        assert scene.world.frame() == 10
        assert scene.world.scene is scene


class TestSceneWorldIntegration:
    """Integration tests for Scene-World interaction."""

    def test_geometry_access_through_scene(self, world, scene):
        """Test that geometry operations work with scene.world reference."""
        world.init(scene)

        # Scene should be able to access world for advanced operations
        assert scene.world is not None
        assert callable(scene.world.features)

    def test_contact_tabular_with_world_reference(self, world, scene):
        """Test that contact_tabular operations work with world reference."""
        world.init(scene)

        # Scene maintains contact tabular
        contact_tabular = scene.contact_tabular()
        assert contact_tabular is not None

        # Can access world from scene (verify by checking scene access)
        assert scene.world.scene is scene
