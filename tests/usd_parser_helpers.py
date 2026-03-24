"""Helper functions for UsdParser tests."""

from pathlib import Path

from pxr import Usd

from u2u.scene import Scene
from u2u.usd_parser import UsdParser
from u2u.usd_utils import read_usd

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_path(usd_path: str) -> str:
    """Resolve a USD file path relative to the project root if not absolute."""
    p = Path(usd_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p)


def load_usd_stage(usd_path: str) -> Usd.Stage:
    """Load a USD stage from the given path.

    Args:
        usd_path: USD file path (relative to project root)

    Returns:
        Usd.Stage: loaded USD stage
    """
    return read_usd(_resolve_path(usd_path))


def parse_usd_with_parser(usd_path: str, config: dict | None = None) -> Scene:
    """Parse a USD file using UsdParser.

    Args:
        usd_path: USD file path (relative to project root)
        config: Scene configuration dict (optional)

    Returns:
        Scene: parsed Scene object
    """
    if config is None:
        config = {
            "gravity": [0, 0, -9.81],
            "dt": 0.01,
            "frame_dt": 0.01,
        }

    scene = Scene(config)
    parser = UsdParser(scene, usd_path)
    return parser.parse_and_build_scene()


def validate_scene_object_counts(scene: Scene, expected: dict) -> None:
    """Validate the counts of scene objects against expected values.

    Args:
        scene: parsed Scene object
        expected: expected values dict from YAML, may contain:
            - num_articulations:     exact count of articulations
            - num_rigid_bodies:      exact count of rigid bodies
            - num_colliders:         exact count of colliders
            - num_deformable_bodies: exact count of deformable bodies
            - num_cloth_bodies:      exact count of cloth bodies
    """
    if "num_articulations" in expected:
        actual = len(scene.robot_dict)
        assert actual == expected["num_articulations"], (
            f"Expected {expected['num_articulations']} articulations, got {actual}"
        )

    count_fields = ("num_rigid_bodies", "num_colliders", "num_deformable_bodies", "num_cloth_bodies")
    if any(k in expected for k in count_fields):
        # Exclude robot link geometries from standalone object counts.
        # Using link_geometry keys directly handles cases where ArticulationRootAPI
        # is on a sibling body rather than the common parent Xform (e.g. ant torso),
        # where a path-prefix check would miss sibling links.
        robot_link_keys = {path for robot in scene.robot_dict.values() for path in robot.link_geometry}

        type_counts: dict[str, int] = {}
        for key, geo_info in scene.geometry_dict.items():
            if key in robot_link_keys:
                continue
            t = geo_info.get("type")
            type_counts[t] = type_counts.get(t, 0) + 1

        type_map = {
            "num_rigid_bodies": "rigid_body",
            "num_colliders": "collider",
            "num_deformable_bodies": "deformable_body",
            "num_cloth_bodies": "cloth",
        }

        for field, type_key in type_map.items():
            if field in expected:
                actual = type_counts.get(type_key, 0)
                assert actual == expected[field], (
                    f"Expected {expected[field]} {field.removeprefix('num_')}, got {actual}"
                )

    # Legacy fields
    if "min_robots" in expected:
        actual = len(scene.robot_dict)
        assert actual >= expected["min_robots"], f"Expected at least {expected['min_robots']} robots, got {actual}"

    if expected.get("has_joints"):
        total_joints = sum(len(robot.joint_names) for robot in scene.robot_dict.values())
        assert total_joints > 0, "Expected at least one joint to be present"


def validate_articulation_details(scene: Scene, expected_articulations: list) -> None:
    """Validate articulation details against expected values.

    Args:
        scene: parsed Scene object
        expected_articulations: list of articulation expectations from YAML, each may contain:
            - name:          robot name or "xxxxx" to match the first robot
            - has_joints:    whether the robot has any joints
            - num_joints:    exact joint count
            - joint_names:   list of short joint name suffixes that must appear
            - is_root_fixed: expected value of robot.is_root_fixed
    """
    for art_expected in expected_articulations:
        name = art_expected.get("name", "xxxxx")

        if name == "xxxxx":
            assert len(scene.robot_dict) > 0, "Expected at least one robot in robot_dict"
            robot = next(iter(scene.robot_dict.values()))
        else:
            matched = [r for key, r in scene.robot_dict.items() if name in key]
            assert matched, f"Robot with name '{name}' not found in robot_dict (keys: {list(scene.robot_dict)})"
            robot = matched[0]

        if art_expected.get("has_joints"):
            assert len(robot.joint_names) > 0, "Expected robot to have joints, got none"

        if "num_joints" in art_expected:
            actual = len(robot.joint_names)
            assert actual == art_expected["num_joints"], f"Expected {art_expected['num_joints']} joints, got {actual}"

        if "joint_names" in art_expected:
            for short_name in art_expected["joint_names"]:
                found = any(short_name in path for path in robot.joint_names)
                assert found, f"Joint '{short_name}' not found in robot's joint_names: {robot.joint_names}"

        if "is_root_fixed" in art_expected:
            assert robot.is_root_fixed == art_expected["is_root_fixed"], (
                f"Expected is_root_fixed={art_expected['is_root_fixed']}, got {robot.is_root_fixed}"
            )
