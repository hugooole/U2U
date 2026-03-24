"""UsdParser test suite.

Tests UsdParser parsing of various USD scenes which include:
- Rigid Body
- Articulation
- Collider
- Deformable Body
- Cloth
"""

from pathlib import Path

import pytest
import yaml
from pxr import UsdGeom

from tests.usd_parser_helpers import (
    load_usd_stage,
    parse_usd_with_parser,
    validate_articulation_details,
    validate_scene_object_counts,
)

_PROJECT_ROOT = Path(__file__).parent.parent


def load_test_cases() -> dict:
    """Load test case configuration file."""
    config_path = Path(__file__).parent / "usd_test_cases.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _skip_if_missing(usd_path: str) -> None:
    """Skip the current test if the USD asset file does not exist."""
    p = Path(usd_path)
    full = p if p.is_absolute() else _PROJECT_ROOT / p
    if not full.exists():
        pytest.skip(f"USD asset not found: {usd_path}")


class TestUsdParserScene:
    """Unified scene parsing tests — parametrized over usd_parser_tests."""

    @pytest.mark.parametrize("test_case", load_test_cases()["usd_parser_tests"])
    def test_object_counts(self, test_case):
        """Test that object counts in the parsed scene match expectations."""
        _skip_if_missing(test_case["file"])
        scene = parse_usd_with_parser(test_case["file"])
        validate_scene_object_counts(scene, test_case["expected"])

    @pytest.mark.parametrize("test_case", load_test_cases()["usd_parser_tests"])
    def test_articulation_details(self, test_case):
        """Test articulation details (skipped when no articulations are expected)."""
        articulations = test_case["expected"].get("articulations")
        if not articulations:
            pytest.skip(f"No articulations field in test case '{test_case['name']}'")

        _skip_if_missing(test_case["file"])
        scene = parse_usd_with_parser(test_case["file"])
        validate_articulation_details(scene, articulations)


class TestUsdParserUtils:
    """Unit conversion, mesh approximation, and utility tests."""

    def test_unit_conversion(self):
        """Test that scene.meters_per_unit matches the USD stage metadata."""
        usd_path = "assets/usd/AnalyticCone.usda"
        stage = load_usd_stage(usd_path)
        scene = parse_usd_with_parser(usd_path)
        assert scene.meters_per_unit == UsdGeom.GetStageMetersPerUnit(stage)

    def test_up_axis_detection(self):
        """Test that scene.up_axis matches the USD stage metadata."""
        usd_path = "assets/usd/AnalyticCone.usda"
        stage = load_usd_stage(usd_path)
        scene = parse_usd_with_parser(usd_path)
        assert scene.up_axis == UsdGeom.GetStageUpAxis(stage)
