"""Unit tests for Articulation joint animation callbacks.

Tests revolute_joint_anim and prismatic_joint_anim by mocking the uipc
geometry objects and patching ``view`` to the identity function, so
numpy arrays backing each attribute are mutated in-place and can be
directly inspected.

Covered scenarios:
- POSITION / VELOCITY / NONE (force) control modes
- Mixed modes across multiple instances
- Joint position limits (upper / lower clamping)
- Prismatic external_force limits (lower / upper clip)
- State tracking: joint_position, joint_velocity, joint_effort
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from uipc import builtin as uipc_builtin

from u2u.scene_builder.articulation import Articulation, JointControlMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DT = 0.01
_JOINT = "joint0"
_FRAME = 5  # > 1 so velocity-update path is exercised

# Patch the ``view`` symbol as imported inside the articulation module.
_PATCH_VIEW = "u2u.scene_builder.articulation.view"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_art(N: int) -> Articulation:
    """Minimal Articulation with one active joint and N instances."""
    robot_prim = MagicMock()  # Usd.Prim not needed for animation callback logic
    art = Articulation(name="robot", robot_prim=robot_prim)
    art.active_joints = [_JOINT]
    art.joint_path_map = {_JOINT: f"/World/robot/{_JOINT}"}
    art.setup_state(N)
    return art


def _revolute_geo(N: int, curr_angles: np.ndarray | None = None):
    """Return (attr_store, geo_mock) for revolute callback tests.

    ``attr_store`` is a plain dict of numpy arrays; mutations to these
    arrays (via ``view(attr)[...] = ...``) are observable after the call.
    """
    if curr_angles is None:
        curr_angles = np.zeros(N, dtype=np.float32)
    attrs = {
        "angle": curr_angles.copy(),
        "external_torque": np.zeros(N, dtype=np.float32),
        "aim_angle": np.zeros(N, dtype=np.float32),
        "driving/is_constrained": np.zeros(N, dtype=np.int8),
        "external_force/is_constrained": np.zeros(N, dtype=np.int8),
    }
    edges = MagicMock()
    edges.find.side_effect = lambda key: attrs[key]
    geo = MagicMock()
    geo.edges.return_value = edges
    return attrs, geo


def _prismatic_geo(N: int, curr_distances: np.ndarray | None = None, init_distance: float = 0.0):
    """Return (attr_store, geo_mock) for prismatic callback tests."""
    if curr_distances is None:
        curr_distances = np.zeros(N, dtype=np.float32)
    attrs = {
        "distance": curr_distances.copy(),
        "init_distance": np.array([init_distance], dtype=np.float32),
        "external_force": np.zeros(N, dtype=np.float32),
        "aim_distance": np.zeros(N, dtype=np.float32),
        "driving/is_constrained": np.zeros(N, dtype=np.int8),
        "external_force/is_constrained": np.zeros(N, dtype=np.int8),
    }
    edges = MagicMock()
    edges.find.side_effect = lambda key: attrs[key]
    geo = MagicMock()
    geo.edges.return_value = edges
    return attrs, geo


def _make_info(geo, frame: int = _FRAME, dt: float = _DT) -> MagicMock:
    joint_obj = MagicMock()
    joint_obj.name.return_value = _JOINT
    geo_slot = MagicMock()
    geo_slot.geometry.return_value = geo
    info = MagicMock()
    info.object.return_value = joint_obj
    info.geo_slots.return_value = [geo_slot]
    info.frame.return_value = frame
    info.dt.return_value = dt
    return info


# ---------------------------------------------------------------------------
# revolute_joint_anim tests
# ---------------------------------------------------------------------------


class TestRevoluteJointAnim:
    def test_position_control_writes_aim_angle(self):
        """POSITION mode: aim_angle equals instruct_position."""
        N = 4
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.POSITION
        art.joint_is_constrained[:, j] = True
        art.joint_instruct_position[:, j] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        attrs, geo = _revolute_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo))

        np.testing.assert_allclose(attrs["aim_angle"], [1.0, 2.0, 3.0, 4.0], rtol=1e-5)

    def test_velocity_control_writes_aim_angle(self):
        """VELOCITY mode: aim_angle = curr_angle + velocity * dt."""
        N = 3
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.VELOCITY
        art.joint_is_constrained[:, j] = True
        art.joint_instruct_velocity[:, j] = np.array([10.0, -5.0, 0.0], dtype=np.float32)

        curr = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        attrs, geo = _revolute_geo(N, curr_angles=curr)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo))

        expected = curr + np.array([10.0, -5.0, 0.0], dtype=np.float32) * _DT
        np.testing.assert_allclose(attrs["aim_angle"], expected, rtol=1e-5)

    def test_torque_control_writes_external_torque(self):
        """NONE+force mode: external_torque written, aim_angle untouched."""
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.NONE
        art.joint_is_force_constrained[:, j] = True
        art.joint_instruct_effort[:, j] = np.array([5.0, 10.0], dtype=np.float32)

        attrs, geo = _revolute_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo))

        np.testing.assert_allclose(attrs["external_torque"], [5.0, 10.0], rtol=1e-5)
        np.testing.assert_allclose(attrs["aim_angle"], [0.0, 0.0], atol=1e-6)

    def test_none_without_force_constraint_does_nothing(self):
        """NONE mode without force constraint: no writes to external_torque or aim_angle."""
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.NONE
        art.joint_is_force_constrained[:, j] = False

        attrs, geo = _revolute_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo))

        np.testing.assert_allclose(attrs["external_torque"], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(attrs["aim_angle"], [0.0, 0.0], atol=1e-6)

    def test_mixed_modes_dispatched_independently(self):
        """POSITION / VELOCITY / NONE+force across 4 instances."""
        N = 4
        art = _make_art(N)
        j = 0
        art.control_mode[0, j] = JointControlMode.POSITION
        art.control_mode[1, j] = JointControlMode.VELOCITY
        art.control_mode[2, j] = JointControlMode.NONE
        art.control_mode[3, j] = JointControlMode.NONE
        art.joint_is_constrained[0, j] = True
        art.joint_is_constrained[1, j] = True
        art.joint_is_force_constrained[2, j] = True
        art.joint_instruct_position[0, j] = 3.0
        art.joint_instruct_velocity[1, j] = 100.0
        art.joint_instruct_effort[2, j] = 7.0

        curr = np.ones(N, dtype=np.float32)
        attrs, geo = _revolute_geo(N, curr_angles=curr)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo))

        assert attrs["aim_angle"][0] == pytest.approx(3.0)
        assert attrs["aim_angle"][1] == pytest.approx(1.0 + 100.0 * _DT)
        assert attrs["aim_angle"][2] == pytest.approx(0.0)  # NONE — untouched
        assert attrs["aim_angle"][3] == pytest.approx(0.0)  # NONE — untouched
        assert attrs["external_torque"][2] == pytest.approx(7.0)
        assert attrs["external_torque"][3] == pytest.approx(0.0)  # no force constraint

    def test_upper_limit_clamped(self):
        """Target angle above upper limit is clamped down."""
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.POSITION
        art.joint_is_constrained[:, j] = True
        art.joint_upper_limits[j] = 1.0
        art.joint_instruct_position[:, j] = np.array([0.5, 5.0], dtype=np.float32)

        attrs, geo = _revolute_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo))

        assert attrs["aim_angle"][0] == pytest.approx(0.5)  # within limit
        assert attrs["aim_angle"][1] == pytest.approx(1.0)  # clamped

    def test_lower_limit_clamped(self):
        """Target angle below lower limit is clamped up."""
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.POSITION
        art.joint_is_constrained[:, j] = True
        art.joint_lower_limits[j] = -1.0
        art.joint_instruct_position[:, j] = np.array([-0.5, -5.0], dtype=np.float32)

        attrs, geo = _revolute_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo))

        assert attrs["aim_angle"][0] == pytest.approx(-0.5)  # within limit
        assert attrs["aim_angle"][1] == pytest.approx(-1.0)  # clamped

    def test_joint_position_state_updated(self):
        """joint_position tracks curr_angles after each callback."""
        N = 3
        art = _make_art(N)
        j = 0
        curr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        attrs, geo = _revolute_geo(N, curr_angles=curr)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo))

        np.testing.assert_allclose(art.joint_position[:, j], curr, rtol=1e-5)

    def test_joint_velocity_computed_on_frame_gt_1(self):
        """joint_velocity = (curr - prev) / dt when frame > 1."""
        N = 2
        art = _make_art(N)
        j = 0
        art.joint_position[:, j] = np.array([0.0, 0.0], dtype=np.float32)
        curr = np.array([0.1, 0.2], dtype=np.float32)
        attrs, geo = _revolute_geo(N, curr_angles=curr)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo, frame=2))

        np.testing.assert_allclose(art.joint_velocity[:, j], curr / _DT, rtol=1e-4)

    def test_joint_velocity_not_updated_on_frame_1(self):
        """joint_velocity unchanged on the very first frame."""
        N = 2
        art = _make_art(N)
        j = 0
        art.joint_velocity[:, j] = np.array([99.0, 99.0], dtype=np.float32)
        attrs, geo = _revolute_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.revolute_joint_anim(_make_info(geo, frame=1))

        np.testing.assert_allclose(art.joint_velocity[:, j], [99.0, 99.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# prismatic_joint_anim tests
# ---------------------------------------------------------------------------


class TestPrismaticJointAnim:
    def test_position_control_writes_aim_distance(self):
        """POSITION mode: aim_distance equals instruct_position."""
        N = 3
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.POSITION
        art.joint_is_constrained[:, j] = True
        art.joint_instruct_position[:, j] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        attrs, geo = _prismatic_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        np.testing.assert_allclose(attrs["aim_distance"], [0.1, 0.2, 0.3], rtol=1e-5)

    def test_velocity_control_writes_aim_distance(self):
        """VELOCITY mode: aim_distance = curr_distance + velocity * dt."""
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.VELOCITY
        art.joint_is_constrained[:, j] = True
        art.joint_instruct_velocity[:, j] = np.array([1.0, -2.0], dtype=np.float32)

        curr = np.array([0.5, 0.5], dtype=np.float32)
        attrs, geo = _prismatic_geo(N, curr_distances=curr)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        expected = curr + np.array([1.0, -2.0], dtype=np.float32) * _DT
        np.testing.assert_allclose(attrs["aim_distance"], expected, rtol=1e-5)

    def test_force_control_writes_external_force(self):
        """NONE+force mode: external_force written, aim_distance untouched."""
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.NONE
        art.joint_is_force_constrained[:, j] = True
        art.joint_instruct_effort[:, j] = np.array([3.0, 6.0], dtype=np.float32)

        attrs, geo = _prismatic_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        np.testing.assert_allclose(attrs["external_force"], [3.0, 6.0], rtol=1e-5)
        np.testing.assert_allclose(attrs["aim_distance"], [0.0, 0.0], atol=1e-6)

    def test_force_clamped_by_lower_limit(self):
        """Force below lower_eff is clipped up to the limit."""
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.NONE
        art.joint_is_force_constrained[:, j] = True
        art.joint_effort_lower_limits[j] = 2.0
        art.joint_instruct_effort[:, j] = np.array([5.0, 1.0], dtype=np.float32)

        attrs, geo = _prismatic_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        assert attrs["external_force"][0] == pytest.approx(5.0)  # above lower → unchanged
        assert attrs["external_force"][1] == pytest.approx(2.0)  # clipped up

    def test_force_clamped_by_upper_limit(self):
        """Force above upper_eff is clipped down to the limit."""
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.NONE
        art.joint_is_force_constrained[:, j] = True
        art.joint_effort_upper_limits[j] = 4.0
        art.joint_instruct_effort[:, j] = np.array([2.0, 10.0], dtype=np.float32)

        attrs, geo = _prismatic_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        assert attrs["external_force"][0] == pytest.approx(2.0)  # below upper → unchanged
        assert attrs["external_force"][1] == pytest.approx(4.0)  # clipped down

    def test_position_limit_accounts_for_init_distance(self):
        """Position limits use combined = target + init_distance.

        init_distance = 0.5, upper = 0.8.
        inst0: target=0.1 → combined=0.6 ≤ 0.8 → unchanged.
        inst1: target=0.4 → combined=0.9 > 0.8 → clamped to 0.8-0.5=0.3.
        """
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.POSITION
        art.joint_is_constrained[:, j] = True
        art.joint_upper_limits[j] = 0.8
        art.joint_instruct_position[:, j] = np.array([0.1, 0.4], dtype=np.float32)

        attrs, geo = _prismatic_geo(N, init_distance=0.5)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        assert attrs["aim_distance"][0] == pytest.approx(0.1)
        assert attrs["aim_distance"][1] == pytest.approx(0.3, abs=1e-5)

    def test_position_lower_limit_accounts_for_init_distance(self):
        """Lower limit uses combined = target + init_distance.

        init_distance = 0.5, lower = 0.6.
        inst0: target=0.2 → combined=0.7 ≥ 0.6 → unchanged.
        inst1: target=0.0 → combined=0.5 < 0.6 → clamped to 0.6-0.5=0.1.
        """
        N = 2
        art = _make_art(N)
        j = 0
        art.control_mode[:, j] = JointControlMode.POSITION
        art.joint_is_constrained[:, j] = True
        art.joint_lower_limits[j] = 0.6
        art.joint_instruct_position[:, j] = np.array([0.2, 0.0], dtype=np.float32)

        attrs, geo = _prismatic_geo(N, init_distance=0.5)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        assert attrs["aim_distance"][0] == pytest.approx(0.2)
        assert attrs["aim_distance"][1] == pytest.approx(0.1, abs=1e-5)

    def test_state_updated_joint_position_and_force(self):
        """joint_position and joint_effort track curr values after callback."""
        N = 3
        art = _make_art(N)
        j = 0
        curr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        curr_eff = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        attrs, geo = _prismatic_geo(N, curr_distances=curr)
        attrs["external_force"][:] = curr_eff
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        np.testing.assert_allclose(art.joint_position[:, j], curr, rtol=1e-5)
        np.testing.assert_allclose(art.joint_effort[:, j], curr_eff, rtol=1e-5)

    def test_mixed_modes_dispatched_independently(self):
        """POSITION / VELOCITY / NONE+force across 3 instances."""
        N = 3
        art = _make_art(N)
        j = 0
        art.control_mode[0, j] = JointControlMode.POSITION
        art.control_mode[1, j] = JointControlMode.VELOCITY
        art.control_mode[2, j] = JointControlMode.NONE
        art.joint_is_constrained[0, j] = True
        art.joint_is_constrained[1, j] = True
        art.joint_is_force_constrained[2, j] = True
        art.joint_instruct_position[0, j] = 0.5
        art.joint_instruct_velocity[1, j] = 10.0
        art.joint_instruct_effort[2, j] = 8.0

        attrs, geo = _prismatic_geo(N)
        with patch(_PATCH_VIEW, new=lambda x: x):
            art.prismatic_joint_anim(_make_info(geo))

        assert attrs["aim_distance"][0] == pytest.approx(0.5)
        assert attrs["aim_distance"][1] == pytest.approx(10.0 * _DT)
        assert attrs["aim_distance"][2] == pytest.approx(0.0)  # NONE — untouched
        assert attrs["external_force"][2] == pytest.approx(8.0)
