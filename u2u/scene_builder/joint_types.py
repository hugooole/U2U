# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
from pxr import Usd
from scipy.spatial.transform import Rotation as R
from uipc.geometry import GeometrySlot


def axis_to_index(axis: str) -> int:
    if axis.upper() == "X":
        return 0
    elif axis.upper() == "Y":
        return 1
    elif axis.upper() == "Z":
        return 2
    else:
        raise ValueError(f"axis {axis} is not supported.")


class JointControlMode(IntEnum):
    """Control mode for joints.

    - NONE: No position/velocity control. Use set_joint_effort to apply forces/torques.
    - POSITION: Position control mode (automatically set when calling set_joint_position).
    - VELOCITY: Velocity control mode (automatically set when calling set_joint_velocity).
    """

    NONE = 0
    POSITION = 1
    VELOCITY = 2


@dataclass(frozen=True)
class PhysicsJoint:
    """Represents a physics joint in the USD stage.

    A physics joint connects two rigid bodies and constrains their relative movement.
    """

    # The prim of this joint in usd stage
    prim: Usd.Prim

    # The first link prim of this joint in usd stage
    body0: Optional[Usd.Prim]

    # The second link prim of this joint in usd stage
    body1: Optional[Usd.Prim]

    # Relative position of the joint frame to body0's frame.
    local_pos0: Optional[np.ndarray]

    # Relative orientation of the joint frame to body0's frame.
    local_orient0: Optional[R]

    # Relative position of the joint frame to body1's frame.
    local_pos1: Optional[np.ndarray]

    # Relative orientation of the joint frame to body1's frame.
    local_orient1: Optional[R]


@dataclass(frozen=True)
class RevoluteJoint(PhysicsJoint):
    """Represents a revolute joint in the USD stage.

    A revolute joint allows rotation around a single axis between two rigid bodies.
    It inherits basic joint properties from PhysicsJoint.
    """

    # The axis which this joint rotates about, root joint has no axis
    axis: Optional[str]

    # The lower limit of this joint in degrees
    lower_limit: Optional[float]

    # The upper limit of this joint in degrees
    upper_limit: Optional[float]


@dataclass(frozen=True)
class PrismaticJoint(PhysicsJoint):
    """Represents a prismatic joint in the USD stage.

    A prismatic joint allows linear movement along a single axis between two rigid bodies.
    It inherits basic joint properties from PhysicsJoint.
    """

    # The axis which this joint moves along, root joint has no axis
    axis: Optional[str]

    # The lower limit of this joint in meters
    lower_limit: Optional[float]

    # The upper limit of this joint in meters
    upper_limit: Optional[float]


@dataclass(frozen=True)
class FixedJoint(PhysicsJoint):
    pass


def is_active_joint(joint: PhysicsJoint) -> bool:
    # TODO(zhiguo): support more types joint
    if isinstance(joint, RevoluteJoint):
        return True
    elif isinstance(joint, PrismaticJoint):
        return True
    else:
        return False


@dataclass(frozen=True)
class GeometryInfo:
    prim: Usd.Prim
    geo_slot: GeometrySlot
    type: str
    robot_name: str
