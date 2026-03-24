# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from __future__ import annotations

import numpy as np
import doctest
from typing import Tuple, Optional


class Pose:
    def __init__(self, p: Optional[np.ndarray] = None, q: Optional[np.ndarray] = None) -> None:
        """
        Initialize a Pose with position and quaternion orientation.

        Args:
            p: Position vector [x, y, z]. Defaults to [0, 0, 0].
            q: Quaternion [w, x, y, z]. Defaults to [1, 0, 0, 0] (identity).

        Examples:
            >>> import numpy as np
            >>> # Default initialization
            >>> pose = Pose()
            >>> np.allclose(pose.p, np.array([[0.0], [0.0], [0.0]]))
            True
            >>> np.allclose(pose.q, np.array([[1.0], [0.0], [0.0], [0.0]]))
            True

            >>> # Custom initialization
            >>> p = np.array([1.0, 2.0, 3.0])
            >>> q = np.array([0.5, 0.5, 0.5, 0.5])
            >>> pose = Pose(p, q)
            >>> np.allclose(pose.p, np.array([[1.0], [2.0], [3.0]]))
            True
            >>> # Quaternion should be normalized
            >>> q_norm = q / np.linalg.norm(q)
            >>> np.allclose(pose.q, q_norm.reshape(4, 1))
            True
        """
        if p is None:
            self._p = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1)
        else:
            self._p = np.asarray(p, dtype=np.float32).reshape(3, 1)

        if q is None:
            self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        else:
            self._q = np.asarray(q, dtype=np.float32).reshape(4, 1)
            # Normalize quaternion
            self._q = self._q / np.linalg.norm(self._q)

    def __getstate__(self) -> Tuple:
        """
        Get the state of the Pose for serialization.

        Returns:
            Tuple containing position and quaternion arrays.
        """
        return (self._p, self._q)

    def __setstate__(self, state: Tuple) -> None:
        """
        Set the state of the Pose from deserialization.

        Args:
            state: Tuple containing position and quaternion arrays.
        """
        self._p, self._q = state

    def __mul__(self, other: Pose) -> Pose:
        """
        Multiply this pose with another pose (composition of transformations).

        Args:
            other: Another Pose to multiply with.

        Returns:
            A new Pose representing the combined transformation.
        """
        return self.transform(other)

    def __repr__(self) -> str:
        """
        String representation of the Pose.

        Returns:
            String representation with position and quaternion.

        Examples:
            >>> import numpy as np
            >>> p = np.array([1.0, 2.0, 3.0])
            >>> q = np.array([1.0, 0.0, 0.0, 0.0])
            >>> pose = Pose(p, q)
            >>> repr(pose)
            'Pose(p=[1. 2. 3.], q=[1. 0. 0. 0.])'
        """
        return f"Pose(p={self._p.flatten()}, q={self._q.flatten()})"

    @staticmethod
    def from_transformation_matrix(mat44: np.ndarray) -> Pose:
        """
        Create a Pose from a 4x4 transformation matrix.

        Args:
            mat44: 4x4 transformation matrix.

        Returns:
            A new Pose object.

        Examples:
            >>> import numpy as np
            >>> # Create a transformation matrix with translation [1,2,3] and identity rotation
            >>> mat = np.eye(4, dtype=np.float32)
            >>> mat[:3, 3] = [1.0, 2.0, 3.0]
            >>> pose = Pose.from_transformation_matrix(mat)
            >>> np.allclose(pose.p, np.array([[1.0], [2.0], [3.0]]))
            True
            >>> np.allclose(pose.q, np.array([[1.0], [0.0], [0.0], [0.0]]))
            True
        """
        # Extract position from the translation part of the matrix
        p = mat44[:3, 3]

        # Extract rotation matrix
        rot_matrix = mat44[:3, :3]

        # Convert rotation matrix to quaternion
        # Using the algorithm from: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        trace = np.trace(rot_matrix)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
            y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
            z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
        elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            z = 0.25 * s

        q = np.array([w, x, y, z], dtype=np.float32)

        return Pose(p, q)

    def inv(self) -> Pose:
        """
        Compute the inverse of this pose.

        Returns:
            A new Pose representing the inverse transformation.

        Examples:
            >>> import numpy as np
            >>> # Create a pose with position [1,2,3] and identity quaternion
            >>> p = np.array([1.0, 2.0, 3.0])
            >>> q = np.array([1.0, 0.0, 0.0, 0.0])
            >>> pose = Pose(p, q)
            >>> inv_pose = pose.inv()
            >>> # For identity quaternion, the inverse position should be negated
            >>> np.allclose(inv_pose.p, np.array([[-1.0], [-2.0], [-3.0]]))
            True
            >>> np.allclose(inv_pose.q, np.array([[1.0], [0.0], [0.0], [0.0]]))
            True

            >>> # Multiplying a pose with its inverse should give identity
            >>> identity = pose * inv_pose
            >>> np.allclose(identity.p, np.array([[0.0], [0.0], [0.0]]))
            True
            >>> np.allclose(identity.q, np.array([[1.0], [0.0], [0.0], [0.0]]))
            True
        """
        # Conjugate of quaternion (inverse for unit quaternions)
        q_inv = self._q.copy()
        q_inv[1:] = -q_inv[1:]

        # Rotate the negative of the translation by the inverse quaternion
        p_inv = self._quaternion_rotate(q_inv, -self._p)

        return Pose(p_inv, q_inv)

    def set_p(self, p: np.ndarray) -> None:
        """
        Set the position vector.

        Args:
            p: New position vector.
        """
        self._p = np.asarray(p, dtype=np.float32).reshape(3, 1)

    def set_q(self, q: np.ndarray) -> None:
        """
        Set the quaternion.

        Args:
            q: New quaternion.
        """
        self._q = np.asarray(q, dtype=np.float32).reshape(4, 1)
        # Normalize quaternion
        self._q = self._q / np.linalg.norm(self._q)

    def set_rotation(self, rotation_matrix: np.ndarray) -> None:
        """
        Set the rotation from a 3x3 rotation matrix.

        Args:
            rotation_matrix: 3x3 rotation matrix.
        """
        # Create a 4x4 transformation matrix with the given rotation
        mat44 = np.eye(4, dtype=np.float32)
        mat44[:3, :3] = rotation_matrix
        mat44[:3, 3] = self._p.flatten()

        # Extract the quaternion from the matrix
        pose = Pose.from_transformation_matrix(mat44)
        self._q = pose._q

    def to_transformation_matrix(self) -> np.ndarray:
        """
        Convert the pose to a 4x4 transformation matrix.

        Returns:
            4x4 transformation matrix.

        Examples:
            >>> import numpy as np
            >>> # Create a pose with position [1,2,3] and identity quaternion
            >>> p = np.array([1.0, 2.0, 3.0])
            >>> q = np.array([1.0, 0.0, 0.0, 0.0])
            >>> pose = Pose(p, q)
            >>> mat = pose.to_transformation_matrix()
            >>> # Check that the rotation part is identity
            >>> np.allclose(mat[:3, :3], np.eye(3))
            True
            >>> # Check that the translation part matches the position
            >>> np.allclose(mat[:3, 3], p)
            True

            >>> # Test round-trip conversion
            >>> pose2 = Pose.from_transformation_matrix(mat)
            >>> np.allclose(pose.p, pose2.p)
            True
            >>> np.allclose(pose.q, pose2.q)
            True
        """
        # Extract quaternion components
        w, x, y, z = self._q.flatten()

        # Compute rotation matrix from quaternion
        xx, xy, xz = x * x, x * y, x * z
        yy, yz, zz = y * y, y * z, z * z
        wx, wy, wz = w * x, w * y, w * z

        rot_matrix = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float32,
        )

        # Create transformation matrix
        mat44 = np.eye(4, dtype=np.float32)
        mat44[:3, :3] = rot_matrix
        mat44[:3, 3] = self._p.flatten()

        return mat44

    def transform(self, other: Pose) -> Pose:
        """
        Apply this pose as a transformation to another pose.

        Args:
            other: The pose to transform.

        Returns:
            A new Pose representing the transformed pose.

        Examples:
            >>> import numpy as np
            >>> # Create two poses
            >>> p1 = np.array([1.0, 0.0, 0.0])
            >>> q1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
            >>> pose1 = Pose(p1, q1)
            >>>
            >>> p2 = np.array([0.0, 1.0, 0.0])
            >>> q2 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
            >>> pose2 = Pose(p2, q2)
            >>>
            >>> # Transform pose2 by pose1
            >>> result = pose1.transform(pose2)
            >>> # For identity quaternions, the result should be a simple addition of positions
            >>> np.allclose(result.p, np.array([[1.0], [1.0], [0.0]]))
            True
            >>> np.allclose(result.q, np.array([[1.0], [0.0], [0.0], [0.0]]))
            True

            >>> # Using the multiplication operator
            >>> result = pose1 * pose2
            >>> np.allclose(result.p, np.array([[1.0], [1.0], [0.0]]))
            True
        """
        # Combine rotations by quaternion multiplication
        q_result = self._quaternion_multiply(self._q, other._q)

        # Transform the position: p_result = p1 + q1 * p2 * q1^-1
        p_rotated = self._quaternion_rotate(self._q, other._p)
        p_result = self._p + p_rotated

        return Pose(p_result, q_result)

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.

        Args:
            q1: First quaternion [w, x, y, z].
            q2: Second quaternion [w, x, y, z].

        Returns:
            Result quaternion [w, x, y, z].
        """
        w1, x1, y1, z1 = q1.flatten()
        w2, x2, y2, z2 = q2.flatten()

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z], dtype=np.float32).reshape(4, 1)

    def _quaternion_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by a quaternion: q * v * q^-1

        Args:
            q: Quaternion [w, x, y, z].
            v: Vector [x, y, z].

        Returns:
            Rotated vector [x, y, z].
        """
        # Convert vector to pure quaternion
        v_quat = np.array([0, v[0, 0], v[1, 0], v[2, 0]], dtype=np.float32).reshape(4, 1)

        # Conjugate of quaternion (inverse for unit quaternions)
        q_conj = q.copy()
        q_conj[1:] = -q_conj[1:]

        # Perform rotation: q * v * q^-1
        rotated_quat = self._quaternion_multiply(self._quaternion_multiply(q, v_quat), q_conj)

        # Extract vector part
        return rotated_quat[1:].reshape(3, 1)

    @property
    def p(self) -> np.ndarray:
        """
        Get the position vector.

        Returns:
            Position vector [x, y, z].
        """
        return self._p

    @property
    def q(self) -> np.ndarray:
        """
        Get the quaternion.

        Returns:
            Quaternion [w, x, y, z].

        Examples:
            >>> import numpy as np
            >>> q = np.array([0.5, 0.5, 0.5, 0.5])
            >>> pose = Pose(q=q)
            >>> # Quaternion should be normalized
            >>> q_norm = q / np.linalg.norm(q)
            >>> np.allclose(pose.q, q_norm.reshape(4, 1))
            True
        """
        return self._q


if __name__ == "__main__":
    # Run doctests
    doctest.testmod(verbose=True)
