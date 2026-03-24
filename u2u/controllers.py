# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
"""
Controllers Module

Provides feedback control implementations for robotics applications, supporting both
linear (force) and angular (torque) control. All controllers inherit from ControllerBase
and implement a unified compute() interface.

Typical Usage:
    >>> # For linear control (prismatic joint)
    >>> pd_linear = PDController(kp=250.0, kd=146.0, force_or_torque_limit=500.0)
    >>> force = pd_linear.compute(target=0.0, position=cart_pos, velocity=cart_vel)
    >>>
    >>> # For angular control (revolute joint)
    >>> pd_angular = PDController(kp=800.0, kd=2.19, force_or_torque_limit=500.0)
    >>> torque = pd_angular.compute(target=0.0, position=joint_angle, velocity=joint_vel)

PD Parameter Tuning:
    For optimal performance, tune Kp and Kd based on system dynamics:

    Linear systems (mass m):
        - Natural frequency: ω_n = √(Kp/m)
        - Critical damping: Kd_crit = 2√(Kp × m)
        - Target damping ratio: ζ = Kd / Kd_crit = 1.0-1.2 (slightly overdamped)

    Rotational systems (inertia I):
        - Natural frequency: ω_n = √(Kp/I)
        - Critical damping: Kd_crit = 2√(Kp × I)
        - Target damping ratio: ζ = Kd / Kd_crit = 1.0-1.2 (slightly overdamped)

See Also:
    - examples/cartpole.py: Linear PD control example
    - examples/cartpole_revolute.py: Angular PD control example
    - examples/cartpole_analysis.py: Parameter tuning theory
"""

from abc import abstractmethod


class ControllerBase:
    """
    Abstract base class for all feedback controllers.

    Provides a unified interface for force/torque control of robotic systems.
    All derived controllers must implement the compute() method.

    Args:
        force_or_torque_limit: Optional output saturation limit.
            - For linear control: force limit in Newtons (N)
            - For angular control: torque limit in Newton-meters (N·m)
            - If None, no limiting is applied

    Attributes:
        force_or_torque_limit (float | None): The configured output limit
    """

    def __init__(self, force_or_torque_limit: float | None = None):
        self.force_or_torque_limit = force_or_torque_limit

    @abstractmethod
    def compute(self, target: float, position: float, velocity: float) -> float:
        """
        Compute control output based on current state.

        Args:
            target: Desired position/angle
            position: Current position/angle
            velocity: Current velocity/angular velocity

        Returns:
            Control force (N) or torque (N·m)
        """
        pass


class PDController(ControllerBase):
    """
    Proportional-Derivative (PD) controller for position/angle tracking.

    Implements the control law:
        u = Kp × (target - position) - Kd × velocity

    The PD controller provides:
    - Proportional term (Kp): Generates force/torque proportional to position error
    - Derivative term (Kd): Provides damping by opposing velocity (stabilization)

    PD controllers are sufficient for most robotics applications and avoid the
    integral windup issues of PID controllers when properly tuned.

    Args:
        kp: Proportional gain (stiffness)
            - Linear: N/m (Newtons per meter)
            - Angular: N·m/rad (Newton-meters per radian)
        kd: Derivative gain (damping)
            - Linear: N·s/m (Newton-seconds per meter)
            - Angular: N·m·s/rad (Newton-meter-seconds per radian)
        force_or_torque_limit: Optional output saturation limit in N or N·m

    Attributes:
        kp (float): Proportional gain
        kd (float): Derivative gain

    Example:
        >>> # Control a 14.8 kg cart with 500N force limit
        >>> mass = 14.8
        >>> kp = 250.0  # Stiffness
        >>> zeta = 1.2  # Target damping ratio (slightly overdamped)
        >>> kd = zeta * 2 * (kp * mass) ** 0.5  # Calculate Kd = 146
        >>>
        >>> controller = PDController(kp=250.0, kd=146.0, force_or_torque_limit=500.0)
        >>> force = controller.compute(target=0.0, position=2.0, velocity=-0.5)
        >>> print(f"Control force: {force:.1f} N")
        Control force: -500.0 N

    Notes:
        - For optimal performance, tune Kd based on system mass/inertia
        - Recommended damping ratio ζ = 1.0-1.2 (slightly overdamped)
        - Underdamped (ζ < 1.0) causes oscillations
        - Overdamped (ζ > 2.0) results in slow response
    """

    def __init__(self, kp: float, kd: float, force_or_torque_limit: float | None = None):
        super().__init__(force_or_torque_limit)
        self.kp = kp
        self.kd = kd

    def compute(self, target: float, position: float, velocity: float) -> float:
        """
        Compute PD control output.

        Args:
            target: Desired position (m) or angle (rad)
            position: Current position (m) or angle (rad)
            velocity: Current velocity (m/s) or angular velocity (rad/s)

        Returns:
            Control force (N) or torque (N·m), saturated if limit is set

        Example:
            >>> pd = PDController(kp=200.0, kd=50.0, force_or_torque_limit=500.0)
            >>> force = pd.compute(target=0.0, position=1.5, velocity=0.2)
            >>> # Returns: 200.0 * (0 - 1.5) - 50.0 * 0.2 = -310.0 N
        """
        u = self.kp * (target - position) - self.kd * velocity
        if self.force_or_torque_limit is not None:
            u = max(-self.force_or_torque_limit, min(self.force_or_torque_limit, u))
        return u


class PIDController(ControllerBase):
    """
    Proportional-Integral-Derivative (PID) controller for position/angle tracking.

    Implements the control law:
        u = Kp × (target - position) + Ki × ∫error dt - Kd × velocity

    The PID controller provides:
    - Proportional term (Kp): Force/torque proportional to position error
    - Integral term (Ki): Eliminates steady-state error by accumulating past errors
    - Derivative term (Kd): Damping by opposing velocity (stabilization)

    Features anti-windup protection to prevent integral term saturation when
    output is limited.

    Args:
        kp: Proportional gain (stiffness)
            - Linear: N/m (Newtons per meter)
            - Angular: N·m/rad (Newton-meters per radian)
        ki: Integral gain (steady-state error elimination)
            - Linear: N/(m·s) (Newtons per meter-second)
            - Angular: N·m/(rad·s) (Newton-meters per radian-second)
        kd: Derivative gain (damping)
            - Linear: N·s/m (Newton-seconds per meter)
            - Angular: N·m·s/rad (Newton-meter-seconds per radian)
        dt: Time step for integration in seconds (e.g., 0.01 for 100 Hz)
        force_or_torque_limit: Optional output saturation limit in N or N·m

    Attributes:
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain
        dt (float): Integration time step
        integral (float): Accumulated integral term

    Example:
        >>> # Create PID controller for 100 Hz control loop
        >>> pid = PIDController(
        ...     kp=250.0,
        ...     ki=10.0,
        ...     kd=146.0,
        ...     dt=0.01,
        ...     force_or_torque_limit=500.0
        ... )
        >>>
        >>> # Control loop
        >>> for step in range(1000):
        ...     force = pid.compute(target=0.0, position=cart_pos, velocity=cart_vel)
        ...     # Apply force to system...
        >>>
        >>> # Reset between tasks to clear integral
        >>> pid.reset()

    Notes:
        - Use PID when steady-state error must be zero (e.g., against gravity)
        - PD is often sufficient for most robotics applications
        - Integral term can cause overshoot if Ki is too large
        - Always call reset() between different control tasks
        - Anti-windup prevents integral from growing when saturated
    """

    def __init__(self, kp: float, ki: float, kd: float, dt: float, force_or_torque_limit: float | None = None):
        super().__init__(force_or_torque_limit)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0

    def compute(self, target: float, position: float, velocity: float) -> float:
        """
        Compute PID control output with anti-windup protection.

        The integral term only accumulates when the controller output is not
        saturated, preventing integral windup.

        Args:
            target: Desired position (m) or angle (rad)
            position: Current position (m) or angle (rad)
            velocity: Current velocity (m/s) or angular velocity (rad/s)

        Returns:
            Control force (N) or torque (N·m), saturated if limit is set

        Example:
            >>> pid = PIDController(kp=200.0, ki=5.0, kd=50.0, dt=0.01, force_or_torque_limit=500.0)
            >>> # First call with error = -1.5 m, velocity = 0.2 m/s
            >>> force = pid.compute(target=0.0, position=1.5, velocity=0.2)
            >>> # Returns: 200.0 * (-1.5) + 5.0 * 0.0 - 50.0 * 0.2 = -310.0 N
            >>> # Integral accumulates: integral = -1.5 * 0.01 = -0.015
        """
        error = target - position

        # Compute raw control output
        u_raw = self.kp * error + self.ki * self.integral - self.kd * velocity

        # Apply limits and anti-windup
        if self.force_or_torque_limit is not None:
            u = max(-self.force_or_torque_limit, min(self.force_or_torque_limit, u_raw))
            # Only integrate if not saturated (anti-windup)
            if abs(u_raw) <= self.force_or_torque_limit:
                self.integral += error * self.dt
        else:
            u = u_raw
            self.integral += error * self.dt

        return u

    def reset(self):
        """
        Reset integral term to zero.

        Call this method:
        - Between different control tasks to prevent integral carryover
        - When switching control targets
        - After long periods without control updates

        Example:
            >>> pid = PIDController(kp=200.0, ki=5.0, kd=50.0, dt=0.01)
            >>> # Task 1: Move to position 1.0
            >>> for _ in range(100):
            ...     force = pid.compute(target=1.0, position=pos, velocity=vel)
            ...     # Apply force...
            >>>
            >>> pid.reset()  # Clear integral before new task
            >>>
            >>> # Task 2: Move to position 2.0
            >>> for _ in range(100):
            ...     force = pid.compute(target=2.0, position=pos, velocity=vel)
            ...     # Apply force...
        """
        self.integral = 0.0
