# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
"""
PD Controller Parameter Analysis Tool

Theoretical analysis tool for PD controller parameter tuning without running simulation.
Supports both linear (mass-based) and rotational (inertia-based) systems.

Usage:
    # Analyze linear system (e.g., cart on rail)
    uv run examples/cartpole_analysis.py --mass 14.8 --limit 500 --error 2.0 --type linear

    # Analyze rotational system (e.g., pendulum)
    uv run examples/cartpole_analysis.py --inertia 0.00104 --limit 500 --error 0.5 --type rotational

    # Auto-generate optimal parameters
    uv run examples/cartpole_analysis.py --mass 14.8 --limit 500 --error 2.0 --auto
"""

import argparse
import math
from typing import Literal


class PDAnalyzer:
    """
    PD Controller Parameter Analyzer

    Supports both linear and rotational systems:
    - Linear: Uses mass (kg) and force limit (N)
    - Rotational: Uses inertia (kg·m²) and torque limit (N·m)
    """

    def __init__(
        self,
        mass_or_inertia: float,
        force_or_torque_limit: float,
        dt: float = 0.01,
        system_type: Literal["linear", "rotational"] = "linear",
    ):
        """
        Args:
            mass_or_inertia: Mass (kg) for linear system or inertia (kg·m²) for rotational
            force_or_torque_limit: Maximum force (N) or torque (N·m)
            dt: Time step in seconds
            system_type: "linear" for mass-based or "rotational" for inertia-based
        """
        self.mass_or_inertia = mass_or_inertia
        self.force_or_torque_limit = force_or_torque_limit
        self.dt = dt
        self.system_type = system_type

    def analyze(self, kp: float, kd: float, name: str = "") -> dict:
        """
        Analyze PD parameters for given system.

        Args:
            kp: Proportional gain (N/m for linear, N·m/rad for rotational)
            kd: Derivative gain (N·s/m for linear, N·m·s/rad for rotational)
            name: Configuration name for display

        Returns:
            Dictionary with analysis results
        """
        omega_n = math.sqrt(kp / self.mass_or_inertia)
        kd_crit = 2 * math.sqrt(kp * self.mass_or_inertia)
        zeta = kd / kd_crit

        # 估算收敛时间（基于 2% 误差准则）
        if zeta < 1:
            settling_time = 4 / (zeta * omega_n)
            behavior = "欠阻尼(有振荡)"
            has_overshoot = True
        elif zeta == 1:
            settling_time = 4 / omega_n
            behavior = "临界阻尼(最优)"
            has_overshoot = False
        else:
            settling_time = 4 / (zeta * omega_n) * (zeta + math.sqrt(zeta**2 - 1))
            if zeta < 1.5:
                behavior = "轻微过阻尼(实用最优)"
            elif zeta < 3:
                behavior = "过阻尼(较慢)"
            else:
                behavior = "严重过阻尼(极慢)"
            has_overshoot = False

        settling_frames = int(settling_time / self.dt)
        max_error = 2.0
        max_force_estimate = kp * max_error

        return {
            "name": name,
            "kp": kp,
            "kd": kd,
            "omega_n": omega_n,
            "kd_crit": kd_crit,
            "zeta": zeta,
            "settling_time_sec": settling_time,
            "settling_frames": settling_frames,
            "behavior": behavior,
            "has_overshoot": has_overshoot,
            "max_force_estimate": max_force_estimate,
        }

    def print_analysis(self, result: dict) -> None:
        """Print analysis results"""
        units = self._get_units()

        print(f"\n{'=' * 70}")
        print(f"📊 {result['name']}")
        print(f"{'=' * 70}")
        print("Parameters:")
        print(f"  Kp = {result['kp']:.1f} {units['kp']}")
        print(f"  Kd = {result['kd']:.1f} {units['kd']}")
        print("\nSystem Characteristics:")
        print(f"  Natural frequency ω_n = {result['omega_n']:.2f} rad/s")
        print(f"  Critical damping Kd_crit = {result['kd_crit']:.2f} {units['kd']}")
        print(f"  Damping ratio ζ = {result['zeta']:.2f}")
        print(f"  Behavior: {result['behavior']}")
        print("\nPerformance Prediction:")
        print(f"  Settling time: {result['settling_time_sec']:.3f} s (~{result['settling_frames']} frames)")
        print(f"  Overshoot: {'Yes ⚠️' if result['has_overshoot'] else 'No ✅'}")
        print(f"  Max output estimate: {result['max_force_estimate']:.1f} {units['output']}")

    def _get_units(self) -> dict:
        """Get units based on system type"""
        if self.system_type == "linear":
            return {"kp": "N/m", "kd": "N·s/m", "output": "N", "state": "m"}
        else:
            return {"kp": "N·m/rad", "kd": "N·m·s/rad", "output": "N·m", "state": "rad"}

    def compare(self, configs: list[tuple[float, float, str]]) -> None:
        """对比多个配置"""
        results = []
        for kp, kd, name in configs:
            result = self.analyze(kp, kd, name)
            results.append(result)
            self.print_analysis(result)

        # 打印对比总结
        print(f"\n{'=' * 70}")
        print("📈 性能对比总结")
        print(f"{'=' * 70}")
        print(f"{'配置':<20} {'ζ':<8} {'收敛帧数':<12} {'收敛时间':<12} {'行为'}")
        print(f"{'-' * 70}")
        for r in results:
            print(
                f"{r['name']:<20} {r['zeta']:<8.2f} {r['settling_frames']:<12} "
                f"{r['settling_time_sec']:<12.3f} {r['behavior']}"
            )

        # 计算改进
        if len(results) >= 2:
            base = results[0]
            for r in results[1:]:
                if base["settling_frames"] > 0:
                    improvement = (base["settling_frames"] - r["settling_frames"]) / base["settling_frames"] * 100
                    print(f"\n{r['name']} 相比 {base['name']}:")
                    print(f"  收敛速度提升: {improvement:+.1f}%")
                    print(f"  收敛时间: {base['settling_frames']} 帧 → {r['settling_frames']} 帧")
                else:
                    print(f"\n{r['name']} 相比 {base['name']}:")
                    print(f"  收敛时间: {base['settling_time_sec']:.4f}s → {r['settling_time_sec']:.4f}s")
                    print("  (Both converge in <1 frame)")


def generate_optimal_configs(
    mass_or_inertia: float, limit: float, max_error: float, system_type: str
) -> list[tuple[float, float, str]]:
    """
    Generate optimal PD configurations automatically.

    Args:
        mass_or_inertia: System mass or inertia
        limit: Force or torque limit
        max_error: Maximum expected error

    Returns:
        List of (kp, kd, name) tuples
    """
    # Calculate maximum Kp based on force/torque limit
    kp_max = limit / max_error

    configs = []
    zeta_target = 1.2  # Optimal damping ratio

    # Aggressive: Use maximum Kp
    kp_aggressive = kp_max
    kd_aggressive = zeta_target * 2 * math.sqrt(kp_aggressive * mass_or_inertia)
    configs.append((kp_aggressive, kd_aggressive, "Aggressive (max Kp)"))

    # Balanced: 80% of max Kp
    kp_balanced = 0.8 * kp_max
    kd_balanced = zeta_target * 2 * math.sqrt(kp_balanced * mass_or_inertia)
    configs.append((kp_balanced, kd_balanced, "Balanced (80% Kp)"))

    # Conservative: 60% of max Kp
    kp_conservative = 0.6 * kp_max
    kd_conservative = zeta_target * 2 * math.sqrt(kp_conservative * mass_or_inertia)
    configs.append((kp_conservative, kd_conservative, "Conservative (60% Kp)"))

    return configs


def print_tuning_guide(mass_or_inertia: float, limit: float, max_error: float, system_type: str) -> None:
    """Print parameter tuning guidelines"""
    units = "N/m, N·s/m" if system_type == "linear" else "N·m/rad, N·m·s/rad"
    param = "mass" if system_type == "linear" else "inertia"
    output = "force" if system_type == "linear" else "torque"

    print(f"\n{'=' * 70}")
    print("💡 Parameter Tuning Guide")
    print(f"{'=' * 70}")
    print("\nOptimal Damping Ratio Selection:")
    print("  ζ < 1.0      : Underdamped, oscillations, not recommended")
    print("  ζ = 1.0      : Critically damped, theoretically fastest (not robust)")
    print("  ζ = 1.0-1.2  : Slightly overdamped, optimal for practice ✅")
    print("  ζ = 1.2-2.0  : Overdamped, slower but smoother")
    print("  ζ > 3.0      : Severely overdamped, very slow response ❌")

    print("\nParameter Calculation:")
    print("  1. Choose Kp (stiffness):")
    print(f"     - Kp × max_error ≤ {output}_limit")
    print(f"     - For error={max_error} and limit={limit}: Kp ≤ {limit / max_error:.1f}")
    print()
    print("  2. Calculate Kd (damping):")
    print(f"     - Kd = ζ × 2√(Kp × {param})")
    print("     - Recommended ζ = 1.2")
    kp_example = limit / max_error
    kd_example = 1.2 * 2 * math.sqrt(kp_example * mass_or_inertia)
    print(
        f"     - Example: Kp={kp_example:.1f} → Kd = 1.2 × 2√({kp_example:.1f}×{mass_or_inertia:.4f}) ≈ {kd_example:.1f}"
    )
    print(f"\nUnits: Kp, Kd in {units}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="PD Controller Parameter Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze linear system (cart with mass)
  %(prog)s --mass 14.8 --limit 500 --error 2.0

  # Analyze rotational system (pendulum with inertia)
  %(prog)s --inertia 0.00104 --limit 500 --error 0.5 --type rotational

  # Auto-generate optimal parameters
  %(prog)s --mass 14.8 --limit 500 --error 2.0 --auto

  # Custom configurations
  %(prog)s --mass 14.8 --limit 500 --configs "200,300,Old" "250,146,New"
        """,
    )

    # System parameters
    system_group = parser.add_mutually_exclusive_group(required=True)
    system_group.add_argument("--mass", type=float, help="System mass in kg (for linear systems)")
    system_group.add_argument("--inertia", type=float, help="System inertia in kg·m² (for rotational systems)")

    parser.add_argument("--limit", type=float, required=True, help="Force (N) or torque (N·m) limit")
    parser.add_argument("--error", type=float, required=True, help="Maximum expected position/angle error")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step in seconds (default: 0.01)")
    parser.add_argument(
        "--type",
        choices=["linear", "rotational"],
        help="System type (auto-detected from --mass or --inertia if not specified)",
    )

    # Configuration options
    parser.add_argument(
        "--auto", action="store_true", help="Auto-generate optimal configurations (Aggressive, Balanced, Conservative)"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        metavar="KP,KD,NAME",
        help='Custom configurations as "kp,kd,name" (e.g., "200,300,Old Config")',
    )

    args = parser.parse_args()

    # Determine system type
    if args.mass is not None:
        mass_or_inertia = args.mass
        system_type = args.type or "linear"
        param_name = "mass"
        param_unit = "kg"
    else:
        mass_or_inertia = args.inertia
        system_type = args.type or "rotational"
        param_name = "inertia"
        param_unit = "kg·m²"

    # Print header
    print("\n" + "=" * 70)
    print(f"🔬 PD Controller Parameter Analysis ({system_type.capitalize()} System)")
    print("=" * 70)
    print("\nSystem Parameters:")
    print(f"  - {param_name.capitalize()}: {mass_or_inertia} {param_unit}")
    print(
        f"  - {'Force' if system_type == 'linear' else 'Torque'} limit: {args.limit} {'N' if system_type == 'linear' else 'N·m'}"
    )
    print(f"  - Time step: {args.dt} s ({1 / args.dt:.0f} Hz)")
    print(f"  - Max error: {args.error} {'m' if system_type == 'linear' else 'rad'}")
    print("  - Convergence criterion: 2% error")

    # Create analyzer
    analyzer = PDAnalyzer(
        mass_or_inertia=mass_or_inertia,
        force_or_torque_limit=args.limit,
        dt=args.dt,
        system_type=system_type,
    )

    # Determine configurations
    if args.auto:
        configs = generate_optimal_configs(mass_or_inertia, args.limit, args.error, system_type)
    elif args.configs:
        configs = []
        for config_str in args.configs:
            parts = config_str.split(",")
            if len(parts) != 3:
                parser.error(f"Invalid config format: {config_str}. Expected 'kp,kd,name'")
            kp, kd, name = float(parts[0]), float(parts[1]), parts[2]
            configs.append((kp, kd, name))
    else:
        # Default: use cartpole example configurations scaled to current mass
        print("\n⚠️  No configurations specified. Use --auto or --configs")
        print_tuning_guide(mass_or_inertia, args.limit, args.error, system_type)
        return

    # Run analysis
    analyzer.compare(configs)

    # Print tuning guide
    print_tuning_guide(mass_or_inertia, args.limit, args.error, system_type)

    # Print conclusion if auto-generated
    if args.auto and len(configs) > 0:
        best = configs[0]  # Aggressive is usually best
        print(f"\n{'=' * 70}")
        print("🎯 Recommended Configuration")
        print(f"{'=' * 70}")
        print(f"  {best[2]}: Kp={best[0]:.1f}, Kd={best[1]:.1f}")
        print("  ✅ Fast convergence")
        print("  ✅ Full utilization of output limit")
        print("  ✅ No overshoot, no oscillation")
        print("  ✅ Optimal damping ratio (ζ=1.2)")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
