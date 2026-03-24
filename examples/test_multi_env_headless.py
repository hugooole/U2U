# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
"""Headless multi-instance environment test.

Verifies that multi-env mode works correctly:
1. Parses cartpole_128.usda with multi_env=True
2. Checks env/robot counts
3. Runs simulation steps
4. Controls joints independently per-env
5. Reads back joint positions to confirm independence

No PipelineBase or polyscope required.
"""

import os
import os.path as osp

from uipc import Engine, Logger

from u2u.scene import Scene
from u2u.usd_parser import UsdParser
from u2u.usd_utils import read_usd
from u2u.world import World


def create_config() -> dict:
    """Return physics config suitable for cartpole."""
    config = Scene.default_config()
    config["sanity_check"] = False
    config["contact"]["enable"] = True
    config["contact"]["d_hat"] = 0.001
    config["newton"]["semi_implicit"] = True
    config["collision_detection"]["method"] = "info_stackless_bvh"
    return config


def main():
    Logger.set_level(Logger.Warn)
    print("=== Multi-Instance Headless Test ===\n")

    # --- 1. Parse USD with multi_env ---
    usd_path = osp.join(osp.dirname(osp.abspath(__file__)), "..", "assets", "usd", "cartpole_128.usda")
    usd_path = osp.abspath(usd_path)
    assert osp.exists(usd_path), f"USD file not found: {usd_path}"

    stage = read_usd(usd_path)
    config = create_config()
    scene = Scene(config)
    parser = UsdParser(scene, stage)
    scene = parser.parse_and_build_scene(
        root_path="/",
        multi_env=True,
        env_scope_path="/World/envs",
        skip_mesh_approximation=False,
        approx_method="convexdecomposition",
    )

    # --- 2. Verify env / robot counts ---
    print(f"num_envs       = {scene.num_envs}")
    print(f"robot_dict     = {len(scene.robot_dict)} robots")
    print(f"env_dict       = {len(scene.env_dict)} envs")
    assert scene.num_envs == 128, f"Expected 128 envs, got {scene.num_envs}"
    assert len(scene.env_dict) == 128

    env0 = scene.get_env(0)
    env5 = scene.get_env(5)
    env127 = scene.get_env(127)
    print(f"env_0  robots  = {list(env0.robot_dict.keys())}")
    print(f"env_5  robots  = {list(env5.robot_dict.keys())}")
    print(f"env_127 robots = {list(env127.robot_dict.keys())}")

    # Inspect template robot
    r0_name = list(env0.robot_dict.keys())[0]
    r0 = env0.robot_dict[r0_name]
    print(f"\nTemplate robot: {r0_name}")
    print(f"  active_joints: {r0.active_joints}")

    # --- 3. Init world and warm up ---
    workdir = "/tmp/multi_env_headless_test"
    os.makedirs(workdir, exist_ok=True)
    engine = Engine(backend_name="cuda", workspace=workdir)
    world = World(engine)
    world.init(scene)

    print("\nWarm-up: 5 steps...")
    for _ in range(5):
        world.advance()
        world.retrieve()
    print(f"  frame = {world.frame()}")

    # --- 4. Independent joint control ---
    r5_name = list(env5.robot_dict.keys())[0]
    r5 = env5.robot_dict[r5_name]
    joint = r0.active_joints[0]  # prismatic joint (slider_to_cart)

    print(f"\nControlling: {joint}")
    joint_path = r0.joint_path_map[joint]
    print(f"  joint type: {r0.joint_geometry[joint_path]['type']}")
    j_idx = r0._get_joint_idx(joint)
    print(f"  limits: [{r0.joint_lower_limits[j_idx]}, {r0.joint_upper_limits[j_idx]}]")

    # Set different targets per instance
    # r0 is env_0, instance 0; r5 is env_5, instance 5
    r0.set_joint_position(joint, 0.5, instance_ids=[0])
    r5.set_joint_position(joint, -0.3, instance_ids=[5])

    print("  r0 target = +0.5")
    print("  r5 target = -0.3")

    print("\nRunning 30 control steps...")
    for _ in range(30):
        world.advance()
        world.retrieve()

    # get_joint_position returns (N,) array; extract specific instances
    pos_array = r0.get_joint_position(joint)
    p0 = pos_array[0]  # instance 0
    p5 = pos_array[5]  # instance 5
    print(f"  r0 pos = {p0:.4f}")
    print(f"  r5 pos = {p5:.4f}")

    assert abs(p0 - 0.5) < 0.01, f"r0 should be near 0.5, got {p0}"
    assert abs(p5 - (-0.3)) < 0.01, f"r5 should be near -0.3, got {p5}"
    assert p0 != p5, "Positions must differ"
    print("  [PASS] Joints move independently across envs.")

    # --- 5. Test second joint (revolute: cart_to_pole) ---
    if len(r0.active_joints) > 1:
        joint2 = r0.active_joints[1]
        print(f"\nControlling second joint: {joint2}")
        joint2_path = r0.joint_path_map[joint2]
        print(f"  joint type: {r0.joint_geometry[joint2_path]['type']}")

        r127_name = list(env127.robot_dict.keys())[0]
        r127 = env127.robot_dict[r127_name]

        r0.set_joint_position(joint2, 0.3, instance_ids=[0])
        r127.set_joint_position(joint2, -0.2, instance_ids=[127])

        for _ in range(30):
            world.advance()
            world.retrieve()

        pos_array2 = r0.get_joint_position(joint2)
        p0_j2 = pos_array2[0]  # instance 0
        p127_j2 = pos_array2[127]  # instance 127
        print(f"  r0   pos = {p0_j2:.4f} (target +0.3)")
        print(f"  r127 pos = {p127_j2:.4f} (target -0.2)")
        assert p0_j2 != p127_j2, "Revolute joint positions must differ"
        print("  [PASS] Revolute joints also independent.")

    # --- 6. Verify all envs have robots ---
    for env_id in range(scene.num_envs):
        env = scene.get_env(env_id)
        assert len(env.robot_dict) > 0, f"env_{env_id} has no robots!"
    print(f"\n  [PASS] All {scene.num_envs} envs have robots.")

    print("\n=== All Tests Passed ===")


if __name__ == "__main__":
    main()
