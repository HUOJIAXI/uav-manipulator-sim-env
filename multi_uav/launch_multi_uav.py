#!/usr/bin/env python3
"""
Multi-UAV Spawning Launch Script.
Spawns multiple UAV-manipulators from a YAML config file.

Usage:
    python launch_multi_uav.py [--config path/to/config.yaml]
"""

import argparse
import math
import os
import sys
import time

import yaml


def load_config(config_path):
    """Load and validate YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    drones = cfg.get("drones", [])
    if not drones:
        print("ERROR: Config must define at least one drone in 'drones' list.")
        sys.exit(1)

    # Validate required fields and drone id uniqueness
    id_set = set()
    for i, d in enumerate(drones):
        if "id" not in d:
            print(f"ERROR: Drone entry {i} missing required 'id' field.")
            sys.exit(1)
        did = d["id"]
        if did in id_set:
            print(f"ERROR: Duplicate drone id={did} in config.")
            sys.exit(1)
        id_set.add(did)

    # Validate px4_vehicle_id uniqueness
    vid_set = set()
    for d in drones:
        vid = d.get("px4_vehicle_id", d["id"])
        if vid in vid_set:
            print(f"ERROR: Duplicate px4_vehicle_id={vid} in config.")
            sys.exit(1)
        vid_set.add(vid)

    # Warn on overlapping spawn positions (< 1m apart)
    for i, a in enumerate(drones):
        for b in drones[i + 1:]:
            dist = math.sqrt((a.get("x", 0) - b.get("x", 0)) ** 2
                             + (a.get("y", 0) - b.get("y", 0)) ** 2)
            if dist < 1.0:
                print(f"WARNING: drone{a['id']} and drone{b['id']} are only "
                      f"{dist:.2f}m apart — risk of physics collision at spawn.")

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Multi-UAV Isaac Sim Launcher")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "default_config.yaml"),
        help="Path to YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Validate asset exists
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    usd_path = os.path.join(assets_dir, "iris_vslam.usd")
    if not os.path.exists(usd_path):
        print(f"ERROR: iris_vslam.usd not found at {usd_path}")
        sys.exit(1)

    cfg = load_config(args.config)
    spawn_height = cfg.get("spawn_height", 0.30)
    env_name = cfg.get("environment", "Curved Gridroom")
    drones_cfg = cfg["drones"]

    num_drones = len(drones_cfg)
    print("\n" + "=" * 70)
    print(f"  Multi-UAV Launcher — {num_drones} drone(s)")
    print("=" * 70 + "\n")

    # --- Init SimulationApp ---
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    import omni.timeline
    import omni.usd
    from omni.isaac.core.world import World
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")

    from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

    # --- Init Pegasus & World ---
    print("Initializing Pegasus...")
    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world

    print(f"Loading environment: {env_name}...")
    pg.load_environment(SIMULATION_ENVIRONMENTS[env_name])
    time.sleep(2.0)
    simulation_app.update()
    print("  Environment loaded\n")

    # --- Spawning and simulation ---
    from multi_uav.spawn_uav import spawn_uav, create_arm, ArmBridgeNode, StereoCamPublisher

    stage = omni.usd.get_context().get_stage()

    # --- Phase 1: Spawn all Multirotors (before world.reset()) ---
    print("Spawning drones...")
    drone_handles = []
    for dcfg in drones_cfg:
        handle = spawn_uav(stage, world, dcfg, spawn_height, assets_dir, simulation_app)
        drone_handles.append(handle)
        time.sleep(0.5)

    # --- world.reset() ---
    print("Resetting world...")
    world.reset()
    stage = omni.usd.get_context().get_stage()

    # --- Phase 2: Create legs + arms (after world.reset()) ---
    print("Creating legs and arms...")
    for i, dcfg in enumerate(drones_cfg):
        arm_result = create_arm(stage, dcfg, spawn_height, simulation_app)
        drone_handles[i].update(arm_result)

    # --- ROS2 setup ---
    print("Setting up ROS2...")
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading

    rclpy.init()
    executor = MultiThreadedExecutor()

    arm_bridges = []
    stereo_pubs = []

    for handle in drone_handles:
        did = handle["drone_id"]

        # Arm bridge
        if handle.get("arm_drives"):
            bridge = ArmBridgeNode(did, handle["arm_drives"])
            executor.add_node(bridge)
            arm_bridges.append(bridge)
            handle["arm_bridge"] = bridge
            print(f"  drone{did}: arm bridge node ready")
        else:
            handle["arm_bridge"] = None

        # Stereo publisher
        if handle.get("stereo_cam_paths"):
            pub = StereoCamPublisher(did)
            stereo_pubs.append(pub)
            handle["stereo_pub"] = pub
            print(f"  drone{did}: stereo publisher node ready")
        else:
            handle["stereo_pub"] = None

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    print("  ROS2 executor spinning\n")

    # --- Timeline ---
    timeline = omni.timeline.get_timeline_interface()

    try:
        timeline.play()

        # --- Stereo viewports (must be after timeline.play()) ---
        import omni.kit.viewport.utility as vp_utils
        STEREO_CAM_RESOLUTION = (640, 480)

        for handle in drone_handles:
            if handle.get("stereo_cam_paths") and handle.get("stereo_pub"):
                did = handle["drone_id"]
                cam_viewports = {}
                for side, cam_path in handle["stereo_cam_paths"].items():
                    vp_name = f"drone{did}_stereo_{side}"
                    vp_window = vp_utils.create_viewport_window(
                        vp_name,
                        width=STEREO_CAM_RESOLUTION[0],
                        height=STEREO_CAM_RESOLUTION[1],
                        visible=False,
                    )
                    vp_api = vp_window.viewport_api
                    vp_api.set_active_camera(cam_path)
                    cam_viewports[side] = {
                        "viewport_api": vp_api,
                        "viewport_window": vp_window,
                    }
                    print(f"  drone{did}: {side} viewport -> {cam_path}")
                handle["stereo_pub"].set_viewports(cam_viewports)

        # Warm up renderer
        print("  Warming up renderer...")
        for _ in range(10):
            world.step(render=True)

        # --- Print summary ---
        print("\n" + "=" * 70)
        print("  READY!")
        print("=" * 70)
        for handle in drone_handles:
            did = handle["drone_id"]
            print(f"\n  drone{did}:")
            if handle.get("arm_bridge"):
                print(f"    drone{did}/arm/joint_command")
                print(f"    drone{did}/arm/joint_states")
            if handle.get("stereo_pub"):
                print(f"    drone{did}/front_stereo_camera/left/image_rect_color")
                print(f"    drone{did}/front_stereo_camera/right/image_rect_color")
        print("\nPress Ctrl+C to exit\n")

        # --- Simulation loop ---
        if not hasattr(world, 'current_time'):
            import carb
            carb.log_warn("World has no 'current_time' attribute; timestamps will be 0.0")

        pub_counter = 0
        while simulation_app.is_running():
            world.step(render=True)

            # Apply arm commands
            for bridge in arm_bridges:
                bridge.apply_commands()

            # Publish at ~10 Hz
            pub_counter += 1
            if pub_counter >= 6:
                pub_counter = 0
                sim_time = world.current_time if hasattr(world, 'current_time') else 0.0
                for bridge in arm_bridges:
                    bridge.publish_states(sim_time)
                for pub in stereo_pubs:
                    pub.capture_and_publish(sim_time)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        timeline.stop()
        for pub in stereo_pubs:
            pub.destroy_node()
        for bridge in arm_bridges:
            bridge.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
        simulation_app.close()
        print("Shutdown complete\n")


if __name__ == "__main__":
    main()
