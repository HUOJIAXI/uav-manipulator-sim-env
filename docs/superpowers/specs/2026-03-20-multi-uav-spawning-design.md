# Multi-UAV Spawning Design

## Overview

Add support for spawning multiple UAV-manipulators in Isaac Sim, each with configurable components (stereo cameras, 2-DOF folding arm, PX4 backend). The existing single-drone launch file (`launch_stereo_vslam_with_arm.py`) remains untouched.

## Architecture

**Approach:** Factory function extraction. A `spawn_uav()` function encapsulates all per-drone setup and returns handles. The main script loads config, loops over drones, and runs a unified simulation loop.

## File Structure

```
uav_sim/
├── multi_uav/
│   ├── __init__.py
│   ├── launch_multi_uav.py      # Main entry point
│   ├── spawn_uav.py             # Factory function
│   └── config/
│       ├── default_config.yaml  # Single drone fallback
│       └── example_config.yaml  # Example multi-drone config
├── assets/
│   └── iris_vslam.usd           # Shared by all drones
├── launch_stereo_vslam_with_arm.py  # Untouched
└── ...
```

## YAML Config Format

```yaml
environment: "Curved Gridroom"
spawn_height: 0.30

drones:
  - id: 0
    x: 0.0
    y: 0.0
    yaw: 0.0                    # degrees
    stereo_camera: true          # default: true
    arm: true                    # default: true
    px4_autolaunch: true
    px4_vehicle_id: 0
    # Optional PX4 port overrides:
    # px4_sim_port: 4560
    # px4_mavlink_port: 14540

  - id: 1
    x: 2.0
    y: 0.0
    yaw: 90.0
    stereo_camera: true
    arm: true
    px4_autolaunch: true
    px4_vehicle_id: 1
```

### Config Defaults

- `stereo_camera`: `true`
- `arm`: `true`
- `yaw`: `0.0`
- `px4_autolaunch`: `true`

When no config file is provided via CLI, the script falls back to `config/default_config.yaml`, which defines a single drone at the origin with all components enabled (matching current single-drone behavior).

## Factory Function: `spawn_uav()`

Located in `multi_uav/spawn_uav.py`.

### Signature

```python
def spawn_uav(stage, world, drone_cfg, assets_dir, sim_app):
    """
    Spawns a single UAV-manipulator and returns its handles.

    Args:
        stage: USD stage
        world: Isaac Sim World
        drone_cfg: dict from YAML config for this drone
        assets_dir: path to assets/ folder (for iris_vslam.usd)
        sim_app: SimulationApp instance

    Returns:
        dict:
            "drone_id": int
            "multirotor": Multirotor instance
            "arm_drives": dict of DriveAPI handles (or None if arm disabled)
            "arm_bridge": ArmBridgeNode (or None)
            "stereo_cam_paths": dict (or None if stereo disabled)
            "cam_viewports": dict (or None)
            "stereo_pub": StereoCamPublisher (or None)
    """
```

### Internal Steps

1. **Create Multirotor** at `/World/droneN/quadrotor` using `iris_vslam.usd`
   - Position: `(x, y, spawn_height)` from config
   - Orientation: yaw from config, converted via `Rotation.from_euler`
   - PX4 backend configured with `px4_vehicle_id`, optional port overrides
   - `graphical_sensors = []` (USD has its own)

2. **Clean stale OmniGraphs** under `/World/droneN/` prim tree

3. **Create stereo camera prims** (if `stereo_camera: true`)
   - Left: `/World/droneN/quadrotor/body/stereo_left`
   - Right: `/World/droneN/quadrotor/body/stereo_right`
   - Same baseline (50 mm) and orientation as current script

4. **Add landing legs** under `/World/droneN/quadrotor/body/`

5. **Create 2-DOF folding arm** (if `arm: true`)
   - Arm root: `/World/droneN/folding_arm/`
   - Links: `base_link`, `upper_link`, `lower_link`
   - Joints: `shoulder_joint`, `elbow_joint`
   - FixedJoint attaches to `/World/droneN/quadrotor/body`
   - Same dimensions and drive parameters as current script

6. **Create ROS2 nodes** (per enabled component)
   - Arm: `ArmBridgeNode` named `droneN_arm_bridge`
   - Stereo: `StereoCamPublisher` named `droneN_stereo_cam_publisher`

## USD Prim Hierarchy

```
/World/
  droneN/
    quadrotor/
      body/
        stereo_left       # Camera prim (if stereo enabled)
        stereo_right      # Camera prim (if stereo enabled)
        leg_front_right   # Landing legs (always)
        leg_front_left
        leg_back_left
        leg_back_right
    folding_arm/          # (if arm enabled)
      base_link
      upper_link
      lower_link
      attach_to_body      # FixedJoint -> droneN/quadrotor/body
      shoulder_joint
      elbow_joint
```

## ROS2 Topic Naming

All topics prefixed with `droneN/`:

| Component | Topic |
|-----------|-------|
| Stereo left image | `droneN/front_stereo_camera/left/image_rect_color` |
| Stereo left info | `droneN/front_stereo_camera/left/camera_info` |
| Stereo right image | `droneN/front_stereo_camera/right/image_rect_color` |
| Stereo right info | `droneN/front_stereo_camera/right/camera_info` |
| Arm joint states | `droneN/arm/joint_states` |
| Arm joint command | `droneN/arm/joint_command` |

## ROS2 Architecture

- `rclpy.init()` called once
- All per-drone nodes added to a single `MultiThreadedExecutor`
- One background thread spins the executor
- No per-drone spin threads

## Main Script Flow (`launch_multi_uav.py`)

1. Parse CLI args (`--config path/to/config.yaml`)
2. Init `SimulationApp`, enable ROS2 bridge extension
3. Init Pegasus, create World, load environment from config
4. Loop over `drones` list: call `spawn_uav()`, collect handles
5. `world.reset()` once after all drones spawned
6. `rclpy.init()`, create `MultiThreadedExecutor`, add all ROS2 nodes, spin in background thread
7. `timeline.play()`
8. Set up stereo camera viewports for all drones with stereo enabled
9. Warm up renderer
10. Simulation loop:
    - `world.step(render=True)`
    - For each drone: apply pending arm commands
    - At ~10 Hz: publish arm states and stereo images for all drones
11. On shutdown: stop timeline, destroy all nodes, `rclpy.try_shutdown()`, close app

## Constraints and Known Issues

- **MonocularCamera pose bug:** Camera prims are created manually via `create_prim()` to avoid Pegasus's `MonocularCamera.initialize()` resetting poses (documented in project memory).
- **Isaac Sim camera orientation:** Cameras look along local -Z; rotation `[-90, 0, 90]` degrees aligns with drone forward (+X body).
- **Import order:** ROS2 bridge extension must be enabled before any ROS2/OmniGraph usage.
- **PX4 ports:** Each PX4 SITL instance needs unique ports. With sequential `vehicle_id` values, PX4 auto-offsets ports. Explicit port config available for non-standard setups.
