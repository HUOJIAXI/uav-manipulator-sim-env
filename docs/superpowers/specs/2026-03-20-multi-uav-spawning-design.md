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

## Functions in `multi_uav/spawn_uav.py`

### `spawn_uav()` — called before `world.reset()`

```python
def spawn_uav(stage, world, drone_cfg, spawn_height, assets_dir, sim_app):
    """
    Spawns a UAV body (Multirotor, cameras, legs) before world.reset().

    Args:
        stage: USD stage
        world: Isaac Sim World
        drone_cfg: dict from YAML config for this drone
        spawn_height: float, Z height for spawning (from top-level config)
        assets_dir: path to assets/ folder (for iris_vslam.usd)
        sim_app: SimulationApp instance

    Returns:
        dict:
            "drone_id": int
            "multirotor": Multirotor instance
            "stereo_cam_paths": dict (or None if stereo disabled)
    """
```

**Steps:**

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

### `create_arm()` — called after `world.reset()`

```python
def create_arm(stage, drone_cfg, spawn_height, sim_app):
    """
    Creates 2-DOF folding arm and landing legs for a drone after world.reset().
    Derives prim paths from drone_cfg["id"] (e.g. id=0 -> /World/drone0/).

    Args:
        stage: USD stage
        drone_cfg: dict from YAML config for this drone
        spawn_height: float, Z height (from top-level config)
        sim_app: SimulationApp instance

    Returns:
        dict:
            "arm_drives": dict of DriveAPI handles {joint_name: DriveAPI}
            "shoulder_path": str
            "elbow_path": str
    """
```

**Steps:**

4. **Add landing legs** under `/World/droneN/quadrotor/body/`
   (legs are created after `world.reset()`, matching the existing script's order)

5. **Create 2-DOF folding arm** (if `arm: true`)
   - Arm root: `/World/droneN/folding_arm/` (derived from `drone_cfg["id"]`)
   - Body path: `/World/droneN/quadrotor/body` (derived from `drone_cfg["id"]`)
   - Links: `base_link`, `upper_link`, `lower_link`
   - Joints: `shoulder_joint`, `elbow_joint`
   - FixedJoint attaches to `/World/droneN/quadrotor/body`
   - Same dimensions and drive parameters as current script
   - **Arm link world positions must be computed from this drone's spawn position `(x, y, spawn_height)`**, not hardcoded to origin
   - Calls `sim_app.update()` after prim creation, matching existing script

### ROS2 Node Classes (in `spawn_uav.py`)

Both classes are self-contained — each instance holds its own drives dict / viewport refs, command lock, and pending targets as instance attributes (not module-level globals/closures). Topic namespacing is done by string-prefixing topic names with `droneN/` in the constructor.

- **`ArmBridgeNode`**: takes `drone_id` and `arm_drives` dict in constructor. Owns its own `cmd_lock` and `cmd_targets` dict.
- **`StereoCamPublisher`**: takes `drone_id` in constructor. Creates publishers but does **not** hold viewport refs at construction time. Viewports are injected later via a `set_viewports(cam_viewports)` method, since viewports can only be created after `timeline.play()`. This node only publishes on-demand from the sim loop (no subscriptions), so it does **not** need to be added to the executor.

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
- `ArmBridgeNode` instances (which have subscriptions) are added to a single `MultiThreadedExecutor`
- `StereoCamPublisher` instances only publish on-demand from the sim loop and do not need the executor
- One background thread spins the executor
- No per-drone spin threads
- This is a deliberate simplification from the existing script which uses a per-node `rclpy.spin()` thread

## Main Script Flow (`launch_multi_uav.py`)

1. Parse CLI args (`--config path/to/config.yaml`)
2. Validate config: check `px4_vehicle_id` uniqueness, warn on overlapping spawn positions
3. Init `SimulationApp`, enable ROS2 bridge extension
4. Init Pegasus, create World, load environment from config
5. Loop over `drones` list: call `spawn_uav()` for each (creates Multirotor, cameras — but **not** legs or arm yet), collect handles
6. `world.reset()` once after all Multirotors spawned (initializes articulations)
7. Loop over drones again: call `create_arm()` for each (creates landing legs, and arm if enabled). Merge returned `arm_drives` into the per-drone handle dict.
8. `rclpy.init()`, create `ArmBridgeNode` for each armed drone, create `StereoCamPublisher` for each stereo drone. Add `ArmBridgeNode` instances to a `MultiThreadedExecutor`, spin in background thread.
9. `timeline.play()`
10. Set up stereo camera viewports for all drones with stereo enabled (must be after `timeline.play()`). Call `stereo_pub.set_viewports(cam_viewports)` for each.
11. Warm up renderer
12. Simulation loop:
    - `world.step(render=True)`
    - For each drone: apply pending arm commands
    - At ~10 Hz: publish arm states and stereo images for all drones
13. On shutdown: stop timeline, destroy all nodes, `rclpy.try_shutdown()`, close app

## Constraints and Known Issues

- **MonocularCamera pose bug:** Camera prims are created manually via `create_prim()` to avoid Pegasus's `MonocularCamera.initialize()` resetting poses (documented in project memory).
- **Isaac Sim camera orientation:** Cameras look along local -Z; rotation `[-90, 0, 90]` degrees aligns with drone forward (+X body).
- **Import order:** ROS2 bridge extension must be enabled before any ROS2/OmniGraph usage.
- **PX4 ports:** Each PX4 SITL instance needs unique ports. With sequential `vehicle_id` values, PX4 auto-offsets ports. Explicit port config available for non-standard setups. `px4_vehicle_id` must be unique across all drones; the script validates this at startup.
- **Arm and legs initialization order:** Arm prims, joints, and landing legs must be created after `world.reset()`, matching the existing script's order where these are added after physics initialization.
- **Asset validation:** The script validates that `iris_vslam.usd` exists at startup before entering the drone spawn loop.
- **Viewport scaling:** Each stereo-enabled drone creates 2 hidden viewports. With N drones, that is 2N viewports which is GPU-intensive. Disable `stereo_camera` per-drone in config to reduce load.
- **Spawn position overlap:** The script warns if any two drones are closer than 1m at spawn, as overlapping physics bodies cause simulation instability.

## Config Defaults (default_config.yaml)

The default config produces identical behavior to the existing `launch_stereo_vslam_with_arm.py`:

```yaml
environment: "Curved Gridroom"
spawn_height: 0.30

drones:
  - id: 0
    x: 0.0
    y: 0.0
    yaw: 0.0
    stereo_camera: true
    arm: true
    px4_autolaunch: true
    px4_vehicle_id: 0
```
