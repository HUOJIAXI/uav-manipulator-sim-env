# UAV Simulation with PX4 + Stereo Camera + Manipulator

Isaac Sim simulation environment with PX4-enabled quadrotor, stereo cameras, and a 2-DOF folding arm.

## Launch Files

| Script | UAV Body | Stereo Camera | Folding Arm | Description |
|--------|----------|---------------|-------------|-------------|
| `launch_with_camera.py` | Iris | No (mono) | No | Basic mono camera setup |
| `launch_with_stereo_camera.py` | Iris | Yes | No | Stereo camera on default Iris |
| `launch_with_arm.py` | Iris | No | Yes | Iris with folding arm |
| `launch_stereo_default.py` | Iris | Yes | No | Stereo on default Iris (Pegasus env) |
| `launch_stereo_default_with_arm.py` | Iris | Yes | Yes | Stereo + arm on default Iris |
| `launch_stereo_vslam_with_arm.py` | Iris VSLAM | Yes | Yes | VSLAM body + stereo + arm (recommended) |

## Quick Start

```bash
cd ~/Desktop/uav_sim

# Recommended: Iris VSLAM body with stereo cameras + folding arm
~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh launch_stereo_vslam_with_arm.py
```

Wait for `READY!` and PX4 console output.

## `launch_stereo_vslam_with_arm.py` Pipeline

This is the main launch file. It uses the `iris_vslam.usd` model (from `assets/`) with a workaround for Isaac Sim 5.1 compatibility:

1. **Load environment** - Pegasus Curved Gridroom
2. **Spawn vehicle** - Iris VSLAM body from `assets/iris_vslam.usd`
3. **Clean stale OmniGraph** - Removes old `omni.isaac.*` action graphs embedded in the USD (incompatible with Isaac Sim 5.1's `isaacsim.*` node types)
4. **Create stereo camera prims** - Fresh pinhole cameras on `/World/quadrotor/body/stereo_left` and `stereo_right` (50 mm baseline, forward-looking)
5. **Reset world** - Initialize physics articulations
6. **Add landing legs** - Cylinder legs below each rotor
7. **Add 2-DOF folding arm** - Shoulder + elbow revolute joints with position drive, attached to body via FixedJoint
8. **Start ROS2 arm bridge** - Subscribes to joint commands, publishes joint states
9. **Play timeline + warm up renderer**
10. **Create viewport-based stereo camera publishers** - Hidden viewports with `schedule_capture(ByteCapture(...))` to bypass broken SyntheticData pipeline, publishing via ROS2

### Why viewport capture instead of replicator writers?

The `iris_vslam.usd` contains embedded OmniGraph nodes using old Isaac Sim 4.x node type names (`omni.isaac.core_nodes.*`, `omni.isaac.ros2_bridge.*`). These are not resolved by Isaac Sim 5.1 and corrupt the global `SyntheticData` singleton, causing all `annotator.attach()` and `writer.attach()` calls to fail with `TypeError: Unable to write from unknown dtype`. The viewport capture path (`viewport_api.schedule_capture`) bypasses SyntheticData entirely.

## ROS2 Topics

### Stereo Camera
| Topic | Type | Description |
|-------|------|-------------|
| `front_stereo_camera/left/image_rect_color` | `sensor_msgs/Image` | Left camera RGBA image |
| `front_stereo_camera/left/camera_info` | `sensor_msgs/CameraInfo` | Left camera intrinsics |
| `front_stereo_camera/right/image_rect_color` | `sensor_msgs/Image` | Right camera RGBA image |
| `front_stereo_camera/right/camera_info` | `sensor_msgs/CameraInfo` | Right camera intrinsics |

### Folding Arm
| Topic | Type | Description |
|-------|------|-------------|
| `drone0/arm/joint_states` | `sensor_msgs/JointState` | Current arm joint positions |
| `drone0/arm/joint_command` | `sensor_msgs/JointState` | Send arm joint commands (radians) |

### Control the Arm
```bash
# Unfold arm (shoulder=0, elbow=0)
ros2 topic pub --once drone0/arm/joint_command sensor_msgs/msg/JointState \
  "{name: ['shoulder_joint','elbow_joint'], position: [0.0, 0.0]}"

# Fold arm (shoulder=-90deg, elbow=180deg)
ros2 topic pub --once drone0/arm/joint_command sensor_msgs/msg/JointState \
  "{name: ['shoulder_joint','elbow_joint'], position: [-1.5708, 3.1416]}"
```

### Launch Offboard Control
```bash
ros2 launch px4_offboard_control offboard.launch.py use_sim_time:=true
```

## Stereo Camera Parameters

| Parameter | Value |
|-----------|-------|
| Baseline | 50 mm |
| Forward offset | 100 mm from body center |
| Resolution | 640 x 480 |
| Projection | Pinhole |
| Encoding | RGBA8 |
| Publish rate | ~10 Hz |

## Folding Arm Parameters

| Parameter | Value |
|-----------|-------|
| Link length | 125 mm per link |
| Link radius | 8 mm |
| Link mass | 50 g per link |
| Joints | Shoulder (Y-axis, -90 to 90 deg), Elbow (Y-axis, 0 to 180 deg) |
| Default state | Folded (shoulder=-90, elbow=180) |
| Drive | Position control (stiffness=1e5, damping=1e4) |

## Project Structure

```
uav_sim/
  assets/
    iris_vslam.usd          # Iris VSLAM quadrotor model
  launch_stereo_vslam_with_arm.py   # Main launch file (recommended)
  launch_stereo_default_with_arm.py # Default Iris + stereo + arm
  launch_stereo_default.py          # Default Iris + stereo
  launch_with_stereo_camera.py      # Stereo camera setup
  launch_with_camera.py             # Basic mono camera
  launch_with_arm.py                # Iris + arm
  run_sim.sh                        # Legacy launcher
  README.md
```

## Requirements

- Isaac Sim 5.1 (with Pegasus Simulator extension)
- PX4-Autopilot
- ROS2 (Jazzy/Humble)
