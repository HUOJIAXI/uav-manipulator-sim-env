# UAV Simulation with PX4 + Stereo Camera + Manipulator

Isaac Sim simulation environment with PX4-enabled quadrotors, stereo cameras, and 2-DOF folding arms. Supports single and multi-UAV configurations.

## Quick Start

### Single UAV

```bash
cd ~/Desktop/uav_sim
isaac_run launch_stereo_vslam_with_arm.py
```

### Multi-UAV

```bash
cd ~/Desktop/uav_sim/multi_uav

# Default config (1 drone at origin)
isaac_run launch_multi_uav.py

# 2 drones from example config
isaac_run launch_multi_uav.py --config config/example_config.yaml
```

Wait for `READY!` and PX4 console output.

## Multi-UAV System

The `multi_uav/` directory contains the multi-UAV spawning system. It uses a YAML config file to define how many drones to spawn and their configurations.

### Config Format

```yaml
environment: "Curved Gridroom"
spawn_height: 0.30

drones:
  - id: 0
    x: 0.0
    y: 0.0
    yaw: 0.0
    stereo_camera: true    # default: true
    arm: true              # default: true
    px4_autolaunch: true
    px4_vehicle_id: 0

  - id: 1
    x: 2.0
    y: 0.0
    yaw: 90.0
    stereo_camera: true
    arm: true
    px4_autolaunch: true
    px4_vehicle_id: 1
```

Each drone gets:
- Its own PX4 SITL instance (unique `px4_vehicle_id`)
- USD prims under `/World/droneN/`
- ROS2 topics prefixed with `droneN/`
- Optional stereo cameras and 2-DOF arm (per-drone toggle)

### ROS2 Topics (Multi-UAV)

For each drone N:

| Topic | Type | Description |
|-------|------|-------------|
| `droneN/front_stereo_camera/left/image_rect_color` | `sensor_msgs/Image` | Left stereo image |
| `droneN/front_stereo_camera/left/camera_info` | `sensor_msgs/CameraInfo` | Left camera intrinsics |
| `droneN/front_stereo_camera/right/image_rect_color` | `sensor_msgs/Image` | Right stereo image |
| `droneN/front_stereo_camera/right/camera_info` | `sensor_msgs/CameraInfo` | Right camera intrinsics |
| `droneN/arm/joint_states` | `sensor_msgs/JointState` | Arm joint positions |
| `droneN/arm/joint_command` | `sensor_msgs/JointState` | Arm joint commands (radians) |

### Launching Controllers

After the simulation is running, launch the multi-drone controller stack:

```bash
cd ~/Desktop/quadrotor_ws
source install/setup.bash
ros2 launch uav_position_controller multi_drone.launch.py num_drones:=2
```

This launches per-drone: mavsdk_server, offboard_node, position controller, arm compensation, and a shared arm GUI with drone selector.

### Arm Control (Multi-UAV)

```bash
# Deploy drone1's arm
ros2 run flexible_arm_controller arm_command --ros-args -p drone_id:=1 -p shoulder:=0.0 -p elbow:=0.0

# Fold drone0's arm
ros2 run flexible_arm_controller arm_command --ros-args -p drone_id:=0 -p shoulder:=-90.0 -p elbow:=180.0
```

## Single UAV Launch Files

| Script | Stereo Camera | Folding Arm | Description |
|--------|---------------|-------------|-------------|
| `launch_stereo_vslam_with_arm.py` | Yes | Yes | VSLAM body + stereo + arm (recommended) |
| `launch_stereo_default_with_arm.py` | Yes | Yes | Default Iris + stereo + arm |
| `launch_stereo_default.py` | Yes | No | Default Iris + stereo |
| `launch_with_stereo_camera.py` | Yes | No | Stereo camera setup |
| `launch_with_camera.py` | No (mono) | No | Basic mono camera |
| `launch_with_arm.py` | No | Yes | Iris + arm |

### Single UAV ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `front_stereo_camera/left/image_rect_color` | `sensor_msgs/Image` | Left RGBA image |
| `front_stereo_camera/left/camera_info` | `sensor_msgs/CameraInfo` | Left intrinsics |
| `front_stereo_camera/right/image_rect_color` | `sensor_msgs/Image` | Right RGBA image |
| `front_stereo_camera/right/camera_info` | `sensor_msgs/CameraInfo` | Right intrinsics |
| `drone0/arm/joint_states` | `sensor_msgs/JointState` | Arm joint positions |
| `drone0/arm/joint_command` | `sensor_msgs/JointState` | Arm joint commands |

## Project Structure

```
uav_sim/
  assets/
    iris_vslam.usd                    # Iris VSLAM quadrotor model
  multi_uav/
    launch_multi_uav.py               # Multi-UAV launch script
    spawn_uav.py                      # Factory functions (spawn_uav, create_arm)
    config/
      default_config.yaml             # Single drone fallback
      example_config.yaml             # Two-drone example
  launch_stereo_vslam_with_arm.py     # Single-UAV launch (recommended)
  launch_stereo_default_with_arm.py
  launch_stereo_default.py
  launch_with_stereo_camera.py
  launch_with_camera.py
  launch_with_arm.py
```

## Hardware Parameters

### Stereo Camera

| Parameter | Value |
|-----------|-------|
| Baseline | 50 mm |
| Forward offset | 100 mm from body center |
| Resolution | 640 x 480 |
| Projection | Pinhole |
| Encoding | RGBA8 |
| Publish rate | ~10 Hz |

### Folding Arm

| Parameter | Value |
|-----------|-------|
| Link length | 125 mm per link |
| Link radius | 8 mm |
| Link mass | 50 g per link |
| Joints | Shoulder (Y-axis, -90 to 90 deg), Elbow (Y-axis, 0 to 180 deg) |
| Default state | Folded (shoulder=-90, elbow=180) |
| Drive | Position control (stiffness=1e5, damping=1e4) |

## Requirements

- Isaac Sim 5.1 (with Pegasus Simulator extension)
- PX4-Autopilot
- ROS2 (Jazzy)
- PyYAML (`pip install pyyaml`)
