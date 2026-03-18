# Flexible Arm Controller Design

## Overview

ROS2 package + simulation-side OmniGraph to control the 2-DOF folding arm joints via GUI sliders or topic commands.

## Part 1: Simulation OmniGraph

Add to `launch_stereo_default_with_arm.py` after arm creation:
- ROS2 JointState publisher on `drone0/arm/joint_states`
- ROS2 JointState subscriber on `drone0/arm/joint_command`
- IsaacArticulationController to apply received commands

Uses the same pattern as the Cobotta arm graph in `launch_with_arm.py`.

## Part 2: ROS2 Package

**Path:** `/home/huojiaxi/Desktop/quadrotor_ws/src/flexible_arm_controller`

### Files

```
flexible_arm_controller/
  package.xml
  setup.py
  resource/flexible_arm_controller
  launch/arm_control.launch.py
  flexible_arm_controller/
    __init__.py
    arm_gui_node.py
    arm_command_node.py
```

### arm_gui_node.py

- Tkinter GUI with two sliders:
  - Shoulder: -90 to +90 deg (default 90 = folded)
  - Elbow: -180 to 0 deg (default -180 = folded)
- Subscribes to `drone0/arm/joint_states` to display current positions
- Publishes `sensor_msgs/JointState` to `drone0/arm/joint_command` at 10 Hz
- Preset buttons: "Fold" (90, -180) and "Deploy" (0, 0)

### arm_command_node.py

- Takes joint positions as CLI parameters
- Publishes a single JointState command and exits
- Usage: `ros2 run flexible_arm_controller arm_command --ros-args -p shoulder:=0.0 -p elbow:=0.0`

### Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `drone0/arm/joint_states` | sensor_msgs/JointState | sim → ROS2 |
| `drone0/arm/joint_command` | sensor_msgs/JointState | ROS2 → sim |

### Joint Names

- `shoulder_joint`
- `elbow_joint`
