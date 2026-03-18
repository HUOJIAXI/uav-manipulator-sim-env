# UAV Warehouse Simulation with PX4 + Camera

This directory contains a complete Isaac Sim simulation with PX4-enabled quadrotor and camera.

## 🚀 Quick Start

```bash
cd ~/Desktop/uav_sim
./run_sim.sh
```

## 📋 What It Does

1. ✅ Loads your warehouse environment (`uav_warehouse.usd`)
2. ✅ Spawns Pegasus Iris quadrotor with PX4 SITL
3. ✅ Attaches forward-facing camera to the UAV
4. ✅ Camera moves with UAV and faces flight direction

## 🎥 Camera Details

- **Location**: `/World/quadrotor/body/camera`
- **Position**: 10cm forward, 5cm up from body center
- **Resolution**: 640x480 @ 30fps
- **Orientation**: Faces forward, aligned with UAV movement

## 📡 Using with ROS2

### Step 1: Launch Simulation
```bash
./run_sim.sh
```

Wait for: `READY!` and PX4 console output

### Step 2: Enable ROS2 Bridge (for camera topics)
In Isaac Sim:
- **Window → Extensions**
- Search "Isaac ROS2 Bridge"
- Enable it

### Step 3: Launch ROS2 Offboard Control
```bash
# Terminal 2
cd ~/Desktop/quadrotor_ws
source install/setup.bash
ros2 launch px4_offboard_control offboard.launch.py use_sim_time:=true
```

**Important:** Always use `use_sim_time:=true` - the simulation publishes clock to `/clock` topic.

### Step 4: Check Camera Topics
```bash
# Terminal 3
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib

ros2 topic list | grep camera
ros2 topic echo /camera/rgb/image_raw --once
```

## 📁 Files

- `uav_warehouse.usd` - Your warehouse scene
- `launch_with_camera.py` - Main Python script
- `run_sim.sh` - Launcher script
- `README.md` - This file

## ⏱️ Simulation Time (use_sim_time)

The simulation automatically publishes simulation time to the `/clock` topic. This allows ROS2 nodes to synchronize with the simulation time instead of using wall clock time.

**Always use `use_sim_time:=true` when launching ROS2 nodes:**
```bash
ros2 launch px4_offboard_control offboard.launch.py use_sim_time:=true
ros2 run your_package your_node --ros-args -p use_sim_time:=true
```

**Benefits:**
- Consistent timing in slow/fast simulations
- Repeatable experiments
- Proper tf transforms and time-based algorithms

**Verify clock is publishing:**
```bash
ros2 topic hz /clock
ros2 topic echo /clock --once
```

## 🔧 Customization

### Adjust Camera Position
Edit `launch_with_camera.py` line ~105:
```python
position=[0.1, 0.0, 0.05],  # [forward, right, up] in meters
```

### Adjust Camera Angle
Edit `launch_with_camera.py` line ~98:
```python
camera_rot = R.from_euler('xyz', [-90, 0, 90], degrees=True)
```

### Change Resolution
Edit `launch_with_camera.py` line ~114:
```python
resolution=(640, 480),  # (width, height)
frequency=30  # FPS
```

## 🎯 System Requirements

- Isaac Sim (with Pegasus Simulator extension)
- PX4-Autopilot at `~/Pegasus/PX4-Autopilot`
- ROS2 Humble (for offboard control)

## ✨ Features

- ✅ PX4 SITL auto-starts with simulation
- ✅ Camera rigidly attached to UAV body
- ✅ MAVLink connection ready for ROS2
- ✅ Forward-facing FPV camera view
- ✅ ROS2 Bridge support for camera topics
- ✅ **Auto-loads action graphs from USD**
- ✅ **Improved timing for reliable startup**
- ✅ **ROS2 Clock publisher for simulation time** (new!)

## 🔧 Troubleshooting

### Intermittent Crashes During Launch

The script now includes strategic timing delays (~6.5s total) to prevent race conditions:
- Waits for stage to fully load
- Waits for physics engine to stabilize
- Verifies vehicle body prim exists before adding camera
- Updates simulation app before starting timeline

**If crashes still occur:**
- Check Isaac Sim console for error messages
- Ensure PX4-Autopilot is at `~/Pegasus/PX4-Autopilot`
- Try closing Isaac Sim completely and relaunching

### Action Graphs Not Working

The script automatically:
1. Searches for action graphs in your USD
2. Enables their pipelines
3. Evaluates them before and after simulation starts

**If action graphs still don't load:**
- Check the launch console - it will list found graphs
- Manually enable ROS2 Bridge: Window → Extensions → "Isaac ROS2 Bridge"
- Verify your USD contains valid action graph prims

### Camera Not Publishing

1. Enable ROS2 Bridge extension manually (if not auto-enabled)
2. Check ROS2 environment in separate terminal:
```bash
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib
ros2 topic list | grep camera
```

---

**Enjoy flying!** 🚁📷
