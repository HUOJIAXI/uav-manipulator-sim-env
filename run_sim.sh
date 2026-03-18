#!/bin/bash
# Launcher with Camera Support

echo "================================================"
echo "  UAV Warehouse with PX4 + Camera"
echo "================================================"
echo ""

# Load isaac_run function (bashrc guard blocks non-interactive sourcing)
export ISAACSIM_PATH="${HOME}/isaac-sim"
export ISAACSIM_PYTHON="${ISAACSIM_PATH}/python.sh"
export ISAACSIM="${ISAACSIM_PATH}/isaac-sim.sh"
eval "$(sed -n '/^isaac_run()/,/^}/p' "${HOME}/.bashrc")"

# Path to Isaac Sim
ISAAC_SIM_PATH="${HOME}/isaac-sim"

# Check if Isaac Sim exists
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "ERROR: Isaac Sim not found at $ISAAC_SIM_PATH"
    exit 1
fi

# Setup ROS2 for Isaac Sim's built-in ROS2 Bridge
echo "Setting up ROS2 environment..."

# Isaac Sim uses its own bundled ROS2 Jazzy
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ISAAC_SIM_PATH}/exts/isaacsim.ros2.bridge/jazzy/lib
export ROS_DOMAIN_ID=0

echo "✓ ROS2 configured"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Launch using Isaac Sim's Python
# "${ISAAC_SIM_PATH}/python.sh" launch_with_camera.py
isaac_run launch_with_camera.py

echo ""
echo "Simulation closed."
