#!/usr/bin/env python3
"""
UAV Warehouse with PX4 + Camera
Spawns Pegasus vehicle with an attached camera
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.sensor import Camera
import omni.graph.core as og
from pxr import Sdf

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

print("\n" + "="*70)
print("  UAV Warehouse with PX4 + Camera")
print("="*70 + "\n")

# Load warehouse
print("Loading warehouse...")
open_stage("/home/huojiaxi/Desktop/uav_sim/uav_warehouse.usd")
import time
time.sleep(10.0)  # Wait for stage to load

# Single update to process stage loading
simulation_app.update()
time.sleep(10.0)
print("✓ Loaded\n")

# Initialize Pegasus BEFORE searching for action graphs
print("Initializing Pegasus...")
try:
    pg = PegasusInterface()
    time.sleep(0.5)  # Let PegasusInterface initialize
    pg._world = World(**pg._world_settings)
    world = pg._world
    time.sleep(0.5)  # Let World initialize
    print("✓ Initialized\n")
except Exception as e:
    print(f"Error initializing Pegasus: {e}")
    print("This may be due to stage/world conflicts.")
    print("Try using a USD without action graphs, or create vehicle manually.")
    raise

# NOW search for action graphs (after Pegasus is initialized)
print("Searching for action graphs...")
stage = omni.usd.get_context().get_stage()
action_graphs = []

# Find all OmniGraph and ActionGraph nodes in the USD
for prim in stage.Traverse():
    prim_type = prim.GetTypeName()
    prim_name = prim.GetName().lower()

    # Check for OmniGraph type or ActionGraph in name
    if prim_type == "OmniGraph" or "actiongraph" in prim_name or "action_graph" in prim_name:
        graph_path = str(prim.GetPath())
        action_graphs.append(graph_path)
        print(f"  Found: {graph_path}")

if action_graphs:
    print(f"  Will initialize {len(action_graphs)} graph(s) after camera creation")
    print("  (Note: You may see warnings during world reset - this is expected)\n")
else:
    print("  No action graphs found in USD\n")

# Clean old vehicles
print("Removing old vehicles...")
# stage already obtained above
to_remove = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    name = prim.GetName().lower()
    if any(k in name for k in ["quadrotor", "drone", "iris"]):
        if path.count('/') == 2 and path.startswith('/World/'):
            to_remove.append(path)

for path in to_remove:
    stage.RemovePrim(path)
    print(f"  Removed: {path}")
print(f"✓ Cleaned\n")

# Reset world
print("Resetting world...")
world.reset()
time.sleep(0.5)  # Wait for physics engine to stabilize
print("✓ Ready\n")

# Configure PX4
print("Creating Pegasus vehicle with PX4...")
config = MultirotorConfig()
px4_cfg = PX4MavlinkBackendConfig({
    "vehicle_id": 0,
    "px4_autolaunch": True,
    "px4_dir": pg.px4_path,
    "px4_vehicle_model": pg.px4_default_airframe
})
config.backends = [PX4MavlinkBackend(px4_cfg)]

# Spawn vehicle
drone = Multirotor(
    "/World/quadrotor",
    ROBOTS['Iris'],
    0,
    [0.0, 0.0, 0.07],
    Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
    config=config,
)
time.sleep(1.0)  # Wait for vehicle prims to be created
print("✓ Vehicle created\n")

# Add camera to the vehicle BODY (not root)
print("Adding camera to vehicle body...")

# Wait for vehicle body prim to be fully created
max_retries = 10
body_prim = None
for i in range(max_retries):
    body_prim = stage.GetPrimAtPath("/World/quadrotor/body")
    if body_prim.IsValid():
        break
    time.sleep(0.2)

if not body_prim or not body_prim.IsValid():
    raise RuntimeError("Failed to find vehicle body prim after vehicle creation")

# The camera must be attached to the BODY prim which actually moves
camera_prim_path = "/World/quadrotor/body/camera"

# Camera orientation: look forward and upright
# Adjust yaw to face correct horizontal direction
from scipy.spatial.transform import Rotation as R
camera_rot = R.from_euler('xyz', [-90, 0, 90], degrees=True)  # Turn 90° more left
camera_quat = camera_rot.as_quat()  # [x, y, z, w]

# Create camera prim as child of quadrotor/body (the moving part!)
camera_prim = create_prim(
    camera_prim_path,
    "Camera",
    position=[0.1, 0.0, 0.05],  # 10cm forward, 5cm up from body
    orientation=camera_quat  # Face forward
)

print(f"✓ Camera attached to body at {camera_prim_path}")

# Configure camera using Isaac Sim Camera API
camera = Camera(
    prim_path=camera_prim_path,
    resolution=(640, 480),
    frequency=30
)

# Initialize camera
camera.initialize()
time.sleep(0.3)  # Wait for camera initialization

# Add camera to world scene
world.scene.add(camera)
time.sleep(0.2)  # Wait for scene to register camera

print(f"  Resolution: 640x480")
print(f"  FPS: 30")
print(f"  Attached to moving body - will follow UAV!\n")

# NOW initialize action graphs (camera exists now!)
if action_graphs:
    print(f"Initializing {len(action_graphs)} action graph(s)...")
    time.sleep(0.5)  # Wait before initializing

    # Use OmniGraph Controller for proper graph management
    controller = og.Controller()

    for graph_path in action_graphs:
        try:
            # Get the graph and evaluate it
            graph = og.get_graph_by_path(graph_path)
            if graph:
                # Use controller to evaluate with camera now present
                controller.evaluate_sync(graph)
                print(f"  ✓ Initialized: {graph_path}")
            else:
                print(f"  ⚠ Could not get graph: {graph_path}")
        except Exception as e:
            print(f"  ⚠ Failed to initialize {graph_path}: {str(e)[:80]}")

    # Force simulation app update to register graphs
    simulation_app.update()
    time.sleep(0.5)  # Wait after initialization
    print("✓ Action graphs initialized\n")

# Create ROS2 Clock publisher for use_sim_time:=true
print("Creating ROS2 Clock publisher...")
try:
    # Create a simple clock publisher graph
    keys = og.Controller.Keys
    (clock_graph, nodes, _, _) = og.Controller.edit(
        {
            "graph_path": "/World/ROS2_Clock",
            "evaluator_name": "execution",
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("IsaacReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("ROS2PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "IsaacReadSimTime.inputs:execIn"),
                ("IsaacReadSimTime.outputs:execOut", "ROS2PublishClock.inputs:execIn"),
                ("IsaacReadSimTime.outputs:simulationTime", "ROS2PublishClock.inputs:timeStamp"),
            ],
        },
    )
    print("✓ ROS2 Clock publisher created")
    print("  Publishing to /clock topic for use_sim_time:=true\n")
except Exception as e:
    print(f"⚠ Could not create clock publisher: {str(e)[:80]}")
    print("  You may need to enable ROS2 Bridge extension manually\n")

print("="*70)
print("  READY!")
print("="*70)
print("\n✓ Pegasus vehicle with camera spawned")
print("✓ PX4 SITL starting...")
print("✓ ROS2 Clock publisher active (/clock topic)\n")
print("🎥 To publish camera via ROS2:")
print("   1. Enable ROS2 Bridge: Window → Extensions → 'Isaac ROS2 Bridge'")
print("   2. Camera data will be available on ROS2 topics\n")
print("🚀 Launch ROS2 offboard control:")
print("   ros2 launch px4_offboard_control offboard.launch.py use_sim_time:=true\n")
print("⏱️  Simulation time is published to /clock topic")
print("   Use 'use_sim_time:=true' with all ROS2 launch commands\n")
print("Press Ctrl+C to exit\n")

timeline = omni.timeline.get_timeline_interface()

try:
    timeline.play()

    # Let simulation initialize
    time.sleep(1.0)

    # Main loop
    while simulation_app.is_running():
        world.step(render=True)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    timeline.stop()
    simulation_app.close()
    print("✓ Shutdown complete\n")
