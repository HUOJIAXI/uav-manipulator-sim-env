#!/usr/bin/env python3
"""
UAV with PX4 + Stereo Camera (Default Pegasus Environment)
Spawns Pegasus vehicle with stereo cameras published via ROS2,
using a built-in Pegasus environment instead of a custom warehouse USD.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from omni.isaac.core.utils.prims import create_prim
import omni.graph.core as og

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation
import numpy as np
import time

# ---------------------------------------------------------------------------
# Stereo camera parameters
# ---------------------------------------------------------------------------
STEREO_BASELINE = 0.055          # 55 mm baseline (similar to RealSense D455)
CAMERA_RESOLUTION = (640, 480)
CAMERA_FPS = 30
CAMERA_FORWARD = 0.10           # 10 cm forward from body centre
CAMERA_UP = 0.05                # 5 cm above body centre

# Camera orientation: forward-looking
# Isaac Sim cameras look along local -Z. This rotation maps -Z_cam -> +X_body (forward).
CAMERA_QUAT = Rotation.from_euler('xyz', [-90, 0, 90], degrees=True).as_quat()  # [x,y,z,w]

print("\n" + "=" * 70)
print("  UAV with PX4 + Stereo Camera (Default Environment)")
print("=" * 70 + "\n")

# ------------------------------------------------------------------
# Initialize Pegasus and load default environment
# ------------------------------------------------------------------
print("Initializing Pegasus...")
pg = PegasusInterface()
pg._world = World(**pg._world_settings)
world = pg.world

# Load a built-in environment (change key to use a different one)
# Options: "Default Environment", "Curved Gridroom", "Black Gridroom",
#          "Flat Plane", "Warehouse", etc. (see pegasus.simulator.params)
print("Loading environment...")
pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
time.sleep(2.0)
simulation_app.update()
print("  Environment loaded\n")

# ------------------------------------------------------------------
# Configure PX4 and spawn vehicle
# ------------------------------------------------------------------
print("Creating Pegasus vehicle with PX4...")
config = MultirotorConfig()
px4_cfg = PX4MavlinkBackendConfig({
    "vehicle_id": 0,
    "px4_autolaunch": True,
    "px4_dir": pg.px4_path,
    "px4_vehicle_model": pg.px4_default_airframe
})
config.backends = [PX4MavlinkBackend(px4_cfg)]

drone = Multirotor(
    "/World/quadrotor",
    ROBOTS['Iris'],
    0,
    [0.0, 0.0, 0.07],
    Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
    config=config,
)
time.sleep(1.0)
print("  Vehicle created\n")

# ------------------------------------------------------------------
# Reset world so articulations are initialized
# ------------------------------------------------------------------
print("Resetting world...")
world.reset()
time.sleep(0.5)
print("  Ready\n")

# ------------------------------------------------------------------
# Add stereo cameras to the vehicle BODY (not root)
# ------------------------------------------------------------------
print("Adding stereo cameras to vehicle body...")
stage = omni.usd.get_context().get_stage()

max_retries = 10
body_prim = None
for i in range(max_retries):
    body_prim = stage.GetPrimAtPath("/World/quadrotor/body")
    if body_prim.IsValid():
        break
    time.sleep(0.2)

if not body_prim or not body_prim.IsValid():
    raise RuntimeError("Failed to find vehicle body prim after vehicle creation")

half_baseline = STEREO_BASELINE / 2.0

stereo_cameras = {
    "stereo_left": {
        "prim_path": "/World/quadrotor/body/stereo_left",
        "position": [CAMERA_FORWARD, half_baseline, CAMERA_UP],
    },
    "stereo_right": {
        "prim_path": "/World/quadrotor/body/stereo_right",
        "position": [CAMERA_FORWARD, -half_baseline, CAMERA_UP],
    },
}

import omni.replicator.core as rep

for name, cfg in stereo_cameras.items():
    create_prim(
        cfg["prim_path"],
        "Camera",
        position=cfg["position"],
        orientation=CAMERA_QUAT,
    )
    print(f"  {name} at {cfg['prim_path']}")

    # Create render product directly via replicator, bypassing Camera.initialize()
    # which fails due to SyntheticData pipeline timing issues
    rp = rep.create.render_product(cfg["prim_path"], CAMERA_RESOLUTION)
    cfg["render_prod"] = rp.path
    time.sleep(0.3)

# Stop the replicator orchestrator — our ROS2 writers are driven by
# OmniGraph simulation ticks, not by the orchestrator's on_update loop
# which crashes with "Unable to write from unknown dtype"
rep.orchestrator.stop()
simulation_app.update()

time.sleep(0.2)
print(f"  Resolution: {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]}")
print(f"  FPS: {CAMERA_FPS}")
print(f"  Baseline: {STEREO_BASELINE * 1000:.0f} mm")
print(f"  Attached to moving body - will follow UAV!\n")

# ------------------------------------------------------------------
# Create ROS2 Clock publisher for use_sim_time:=true
# ------------------------------------------------------------------
print("Creating ROS2 Clock publisher...")
try:
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
    print("  Publishing to /clock topic for use_sim_time:=true\n")
except Exception as e:
    print(f"  Could not create clock publisher: {str(e)[:80]}\n")

# ------------------------------------------------------------------
# Create ROS2 stereo image publishers
# ------------------------------------------------------------------
print("Creating ROS2 stereo image publishers...")
try:
    import omni.replicator.core as rep
    from isaacsim.ros2.bridge import read_camera_info

    for name, cfg in stereo_cameras.items():
        render_prod = cfg["render_prod"]

        # RGB image publisher
        writer_rgb = rep.writers.get("LdrColorSDROS2PublishImage")
        writer_rgb.initialize(
            nodeNamespace="drone0",
            topicName=f"{name}/color/image_raw",
            frameId=f"{name}_optical",
            queueSize=1,
        )
        writer_rgb.attach([render_prod])

        # Camera info publisher
        camera_info, _ = read_camera_info(render_product_path=render_prod)
        writer_info = rep.writers.get("ROS2PublishCameraInfo")
        writer_info.initialize(
            nodeNamespace="drone0",
            topicName=f"{name}/color/camera_info",
            frameId=f"{name}_optical",
            queueSize=1,
            width=camera_info.width,
            height=camera_info.height,
            projectionType=camera_info.distortion_model,
            k=camera_info.k.reshape([1, 9]),
            r=camera_info.r.reshape([1, 9]),
            p=camera_info.p.reshape([1, 12]),
            physicalDistortionModel=camera_info.distortion_model,
            physicalDistortionCoefficients=camera_info.d,
        )
        writer_info.attach([render_prod])

        print(f"  {name} -> drone0/{name}/color/image_raw")

    print("  Stereo image publishers ready\n")
except Exception as e:
    print(f"  Could not create image publishers: {e}\n")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("=" * 70)
print("  READY!")
print("=" * 70)
print(f"""
Pegasus vehicle with stereo cameras spawned (Curved Gridroom)
PX4 SITL starting...
ROS2 Clock publisher active (/clock topic)

ROS2 Topics:
  drone0/stereo_left/color/image_raw
  drone0/stereo_left/color/camera_info
  drone0/stereo_right/color/image_raw
  drone0/stereo_right/color/camera_info
  /clock

Launch ROS2 offboard control:
  ros2 launch px4_offboard_control offboard.launch.py use_sim_time:=true

Press Ctrl+C to exit
""")

# ------------------------------------------------------------------
# Run simulation
# ------------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()

try:
    timeline.play()
    time.sleep(1.0)

    while simulation_app.is_running():
        world.step(render=True)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    timeline.stop()
    simulation_app.close()
    print("Shutdown complete\n")
