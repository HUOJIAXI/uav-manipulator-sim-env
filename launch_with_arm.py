#!/usr/bin/env python3
"""
UAV Warehouse with PX4 + Stereo Camera + Cobotta Pro 900 Arm
Spawns Pegasus vehicle with stereo cameras and a physically-attached
Cobotta Pro 900 robotic arm controllable via ROS2 JointState topics.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import omni.timeline
import omni.usd
import omni.kit.commands
from omni.isaac.core.world import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.sensor import Camera
import omni.graph.core as og
from pxr import Gf, Sdf, UsdPhysics
import usdrt.Sdf

from pegasus.simulator.params import ROBOTS
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

# ---------------------------------------------------------------------------
# Cobotta Pro 900 arm parameters
# ---------------------------------------------------------------------------
COBOTTA_URDF = "/home/huojiaxi/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/cobotta_pro_900/cobotta_pro_900_drone.urdf"
ARM_ATTACH_OFFSET = Gf.Vec3f(0.0, 0.0, -0.05)  # 5 cm below drone body centre (arm is 1/10 scale)
ARM_ROOT = None  # set after import

print("\n" + "=" * 70)
print("  UAV Warehouse with PX4 + Stereo Camera + Cobotta Arm")
print("=" * 70 + "\n")

# ------------------------------------------------------------------
# Load warehouse
# ------------------------------------------------------------------
print("Loading warehouse...")
open_stage("/home/huojiaxi/Desktop/uav_sim/uav_warehouse.usd")
time.sleep(5.0)

# Remove stale cobotta prims BEFORE first update — these have broken variant
# payloads (referencing non-existent uav_warehouse_physics.usd etc.) that crash
# USD recomposition during simulation_app.update().
_stage = omni.usd.get_context().get_stage()
if _stage:
    for _stale in ["/cobotta_pro_900_drone", "/cobotta_pro_900"]:
        _p = _stage.GetPrimAtPath(_stale)
        if _p.IsValid():
            _stage.RemovePrim(_stale)
            print(f"  Removed stale prim: {_stale}")

time.sleep(45.0)

# Single update to process stage loading
simulation_app.update()
time.sleep(30.0)
print("  Loaded\n")

# ------------------------------------------------------------------
# Initialize Pegasus BEFORE searching for action graphs
# ------------------------------------------------------------------
print("Initializing Pegasus...")
try:
    pg = PegasusInterface()
    time.sleep(0.5)
    pg._world = World(**pg._world_settings)
    world = pg._world
    time.sleep(0.5)
    print("  Initialized\n")
except Exception as e:
    print(f"Error initializing Pegasus: {e}")
    raise

# ------------------------------------------------------------------
# Search for action graphs (after Pegasus is initialized)
# ------------------------------------------------------------------
print("Searching for action graphs...")
stage = omni.usd.get_context().get_stage()
action_graphs = []

for prim in stage.Traverse():
    try:
        prim_type = prim.GetTypeName()
        prim_name = prim.GetName().lower()
        if prim_type == "OmniGraph" or "actiongraph" in prim_name or "action_graph" in prim_name:
            graph_path = str(prim.GetPath())
            action_graphs.append(graph_path)
            print(f"  Found: {graph_path}")
    except Exception:
        continue

if action_graphs:
    print(f"  Will initialize {len(action_graphs)} graph(s) after camera creation\n")
else:
    print("  No action graphs found in USD\n")

# ------------------------------------------------------------------
# Clean old vehicles
# ------------------------------------------------------------------
print("Removing old vehicles...")
to_remove = []
for prim in stage.Traverse():
    try:
        path = str(prim.GetPath())
        name = prim.GetName().lower()
        if any(k in name for k in ["quadrotor", "drone", "iris"]):
            if path.count('/') == 2 and path.startswith('/World/'):
                to_remove.append(path)
    except Exception:
        continue

for path in to_remove:
    stage.RemovePrim(path)
    print(f"  Removed: {path}")
print("  Cleaned\n")

# ------------------------------------------------------------------
# Reset world
# ------------------------------------------------------------------
print("Resetting world...")
world.reset()
time.sleep(0.5)
print("  Ready\n")

# ------------------------------------------------------------------
# Configure PX4
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

# Spawn vehicle
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
# Add stereo cameras to the vehicle BODY (not root)
# ------------------------------------------------------------------
print("Adding stereo cameras to vehicle body...")

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

for name, cfg in stereo_cameras.items():
    # Create camera prim as child of quadrotor/body (the moving part)
    create_prim(
        cfg["prim_path"],
        "Camera",
        position=cfg["position"],
        orientation=CAMERA_QUAT,
    )
    print(f"  {name} at {cfg['prim_path']}")

    # Configure camera using Isaac Sim Camera API
    cam = Camera(
        prim_path=cfg["prim_path"],
        resolution=CAMERA_RESOLUTION,
        frequency=CAMERA_FPS,
    )
    cam.initialize()
    time.sleep(0.3)

    # Store render product path for later ROS2 publishing
    cfg["render_prod"] = cam._render_product_path

time.sleep(0.2)
print(f"  Resolution: {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]}")
print(f"  FPS: {CAMERA_FPS}")
print(f"  Baseline: {STEREO_BASELINE * 1000:.0f} mm")
print(f"  Attached to moving body - will follow UAV!\n")

# ------------------------------------------------------------------
# Import Cobotta Pro 900 arm and attach to drone
# ------------------------------------------------------------------
print("Importing Cobotta Pro 900 arm...")

try:
    # Enable URDF importer extension
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    time.sleep(1.0)

    # Create import configuration
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.fix_base = False           # not fixed to world — will hang from drone
    import_config.merge_fixed_joints = False  # keep base_link separate from world link
    import_config.create_physics_scene = False # scene already exists
    import_config.make_default_prim = False   # don't override stage default prim
    from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType
    import_config.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.set_default_drive_strength(1e4)          # joint stiffness
    import_config.set_default_position_drive_damping(1e3)  # joint damping

    # Import URDF into the current stage
    result, arm_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=COBOTTA_URDF,
        import_config=import_config,
    )
    print(f"  Imported at: {arm_path}")

    ARM_ROOT = arm_path

    simulation_app.update()
    time.sleep(0.5)

    # Find the articulation root using direct path access (avoid stage.Traverse
    # which can crash on stale prims from the warehouse USD)
    ARM_ART_ROOT = ARM_ROOT  # fallback: root prim itself
    for candidate in [ARM_ROOT, ARM_ROOT + "/world", ARM_ROOT + "/base_link"]:
        p = stage.GetPrimAtPath(candidate)
        if p.IsValid() and p.HasAPI(UsdPhysics.ArticulationRootAPI):
            ARM_ART_ROOT = candidate
            break
    print(f"  Articulation root: {ARM_ART_ROOT}")

    # Find the base_link prim (attachment point) using direct path access
    arm_base_link = None
    for candidate in [ARM_ROOT + "/base_link", ARM_ROOT + "/world/base_link"]:
        p = stage.GetPrimAtPath(candidate)
        if p.IsValid():
            arm_base_link = candidate
            break

    if not arm_base_link:
        raise RuntimeError(f"Could not find base_link under {ARM_ROOT}")
    print(f"  Base link: {arm_base_link}")

    # Create fixed joint: drone body <-> arm base_link
    print("  Creating fixed joint attachment...")
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/World/arm_attachment_joint"))
    fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/quadrotor/body")])
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(arm_base_link)])
    fixed_joint.CreateLocalPos0Attr().Set(ARM_ATTACH_OFFSET)
    # 180° rotation around Y so arm hangs downward (flips Z axis, gripper below drone)
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(0, 0, 1, 0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    print("  Fixed joint: /World/quadrotor/body -> arm base_link")

    # Lower stiffness/damping on gripper finger joint for softer grip control
    # Use direct path access instead of stage.Traverse()
    for finger_candidate in [ARM_ROOT + "/finger_joint", ARM_ROOT + "/world/finger_joint",
                              ARM_ROOT + "/base_link/finger_joint"]:
        fp = stage.GetPrimAtPath(finger_candidate)
        if fp.IsValid() and fp.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Get(fp, "angular")
            if drive:
                drive.GetStiffnessAttr().Set(1e3)
                drive.GetDampingAttr().Set(1e2)
                print(f"  Gripper drive adjusted: {finger_candidate}")
            break

    simulation_app.update()
    print("  Cobotta arm attached to drone!\n")

except Exception as e:
    print(f"  ERROR importing arm: {e}")
    import traceback
    traceback.print_exc()
    ARM_ROOT = None
    print()

# ------------------------------------------------------------------
# Initialize action graphs (cameras exist now)
# ------------------------------------------------------------------
if action_graphs:
    print(f"Initializing {len(action_graphs)} action graph(s)...")
    time.sleep(0.5)

    controller = og.Controller()

    for graph_path in action_graphs:
        try:
            graph = og.get_graph_by_path(graph_path)
            if graph:
                controller.evaluate_sync(graph)
                print(f"  Initialized: {graph_path}")
            else:
                print(f"  Could not get graph: {graph_path}")
        except Exception as e:
            print(f"  Failed to initialize {graph_path}: {str(e)[:80]}")

    simulation_app.update()
    time.sleep(0.5)
    print("  Action graphs initialized\n")

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
# Create ROS2 stereo image publishers (deferred to after clock graph
# so the ROS2 bridge extension is already loaded)
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
# Create ROS2 arm joint state publisher + subscriber/controller
# ------------------------------------------------------------------
if ARM_ROOT:
    print("Creating ROS2 arm joint control graph...")
    try:
        keys = og.Controller.Keys
        (arm_graph, arm_nodes, _, _) = og.Controller.edit(
            {
                "graph_path": "/World/ROS2_ArmControl",
                "evaluator_name": "execution",
            },
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                keys.SET_VALUES: [
                    ("PublishJointState.inputs:topicName", "drone0/arm/joint_states"),
                    ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ARM_ART_ROOT)]),
                    ("SubscribeJointState.inputs:topicName", "drone0/arm/joint_command"),
                    ("ArticulationController.inputs:targetPrim", [usdrt.Sdf.Path(ARM_ART_ROOT)]),
                ],
                keys.CONNECT: [
                    # Joint state publisher
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    # Joint command subscriber -> articulation controller
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ],
            },
        )
        print("  Publisher:  drone0/arm/joint_states")
        print("  Subscriber: drone0/arm/joint_command")
        print("  Arm joint control graph ready\n")
    except Exception as e:
        print(f"  Could not create arm control graph: {e}")
        import traceback
        traceback.print_exc()
        print()

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("=" * 70)
print("  READY!")
print("=" * 70)
print(f"""
Pegasus vehicle with stereo cameras and Cobotta arm spawned
PX4 SITL starting...
ROS2 Clock publisher active (/clock topic)

ROS2 Topics:
  drone0/stereo_left/color/image_raw      (stereo left RGB)
  drone0/stereo_left/color/camera_info    (stereo left camera info)
  drone0/stereo_right/color/image_raw     (stereo right RGB)
  drone0/stereo_right/color/camera_info   (stereo right camera info)
  drone0/arm/joint_states                 (arm joint positions/velocities)
  drone0/arm/joint_command                (send arm joint commands)
  /clock                                  (simulation clock)

Command arm joints:
  ros2 topic pub /drone0/arm/joint_command sensor_msgs/msg/JointState \\
    "{{name: ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6'], \\
     position: [0.0, -0.5, 0.5, 0.0, 0.5, 0.0]}}"

Read arm joint states:
  ros2 topic echo /drone0/arm/joint_states

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
