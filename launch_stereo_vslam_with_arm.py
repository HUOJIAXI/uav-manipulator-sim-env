#!/usr/bin/env python3
"""
UAV with PX4 + Stereo Camera (Default Pegasus Environment)
Uses iris_vslam.usd body with stereo cameras published via ROS2,
using a built-in Pegasus environment instead of a custom warehouse USD.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import omni.timeline
import omni.usd
from omni.isaac.core.world import World

# Enable ROS2 bridge extension early (before any ROS2/OmniGraph usage)
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

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
# UAV body USD
# ---------------------------------------------------------------------------
import os
IRIS_VSLAM_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "iris_vslam.usd")


print("\n" + "=" * 70)
print("  UAV (iris_vslam) with PX4 + Stereo Camera (Default Environment)")
print("=" * 70 + "\n")

# ------------------------------------------------------------------
# Initialize Pegasus and load default environment
# ------------------------------------------------------------------
print("Initializing Pegasus...")
pg = PegasusInterface()
pg._world = World(**pg._world_settings)
world = pg.world

print("Loading environment...")
pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
time.sleep(2.0)
simulation_app.update()
print("  Environment loaded\n")

# ------------------------------------------------------------------
# Configure PX4 and spawn vehicle
# ------------------------------------------------------------------
print("Creating Pegasus vehicle with PX4 (iris_vslam)...")
config = MultirotorConfig()
px4_cfg = PX4MavlinkBackendConfig({
    "vehicle_id": 0,
    "px4_autolaunch": True,
    "px4_dir": pg.px4_path,
    "px4_vehicle_model": pg.px4_default_airframe
})
config.backends = [PX4MavlinkBackend(px4_cfg)]
config.graphical_sensors = []  # Don't let Pegasus create sensors; USD has its own

drone = Multirotor(
    "/World/quadrotor",
    IRIS_VSLAM_USD,
    0,
    [0.0, 0.0, 0.30],
    Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
    config=config,
)
time.sleep(1.0)
print("  Vehicle created\n")

# ------------------------------------------------------------------
# Remove stale OmniGraph action graphs from the USD
# (iris_vslam.usd has graphs using old omni.isaac.* node types from Isaac Sim 4.x)
# ------------------------------------------------------------------
print("Cleaning stale OmniGraph graphs from USD...")
import omni.usd
stage = omni.usd.get_context().get_stage()
stale_graph_paths = []
for prim in stage.Traverse():
    if prim.GetTypeName() == "OmniGraph":
        stale_graph_paths.append(str(prim.GetPath()))
for gpath in stale_graph_paths:
    stage.RemovePrim(gpath)
    print(f"  Removed: {gpath}")
if not stale_graph_paths:
    print("  No stale graphs found")

# ------------------------------------------------------------------
# Create clean stereo camera prims on the vehicle body
# (the USD's RSD455 cameras have fisheye/aperture attributes incompatible
# with Isaac Sim 5.1's replicator pipeline, so we create fresh pinhole cameras)
# ------------------------------------------------------------------
print("Creating clean stereo camera prims...")
from pxr import UsdGeom, Gf
from omni.isaac.core.utils.prims import create_prim
from scipy.spatial.transform import Rotation as R

STEREO_BASELINE = 0.050    # 50 mm (matching RSD455 in the USD)
CAMERA_FORWARD = 0.10      # 10 cm forward from body centre
CAMERA_UP = 0.0
half_baseline = STEREO_BASELINE / 2.0

# Camera orientation: forward-looking on the drone body
# Isaac Sim cameras look along local -Z → rotate so -Z_cam aligns with +X_body (forward)
cam_quat = R.from_euler('xyz', [-90, 0, 90], degrees=True).as_quat()  # [x,y,z,w]

stereo_cam_paths = {}
for side, y_offset in [("left", half_baseline), ("right", -half_baseline)]:
    cam_path = f"/World/quadrotor/body/stereo_{side}"
    create_prim(
        cam_path,
        "Camera",
        position=[CAMERA_FORWARD, y_offset, CAMERA_UP],
        orientation=cam_quat,
    )
    stereo_cam_paths[side] = cam_path
    print(f"  {side}: {cam_path} (y={y_offset:+.4f})")

simulation_app.update()
print(f"  Baseline: {STEREO_BASELINE * 1000:.0f} mm\n")

# ------------------------------------------------------------------
# Reset world so articulations are initialized
# ------------------------------------------------------------------
print("Resetting world...")
world.reset()


stage = omni.usd.get_context().get_stage()

# ------------------------------------------------------------------
# Add landing legs below each rotor
# ------------------------------------------------------------------
print("Adding landing legs below rotors...")
from pxr import UsdGeom, UsdPhysics, Gf

LEG_LENGTH = 0.20   # 20 cm
LEG_RADIUS = 0.005  # 5 mm thin strut

# Rotor XY positions (from Iris SDF), Z at body bottom
ROTOR_POSITIONS = {
    "leg_front_right": (0.13, -0.22),
    "leg_front_left":  (0.13,  0.22),
    "leg_back_left":   (-0.13,  0.20),
    "leg_back_right":  (-0.13, -0.20),
}

# Place leg so its top touches the body origin (Z=0 of body frame)
LEG_Z = -LEG_LENGTH / 2.0  # center of leg cylinder

stage = omni.usd.get_context().get_stage()
body_path = "/World/quadrotor/body"

for leg_name, (rx, ry) in ROTOR_POSITIONS.items():
    leg_path = f"{body_path}/{leg_name}"
    leg_prim = UsdGeom.Cylinder.Define(stage, leg_path)
    leg_prim.GetHeightAttr().Set(LEG_LENGTH)
    leg_prim.GetRadiusAttr().Set(LEG_RADIUS)
    leg_prim.GetAxisAttr().Set("Z")

    # Set local transform
    xform = UsdGeom.Xformable(leg_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(rx, ry, LEG_Z))

    # Add collision
    UsdPhysics.CollisionAPI.Apply(leg_prim.GetPrim())

    # Dark gray appearance
    leg_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])

    print(f"  {leg_name} at ({rx}, {ry}, {LEG_Z:.3f})")

simulation_app.update()
print("  Landing legs added\n")

# ------------------------------------------------------------------
# Add 2-DOF folding arm as separate articulation, attached via FixedJoint
# ------------------------------------------------------------------
print("Adding 2-DOF folding arm...")
from pxr import Sdf

ARM_LINK_LENGTH = 0.125      # 12.5 cm per link
ARM_LINK_RADIUS = 0.008      # 8 mm radius
ARM_LINK_MASS = 0.05          # 50 g per link
ARM_HALF_LEN = ARM_LINK_LENGTH / 2.0  # 0.0625 m
SHOULDER_MOUNT_Z = -0.08     # mount point below body bottom (~0.055 half-height + margin)
JOINT_STIFFNESS = 1e5
JOINT_DAMPING = 1e4

arm_root = "/World/folding_arm"
DRONE_Z = 0.30  # must match drone spawn height
MOUNT_WX, MOUNT_WY, MOUNT_WZ = 0.0, 0.0, DRONE_Z + SHOULDER_MOUNT_Z

UsdGeom.Xform.Define(stage, arm_root)

# Base link
base_path = f"{arm_root}/base_link"
base_xform = UsdGeom.Xform.Define(stage, base_path)
bx = UsdGeom.Xformable(base_xform)
bx.ClearXformOpOrder()
bx.AddTranslateOp().Set(Gf.Vec3d(MOUNT_WX, MOUNT_WY, MOUNT_WZ))
base_prim = base_xform.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(base_prim)
mass_base = UsdPhysics.MassAPI.Apply(base_prim)
mass_base.GetMassAttr().Set(0.01)

# Upper link
upper_path = f"{arm_root}/upper_link"
upper_capsule = UsdGeom.Capsule.Define(stage, upper_path)
upper_capsule.GetHeightAttr().Set(ARM_LINK_LENGTH)
upper_capsule.GetRadiusAttr().Set(ARM_LINK_RADIUS)
upper_capsule.GetAxisAttr().Set("Z")
upper_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.2)])

ux = UsdGeom.Xformable(upper_capsule)
ux.ClearXformOpOrder()
ux.AddTranslateOp().Set(Gf.Vec3d(MOUNT_WX + ARM_HALF_LEN, MOUNT_WY, MOUNT_WZ))
ux.AddOrientOp().Set(Gf.Quatf(0.7071, 0.0, -0.7071, 0.0))

upper_prim = upper_capsule.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(upper_prim)
mass_upper = UsdPhysics.MassAPI.Apply(upper_prim)
mass_upper.GetMassAttr().Set(ARM_LINK_MASS)

# Lower link
lower_path = f"{arm_root}/lower_link"
lower_capsule = UsdGeom.Capsule.Define(stage, lower_path)
lower_capsule.GetHeightAttr().Set(ARM_LINK_LENGTH)
lower_capsule.GetRadiusAttr().Set(ARM_LINK_RADIUS)
lower_capsule.GetAxisAttr().Set("Z")
lower_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.8)])

lx = UsdGeom.Xformable(lower_capsule)
lx.ClearXformOpOrder()
lx.AddTranslateOp().Set(Gf.Vec3d(MOUNT_WX + ARM_HALF_LEN, MOUNT_WY, MOUNT_WZ - 2 * ARM_LINK_RADIUS))
lx.AddOrientOp().Set(Gf.Quatf(0.7071, 0.0, 0.7071, 0.0))

lower_prim = lower_capsule.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(lower_prim)
mass_lower = UsdPhysics.MassAPI.Apply(lower_prim)
mass_lower.GetMassAttr().Set(ARM_LINK_MASS)

simulation_app.update()
print("  Arm link prims created")

# FixedJoint: drone body <-> arm base_link
attach_joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{arm_root}/attach_to_body"))
attach_joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
attach_joint.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])
attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, SHOULDER_MOUNT_Z))
attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
attach_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
attach_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
print("  Fixed joint: drone body -> arm base_link")

# Shoulder revolute joint
shoulder_path = f"{arm_root}/shoulder_joint"
shoulder_joint = UsdPhysics.RevoluteJoint.Define(stage, shoulder_path)
shoulder_joint.GetAxisAttr().Set("Y")
shoulder_joint.GetLowerLimitAttr().Set(-90.0)
shoulder_joint.GetUpperLimitAttr().Set(90.0)
shoulder_joint.GetBody0Rel().SetTargets([base_path])
shoulder_joint.GetBody1Rel().SetTargets([upper_path])
shoulder_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
shoulder_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, ARM_HALF_LEN))
shoulder_joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
shoulder_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

shoulder_drive = UsdPhysics.DriveAPI.Apply(shoulder_joint.GetPrim(), "angular")
shoulder_drive.GetTypeAttr().Set("force")
shoulder_drive.GetStiffnessAttr().Set(JOINT_STIFFNESS)
shoulder_drive.GetDampingAttr().Set(JOINT_DAMPING)
shoulder_drive.GetTargetPositionAttr().Set(-90.0)
print("  Shoulder joint created (target: -90 deg)")

# Elbow revolute joint
elbow_path = f"{arm_root}/elbow_joint"
elbow_joint = UsdPhysics.RevoluteJoint.Define(stage, elbow_path)
elbow_joint.GetAxisAttr().Set("Y")
elbow_joint.GetLowerLimitAttr().Set(0.0)
elbow_joint.GetUpperLimitAttr().Set(180.0)
elbow_joint.GetBody0Rel().SetTargets([upper_path])
elbow_joint.GetBody1Rel().SetTargets([lower_path])
elbow_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -ARM_HALF_LEN))
elbow_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, ARM_HALF_LEN))
elbow_joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
elbow_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

elbow_drive = UsdPhysics.DriveAPI.Apply(elbow_joint.GetPrim(), "angular")
elbow_drive.GetTypeAttr().Set("force")
elbow_drive.GetStiffnessAttr().Set(JOINT_STIFFNESS)
elbow_drive.GetDampingAttr().Set(JOINT_DAMPING)
elbow_drive.GetTargetPositionAttr().Set(180.0)

simulation_app.update()
print("  Elbow joint created (target: 180 deg)")
print("  2-DOF folding arm ready (folded state)\n")

# ------------------------------------------------------------------
# Set up ROS2 arm joint control via direct DriveAPI target updates
# ------------------------------------------------------------------
print("Setting up ROS2 arm joint control (direct drive)...")
import rclpy
from rclpy.node import Node as RclpyNode
from sensor_msgs.msg import JointState
import math, threading

arm_joint_drives = {
    "shoulder_joint": UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath(shoulder_path), "angular"
    ),
    "elbow_joint": UsdPhysics.DriveAPI.Get(
        stage.GetPrimAtPath(elbow_path), "angular"
    ),
}

arm_cmd_lock = threading.Lock()
arm_cmd_targets = {}


class ArmBridgeNode(RclpyNode):
    def __init__(self):
        super().__init__("arm_bridge")
        self.sub = self.create_subscription(
            JointState, "drone0/arm/joint_command", self._cmd_cb, 10
        )
        self.pub = self.create_publisher(JointState, "drone0/arm/joint_states", 10)

    def _cmd_cb(self, msg):
        with arm_cmd_lock:
            for i, name in enumerate(msg.name):
                if name in arm_joint_drives and i < len(msg.position):
                    arm_cmd_targets[name] = math.degrees(msg.position[i])

    def publish_states(self, stamp_sec):
        msg = JointState()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.name = list(arm_joint_drives.keys())
        msg.position = []
        for name, drive in arm_joint_drives.items():
            target = drive.GetTargetPositionAttr().Get()
            msg.position.append(math.radians(target if target else 0.0))
        self.pub.publish(msg)


rclpy.init()
arm_bridge = ArmBridgeNode()
arm_spin_thread = threading.Thread(
    target=lambda: rclpy.spin(arm_bridge), daemon=True
)
arm_spin_thread.start()
print("  ROS2 arm bridge node running")
print("  Subscriber: drone0/arm/joint_command")
print("  Publisher:  drone0/arm/joint_states\n")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("=" * 70)
print("  READY!")
print("=" * 70)
print(f"""
Pegasus vehicle (iris_vslam) with folding arm spawned (Curved Gridroom)
PX4 SITL starting...

ROS2 Topics:
  front_stereo_camera/left/image_rect_color   (left stereo image)
  front_stereo_camera/left/camera_info        (left camera info)
  front_stereo_camera/right/image_rect_color  (right stereo image)
  front_stereo_camera/right/camera_info       (right camera info)
  drone0/arm/joint_states                     (arm joint positions)
  drone0/arm/joint_command                    (send arm joint commands)

Control arm joints:
  ros2 run flexible_arm_controller arm_gui
  ros2 topic pub --once drone0/arm/joint_command sensor_msgs/msg/JointState \\
    "{{name: ['shoulder_joint','elbow_joint'], position: [0.0, 0.0]}}"

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

    # ------------------------------------------------------------------
    # Set up stereo camera publishers via viewport capture + ROS2
    # (uses viewport schedule_capture to bypass broken SyntheticData)
    # ------------------------------------------------------------------
    print("Setting up ROS2 stereo camera publishers (viewport capture)...")
    import omni.kit.viewport.utility as vp_utils
    from omni.kit.widget.viewport.capture import ByteCapture
    from sensor_msgs.msg import Image, CameraInfo as CameraInfoMsg
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Time as TimeMsg
    import numpy as np

    STEREO_CAM_RESOLUTION = (640, 480)
    cam_viewports = {}

    for side, cam_prim_path in stereo_cam_paths.items():
        vp_name = f"stereo_{side}"
        vp_window = vp_utils.create_viewport_window(
            vp_name,
            width=STEREO_CAM_RESOLUTION[0],
            height=STEREO_CAM_RESOLUTION[1],
            visible=False,
        )
        vp_api = vp_window.viewport_api
        vp_api.set_active_camera(cam_prim_path)

        cam_viewports[side] = {
            "viewport_api": vp_api,
            "viewport_window": vp_window,
            "latest_frame": None,  # will be filled by capture callback
        }
        print(f"  {side}: {cam_prim_path} -> viewport '{vp_name}'")

    # Warm up renderer
    print("  Warming up renderer...")
    for _ in range(10):
        world.step(render=True)

    # Create ROS2 publishers
    class StereoCamPublisher(RclpyNode):
        def __init__(self):
            super().__init__("stereo_cam_publisher")
            self.img_pubs = {}
            self.info_pubs = {}
            for side in cam_viewports:
                self.img_pubs[side] = self.create_publisher(
                    Image, f"front_stereo_camera/{side}/image_rect_color", 1
                )
                self.info_pubs[side] = self.create_publisher(
                    CameraInfoMsg, f"front_stereo_camera/{side}/camera_info", 1
                )

        def capture_and_publish(self, stamp_sec):
            stamp = TimeMsg()
            stamp.sec = int(stamp_sec)
            stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)

            for side, vp_data in cam_viewports.items():
                vp_api = vp_data["viewport_api"]
                img_pub = self.img_pubs[side]
                info_pub = self.info_pubs[side]
                frame_id = f"camera_{side}_optical"

                def _on_capture(buffer, buffer_size, width, height, fmt,
                                _pub=img_pub, _info_pub=info_pub,
                                _stamp=stamp, _frame_id=frame_id):
                    if buffer is None or buffer_size == 0:
                        return
                    try:
                        import ctypes
                        # buffer is a PyCapsule; extract raw bytes via ctypes
                        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                        ptr = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
                        raw_bytes = (ctypes.c_ubyte * buffer_size).from_address(ptr)

                        header = Header()
                        header.stamp = _stamp
                        header.frame_id = _frame_id

                        img_msg = Image()
                        img_msg.header = header
                        img_msg.height = height
                        img_msg.width = width
                        img_msg.encoding = "rgba8"
                        img_msg.step = width * 4
                        img_msg.is_bigendian = False
                        img_msg.data = bytes(raw_bytes)
                        _pub.publish(img_msg)

                        info_msg = CameraInfoMsg()
                        info_msg.header = header
                        info_msg.height = height
                        info_msg.width = width
                        info_msg.distortion_model = "plumb_bob"
                        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
                        fx = float(width)
                        fy = fx
                        cx = width / 2.0
                        cy = height / 2.0
                        info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
                        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                        info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
                        _info_pub.publish(info_msg)
                    except Exception as e:
                        carb.log_warn(f"Capture publish error: {e}")

                vp_api.schedule_capture(ByteCapture(_on_capture))

    stereo_pub = StereoCamPublisher()
    print("  Stereo camera publishers ready")
    print("    -> front_stereo_camera/left/image_rect_color")
    print("    -> front_stereo_camera/left/camera_info")
    print("    -> front_stereo_camera/right/image_rect_color")
    print("    -> front_stereo_camera/right/camera_info\n")

    pub_counter = 0
    while simulation_app.is_running():
        world.step(render=True)

        # Apply any pending arm joint commands from ROS2
        with arm_cmd_lock:
            for name, target_deg in arm_cmd_targets.items():
                drive = arm_joint_drives.get(name)
                if drive:
                    drive.GetTargetPositionAttr().Set(float(target_deg))
            arm_cmd_targets.clear()

        # Publish at ~10 Hz (every 6 steps at 60 Hz sim)
        pub_counter += 1
        if pub_counter >= 6:
            pub_counter = 0
            sim_time = world.current_time if hasattr(world, 'current_time') else 0.0
            arm_bridge.publish_states(sim_time)
            stereo_pub.capture_and_publish(sim_time)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    timeline.stop()
    stereo_pub.destroy_node()
    arm_bridge.destroy_node()
    rclpy.try_shutdown()
    simulation_app.close()
    print("Shutdown complete\n")
