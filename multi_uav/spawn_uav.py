"""Factory functions for spawning UAV-manipulators in Isaac Sim."""

import os
import math
import threading
import ctypes

import carb
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
from omni.isaac.core.utils.prims import create_prim

from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

import rclpy
from rclpy.node import Node as RclpyNode
from sensor_msgs.msg import JointState, Image, CameraInfo as CameraInfoMsg
from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header

# Camera constants (matching iris_vslam.usd RSD455)
STEREO_BASELINE = 0.050       # 50 mm
CAMERA_FORWARD = 0.10         # 10 cm forward from body centre
CAMERA_UP = 0.0

# Landing leg constants
LEG_LENGTH = 0.20             # 20 cm
LEG_RADIUS = 0.005            # 5 mm
ROTOR_POSITIONS = {
    "leg_front_right": (0.13, -0.22),
    "leg_front_left":  (0.13,  0.22),
    "leg_back_left":   (-0.13,  0.20),
    "leg_back_right":  (-0.13, -0.20),
}

# Arm constants
ARM_LINK_LENGTH = 0.125       # 12.5 cm per link
ARM_LINK_RADIUS = 0.008       # 8 mm radius
ARM_LINK_MASS = 0.05          # 50 g per link
ARM_HALF_LEN = ARM_LINK_LENGTH / 2.0
SHOULDER_MOUNT_Z = -0.08      # mount point below body
JOINT_STIFFNESS = 1e5
JOINT_DAMPING = 1e4

# Camera orientation: forward-looking on drone body
# Isaac Sim cameras look along local -Z -> rotate so -Z_cam aligns with +X_body
CAM_QUAT = Rotation.from_euler('xyz', [-90, 0, 90], degrees=True).as_quat()


def spawn_uav(stage, world, drone_cfg, spawn_height, assets_dir, sim_app):
    """
    Spawns a UAV body (Multirotor, cameras) before world.reset().

    Args:
        stage: USD stage
        world: Isaac Sim World
        drone_cfg: dict from YAML config for this drone
        spawn_height: float, Z height for spawning
        assets_dir: path to assets/ folder (for iris_vslam.usd)
        sim_app: SimulationApp instance

    Returns:
        dict with keys: "drone_id", "multirotor", "stereo_cam_paths"
    """
    drone_id = drone_cfg["id"]
    prefix = f"/World/drone{drone_id}"
    usd_path = os.path.join(assets_dir, "iris_vslam.usd")

    # --- 1. Create Multirotor ---
    x = drone_cfg.get("x", 0.0)
    y = drone_cfg.get("y", 0.0)
    yaw = drone_cfg.get("yaw", 0.0)

    pg = PegasusInterface()
    config = MultirotorConfig()
    px4_params = {
        "vehicle_id": drone_cfg.get("px4_vehicle_id", drone_id),
        "px4_autolaunch": drone_cfg.get("px4_autolaunch", True),
        "px4_dir": pg.px4_path,
        "px4_vehicle_model": pg.px4_default_airframe,
    }
    if "px4_sim_port" in drone_cfg:
        px4_params["px4_sim_port"] = drone_cfg["px4_sim_port"]
    if "px4_mavlink_port" in drone_cfg:
        px4_params["px4_mavlink_port"] = drone_cfg["px4_mavlink_port"]

    px4_cfg = PX4MavlinkBackendConfig(px4_params)
    config.backends = [PX4MavlinkBackend(px4_cfg)]
    config.graphical_sensors = []

    orientation = Rotation.from_euler("XYZ", [0.0, 0.0, yaw], degrees=True).as_quat()

    multirotor = Multirotor(
        f"{prefix}/quadrotor",
        usd_path,
        drone_id,
        [x, y, spawn_height],
        orientation,
        config=config,
    )
    sim_app.update()
    print(f"  drone{drone_id}: Multirotor created at ({x}, {y}, {spawn_height}), yaw={yaw}")

    # --- 2. Clean stale OmniGraphs ---
    stale_paths = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(f"{prefix}/") and prim.GetTypeName() == "OmniGraph":
            stale_paths.append(prim_path)
    for gpath in stale_paths:
        stage.RemovePrim(gpath)
        print(f"  drone{drone_id}: Removed stale graph: {gpath}")

    # --- 3. Stereo camera prims ---
    stereo_cam_paths = None
    if drone_cfg.get("stereo_camera", True):
        half_baseline = STEREO_BASELINE / 2.0
        stereo_cam_paths = {}
        body_path = f"{prefix}/quadrotor/body"
        for side, y_offset in [("left", half_baseline), ("right", -half_baseline)]:
            cam_path = f"{body_path}/stereo_{side}"
            create_prim(
                cam_path,
                "Camera",
                position=[CAMERA_FORWARD, y_offset, CAMERA_UP],
                orientation=CAM_QUAT,
            )
            stereo_cam_paths[side] = cam_path
            print(f"  drone{drone_id}: stereo_{side} at {cam_path}")
        sim_app.update()

    return {
        "drone_id": drone_id,
        "multirotor": multirotor,
        "stereo_cam_paths": stereo_cam_paths,
    }


def create_arm(stage, drone_cfg, spawn_height, sim_app):
    """
    Creates landing legs and 2-DOF folding arm for a drone after world.reset().

    Args:
        stage: USD stage
        drone_cfg: dict from YAML config for this drone
        spawn_height: float, Z height
        sim_app: SimulationApp instance

    Returns:
        dict with keys: "arm_drives" (or None), "shoulder_path", "elbow_path"
    """
    drone_id = drone_cfg["id"]
    prefix = f"/World/drone{drone_id}"
    body_path = f"{prefix}/quadrotor/body"
    dx = drone_cfg.get("x", 0.0)
    dy = drone_cfg.get("y", 0.0)

    # --- Landing legs ---
    leg_z = -LEG_LENGTH / 2.0
    for leg_name, (rx, ry) in ROTOR_POSITIONS.items():
        leg_path = f"{body_path}/{leg_name}"
        leg_prim = UsdGeom.Cylinder.Define(stage, leg_path)
        leg_prim.GetHeightAttr().Set(LEG_LENGTH)
        leg_prim.GetRadiusAttr().Set(LEG_RADIUS)
        leg_prim.GetAxisAttr().Set("Z")

        xform = UsdGeom.Xformable(leg_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(rx, ry, leg_z))

        UsdPhysics.CollisionAPI.Apply(leg_prim.GetPrim())
        leg_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])

    sim_app.update()
    print(f"  drone{drone_id}: Landing legs added")

    # --- Arm (if enabled) ---
    arm_enabled = drone_cfg.get("arm", True)
    if not arm_enabled:
        return {"arm_drives": None, "shoulder_path": None, "elbow_path": None}

    arm_root = f"{prefix}/folding_arm"
    mount_wx = dx
    mount_wy = dy
    mount_wz = spawn_height + SHOULDER_MOUNT_Z

    UsdGeom.Xform.Define(stage, arm_root)

    # Base link
    base_path = f"{arm_root}/base_link"
    base_xform = UsdGeom.Xform.Define(stage, base_path)
    bx = UsdGeom.Xformable(base_xform)
    bx.ClearXformOpOrder()
    bx.AddTranslateOp().Set(Gf.Vec3d(mount_wx, mount_wy, mount_wz))
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
    ux.AddTranslateOp().Set(Gf.Vec3d(mount_wx + ARM_HALF_LEN, mount_wy, mount_wz))
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
    lx.AddTranslateOp().Set(
        Gf.Vec3d(mount_wx + ARM_HALF_LEN, mount_wy, mount_wz - 2 * ARM_LINK_RADIUS)
    )
    lx.AddOrientOp().Set(Gf.Quatf(0.7071, 0.0, 0.7071, 0.0))

    lower_prim = lower_capsule.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(lower_prim)
    mass_lower = UsdPhysics.MassAPI.Apply(lower_prim)
    mass_lower.GetMassAttr().Set(ARM_LINK_MASS)

    sim_app.update()

    # FixedJoint: drone body <-> arm base_link
    attach_joint = UsdPhysics.FixedJoint.Define(
        stage, Sdf.Path(f"{arm_root}/attach_to_body")
    )
    attach_joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
    attach_joint.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])
    attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, SHOULDER_MOUNT_Z))
    attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    attach_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    attach_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

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

    sim_app.update()
    print(f"  drone{drone_id}: 2-DOF arm created (folded)")

    arm_drives = {
        "shoulder_joint": UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(shoulder_path), "angular"
        ),
        "elbow_joint": UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(elbow_path), "angular"
        ),
    }

    return {
        "arm_drives": arm_drives,
        "shoulder_path": shoulder_path,
        "elbow_path": elbow_path,
    }


class ArmBridgeNode(RclpyNode):
    """Per-drone ROS2 node for arm joint command/state bridging."""

    def __init__(self, drone_id, arm_drives):
        super().__init__(f"drone{drone_id}_arm_bridge")
        self.arm_drives = arm_drives
        self.cmd_lock = threading.Lock()
        self.cmd_targets = {}

        prefix = f"drone{drone_id}"
        self.sub = self.create_subscription(
            JointState, f"{prefix}/arm/joint_command", self._cmd_cb, 10
        )
        self.pub = self.create_publisher(
            JointState, f"{prefix}/arm/joint_states", 10
        )

    def _cmd_cb(self, msg):
        with self.cmd_lock:
            for i, name in enumerate(msg.name):
                if name in self.arm_drives and i < len(msg.position):
                    self.cmd_targets[name] = math.degrees(msg.position[i])

    def apply_commands(self):
        """Apply pending joint commands to USD drives. Call from sim loop."""
        with self.cmd_lock:
            for name, target_deg in self.cmd_targets.items():
                drive = self.arm_drives.get(name)
                if drive:
                    drive.GetTargetPositionAttr().Set(float(target_deg))
            self.cmd_targets.clear()

    def publish_states(self, stamp_sec):
        msg = JointState()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.name = list(self.arm_drives.keys())
        msg.position = []
        for name, drive in self.arm_drives.items():
            target = drive.GetTargetPositionAttr().Get()
            msg.position.append(math.radians(target if target else 0.0))
        self.pub.publish(msg)


class StereoCamPublisher(RclpyNode):
    """Per-drone ROS2 node for stereo camera image publishing."""

    def __init__(self, drone_id):
        super().__init__(f"drone{drone_id}_stereo_cam_publisher")
        self.drone_id = drone_id
        self.cam_viewports = None  # injected after timeline.play()

        prefix = f"drone{drone_id}"
        self.img_pubs = {}
        self.info_pubs = {}
        for side in ("left", "right"):
            self.img_pubs[side] = self.create_publisher(
                Image, f"{prefix}/front_stereo_camera/{side}/image_rect_color", 1
            )
            self.info_pubs[side] = self.create_publisher(
                CameraInfoMsg, f"{prefix}/front_stereo_camera/{side}/camera_info", 1
            )

    def set_viewports(self, cam_viewports):
        """Inject viewport references after timeline.play()."""
        self.cam_viewports = cam_viewports

    def capture_and_publish(self, stamp_sec):
        if self.cam_viewports is None:
            return

        from omni.kit.widget.viewport.capture import ByteCapture

        stamp = TimeMsg()
        stamp.sec = int(stamp_sec)
        stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)

        for side, vp_data in self.cam_viewports.items():
            vp_api = vp_data["viewport_api"]
            img_pub = self.img_pubs[side]
            info_pub = self.info_pubs[side]
            frame_id = f"drone{self.drone_id}_camera_{side}_optical"

            def _on_capture(buffer, buffer_size, width, height, fmt,
                            _pub=img_pub, _info_pub=info_pub,
                            _stamp=stamp, _frame_id=frame_id):
                if buffer is None or buffer_size == 0:
                    return
                try:
                    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
                        ctypes.py_object, ctypes.c_char_p
                    ]
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
                    carb.log_warn(f"Capture publish error (drone{self.drone_id}): {e}")

            vp_api.schedule_capture(ByteCapture(_on_capture))
