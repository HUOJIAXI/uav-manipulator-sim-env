"""Factory functions for spawning UAV-manipulators in Isaac Sim."""

import os
import math
import threading
import ctypes

import numpy as np
import carb
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
from omni.isaac.core.utils.prims import create_prim

from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

import rclpy
from rclpy.node import Node as RclpyNode
from sensor_msgs.msg import JointState, Image, CameraInfo as CameraInfoMsg
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header

# Camera constants (matching iris_vslam.usd RSD455)
STEREO_BASELINE = 0.050       # 50 mm
CAMERA_FORWARD = 0.10         # 10 cm forward from body centre
CAMERA_UP = 0.05              # 5 cm above body centre

# Downward camera constants
DOWNWARD_CAM_Z = -0.05            # 5 cm below body centre
# Downward-facing: camera -Z should point along body -Z (down).
# With identity rotation, camera axes = body axes, so -Z_cam = -Z_body = down.
# Rotate 90° around Z so image top = body forward (+X).
DOWNWARD_CAM_QUAT = Rotation.from_euler('z', -90, degrees=True).as_quat()

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
# Isaac Sim cameras look along local -Z with +Y up (OpenGL convention).
# Rotate so: -Z_cam -> +X_body (forward), +Y_cam -> +Z_body (up), +X_cam -> -Y_body (right)
CAM_QUAT = Rotation.from_euler('xyz', [90, 0, -90], degrees=True).as_quat()


def spawn_uav(stage, world, drone_cfg, spawn_height, assets_dir, sim_app):
    """
    Spawns a UAV body (Multirotor only) before world.reset().

    Args:
        stage: USD stage
        world: Isaac Sim World
        drone_cfg: dict from YAML config for this drone
        spawn_height: float, Z height for spawning
        assets_dir: path to assets/ folder (for iris_vslam.usd)
        sim_app: SimulationApp instance

    Returns:
        dict with keys: "drone_id", "multirotor"
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
        "enable_lockstep": True,
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

    return {
        "drone_id": drone_id,
        "multirotor": multirotor,
    }


def create_stereo_cameras(stage, drone_cfg, spawn_height, sim_app):
    """
    Creates stereo camera prims on a drone. Must be called AFTER world.reset().

    Cameras are children of the drone body prim so they follow it via
    USD hierarchy — no separate rigid bodies or joints needed.

    Args:
        stage: USD stage
        drone_cfg: dict from YAML config for this drone
        spawn_height: float, Z height for spawning
        sim_app: SimulationApp instance

    Returns:
        dict mapping side name to cam prim path, or None if disabled
    """
    drone_id = drone_cfg["id"]
    if not drone_cfg.get("stereo_camera", True):
        return None

    prefix = f"/World/drone{drone_id}"
    body_path = f"{prefix}/quadrotor/body"
    half_baseline = STEREO_BASELINE / 2.0
    stereo_cam_paths = {}

    # Camera orientation: scipy [x,y,z,w] -> Gf.Quatf(w,x,y,z)
    cam_q = CAM_QUAT
    cam_quat_gf = Gf.Quatf(float(cam_q[3]), float(cam_q[0]), float(cam_q[1]), float(cam_q[2]))

    for side, y_offset in [("left", half_baseline), ("right", -half_baseline)]:
        # Mount as child of body — inherits body transform automatically
        mount_path = f"{body_path}/stereo_{side}"
        mount_xform = UsdGeom.Xform.Define(stage, mount_path)
        mx = UsdGeom.Xformable(mount_xform)
        mx.ClearXformOpOrder()
        # Local offset relative to body (body-frame coordinates, no yaw needed)
        mx.AddTranslateOp().Set(Gf.Vec3d(CAMERA_FORWARD, y_offset, CAMERA_UP))

        # Camera prim with optical-frame orientation
        cam_path = f"{mount_path}/camera"
        cam_prim = stage.DefinePrim(cam_path, "Camera")
        cam_xf = UsdGeom.Xformable(cam_prim)
        cam_xf.ClearXformOpOrder()
        cam_xf.AddOrientOp().Set(cam_quat_gf)

        stereo_cam_paths[side] = cam_path
        print(f"  drone{drone_id}: stereo_{side} at {cam_path}")

    sim_app.update()

    return stereo_cam_paths


def create_downward_camera(stage, drone_cfg, spawn_height, sim_app):
    """
    Creates a downward-facing camera under a drone. Must be called AFTER world.reset().

    Camera is a child of the drone body prim so it follows via USD hierarchy.

    Args:
        stage: USD stage
        drone_cfg: dict from YAML config for this drone
        spawn_height: float, Z height for spawning
        sim_app: SimulationApp instance

    Returns:
        str cam prim path, or None if disabled
    """
    drone_id = drone_cfg["id"]
    if not drone_cfg.get("downward_camera", True):
        return None

    prefix = f"/World/drone{drone_id}"
    body_path = f"{prefix}/quadrotor/body"

    # Mount as child of body
    mount_path = f"{body_path}/downward_cam"
    mount_xform = UsdGeom.Xform.Define(stage, mount_path)
    mx = UsdGeom.Xformable(mount_xform)
    mx.ClearXformOpOrder()
    mx.AddTranslateOp().Set(Gf.Vec3d(0, 0, DOWNWARD_CAM_Z))

    # Camera prim with downward orientation
    cam_q = DOWNWARD_CAM_QUAT
    cam_quat_gf = Gf.Quatf(float(cam_q[3]), float(cam_q[0]), float(cam_q[1]), float(cam_q[2]))

    cam_path = f"{mount_path}/camera"
    cam_prim = stage.DefinePrim(cam_path, "Camera")
    cam_xf = UsdGeom.Xformable(cam_prim)
    cam_xf.ClearXformOpOrder()
    cam_xf.AddOrientOp().Set(cam_quat_gf)

    sim_app.update()
    print(f"  drone{drone_id}: downward camera at {cam_path}")

    return cam_path


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
    yaw = drone_cfg.get("yaw", 0.0)
    yaw_rot = Rotation.from_euler("Z", yaw, degrees=True)

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
    mount_wz = spawn_height + SHOULDER_MOUNT_Z

    UsdGeom.Xform.Define(stage, arm_root)

    # Arm starts at shoulder angle=-90° (deployed forward).
    # Positions and orientations are yaw-aware so the initial joint angle
    # matches the drive target exactly — no correction needed.
    upper_local = np.array([ARM_HALF_LEN, 0.0, 0.0])
    upper_world = yaw_rot.apply(upper_local)
    upper_orient = yaw_rot * Rotation.from_quat([0.0, -0.7071068, 0.0, 0.7071068])
    uq = upper_orient.as_quat()
    upper_quat_gf = Gf.Quatf(float(uq[3]), float(uq[0]), float(uq[1]), float(uq[2]))

    upper_path = f"{arm_root}/upper_link"
    upper_capsule = UsdGeom.Capsule.Define(stage, upper_path)
    upper_capsule.GetHeightAttr().Set(ARM_LINK_LENGTH)
    upper_capsule.GetRadiusAttr().Set(ARM_LINK_RADIUS)
    upper_capsule.GetAxisAttr().Set("Z")
    upper_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.2)])

    ux = UsdGeom.Xformable(upper_capsule)
    ux.ClearXformOpOrder()
    ux.AddTranslateOp().Set(Gf.Vec3d(
        dx + upper_world[0], dy + upper_world[1], mount_wz + upper_world[2]
    ))
    ux.AddOrientOp().Set(upper_quat_gf)

    upper_prim = upper_capsule.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(upper_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(upper_prim).GetSleepThresholdAttr().Set(0.0)
    mass_upper = UsdPhysics.MassAPI.Apply(upper_prim)
    mass_upper.GetMassAttr().Set(ARM_LINK_MASS)

    # Lower link — folded back on upper (elbow=180°)
    lower_orient = yaw_rot * Rotation.from_quat([0.0, 0.7071068, 0.0, 0.7071068])
    lq = lower_orient.as_quat()
    lower_quat_gf = Gf.Quatf(float(lq[3]), float(lq[0]), float(lq[1]), float(lq[2]))

    lower_path = f"{arm_root}/lower_link"
    lower_capsule = UsdGeom.Capsule.Define(stage, lower_path)
    lower_capsule.GetHeightAttr().Set(ARM_LINK_LENGTH)
    lower_capsule.GetRadiusAttr().Set(ARM_LINK_RADIUS)
    lower_capsule.GetAxisAttr().Set("Z")
    lower_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.8)])

    lx = UsdGeom.Xformable(lower_capsule)
    lx.ClearXformOpOrder()
    lx.AddTranslateOp().Set(Gf.Vec3d(
        dx + upper_world[0], dy + upper_world[1], mount_wz + upper_world[2] - 2 * ARM_LINK_RADIUS
    ))
    lx.AddOrientOp().Set(lower_quat_gf)

    lower_prim = lower_capsule.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(lower_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(lower_prim).GetSleepThresholdAttr().Set(0.0)
    mass_lower = UsdPhysics.MassAPI.Apply(lower_prim)
    mass_lower.GetMassAttr().Set(ARM_LINK_MASS)

    sim_app.update()

    # Shoulder revolute joint — connects directly to drone body
    shoulder_path = f"{arm_root}/shoulder_joint"
    shoulder_joint = UsdPhysics.RevoluteJoint.Define(stage, shoulder_path)
    shoulder_joint.GetAxisAttr().Set("Y")
    shoulder_joint.GetLowerLimitAttr().Set(-90.0)
    shoulder_joint.GetUpperLimitAttr().Set(90.0)
    shoulder_joint.GetBody0Rel().SetTargets([body_path])
    shoulder_joint.GetBody1Rel().SetTargets([upper_path])
    shoulder_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, SHOULDER_MOUNT_Z))
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
    print(f"  drone{drone_id}: 2-DOF arm created")

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

    def __init__(self, drone_id, arm_drives, initial_targets=None):
        super().__init__(f"drone{drone_id}_arm_bridge")
        self.drone_id = drone_id
        self.arm_drives = arm_drives
        self.cmd_lock = threading.Lock()
        # Pre-populate with initial targets so they get applied during first physics step
        self.cmd_targets = dict(initial_targets) if initial_targets else {}

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
        """Apply joint targets to USD drives. Call from sim loop.
        Targets persist until overridden by a new ROS2 command."""
        with self.cmd_lock:
            for name, target_deg in self.cmd_targets.items():
                drive = self.arm_drives.get(name)
                if drive:
                    drive.GetTargetPositionAttr().Set(float(target_deg))

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


class GroundTruthPublisher(RclpyNode):
    """Per-drone ROS2 node for publishing ground truth pose from Isaac Sim."""

    def __init__(self, drone_id, body_prim_path):
        super().__init__(f"drone{drone_id}_ground_truth")
        self.body_prim_path = body_prim_path
        self.drone_id = drone_id

        self.pub = self.create_publisher(
            PoseStamped, f"drone{drone_id}/state/pose", 10
        )

    def publish_pose(self, stamp_sec, stage):
        """Read body pose from USD stage and publish. Call from sim loop."""
        body_prim = stage.GetPrimAtPath(self.body_prim_path)
        if not body_prim.IsValid():
            return

        xformable = UsdGeom.Xformable(body_prim)
        world_tf = xformable.ComputeLocalToWorldTransform(0)

        translation = world_tf.ExtractTranslation()
        rotation = world_tf.ExtractRotation()
        quat = rotation.GetQuat()
        qi = quat.GetImaginary()
        qr = quat.GetReal()

        msg = PoseStamped()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.header.frame_id = "map"
        msg.pose.position.x = float(translation[0])
        msg.pose.position.y = float(translation[1])
        msg.pose.position.z = float(translation[2])
        msg.pose.orientation.x = float(qi[0])
        msg.pose.orientation.y = float(qi[1])
        msg.pose.orientation.z = float(qi[2])
        msg.pose.orientation.w = float(qr)
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


class DownwardCamPublisher(RclpyNode):
    """Per-drone ROS2 node for downward camera image publishing."""

    def __init__(self, drone_id):
        super().__init__(f"drone{drone_id}_downward_cam_publisher")
        self.drone_id = drone_id
        self.viewport_api = None  # injected after timeline.play()

        prefix = f"drone{drone_id}"
        self.img_pub = self.create_publisher(
            Image, f"{prefix}/downward_camera/image_rect", 1
        )
        self.info_pub = self.create_publisher(
            CameraInfoMsg, f"{prefix}/downward_camera/camera_info", 1
        )

    def set_viewport(self, viewport_api, viewport_window):
        """Inject viewport reference after timeline.play()."""
        self.viewport_api = viewport_api
        self._viewport_window = viewport_window  # prevent garbage collection

    def capture_and_publish(self, stamp_sec):
        if self.viewport_api is None:
            return

        from omni.kit.widget.viewport.capture import ByteCapture

        stamp = TimeMsg()
        stamp.sec = int(stamp_sec)
        stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)

        img_pub = self.img_pub
        info_pub = self.info_pub
        frame_id = f"drone{self.drone_id}_downward_camera_optical"

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
                carb.log_warn(f"Downward capture error (drone{self.drone_id}): {e}")

        self.viewport_api.schedule_capture(ByteCapture(_on_capture))
