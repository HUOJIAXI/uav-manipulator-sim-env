"""Factory functions for spawning UAV-manipulators in Isaac Sim."""

import os
import math
import threading
import ctypes

import numpy as np
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
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header

# Camera constants (matching iris_vslam.usd RSD455)
STEREO_BASELINE = 0.050       # 50 mm
CAMERA_FORWARD = 0.10         # 10 cm forward from body centre
CAMERA_UP = 0.05              # 5 cm above body centre

# Landing leg constants
LEG_LENGTH = 0.20             # 20 cm (2-DOF arm version)
LEG_LENGTH_5DOF = 0.55        # 55 cm (5-DOF arm version – clears straight-down arm)
LEG_RADIUS = 0.005            # 5 mm
ROTOR_POSITIONS = {
    "leg_front_right": (0.13, -0.22),
    "leg_front_left":  (0.13,  0.22),
    "leg_back_left":   (-0.13,  0.20),
    "leg_back_right":  (-0.13, -0.20),
}

# 2-DOF arm constants
ARM_LINK_LENGTH = 0.125       # 12.5 cm per link
ARM_LINK_RADIUS = 0.008       # 8 mm radius
ARM_LINK_MASS = 0.05          # 50 g per link
ARM_HALF_LEN = ARM_LINK_LENGTH / 2.0
SHOULDER_MOUNT_Z = -0.08      # mount point below body
JOINT_STIFFNESS = 1e5
JOINT_DAMPING = 1e4

# OpenMANIPULATOR-X 5-DOF arm constants (4-DOF + 1-DOF gripper)
# Dimensions from ROBOTIS URDF: https://emanual.robotis.com/docs/en/platform/openmanipulator_x/specification/
OM_LINK1_LENGTH = 0.060       # Base rotation housing (J1->J2)
OM_LINK2_LENGTH = 0.130       # Upper arm (J2->J3)
OM_LINK3_LENGTH = 0.124       # Forearm (J3->J4)
OM_LINK4_LENGTH = 0.082       # Wrist (J4->gripper base)
OM_LINK_RADIUS = 0.010        # Capsule radius for arm links
OM_FINGER_LENGTH = 0.040      # Gripper finger length
OM_FINGER_RADIUS = 0.005      # Gripper finger radius
OM_FINGER_OFFSET_Y = 0.021    # Finger Y-offset from centerline
OM_MOUNT_Z = -0.08            # Mount point below drone body
OM_JOINT_STIFFNESS = 1e5
OM_JOINT_DAMPING = 1e4
OM_GRIPPER_STIFFNESS = 1e4
OM_GRIPPER_DAMPING = 1e3
# Link masses from URDF (kg)
OM_LINK1_MASS = 0.079
OM_LINK2_MASS = 0.098
OM_LINK3_MASS = 0.139
OM_LINK4_MASS = 0.133
OM_FINGER_MASS = 0.020

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

    Cameras are mounted via FixedJoint (same pattern as the arm) so they
    follow the drone body during physics simulation.

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

    dx = drone_cfg.get("x", 0.0)
    dy = drone_cfg.get("y", 0.0)
    yaw = drone_cfg.get("yaw", 0.0)
    yaw_rot = Rotation.from_euler("Z", yaw, degrees=True)

    # Camera orientation: scipy [x,y,z,w] -> Gf.Quatf(w,x,y,z)
    cam_q = CAM_QUAT
    cam_quat_gf = Gf.Quatf(float(cam_q[3]), float(cam_q[0]), float(cam_q[1]), float(cam_q[2]))

    cam_root = f"{prefix}/stereo_cameras"
    UsdGeom.Xform.Define(stage, cam_root)

    for side, y_offset in [("left", half_baseline), ("right", -half_baseline)]:
        # --- Rigid-body mount (follows drone body via FixedJoint) ---
        mount_path = f"{cam_root}/mount_{side}"
        mount_xform = UsdGeom.Xform.Define(stage, mount_path)

        # Initial world position (so physics starts from correct pose)
        local_pos = np.array([CAMERA_FORWARD, y_offset, CAMERA_UP])
        world_offset = yaw_rot.apply(local_pos)
        mx = UsdGeom.Xformable(mount_xform)
        mx.ClearXformOpOrder()
        mx.AddTranslateOp().Set(Gf.Vec3d(
            dx + world_offset[0], dy + world_offset[1], spawn_height + world_offset[2]
        ))

        mount_prim = mount_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(mount_prim)
        mass_api = UsdPhysics.MassAPI.Apply(mount_prim)
        mass_api.GetMassAttr().Set(0.001)

        # FixedJoint: drone body <-> camera mount (position only, no rotation)
        joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{mount_path}/attach"))
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(mount_path)])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(CAMERA_FORWARD, y_offset, CAMERA_UP)
        )
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        # --- Camera prim (child of mount, with camera orientation) ---
        cam_path = f"{mount_path}/camera"
        cam_prim = stage.DefinePrim(cam_path, "Camera")
        cam_xf = UsdGeom.Xformable(cam_prim)
        cam_xf.ClearXformOpOrder()
        cam_xf.AddOrientOp().Set(cam_quat_gf)

        stereo_cam_paths[side] = cam_path
        print(f"  drone{drone_id}: stereo_{side} at {cam_path}")

    sim_app.update()

    return stereo_cam_paths


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


def create_5dof_arm(stage, drone_cfg, spawn_height, sim_app):
    """
    Creates landing legs and 5-DOF OpenMANIPULATOR-X arm for a drone after world.reset().

    Kinematic chain: base_link -> J1(Z) -> link1 -> J2(Y) -> link2 -> J3(Y) -> link3
                     -> J4(Y) -> link4 -> gripper(prismatic Y, 2 fingers)

    Args:
        stage: USD stage
        drone_cfg: dict from YAML config for this drone
        spawn_height: float, Z height
        sim_app: SimulationApp instance

    Returns:
        dict with keys: "arm_drives", "linear_joints", "mimic_drives"
    """
    drone_id = drone_cfg["id"]
    prefix = f"/World/drone{drone_id}"
    body_path = f"{prefix}/quadrotor/body"
    dx = drone_cfg.get("x", 0.0)
    dy = drone_cfg.get("y", 0.0)

    # --- Landing legs (longer than 2-DOF to clear the 5-DOF arm) ---
    leg_z = -LEG_LENGTH_5DOF / 2.0
    for leg_name, (rx, ry) in ROTOR_POSITIONS.items():
        leg_path = f"{body_path}/{leg_name}"
        leg_prim = UsdGeom.Cylinder.Define(stage, leg_path)
        leg_prim.GetHeightAttr().Set(LEG_LENGTH_5DOF)
        leg_prim.GetRadiusAttr().Set(LEG_RADIUS)
        leg_prim.GetAxisAttr().Set("Z")
        xform = UsdGeom.Xformable(leg_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(rx, ry, leg_z))
        UsdPhysics.CollisionAPI.Apply(leg_prim.GetPrim())
        leg_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])

    sim_app.update()
    print(f"  drone{drone_id}: Landing legs added (5-DOF, {LEG_LENGTH_5DOF}m)")

    # --- 5-DOF Arm ---
    arm_enabled = drone_cfg.get("arm", True)
    if not arm_enabled:
        return {
            "arm_drives": None, "linear_joints": set(), "mimic_drives": {},
        }

    arm_root = f"{prefix}/manipulator_5dof"
    mount_wz = spawn_height + OM_MOUNT_Z

    # Half-lengths for joint local positions
    L1H = OM_LINK1_LENGTH / 2.0
    L2H = OM_LINK2_LENGTH / 2.0
    L3H = OM_LINK3_LENGTH / 2.0
    L4H = OM_LINK4_LENGTH / 2.0
    FH = OM_FINGER_LENGTH / 2.0

    UsdGeom.Xform.Define(stage, arm_root)

    # --- Base link (massless anchor) ---
    base_path = f"{arm_root}/base_link"
    base_xform = UsdGeom.Xform.Define(stage, base_path)
    bx = UsdGeom.Xformable(base_xform)
    bx.ClearXformOpOrder()
    bx.AddTranslateOp().Set(Gf.Vec3d(dx, dy, mount_wz))
    base_prim = base_xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base_prim)
    mass_base = UsdPhysics.MassAPI.Apply(base_prim)
    mass_base.GetMassAttr().Set(0.01)

    # --- Link 1 (base rotation housing, vertical) ---
    link1_path = f"{arm_root}/link1"
    link1_cap = UsdGeom.Capsule.Define(stage, link1_path)
    link1_cap.GetHeightAttr().Set(OM_LINK1_LENGTH)
    link1_cap.GetRadiusAttr().Set(OM_LINK_RADIUS + 0.002)
    link1_cap.GetAxisAttr().Set("Z")
    link1_cap.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.3)])
    l1x = UsdGeom.Xformable(link1_cap)
    l1x.ClearXformOpOrder()
    l1x.AddTranslateOp().Set(Gf.Vec3d(dx, dy, mount_wz - L1H))
    link1_prim = link1_cap.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link1_prim)
    UsdPhysics.MassAPI.Apply(link1_prim).GetMassAttr().Set(OM_LINK1_MASS)

    # --- Link 2 (upper arm) ---
    link2_path = f"{arm_root}/link2"
    link2_cap = UsdGeom.Capsule.Define(stage, link2_path)
    link2_cap.GetHeightAttr().Set(OM_LINK2_LENGTH)
    link2_cap.GetRadiusAttr().Set(OM_LINK_RADIUS)
    link2_cap.GetAxisAttr().Set("Z")
    link2_cap.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.2)])
    l2x = UsdGeom.Xformable(link2_cap)
    l2x.ClearXformOpOrder()
    l2x.AddTranslateOp().Set(Gf.Vec3d(dx, dy, mount_wz - OM_LINK1_LENGTH - L2H))
    link2_prim = link2_cap.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link2_prim)
    UsdPhysics.MassAPI.Apply(link2_prim).GetMassAttr().Set(OM_LINK2_MASS)

    # --- Link 3 (forearm) ---
    link3_path = f"{arm_root}/link3"
    link3_cap = UsdGeom.Capsule.Define(stage, link3_path)
    link3_cap.GetHeightAttr().Set(OM_LINK3_LENGTH)
    link3_cap.GetRadiusAttr().Set(OM_LINK_RADIUS)
    link3_cap.GetAxisAttr().Set("Z")
    link3_cap.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.8)])
    l3x = UsdGeom.Xformable(link3_cap)
    l3x.ClearXformOpOrder()
    l3x.AddTranslateOp().Set(Gf.Vec3d(
        dx, dy, mount_wz - OM_LINK1_LENGTH - OM_LINK2_LENGTH - L3H
    ))
    link3_prim = link3_cap.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link3_prim)
    UsdPhysics.MassAPI.Apply(link3_prim).GetMassAttr().Set(OM_LINK3_MASS)

    # --- Link 4 (wrist) ---
    link4_path = f"{arm_root}/link4"
    link4_cap = UsdGeom.Capsule.Define(stage, link4_path)
    link4_cap.GetHeightAttr().Set(OM_LINK4_LENGTH)
    link4_cap.GetRadiusAttr().Set(OM_LINK_RADIUS - 0.002)
    link4_cap.GetAxisAttr().Set("Z")
    link4_cap.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.7, 0.2)])
    l4x = UsdGeom.Xformable(link4_cap)
    l4x.ClearXformOpOrder()
    l4x.AddTranslateOp().Set(Gf.Vec3d(
        dx, dy,
        mount_wz - OM_LINK1_LENGTH - OM_LINK2_LENGTH - OM_LINK3_LENGTH - L4H,
    ))
    link4_prim = link4_cap.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link4_prim)
    UsdPhysics.MassAPI.Apply(link4_prim).GetMassAttr().Set(OM_LINK4_MASS)

    # --- Gripper fingers ---
    finger_z = (mount_wz - OM_LINK1_LENGTH - OM_LINK2_LENGTH
                - OM_LINK3_LENGTH - OM_LINK4_LENGTH - FH)

    left_finger_path = f"{arm_root}/finger_left"
    lf_cap = UsdGeom.Capsule.Define(stage, left_finger_path)
    lf_cap.GetHeightAttr().Set(OM_FINGER_LENGTH)
    lf_cap.GetRadiusAttr().Set(OM_FINGER_RADIUS)
    lf_cap.GetAxisAttr().Set("Z")
    lf_cap.GetDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.7, 0.2)])
    lfx = UsdGeom.Xformable(lf_cap)
    lfx.ClearXformOpOrder()
    lfx.AddTranslateOp().Set(Gf.Vec3d(dx, dy + OM_FINGER_OFFSET_Y, finger_z))
    lf_prim = lf_cap.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(lf_prim)
    UsdPhysics.MassAPI.Apply(lf_prim).GetMassAttr().Set(OM_FINGER_MASS)

    right_finger_path = f"{arm_root}/finger_right"
    rf_cap = UsdGeom.Capsule.Define(stage, right_finger_path)
    rf_cap.GetHeightAttr().Set(OM_FINGER_LENGTH)
    rf_cap.GetRadiusAttr().Set(OM_FINGER_RADIUS)
    rf_cap.GetAxisAttr().Set("Z")
    rf_cap.GetDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.7, 0.2)])
    rfx = UsdGeom.Xformable(rf_cap)
    rfx.ClearXformOpOrder()
    rfx.AddTranslateOp().Set(Gf.Vec3d(dx, dy - OM_FINGER_OFFSET_Y, finger_z))
    rf_prim = rf_cap.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(rf_prim)
    UsdPhysics.MassAPI.Apply(rf_prim).GetMassAttr().Set(OM_FINGER_MASS)

    sim_app.update()

    # --- FixedJoint: drone body <-> arm base_link ---
    attach_joint = UsdPhysics.FixedJoint.Define(
        stage, Sdf.Path(f"{arm_root}/attach_to_body")
    )
    attach_joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
    attach_joint.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])
    attach_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, OM_MOUNT_Z))
    attach_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    attach_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    attach_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # --- Joint 1: base yaw (revolute Z) ---
    j1_path = f"{arm_root}/joint1"
    j1 = UsdPhysics.RevoluteJoint.Define(stage, j1_path)
    j1.GetAxisAttr().Set("Z")
    j1.GetLowerLimitAttr().Set(-180.0)
    j1.GetUpperLimitAttr().Set(180.0)
    j1.GetBody0Rel().SetTargets([base_path])
    j1.GetBody1Rel().SetTargets([link1_path])
    j1.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    j1.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, L1H))
    j1.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j1.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j1_drive = UsdPhysics.DriveAPI.Apply(j1.GetPrim(), "angular")
    j1_drive.GetTypeAttr().Set("force")
    j1_drive.GetStiffnessAttr().Set(OM_JOINT_STIFFNESS)
    j1_drive.GetDampingAttr().Set(OM_JOINT_DAMPING)
    j1_drive.GetTargetPositionAttr().Set(0.0)

    # --- Joint 2: shoulder pitch (revolute Y) ---
    j2_path = f"{arm_root}/joint2"
    j2 = UsdPhysics.RevoluteJoint.Define(stage, j2_path)
    j2.GetAxisAttr().Set("Y")
    j2.GetLowerLimitAttr().Set(-85.9)
    j2.GetUpperLimitAttr().Set(85.9)
    j2.GetBody0Rel().SetTargets([link1_path])
    j2.GetBody1Rel().SetTargets([link2_path])
    j2.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -L1H))
    j2.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, L2H))
    j2.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j2.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j2_drive = UsdPhysics.DriveAPI.Apply(j2.GetPrim(), "angular")
    j2_drive.GetTypeAttr().Set("force")
    j2_drive.GetStiffnessAttr().Set(OM_JOINT_STIFFNESS)
    j2_drive.GetDampingAttr().Set(OM_JOINT_DAMPING)
    j2_drive.GetTargetPositionAttr().Set(0.0)  # straight down at rest

    # --- Joint 3: elbow pitch (revolute Y) ---
    j3_path = f"{arm_root}/joint3"
    j3 = UsdPhysics.RevoluteJoint.Define(stage, j3_path)
    j3.GetAxisAttr().Set("Y")
    j3.GetLowerLimitAttr().Set(-85.9)
    j3.GetUpperLimitAttr().Set(80.2)
    j3.GetBody0Rel().SetTargets([link2_path])
    j3.GetBody1Rel().SetTargets([link3_path])
    j3.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -L2H))
    j3.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, L3H))
    j3.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j3.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j3_drive = UsdPhysics.DriveAPI.Apply(j3.GetPrim(), "angular")
    j3_drive.GetTypeAttr().Set("force")
    j3_drive.GetStiffnessAttr().Set(OM_JOINT_STIFFNESS)
    j3_drive.GetDampingAttr().Set(OM_JOINT_DAMPING)
    j3_drive.GetTargetPositionAttr().Set(0.0)  # straight down at rest

    # --- Joint 4: wrist pitch (revolute Y) ---
    j4_path = f"{arm_root}/joint4"
    j4 = UsdPhysics.RevoluteJoint.Define(stage, j4_path)
    j4.GetAxisAttr().Set("Y")
    j4.GetLowerLimitAttr().Set(-97.4)
    j4.GetUpperLimitAttr().Set(112.9)
    j4.GetBody0Rel().SetTargets([link3_path])
    j4.GetBody1Rel().SetTargets([link4_path])
    j4.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -L3H))
    j4.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, L4H))
    j4.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j4.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j4_drive = UsdPhysics.DriveAPI.Apply(j4.GetPrim(), "angular")
    j4_drive.GetTypeAttr().Set("force")
    j4_drive.GetStiffnessAttr().Set(OM_JOINT_STIFFNESS)
    j4_drive.GetDampingAttr().Set(OM_JOINT_DAMPING)
    j4_drive.GetTargetPositionAttr().Set(0.0)

    # --- Gripper left finger (prismatic Y) ---
    gl_path = f"{arm_root}/gripper_left_joint"
    gl = UsdPhysics.PrismaticJoint.Define(stage, gl_path)
    gl.GetAxisAttr().Set("Y")
    gl.GetLowerLimitAttr().Set(-0.010)
    gl.GetUpperLimitAttr().Set(0.019)
    gl.GetBody0Rel().SetTargets([link4_path])
    gl.GetBody1Rel().SetTargets([left_finger_path])
    gl.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, OM_FINGER_OFFSET_Y, -L4H))
    gl.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, FH))
    gl.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    gl.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    gl_drive = UsdPhysics.DriveAPI.Apply(gl.GetPrim(), "linear")
    gl_drive.GetTypeAttr().Set("force")
    gl_drive.GetStiffnessAttr().Set(OM_GRIPPER_STIFFNESS)
    gl_drive.GetDampingAttr().Set(OM_GRIPPER_DAMPING)
    gl_drive.GetTargetPositionAttr().Set(0.0)

    # --- Gripper right finger (prismatic Y, mimic of left) ---
    gr_path = f"{arm_root}/gripper_right_joint"
    gr = UsdPhysics.PrismaticJoint.Define(stage, gr_path)
    gr.GetAxisAttr().Set("Y")
    gr.GetLowerLimitAttr().Set(-0.019)
    gr.GetUpperLimitAttr().Set(0.010)
    gr.GetBody0Rel().SetTargets([link4_path])
    gr.GetBody1Rel().SetTargets([right_finger_path])
    gr.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, -OM_FINGER_OFFSET_Y, -L4H))
    gr.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, FH))
    gr.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    gr.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    gr_drive = UsdPhysics.DriveAPI.Apply(gr.GetPrim(), "linear")
    gr_drive.GetTypeAttr().Set("force")
    gr_drive.GetStiffnessAttr().Set(OM_GRIPPER_STIFFNESS)
    gr_drive.GetDampingAttr().Set(OM_GRIPPER_DAMPING)
    gr_drive.GetTargetPositionAttr().Set(0.0)

    sim_app.update()
    print(f"  drone{drone_id}: 5-DOF OpenMANIPULATOR-X arm created (folded)")

    # Build drive dictionaries
    arm_drives = {
        "joint1": UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(j1_path), "angular"),
        "joint2": UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(j2_path), "angular"),
        "joint3": UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(j3_path), "angular"),
        "joint4": UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(j4_path), "angular"),
        "gripper": UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(gl_path), "linear"),
    }
    # Right finger mimics left with multiplier=-1
    mimic_drives = {
        "gripper": [(
            UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(gr_path), "linear"),
            -1.0,
        )],
    }

    return {
        "arm_drives": arm_drives,
        "linear_joints": {"gripper"},
        "mimic_drives": mimic_drives,
    }


class ArmBridgeNode(RclpyNode):
    """Per-drone ROS2 node for arm joint command/state bridging."""

    def __init__(self, drone_id, arm_drives, linear_joints=None, mimic_drives=None):
        super().__init__(f"drone{drone_id}_arm_bridge")
        self.arm_drives = arm_drives
        self.linear_joints = linear_joints or set()
        self.mimic_drives = mimic_drives or {}  # {name: [(drive, multiplier), ...]}
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
                    if name in self.linear_joints:
                        self.cmd_targets[name] = msg.position[i]  # meters
                    else:
                        self.cmd_targets[name] = math.degrees(msg.position[i])

    def apply_commands(self):
        """Apply pending joint commands to USD drives. Call from sim loop."""
        with self.cmd_lock:
            for name, target in self.cmd_targets.items():
                drive = self.arm_drives.get(name)
                if drive:
                    drive.GetTargetPositionAttr().Set(float(target))
                # Apply mimic drives (e.g. right gripper finger mirrors left)
                for mimic_drive, mult in self.mimic_drives.get(name, []):
                    mimic_drive.GetTargetPositionAttr().Set(float(target * mult))
            self.cmd_targets.clear()

    def publish_states(self, stamp_sec):
        msg = JointState()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.name = list(self.arm_drives.keys())
        msg.position = []
        for name, drive in self.arm_drives.items():
            target = drive.GetTargetPositionAttr().Get()
            if target is None:
                target = 0.0
            if name in self.linear_joints:
                msg.position.append(float(target))  # meters
            else:
                msg.position.append(math.radians(target))
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
