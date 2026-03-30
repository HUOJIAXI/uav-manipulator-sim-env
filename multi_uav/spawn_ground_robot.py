"""Factory functions for spawning ground robots in Isaac Sim."""

import math
import threading

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdShade, Sdf, Gf

import rclpy
from rclpy.node import Node as RclpyNode
from geometry_msgs.msg import Twist, PoseStamped, PointStamped

# Nova Carter parameters
ROBOT_ASSET_SUBPATH = "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
ROBOT_WHEEL_DOF_NAMES = ["joint_wheel_left", "joint_wheel_right"]
ROBOT_WHEEL_RADIUS = 0.14       # meters
ROBOT_WHEEL_BASE = 0.4132       # meters
ROBOT_SPAWN_Z = 0.0             # Nova Carter sits at ground level

DEFAULT_UWB_NOISE_STD = 0.05    # meters
DEFAULT_MAX_SPEED = 2.0         # m/s, configurable via YAML

# AprilTag parameters
APRILTAG_MDL_SUBPATH = "/Isaac/Materials/AprilTag/AprilTag.mdl"
APRILTAG_TEXTURE_SUBPATH = "/Isaac/Materials/AprilTag/Textures/tag36h11.png"
APRILTAG_SIZE = 0.40            # meters — roughly matching Nova Carter top surface
APRILTAG_X_OFFSET = 0.05         # meters forward from robot origin
APRILTAG_HEIGHT_OFFSET = 0.45   # meters above robot origin (top of Nova Carter)


def spawn_ground_robot(world, robot_cfg, sim_app):
    """
    Spawn a Nova Carter ground robot into the Isaac Sim world.

    Must be called BEFORE world.reset().

    Args:
        world: Isaac Sim World instance
        robot_cfg: dict from YAML config for this ground robot
        sim_app: SimulationApp instance

    Returns:
        dict with keys: "robot_id", "wheeled_robot", "uwb_noise_std"
    """
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    from isaacsim.storage.native import get_assets_root_path

    robot_id = robot_cfg["id"]
    x = robot_cfg.get("x", 0.0)
    y = robot_cfg.get("y", 0.0)
    yaw = robot_cfg.get("yaw", 0.0)
    uwb_noise_std = robot_cfg.get("uwb_noise_std", DEFAULT_UWB_NOISE_STD)

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not find Isaac Sim assets folder (Nucleus)")
    robot_usd = assets_root + ROBOT_ASSET_SUBPATH

    prim_path = f"/World/ground_robot{robot_id}"

    # Compute spawn orientation from yaw
    yaw_rad = math.radians(yaw)
    orientation = np.array([
        math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)
    ])  # [w, x, y, z] for Isaac Sim

    wheeled_robot = world.scene.add(
        WheeledRobot(
            prim_path=prim_path,
            name=f"ground_robot{robot_id}",
            wheel_dof_names=ROBOT_WHEEL_DOF_NAMES,
            create_robot=True,
            usd_path=robot_usd,
            position=np.array([x, y, ROBOT_SPAWN_Z]),
            orientation=orientation,
        )
    )
    sim_app.update()
    print(f"  ground_robot{robot_id}: Nova Carter spawned at ({x}, {y}), yaw={yaw}")

    return {
        "robot_id": robot_id,
        "wheeled_robot": wheeled_robot,
        "uwb_noise_std": uwb_noise_std,
    }


def configure_wheel_drives(wheeled_robot, robot_id):
    """
    Increase wheel joint velocity and effort limits after world.reset().

    Targets only the two wheel joints by index, leaving other DOFs unchanged.
    """
    av = wheeled_robot._articulation_view
    num_dof = av.num_dof

    # Find wheel joint indices
    wheel_indices = []
    for name in ROBOT_WHEEL_DOF_NAMES:
        idx = av.get_dof_index(name)
        wheel_indices.append(idx)
    print(f"  ground_robot{robot_id}: num_dof={num_dof}, "
          f"wheel indices={wheel_indices}")

    # Set limits only on wheel joints
    n_wheels = len(wheel_indices)
    max_vels = np.ones((1, n_wheels)) * 50.0    # rad/s (~7 m/s with 0.14m radius)
    max_efforts = np.ones((1, n_wheels)) * 100.0  # Nm
    av.set_max_joint_velocities(max_vels, joint_indices=wheel_indices)
    av.set_max_efforts(max_efforts, joint_indices=wheel_indices)

    # Verify the values were applied
    cur_vels = av.get_joint_max_velocities(joint_indices=wheel_indices)
    cur_efforts = av.get_max_efforts(joint_indices=wheel_indices)
    print(f"  ground_robot{robot_id}: wheel limits set — "
          f"max_vel={cur_vels}, max_effort={cur_efforts}")


def create_apriltag_on_robot(stage, robot_id, sim_app):
    """
    Place an AprilTag (tag36h11, ID 0) on top of the Nova Carter.

    Creates a flat plane as a child of the robot's chassis_link so it
    moves with the robot. Applies the Isaac Sim built-in AprilTag MDL material.

    Must be called AFTER world.reset() and timeline.play() for material creation.
    """
    import omni.kit.commands
    from isaacsim.storage.native import get_assets_root_path

    assets_root = get_assets_root_path()
    robot_prim_path = f"/World/ground_robot{robot_id}"

    # Find the chassis link to parent the tag to
    chassis_path = f"{robot_prim_path}/chassis_link"
    chassis_prim = stage.GetPrimAtPath(chassis_path)
    if not chassis_prim.IsValid():
        # Fallback: parent to the robot root
        chassis_path = robot_prim_path
        print(f"  ground_robot{robot_id}: WARNING — chassis_link not found, "
              f"parenting AprilTag to robot root")

    # Create a plane mesh as child of chassis
    tag_path = f"{chassis_path}/apriltag_plane"
    plane = UsdGeom.Mesh.Define(stage, tag_path)

    # A flat quad facing up (+Z), centered on the robot
    half = APRILTAG_SIZE / 2.0
    plane.GetPointsAttr().Set([
        Gf.Vec3f(-half, -half, 0),
        Gf.Vec3f(half, -half, 0),
        Gf.Vec3f(half, half, 0),
        Gf.Vec3f(-half, half, 0),
    ])
    plane.GetFaceVertexCountsAttr().Set([4])
    plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    plane.GetNormalsAttr().Set([
        Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1),
        Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 0, 1),
    ])
    plane.SetNormalsInterpolation("vertex")

    # UV coordinates for the texture
    primvars_api = UsdGeom.PrimvarsAPI(plane)
    texcoords = primvars_api.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    texcoords.Set([
        Gf.Vec2f(0, 0), Gf.Vec2f(1, 0),
        Gf.Vec2f(1, 1), Gf.Vec2f(0, 1),
    ])

    # Position on top of the robot
    xform = UsdGeom.Xformable(plane)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(APRILTAG_X_OFFSET, 0, APRILTAG_HEIGHT_OFFSET))

    sim_app.update()

    # Create AprilTag MDL material
    mtl_path = f"{robot_prim_path}/Looks/AprilTag"
    omni.kit.commands.execute(
        "CreateMdlMaterialPrim",
        mtl_url=assets_root + APRILTAG_MDL_SUBPATH,
        mtl_name="AprilTag",
        mtl_path=mtl_path,
        select_new_prim=False,
    )
    sim_app.update()

    # Set the tag texture (tag36h11 mosaic — ID 0 is the default tile)
    shader_prim = stage.GetPrimAtPath(mtl_path + "/Shader")
    if shader_prim.IsValid():
        attr = shader_prim.CreateAttribute("inputs:tag_mosaic", Sdf.ValueTypeNames.Asset)
        attr.Set(Sdf.AssetPath(assets_root + APRILTAG_TEXTURE_SUBPATH))

    # Bind material to the plane
    material = UsdShade.Material(stage.GetPrimAtPath(mtl_path))
    if material:
        UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material)

    sim_app.update()
    print(f"  ground_robot{robot_id}: AprilTag (tag36h11) placed on top "
          f"({APRILTAG_SIZE}m, z_offset={APRILTAG_HEIGHT_OFFSET}m)")


class GroundRobotBridge(RclpyNode):
    """ROS2 node for ground robot velocity control and UWB position publishing."""

    def __init__(self, robot_id, wheeled_robot, uwb_noise_std=DEFAULT_UWB_NOISE_STD):
        super().__init__(f"ground_robot{robot_id}_bridge")
        self.robot_id = robot_id
        self.wheeled_robot = wheeled_robot
        self.uwb_noise_std = uwb_noise_std
        self.prim_path = f"/World/ground_robot{robot_id}"

        self.cmd_lock = threading.Lock()
        self.linear_vel = 0.0   # forward velocity (m/s)
        self.angular_vel = 0.0  # yaw rate (rad/s)

        from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
        self.controller = DifferentialController(
            name=f"ground_robot{robot_id}_controller",
            wheel_radius=ROBOT_WHEEL_RADIUS,
            wheel_base=ROBOT_WHEEL_BASE,
        )

        prefix = f"ground_robot{robot_id}"
        self.cmd_sub = self.create_subscription(
            Twist, f"{prefix}/cmd_vel", self._cmd_vel_cb, 10
        )
        self.uwb_pub = self.create_publisher(
            PointStamped, f"{prefix}/uwb/position", 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, f"{prefix}/state/pose", 10
        )

    def _cmd_vel_cb(self, msg):
        with self.cmd_lock:
            self.linear_vel = msg.linear.x
            self.angular_vel = msg.angular.z

    def apply_velocity(self):
        """Apply latest cmd_vel to the Jetbot. Call every physics step."""
        with self.cmd_lock:
            lin = self.linear_vel
            ang = self.angular_vel
        wheel_actions = self.controller.forward(command=[lin, ang])
        self.wheeled_robot.apply_wheel_actions(wheel_actions)

    def publish_uwb(self, stamp_sec, stage):
        """Publish noisy UWB position. Call at 10 Hz from sim loop."""
        position, _ = self.wheeled_robot.get_world_pose()
        if position is None:
            return

        msg = PointStamped()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.header.frame_id = "map"
        msg.point.x = float(position[0]) + np.random.normal(0.0, self.uwb_noise_std)
        msg.point.y = float(position[1]) + np.random.normal(0.0, self.uwb_noise_std)
        msg.point.z = float(position[2]) + np.random.normal(0.0, self.uwb_noise_std)
        self.uwb_pub.publish(msg)

    def publish_pose(self, stamp_sec, stage):
        """Publish clean ground truth pose. Call at 10 Hz from sim loop."""
        position, orientation = self.wheeled_robot.get_world_pose()
        if position is None:
            return

        msg = PoseStamped()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.header.frame_id = "map"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        # orientation is [w, x, y, z] from Isaac Sim
        msg.pose.orientation.x = float(orientation[1])
        msg.pose.orientation.y = float(orientation[2])
        msg.pose.orientation.z = float(orientation[3])
        msg.pose.orientation.w = float(orientation[0])
        self.pose_pub.publish(msg)
