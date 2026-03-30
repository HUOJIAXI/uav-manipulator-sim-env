"""Factory functions for spawning ground robots in Isaac Sim."""

import math
import threading

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, Gf

import rclpy
from rclpy.node import Node as RclpyNode
from geometry_msgs.msg import Twist, PoseStamped, PointStamped

# Jetbot parameters (from Isaac Sim example)
JETBOT_ASSET_SUBPATH = "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
JETBOT_WHEEL_DOF_NAMES = ["left_wheel_joint", "right_wheel_joint"]
JETBOT_WHEEL_RADIUS = 0.03      # meters
JETBOT_WHEEL_BASE = 0.1125      # meters
JETBOT_SPAWN_Z = 0.05           # slight offset above ground

DEFAULT_UWB_NOISE_STD = 0.1     # meters


def spawn_ground_robot(world, robot_cfg, sim_app):
    """
    Spawn a Jetbot ground robot into the Isaac Sim world.

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
    jetbot_usd = assets_root + JETBOT_ASSET_SUBPATH

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
            wheel_dof_names=JETBOT_WHEEL_DOF_NAMES,
            create_robot=True,
            usd_path=jetbot_usd,
            position=np.array([x, y, JETBOT_SPAWN_Z]),
            orientation=orientation,
        )
    )
    sim_app.update()
    print(f"  ground_robot{robot_id}: Jetbot spawned at ({x}, {y}), yaw={yaw}")

    return {
        "robot_id": robot_id,
        "wheeled_robot": wheeled_robot,
        "uwb_noise_std": uwb_noise_std,
    }


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
            wheel_radius=JETBOT_WHEEL_RADIUS,
            wheel_base=JETBOT_WHEEL_BASE,
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
        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        world_tf = xformable.ComputeLocalToWorldTransform(0)
        pos = world_tf.ExtractTranslation()

        msg = PointStamped()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.header.frame_id = "map"
        msg.point.x = float(pos[0]) + np.random.normal(0.0, self.uwb_noise_std)
        msg.point.y = float(pos[1]) + np.random.normal(0.0, self.uwb_noise_std)
        msg.point.z = float(pos[2]) + np.random.normal(0.0, self.uwb_noise_std)
        self.uwb_pub.publish(msg)

    def publish_pose(self, stamp_sec, stage):
        """Publish clean ground truth pose. Call at 10 Hz from sim loop."""
        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        world_tf = xformable.ComputeLocalToWorldTransform(0)
        pos = world_tf.ExtractTranslation()
        rot = world_tf.ExtractRotation()
        quat = rot.GetQuat()
        qi = quat.GetImaginary()
        qr = quat.GetReal()

        msg = PoseStamped()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.header.frame_id = "map"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(qi[0])
        msg.pose.orientation.y = float(qi[1])
        msg.pose.orientation.z = float(qi[2])
        msg.pose.orientation.w = float(qr)
        self.pose_pub.publish(msg)
