# Ground Robot with Simulated UWB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Jetbot wheeled robot to the multi-UAV Isaac Sim simulation that is controlled via ROS2 `cmd_vel` and publishes its position with configurable Gaussian noise simulating UWB.

**Architecture:** New `spawn_ground_robot.py` module following the same patterns as `spawn_uav.py`. The Jetbot is loaded from Isaac Sim's Nucleus assets, driven by a `DifferentialController`, and a `GroundRobotBridge` ROS2 node handles cmd_vel subscription and noisy position publishing. Integration into `launch_multi_uav.py` follows the existing phased startup sequence.

**Tech Stack:** Isaac Sim (WheeledRobot, DifferentialController), ROS2 (rclpy), NumPy (Gaussian noise), PyYAML (config)

---

### Task 1: Create `spawn_ground_robot.py` — robot spawning function

**Files:**
- Create: `multi_uav/spawn_ground_robot.py`

- [ ] **Step 1: Create the module with imports and constants**

```python
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
```

- [ ] **Step 2: Add the `spawn_ground_robot` function**

This function loads a Jetbot from Nucleus assets and adds it to the World scene, returning the WheeledRobot handle. It must be called before `world.reset()`.

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add multi_uav/spawn_ground_robot.py
git commit -m "feat: add ground robot spawning function with Jetbot asset"
```

---

### Task 2: Add `GroundRobotBridge` ROS2 node

**Files:**
- Modify: `multi_uav/spawn_ground_robot.py`

- [ ] **Step 1: Add the `GroundRobotBridge` class**

This ROS2 node subscribes to `cmd_vel`, applies differential drive commands to the Jetbot each physics step, and publishes noisy UWB position + clean ground truth pose at 10 Hz.

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add multi_uav/spawn_ground_robot.py
git commit -m "feat: add GroundRobotBridge ROS2 node with UWB noise and cmd_vel"
```

---

### Task 3: Update config with `ground_robots` section

**Files:**
- Modify: `multi_uav/config/example_config.yaml`
- Modify: `multi_uav/config/default_config.yaml`

- [ ] **Step 1: Add ground_robots to `example_config.yaml`**

Add after the `drones` section:

```yaml
ground_robots:
  - id: 0
    x: 3.0
    y: 0.0
    yaw: 0.0
    uwb_noise_std: 0.1
```

- [ ] **Step 2: Add empty ground_robots to `default_config.yaml`**

Add after the `drones` section (no ground robots by default):

```yaml
ground_robots: []
```

- [ ] **Step 3: Commit**

```bash
git add multi_uav/config/example_config.yaml multi_uav/config/default_config.yaml
git commit -m "feat: add ground_robots config section with UWB noise parameter"
```

---

### Task 4: Integrate ground robot into `launch_multi_uav.py`

**Files:**
- Modify: `multi_uav/launch_multi_uav.py`

- [ ] **Step 1: Add ground robot config validation in `load_config`**

After the existing drone validation (after line 58, before `return cfg`), add validation for ground robots:

```python
    # Validate ground robot IDs
    ground_robots = cfg.get("ground_robots", [])
    gr_id_set = set()
    for i, gr in enumerate(ground_robots):
        if "id" not in gr:
            print(f"ERROR: Ground robot entry {i} missing required 'id' field.")
            sys.exit(1)
        gid = gr["id"]
        if gid in gr_id_set:
            print(f"ERROR: Duplicate ground_robot id={gid} in config.")
            sys.exit(1)
        gr_id_set.add(gid)

    # Warn if ground robot spawns near a drone
    for gr in ground_robots:
        for d in drones:
            dist = math.sqrt((gr.get("x", 0) - d.get("x", 0)) ** 2
                             + (gr.get("y", 0) - d.get("y", 0)) ** 2)
            if dist < 1.0:
                print(f"WARNING: ground_robot{gr['id']} and drone{d['id']} are only "
                      f"{dist:.2f}m apart — risk of collision at spawn.")
```

- [ ] **Step 2: Parse ground_robots config in `main()`**

After line 81 (`drones_cfg = cfg["drones"]`), add:

```python
    ground_robots_cfg = cfg.get("ground_robots", [])
```

Update the print banner to include ground robots:

```python
    num_ground_robots = len(ground_robots_cfg)
    print("\n" + "=" * 70)
    print(f"  Multi-UAV Launcher — {num_drones} drone(s), {num_ground_robots} ground robot(s)")
    print("=" * 70 + "\n")
```

- [ ] **Step 3: Add ground robot import**

After the existing `from spawn_uav import ...` line (line 114), add:

```python
    from spawn_ground_robot import spawn_ground_robot, GroundRobotBridge
```

- [ ] **Step 4: Spawn ground robots in Phase 1 (before world.reset())**

After the drone spawning loop (after line 127, before the `world.reset()` comment), add:

```python
    # --- Spawn ground robots (before world.reset()) ---
    ground_robot_handles = []
    if ground_robots_cfg:
        print("Spawning ground robots...")
        for gr_cfg in ground_robots_cfg:
            gr_handle = spawn_ground_robot(world, gr_cfg, simulation_app)
            ground_robot_handles.append(gr_handle)
            time.sleep(0.5)
```

- [ ] **Step 5: Create GroundRobotBridge nodes in ROS2 setup**

After the drone ROS2 node creation loop (after line 204, before `spin_thread`), add:

```python
    ground_robot_bridges = []
    for gr_handle in ground_robot_handles:
        gid = gr_handle["robot_id"]
        bridge = GroundRobotBridge(
            gid,
            gr_handle["wheeled_robot"],
            uwb_noise_std=gr_handle["uwb_noise_std"],
        )
        executor.add_node(bridge)
        ground_robot_bridges.append(bridge)
        print(f"  ground_robot{gid}: bridge node ready "
              f"(uwb_noise_std={gr_handle['uwb_noise_std']}m)")
```

- [ ] **Step 6: Add ground robot velocity + publishing to the main simulation loop**

In the simulation loop, after `bridge.apply_commands()` for arm bridges (after line 307), add velocity application:

```python
            # Apply ground robot velocities
            for gr_bridge in ground_robot_bridges:
                gr_bridge.apply_velocity()
```

Inside the 10 Hz publish block (after `gt.publish_pose(sim_time, stage)` on line 319), add:

```python
                for gr_bridge in ground_robot_bridges:
                    gr_bridge.publish_uwb(sim_time, stage)
                    gr_bridge.publish_pose(sim_time, stage)
```

- [ ] **Step 7: Add ground robot topics to the READY summary**

After the drone summary loop (after line 293), add:

```python
        for gr_handle in ground_robot_handles:
            gid = gr_handle["robot_id"]
            print(f"\n  ground_robot{gid}:")
            print(f"    ground_robot{gid}/cmd_vel (geometry_msgs/Twist)")
            print(f"    ground_robot{gid}/uwb/position (PointStamped, noise={gr_handle['uwb_noise_std']}m)")
            print(f"    ground_robot{gid}/state/pose (ground truth)")
```

- [ ] **Step 8: Add ground robot cleanup in `finally` block**

After the existing `bridge.destroy_node()` line (line 330), add:

```python
        for gr_bridge in ground_robot_bridges:
            gr_bridge.destroy_node()
```

- [ ] **Step 9: Commit**

```bash
git add multi_uav/launch_multi_uav.py
git commit -m "feat: integrate ground robot spawning and UWB publishing into launcher"
```

---

### Task 5: Smoke test the full integration

- [ ] **Step 1: Launch with example config and verify**

```bash
cd /home/huojiaxi/Desktop/uav_sim/multi_uav
# Launch simulation (will need Isaac Sim runtime)
python launch_multi_uav.py --config config/example_config.yaml
```

Expected: simulation starts, prints ground_robot0 in READY summary, Jetbot visible in scene.

- [ ] **Step 2: Verify ROS2 topics exist**

In a separate terminal:

```bash
ros2 topic list | grep ground_robot
```

Expected output:
```
/ground_robot0/cmd_vel
/ground_robot0/uwb/position
/ground_robot0/state/pose
```

- [ ] **Step 3: Verify UWB publishing with noise**

```bash
ros2 topic echo /ground_robot0/uwb/position --once
```

Expected: `PointStamped` message with x/y/z values near (3.0, 0.0, 0.05) but with noise.

- [ ] **Step 4: Test cmd_vel driving**

```bash
ros2 topic pub /ground_robot0/cmd_vel geometry_msgs/Twist "{linear: {x: 0.1}, angular: {z: 0.0}}" --rate 10
```

Expected: Jetbot moves forward in the simulation; UWB position values change over time.

- [ ] **Step 5: Compare UWB vs ground truth**

```bash
ros2 topic echo /ground_robot0/state/pose --once
ros2 topic echo /ground_robot0/uwb/position --once
```

Expected: ground truth is clean, UWB has visible noise (differences ~0.1m std dev).

- [ ] **Step 6: Final commit if any fixes needed**

```bash
git add -u
git commit -m "fix: address smoke test findings for ground robot"
```
