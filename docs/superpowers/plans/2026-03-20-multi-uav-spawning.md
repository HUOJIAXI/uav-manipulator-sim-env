# Multi-UAV Spawning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable spawning multiple UAV-manipulators from a YAML config file in Isaac Sim, with per-drone component toggles.

**Architecture:** Factory function approach — `spawn_uav()` and `create_arm()` in `spawn_uav.py` handle per-drone setup, called from `launch_multi_uav.py` which manages config loading, initialization ordering, and the simulation loop.

**Tech Stack:** Isaac Sim 5.1, Pegasus Simulator, PX4 SITL, ROS2 (rclpy), USD/OpenUSD (pxr), PyYAML

**Spec:** `docs/superpowers/specs/2026-03-20-multi-uav-spawning-design.md`

**Note:** This project runs inside Isaac Sim and cannot be unit-tested with pytest. Each task includes manual verification steps to run inside the simulator.

---

## File Structure

```
multi_uav/
├── __init__.py                  # Empty package marker
├── launch_multi_uav.py          # Main entry point (CLI arg parsing, init, sim loop)
├── spawn_uav.py                 # spawn_uav(), create_arm(), ArmBridgeNode, StereoCamPublisher
└── config/
    ├── default_config.yaml      # Single drone at origin (matches existing script)
    └── example_config.yaml      # Two drones for testing
```

---

### Task 1: Create config files and package structure

**Files:**
- Create: `multi_uav/__init__.py`
- Create: `multi_uav/config/default_config.yaml`
- Create: `multi_uav/config/example_config.yaml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p multi_uav/config
```

- [ ] **Step 2: Create empty `__init__.py`**

Create `multi_uav/__init__.py` as an empty file.

- [ ] **Step 3: Create `default_config.yaml`**

```yaml
environment: "Curved Gridroom"
spawn_height: 0.30

drones:
  - id: 0
    x: 0.0
    y: 0.0
    yaw: 0.0
    stereo_camera: true
    arm: true
    px4_autolaunch: true
    px4_vehicle_id: 0
```

- [ ] **Step 4: Create `example_config.yaml`**

```yaml
environment: "Curved Gridroom"
spawn_height: 0.30

drones:
  - id: 0
    x: 0.0
    y: 0.0
    yaw: 0.0
    stereo_camera: true
    arm: true
    px4_autolaunch: true
    px4_vehicle_id: 0

  - id: 1
    x: 2.0
    y: 0.0
    yaw: 90.0
    stereo_camera: true
    arm: true
    px4_autolaunch: true
    px4_vehicle_id: 1
```

- [ ] **Step 5: Commit**

```bash
git add multi_uav/
git commit -m "Add multi_uav package skeleton and config files"
```

---

### Task 2: Implement `spawn_uav()` function

**Files:**
- Create: `multi_uav/spawn_uav.py`

This function is called before `world.reset()`. It creates the Multirotor, cleans stale OmniGraphs, and optionally creates stereo camera prims. Reference: `launch_stereo_vslam_with_arm.py` lines 58-129.

- [ ] **Step 1: Create `spawn_uav.py` with imports and constants**

```python
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
```

- [ ] **Step 2: Implement `spawn_uav()` function**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add multi_uav/spawn_uav.py
git commit -m "Add spawn_uav() factory function"
```

---

### Task 3: Implement `create_arm()` function

**Files:**
- Modify: `multi_uav/spawn_uav.py`

This function is called after `world.reset()`. It adds landing legs and optionally the 2-DOF arm. Reference: `launch_stereo_vslam_with_arm.py` lines 140-307. Key change: arm link world positions are offset by drone's `(x, y)` spawn position.

- [ ] **Step 1: Add `create_arm()` function to `spawn_uav.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add multi_uav/spawn_uav.py
git commit -m "Add create_arm() factory function for legs and arm"
```

---

### Task 4: Implement ROS2 node classes

**Files:**
- Modify: `multi_uav/spawn_uav.py`

Add self-contained `ArmBridgeNode` and `StereoCamPublisher` classes. Key difference from existing script: instance attributes instead of module-level globals, drone-prefixed topics. Reference: `launch_stereo_vslam_with_arm.py` lines 331-354 (ArmBridgeNode) and 443-511 (StereoCamPublisher).

- [ ] **Step 1: Add `ArmBridgeNode` class**

Append to `spawn_uav.py` (imports already at top of file from Task 2):

```python
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
```

- [ ] **Step 2: Add `StereoCamPublisher` class**

Append to `spawn_uav.py` (imports already at top of file from Task 2):

```python
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
                    import ctypes
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
```

- [ ] **Step 3: Commit**

```bash
git add multi_uav/spawn_uav.py
git commit -m "Add ArmBridgeNode and StereoCamPublisher ROS2 classes"
```

---

### Task 5: Implement `launch_multi_uav.py` — config loading and validation

**Files:**
- Create: `multi_uav/launch_multi_uav.py`

This task covers steps 1-4 of the main script flow: CLI arg parsing, config loading/validation, SimulationApp init, and environment loading. The simulation loop is added in Task 6.

- [ ] **Step 1: Create `launch_multi_uav.py` with config loading and validation**

```python
#!/usr/bin/env python3
"""
Multi-UAV Spawning Launch Script.
Spawns multiple UAV-manipulators from a YAML config file.

Usage:
    python launch_multi_uav.py [--config path/to/config.yaml]
"""

import argparse
import math
import os
import sys
import time

import yaml


def load_config(config_path):
    """Load and validate YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    drones = cfg.get("drones", [])
    if not drones:
        print("ERROR: Config must define at least one drone in 'drones' list.")
        sys.exit(1)

    # Validate drone id uniqueness
    id_set = set()
    for d in drones:
        did = d["id"]
        if did in id_set:
            print(f"ERROR: Duplicate drone id={did} in config.")
            sys.exit(1)
        id_set.add(did)

    # Validate px4_vehicle_id uniqueness
    vid_set = set()
    for d in drones:
        vid = d.get("px4_vehicle_id", d["id"])
        if vid in vid_set:
            print(f"ERROR: Duplicate px4_vehicle_id={vid} in config.")
            sys.exit(1)
        vid_set.add(vid)

    # Warn on overlapping spawn positions (< 1m apart)
    for i, a in enumerate(drones):
        for b in drones[i + 1:]:
            dist = math.sqrt((a.get("x", 0) - b.get("x", 0)) ** 2
                             + (a.get("y", 0) - b.get("y", 0)) ** 2)
            if dist < 1.0:
                print(f"WARNING: drone{a['id']} and drone{b['id']} are only "
                      f"{dist:.2f}m apart — risk of physics collision at spawn.")

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Multi-UAV Isaac Sim Launcher")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "default_config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # Validate asset exists
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    usd_path = os.path.join(assets_dir, "iris_vslam.usd")
    if not os.path.exists(usd_path):
        print(f"ERROR: iris_vslam.usd not found at {usd_path}")
        sys.exit(1)

    cfg = load_config(args.config)
    spawn_height = cfg.get("spawn_height", 0.30)
    env_name = cfg.get("environment", "Curved Gridroom")
    drones_cfg = cfg["drones"]

    num_drones = len(drones_cfg)
    print("\n" + "=" * 70)
    print(f"  Multi-UAV Launcher — {num_drones} drone(s)")
    print("=" * 70 + "\n")

    # --- Init SimulationApp ---
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    import omni.timeline
    import omni.usd
    from omni.isaac.core.world import World
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")

    from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

    # --- Init Pegasus & World ---
    print("Initializing Pegasus...")
    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world

    print(f"Loading environment: {env_name}...")
    pg.load_environment(SIMULATION_ENVIRONMENTS[env_name])
    time.sleep(2.0)
    simulation_app.update()
    print("  Environment loaded\n")

    # --- Placeholder for spawning + sim loop (Task 6) ---


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add multi_uav/launch_multi_uav.py
git commit -m "Add launch_multi_uav.py with config loading and validation"
```

---

### Task 6: Implement spawning loop and simulation loop

**Files:**
- Modify: `multi_uav/launch_multi_uav.py`

Replace the placeholder at the end of `main()` with the full spawning sequence (steps 5-13 from spec) and simulation loop.

- [ ] **Step 1: Add spawning and simulation loop**

Replace the `# --- Placeholder ---` comment with:

```python
    from multi_uav.spawn_uav import spawn_uav, create_arm, ArmBridgeNode, StereoCamPublisher

    stage = omni.usd.get_context().get_stage()

    # --- Phase 1: Spawn all Multirotors (before world.reset()) ---
    print("Spawning drones...")
    drone_handles = []
    for dcfg in drones_cfg:
        handle = spawn_uav(stage, world, dcfg, spawn_height, assets_dir, simulation_app)
        drone_handles.append(handle)
        time.sleep(0.5)

    # --- world.reset() ---
    print("Resetting world...")
    world.reset()
    stage = omni.usd.get_context().get_stage()

    # --- Phase 2: Create legs + arms (after world.reset()) ---
    print("Creating legs and arms...")
    for i, dcfg in enumerate(drones_cfg):
        arm_result = create_arm(stage, dcfg, spawn_height, simulation_app)
        drone_handles[i].update(arm_result)

    # --- ROS2 setup ---
    print("Setting up ROS2...")
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading

    rclpy.init()
    executor = MultiThreadedExecutor()

    arm_bridges = []
    stereo_pubs = []

    for handle in drone_handles:
        did = handle["drone_id"]

        # Arm bridge
        if handle.get("arm_drives"):
            bridge = ArmBridgeNode(did, handle["arm_drives"])
            executor.add_node(bridge)
            arm_bridges.append(bridge)
            handle["arm_bridge"] = bridge
            print(f"  drone{did}: arm bridge node ready")
        else:
            handle["arm_bridge"] = None

        # Stereo publisher
        if handle.get("stereo_cam_paths"):
            pub = StereoCamPublisher(did)
            stereo_pubs.append(pub)
            handle["stereo_pub"] = pub
            print(f"  drone{did}: stereo publisher node ready")
        else:
            handle["stereo_pub"] = None

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    print("  ROS2 executor spinning\n")

    # --- Timeline ---
    timeline = omni.timeline.get_timeline_interface()

    try:
        timeline.play()

        # --- Stereo viewports (must be after timeline.play()) ---
        import omni.kit.viewport.utility as vp_utils
        STEREO_CAM_RESOLUTION = (640, 480)

        for handle in drone_handles:
            if handle.get("stereo_cam_paths") and handle.get("stereo_pub"):
                did = handle["drone_id"]
                cam_viewports = {}
                for side, cam_path in handle["stereo_cam_paths"].items():
                    vp_name = f"drone{did}_stereo_{side}"
                    vp_window = vp_utils.create_viewport_window(
                        vp_name,
                        width=STEREO_CAM_RESOLUTION[0],
                        height=STEREO_CAM_RESOLUTION[1],
                        visible=False,
                    )
                    vp_api = vp_window.viewport_api
                    vp_api.set_active_camera(cam_path)
                    cam_viewports[side] = {
                        "viewport_api": vp_api,
                        "viewport_window": vp_window,
                    }
                    print(f"  drone{did}: {side} viewport -> {cam_path}")
                handle["stereo_pub"].set_viewports(cam_viewports)

        # Warm up renderer
        print("  Warming up renderer...")
        for _ in range(10):
            world.step(render=True)

        # --- Print summary ---
        print("\n" + "=" * 70)
        print("  READY!")
        print("=" * 70)
        for handle in drone_handles:
            did = handle["drone_id"]
            print(f"\n  drone{did}:")
            if handle.get("arm_bridge"):
                print(f"    drone{did}/arm/joint_command")
                print(f"    drone{did}/arm/joint_states")
            if handle.get("stereo_pub"):
                print(f"    drone{did}/front_stereo_camera/left/image_rect_color")
                print(f"    drone{did}/front_stereo_camera/right/image_rect_color")
        print("\nPress Ctrl+C to exit\n")

        # --- Simulation loop ---
        pub_counter = 0
        while simulation_app.is_running():
            world.step(render=True)

            # Apply arm commands
            for bridge in arm_bridges:
                bridge.apply_commands()

            # Publish at ~10 Hz
            pub_counter += 1
            if pub_counter >= 6:
                pub_counter = 0
                sim_time = world.current_time if hasattr(world, 'current_time') else 0.0
                for bridge in arm_bridges:
                    bridge.publish_states(sim_time)
                for pub in stereo_pubs:
                    pub.capture_and_publish(sim_time)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        timeline.stop()
        for pub in stereo_pubs:
            pub.destroy_node()
        for bridge in arm_bridges:
            bridge.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
        simulation_app.close()
        print("Shutdown complete\n")
```

- [ ] **Step 2: Commit**

```bash
git add multi_uav/launch_multi_uav.py
git commit -m "Add spawning loop and simulation loop to launch script"
```

---

### Task 7: Manual verification — single drone (default config)

**Files:** None (verification only)

- [ ] **Step 1: Run with default config (single drone)**

```bash
cd /home/huojiaxi/Desktop/uav_sim
python multi_uav/launch_multi_uav.py
```

**Expected:**
- One drone spawns at origin with stereo cameras, arm, and landing legs
- PX4 SITL launches (vehicle_id=0)
- ROS2 topics visible: `drone0/arm/joint_command`, `drone0/arm/joint_states`, `drone0/front_stereo_camera/left/image_rect_color`, etc.

- [ ] **Step 2: Verify ROS2 topics**

In a separate terminal:

```bash
ros2 topic list | grep drone0
```

**Expected:** All 6 topics for drone0.

- [ ] **Step 3: Verify arm control**

```bash
ros2 topic pub --once drone0/arm/joint_command sensor_msgs/msg/JointState \
  "{name: ['shoulder_joint','elbow_joint'], position: [0.0, 0.0]}"
```

**Expected:** Arm unfolds in the simulator.

---

### Task 8: Manual verification — multi-drone (example config)

**Files:** None (verification only)

- [ ] **Step 1: Run with example config (two drones)**

```bash
cd /home/huojiaxi/Desktop/uav_sim
python multi_uav/launch_multi_uav.py --config multi_uav/config/example_config.yaml
```

**Expected:**
- Two drones spawn: drone0 at (0,0) yaw=0, drone1 at (2,0) yaw=90
- Both have arms, stereo cameras, landing legs
- Two PX4 SITL instances launch

- [ ] **Step 2: Verify ROS2 topics for both drones**

```bash
ros2 topic list | grep drone
```

**Expected:** 12 topics total (6 per drone).

- [ ] **Step 3: Test arm on each drone independently**

```bash
ros2 topic pub --once drone0/arm/joint_command sensor_msgs/msg/JointState \
  "{name: ['shoulder_joint','elbow_joint'], position: [0.0, 0.0]}"

ros2 topic pub --once drone1/arm/joint_command sensor_msgs/msg/JointState \
  "{name: ['shoulder_joint','elbow_joint'], position: [0.0, 0.0]}"
```

**Expected:** Each drone's arm unfolds independently.
