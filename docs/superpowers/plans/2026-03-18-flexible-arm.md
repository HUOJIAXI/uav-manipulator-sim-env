# 2-DOF Flexible Arm Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 2-DOF pitch-pitch folding arm to the Iris UAV underside in Isaac Sim.

**Architecture:** Two capsule links connected by revolute joints with position-drive control, inserted into the existing launch script between the landing legs and stereo camera sections.

**Tech Stack:** USD Physics (UsdPhysics, UsdGeom), Isaac Sim, Pegasus Simulator

**Spec:** `docs/superpowers/specs/2026-03-18-flexible-arm-design.md`

---

### Task 1: Add arm link prims with physics

**Files:**
- Modify: `launch_stereo_default_with_arm.py:138-139` (insert after "Landing legs added", before stereo camera section)

- [ ] **Step 1: Add arm constants and create upper link capsule**

Insert after line 138 (`print("  Landing legs added\n")`), before the stereo camera section:

```python
# ------------------------------------------------------------------
# Add 2-DOF folding arm under the body
# ------------------------------------------------------------------
print("Adding 2-DOF folding arm...")

ARM_LINK_LENGTH = 0.125      # 12.5 cm per link
ARM_LINK_RADIUS = 0.008      # 8 mm radius
ARM_LINK_MASS = 0.05          # 50 g per link
ARM_HALF_LEN = ARM_LINK_LENGTH / 2.0  # 0.0625 m
SHOULDER_MOUNT_Z = -0.02     # mount point below body center
JOINT_STIFFNESS = 100.0
JOINT_DAMPING = 10.0

# -- Upper link --
upper_path = f"{body_path}/arm_upper_link"
upper_capsule = UsdGeom.Capsule.Define(stage, upper_path)
upper_capsule.GetHeightAttr().Set(ARM_LINK_LENGTH)
upper_capsule.GetRadiusAttr().Set(ARM_LINK_RADIUS)
upper_capsule.GetAxisAttr().Set("Z")
upper_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(0.8, 0.2, 0.2)])  # red

upper_prim = upper_capsule.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(upper_prim)
UsdPhysics.CollisionAPI.Apply(upper_prim)
mass_api = UsdPhysics.MassAPI.Apply(upper_prim)
mass_api.GetMassAttr().Set(ARM_LINK_MASS)
```

- [ ] **Step 2: Create lower link capsule**

```python
# -- Lower link --
lower_path = f"{body_path}/arm_lower_link"
lower_capsule = UsdGeom.Capsule.Define(stage, lower_path)
lower_capsule.GetHeightAttr().Set(ARM_LINK_LENGTH)
lower_capsule.GetRadiusAttr().Set(ARM_LINK_RADIUS)
lower_capsule.GetAxisAttr().Set("Z")
lower_capsule.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.8)])  # blue

lower_prim = lower_capsule.GetPrim()
UsdPhysics.RigidBodyAPI.Apply(lower_prim)
UsdPhysics.CollisionAPI.Apply(lower_prim)
mass_api_lower = UsdPhysics.MassAPI.Apply(lower_prim)
mass_api_lower.GetMassAttr().Set(ARM_LINK_MASS)
```

- [ ] **Step 3: Run `simulation_app.update()` to confirm prims are created**

```python
simulation_app.update()
print("  Arm link prims created")
```

### Task 2: Add revolute joints with drive control

**Files:**
- Modify: `launch_stereo_default_with_arm.py` (continuing in the same insertion block)

- [ ] **Step 1: Create shoulder revolute joint (body <-> upper link)**

```python
# -- Shoulder joint (body <-> upper link) --
shoulder_path = f"{body_path}/shoulder_joint"
shoulder_joint = UsdPhysics.RevoluteJoint.Define(stage, shoulder_path)
shoulder_joint.GetAxisAttr().Set("Y")
shoulder_joint.GetLowerLimitAttr().Set(-90.0)
shoulder_joint.GetUpperLimitAttr().Set(90.0)

# Connect body0=drone body, body1=upper link
shoulder_joint.GetBody0Rel().SetTargets([body_path])
shoulder_joint.GetBody1Rel().SetTargets([upper_path])

# Anchor positions
shoulder_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, SHOULDER_MOUNT_Z))
shoulder_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, ARM_HALF_LEN))

# No rotation on anchors (Y-axis is already correct)
shoulder_joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
shoulder_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

# Position drive — folded at +90 deg
shoulder_drive = UsdPhysics.DriveAPI.Apply(shoulder_joint.GetPrim(), "angular")
shoulder_drive.GetTypeAttr().Set("force")
shoulder_drive.GetStiffnessAttr().Set(JOINT_STIFFNESS)
shoulder_drive.GetDampingAttr().Set(JOINT_DAMPING)
shoulder_drive.GetTargetPositionAttr().Set(90.0)  # folded: upper link forward

print("  Shoulder joint created (target: 90 deg)")
```

- [ ] **Step 2: Create elbow revolute joint (upper link <-> lower link)**

```python
# -- Elbow joint (upper link <-> lower link) --
elbow_path = f"{body_path}/elbow_joint"
elbow_joint = UsdPhysics.RevoluteJoint.Define(stage, elbow_path)
elbow_joint.GetAxisAttr().Set("Y")
elbow_joint.GetLowerLimitAttr().Set(-180.0)
elbow_joint.GetUpperLimitAttr().Set(0.0)

# Connect body0=upper link, body1=lower link
elbow_joint.GetBody0Rel().SetTargets([upper_path])
elbow_joint.GetBody1Rel().SetTargets([lower_path])

# Anchor positions
elbow_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -ARM_HALF_LEN))
elbow_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, ARM_HALF_LEN))

elbow_joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
elbow_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

# Position drive — folded at -180 deg
elbow_drive = UsdPhysics.DriveAPI.Apply(elbow_joint.GetPrim(), "angular")
elbow_drive.GetTypeAttr().Set("force")
elbow_drive.GetStiffnessAttr().Set(JOINT_STIFFNESS)
elbow_drive.GetDampingAttr().Set(JOINT_DAMPING)
elbow_drive.GetTargetPositionAttr().Set(-180.0)  # folded: lower link back

simulation_app.update()
print("  Elbow joint created (target: -180 deg)")
print("  2-DOF folding arm ready (folded state)\n")
```

### Task 3: Verify in simulation

- [ ] **Step 1: Launch the simulation**

Run: `cd /home/huojiaxi/Desktop/uav_sim && ./python.sh launch_stereo_default_with_arm.py`

Expected:
- No errors during arm creation
- Two colored capsule links visible folded flat under the drone body
- Arm stays folded when drone is on the ground
- Drone flies normally with PX4
