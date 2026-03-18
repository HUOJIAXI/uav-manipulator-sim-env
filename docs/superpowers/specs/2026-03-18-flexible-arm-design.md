# 2-DOF Flexible Arm Design for Iris UAV

## Overview

Add a 2-DOF pitch-pitch articulated arm to the underside of the Iris quadrotor. The arm has two revolute joints (shoulder and elbow), each rotating about the Y-axis. Total arm length is 25 cm (two 12.5 cm links). The arm folds flat under the body in its initial state and deploys straight down when extended.

## Purpose

General-purpose manipulator for future grasping tasks. This iteration focuses on the mechanical structure and joint control — no end-effector yet.

## Physical Parameters

| Parameter            | Value         |
|----------------------|---------------|
| Upper link length    | 0.125 m       |
| Lower link length    | 0.125 m       |
| Link radius          | 0.008 m       |
| Link mass            | 0.05 kg each  |
| Shoulder mount point | (0, 0, -0.02) in body frame |
| Joint axis           | Y (both)      |
| Shoulder range       | [-90, +90] deg |
| Elbow range          | [-180, 0] deg  |
| Joint stiffness      | 100.0 (force mode, N*m/rad) |
| Joint damping        | 10.0 (force mode, N*m*s/rad) |

## Prim Hierarchy

```
/World/quadrotor/body/
  arm_upper_link        (Capsule, rigid body)
  arm_lower_link        (Capsule, rigid body)
  shoulder_joint        (Revolute: body <-> upper_link)
  elbow_joint           (Revolute: upper_link <-> lower_link)
```

Joints are relationship-based connectors — prim tree location does not determine physics connectivity. All placed as siblings under `body/` for simplicity.

## Physics APIs per Link

Each link requires:
- `UsdGeom.Capsule` with axis="Z", height and radius as specified
- `UsdPhysics.RigidBodyAPI` (enables dynamics)
- `UsdPhysics.MassAPI` with mass = 0.05 kg
- `UsdPhysics.CollisionAPI` (enables collision)

## Joint Anchor Positions

Capsule axis is Z. Half-height of each link = 0.0625 m.

| Joint     | localPos0 (parent body)  | localPos1 (child link)   |
|-----------|--------------------------|--------------------------|
| Shoulder  | (0, 0, -0.02) on body   | (0, 0, +0.0625) on upper |
| Elbow     | (0, 0, -0.0625) on upper | (0, 0, +0.0625) on lower |

## Folded State (Initial)

- Shoulder joint: +90 deg (upper link points forward along +X)
- Elbow joint: -180 deg (lower link folds back along -X)
- Both links lie at approximately Z = -0.02 to -0.04, horizontally under the body
- Fits within the 20 cm vertical clearance provided by the landing legs

## Unfolded State (Deployed)

- Both joints at 0 deg
- Arm hangs straight down, end-effector at Z = -0.27 from body center
- Arm should only be deployed during flight; ground clearance is tight when landed

## Joint Implementation

Each joint uses:
- `UsdPhysics.RevoluteJoint` connecting two rigid body links via body0/body1 relationships
- `UsdPhysics.DriveAPI` (force mode) with position target for stiffness-based control
- Initial position targets set to folded angles
- Elbow constrained to [-180, 0] deg to prevent self-collision with drone body

The arm joins the quadrotor's existing articulation tree — no separate ArticulationRootAPI needed.

## Integration Notes

- All arm prims created after `world.reset()` and before camera setup
- Arm links are siblings under `/World/quadrotor/body/`
- Spawn height at 0.30 m (sufficient for folded arm; deploy only in flight)

## Future Extensions

- Add gripper end-effector to lower link tip
- ROS2 JointState publisher/subscriber for arm control
- Coupled UAV-manipulator stability controller
