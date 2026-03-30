# Ground Robot with Simulated UWB Position Publishing

## Overview

Add a wheeled mobile ground robot to the multi-UAV Isaac Sim simulation. The robot is a physical USD asset controlled via ROS2 `cmd_vel` and publishes its position with configurable Gaussian noise to simulate UWB tag measurements. UAVs can subscribe to this noisy position to track the ground robot.

## Components

### 1. `multi_uav/spawn_ground_robot.py`

New module containing:

#### `spawn_ground_robot(world, robot_id, x, y, yaw)`

- Loads a built-in Isaac Sim wheeled robot USD asset (Jetbot or similar from Nucleus/Isaac asset library)
- Places it at `(x, y, 0)` with the given yaw on the ground plane
- Returns the robot prim path

#### `GroundRobotBridge` (ROS2 Node)

Handles all ROS2 communication for one ground robot.

**Subscriptions:**
- `ground_robot{id}/cmd_vel` (geometry_msgs/Twist) — linear.x, linear.y, angular.z used to drive the robot

**Publications (10 Hz, every 6 sim steps):**
- `ground_robot{id}/uwb/position` (geometry_msgs/PointStamped) — ground truth position + Gaussian noise on x, y, z
- `ground_robot{id}/state/pose` (geometry_msgs/PoseStamped) — clean ground truth for debugging/validation

**Parameters:**
- `uwb_noise_std` (float): standard deviation in meters for Gaussian noise on each axis, default 0.1

#### Velocity Application

Each physics step:
1. Read latest `cmd_vel` from the subscription callback (thread-safe with a lock)
2. Get the robot's current world orientation from USD
3. Transform linear velocity from body frame to world frame using yaw
4. Set the rigid body's linear and angular velocity via `UsdPhysics` or `PhysxAPI`

#### UWB Noise Model

Every publish cycle (10 Hz):
1. Read ground truth position from USD prim world transform
2. Add independent Gaussian noise: `x += N(0, std^2)`, `y += N(0, std^2)`, `z += N(0, std^2)`
3. Publish as `PointStamped` with simulation timestamp

### 2. Configuration

Add `ground_robots` section to YAML config files.

**Schema:**
```yaml
ground_robots:          # optional, default: empty list
  - id: 0              # unique integer ID
    x: 3.0             # spawn X position (meters)
    y: 0.0             # spawn Y position (meters)
    yaw: 0.0           # initial heading (degrees)
    uwb_noise_std: 0.1 # UWB noise std dev (meters), default 0.1
```

**Validation:**
- Ground robot IDs must be unique
- Warn if a ground robot spawns within 1m of any drone spawn position

### 3. Integration in `launch_multi_uav.py`

**Phase 2 (Spawn):** After spawning drones, spawn ground robots from config.

**Phase 5 (ROS2 Setup):** Create `GroundRobotBridge` node per ground robot, add to the executor.

**Phase 8 (Main Loop):** Each step:
- Call `bridge.apply_velocity()` to set rigid body velocity from latest `cmd_vel`
- Every 6 steps: call `bridge.publish_uwb(sim_time, stage)` and `bridge.publish_pose(sim_time, stage)`

### 4. USD Prim Hierarchy

```
/World/ground_robot{id}/
  └── (loaded USD asset hierarchy — e.g., Jetbot chassis, wheels, etc.)
```

## ROS2 Topic Summary

| Topic | Type | Direction | Rate |
|-------|------|-----------|------|
| `ground_robot{id}/cmd_vel` | geometry_msgs/Twist | Subscribe | — |
| `ground_robot{id}/uwb/position` | geometry_msgs/PointStamped | Publish | 10 Hz |
| `ground_robot{id}/state/pose` | geometry_msgs/PoseStamped | Publish | 10 Hz |

## Asset Selection

Use a built-in Isaac Sim wheeled robot asset. Candidates in order of preference:
1. **Jetbot** — simple 2-wheel differential drive, small footprint, available in Isaac Sim assets
2. **Carter** — larger wheeled robot, also available in Isaac Sim

The exact asset path will be resolved at implementation time by checking available Nucleus assets.

## Out of Scope

- Obstacle avoidance or autonomous navigation on the ground robot
- Realistic UWB multipath/NLOS error modeling (Gaussian only)
- AprilTag visual tracking (future work)
- Multiple UWB tags per robot
