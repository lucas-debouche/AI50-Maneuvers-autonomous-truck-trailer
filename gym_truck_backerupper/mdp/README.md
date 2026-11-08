# MDP Documentation — Autonomous Truck-Trailer

This document describes the design choices and entities that compose the Markov Decision Process (MDP) layer of the autonomous truck-trailer project. The goal of this layer is to bridge the raw physics simulator with any standard reinforcement learning (RL) library.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Entity: `Snapshot`](#2-entity-snapshot)
3. [Entity: `TruckBackend` Protocol](#3-entity-truckbackend-protocol)
4. [Entity: `TruckMDP`](#4-entity-truckmdp)
   - [Observation Space](#41-observation-space)
   - [Action Space](#42-action-space)
   - [Reward Function](#43-reward-function)
5. [Entity: `GymBackerUpperBackend`](#5-entity-gymbackerupperbackend)
6. [Entity: `TruckMDPEnv`](#6-entity-truckmdpenv)
7. [Configuration Dataclasses](#7-configuration-dataclasses)
8. [Helper Functions](#8-helper-functions)
9. [Design Rationale Summary](#9-design-rationale-summary)

---

## 1. Architecture Overview

The codebase is organized in **three distinct layers**:

```
┌─────────────────────────────────────────────────────────────┐
│          TruckBackerUpperEnv  (low-level simulator)         │
│  - Kinematic model (dopri5 ODE integrator)                  │
│  - Dubins path planning                                     │
│  - Raw observation: [psi_1_err, psi_2_err, lat_err]        │
└───────────────────────────┬─────────────────────────────────┘
                            │  wrapped by
                            ▼
┌─────────────────────────────────────────────────────────────┐
│        GymBackerUpperBackend  (adapter — wrap_env.py)       │
│  - Reads simulator internals (psi_1, x2, track_vector…)     │
│  - Packages them into a neutral Snapshot object             │
└───────────────────────────┬─────────────────────────────────┘
                            │  fed into
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              TruckMDP  (MDP layer — truck_mdp.py)           │
│  - Builds a rich 15-dim observation                         │
│  - Computes a shaped reward signal                          │
│  - Normalizes actions                                       │
└───────────────────────────┬─────────────────────────────────┘
                            │  re-exposed as
                            ▼
┌─────────────────────────────────────────────────────────────┐
│          TruckMDPEnv  (Gymnasium wrapper — wrap_env.py)     │
│  - Standard gym.Env interface                               │
│  - Plug-and-play with any RL library (SB3, CleanRL…)        │
└─────────────────────────────────────────────────────────────┘
```

**Key design principle**: the `TruckMDP` knows nothing about the simulator internals. It only interacts with a `TruckBackend` and consumes `Snapshot` objects. This makes it trivially possible to swap the simulator without touching the MDP logic.

---

## 2. Entity: `Snapshot`

**File**: `truck_mdp.py`

```python
@dataclass(frozen=True)
class Snapshot:
    obs: np.ndarray  # [tractor_heading_err, trailer_heading_err, trailer_lat_err]
    terminated: bool
    truncated: bool
    jackknife: bool
    out_of_bounds: bool
    off_track: bool
    success: bool
    extras: dict     # hitch_angle, steering, velocity, min_dist_goal, t
    geometry: dict   # trailer_pose, path, path_index, goal, goal_offset
```

The `Snapshot` is an **immutable, simulator-agnostic view** of the current environment state. It is the central data contract between the simulator adapter and the MDP.

### Design choices

- **`frozen=True`**: Snapshots are immutable by design. Once created by the backend, neither the MDP nor anything else should modify them. This prevents subtle bugs where the same snapshot object is mutated between `observation()` and `reward()` calls.

- **`obs`**: Contains the three scalar errors that the low-level simulator natively computes — tractor heading error, trailer heading error, and trailer lateral error. These are the minimal signals available from the physics engine.

- **`extras`**: A flexible dictionary for scalar quantities that the MDP needs but that are not part of the simulator's native observation. These include the **hitch angle** (articulation between tractor and trailer), the current **steering angle**, and the current **velocity**.

- **`geometry`**: Raw geometric data needed to construct the extended MDP state — the trailer pose in world coordinates, the full reference path (`track_vector`), the current progress index along that path, the goal configuration, and the goal offset (distance from trailer center to rear bumper).

The separation of `extras` and `geometry` reflects their different roles: `extras` are scalars used directly in the observation and reward, while `geometry` is structural data used to compute derived quantities (preview points, relative goal).

---

## 3. Entity: `TruckBackend` Protocol

**File**: `truck_mdp.py`

```python
class TruckBackend(Protocol):
    max_steer: float
    def reset(self, seed=None, options=None) -> Snapshot: ...
    def step(self, steer: float) -> Snapshot: ...
    def close(self) -> None: ...
```

`TruckBackend` is a **structural interface** (Python `Protocol`) that any simulator adapter must satisfy. It defines the minimum contract required by the `TruckMDP`:

| Member | Description |
|--------|-------------|
| `max_steer` | Maximum steering angle in radians (used to normalize actions) |
| `reset()` | Initializes the simulator and returns the initial `Snapshot` |
| `step(steer)` | Advances the simulation by one timestep with the given steering command (in radians) and returns the resulting `Snapshot` |
| `close()` | Releases resources |

### Design choice

Using a `Protocol` instead of an abstract base class means that any object satisfying this interface can be used as a backend — **no inheritance required**. This is structural typing (duck typing with static analysis support), making it easy to write lightweight mock backends for testing or to integrate a completely different simulator.

The `step()` method receives the steering angle directly in **physical units (radians)**, not normalized. The normalization/denormalization is handled by `TruckMDP.step()`, which keeps the backend contract as simple as possible.

---

## 4. Entity: `TruckMDP`

**File**: `truck_mdp.py`

`TruckMDP` is the central entity of the layer. It is **not** a `gym.Env` itself — it is a pure MDP object that holds the observation, reward, and transition logic. It delegates state transitions to the backend and translates `Snapshot` objects into RL-friendly tensors.

```python
class TruckMDP:
    def __init__(self, backend: TruckBackend, gamma=0.99,
                 reward_cfg=None, obs_cfg=None): ...
    def reset(self, seed, options) -> (obs, info): ...
    def step(self, a)  -> (obs, reward, terminated, truncated, info): ...
    def observation(self, snap: Snapshot) -> np.ndarray: ...
    def reward(self, snap: Snapshot, a) -> float: ...
```

### 4.1 Observation Space

The observation space is a **15-dimensional continuous vector**, all components normalized to be roughly in `[-1, 1]`:

```
state = [
    e_tractor,          # tractor heading error      / 45°
    e_trailer,          # trailer heading error       / 45°
    e_lat,              # trailer lateral error       / 5 m

    hitch_angle,        # articulation angle          / 90°
    steering,           # current steering command    / max_steer
    velocity,           # longitudinal velocity       / 2.012 m/s

    preview_x_3m,       # trajectory preview at 3 m  (trailer local frame) / 10 m
    preview_y_3m,
    preview_x_6m,       # trajectory preview at 6 m  (trailer local frame) / 10 m
    preview_y_6m,
    preview_x_10m,      # trajectory preview at 10 m (trailer local frame) / 10 m
    preview_y_10m,

    goal_x,             # goal position x            (trailer local frame) / 40 m
    goal_y,             # goal position y            (trailer local frame) / 40 m
    goal_heading,       # goal heading error                               / π
]
```

**Why 15 dimensions?** The raw simulator observation has only 3 values. While sufficient to close a simple tracking loop, 3 values do not give the agent any sense of what the road will do next or how far the goal is. The 15-dim observation adds two key categories of information:

#### Preview Points (indices 6–11)

Computed by `preview_points()`. Three waypoints are sampled from the reference path at cumulative arc-length distances of **3 m, 6 m, and 10 m** ahead of the trailer's current position. Each point is projected into the **trailer's local frame** using `world_to_local()`.

This gives the agent an **ego-centric view of upcoming trajectory curvature**, which is essential for anticipating turns. Because the points are expressed in the local frame, the representation is invariant to global position and heading — the agent sees the road relative to itself, not relative to the world origin.

#### Relative Goal (indices 12–14)

Computed by `goal_relative()`. The goal is expressed in the trailer's local frame, **anchored at the rear bumper** of the trailer (offset of `trail = 2.0 m` from the trailer center). This is the physically meaningful docking point.

The heading component is the angular error between the goal orientation and the current trailer heading. Together, these three values give the agent a persistent sense of direction toward the final objective.

#### Normalization

All components are divided by a physically meaningful scale factor (from `ObsConfig`). This keeps the observation in a consistent numerical range, which is important for gradient-based optimization: unnormalized observations with very different scales would cause the neural network to focus disproportionately on large-magnitude inputs.

---

### 4.2 Action Space

```python
self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
```

The agent outputs a **normalized steering command** in `[-1, 1]`. Inside `step()`, it is mapped to physical units:

```python
steer_rad = a[0] * self.backend.max_steer   # max_steer = 45° = π/4 rad
```

The action is clipped before being passed to the backend. Working in normalized space simplifies hyperparameter tuning — action bounds are always `±1` regardless of the simulator's physical units.

---

### 4.3 Reward Function

The reward is computed by `reward()` and is a sum of several terms, governed by `RewardConfig`:

```
r = survival
  - w_lat    × (|e_lat|    / lat_scale)^exponent      # lateral tracking error
  - w_trailer × (|e_trailer| / head_scale)^exponent   # trailer heading error
  - w_tractor × (|e_tractor| / head_scale)^exponent   # tractor heading error
  - w_hitch  × (|hitch|   / hitch_scale)^2            # jackknife prevention
  - w_action × a²                                     # steering effort penalty

  + r_success  if success
  + r_failure  if jackknife / out_of_bounds / off_track
  + r_timeout  if truncated (time limit reached)
```

**Default weights:**

| Term | Weight | Scale |
|------|--------|-------|
| Survival bonus | +0.01 / step | — |
| Lateral error | 0.5 | 5 m |
| Trailer heading | 0.5 | 45° |
| Tractor heading | 0.1 | 45° |
| Hitch angle | 0.2 | 90° |
| Steering effort | 0.05 | — |
| Success | +100 | — |
| Failure | −100 | — |
| Timeout | −100 | — |

#### Why `exponent = 0.4`?

The error terms use a **concave power function** (fractional exponent < 1) instead of the quadratic penalty used in classic LQR. With a concave function:

- Small errors are penalized **relatively more** than with a quadratic, discouraging the agent from tolerating even small tracking deviations.
- Large errors are penalized **relatively less**, which provides smoother gradients and avoids exploding penalties that could destabilize training in early episodes.

This is in contrast to the base simulator's native reward, which uses `exponent = 0.4` for some terms and a full quadratic LQR cost `s^T Q s + a^T R a` for others. The MDP reward is **fully decoupled** from the simulator's internal reward.

#### Why separate trailer and tractor heading terms?

The primary control objective is to guide the **trailer** along the reference path. However, allowing the tractor to adopt arbitrary orientations relative to the trailer would lead to unstable, jackknife-prone configurations. The smaller tractor weight (`w_tractor = 0.1` vs. `w_trailer = 0.5`) encourages alignment without over-constraining the agent.

#### Why a quadratic penalty on hitch angle?

The hitch angle penalty uses `exponent = 2` (quadratic), not the same `0.4` used for heading and lateral errors. A quadratic penalty **grows rapidly** as the articulation angle approaches its physical limit (±90°), creating a strong repulsive force that keeps the agent away from jackknife configurations.

#### Survival bonus

The small `+0.01` per-step bonus encourages the agent to **stay alive** as long as possible, which implicitly rewards reaching distant goals rather than terminating early.

---

## 5. Entity: `GymBackerUpperBackend`

**File**: `wrap_env.py`

```python
class GymBackerUpperBackend:
    def __init__(self, env, velocity=None, course=None): ...
    def reset(self, seed=None, options=None) -> Snapshot: ...
    def step(self, steer: float) -> Snapshot: ...
    def close(self): ...
```

This class is the **only component that knows the internals of `TruckBackerUpperEnv`**. It implements `TruckBackend` and translates the simulator's native output into `Snapshot` objects.

### Key responsibilities

#### Building `extras`

| Key | Source in simulator | Meaning |
|-----|---------------------|---------|
| `hitch_angle` | `s.psi_1[i] - s.psi_2[i]` | Articulation angle between tractor and trailer |
| `steering` | `s.u` | Current steering command (physical rad) |
| `velocity` | `s.v1x` | Longitudinal velocity of the tractor |
| `min_dist_goal` | `info["min_d"]` | Minimum distance to goal seen so far |
| `t` | `s.t[i]` | Current simulation time |

#### Building `geometry`

| Key | Source | Meaning |
|-----|--------|---------|
| `trailer_pose` | `(s.x2[i], s.y2[i], s.psi_2[i])` | Trailer center position and heading |
| `path` | `s.track_vector` | Full Dubins reference path (N×5 array) |
| `path_index` | `s.last_index` | Current progress index along the path |
| `goal` | `s.qg[:3]` | Goal configuration `[x, y, psi]` |
| `goal_offset` | `s.trail` | Rear bumper offset (2.0 m) |

#### `off_track` mapping

The simulator flags two distinct error conditions: `dist_too_large` (lateral deviation > 5 m) and `angle_too_large` (heading error > 45°). Both map to the single `off_track` flag in the `Snapshot`, which the MDP treats as a terminal failure.

#### Course reset issue

When a fixed `course` (start and goal configurations) is provided, the backend **reassigns `q0` and `qg` at every reset**. This is necessary because the simulator modifies these arrays **in-place** during `reset()` (it adds `+π` to orientations when driving in reverse). Without reassignment, the first reset would corrupt the course for all subsequent resets.

---

## 6. Entity: `TruckMDPEnv`

**File**: `wrap_env.py`

```python
class TruckMDPEnv(gym.Env):
    def __init__(self, backend, **mdp_kwargs): ...
    def reset(self, *, seed=None, options=None): ...
    def step(self, action): ...
    def close(self): ...
```

A thin `gymnasium.Env` wrapper that exposes `TruckMDP` through the standard Gymnasium interface. All logic is delegated to `self.mdp`.

### Design choice

The `TruckMDP` is **not** a `gym.Env` by itself — this is intentional. By keeping the MDP as a pure Python object, it can be used:
- As a standard Gymnasium environment (via `TruckMDPEnv`)
- Directly in custom training loops without the Gymnasium overhead
- As a component in a vectorized environment stack

`TruckMDPEnv` simply forwards `action_space` and `observation_space` from the MDP, making it trivially compatible with any RL library that expects a `gym.Env`.

---

## 7. Configuration Dataclasses

### `RewardConfig`

Controls the shape of the reward function. All weights and scales can be tuned without touching the reward logic.

```python
@dataclass
class RewardConfig:
    survival: float = 0.01       # per-step survival bonus
    w_lat: float = 0.5           # weight: lateral error
    lat_scale: float = 5.0       # normalization: lateral error (m)
    w_trailer: float = 0.5       # weight: trailer heading error
    w_tractor: float = 0.1       # weight: tractor heading error
    head_scale: float = 45°      # normalization: heading errors
    exponent: float = 0.4        # power for heading/lateral terms
    w_hitch: float = 0.2         # weight: hitch angle (jackknife)
    hitch_scale: float = 90°     # normalization: hitch angle
    w_action: float = 0.05       # weight: steering effort
    r_success: float = 100.0     # terminal reward on success
    r_failure: float = -100.0    # terminal reward on failure
    r_timeout: float = -100.0    # terminal reward on timeout
```

### `ObsConfig`

Controls the normalization scales applied to each component of the observation vector.

```python
@dataclass
class ObsConfig:
    head_scale: float = 45°      # heading error normalization
    lat_scale: float = 5.0       # lateral error normalization (m)
    hitch_scale: float = 90°     # hitch angle normalization
    velocity_scale: float = 2.012 # velocity normalization (m/s, nominal speed)
    preview_scale: float = 10.0  # preview point normalization (m)
    goal_scale: float = 40.0     # goal position normalization (m)
    goal_heading_scale: float = π # goal heading normalization
```

---

## 8. Helper Functions

### `world_to_local(point, pose)`

Converts a 2D point from world coordinates to a local frame defined by `pose = (x, y, psi)`.

```
dx = point_x - pose_x
dy = point_y - pose_y

x_local =  cos(psi) * dx + sin(psi) * dy
y_local = -sin(psi) * dx + cos(psi) * dy
```

This is a standard 2D rotation by `-psi` (inverse rotation of the frame). It is used both for preview points (expressed in trailer frame) and for the goal position (expressed in trailer rear-bumper frame).

**Why ego-centric coordinates?** A neural network generalizes much better when the input representation is invariant to global position and orientation. By expressing everything in the trailer's local frame, two identical local situations (same road curvature ahead, same goal distance) produce identical observation vectors regardless of where in the world the truck currently is.

### `wrap_angle(angle)`

Wraps an angle to `[-π, π]` using the `arctan2(sin, cos)` identity. Used to normalize the hitch angle and goal heading error, ensuring they remain in a consistent range regardless of accumulated rotation.

---

## 9. Design Rationale Summary

| Design Decision | Rationale |
|-----------------|-----------|
| `Snapshot` as immutable data contract | Decouples the MDP from the simulator; prevents accidental mutation between observation and reward computation |
| `TruckBackend` as a Protocol | Structural typing: any simulator satisfying the interface can be used without inheritance; makes mocking and testing easy |
| 15-dim observation (vs. 3 raw) | Adds anticipation (preview) and global sense of direction (goal); both are critical for learning complex maneuvers |
| Ego-centric observation (local frame) | Representation invariant to global pose; improves generalization across different trajectory shapes and positions |
| Normalized actions in `[-1, 1]` | Uniform action range regardless of physical units; simplifies hyperparameter tuning across simulators |
| Concave reward exponent (0.4) | Stronger gradient signal for small errors; smoother gradient for large errors compared to quadratic |
| Quadratic hitch penalty | Rapidly growing cost as the articulation limit is approached; acts as a safety barrier against jackknife |
| `TruckMDP` separate from `gym.Env` | Can be used in custom loops, Gymnasium wrappers, or vectorized stacks; avoids coupling to Gymnasium API internals |
| `GymBackerUpperBackend` as the only simulator-aware code | All simulator-specific knowledge is isolated in one class; swapping the simulator only requires rewriting this adapter |
| Course re-assignment on every reset | The simulator mutates `q0/qg` in-place during initialization (reverse-driving offset); re-assignment ensures deterministic resets with a fixed course |
