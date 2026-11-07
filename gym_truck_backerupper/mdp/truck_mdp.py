from dataclasses import dataclass, field
from typing import Protocol

from gymnasium.spaces import Box
import numpy as np


@dataclass(frozen=True)
class Snapshot:
    """Neutral view of the simulator state."""
    obs: np.ndarray # [tractor_heading_err, trailer_heading_err, trailer_lat_err]
    terminated: bool = False
    truncated: bool = False
    jackknife: bool = False
    out_of_bounds: bool = False    
    off_track: bool = False
    success: bool = False
    extras: dict = field(default_factory=dict)  # articulation, steering, velocity, dist_goal...
    geometry: dict = field(default_factory=dict) #Raw geometric information needed to construct the extended MDP state

class TruckBackend(Protocol):
    """Contract to be implemented for each simulator."""
    max_steer: float
    def reset(self, seed=None, options=None) -> Snapshot: ...
    def step(self, steer: float) -> Snapshot: ... # rad steer
    def close(self) -> None: ...

@dataclass
class RewardConfig:
    survival: float = 0.01
    w_lat: float = 0.5
    lat_scale: float = 5.0
    w_trailer: float = 0.5
    w_tractor: float = 0.1
    head_scale: float = np.radians(45.0)
    exponent: float = 0.4
    w_hitch: float = 0.2
    hitch_scale: float = np.radians(90.0)
    w_action: float = 0.05
    r_success: float = 100.0
    r_failure: float = -100.0
    r_timeout: float = -100.0

@dataclass
class ObsConfig:
    head_scale: float = np.radians(45.0)
    lat_scale: float = 5.0
    hitch_scale: float = np.radians(90.0)
    velocity_scale: float = 2.012
    preview_scale: float = 10.0
    goal_scale: float = 40.0
    goal_heading_scale: float = np.pi

def world_to_local(point, pose):
    """
    Convert a point from world coordinates to the trailer local frame.

    point = (x, y)
    pose  = (x_trailer, y_trailer, psi_trailer)
    """
    x, y = point
    x_trailer, y_trailer, psi_trailer = pose

    dx = x - x_trailer
    dy = y - y_trailer

    c = np.cos(psi_trailer)
    s = np.sin(psi_trailer)

    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy

    return np.array([x_local, y_local], dtype=np.float32)

def wrap_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))

class TruckMDP:    
    def __init__(self, backend: TruckBackend, gamma=0.99, reward_cfg=None, obs_cfg=None):
        self.backend = backend
        self.r_cfg = reward_cfg if reward_cfg is not None else RewardConfig()
        self.obs_cfg = obs_cfg if obs_cfg is not None else ObsConfig()
        self.gamma = gamma
        # Continuous action space based on the steering angle [-45°, 45°]
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)

    def info(self, snap: Snapshot):
        return {
            **snap.extras,
            "success": snap.success,
            "jackknife": snap.jackknife,
            "out_of_bounds": snap.out_of_bounds,
            "off_track": snap.off_track,
        }

    def observation(self, snap: Snapshot):
        c = self.obs_cfg

        # 1. Original simulator errors
        e_tractor, e_trailer, e_lat = snap.obs
        e_tractor /= c.head_scale
        e_trailer /= c.head_scale
        e_lat /= c.lat_scale

        # 2. Current vehicle configuration
        hitch_angle = wrap_angle(
            snap.extras["hitch_angle"]
        ) / c.hitch_scale

        steering = (
            snap.extras["steering"]
            / self.backend.max_steer
        )

        velocity = (
            snap.extras["velocity"]
            / c.velocity_scale
        )

        # 3. Future trajectory preview
        previews = (
            self.preview_points(snap)
            / c.preview_scale
        )

        # 4. Relative goal information
        goal = self.goal_relative(snap)

        goal_x = goal[0] / c.goal_scale
        goal_y = goal[1] / c.goal_scale
        goal_heading = goal[2] / c.goal_heading_scale

        state = np.array(
            [
                e_tractor,
                e_trailer,
                e_lat,

                hitch_angle,
                steering,
                velocity,

                previews[0],
                previews[1],
                previews[2],
                previews[3],
                previews[4],
                previews[5],

                goal_x,
                goal_y,
                goal_heading,
            ],
            dtype=np.float32
        )

        return state

    def reward(self, snap: Snapshot, a) -> float:
        c = self.r_cfg
        e_tractor, e_trailer, e_lat = (
            float(v) for v in snap.obs
        )
        hitch = wrap_angle(
            snap.extras["hitch_angle"]
        )
        r = c.survival

        # Trajectory tracking
        r -= c.w_lat * (
            abs(e_lat) / c.lat_scale
        ) ** c.exponent

        # Vehicle orientation
        r -= c.w_trailer * (
            abs(e_trailer) / c.head_scale
        ) ** c.exponent

        r -= c.w_tractor * (
            abs(e_tractor) / c.head_scale
        ) ** c.exponent

        # Articulation stability
        r -= c.w_hitch * (
            abs(hitch) / c.hitch_scale
        ) ** 2

        # Steering effort
        r -= c.w_action * float(a[0]) ** 2

        # Terminal rewards
        if snap.success:
            r += c.r_success
        elif snap.jackknife or snap.out_of_bounds or snap.off_track:
            r += c.r_failure
        elif snap.truncated:
            r += c.r_timeout

        return r
 
    def reset(self, seed=None, options=None):
        snap = self.backend.reset(seed=seed, options=options)
        return self.observation(snap), self.info(snap)
 
    def step(self, a):
        a = np.clip(
            np.asarray(a, dtype=np.float32).reshape(1),
            -1.0,
            1.0
        )

        snap = self.backend.step(
            float(a[0]) * self.backend.max_steer
        )

        return (
            self.observation(snap),
            self.reward(snap, a),
            snap.terminated,
            snap.truncated,
            self.info(snap)
        )

    def preview_points(self, snap: Snapshot):
        geometry = snap.geometry
        pose = geometry["trailer_pose"]
        path = geometry["path"]
        current_index = geometry["path_index"]
        preview_distances = (3.0, 6.0, 10.0)
        current_distance = path[current_index, 4]
        previews = []

        for distance in preview_distances:
            target_distance = current_distance + distance
            index = int(np.searchsorted(
                path[:, 4],
                target_distance
            ))
            index = min(index, len(path) - 1)
            point_world = path[index, :2]
            point_local = world_to_local(
                point_world,
                pose
            )
            previews.extend(point_local)
        return np.asarray(previews, dtype=np.float32)

    def goal_relative(self, snap: Snapshot):
        geometry = snap.geometry

        x_trailer, y_trailer, psi_trailer = geometry["trailer_pose"]
        goal = geometry["goal"]
        offset = geometry["goal_offset"]

        # Rear bumper position of the trailer
        x_bumper = x_trailer - offset * np.cos(psi_trailer)
        y_bumper = y_trailer - offset * np.sin(psi_trailer)

        # Goal position expressed in the trailer local frame,
        # using the rear bumper as origin
        goal_local = world_to_local(
            goal[:2],
            (x_bumper, y_bumper, psi_trailer)
        )

        # Final orientation error
        goal_heading_error = wrap_angle(
            goal[2] - psi_trailer
        )

        return np.array(
            [
                goal_local[0],
                goal_local[1],
                goal_heading_error
            ],
            dtype=np.float32
        )