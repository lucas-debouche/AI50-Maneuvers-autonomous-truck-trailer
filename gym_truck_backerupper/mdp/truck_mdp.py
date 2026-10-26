from dataclasses import dataclass, field
from typing import Protocol

from gymnasium.spaces import Box
import numpy as np


@dataclass(frozen=True)
class Snapshot:
    """Neutral vue of the simulator state."""
    obs: np.ndarray # [tractor_heading_err, trailer_heading_err, trailer_lat_err]
    terminated: bool = False
    truncated: bool = False
    jackknife: bool = False
    out_of_bounds: bool = False    
    off_track: bool = False
    success: bool = False
    extras: dict = field(default_factory=dict)  # articulation, steering, velocity, dist_goal...

class TruckBackend(Protocol):
    """Contract to be implemented for each simulator."""
    max_steer: float
    def reset(self, seed=None, options=None) -> Snapshot: ...
    def step(self, steer: float) -> Snapshot: ... # rad steer
    def close(self) -> None: ...

@dataclass
class RewardConfig:
    survival: float = 1.0
    w_lat: float = 0.5
    lat_scale: float = 5.0
    w_trailer: float = 0.5
    w_tractor: float = 0.1
    head_scale: float = np.radians(45.0)
    exponent: float = 0.4
    w_action: float = 0.05
    r_success: float = 100.0
    r_failure: float = -100.0


class TruckMDP:    
    def __init__(self, backend: TruckBackend, gamma=0.99, reward_cfg=RewardConfig()):
        self.backend = backend
        self.r_cfg = reward_cfg
        self.gamma=gamma
        # Continuous action space based on the steering angle [-45°, 45°]
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32) # Normalized action, steering = a * max_steer
        self.observation_space = Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

    def observation(self, snap: Snapshot):
        return np.asarray(snap.obs, dtype=np.float32)

    def reward(self, snap: Snapshot, a) -> float:
        c = self.r_cfg
        e_tractor, e_trailer, e_lat = (float(v) for v in snap.obs)
        r = c.survival
        r -= c.w_lat * (abs(e_lat) / c.lat_scale) ** c.exponent
        r -= c.w_trailer * (abs(e_trailer) / c.head_scale) ** c.exponent
        r -= c.w_tractor * (abs(e_tractor) / c.head_scale) ** c.exponent
        r -= c.w_action * float(a[0]) ** 2
        if snap.success:
            r += c.r_success
        if snap.jackknife or snap.out_of_bounds or snap.off_track:
            r += c.r_failure
        return r
 
    def reset(self, seed=None, options=None):
        snap = self.backend.reset(seed=seed, options=options)
        return self.observation(snap), dict(snap.extras)
 
    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float32).reshape(1), -1.0, 1.0)
        snap = self.backend.step(float(a[0]) * self.backend.max_steer)
        return (self.observation(snap), self.reward(snap, a),
                snap.terminated, snap.truncated, dict(snap.extras))