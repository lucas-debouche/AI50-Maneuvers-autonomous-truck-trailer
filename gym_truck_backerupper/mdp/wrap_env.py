# TODO : Wrap the MDP to the simulator

import gymnasium as gym
import numpy as np
from mdp.truck_mdp import Snapshot, TruckMDP


class GymBackerUpperBackend:
    """Adapter for gym-truck-backerupper. The only part that needs to be rewritten for another simulator."""

    def __init__(self, env, velocity=None, course=None):
        self.env = env
        self.sim = env.unwrapped
        self.max_steer = float(self.sim.max_action) # rad (45°)
        self.course = course
        if velocity is not None:
            self.sim.manual_velocity(velocity)

    def _snapshot(self, obs, info, terminated=False, truncated=False) -> Snapshot:
        s, i = self.sim, self.sim.sim_i - 1
        return Snapshot(
            obs=np.asarray(obs, dtype=np.float32),
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=bool(info.get("goal", False)),
            jackknife=bool(info.get("jackknife", False)),
            out_of_bounds=bool(info.get("out_of_bounds", False)),
            off_track=bool(info.get("dist_too_large", False) or info.get("angle_too_large", False)),
            extras={
                "hitch_angle": float(s.psi_1[i] - s.psi_2[i]),
                "steering": float(np.asarray(s.u).ravel()[0]),
                "velocity": float(s.v1x),
                "min_dist_goal": float(info.get("min_d", np.nan)),
                "t": float(s.t[i]),
            },
        )

    def reset(self, seed=None, options=None) -> Snapshot:
        if self.course is not None:
            # q0/qg is recalculated at every reset: the simulator modifies these arrays on the fly
            # (+pi in reverse), which would alter the trajectory from one reset to the next.
            (x0, y0, p0), (xg, yg, pg) = self.course
            self.sim.q0 = np.array([x0, y0, np.radians(p0)], dtype=float)
            self.sim.qg = np.array([xg, yg, np.radians(pg)], dtype=float)
            self.sim.manual_track = True
        obs, info = self.env.reset(seed=seed, options=options)
        return self._snapshot(obs, info)
 
    def step(self, steer: float) -> Snapshot:
        a = np.array([steer], dtype=np.float32)
        obs, _, terminated, truncated, info = self.env.step(a)
        return self._snapshot(obs, info, terminated, truncated)
 
    def close(self):
        self.env.close()


class TruckMDPEnv(gym.Env):
    """Exposes the MDP as a gymnasium env."""

    metadata = {"render_modes": []}
 
    def __init__(self, backend, **mdp_kwargs):
        self.mdp = TruckMDP(backend, **mdp_kwargs)
        self.action_space = self.mdp.action_space
        self.observation_space = self.mdp.observation_space
 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.mdp.reset(seed=seed, options=options)
 
    def step(self, action):
        return self.mdp.step(action)
 
    def close(self):
        self.mdp.backend.close()