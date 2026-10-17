# gym-truck-backerupper

Gymnasium simulator for a tractor–trailer (2D kinematics).

## Install

```bash
cd gym-truck-backerupper
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate           # Linux / macOS
pip install -r requirements.txt
pip install -e .
```

Python **3.9 or 3.10**.

## Quick check

```python
import gymnasium as gym
import gym_truck_backerupper

env = gym.make("TruckBackerUpper-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Manual driving

```bash
python manual_control_pygame.py
```

`1` / `2` / `3` = courses · `A`/`D` or arrows = steer · `R` = reset · `F` = forward/reverse · `Q`/Esc = quit

## Env API

| | |
|--|--|
| ID | `TruckBackerUpper-v0` |
| Obs | tractor heading error, trailer heading error, trailer lateral error |
| Action | steering `[-45°, +45°]` |
| Helpers | `manual_course`, `manual_velocity`, `manual_params` on `env.unwrapped` |

## Where to code (exemple)

Do **not** put course work inside `gym_truck_backerupper/` (simulator only).

| Folder   | When              | What                                                              |
| -------- | ----------------- | ----------------------------------------------------------------- |
| `mdp/`   | **first**         | Our MDP (state, action, reward, termination) + wrapper on the sim |
| `train/` | **after** the MDP | Training / eval scripts (PPO, etc.)                               |

Target tree:

```
gym-truck-backerupper/
├── gym_truck_backerupper/     # simulator (leave as-is)
│   └── envs/
│       ├── truck_backerupper_env.py
│       ├── PathPlanner.py
│       └── dubins_pure.py
├── mdp/                       # us — MDP first
│   ├── truck_mdp.py
│   └── wrap_env.py
├── train/                     # us — AI later
│   ├── train_ppo.py
│   └── eval.py
├── manual_control_pygame.py
├── requirements.txt
├── setup.py
└── LICENSE.md
```

## Files (simulator)

| Path | Role |
|------|------|
| `gym_truck_backerupper/__init__.py` | Registers the env |
| `gym_truck_backerupper/envs/truck_backerupper_env.py` | Physics, `reset` / `step` |
| `gym_truck_backerupper/envs/PathPlanner.py` | Episode paths (Dubins) |
| `gym_truck_backerupper/envs/dubins_pure.py` | Dubins fallback (no native build) |
| `manual_control_pygame.py` | Keyboard drive |
| `requirements.txt` / `setup.py` | Dependencies + install |
| `LICENSE.md` | License |
