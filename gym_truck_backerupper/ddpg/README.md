# ddpg/ — DDPG for the autonomous truck-trailer

This folder contains the **DDPG** (Deep Deterministic Policy Gradient) implementation used to train the truck-trailer agent on top of the MDP layer (`gym_truck_backerupper/mdp/`). It also provides the hooks needed to plug in **Imitation Learning** (Behavioral Cloning + human demonstrations).

## 1. Layout

```
ddpg/
├── __init__.py
├── networks.py        # Actor and Critic (PyTorch MLPs)
├── replay_buffer.py   # Replay buffer + add_demonstrations()
├── ddpg.py            # DDPGAgent: act(), update(), save/load, load_actor()
└── train_ddpg.py      # Training loop, evaluation, CSV metrics
```

## 2. Quick start

```bash
# Reverse straight line (default), 500 episodes
python -m train.train_ddpg --course straight --velocity -2.012 --episodes 500

# Easier first validation: forward straight line
python -m train.train_ddpg --course straight --velocity 2.012 --run-name ddpg_straight_forward
```

| Argument                        | Default                 | Meaning                                                                      |
| ------------------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| `--course`                      | `straight`              | `straight`, `turn`, `scurve` (fixed courses) or `random` (random start/goal) |
| `--velocity`                    | `-2.012`                | Constant speed in m/s. Negative = reverse, positive = forward                |
| `--episodes`                    | `500`                   | Number of training episodes                                                  |
| `--warmup`                      | `5000`                  | Initial steps with random actions to fill the buffer                         |
| `--batch-size`                  | `256`                   | Mini-batch size for each update                                              |
| `--sigma-start` / `--sigma-end` | `0.3` / `0.05`          | Exploration noise (std), decayed linearly                                    |
| `--sigma-decay-steps`           | `100000`                | Steps over which the noise decays                                            |
| `--eval-every`                  | `25`                    | Run 10 deterministic evaluation episodes every N episodes                    |
| `--seed`                        | `0`                     | Random seed (use several seeds for comparisons)                              |
| `--run-name`                    | `ddpg_straight_reverse` | Name of the metrics CSV and of the checkpoint folder                         |
| `--init-actor`                  | —                       | Path to Behavioral Cloning actor weights (`.pt`)                             |
| `--demos`                       | —                       | `.npz` file with human demonstrations (see §6)                               |

**Outputs**

- `training_metrics/<run-name>.csv` — one line per episode, via `TrainingMetricsDirectory` (`;`-separated).
- `runs/<run-name>/last.pt` and `best.pt` — checkpoints (actor + critic), saved at each evaluation.

## 3. How DDPG works here

DDPG is an **actor-critic** algorithm for continuous actions. Here the action is a single normalized steering command in `[-1, 1]` (mapped to ±45° by the MDP).

| Component                  | Role                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Actor** μ(s)             | Takes the 15-dim state, outputs the steering. MLP 15→256→256→1 with `tanh` so the output stays in `[-1, 1]`.        |
| **Critic** Q(s, a)         | Takes state and action, estimates the expected discounted return. MLP 16→256→256→1.                                 |
| **Target networks** μ′, Q′ | Slow copies of the actor and the critic. They keep the Bellman target almost fixed, which stabilizes learning.      |
| **Replay buffer**          | Stores past transitions `(s, a, r, s′, done)`; random mini-batches break the correlation between consecutive steps. |
| **Exploration noise**      | Gaussian noise added to the actor output during training (the actor itself is deterministic).                       |

One update step (`DDPGAgent.update`):

1. **Critic**: target `y = r + γ·(1 − done)·Q′(s′, μ′(s′))`, then minimize `MSE(Q(s, a), y)`.
2. **Actor**: maximize `Q(s, μ(s))`, i.e. minimize `−Q(s, μ(s))`.
3. **Soft update**: `θ′ ← τ·θ + (1 − τ)·θ′` for both target networks.

**Default hyperparameters**

| Parameter                    | Value             |
| ---------------------------- | ----------------- |
| γ (discount)                 | 0.99              |
| τ (soft update)              | 5e-3              |
| Actor / critic learning rate | 1e-4 / 1e-3       |
| Hidden layers                | 256 × 256         |
| Buffer capacity              | 1,000,000         |
| Gradient clipping            | 1.0 (global norm) |

**Design choices**

- `done` is **`terminated` only** (crash or success). A time-out (`truncated`) is not a real end of the MDP, so the agent keeps bootstrapping on it.
- The last layer of each network is initialized with tiny weights (±3e-3) : initial steering and Q-values are close to 0 and `tanh` does not start saturated.
- Gaussian noise is used: simpler, and generally as effective.
- Training metrics reuse the existing `metrics` module:

| CSV column  | Computed as                                                                           |
| ----------- | ------------------------------------------------------------------------------------- |
| `reward`    | Sum of rewards over the episode                                                       |
| `loss`      | Mean critic loss over the episode (`nan` during warm-up)                              |
| `success`   | `info["success"]` at the end of the episode                                           |
| `precision` | Mean absolute lateral error of the trailer, in metres                                 |
| `stability` | Mean absolute steering variation between steps (zig-zag indicator; lower is smoother) |

## 4. Integration with the MDP layer

```python
raw     = gym.make("TruckBackerUpper-v0")
backend = GymBackerUpperBackend(raw, velocity=-2.012, course=((28, 0, 180), (-28, 0, 180)))
env     = TruckMDPEnv(backend)          # 15-dim observation, action in [-1, 1]
```

Courses use the same convention as `manual_control_pygame.py`: `((x0, y0, heading_deg), (xg, yg, heading_deg))`.

## 5. Imitation Learning hooks

Two entry points are ready, the data pipeline is not (yet):

- `ReplayBuffer.add_demonstrations(states, actions, rewards, next_states, terminated)` injects human transitions into the buffer, so the critic sees good actions from the start.
- `DDPGAgent.load_actor(path)` initializes the actor from a Behavioral Cloning network (same architecture).

`--demos` expects an `.npz` file with the arrays `s`, `a`, `r`, `s2`, `done`.

**Important: the current `demonstrations.csv` cannot be used as is.** `manual_control_pygame.py` records:

| Aspect      | `manual_control_pygame.py`                            | Required by DDPG                           |
| ----------- | ----------------------------------------------------- | ------------------------------------------ |
| Observation | 3 dims (raw simulator errors)                         | 15 dims (MDP observation)                  |
| Action      | steering in radians, limited to ±25°                  | `steer / max_steer` in `[-1, 1]`           |
| Reward      | simulator reward                                      | MDP reward                                 |
| Environment | raw env with "easy" parameters (`L2 = 7`, speed 0.45) | `TruckMDPEnv` with the training parameters |

Demonstrations must therefore be re-recorded through `TruckMDPEnv`.

## 6. Known limitations and points to watch

- **Sparse success**: the simulator requires `min_d ≤ 0.15 m` and `|ψ| ≤ 0.1 rad`. Early on, the +100 bonus will rarely be reached; follow `precision` rather than the success rate at first.
- **Reward balance**: the per-step reward is mostly negative (survival +0.01 vs. penalties that can exceed 0.5). With γ = 0.99 (effective horizon ≈ 100 steps), crashing early can look attractive when errors stay large. If the agent learns to terminate quickly, revisit the reward weights.
- **Fixed course overfitting**: a fixed course trains the agent on a single trajectory. Use `--course random` to test generalization.
- **Infinite loop risk**: with a manual course, a start/goal placed too close to the ±40 m borders makes the simulator's `reset()` loop forever. Stick to the predefined courses.
- **Seed sensitivity**: DDPG is very sensitive to the seed; compare methods over at least 3 seeds.
- **Side bug**: `manual_control_pygame.py` adds the reward twice per step (`total_reward`), so the displayed reward is doubled.

## 7. Suggested roadmap

1. Validate the pipeline on the **forward straight line**.
2. Move to the **reverse straight line**, then `turn` and `scurve`.
3. Write `record_demos.py` (record via `TruckMDPEnv`, export the `.npz`).
4. Write `bc_pretrain.py` (supervised regression `state → action`, saves actor weights for `--init-actor`).
5. Compare **DDPG alone** vs. **BC + DDPG** over several seeds, using reward, success rate, precision and stability.
6. Test generalization with `--course random`.

## 8. About section

This README.md and the code was generated by AI (Claude) therefore we checked all the report and code by hand to be sure of what have been generated.
