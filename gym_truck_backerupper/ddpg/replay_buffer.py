import numpy as np

# Replay buffer for DDPG, with a hook to inject human demonstrations (Imitation Learning).
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=1_000_000):
        self.capacity = int(capacity) # set the maximum number of transitions the buffer can hold
        self.s = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        # `done` = TERMINATED only (crash / success). A time-out (truncated)
        # is NOT a real end of the MDP, so we keep bootstrapping on it.
        self.d = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0 # pointer to the next index to insert a new transition
        self.size = 0 # current number of transitions stored in the buffer
        self.n_demo = 0 # number of transitions that come from human demonstrations (for imitation learning)

    # add a single transition to the buffer
    def add(self, s, a, r, s2, terminated):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(terminated)
        # circular buffer: when the buffer is full, new transitions overwrite the oldest ones.
        self.ptr = (self.ptr + 1) % self.capacity # % capacity get back to 0 when we reach the end of the buffer
        self.size = min(self.size + 1, self.capacity) # size stop capacity

    # inject human demonstrations into the buffer
    # At the beginning of training, we can fill the buffer with human demonstrations to help the agent learn faster.
    # At the end, demonstrations are ecrased by the agent's own experience.
    def add_demonstrations(self, states, actions, rewards, next_states, terminated):
        """Insert (s, a, r, s', done) transitions recorded from a human driver.
        They must use the SAME 15-dim state and normalized action in [-1, 1]
        as the agent (see README of train/)."""
        for s, a, r, s2, d in zip(states, actions, rewards, next_states, terminated):
            self.add(s, np.atleast_1d(a), r, s2, d) # np.atleast_1d(a) ensures that the action is at least 1-dimensional. 0.3 instead of [0.3] for example.
        self.n_demo += len(states)

    # Sample a batch of transitions from the buffer 
    def sample(self, batch_size):
        # Select randomly 256 indexes between 0 and current size.
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.size
