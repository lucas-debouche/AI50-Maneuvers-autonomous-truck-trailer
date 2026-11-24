"""DDPG agent: Actor mu(s), Critic Q(s,a), target networks, soft update."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, Critic


# Constructor arguments for DDPGAgent:
# obs_dim : int, dimension of the state space (15 for TruckMDPEnv)
# act_dim : int, dimension of the action space (1 for TruckMDPEnv)
# gamma : float, discount factor
# tau : float, soft update rate for target networks
# actor_lr : float, learning rate for the actor network
# critic_lr : float, learning rate for the critic network
# hidden : tuple, sizes of hidden layers for both actor and critic networks
# max_grad_norm : float, maximum gradient norm for clipping
# device : str or torch.device, device to run the networks on (default: "cuda" if available, else "cpu")
class DDPGAgent:
    def __init__(self, obs_dim, act_dim, gamma=0.99, tau=5e-3,
                 actor_lr=1e-4, critic_lr=1e-3, hidden=(256, 256),
                 max_grad_norm=1.0, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma, self.tau, self.max_grad_norm = gamma, tau, max_grad_norm
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        # Target networks = slow copies, they keep the Bellman target almost fixed.
        self.actor_t = copy.deepcopy(self.actor)
        self.critic_t = copy.deepcopy(self.critic)

        # Disable gradient computation for target networks to save memory and computation, since they are updated differently (soft update).
        for p in list(self.actor_t.parameters()) + list(self.critic_t.parameters()):
            p.requires_grad_(False) # don't compute gradients for target networks (they are updated differently), update à la main

        # Adams optimizers for actor and critic networks (standard choice for DDPG)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # Choose an action
    @torch.no_grad() # we only need the forward pass, no gradient computation needed
    def act(self, obs, noise_sigma=0.0):
        # Convert the observation to a PyTorch tensor, add a batch dimension, and move it to the appropriate device (CPU or GPU).
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Pass the state through the actor network to get the action, convert it to a NumPy array, and remove the batch dimension.
        a = self.actor(s).cpu().numpy()[0]
        # adding Gaussian noise for exploration during training, but not during evaluation. The noise is clipped to ensure the action remains within valid bounds.
        if noise_sigma > 0:  # exploration: Gaussian noise on the deterministic action
            a = a + np.random.normal(0.0, noise_sigma, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32) # Guarantee the action is within the valid range [-1, 1].

    #
    def update(self, batch):
        s, a, r, s2, d = (torch.as_tensor(x, device=self.device) for x in batch)

        # 1. Critic: regress Q(s,a) toward y = r + gamma * (1 - done) * Q'(s', mu'(s'))
        with torch.no_grad():
            y = r + self.gamma * (1.0 - d) * self.critic_t(s2, self.actor_t(s2)) 

        # Predicted Q(s,a) vs target y
        critic_loss = F.mse_loss(self.critic(s, a), y) 
        # Erase previous gradients, compute new gradients, clip them to avoid exploding gradients, and update the critic network's parameters using the optimizer.
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        # 2. Actor: maximize Q(s, mu(s))  <=>  minimize -Q
        # We want to have the score as high as possible, so we take the negative of the critic's output to turn it into a minimization problem.
        actor_loss = -self.critic(s, self.actor(s)).mean()
        # Critic does not get updated here, only the actor. The critic is used to evaluate how good the actor's actions are, but we don't want to change the critic's parameters in this step.
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # 3. Soft update: theta' <- tau * theta + (1 - tau) * theta'
        with torch.no_grad():
            # We apply the soft update in each weight (actor and critic) of the target networks. The target networks slowly track the learned networks, which helps stabilize training.
            for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1.0 - self.tau).add_(self.tau * p)

        return float(critic_loss.item()), float(actor_loss.item()) # return the losses as floats

    # Load either a path file or a state_dict directly.
    def load_actor(self, path_or_state_dict):
        """Initialise the actor from a Behavioral-Cloning network (same architecture)."""
        sd = torch.load(path_or_state_dict, map_location=self.device) \
            if isinstance(path_or_state_dict, str) else path_or_state_dict
        self.actor.load_state_dict(sd)
        self.actor_t.load_state_dict(sd)

    # save weight of actor and critic in a single file (.pt). Target networks are not saved.
    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    # load weights and copy them to the target networks. However, the target networks are not saved, so they will be re-initialized to the same weights as the main networks when loading.
    # We are not sure if this is a good idea, but it is convenient to be able to load a single file for both actor and critic.
    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.actor_t.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"]); 
        self.critic_t.load_state_dict(ck["critic"])
