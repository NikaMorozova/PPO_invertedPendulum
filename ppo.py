import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Actor(nn.Module):
    """
    Actor network, input_dim is a dimension of state space (also observation),
                   output_dim is a action space (in our env change in force for pendulum cart).
    Methods in my imolementation of this class:
                   forward(state) for passing trough netwrok
                                  and get mean and standard deviation of the action distribution
                   get_actions(state, actions: Optional) for sampling action from distribution
                                                         and it's log probability
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean3 = nn.Linear(hidden_dim, output_dim)

        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        mu = self.mean3(x)
        sigma = torch.exp(self.log_std)
        return mu, sigma

    def get_actions(self, state, actions = None):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        mus, sigmas = self.forward(state)
        dist = torch.distributions.MultivariateNormal(mus, torch.diag_embed(sigmas, 0))
        if actions == None:
            actions = dist.sample()
        if self.output_dim == 1 and actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        log_probs = dist.log_prob(actions)
        return actions, log_probs


class Critic(nn.Module):
    """
    Critic network, input_dim is a dimension of state space (also observation, as above for Actor)
    Method in my imolementation in this class:
                   forward(state) for passing trough netwrok
                                  and get state value
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        return self.dense3(x).squeeze()

class PPO:
    """
    PPO algorithm class

    Defiened Computation of advantages and returns trough GAE,
    then using it for computation Actor and Critic loss and update PPO agent for defined num of steps (num_eposhs)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        num_epochs: int = 16,
        mini_batch_size: int = 64,
        cliprange=0.2,
        critic_loss_coef=0.5
    ):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.actor = Actor(input_dim, output_dim, hidden_dim)
        self.critic = Critic(input_dim, hidden_dim)
        self.optimizer = torch.optim.Adam([
            {"params": self.actor.parameters(), "lr": 3e-4},
            {"params": self.critic.parameters(), "lr": 3e-4},
        ])
        self.cliprange = cliprange
        self.critic_loss_coef = critic_loss_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def gae_compute(self, rewards, values, resets):
        masks = 1 - resets
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0
                
            else:
                next_value = values[step + 1]
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.lambda_* masks[step] * gae

            advantages[step] = gae
            returns[step]= gae + values[step]

        return advantages, returns

    def actor_loss(self, states, actions, old_log_probs, advantages):
        _, log_probs_all = self.actor.get_actions(states, actions)
        log_old_probs = torch.tensor(old_log_probs)
        ratio = (log_probs_all - log_old_probs).exp()
        surrogate = ratio * advantages
        surrogate_clipped = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange) * advantages
        return -torch.mean(torch.min(surrogate, surrogate_clipped))

    def critic_loss(self, states, returns):
        value_predict = self.critic(states)
        critic_losses = F.mse_loss(value_predict, returns)
        return critic_losses

    def step(self, trajectory):
        states = torch.as_tensor(trajectory['observations'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(trajectory['actions'], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(trajectory['rewards'], dtype=torch.float32, device=self.device)
        resets = torch.as_tensor(trajectory['resets'], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(trajectory['log_probs'], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = self.gae_compute(rewards, values, resets)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

        actor_loss_epoch = 0
        critic_loss_epoch = 0
        for _ in range(self.num_epochs):
            for (states, actions, old_log_probs, advantages, returns) in dataloader:
                # Total loss
                actor_loss = self.actor_loss(states, actions, old_log_probs, advantages)
                critic_loss = self.critic_loss(states, returns)
                loss = actor_loss + self.critic_loss_coef * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
        return {
            "actor_loss": actor_loss_epoch / (len(dataloader)),
            "critic_loss": critic_loss_epoch / (len(dataloader)),
        }