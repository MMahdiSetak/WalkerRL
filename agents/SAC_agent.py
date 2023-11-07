import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.distributions import Normal

from agents.base_agent import BaseAgent, Algorithm


class ActorSAC(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(ActorSAC, self).__init__()

        self.latent_pi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        x = self.latent_pi(x)
        mean_actions = self.mu(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return mean_actions, log_std

    def action_log_prob(self, state):
        action_mu, log_std = self.forward(state)
        action_std = torch.ones_like(action_mu) * log_std.exp()
        distribution = Normal(action_mu, action_std)
        gaussian_actions = distribution.rsample()
        action = torch.tanh(gaussian_actions)

        log_prob = distribution.log_prob(gaussian_actions)
        log_prob = torch.sum(log_prob, dim=1)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - action ** 2 + 1e-6), dim=1)
        return action, log_prob


def sample_action(action_mu, log_std):
    action_std = torch.ones_like(action_mu) * log_std.exp()
    distribution = Normal(action_mu, action_std)
    gaussian_actions = distribution.rsample()
    action = torch.tanh(gaussian_actions)
    return action


class CriticSAC(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims=256):
        super(CriticSAC, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, state, action):
        q_input = torch.cat([state, action], dim=1)
        q1 = self.q1(q_input)
        q2 = self.q2(q_input)
        return q1, q2


class SACAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env=env, alg=Algorithm.SAC)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = ActorSAC(state_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = CriticSAC(state_dim, action_dim).to(self.device)
        self.critic_target = CriticSAC(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.log_ent_coef = torch.log(torch.ones(1, device=self.device)).requires_grad_(True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action_mu, log_std = self.actor(state)
        self.actor.train()
        action = sample_action(action_mu, log_std)
        return action.cpu().numpy()

    def train(self):
        # Sample transitions from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, device=self.device)

        actions_pi, log_prob = self.actor.action_log_prob(state)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (log_prob - 6).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        # Compute the target Q value
        with torch.no_grad():
            next_action, next_log_prob = self.actor.action_log_prob(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
            target_q = reward + (1 - done) * self.discount * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        # Compute critic loss
        critic_loss = (mse_loss(current_q1, target_q) + mse_loss(current_q2, target_q)) / 2

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q1_value, q2_value = self.critic(state, actions_pi)
        min_qf_pi = torch.min(q1_value, q2_value)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.mul_(1 - self.tau)
            torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

        self.log_record(actor_losses=actor_loss.item(), critic_losses=critic_loss.item(), ent_coefs=ent_coef.item(),
                        ent_coef_losses=ent_coef_loss.item())
