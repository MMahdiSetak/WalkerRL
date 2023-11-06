import numpy as np
import torch
from torch import nn

from agents.base_agent import BaseAgent, Algorithm


class ActorTD3(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(400, 300)):
        super(ActorTD3, self).__init__()
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
            nn.Tanh()
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class CriticTD3(nn.Module):
    def forward(self, state, action):
        q_input = torch.cat([state, action], dim=1)
        q1 = self.q1(q_input)
        q2 = self.q2(q_input)
        return q1, q2

    def __init__(self, input_dim, action_dim, hidden_dims=(400, 300)):
        super(CriticTD3, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )


class TD3Agent(BaseAgent):
    def __init__(self, env, policy_noise=0.2, noise_clip=0.5):
        super().__init__(env=env, alg=Algorithm.TD3)
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = ActorTD3(state_dim, action_dim).to(self.device)
        self.actor_target = ActorTD3(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = CriticTD3(state_dim, action_dim).to(self.device)
        self.critic_target = CriticTD3(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.hparams['policy_noise'] = policy_noise
        self.hparams['noise_clip'] = noise_clip

    # Selects an action with exploration noise
    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if noise:
            action = (action + np.random.normal(0, self.policy_noise, size=action.shape)).clip(-1, 1)
        return action

    def train(self):
        # Sample transitions from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Compute the target Q value
        with torch.no_grad():
            noise = action.data.normal_(0, self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.discount * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + torch.nn.functional.mse_loss(current_q2,
                                                                                                        target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Policy updates
        actor_loss = -self.critic(state, self.actor(state))[0].mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.mul_(1 - self.tau)
            torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.mul_(1 - self.tau)
            torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
