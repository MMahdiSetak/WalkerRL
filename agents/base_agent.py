import random
import time
from pathlib import Path

import numpy as np
import torch
from collections import namedtuple, deque
from enum import Enum

from utils.logging import get_writer, get_run_directory

# Define a transition structure for the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.FloatTensor(np.array(batch.state)).to(device)
        action = torch.FloatTensor(np.array(batch.action)).to(device)
        reward = torch.FloatTensor(batch.reward).to(device).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done = torch.FloatTensor(batch.done).to(device).unsqueeze(1)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Algorithm(Enum):
    DQN = 1
    TD3 = 2
    SAC = 3


class BaseAgent:
    def __init__(self, alg, env, device="cuda", buffer_size=1_000_000, lr=3e-4, batch_size=256, discount=0.999,
                 tau=0.005, save_interval=2e5):
        self.alg = alg
        self.writer = None
        self.env = env
        self.device = device
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.learning_rate = lr
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.save_interval = save_interval
        self.num_steps = self.last_save = self.episode_length = self.episode_reward = 0
        self.save_path = ""
        self.actor = self.actor_optimizer = self.critic = self.critic_optimizer = self.critic_target = None
        self.last_obs = None

        self.hparams = {
            'algorithm': self.alg.value,
            'learning_rate': lr,
            'memory_size': buffer_size,
            'batch_size': batch_size,
            'discount': discount,
            'tau': tau
        }

    def select_action(self, state):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_models(self):
        if self.num_steps >= self.last_save + self.save_interval:
            path = self.save_path + f"/{self.num_steps}/"
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(self.actor.state_dict(), path + "actor.pth")
            torch.save(self.actor_optimizer.state_dict(), path + "actor_optimizer")
            torch.save(self.critic.state_dict(), path + "critic.pth")
            torch.save(self.critic_optimizer.state_dict(), path + "critic_optimizer")
            self.last_save += self.save_interval

    def load_model(self, path=None):
        self.actor.load_state_dict(torch.load(path + "actor.pth"))
        self.actor_optimizer.load_state_dict(torch.load(path + "actor_optimizer"))
        self.critic.load_state_dict(torch.load(path + "critic.pth"))
        self.critic_optimizer.load_state_dict(torch.load(path + "critic_optimizer"))
        self.critic_target.load_state_dict(torch.load(path + "critic.pth"))

    def populate_initial_buffer(self):
        """Populate the replay buffer with initial experiences"""
        state = self.env.reset()[0]
        episode_len = 0
        while True:
            episode_len += 1
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            if terminated or truncated:
                break
        self.num_steps += episode_len
        return episode_len

    def interact_with_environment(self, steps=-1):
        """Interact with the environment and train the agent"""
        if self.last_obs is None:
            self.last_obs = self.env.reset()[0]
        iteration = 0
        while steps == -1 or iteration < steps:
            iteration += 1
            self.episode_length += 1
            action = self.select_action(self.last_obs)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.num_steps += 1
            done = terminated or truncated
            self.episode_reward += reward
            self.replay_buffer.push(self.last_obs, action, reward, next_state, done)
            self.last_obs = next_state
            if done:
                self.last_obs = self.env.reset()[0]
                self.writer.add_scalar("episode/reward", self.episode_reward, self.num_steps)
                self.writer.add_scalar("episode/length", self.episode_length, self.num_steps)
                self.episode_length = 0
                self.episode_reward = 0
                break

    def learn(self, total_timesteps):
        self.save_path = get_run_directory(f"models/{self.alg.name}")
        self.writer = get_writer(self.alg.name)
        start = time.time()
        while self.num_steps < self.batch_size:
            self.populate_initial_buffer()
        while self.num_steps < total_timesteps:
            self.interact_with_environment(steps=1)
            self.train()
            self.save_models()
        end = time.time()
        elapsed_time = end - start
        elapsed_time /= 60
        final_result = self.evaluate_agent(100) / elapsed_time
        self.writer.add_hparams(self.hparams, {'final_result': final_result})

    def evaluate_agent(self, episodes):
        total_reward = 0
        for _ in range(episodes):
            state = self.env.reset()[0]
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                if terminated or truncated:
                    break
        return total_reward / episodes

    def log_record(self, actor_losses, critic_losses, ent_coefs=None, ent_coef_losses=None):
        self.writer.add_scalar("train/ent_coef", ent_coefs, self.num_steps)
        self.writer.add_scalar("train/actor_loss", actor_losses, self.num_steps)
        if self.alg == Algorithm.SAC and ent_coefs is not None and ent_coef_losses is not None:
            self.writer.add_scalar("train/critic_loss", critic_losses, self.num_steps)
            self.writer.add_scalar("train/ent_coef_loss", ent_coef_losses, self.num_steps)
