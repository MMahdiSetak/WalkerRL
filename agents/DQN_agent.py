import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

from torch import optim

from environments.discretize_action_wrapper import DiscretizeActionWrapper


# Define a simple feed forward neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 44),
            nn.ReLU(),
            nn.Linear(44, 111),
            nn.ReLU(),
            nn.Linear(111, 285),
            nn.ReLU(),
            nn.Linear(285, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(self, env, ddqn=False, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.01,
                 batch_size=6000, memory_size=10000, target_update_frequency=10):
        self.env = DiscretizeActionWrapper(env, n_discrete_actions=3)
        self.ddqn = ddqn
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_frequency = target_update_frequency
        self.hparams = {
            'learning_rate': learning_rate,
            'memory_size': memory_size,
            'batch_size': batch_size,
            'gamma': gamma,
            'target_update_frequency': target_update_frequency,
            'epsilon_min': epsilon_min,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'ddqn': ddqn,
        }
        self.memory = deque(maxlen=memory_size)

        # Initialize model
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        self.model = DQN(state_size, action_size).cuda()
        # model.load_state_dict(torch.load("DDQN.pth"))
        self.target_model = DQN(state_size, action_size).cuda()
        self.target_model.load_state_dict(self.model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    # Replay and train
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state = torch.FloatTensor(np.array([x[0] for x in batch])).squeeze(1).cuda()
        action = torch.LongTensor([x[1] for x in batch]).view(-1, 1).cuda()
        reward = torch.FloatTensor([x[2] for x in batch]).view(-1, 1).cuda()
        next_state = torch.FloatTensor(np.array([x[3] for x in batch])).squeeze(1).cuda()
        done = torch.FloatTensor([x[4] for x in batch]).view(-1, 1).cuda()

        current_Q = self.model(state).gather(1, action)
        if self.ddqn:
            next_Q = self.target_model(next_state).gather(1, self.model(next_state).max(1)[1].unsqueeze(1))  # DDQN part
        else:
            next_Q = self.model(next_state).detach().max(1)[1].unsqueeze(1)  # Normal DQN
        expected_Q = reward + (1 - done) * self.gamma * next_Q

        loss = self.criterion(expected_Q, current_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes, writer):
        final_result = 0
        # Main loop
        for episode in range(num_episodes):
            state = self.env.reset()
            state = state[0]
            total_reward = 0
            while True:
                # env.render()
                state = torch.FloatTensor(state).cuda()
                action = self.env.action_space.sample() if np.random.rand() <= self.epsilon else np.argmax(
                    self.model(state).cpu().data.numpy())
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                exp = [state.cpu().numpy(), action, reward, next_state, done]
                self.memory.append(exp)
                state = next_state
                total_reward += reward
                if done:
                    # print("Episode: {}/{}, Score: {}".format(episode + 1, num_episodes, t))
                    break
            writer.add_scalar("reward", total_reward, episode)
            final_result += total_reward / num_episodes
            self.replay()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            writer.add_scalar("epsilon", self.epsilon, episode)

            if episode % self.target_update_frequency == 0 and self.ddqn:
                self.target_model.load_state_dict(self.model.state_dict())

        torch.save(self.model.state_dict(), "DDQN.pth" if self.ddqn else "DQN.pth")
        writer.add_hparams(self.hparams, {'final_result': final_result})
