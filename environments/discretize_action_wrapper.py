import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_discrete_actions):
        super().__init__(env)
        self.n_discrete_actions = n_discrete_actions
        self.action_space = spaces.Discrete(n_discrete_actions ** env.action_space.shape[0])

    def action(self, action):
        n = self.env.action_space.shape[0]
        actions = np.zeros(n)
        for i in range(n):
            actions[i] = action % self.n_discrete_actions
            action //= self.n_discrete_actions
        actions = 2 * actions / self.n_discrete_actions - 1
        return actions
