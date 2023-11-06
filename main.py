import gymnasium as gym
from stable_baselines3 import SAC
import torch
import numpy as np

from agents.DQN_agent import DQNAgent
from agents.SAC_agent import SACAgent
from agents.TD3_agent import TD3Agent
from utils.logging import get_writer
from utils.visual import human_render

# Create the environment
env = gym.make('Walker2d-v4')
num_episodes = 10_000_000
model = SAC("MlpPolicy", env, tensorboard_log="./baseline3_tensorboard/", device="cuda")
model.learn(total_timesteps=num_episodes)
model.save("models/baseline3/SAC")
# exit()

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render()

#
# dqn_agent = DQNAgent(env)
# dqn_agent.train(num_episodes, writer)
# env.close()

agent = SACAgent(env)
agent.load_model("models/SAC_7/10000348/")
# agent = TD3Agent(env)
agent.learn(num_episodes)


human_render()
exit()

# Test the trained agent
env = gym.make('Walker2d-v4', render_mode='human')
obs = env.reset()
while True:
    state = torch.FloatTensor(obs[0]).cuda()
    action = np.argmax(dqn_agent.model(state).cpu().data.numpy())
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated:
        obs = env.reset()
    env.render()
