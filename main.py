import gymnasium as gym
from agents.SAC_agent import SACAgent

# Create the environment
env = gym.make('Walker2d-v4')
num_episodes = 1_000_000

# dqn_agent = DQNAgent(env)
# dqn_agent.train(num_episodes, writer)

agent = SACAgent(env)
agent.learn(num_episodes)

# Test the trained agent
# env = gym.make('Walker2d-v4', render_mode='human')
# obs = env.reset()
# while True:
#     state = torch.FloatTensor(obs[0]).cuda()
#     action = np.argmax(dqn_agent.model(state).cpu().data.numpy())
#     obs, rewards, terminated, truncated, info = env.step(action)
#     if terminated:
#         obs = env.reset()
#     env.render()
