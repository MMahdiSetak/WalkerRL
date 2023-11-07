import os

import gymnasium as gym

from agents.SAC_agent import SACAgent


def human_render():
    env = gym.make('Walker2d-v4', render_mode='human')
    agent = SACAgent(env)
    agent.load_model("models/SAC_7/10000348/")
    # agent.load_model("models/SAC_6/10000381/")
    state = env.reset()[0]
    while True:
        action = agent.select_action(state)
        state, rewards, terminated, truncated, info = env.step(action)
        if terminated:
            state = env.reset()[0]
        env.render()


def training_visualization():
    env = gym.make('Walker2d-v4', render_mode='human')
    agent = SACAgent(env)
    path = "models/SAC_9/"
    dirs = os.listdir(path)
    dirs = list(map(int, dirs))
    dirs.sort()
    for d in dirs:
        agent.load_model(path + str(d) + "/")
        state = env.reset()[0]
        print(d)
        while True:
            action = agent.select_action(state)
            state, rewards, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            env.render()
