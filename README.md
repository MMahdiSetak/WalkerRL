# Reinforcement Learning Algorithms for Walker2d

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MMahdiSetak/WalkerRL/blob/main/LICENSE)

This repository contains implementations of three reinforcement learning algorithms: Double Deep Q-Network (DDQN), Twin
Delayed DDPG (TD3), and Soft Actor-Critic (SAC). These algorithms are applied to the `Walker2d-v2` environment from
OpenAI Gym, demonstrating various strategies to address a complex control task with a bipedal robot.

## Installation

Follow these steps to set up the project environment and run the code:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/MMahdiSetak/WalkerRL.git
   ```

2. Navigate to the project directory and install the required libraries using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the following command to train the agents with the default settings:

```bash
python main.py
```

Set common hyperparameters in the `BaseAgent` class and algorithm-specific hyperparameters in their respective agent
files.

## Algorithms Implemented

- **DDQN (Double Deep Q-Network):** This algorithm mitigates overestimation by decomposing the max operation in the
  target into action selection and action evaluation. Though DDQN is better suited for discrete action spaces, our
  experiments focus on the following two algorithms.

- **TD3 (Twin Delayed DDPG):** TD3 rectifies function approximation errors in actor-critic methods by introducing twin
  Q-networks and delayed policy updates.

- **SAC (Soft Actor-Critic):** SAC is an off-policy algorithm that optimizes a stochastic policy within an
  entropy-regularized reinforcement learning framework. This optimization encourages a balance between exploration and
  exploitation.

## Environment

The `Walker2d-v2` environment presents a challenge for agents to learn to walk forward without falling. It features a
continuous state and action space, requiring nuanced exploration strategies.

## Logging and Visualization

The logging system tracks various training metrics, including rewards, episode lengths, and neural network losses. These
metrics can be visualized using TensorBoard:

```bash
tensorboard --logdir=walker2d_tensorboard
```

## Results

This section showcases the performance of the reinforcement learning agent in the `Walker2d-v2` environment at various
stages of training, from 200,000 to 1,000,000 iterations. The video below demonstrates the incremental learning process
and the agent's improved ability to balance and walk as training progresses.

https://github.com/MMahdiSetak/WalkerRL/assets/21222218/e6868652-78b4-4dd9-9022-6c373d34ee54

## Acknowledgements

Parts of this codebase draw inspiration and adaptation from
the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) repository. I am thankful for their open-source
contributions, which were instrumental in the development of this project.

## License

This project is open source and available under the [MIT License](LICENSE).
