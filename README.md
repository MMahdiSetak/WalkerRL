# WalkerRL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MMahdiSetak/WalkerRL/blob/main/LICENSE)

This repository contains implementations of three reinforcement learning algorithms: Double Deep Q-Network (DDQN), Twin
Delayed DDPG (TD3), and Soft Actor-Critic (SAC). These algorithms have been applied to the `Walker2d-v2` environment
from OpenAI Gym, showcasing different strategies to solve a complex control task with a bipedal robot.

## Installation

To set up the project environment to run the code, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rl-walker2d.git
   cd rl-walker2d
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the agents with the default settings, run the following command:

```bash
python main.py
```

Common hyperparameters can be set in BaseAgent class and algorithm specific hyperparameters can be set in their own
agent file.

## Algorithms Implemented

- **DDQN (Double Deep Q-Network)**: An extension of DQN that reduces overestimation by decomposing the max operation in
  the target into action selection and action evaluation. Since this algorithm is well-suited for discrete action spaces
  the experiments used the next two algorithms.

- **TD3 (Twin Delayed DDPG)**: An algorithm that addresses the function approximation errors in actor-critic methods by
  introducing twin Q-networks and delayed policy updates.

- **SAC (Soft Actor-Critic)**: An off-policy algorithm that optimizes a stochastic policy in an entropy-regularized
  reinforcement learning framework, leading to a balance between exploration and exploitation.

## Environment

The `Walker2d-v2` environment challenges agents to learn to walk forward without falling over, which involves a
continuous state and action space.

## Logging and Visualization

Logging is set up to track various metrics during training, including rewards, episode length and neural networks
losses, which can be visualized using TensorBoard:

```bash
tensorboard --logdir=walker2d_tensorboard
```

## Acknowledgements

Parts of this codebase are inspired by and adapted from
the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) repository. I am grateful for their open-source
contributions which were instrumental in the development of this project.

## License

This project is open source and available under the [MIT License](LICENSE).