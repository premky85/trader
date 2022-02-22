# Deep cryptocurrency trading agent

## About the project

Since computers can outperform humans by far in terms of how much data they can process, cryptocurrency trading agents are becoming more and more popular by the day. Most trading bots use human generated strategies and apply them at volume. This project will cover trading using deep learning methods. With usage of neural networks, new trading strategies, previously unknown or too abstract can be found, hoping to enhance automated trading.

## Algorithms

### DDPG

Deep Deterministic Policy Gradient is actor-critic agent tightly connected do Q-learning since it uses model model-free approach. In contrast to model-based reinforcement learning, it tries to maximize reward instead of minimizing error. It's main advantage is, that it can be used in continuous action spaces, which is a big plus in trading, since it can not only select appropriate action (buy, sell, hold), but also the magnitude of that action. 

## Examples

You can find examples of training and usage in `examples/`.
