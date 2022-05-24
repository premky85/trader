from model_loader import DDPG, DQN


# DDPG model with 64 previous minutes as input
# model = DDPG(sample_size=128, batch_size=128, buffer_size=100000)
# model.train()


agent = DDPG(buffer_size=5000, batch_size=128, sample_size=256)
agent.train(episodes=100000, validation_freq=100, validation_runs=1)
