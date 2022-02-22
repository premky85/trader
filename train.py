from model_loader import DDPG

# DDPG model with 64 previous minutes as input
model = DDPG(sample_size=64)
model.train()
