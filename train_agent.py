from model_loader import LSTMNet, TraderEnv, AgentNetActor, AgentNetCritic
from rl.agents import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

e = TraderEnv()
print(e.state)


actions = e.action_space.shape[0]
observations = e.observation_space.shape
action_input = Input(shape=(actions,), name='action_input')

actor = AgentNetActor(input_shape=(1,) + observations).get_model()
critic = AgentNetCritic(action_input, input_shape=observations).get_model()

# for i in range(100):
#     action = e.action_space.sample()
#     step = e.step(action)


def build_agent(actor, critic, actions, action_input):
    memory = SequentialMemory(limit=1000,
     window_length=1)
    random_process = None #OrnsteinUhlenbeckProcess(theta=.001, mu=0., sigma=.005)
    dqn = DDPGAgent(nb_actions=actions, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, random_process=random_process,
                  nb_steps_warmup_actor=100, nb_steps_warmup_critic=100, target_model_update=1e-3)
    return dqn

dqn = build_agent(actor, critic, actions, action_input)
dqn.compile([Adam(lr=1e-3), Adam(lr=1e-3)])
dqn.fit(e, nb_steps=50000, visualize=False, verbose=1, log_interval=1000)