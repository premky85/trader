import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents import networks
from keras import backend as K
from environments.environments import TFTraderEnv
from agents.models import ActorNet
from agents import agent


class DQN(agent.Agent):
    def __init__(self, sample_size=64, batch_size=64, buffer_size=50000) -> None:
        super(DQN, self).__init__(
            sample_size=sample_size, batch_size=batch_size, discrete=True
        )
        self.name = "DQN"

        self.n_actions = 3

        # Learning rates for actor-critic models
        self.learning_rates = [5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 5e-7, 1e-7]

        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rates[self.learning_rate_i], clipnorm=0.001
        )

        self._create_agent()

        buffer_spec = self.agent.collect_data_spec
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            buffer_spec, batch_size=1, max_length=buffer_size
        )

        self._get_dataset(self.buffer)

        replay_observer = [self.buffer.add_batch]
        train_metrics = [tf_metrics.EnvironmentSteps()]

        self.observers = replay_observer + train_metrics
        self.lr_counter_threshold = 15

    def _create_agent(self):
        q_net = ActorNet(
            num_states=self.n_states,
            num_actions=self.n_actions,
            states_spec=self.sample_sizes,
        ).get_model()
        q_net = networks.Sequential([q_net])
        self.agent = dqn_agent.DqnAgent(
            time_step_spec=self.env.time_step_spec(),
            action_spec=self.env.action_spec(),
            q_network=q_net,
            optimizer=self.optimizer,
            gamma=0.95,
            epsilon_greedy=0.2,
            gradient_clipping=0.1,
            target_update_period=2000,
        )
        self.agent.initialize()

    def _update_lr(self):
        K.set_value(
            self.agent._optimizer.learning_rate,
            self.learning_rates[self.learning_rate_i],
        )
