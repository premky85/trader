import tensorflow as tf
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.keras_layers import inner_reshape
from tf_agents import networks
from keras import backend as K
from agents.models import ActorNet, CriticNet, RefineNet
from agents import agent


class DDPG(agent.Agent):
    def __init__(self, sample_size=128, batch_size=64, buffer_size=50000) -> None:
        super(DDPG, self).__init__(
            sample_size=sample_size, batch_size=batch_size, discrete=False
        )
        self.name = "DDPG"

        self.n_actions = 1
        self.qnet_actions = 3

        # Learning rates for actor-critic models
        self.learning_rates = [
            (2e-3, 5e-4),
            (6e-4, 2e-4),
            (1e-4, 5e-5),
            # (5e-5, 1e-4),
            # (1e-5, 5e-5),
            # (1e-6, 5e-6),
            # (5e-7, 5e-7),
            # (1e-7, 1e-7),
        ]

        self.actor_optimizer = tf.keras.optimizers.Adam(
            self.learning_rates[self.learning_rate_i][0], clipnorm=0.01
        )

        self.critic_optimizer = tf.keras.optimizers.Adam(
            self.learning_rates[self.learning_rate_i][1], clipnorm=0.0001
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
        self.lr_counter_threshold = 20
        self.reset_lr_threshold = True

    def _create_agent(self):
        q_net = ActorNet(
            num_states=self.n_states,
            num_actions=self.qnet_actions,
            states_spec=self.sample_sizes,
        ).get_model(interposed=True)
        refine_net = RefineNet(
            num_states=128,
        ).get_model()
        actor_net = networks.Sequential([q_net] + [refine_net])
        critic_net = CriticNet(
            num_states=self.n_states,
            num_actions=self.n_actions,
            states_spec=self.sample_sizes,
        ).get_model()
        critic_net = networks.Sequential(
            [critic_net, inner_reshape.InnerReshape([1], [])]
        )
        self.agent = ddpg_agent.DdpgAgent(
            time_step_spec=self.env.time_step_spec(),
            action_spec=self.env.action_spec(),
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            ou_stddev=0.4,
            gamma=0.8,
            # gradient_clipping=0.,
            train_step_counter=self.global_step,
            target_update_period=1000,
        )
        self.agent.initialize()

    def _update_lr(self):
        K.set_value(
            self.agent._actor_optimizer.learning_rate,
            self.learning_rates[self.learning_rate_i][0],
        )
        K.set_value(
            self.agent._critic_optimizer.learning_rate,
            self.learning_rates[self.learning_rate_i][1],
        )
