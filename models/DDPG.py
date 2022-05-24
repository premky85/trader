from utils.helpers import OUActionNoise
from gym import spaces
from environments.environments import TraderEnv
from environments.dataloaders.binance_dataloader import BinanceDataLoader
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.Network import *
import random


class ActorModel(Network):
    def __init__(
        self,
        num_states,
        output_size=1,
        num_classes=None,
        weights=None,
        save_weights_path=None,
        dataset_name="default",
        lr=1e-3,
    ):
        self.model_name = "AgentNetActor"
        self.num_states = num_states
        self.output_size = output_size
        super(ActorModel, self).__init__(
            num_classes=num_classes,
            weights=weights,
            save_weights_path=save_weights_path,
            dataset_name=dataset_name,
            lr=lr,
        )

    def get_model(self):
        in_shape = self.num_states
        out_size = self.output_size

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = keras.Input(shape=(in_shape,), name="observation_input_0_actor")

        data_size = in_shape // 2
        inputs_0 = tf.reshape(inputs[:, :data_size], [-1, data_size, 1])
        inputs_1 = tf.reshape(inputs[:, data_size:], [-1, data_size, 1])

        x_1 = layers.Conv1D(8, 3, padding="valid")(inputs_0)
        x_1 = layers.BatchNormalization()(x_1)
        x_1 = layers.ReLU()(x_1)

        x_2 = layers.Conv1D(8, 3, padding="valid")(inputs_1)
        x_2 = layers.BatchNormalization()(x_2)
        x_2 = layers.ReLU()(x_2)

        x = layers.add([x_1, x_2])

        for filters in [16, 32, 32]:
            x_1 = layers.Conv1D(filters, 3, padding="valid")(x)
            x_1 = layers.BatchNormalization()(x_1)
            x_1 = layers.ReLU()(x_1)
            x_1 = layers.Conv1D(filters, 3, padding="valid")(x_1)
            x_1 = layers.BatchNormalization()(x_1)
            x_1 = layers.ReLU()(x_1)
            x_1 = layers.MaxPool1D()(x_1)

            x_2 = layers.Conv1D(filters, 3, padding="valid")(x)
            x_2 = layers.BatchNormalization()(x_2)
            x_2 = layers.ReLU()(x_2)
            x_2 = layers.Conv1D(filters, 3, padding="valid")(x_2)
            x_2 = layers.BatchNormalization()(x_2)
            x_2 = layers.ReLU()(x_2)
            x_2 = layers.MaxPool1D()(x_2)

            x = layers.concatenate([x_1, x_2])

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.BatchNormalization()(x)

        outputs = layers.Dense(out_size, activation="sigmoid")(x)

        model = keras.Model(inputs, outputs)
        model.summary()

        if self.weights is not None:
            model.load_weights(self.weights)

        return model


class DDPG:
    def __init__(
        self, sample_size=64, batch_size=128, buffer_size=50000, model_weights_id=None
    ) -> None:
        self.sample_size = sample_size
        action_space = spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)
        observation_space = spaces.Box(
            np.array([-1 for i in range(sample_size * 2)]),
            np.array([1 for i in range(sample_size * 2)]),
            dtype=np.float32,
        )
        self.env = TraderEnv(
            action_space=action_space,
            observation_space=observation_space,
            path="./data/Binance_BTCUSDT_minute.csv",
            sample_size=sample_size,
            train_steps=5000,
        )

        self.batch_size = batch_size
        self.n_actions = self.env.action_space.shape[0]
        self.n_states = self.env.observation_space.shape[0]
        self.min_action = self.env.action_space.low[0]
        self.max_action = self.env.action_space.high[0]

        self._get_models(model_weights_id)

        # Learning rates for actor-critic models
        self.learning_rates = [
            (1e-3, 1e-3),
            (5e-4, 1e-3),
            (1e-4, 5e-4),
            (5e-5, 1e-4),
            (1e-5, 5e-5),
            (1e-6, 5e-6),
            (5e-7, 5e-7),
            (1e-7, 1e-7),
        ]
        self.learning_rate_i = 0

        self.critic_optimizer = tf.keras.optimizers.Adam(
            self.learning_rates[self.learning_rate_i][1]
        )
        self.actor_optimizer = tf.keras.optimizers.Adam(
            self.learning_rates[self.learning_rate_i][0]
        )

        data_spec = (
            tf.TensorSpec([sample_size * 2], tf.float64, "prev_state"),
            tf.TensorSpec([1], tf.float64, "action"),
            tf.TensorSpec([], tf.float32, "reward"),
            tf.TensorSpec([sample_size * 2], tf.float64, "state"),
        )
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec, batch_size=1, max_length=buffer_size
        )

        self.gamma = 0.99
        self.tau = 0.005
        self.ou_noise = OUActionNoise(
            mean=np.zeros(1), std_deviation=float(0.15) * np.ones(1)
        )

    def _get_models(self, ep_balance=None):
        if ep_balance is not None:
            self.actor_main = ActorModel(
                self.n_states,
                weights="./checkpoints/actor_model_{}.h5".format(ep_balance),
            ).get_model()
            self.critic_main = CriticModel(
                self.n_actions,
                self.n_states,
                weights="./checkpoints/critic_model_{}.h5".format(ep_balance),
            ).get_model()

            self.actor_target = ActorModel(
                self.n_states,
                weights="./checkpoints/target_actor_{}.h5".format(ep_balance),
            ).get_model()
            self.critic_target = CriticModel(
                self.n_actions,
                self.n_states,
                weights="./checkpoints/target_critic_{}.h5".format(ep_balance),
            ).get_model()

        else:
            self.actor_main = ActorModel(self.n_states).get_model()
            self.critic_main = CriticModel(self.n_actions, self.n_states).get_model()

            self.actor_target = ActorModel(self.n_states).get_model()
            self.critic_target = CriticModel(self.n_actions, self.n_states).get_model()

            # Making the weights equal initially
            self.actor_target.set_weights(self.actor_main.get_weights())
            self.critic_target.set_weights(self.critic_main.get_weights())

    @tf.function
    def _update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def _policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_main(state))
        chance = random.random()

        if chance < 0.2:
            if sampled_actions < 0.5:
                sampled_actions = tf.constant([random.uniform(0.15, 0.35)])
            else:
                sampled_actions = tf.constant([random.uniform(0.65, 0.85)])

        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.min_action, self.max_action)

        return [np.squeeze(legal_action)]

    @tf.function
    def _learn(self):
        sample = self.buffer.get_next(sample_batch_size=self.batch_size)

        state_batch = sample[0][0]
        action_batch = sample[0][1]
        reward_batch = sample[0][2]
        next_state_batch = sample[0][3]

        with tf.GradientTape() as tape:
            target_actions = self.actor_target(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.critic_target(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_main([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_main.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_main.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_main(state_batch, training=True)
            critic_value = self.critic_main([state_batch, actions], training=True)

            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_main.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_main.trainable_variables)
        )

    def train(self):
        ep_balance = 5000

        for ep in range(1000):
            prev_state = self.env.reset()
            episodic_reward = 0

            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self._policy(tf_prev_state, self.ou_noise)

                # Recieve state and reward from environment.
                state, reward, done, info = self.env.step(action)
                self.env.render()

                buffer_batch = (
                    tf.convert_to_tensor(prev_state),
                    tf.convert_to_tensor(action),
                    tf.convert_to_tensor(reward, dtype=tf.float32),
                    tf.convert_to_tensor(state),
                )
                values_batched = tf.nest.map_structure(
                    lambda t: tf.stack([t]), buffer_batch
                )

                # Store info in buffer
                self.buffer.add_batch(values_batched)
                episodic_reward += reward

                # Update model weights
                self._learn()
                self._update_target(
                    self.actor_target.variables, self.actor_main.variables
                )
                self._update_target(
                    self.critic_target.variables, self.critic_main.variables
                )

                # End this episode when done
                if done:
                    break

                prev_state = state

            if info != "error":
                net = self.validate()
            else:
                net = 0

            # Save model if it's a best performer
            if net > 0:
                ep_balance = net
                tqdm.write("Saving weights, Reward: {}".format(ep_balance))

                self.actor_main.save_weights(
                    "./checkpoints/actor_model_{:03.4g}.h5".format(ep_balance)
                )
                self.critic_main.save_weights(
                    "./checkpoints/critic_model_{:03.4g}.h5".format(ep_balance)
                )
                self.actor_target.save_weights(
                    "./checkpoints/target_actor_{:03.4g}.h5".format(ep_balance)
                )
                self.critic_target.save_weights(
                    "./checkpoints/target_critic_{:03.4g}.h5".format(ep_balance)
                )

                # if self.learning_rate_i >= len(self.learning_rates):
                #     break

                # actor_lr, critic_lr = self.learning_rates[self.learning_rate_i]
                # self.learning_rate_i += 1

                # # Reduce learning rate
                # critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
                # actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

                # self.buffer.actor_optimizer = actor_optimizer
                # self.buffer.critic_optimizer = critic_optimizer

    def validate(self, steps=None):
        init_balance = 1000
        balance_usd = init_balance
        balance_crypto = 1e-5
        if steps is None:
            steps = self.env.loader.steps

        loader = BinanceDataLoader(
            self.env.path, sample_size=self.sample_size, train_steps=steps
        )
        data_iterator = loader.process_data()
        self.env.pbar = tqdm(total=steps)
        nets = []
        prices = []
        total_reward = 0

        data = next(data_iterator)
        init_price = data["price_old"]
        while data:

            obs = data["observation"]
            price = data["price_old"]

            action = tf.squeeze(self.actor_main(np.array([obs]))).numpy()
            action = (action - 0.5) * 2

            reward = action * (data["price_n"] - data["price_old_n"])

            total_reward += reward

            # buy or hold
            if action >= 0:
                transaction_ammount = action * balance_usd
                # Prevent trade if agent wants to spend less than one dollar
                if transaction_ammount > 10:
                    balance_usd -= transaction_ammount
                    balance_crypto += transaction_ammount / price

            # sell
            else:
                transaction_ammount = action * balance_crypto
                # Prevent trade if agent wants to spend less than one dollar
                if transaction_ammount * price < -10:
                    balance_crypto += transaction_ammount
                    balance_usd -= transaction_ammount * price

            net = balance_usd + balance_crypto * price
            nets.append(net)
            prices.append(init_balance * price / init_price)

            self.env.pbar.set_postfix({"Net": net}, refresh=False)
            self.env.pbar.update(1)

            data = next(data_iterator)

        tqdm.write(
            "Environment reset!, Validation - Net worth: {}, Reward: {:.4f}".format(
                net, total_reward
            )
        )

        plt.title("Reward: {:.4f}".format(total_reward))
        plt.plot(prices, label="Price")
        plt.plot(nets, label="Net")
        plt.legend()
        plt.savefig("./plots/{:.4f}.png".format(total_reward))
        plt.clf()

        return total_reward
