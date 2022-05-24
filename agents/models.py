from __future__ import print_function
import tensorflow as tf
from models.Network import *


class ActorNet(Network):
    def __init__(
        self,
        num_states,
        states_spec,
        num_actions=1,
        weights=None,
        save_weights_path=None,
        dataset_name="default",
        lr=1e-3,
    ):
        self.model_name = "ActorNet"
        self.num_states = num_states
        self.num_actions = num_actions
        self.states_spec = states_spec

        super(ActorNet, self).__init__(
            num_classes=None,
            weights=weights,
            save_weights_path=save_weights_path,
            dataset_name=dataset_name,
            lr=lr,
        )

    def get_model(self, interposed=False):
        in_shape = self.num_states
        out_size = self.num_actions
        states_spec = self.states_spec

        inputs = keras.Input(shape=(in_shape,), name="observation_input_0_actor")

        inputs_balance = tf.reshape(inputs[:, :2], [-1, 2])

        i = states_spec[0] + states_spec[1] + 2
        inputs_prices = tf.reshape(inputs[:, 2:i], [-1, states_spec[0], 2])
        inputs_indicators = tf.reshape(
            inputs[:, i:], [-1, states_spec[2], len(states_spec) - 2]
        )

        x_1 = inputs_prices
        x_2 = inputs_indicators

        for filters in [16, 16, 32, 32, 128, 128]:

            x_1 = layers.Conv1D(
                filters, 3, padding="valid", activation=layers.LeakyReLU(alpha=0.01)
            )(x_1)
            x_1 = layers.BatchNormalization()(x_1)
            x_1 = layers.MaxPool1D(strides=2)(x_1)

            x_2 = layers.Conv1D(
                filters, 3, padding="valid", activation=layers.LeakyReLU(alpha=0.01)
            )(x_2)
            x_2 = layers.BatchNormalization()(x_2)
            x_2 = layers.MaxPool1D(strides=2)(x_2)

        x_1 = layers.Flatten()(x_1)
        x_2 = layers.Flatten()(x_2)
        x = layers.concatenate([x_1, x_2, inputs_balance], axis=-1)
        # x = layers.BatchNormalization()(x)

        x = layers.Dense(
            256,
            activation=layers.LeakyReLU(alpha=0.01),
        )(x)
        x = layers.BatchNormalization()(x)
        x_i = layers.Dense(
            128, activation=layers.LeakyReLU(alpha=0.01), name="interposed_output"
        )(x)
        x = layers.BatchNormalization()(x)

        outputs = layers.Dense(out_size)(x)

        if interposed:
            outputs = x_i  # layers.concatenate([outputs, x_i], axis=-1)

        model = keras.Model(inputs, outputs)
        model.summary()

        if self.weights is not None:
            model.load_weights(self.weights)

        return model


class CriticNet(Network):
    def __init__(
        self,
        num_actions,
        num_states,
        states_spec,
        output_size=1,
        weights=None,
        save_weights_path=None,
        dataset_name="default",
        lr=1e-3,
        observation_spec=None,
        action_spec=None,
    ):
        self.model_name = "CriticNet"
        self.num_actions = num_actions
        self.num_states = num_states
        self.output_size = output_size
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.states_spec = states_spec
        super(CriticNet, self).__init__(
            num_classes=None,
            weights=weights,
            save_weights_path=save_weights_path,
            dataset_name=dataset_name,
            lr=lr,
        )

    def get_model(self):
        out_size = self.output_size
        states_spec = self.states_spec

        action_input = keras.Input(
            shape=(self.num_actions,), name="action_input_critic"
        )

        observation_input = keras.Input(
            shape=(self.num_states,), name="observation_input_critic"
        )

        inputs_balance = tf.reshape(observation_input[:, :2], [-1, 2])

        i = states_spec[0] + states_spec[1] + 2
        inputs_prices = tf.reshape(observation_input[:, 2:i], [-1, states_spec[0], 2])
        inputs_indicators = tf.reshape(
            observation_input[:, i:], [-1, states_spec[2], len(states_spec) - 2]
        )

        x_1 = inputs_prices
        x_2 = inputs_indicators

        for filters in [32, 64, 64, 128]:

            x_1 = layers.Conv1D(
                filters, 3, padding="valid", activation=layers.LeakyReLU(alpha=0.01)
            )(x_1)
            x_1 = layers.BatchNormalization()(x_1)
            x_1 = layers.MaxPool1D(strides=4)(x_1)

            x_2 = layers.Conv1D(
                filters, 3, padding="valid", activation=layers.LeakyReLU(alpha=0.01)
            )(x_2)
            x_2 = layers.BatchNormalization()(x_2)
            x_2 = layers.MaxPool1D(strides=4)(x_2)

        x_1 = layers.Flatten()(x_1)
        x_2 = layers.Flatten()(x_2)

        x = layers.concatenate([x_1, x_2, action_input, inputs_balance], axis=-1)

        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.01))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01))(x)
        x = layers.BatchNormalization()(x)

        outputs = layers.Dense(out_size, activation="tanh", name="critic_output")(x)

        model = keras.Model(inputs=[observation_input, action_input], outputs=outputs)
        model.summary()

        if self.weights is not None:
            model.load_weights(self.weights)

        return model


class RefineNet(Network):
    def __init__(
        self,
        num_actions=1,
        num_states=128,
        num_classes=None,
        weights=None,
        save_weights_path=None,
        dataset_name="default",
        lr=1e-3,
    ):
        self.model_name = "RefineNet"
        self.num_actions = num_actions
        self.num_states = num_states
        super(RefineNet, self).__init__(
            num_classes=num_classes,
            weights=weights,
            save_weights_path=save_weights_path,
            dataset_name=dataset_name,
            lr=lr,
        )

    def get_model(self):
        inputs = keras.Input(shape=(self.num_states), name="refine_net_input")
        x = layers.Dense(64, activation=layers.LeakyReLU(alpha=0.01))(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.num_actions)(x)

        model = keras.Model(inputs=inputs, outputs=tf.reshape(outputs, [-1]))
        model.summary()

        if self.weights is not None:
            model.load_weights(self.weights)

        return model
