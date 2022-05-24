from __future__ import print_function
import tensorflow as tf
from models.Network import *


class ActorModel(Network):
    def __init__(
        self,
        num_states,
        num_actions=1,
        num_classes=None,
        weights=None,
        save_weights_path=None,
        dataset_name="default",
        lr=1e-3,
    ):
        self.model_name = "AgentNetActor"
        self.num_states = num_states
        self.num_actions = num_actions
        super(ActorModel, self).__init__(
            num_classes=num_classes,
            weights=weights,
            save_weights_path=save_weights_path,
            dataset_name=dataset_name,
            lr=lr,
        )

    def get_model(self):
        in_shape = self.num_states
        out_size = self.num_actions

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
        # model.summary()

        if self.weights is not None:
            model.load_weights(self.weights)

        return model
