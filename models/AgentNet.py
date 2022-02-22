from xml.sax.xmlreader import InputSource
from tensorflow.python.ops.gen_array_ops import shape
from models.Network import *

class AgentNetActor(Network):
    def __init__(self, num_states, output_size=1, num_classes=None, weights=None, save_weights_path=None, dataset_name='default', lr=1e-3):
        self.model_name = 'AgentNetActor'
        self.num_states = num_states
        self.output_size = output_size
        super(AgentNetActor, self).__init__(num_classes=num_classes, weights=weights, save_weights_path=save_weights_path, dataset_name=dataset_name, lr=lr)

    def get_model(self):
        in_shape = self.num_states
        out_size = self.output_size

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = keras.Input(shape=(in_shape,), name='observation_input_0_actor')

        data_size = in_shape // 2
        inputs_0 = tf.reshape(inputs[:, :data_size], [-1, 1, data_size])
        inputs_1 = tf.reshape(inputs[:, data_size:], [-1, 1, data_size])

        x_conv = layers.Conv1D(32, 3, padding='causal')(inputs_0)
        x_conv = layers.BatchNormalization()(x_conv)
        x_conv = layers.ReLU()(x_conv)
        x_conv = layers.Conv1D(32, 3, padding='causal')(x_conv)
        x_conv = layers.BatchNormalization()(x_conv)
        x_conv = layers.ReLU()(x_conv)
        x_conv = layers.Conv1D(32, 3, padding='causal')(x_conv)
        x_conv = layers.BatchNormalization()(x_conv)
        x_conv = layers.ReLU()(x_conv)

        x = layers.GlobalAveragePooling1D()(x_conv)

        x_conv = layers.Conv1D(32, 3, padding='causal')(inputs_1)
        x_conv = layers.BatchNormalization()(x_conv)
        x_conv = layers.ReLU()(x_conv)
        x_conv = layers.Conv1D(32, 3, padding='causal')(x_conv)
        x_conv = layers.BatchNormalization()(x_conv)
        x_conv = layers.ReLU()(x_conv)
        x_conv = layers.Conv1D(32, 3, padding='causal')(x_conv)
        x_conv = layers.BatchNormalization()(x_conv)
        x_conv = layers.ReLU()(x_conv)

        x = layers.add([x, layers.GlobalAveragePooling1D()(x_conv)])
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.BatchNormalization()(x)

        outputs = layers.Dense(out_size, activation='sigmoid', kernel_initializer=last_init)(x)

        model = keras.Model(inputs, outputs)
        model.summary()
        
        if self.weights is not None:
            model.load_weights(self.weights)


        return model

class AgentNetCritic(Network):
    def __init__(self, num_actions, num_states, output_size=1, num_classes=None, weights=None, save_weights_path=None, dataset_name='default', lr=1e-3):
        self.model_name = 'AgentNetCritic'
        self.num_actions = num_actions
        self.num_states = num_states
        self.output_size = output_size
        super(AgentNetCritic, self).__init__(num_classes=num_classes, weights=weights, save_weights_path=save_weights_path, dataset_name=dataset_name, lr=lr)

    def get_model(self):
        out_size = self.output_size

        action_input = keras.Input(shape=(self.num_actions), name='action_input_critic')
        x_a = layers.Dense(32, activation='relu')(action_input)
        x_a = layers.BatchNormalization()(x_a)
        x_a = layers.Dense(64, activation='relu')(x_a)

        observation_input = keras.Input(shape=(self.num_states), name='observation_input_critic')

        x_o = layers.Dense(32, activation='relu')(observation_input)
        x_o = layers.BatchNormalization()(x_o)
        x_o = layers.Dense(64, activation='relu')(x_o)

        x = layers.Concatenate()([x_o, x_a])

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(out_size)(x)

        model = keras.Model(inputs=[observation_input, action_input], outputs=outputs)
        model.summary()
        
        if self.weights is not None:
            model.load_weights(self.weights)


        return model