from tensorflow.python.ops.gen_array_ops import shape
from models.Network import *

class AgentNetActor(Network):
    def __init__(self, input_shape=(1, 3,), output_size=1, num_classes=None, weights=None, save_weights_path=None, dataset_name='default', lr=1e-3):
        self.model_name = 'AgentNetActor'
        self.output_size = output_size
        super(AgentNetActor, self).__init__(input_shape=input_shape, num_classes=num_classes, weights=weights, save_weights_path=save_weights_path, dataset_name=dataset_name, lr=lr)

    def get_model(self):
        in_shape = self.input_shape
        out_size = self.output_size

        inputs = keras.Input(shape=in_shape, name='observation_input_actor')
        x = layers.Flatten()(inputs)

        x = layers.Dense(32, activation='relu')(x)
        #x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        #x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(8, activation='relu')(x)
        #x = layers.BatchNormalization()(x)
        outputs = layers.Dense(out_size, activation='tanh')(x) / 100
        #x = layers.Flatten()(x)
        #outputs = x#layers.Softmax()(x)

        model = keras.Model(inputs, outputs)
        model.summary()
        
        
        #model.compile()#optimizer=Adam(learning_rate=self.lr), loss='mean_squared_error', metrics='mean_absolute_error')

        if self.weights is not None:
            model.load_weights(self.weights)


        return model

class AgentNetCritic(Network):
    def __init__(self, critic_action_input, input_shape=(3,), output_size=1, num_classes=None, weights=None, save_weights_path=None, dataset_name='default', lr=1e-3):
        self.model_name = 'AgentNetCritic'
        self.output_size = output_size
        self.critic_action_input = critic_action_input
        super(AgentNetCritic, self).__init__(input_shape=input_shape, num_classes=num_classes, weights=weights, save_weights_path=save_weights_path, dataset_name=dataset_name, lr=lr)

    def get_model(self):
        out_size = self.output_size

        action_input = self.critic_action_input#keras.Input(shape=(nb_actions,), name='action_input')
        observation_input = keras.Input(shape=(1,) + self.input_shape, name='observation_input_critic')
        flattened_observation = layers.Flatten()(observation_input)
        x = layers.Concatenate()([action_input, flattened_observation])




        #inputs = keras.Input(shape=in_shape)

        x = layers.Dense(32, activation='relu')(x)
        #x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        #x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(8, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(out_size)(x)
        #x = layers.Flatten()(x)
        #outputs = x#layers.Softmax()(x)

        model = keras.Model(inputs=[action_input, observation_input], outputs=outputs)
        model.summary()
        
        
        #model.compile()#optimizer=Adam(learning_rate=self.lr), loss='mean_squared_error', metrics='mean_absolute_error')

        if self.weights is not None:
            model.load_weights(self.weights)


        return model