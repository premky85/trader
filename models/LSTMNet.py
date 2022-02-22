from models.Network import *

class LSTMNet(Network):
    def __init__(self, input_shape, output_size, num_classes=None, weights=None, save_weights_path=None, dataset_name='default', lr=1e-3):
        self.model_name = 'LSTMNet'
        self.output_size = output_size
        super(LSTMNet, self).__init__(input_shape=input_shape, num_classes=num_classes, weights=weights, save_weights_path=save_weights_path, dataset_name=dataset_name, lr=lr)

    def get_model(self):
        in_shape = self.input_shape
        out_size = self.output_size

        inputs = keras.Input(shape=in_shape)

        x = layers.LSTM(32, return_sequences=True)(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(8, activation='relu')(x)
        outputs = layers.Dense(out_size, activation='relu')(x)

        model = keras.Model(inputs, outputs)
        model.summary()
        
        
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mean_squared_error', metrics=['mean_absolute_error'])

        if self.weights is not None:
            model.load_weights(self.weights)


        return model

    