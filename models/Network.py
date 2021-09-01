from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend, layers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from utils.callbacks import *
from utils.layers import *
from utils.metrics import *
from tensorflow.keras.optimizers import Adam


def cosine_decay(max_epochs, max_lr, min_lr=1e-5, warmup=False):
    """
    cosine annealing scheduler.
    :param max_epochs: max epochs
    :param max_lr: max lr
    :param min_lr: min lr
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = min_lr + (max_lr - min_lr) * (
                1 + np.cos(np.pi * epoch / max_epochs)) / 2
        return lrate

    return decay

class Network():
    def __init__(self, input_shape, num_classes=None, weights=None, save_weights_path=None, dataset_name='default', lr=1e-3) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self.save_weights_path = save_weights_path
        self.dataset_name = dataset_name
        self.lr = lr

    def get_callbacks(self, num_epochs, save_weights=None, dataset_name='default', log_dir=None):
        callbacks = []

        if log_dir is not None:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            callbacks.append(tensorboard)
        
        if save_weights is not None:
            if save_weights[-1] != '/':
                save_weights += '/'

            Path(save_weights).mkdir(parents=True, exist_ok=True)

            filepath = save_weights + '{}_{}_'.format(self.model_name, dataset_name) + '{epoch:04d}_mae_{val_mean_absolute_error:05.4f}.h5'

            cp_callback_0 = tf.keras.callbacks.ModelCheckpoint(
                    filepath=filepath,
                    verbose=1,
                    save_weights_only=True,
                    monitor='val_mean_absolute_error',
                    mode='min',
                    save_best_only=True
            )
            callbacks.append(cp_callback_0)


        lr_scheduler = LearningRateScheduler(cosine_decay(num_epochs, self.lr), verbose=1)
        callbacks.append(lr_scheduler)

        return callbacks
    
    def categorical_crossentropy_with_logits(self, y_true, y_pred):
        # compute cross entropy
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

        # compute loss
        loss = backend.mean(backend.sum(cross_entropy, axis=[1, 2]))
        return loss

    
        














    







