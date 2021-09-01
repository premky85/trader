import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import init_ops
import numpy as np

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

class IoU(tf.keras.metrics.Metric):
    def __init__(self, name='IoU', num_classes=2, dtype=None, **kwargs):
        super(IoU, self).__init__(name=name, dtype=dtype, **kwargs)
        self.errors = []
        self.num_classes = num_classes
        self.c_m = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=tf.dtypes.int32
        )#tf.zeros((num_classes, num_classes), dtype=tf.dtypes.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])#tf.reshape(y_true, [-1])#
        y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])#tf.reshape(tf.round(y_pred), [-1])#

        c_m = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.dtypes.int32)
        self.c_m.assign_add(c_m)

        #err = np.sum((y_true == 0) == (y_pred == 0)) / (np.sum(y_pred == 0) + np.sum(y_true == 0))

        #self.errors.append(c_m[1][1] / (c_m[0][1] + c_m[1][0] + c_m[1][1]))

    def result(self):
        c_m = self.c_m
        return c_m[1][1] / (c_m[1][0] + c_m[0][1] + c_m[1][1])#np.mean(self.errors)

    def reset_states(self):
        K.set_value(self.c_m, tf.zeros((self.num_classes, self.num_classes)))

class MDR(tf.keras.metrics.Metric):
    def __init__(self, name='MDR', num_classes=2, dtype=None, **kwargs):
        super(MDR, self).__init__(name=name, dtype=dtype, **kwargs)
        self.errors = []
        self.num_classes = num_classes
        self.c_m = self.c_m = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=tf.dtypes.int32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])#.numpy()
        y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])#.numpy()

        c_m = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.dtypes.int32)
        self.c_m.assign_add(c_m)

    def result(self):
        c_m = self.c_m
        return c_m[1][0] / (c_m[1][0] + c_m[1][1])

    def reset_states(self):
        K.set_value(self.c_m, tf.zeros((self.num_classes, self.num_classes)))

class FDR(tf.keras.metrics.Metric):
    def __init__(self, name='FDR', num_classes=2, dtype=None, **kwargs):
        super(FDR, self).__init__(name=name, dtype=dtype, **kwargs)
        self.errors = []
        self.num_classes = num_classes
        self.c_m = self.c_m = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=tf.dtypes.int32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.argmax(y_true, axis=-1), [-1])#.numpy()
        y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])#.numpy()

        c_m = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.dtypes.int32)
        self.c_m.assign_add(c_m)

    def result(self):
        c_m = self.c_m
        return c_m[0][1] / (c_m[0][1] + c_m[1][1])

    def reset_states(self):
        K.set_value(self.c_m, tf.zeros((self.num_classes, self.num_classes)))

class E1_Error(tf.keras.metrics.Metric):
    def __init__(self, name='E1', dtype=None, **kwargs):
        super(E1_Error, self).__init__(name=name, dtype=dtype, **kwargs)
        self.errors = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1).numpy()
        y_pred = tf.argmax(y_pred, axis=-1).numpy()

        self.errors.append(np.sum(y_true != y_pred) / np.size(y_true))

    def result(self):
        return np.mean(self.errors)

    def reset_states(self):
        self.errors = []

class E2_Error(tf.keras.metrics.Metric):
    def __init__(self, name='E2', dtype=None, **kwargs):
        super(E2_Error, self).__init__(name=name, dtype=dtype, **kwargs)
        self.errors = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1).numpy()
        y_pred = tf.argmax(y_pred, axis=-1).numpy()

        n = tf.size(y_true)

        fpr = np.sum((y_pred - y_true) > 0) / n
        fnr = np.sum((y_true - y_pred) > 0) / n

        self.errors.append(fpr * 0.5 + fnr * 0.5)

    def result(self):
        return np.mean(self.errors)

    def reset_states(self):
        self.errors = []