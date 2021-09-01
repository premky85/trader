import tensorflow as tf
from tensorflow.keras import layers, backend

class GlobalAveragePooling2D(layers.GlobalAveragePooling2D):
    def __init__(self, keep_dims=False, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).call(inputs)
        else:
            return backend.mean(inputs, axis=[1, 2], keepdims=True)

    def compute_output_shape(self, input_shape):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).compute_output_shape(input_shape)
        else:
            input_shape = tf.TensorShape(input_shape).as_list()
            return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])

    def get_config(self):
        config = super(GlobalAveragePooling2D, self).get_config()
        config['keep_dim'] = self.keep_dims
        return config

def Unpooling(inputs, output_shape, argmax):
        """
        Performs unpooling, as explained in:
        https://www.oreilly.com/library/view/hands-on-convolutional-neural/9781789130331/6476c4d5-19f2-455f-8590-c6f99504b7a5.xhtml
        :param inputs: Input Tensor.
        :param output_shape: Desired output shape. For example, on 2D unpooling, this should be 4D (because of number of samples and channels).
        :param argmax: Result argmax from tf.nn.max_pool_with_argmax
            https://www.tensorflow.org/api_docs/python/tf/nn/max_pool_with_argmax
        """
        flat_output_shape = tf.cast(tf.reduce_prod(output_shape), tf.int64)

        updates = tf.reshape(inputs, [-1])
        indices = tf.expand_dims(tf.reshape(argmax, [-1]), axis=-1)

        ret = tf.scatter_nd(indices, updates, shape=[flat_output_shape])
        ret = tf.reshape(ret, output_shape)
        return ret

def Unpool(pool, ind, ksize=[1, 2, 2, 1], name=None):
    input_shape = tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

    flat_input_size = tf.math.cumprod(input_shape)[-1]
    flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

    pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                    shape=tf.stack([input_shape[0], 1, 1, 1]))
    b = tf.ones_like(ind) * batch_range
    b = tf.reshape(b, tf.stack([flat_input_size, 1]))
    ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
    ind_ = tf.concat([b, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, tf.stack(output_shape))

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
    ret.set_shape(set_output_shape)

    return ret