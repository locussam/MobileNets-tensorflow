import tensorflow as tf
import math

initial_a_parameter = 0.01


def _fully_connected(X, size):
    input_dim = X.shape.as_list()[1]
    maxval = math.sqrt(2.0/(input_dim*(1.0 + initial_a_parameter**2)))

    W = tf.get_variable(
        'W', [input_dim, size], tf.float32,
        tf.random_uniform_initializer(-maxval, maxval)
    )

    return tf.matmul(X, W)


def _dropout(X, rate, is_training):
    keep_prob = tf.constant(
        1.0 - rate, tf.float32,
        [], 'keep_prob'
    )
    result = tf.cond(
        is_training,
        lambda: tf.nn.dropout(X, keep_prob),
        lambda: tf.identity(X),
        name='dropout'
    )
    return result


def _nonlinearity(X):
    a = tf.get_variable(
        'a', [], tf.float32,
        tf.constant_initializer(initial_a_parameter)
    )
    return tf.where(tf.greater_equal(X, 0.0), X, tf.scalar_mul(a, X), name='PReLU')


def _batch_norm(X, is_training):
    return tf.contrib.layers.batch_norm(
        X, is_training=is_training, scale=True,
        fused=True, scope='BatchNorm'
    )


def _affine(X, size):
    input_dim = X.shape.as_list()[1]
    maxval = math.sqrt(2.0/(input_dim*(1.0 + initial_a_parameter**2)))

    W = tf.get_variable(
        'W', [input_dim, size], tf.float32,
        tf.random_uniform_initializer(-maxval, maxval)
    )

    b = tf.get_variable(
        'b', [size], tf.float32,
        tf.zeros_initializer()
    )

    return tf.nn.bias_add(tf.matmul(X, W), b)


def _conv(X, filters, kernel=1, strides=1, padding='SAME', use_bias=False, trainable=True):

    in_channels = X.shape.as_list()[-1]

    W = tf.get_variable(
        'weights', [kernel, kernel, in_channels, filters],
        tf.float32, trainable=trainable
    )

    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, W)

    if use_bias:
        b = tf.get_variable(
            'biases', [filters], tf.float32,
            tf.zeros_initializer(), trainable=trainable
        )
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)
        return tf.nn.bias_add(
            tf.nn.conv2d(X, W, [1, strides, strides, 1], padding), b
        )

    return tf.nn.conv2d(X, W, [1, strides, strides, 1], padding)




def _depthwise_conv2d(X, kernel=3, strides=1, padding='SAME', trainable=True):

    in_channels = X.shape.as_list()[-1]

    W = tf.get_variable(
        'depthwise_weights', [kernel, kernel, in_channels, 1],
        tf.float32, trainable=trainable
    )

    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, W)

    return tf.nn.depthwise_conv2d(X, W, [1, strides, strides, 1], padding)


def _global_average_pooling(X):
    return tf.reduce_mean(X, axis=[1, 2], keep_dims=True)


def _mapping(X, num_classes, is_training):

    with tf.variable_scope('Conv2d_0'):
        result = _conv(X, 32, kernel=3, strides=2)
        result = _batch_norm(result, is_training)
        result = tf.nn.relu6(result, name='ReLU')

    filters = [
        64, 128, 128, 256, 256, 512,
        512, 512, 512, 512, 512, 1024,
        1024
    ]

    strides = [
        1, 2, 1, 2, 1, 2,
        1, 1, 1, 1, 1, 2,
        1
    ]

    for i in range(0, 13):
        with tf.variable_scope('Conv2d_' + str(i + 1) + '_depthwise'):
            result = _depthwise_conv2d(result, strides=strides[i])
            result = _batch_norm(result, is_training)
            result = tf.nn.relu6(result, name='ReLU')

        with tf.variable_scope('Conv2d_' + str(i + 1) + '_pointwise'):
            result = _conv(result, filters[i])
            result = _batch_norm(result, is_training)
            result = tf.nn.relu6(result, name='ReLU')

    result = _global_average_pooling(result)
    result = _dropout(result, 1e-3, is_training)

    with tf.variable_scope('Logits/Conv2d_1c_1x1'):
        logits = _conv(result, num_classes, use_bias=True, trainable=True)
        logits = tf.squeeze(logits, axis=[1, 2])

    return logits


def _add_weight_decay(weight_decay):

    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )

    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = [v for v in trainable if 'kernel' in v.name]

    for K in kernels:
        l2_loss = tf.multiply(
            weight_decay, tf.nn.l2_loss(K), name='l2_loss'
        )
        tf.losses.add_loss(l2_loss)
