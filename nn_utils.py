import tensorflow as tf


def _get_data(num_classes):

    X_train = tf.Variable(
        tf.placeholder(tf.float32, [None, 224, 224, 3], 'X_train'),
        trainable=False, collections=[],
        validate_shape=False, expected_shape=[None, 224, 224, 3]
    )
    Y_train = tf.Variable(
        tf.placeholder(tf.float32, [None, num_classes], 'Y_train'),
        trainable=False, collections=[],
        validate_shape=False, expected_shape=[None, num_classes]
    )
    batch_size = tf.Variable(
        tf.placeholder(tf.int32, [], 'batch_size'),
        trainable=False, collections=[]
    )
    init = tf.variables_initializer([X_train, Y_train, batch_size])

    # three values that you need to tweak
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3*64
    num_threads = 2

    x_batch, y_batch = tf.train.shuffle_batch(
        [X_train, Y_train], batch_size, capacity, min_after_dequeue,
        num_threads, enqueue_many=True,
        shapes=[[224, 224, 3], [num_classes]]
    )
    return init, x_batch, y_batch


def _add_summaries():
    summaries = []
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)

    for v in trainable_vars:
        summaries += [tf.summary.histogram(v.name[:-2] + '_hist', v)]
    for a in activations:
        summaries += [tf.summary.histogram(a.name[:-2] + '_activ_hist', a)]

    return tf.summary.merge(summaries)


def _is_early_stopping(losses, patience, index_to_watch):
    test_losses = [x[index_to_watch] for x in losses]
    if len(losses) > (patience + 4):
        average = (test_losses[-(patience + 4)] +
                   test_losses[-(patience + 3)] +
                   test_losses[-(patience + 2)] +
                   test_losses[-(patience + 1)] +
                   test_losses[-patience])/5.0
        return test_losses[-1] > average
    else:
        return False


def _assign_weights():

    assign_weights_dict = {}
    model_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)

    for v in model_vars:
        assign_weights_dict[v.name] = v.assign(
            tf.placeholder(tf.float32, v.shape, v.name[:-2])
        )

    return assign_weights_dict
