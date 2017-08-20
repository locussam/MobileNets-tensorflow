import tensorflow as tf
import shutil
import os
import time


# it decides if training must stop
def _is_early_stopping(losses, patience, index_to_watch):
    test_losses = [x[index_to_watch] for x in losses]
    if len(losses) > (patience + 4):
        # running average
        average = (test_losses[-(patience + 4)] +
                   test_losses[-(patience + 3)] +
                   test_losses[-(patience + 2)] +
                   test_losses[-(patience + 1)] +
                   test_losses[-patience])/5.0
        return test_losses[-1] > average
    else:
        return False


def train(run, graph, ops, train_tfrecords, val_tfrecords, batch_size,
          num_epochs, steps_per_epoch, validation_steps, patience=5,
          initial_weights=None, warm=False, initial_epoch=1, verbose=True):
    """Fit a defined network.

    Arguments:
        run: An integer that determines a folder where logs and the fitted model
            will be saved.
        graph: A Tensorflow graph.
        ops: A list of ops from the graph.
        train_tfrecords: A string, path to tfrecords train dataset file.
        val_tfrecords: A string, path to tfrecords validation dataset file.
        batch_size: An integer.
        num_epochs: An integer.
        steps_per_epoch: An integer, number of optimization steps per epoch.
        validation_steps: An integer, number of batches from validation dataset
            to evaluate on.
        patience: An integer, number of epochs before early stopping if
            test logloss isn't improving.
        initial_weights: A dictionary of weights to initialize network with or None.
        warm: Boolean, if `True` then resume training from the previously
            saved model.
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)
        verbose: Boolean, whether to print train and test logloss/accuracy
            during fitting.

    Returns:
        losses: A list of tuples containing train and test logloss/accuracy.
        is_early_stopped: Boolean, if `True` then fitting is stopped early.
    """

    # create folders for logging and saving
    dir_to_log = 'logs/run' + str(run)
    dir_to_save = 'saved/run' + str(run)
    if os.path.exists(dir_to_log) and not warm:
        shutil.rmtree(dir_to_log)
    if os.path.exists(dir_to_save) and not warm:
        shutil.rmtree(dir_to_save)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    sess = tf.Session(graph=graph)
    writer = tf.summary.FileWriter(dir_to_log, sess.graph)

    # get graph's ops
    data_init_op, predictions_op, log_loss_op, optimize_op,\
        grad_summaries_op, init_op, saver_op, assign_weights_op,\
        accuracy_op, summaries_op = ops

    if warm:
        saver_op.restore(sess, dir_to_save + '/model')
    else:
        sess.run(init_op)
        if initial_weights is not None:
            # keys in 'initial_weights' dict are full paths to the weights
            # for example: 'features/fire9/expand3x3/kernel:0'
            for w in initial_weights:
                op = assign_weights_op[w]
                sess.run(op, {'utilities/' + w: initial_weights[w]})

    # things that will be returned
    losses = []
    is_early_stopped = False

    training_epochs = range(
        initial_epoch,
        initial_epoch + num_epochs
    )

    # initialize data sources
    data_dict = {
        'input_pipeline/train_file:0': train_tfrecords,
        'input_pipeline/val_file:0': val_tfrecords,
        'input_pipeline/batch_size:0': batch_size
    }
    sess.run(data_init_op, data_dict)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # begin training
    for epoch in training_epochs:

        start_time = time.time()
        running_loss, running_accuracy = 0.0, 0.0

        # at zeroth step also collect metadata and summaries
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE
        )
        run_metadata = tf.RunMetadata()
        # do epoch's zeroth step
        _, batch_loss, batch_accuracy, summary, grad_summary = sess.run(
            [optimize_op, log_loss_op, accuracy_op, summaries_op, grad_summaries_op],
            options=run_options, run_metadata=run_metadata
        )
        writer.add_run_metadata(run_metadata, str(epoch))
        writer.add_summary(summary, epoch)
        writer.add_summary(grad_summary, epoch)
        running_loss += batch_loss
        running_accuracy += batch_accuracy

        # main training loop
        for step in range(1, steps_per_epoch):

            _, batch_loss, batch_accuracy = sess.run(
                [optimize_op, log_loss_op, accuracy_op]
            )

            running_loss += batch_loss
            running_accuracy += batch_accuracy

        # evaluate on the validation set
        test_loss, test_accuracy = _evaluate(
            sess, validation_steps, log_loss_op, accuracy_op
        )
        train_loss = running_loss/steps_per_epoch
        train_accuracy = running_accuracy/steps_per_epoch

        if verbose:
            print('{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f}  {5:.3f}'.format(
                epoch, train_loss, test_loss,
                train_accuracy, test_accuracy, time.time() - start_time
            ))

        # collect all losses and accuracies
        losses += [(epoch, train_loss, test_loss, train_accuracy, test_accuracy)]

        # consider a possibility of early stopping
        if _is_early_stopping(losses, patience, 2):
            is_early_stopped = True
            break

    coord.request_stop()
    coord.join(threads)

    saver_op.save(sess, dir_to_save + '/model')
    sess.close()

    return losses, is_early_stopped


def _evaluate(sess, validation_steps, log_loss_op, accuracy_op):

    test_loss, test_accuracy = 0.0, 0.0
    for i in range(validation_steps):
        batch_loss, batch_accuracy = sess.run(
            [log_loss_op, accuracy_op], {'control/is_training:0': False}
        )
        test_loss += batch_loss
        test_accuracy += batch_accuracy

    test_loss /= validation_steps
    test_accuracy /= validation_steps

    return test_loss, test_accuracy


def predict_proba(graph, ops, X, run=None, network_weights=None):
    """Predict probabilities with a fitted model.

    Arguments:
        graph: A Tensorflow graph.
        ops: A list of ops from the graph.
        X: A numpy array of shape [n_samples, n_features]
            and of type 'float32'.
        run: An integer that determines a folder where a fitted model
            is saved or None.
        network_weights: A dictionary of weights to initialize
            network with or None. You must specify either run or network_weights.

    Returns:
        predictions: A numpy array of shape [n_samples, n_classes]
            and of type 'float32'.
    """
    sess = tf.Session(graph=graph)

    # get graph's ops
    data_init_op, predictions_op, log_loss_op, optimize_op,\
        grad_summaries_op, init_op, saver_op, assign_weights_op,\
        accuracy_op, summaries_op = ops
    # only predictions_op, saver_op, and assign_weights_op are used here

    if run is not None:
        saver_op.restore(sess, 'saved/run' + str(run) + '/model')
    elif network_weights is not None:
        for w in network_weights:
            op = assign_weights_op[w]
            sess.run(op, {'utilities/' + w: network_weights[w]})
    else:
        print('Specify run or network_weights!')

    feed_dict = {'inputs/X:0': X, 'control/is_training:0': False}
    predictions = sess.run(predictions_op, feed_dict)
    sess.close()

    return predictions
