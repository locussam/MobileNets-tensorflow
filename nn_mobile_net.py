import tensorflow as tf
import numpy as np
import shutil
import os
import time
from nn_utils import _get_data, _add_summaries, _is_early_stopping, _assign_weights
from nn_parts import _mapping, _add_weight_decay


class MobileNet:
    def __init__(self, num_classes, optimizer, weight_decay=None):
        self.graph = tf.Graph()
        with self.graph.as_default():

            with tf.device('/cpu:0'), tf.variable_scope('input_pipeline'):
                self.init_data, x_batch, y_batch = _get_data(num_classes)

            with tf.variable_scope('inputs'):
                X = tf.placeholder_with_default(x_batch, [None, 224, 224, 3], 'X')
                Y = tf.placeholder_with_default(y_batch, [None, num_classes], 'Y')

            with tf.variable_scope('control'):
                is_training = tf.placeholder_with_default(True, [], 'is_training')

            with tf.variable_scope('MobilenetV1'):
                logits = _mapping(X, num_classes, is_training)

            with tf.variable_scope('softmax'):
                self.predictions = tf.nn.softmax(logits)

            with tf.variable_scope('log_loss'):
                self.log_loss = tf.losses.softmax_cross_entropy(Y, logits)

            if weight_decay is not None:
                _add_weight_decay(weight_decay)

            with tf.variable_scope('total_loss'):
                total_loss = tf.losses.get_total_loss()

            ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            #with tf.variable_scope('optimizer'):
            grads_and_vars = optimizer.compute_gradients(total_loss)
            self.optimize = optimizer.apply_gradients(grads_and_vars)

            self.grad_summaries = tf.summary.merge(
                [tf.summary.histogram(v.name[:-2] + '_grad_hist', g)
                 for g, v in grads_and_vars]
            )

            with tf.variable_scope('utilities'):
                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()
                self.assign_weights = _assign_weights()
                is_equal = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(Y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))

            self.merged = _add_summaries()

        self.graph.finalize()

    def fit(self, run, X_train, Y_train, X_test, Y_test,
            batch_size, num_epochs, validation_step, patience,
            initial_weights, warm=False, verbose=True):

        dir_to_log = 'logs/run' + str(run)
        dir_to_save = 'saved/run' + str(run)
        if os.path.exists(dir_to_log) and not warm:
            shutil.rmtree(dir_to_log)
        if os.path.exists(dir_to_save) and not warm:
            shutil.rmtree(dir_to_save)
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save)

        sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(dir_to_log, sess.graph)

        feed_dict_train = {
            'input_pipeline/X_train:0': X_train,
            'input_pipeline/Y_train:0': Y_train,
            'input_pipeline/batch_size:0': batch_size
        }
        sess.run(self.init_data, feed_dict_train)

        if not warm:
            sess.run(self.init)
            for w in initial_weights:
                op = self.assign_weights[w]
                sess.run(op, {'utilities/' + w: initial_weights[w]})
        else:
            self.saver.restore(sess, dir_to_save + '/model')

        num_batches = int(len(X_train)/batch_size)
        losses = []
        is_early_stopped = False
        self.steps_trained = 0 if not warm else self.steps_trained
        training_steps = range(
            self.steps_trained + 1,
            self.steps_trained + num_epochs*num_batches + 1
        )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        self.start = time.time()
        for step in training_steps:
            if (step % validation_step != 0):
                sess.run(self.optimize)
            else:
                losses += [self._get_summaries(
                    step, sess, X_train, Y_train, X_test, Y_test,
                    batch_size, num_batches, verbose
                )]
                if _is_early_stopping(losses, patience, 1):
                    is_early_stopped = True
                    break
                self.start = time.time()

        coord.request_stop()
        coord.join(threads)
        self.saver.save(sess, dir_to_save + '/model')
        self.run = run
        self.steps_trained = step
        sess.close()

        return losses, is_early_stopped

    def predict_proba(self, X):
        sess = tf.Session(graph=self.graph)
        self.saver.restore(sess, 'saved/run' + str(self.run) + '/model')
        feed_dict = {'inputs/X:0': X, 'control/is_training:0': False}
        predictions = sess.run(self.predictions, feed_dict)
        sess.close()
        return predictions

    def _get_summaries(self, step, sess, X_train, Y_train,
                       X_test, Y_test, batch_size, num_batches, verbose):

            entry = '{0:.2f}'.format(step/num_batches)
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE
            )
            run_metadata = tf.RunMetadata()
            _, train_summary, train_grad_summary = sess.run(
                [self.optimize, self.merged, self.grad_summaries],
                options=run_options, run_metadata=run_metadata
            )
            self.writer.add_run_metadata(run_metadata, entry)
            self.writer.add_summary(train_summary, step)
            self.writer.add_summary(train_grad_summary, step)

            choice = np.random.choice(
                np.arange(0, len(X_train)),
                size=8*batch_size, replace=False
            )

            feed_dict_train = {
                'inputs/X:0': X_train[choice],
                'inputs/Y:0': Y_train[choice],
                'control/is_training:0': False
            }
            feed_dict_test = {
                'inputs/X:0': X_test,
                'inputs/Y:0': Y_test,
                'control/is_training:0': False
            }
            score_ops = [self.log_loss, self.accuracy]
            train_loss, train_accuracy = sess.run(score_ops, feed_dict_train)
            test_loss, test_accuracy = sess.run(score_ops, feed_dict_test)
            end = time.time()

            if verbose:
                print('{0}  {1:.3f} {2:.3f} {3:.3f} {4:.3f}  {5:.3f}'.format(
                    entry, train_loss, test_loss,
                    train_accuracy, test_accuracy, end - self.start
                ))

            return train_loss, test_loss
