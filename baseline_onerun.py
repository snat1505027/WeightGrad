from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import cPickle as pickle
import socket

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)

def train():
  """Train CIFAR-10 for a number of steps."""

  g1 = tf.Graph()
  with g1.as_default():
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)
    grads  = cifar10.train_part1(loss, global_step)

    only_gradients = [g for g,_ in grads]
    only_vars = [v for _,v in grads]
       
    placeholder_gradients = []

    #with tf.device("/gpu:0"):
    for grad_var in grads :
        placeholder_gradients.append((tf.placeholder('float', shape=grad_var[0].get_shape()) ,grad_var[1]))
    feed_dict = {}
       
    for i,grad_var in enumerate(grads): 
       feed_dict[placeholder_gradients[i][0]] = np.zeros(placeholder_gradients[i][0].shape)
      
    train_op = cifar10.train_part2(global_step,placeholder_gradients)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.
        

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      while not mon_sess.should_stop():
        
        gradients = mon_sess.run(only_gradients,feed_dict = feed_dict)
        gradients2 = gradients

        feed_dict = {}
        for i,grad_var in enumerate(gradients2): 
           feed_dict[placeholder_gradients[i][0]] = gradients2[i]

        res = mon_sess.run(train_op, feed_dict=feed_dict)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  total_start_time = time.time()
  train()
  print("--- %s seconds ---" % (time.time() - total_start_time))

if __name__ == '__main__':
  tf.app.run()
