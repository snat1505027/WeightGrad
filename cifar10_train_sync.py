from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import pickle as pickle
import socket

from datetime import datetime
import time

import tensorflow as tf

import cifar10

#TCP_IP = '127.0.0.1'
TCP_IP = '18.191.119.142'
TCP_PORT = 5014

port = 0
port_main = 0
s = 0

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
def safe_recv(size, server_socket):
  data = ""
  temp = ""
  data = bytearray()
  recv_size = 0
  while 1:
    try:
        #print("~~~~~~~~~INSIDE safe_recv")
        temp = server_socket.recv(size-len(data))
        data.extend(temp)
        recv_size = len(data)
        if recv_size >= size:
            break
    except:
        print("Error")
  data = bytes(data)
  #print(type(data))
  return data

def train():
  """Train CIFAR-10 for a number of steps."""

  g1 = tf.Graph()
  with g1.as_default():
    #global_step = tf.contrib.framework.get_or_create_global_step()
    global_step = tf.Variable(-1, name='global_step', trainable=False, dtype=tf.int32)
    increment_global_step_op = tf.assign(global_step, global_step+1)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs1()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)
    grads  = cifar10.train_part1(loss, global_step)

    only_gradients = [g for g,_ in grads]

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
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:
            #log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      global port
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.connect((TCP_IP, port_main))
      #receiving the variable values
      #print("Connected::::::::::::::")
      #recv_size = safe_recv(8, s)
      recv_size = safe_recv(17, s)
      #print("recv_size:::{}, {}".format(recv_size, type(recv_size)))
      #recv_size = bytes(recv_size,'utf-8')
      recv_size = pickle.loads(recv_size)
      #print("recv_size in number:::{}".format(recv_size))
      recv_data = safe_recv(recv_size, s)
      var_vals = pickle.loads(recv_data)
      s.close()
      feed_dict = {}
      i=0
      for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        feed_dict[v] = var_vals[i]
        i=i+1
      print("Received variable values from ps")
      # Opening the socket and connecting to server
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.connect((TCP_IP, port))
      while not mon_sess.should_stop():
        gradients, step_val = mon_sess.run([only_gradients,increment_global_step_op], feed_dict=feed_dict)
        #print("iteration: ", step_val)
        #print("Sending grads port: ", port)
        # sending the gradients
        send_data = pickle.dumps(gradients,pickle.HIGHEST_PROTOCOL)
        to_send_size = len(send_data)
        send_size = pickle.dumps(to_send_size, pickle.HIGHEST_PROTOCOL)
        #print("~~~~~~~~~~~Size of grads: {}, daat: {}".format(to_send_size, send_size))
        s.sendall(send_size)
        #print("~~~~~~~~~~~Size of size: ", len(send_size))
        s.sendall(send_data)
        #print("~~~~~~~~~~~sent grads")
        #receiving the variable values
        recv_size = safe_recv(17, s)
        recv_size = pickle.loads(recv_size)
        recv_data = safe_recv(recv_size, s)
        var_vals = pickle.loads(recv_data)
        #print("recved grads")
        
        feed_dict = {}
        i=0
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
           feed_dict[v] = var_vals[i]
           i=i+1
      s.close()

def main(argv=None):  # pylint: disable=unused-argument
  global port
  global port_main
  global s
  if(len(sys.argv) != 3):
      print("<port> <worker-id> required")
      sys.exit()
  port = int(sys.argv[1]) + int(sys.argv[2])
  port_main = int(sys.argv[1])
  print("Connecting to port ", port)
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  total_start_time = time.time()
  train()
  print("--- %s seconds ---" % (time.time() - total_start_time))


if __name__ == '__main__':
  tf.app.run()
