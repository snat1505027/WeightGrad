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

TCP_IP = '127.0.0.1'
TCP_PORT = 5014

port_ps1 = 0
port_ps2 = 0
port_main_1 = 0
port_main_2 = 0
s = 0
half_index = 5

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
def safe_recv(size, server_socket):
  data = ''
  temp = ''
  recv_size = 0
  while 1:
    try:
        temp = server_socket.recv(size-len(data))
        data += temp
        recv_size = len(data)
        if recv_size >= size:
            break
    except:
        print("Error")
  return data

def train():
  """Train CIFAR-10 for a number of steps."""

  g1 = tf.Graph()
  with g1.as_default():
    #global_step = tf.contrib.framework.get_or_create_global_step()
    
    global_step = tf.Variable(-1, name='global_step', trainable=False, dtype=tf.int32)
    increment_global_step_op = tf.assign(global_step, global_step+1)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

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
      # Getting first set of variables
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.connect((TCP_IP, port_main_1))
      recv_size = safe_recv(8, s)
      recv_size = pickle.loads(recv_size)
      recv_data = safe_recv(recv_size, s)
      var_vals_1 = pickle.loads(recv_data)
      s.close()
      # Getting second set of variables
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.connect((TCP_IP, port_main_2))
      recv_size = safe_recv(8, s)
      recv_size = pickle.loads(recv_size)
      recv_data = safe_recv(recv_size, s)
      var_vals_2 = pickle.loads(recv_data)
      s.close()

      feed_dict = {}
      i=0
      for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if(i < half_index):
            feed_dict[v] = var_vals_1[i]
        else:
            feed_dict[v] = var_vals_2[i-half_index]
        i=i+1
      print("Received variable values from ps")
      s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s1.connect((TCP_IP, port_ps1))
      s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s2.connect((TCP_IP, port_ps2))
      print("Connected to both PSs")
      while not mon_sess.should_stop():
        gradients, step_val = mon_sess.run([only_gradients,increment_global_step_op], feed_dict=feed_dict)
        #print("Sending grads port: ", port)
        # Opening the socket and connecting to server
        # sending the gradients
        grad_part1 = []
        grad_part2 = []
        i=0
        for g in gradients:
            if(i < half_index):
                grad_part1.append(g)
            else:
                grad_part2.append(g)
            i=i+1

        send_data_1 = pickle.dumps(grad_part1,pickle.HIGHEST_PROTOCOL)
        to_send_size_1 = len(send_data_1)
        send_size_1 = pickle.dumps(to_send_size_1, pickle.HIGHEST_PROTOCOL)
        s1.sendall(send_size_1)
        s1.sendall(send_data_1)

        send_data_2 = pickle.dumps(grad_part2,pickle.HIGHEST_PROTOCOL)
        to_send_size_2 = len(send_data_2)
        send_size_2 = pickle.dumps(to_send_size_2, pickle.HIGHEST_PROTOCOL)
        s2.sendall(send_size_2)
        s2.sendall(send_data_2)
        #print("sent grads")
        #receiving the variable values
        recv_size = safe_recv(8, s1)
        recv_size = pickle.loads(recv_size)
        recv_data = safe_recv(recv_size, s1)
        var_vals_1 = pickle.loads(recv_data)

        recv_size = safe_recv(8, s2)
        recv_size = pickle.loads(recv_size)
        recv_data = safe_recv(recv_size, s2)
        var_vals_2 = pickle.loads(recv_data)
        #print("recved grads")
        
        feed_dict = {}
        i=0
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if(i < half_index):
                feed_dict[v] = var_vals_1[i]
            else:
                feed_dict[v] = var_vals_2[i-half_index]
            i=i+1
      
      s1.close()
      s2.close()

def main(argv=None):  # pylint: disable=unused-argument
  global port_ps1
  global port_ps2
  global port_main_1
  global port_main_2
  if(len(sys.argv) != 4):
      print("<port ps 1> <port ps 2> <worker-id> required")
      sys.exit()
  port_ps1 = int(sys.argv[1]) + int(sys.argv[3])
  port_ps2 = int(sys.argv[2]) + int(sys.argv[3])
  port_main_1 = int(sys.argv[1])
  port_main_2 = int(sys.argv[2])
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  total_start_time = time.time()
  train()
  print("--- %s seconds ---" % (time.time() - total_start_time))


if __name__ == '__main__':
  tf.app.run()
