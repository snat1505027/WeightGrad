from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import pickle as pickle
import socket
import multiprocessing
from multiprocessing import Process, Queue, Value, Manager
from ctypes import c_char_p 

from datetime import datetime
import time

import tensorflow as tf

import cifar10

#TCP_IP = '127.0.0.1'
TCP_IP = '0.0.0.0'
s = 0
MAX_WORKERS = 0
global_var_vals = None

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100002,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)

# def safe_recv(size, server_socket):
#   data = ''
#   temp = ''
#   recv_size = 0
#   while 1:
#     try:
#         temp = server_socket.recv(size-len(data))
#         data += temp
#         recv_size = len(data)
#         if recv_size >= size:
#             break
#     except:
#         print("Error")
#   return data


def safe_recv(size, server_socket):
  data = ""
  temp = ""
  data = bytearray()
  recv_size = 0
  while 1:
    try:
        #print("~~~~~~~~~INSIDE safe_recv")
        temp = server_socket.recv(size-len(data))
        #print(type(temp))
        #print("temp size:: ",size-len(data))
        #print("~~~~~~~~~INSIDE safe_recv:: {}".format(temp))
        #xs = bytearray(temp)
        data.extend(temp)
        #data += str(temp)
        
        #print("~~~~~~~~~INSIDE safe_recv data:: {}".format(data))
        
        recv_size = len(data)
        #print("~~~~~~~~~INSIDE safe_recv recv_size:: {}".format(recv_size))
        if recv_size >= size:
            break
        #print("~~~~~~~~~INSIDE safe_recv")
    except:
        #print("Unexpected error:", sys.exc_info()[0])
        print("Error")
  data = bytes(data)
  #print(type(data))
  return data

def handleWorker(port,gradients_q,done_flag,global_var_vals,ack_q,n):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting to port : ", port)
    s.bind((TCP_IP, port))
    s.listen(1)
    conn, addr = s.accept()
    print('Connection address:', addr)
    k=0
    while 1:
        size = safe_recv(17,conn)
        size = pickle.loads(size)
        data = safe_recv(size,conn)
        #print("Received size: ", size)
        local_worker_gradients = pickle.loads(data)
        gradients_q.put(local_worker_gradients)
        while(done_flag.value == 0):
            pass
        size = len(global_var_vals.value)
        size = pickle.dumps(size, pickle.HIGHEST_PROTOCOL)
        conn.sendall(size)
        conn.sendall(global_var_vals.value)
        ack_q.put(1)
        k=k+1
        #print("Worker: ", k)
        if(k==(n+1)):
            print("Working: Breaking from loop")
            break
    conn.close()
    s.close()


def train():
  """Train CIFAR-10 for a number of steps."""

  g1 = tf.Graph()
  with g1.as_default():
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #global_step = tf.contrib.framework.get_or_create_global_step()

    
    global_step = tf.Variable(-1, name='global_step', trainable=False, dtype=tf.int32)
    increment_global_step_op = tf.assign(global_step, global_step+1)
    
    cifar10.build_graph()
       
    placeholder_gradients = []

    #with tf.device("/gpu:0"):
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        placeholder_gradients.append((tf.placeholder('float', shape=var.get_shape()) ,var))
    feed_dict = {}
       
    i=0
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
       feed_dict[placeholder_gradients[i][0]] = np.zeros(placeholder_gradients[i][0].shape)
       i=i+1
    train_op = cifar10.train_part2(global_step,placeholder_gradients)
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d,(%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, 
              examples_per_sec, sec_per_batch))
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)) as mon_sess:

      # Sending the initial value of variables
      global global_var_vals
      global done_flag
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      print("Connecting to port : ", port, " and no of workers: ", MAX_WORKERS)
      s.bind((TCP_IP, port))
      s.listen(5)
      var_val = []
      for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        var_val.append(mon_sess.run(v, feed_dict=feed_dict))
      
      #print("Var val:: ",var_val)
      send_data = pickle.dumps(var_val, pickle.HIGHEST_PROTOCOL)
      #print("Send the send_data:: ",send_data)
      global_var_vals.value = send_data
      size = len(send_data)
      #print("Send the size:: ",size)
      size = pickle.dumps(size, pickle.HIGHEST_PROTOCOL)
      #print("Send the size in bytes:: ",size)
      for i in range(MAX_WORKERS):
        conn, addr = s.accept()
        #print("Conn: {}, Addr: {}".format(conn, addr))
        conn.sendall(size)
        #print("Sent Size")
        conn.sendall(send_data)
        conn.close()
      s.close()
      print("Sent initial var values to workers")

      while not mon_sess.should_stop():
        #print("Done with Sending")
        val = mon_sess.run(global_step, feed_dict=feed_dict)
        #print("Iteration: ", val)
        if(val == (FLAGS.max_steps - 1)):
            print("Global step val while stoping.")
            return
        #print("Before For")
        for i in range(MAX_WORKERS):
            recv_grads = gradients_q.get()
            feed_dict = {}
            for i,grad_var in enumerate(recv_grads): 
               feed_dict[placeholder_gradients[i][0]] = recv_grads[i]

            res = mon_sess.run(train_op, feed_dict=feed_dict)
        var_val = []
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            var_val.append(mon_sess.run(v, feed_dict=feed_dict))
        global_var_vals.value = pickle.dumps(var_val, pickle.HIGHEST_PROTOCOL)
        #print("New values of variables ready")
        done_flag.value = 1
        for i in range(MAX_WORKERS):
            val = ack_q.get()
        done_flag.value = 0
        #print("Done with Train")


def main(argv=None):  # pylint: disable=unused-argument
  if(len(sys.argv) != 3):
      print("Port number and no of workers required")
      sys.exit()
  global s
  global port
  global MAX_WORKERS
  global gradients_q
  global global_var_vals
  global ack_q
  global done_flag
  port = int(sys.argv[1])
  MAX_WORKERS = int(sys.argv[2])
  gradients_q = Queue()
  ack_q = Queue()
  manager = Manager()
  global_var_vals = manager.Value(c_char_p, "")
  done_flag = manager.Value('i', 0)
  n = int(FLAGS.max_steps/MAX_WORKERS)
  print("Each worker does ", n, " iterations")
  process_list = []
  for i in range(MAX_WORKERS):
        process_port = port + i + 1
        p = Process(target=handleWorker, args=(process_port,gradients_q,done_flag,global_var_vals, ack_q,n))
        p.start()
        process_list.append(p)

  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  total_start_time = time.time()
  train()
  print("--- %s seconds ---" % (time.time() - total_start_time))
  for p in process_list:
      p.join()

if __name__ == '__main__':
  tf.app.run()
