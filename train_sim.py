# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")
flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 100, "Log period.")
flags.DEFINE_integer("evaluation_period", 1000, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_string("problem", "mnist", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 5, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")
flags.DEFINE_integer("reset_interval", 5, "Number of training epochs.")

def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

  if FLAGS.save_path is not None:
    if os.path.exists(FLAGS.save_path):
      raise ValueError("Folder {} already exists".format(FLAGS.save_path))
    else:
      os.mkdir(FLAGS.save_path)

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem)

  # Optimizer setup.
  optimizer = meta.MetaOptimizer(**net_config)
  minimize = optimizer.meta_minimize(
      problem, FLAGS.unroll_length,
      learning_rate=FLAGS.learning_rate,
      net_assignments=net_assignments,
      second_derivatives=FLAGS.second_derivatives)
  step, update, reset, cost_op, _ = minimize
  no_op = tf.no_op()
  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    trainable_vars = tf.trainable_variables()
    print('trainable variables')
    for v in trainable_vars:
        print("parameter:", v.name, "device:", v.device, "shape:", v.get_shape())

    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0
    curr_step = 0
    for e in xrange(FLAGS.num_epochs):
        #time, curr_loss = util.run_epoch(sess, cost_op, [update, step], no_op, 1)
        curr_loss = sess.run([cost_op, update, step])[0]
        total_cost += curr_loss
        curr_step += 1
        if curr_step % 100 ==0:
            print('step:%d,loss:%f' % (curr_step,total_cost/100))
            total_cost = 0

        if curr_step % FLAGS.reset_interval == 0:
            print('reset states')
            sess.run(reset)



if __name__ == "__main__":
  tf.app.run()
