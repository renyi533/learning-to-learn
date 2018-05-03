from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import l2l_optimizer as l2l_optimizer
import utils
import util
from preprocess2 import LogAndSign
flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
flags.DEFINE_integer("mode", 0, "trainer selection")
flags.DEFINE_integer("layer", 2, "trainer selection")
flags.DEFINE_integer("unroll_len", 20, "trainer selection")
flags.DEFINE_integer("reset_interval", 5, "trainer selection")
flags.DEFINE_boolean("opt_last", False, "trainer selection")
flags.DEFINE_boolean("co_opt", False, "trainer selection")
flags.DEFINE_boolean("dynamic_unroll", True, "trainer selection")

flags.DEFINE_string("problem", "mnist", "Type of problem.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float("delta_ratio", 1000.0, "Learning rate.")
flags.DEFINE_float("update_ratio", 1.0, "Learning rate.")


def main(_):
  # Configuration.

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem)

  loss = problem()
  global_step = tf.Variable(0, dtype=tf.int64)
  # Optimizer setup.

  adam_opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
  opt2 = adam_opt.minimize(loss, global_step)

  adagrad_opt = tf.train.AdagradOptimizer(FLAGS.learning_rate)
  optimizer = l2l_optimizer.L2LOptimizer(internal_optimizer=adam_opt, loss_func=problem, opt_last=FLAGS.opt_last, preprocessor=LogAndSign(10),
          co_opt=FLAGS.co_opt, rnn_layer_cnt=FLAGS.layer, delta_ratio=FLAGS.delta_ratio, update_ratio=FLAGS.update_ratio,
          dynamic_unroll=FLAGS.dynamic_unroll)

  opt = optimizer.minimize(loss, global_step = global_step, unroll_len=FLAGS.unroll_len)
  if FLAGS.mode == 1:
    print('use adam opt')
    opt = opt2
  else:
    print('use l2l opt')
  slot_reset = tf.variables_initializer(optimizer._slot_vars+optimizer._opt_vars)
  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    print('trainable variables')
    trainable_vars = tf.trainable_variables()
    for v in trainable_vars:
        print("parameter:", v.name, "device:", v.device, "shape:", v.get_shape())

    best_evaluation = float("inf")
    total_time = 0
    accum_loss = 0.0
    total_cost = 0
    for e in xrange(FLAGS.num_epochs):
      # Training.
      step, curr_loss, _ = sess.run([global_step, loss, opt])
      accum_loss += curr_loss
      if step % 100 == 0:
        print('step:%d,loss:%f' % (step, accum_loss/100))
        accum_loss = 0

      if step % FLAGS.reset_interval == 0:
        #print('reset')
        sess.run(slot_reset)


if __name__ == "__main__":
  tf.app.run()
