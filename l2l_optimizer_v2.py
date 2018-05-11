from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os

import mock
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.ops import variables
from utils import get_created_variables
from utils import make_with_custom_variables
from tensorflow.python.summary import summary

class L2LOptimizer(optimizer.Optimizer):
  """Learning to learn (meta) optimizer.

  Optimizer which has an internal RNN which takes as input, at each iteration,
  the gradient of the function being minimized and returns a step direction.
  This optimizer can then itself be optimized to learn optimization on a set of
  tasks.
  """

  def __init__(self, internal_optimizer, loss_func, lstm_units=20, train_opt=True, opt_last=False,
               dynamic_unroll=False, delta_ratio=1.0, update_ratio=1.0, co_opt=True, rnn_layer_cnt=1,
               corr_smooth=0.999, optimizer_ckpt=None, preprocessor = None, activation=tf.nn.tanh, name="L2L"):
    super(L2LOptimizer, self).__init__(False, name)
    self._internal_optimizer = internal_optimizer
    self._loss_func = loss_func
    self._original_vars = None
    self._opt_vars = None
    self._lstm_units = lstm_units
    with tf.variable_scope('', reuse=True):
      self._original_vars, constants = get_created_variables(loss_func)

    self._slot_map = {}
    self._cells = [tf.contrib.rnn.BasicLSTMCell(lstm_units, state_is_tuple=False, activation=activation) for i in range(rnn_layer_cnt)]
    self._cell = tf.contrib.rnn.MultiRNNCell(self._cells, state_is_tuple=False)
    self._omitted_items = set()
    self._reuse_var = None
    self._train_opt = train_opt
    self._opt_last = opt_last
    self._dynamic_unroll = dynamic_unroll
    self._update_ratio = update_ratio
    self._delta_ratio = delta_ratio
    self._optimizer_vars = []
    self._co_opt = co_opt
    self._corr_smooth = corr_smooth
    self._rnn_layer_cnt = rnn_layer_cnt
    self._optimizer_ckpt = optimizer_ckpt
    self._preprocessor = preprocessor

  def _create_slot(self):
    i = 0

    print('total variables in graph:')
    print(self._original_vars)

    print('total variables after partition')
    self._partitioned_original_vars = []
    self._var_partition_map = {}
    self._var_idx_map = {}
    for i in range(len(self._original_vars)):
      v = self._original_vars[i]
      if isinstance(v, variables.PartitionedVariable):
        self._partitioned_original_vars.extend(list(v))
        self._var_partition_map[v] = list(v)
        for w in list(v):
          self._var_idx_map[w] = i
      else:
        self._partitioned_original_vars.append(v)
        self._var_idx_map[v] = i
    print(self._partitioned_original_vars)

    opt_vars = []
    slot_vars = []
    for v in self._partitioned_original_vars:
      if (self._opt_vars is not None) and (v not in self._opt_vars):
        self._omitted_items.add( self._var_idx_map[v] )
        continue
      else:
        opt_vars.append(v)

      with ops.colocate_with(v):
        shape = v.get_shape().as_list()
        shape.append(self._lstm_units * 2 * self._rnn_layer_cnt)
        init = tf.constant(0.0, shape = tensor_shape.as_shape(shape), dtype=tf.float32)
        slot = tf.Variable(init, trainable=False)

      #slot = self._get_or_make_slot_with_initializer(v, init, tensor_shape.as_shape(shape), dtype,
      #                                               "state", self._name)

      self._slot_map[v] = slot
      slot_vars.append(slot)

    self._opt_vars = opt_vars #list(self._slot_map.keys())
    self._slot_vars = slot_vars
    print('omitted position:')
    print(self._omitted_items)
    print('variables to be optimized by L2L:')
    print(self._opt_vars)

  def _get_prediction(self, inputs, states):
    input_shape = inputs.get_shape()
    state_shape = states.get_shape()

    inputs = tf.reshape(inputs, [-1, 1])
    #inputs = tf.minimum(inputs, 0.01)
    #inputs = tf.maximum(inputs, -0.01)
    if self._preprocessor is not None:
      inputs = self._preprocessor(inputs)
    states = tf.reshape(states, [-1, 2*self._lstm_units*self._rnn_layer_cnt])

    #states = tuple(tf.split(states, num_or_size_splits=self._rnn_layers, axis=1))

    #with tf.variable_scope(self._name, initializer=tf.contrib.layers.xavier_initializer(uniform=True), reuse=self._reuse_var) as curr_scope:
    with tf.variable_scope(self._name, initializer=tf.truncated_normal_initializer(stddev=1e-3), reuse=self._reuse_var) as curr_scope:
      cell_outputs, new_states = self._cell(inputs, states)
      weights = tf.get_variable(
          'out_weights', [self._lstm_units, 1])

      res = tf.matmul(cell_outputs, weights)

      biases = tf.get_variable('out_bias', [1], initializer=tf.constant_initializer(0.0))
      res = tf.nn.bias_add(res, biases)

      res = tf.reshape(res, input_shape)
      new_states = tf.reshape(new_states, state_shape)

      if self._reuse_var is None:
        self._optimizer_vars = curr_scope.trainable_variables()

        print('optimizer variables:')
        print(self._optimizer_vars)

        if self._optimizer_ckpt is not None:
          scope_name = curr_scope.name + '/'
          print('load variables in (%s) from %s' % (scope_name, self._optimizer_ckpt))
          tf.contrib.framework.init_from_checkpoint(self._optimizer_ckpt, {scope_name: scope_name})

      if self._reuse_var is None:
        self._reuse_var = True

      return res, new_states

  def _get_updated_vars(self, loss):

    gradients = tf.gradients(loss, self._opt_vars)

    gradient_map = {}
    for i in range(len(gradients)):
      gradient_map[self._opt_vars[i]] = tf.stop_gradient(gradients[i])

    states_assign = []
    vars_assign = []
    updated_vars = []
    for i in range(len(self._original_vars)):
      var = self._original_vars[i]
      if i in self._omitted_items:
        updated_vars.append(tf.identity(var))
      else:
        v_list = []
        if isinstance(var, variables.PartitionedVariable):
          v_list = list(var)
        else:
          v_list = [var]

        new_var_list = []
        for v in v_list:
          with ops.colocate_with(v):
            g = gradient_map[v]
            grad_dot = tf.sqrt(tf.reduce_sum(g * g))
            delta, state = self._get_prediction(gradient_map[v], self._slot_map[v])
            delta_dot = tf.sqrt(tf.reduce_sum(delta * delta))

            denominator = grad_dot * delta_dot
            correlation = tf.cond(denominator > 0,
                            lambda: tf.reduce_sum(g * delta) / denominator,
                            lambda: ops.convert_to_tensor(0.0))

            correlation_var = tf.Variable(0.0, trainable=False)
            smoothed_correlation = correlation_var * self._corr_smooth + correlation * (1 - self._corr_smooth)
            corr_assign = tf.assign(correlation_var, smoothed_correlation, use_locking=True)
            states_assign.append(corr_assign)
            summary.histogram(v.name, v)
            summary.histogram(v.name+"_gradient", g)
            summary.histogram(v.name+"_delta", delta)

            summary.scalar(v.name+"_Gradient/dir correlation", correlation)
            summary.scalar(v.name+"_Gradient/dir smoothed correlation", smoothed_correlation)
            summary.scalar(v.name+"_grad_dot", grad_dot)
            summary.scalar(v.name+"_delta_dot", delta_dot)
            ratio = tf.cond(grad_dot > 0,
                            lambda: delta_dot/grad_dot,
                            lambda: ops.convert_to_tensor(0.0))
            summary.scalar(v.name+"_delta_grad_ratio", ratio)
            delta = tf.cond(ratio > self._delta_ratio,
                            lambda: delta * self._delta_ratio / ratio,
                            lambda: delta)
            summary.histogram(v.name+"_final_delta", delta)
            new_var_list.append(delta + v)
            state_update_op = tf.assign(self._slot_map[v], state, use_locking=True)
            var_update_op = tf.assign_add(v, delta * self._update_ratio, use_locking=True)
            vars_assign.append(var_update_op)
            states_assign.append(state_update_op)
        if len(new_var_list) == 1:
          updated_vars.append(new_var_list[0])
        else:
          axis = var._partition_axes()[0]
          updated_vars.append(tf.identity(tf.concat(new_var_list, axis = axis)))

    return updated_vars, states_assign, vars_assign

  def _simple_update(self, loss):
    updated_vars, states_assign, vars_assign = self._get_updated_vars(loss)
    new_loss = make_with_custom_variables(self._loss_func, updated_vars)
    optimizer = self._internal_optimizer

    if not self._opt_last:
      new_loss = new_loss + loss

    var_list = self._optimizer_vars
    if self._co_opt:
      var_list = self._opt_vars + self._optimizer_vars

    step = optimizer.minimize(new_loss, var_list=var_list)
    if self._train_opt:
      states_assign.append(step)
    states_assign.extend(vars_assign)
    update_ops = states_assign
    return update_ops

  def _get_update_ops(self, loss, unroll_len):
      if unroll_len <= 1:
        return self._simple_update(loss)
      else:
        assert('unroll len >1 unsupported')


  def get_opt_var(self):
    return self._opt_vars

  def get_slot_var(self):
    return self._slot_vars

  def minimize(self, loss, unroll_len=1, var_list=None, global_step=None):
    """Returns an operator minimizing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      learning_rate: Learning rate for the Adam optimizer.
      **kwargs: keyword arguments forwarded to meta_loss.

    Returns:
      namedtuple containing (step, update, reset, fx, x)
    """
    self._opt_vars = var_list

    if var_list is None:
      self._opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)


    self._create_slot()

    # just create the variables of optimizer, no use this sub-graph
    self._get_prediction(self._opt_vars[0], self._slot_map[self._opt_vars[0]])


    update_ops = self._get_update_ops(loss, unroll_len)


    if global_step is None:
      apply_updates = tf.group(*update_ops)
    else:
      with tf.control_dependencies(update_ops):
        reset_ops = []
        with ops.colocate_with(global_step):
          global_step_update = tf.assign_add(global_step, 1, use_locking=True).op
        reset_ops.append(global_step_update)
        apply_updates = tf.group(*reset_ops)

    return apply_updates

