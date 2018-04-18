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
from preprocess import LogAndSign

class L2LOptimizer(optimizer.Optimizer):
  """Learning to learn (meta) optimizer.

  Optimizer which has an internal RNN which takes as input, at each iteration,
  the gradient of the function being minimized and returns a step direction.
  This optimizer can then itself be optimized to learn optimization on a set of
  tasks.
  """

  def __init__(self, adam_lr, loss_func, opt_vars=None, lstm_units=20, train_opt=True, opt_last=False, dynamic_unroll=True, name="L2L"):
    super(L2LOptimizer, self).__init__(False, name)
    self._adam_lr = adam_lr
    self._loss_func = loss_func
    self._original_vars = None
    self._opt_vars = opt_vars
    self._lstm_units = lstm_units
    with tf.variable_scope('', reuse=True):
      self._original_vars, constants = get_created_variables(loss_func)

    self._slot_map = {}
    self._create_slot()
    self._cell = tf.contrib.rnn.BasicLSTMCell(lstm_units, state_is_tuple=False)
    self._learning_rate = adam_lr
    self._preprocess = LogAndSign(10)
    self._omitted_items = set()
    self._reuse_var = None
    self._train_opt = train_opt
    self._opt_last = opt_last
    self._dynamic_unroll = dynamic_unroll

  def _create_slot(self):
    i = 0

    print('total variables in graph:')
    print(self._original_vars)

    for v in self._original_vars:
      if isinstance(v, variables.PartitionedVariable) or ((self._opt_vars is not None) and (v not in self._opt_vars)):
        self._omitted_items.add(i)
        continue

      with ops.colocate_with(v):
        dtype = v.dtype.base_dtype
        init = init_ops.constant_initializer(0.0,
                                             dtype=dtype)

      shape = v.get_shape().as_list()
      shape.append(self._lstm_units * 2)

      slot = self._get_or_make_slot_with_initializer(v, init, tensor_shape.as_shape(shape), dtype,
                                                     "state", self._name)
      self._slot_map[v] = slot
      i = i + 1

    self._opt_vars = list(self._slot_map.keys())
    print('variables to be optimized by L2L:')
    print(self._opt_vars)

  def _get_prediction(self, inputs, states):
    input_shape = inputs.get_shape()
    state_shape = states.get_shape()

    inputs = tf.reshape(inputs, [-1, 1])
    #inputs = tf.minimum(inputs, 0.01)
    #inputs = tf.maximum(inputs, -0.01)
    inputs = self._preprocess(inputs)
    states = tf.reshape(states, [-1, 2*self._lstm_units])

    with tf.variable_scope('l2l_weight', initializer=tf.random_normal_initializer(stddev=0.001), reuse=self._reuse_var):
      cell_outputs, new_states = self._cell(inputs, states)
      weights = tf.get_variable(
          'out_weights', [self._lstm_units, 1])

      res = tf.matmul(cell_outputs, weights)

      biases = tf.get_variable('out_bias', [1], initializer=tf.constant_initializer(0.0))
      res = tf.nn.bias_add(res, biases)

      res = tf.reshape(res, input_shape)
      new_states = tf.reshape(new_states, state_shape)

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
    for v in self._original_vars:
      if v not in gradient_map:
        updated_vars.append(v)
      else:
        delta, state = self._get_prediction(gradient_map[v], self._slot_map[v])
        updated_vars.append(delta + v)
        #updated_vars.append(delta + tf.stop_gradient(v))
        state_update_op = tf.assign(self._slot_map[v], state)
        var_update_op = tf.assign_add(v, delta)
        vars_assign.append(var_update_op)
        states_assign.append(state_update_op)

    return updated_vars, states_assign, vars_assign

  def _simple_update(self, loss):
    updated_vars, states_assign, vars_assign = self._get_updated_vars(loss)
    new_loss = make_with_custom_variables(self._loss_func, updated_vars)
    optimizer = tf.train.AdamOptimizer(self._learning_rate)

    if not self._opt_last:
      new_loss = new_loss + loss

    step = optimizer.minimize(new_loss)#, var_list=self._opt_vars)
    if self._train_opt:
      states_assign.append(step)
    states_assign.extend(vars_assign)
    update_ops = states_assign
    return update_ops

  def _gen_curr_vars(self, x):
    curr_vars = []

    j = 0
    for i in range(len(self._original_vars)):
      if i in self._omitted_items:
        curr_vars.append(self._original_vars[i])
      else:
        curr_vars.append(x[j])
        j = j+1
    return curr_vars

  def _rnn_update(self, loss, unroll_len):
    x = []
    for i in range(len(self._original_vars)):
      if i in self._omitted_items:
        continue
      x.append(self._original_vars[i])

    initial_states = [self._slot_map[v] for v in x]

    def _update(fx, x, state):
      """Parameter and RNN state update."""
      with tf.name_scope("gradients"):
        gradients = tf.gradients(fx, x)
        gradients = [tf.stop_gradient(g) for g in gradients]
        #gradients = [tf.stop_gradient(g) if g is not None else None for g in gradients]

      with tf.name_scope("deltas"):
        deltas, state_next = zip(*[self._get_prediction(g, s) for g, s in zip(gradients, state)])
        state_next = list(state_next)

      return deltas, state_next

    def _step(t, fx_array, fx, x, state):
      x_next = []
      with tf.name_scope("fx"):
        fx_array = fx_array.write(t, fx)

      with tf.name_scope("dx"):
        deltas, state_next = _update(fx, x, state)

        for j in range(len(deltas)):
          x_next.append(x[j] + deltas[j])

      curr_vars = self._gen_curr_vars(x_next)
      fx_next = make_with_custom_variables(self._loss_func, curr_vars)
      with tf.name_scope("t_next"):
        t_next = t + 1

      return t_next, fx_array, fx_next, x_next, state_next

    def _dynamic_step(t, fx_array, x, state):
      x_next = []

      curr_vars = self._gen_curr_vars(x)
      fx = make_with_custom_variables(self._loss_func, curr_vars)

      with tf.name_scope("fx"):
        fx_array = fx_array.write(t, fx)

      with tf.name_scope("dx"):
        deltas, state_next = _update(fx, x, state)

        for j in range(len(deltas)):
          x_next.append(x[j] + deltas[j])

      with tf.name_scope("t_next"):
        t_next = t + 1

      return t_next, fx_array, x_next, state_next

    fx_array = tf.TensorArray(tf.float32, size=unroll_len + 1,
                              clear_after_read=False)

    if not self._dynamic_unroll:
      next_x = x
      next_states = initial_states
      t = 0
      fx_next = loss
      for i in range(unroll_len):
        t, fx_array, fx_next, next_x, next_states = _step(t, fx_array, fx_next, next_x, next_states)

      x_final = next_x
      s_final = next_states
      loss_final = fx_next
    else:
      _, fx_array, x_final, s_final = tf.while_loop(
        cond=lambda t, *_: t < unroll_len,
        body=_dynamic_step,
        loop_vars=(0, fx_array, x, initial_states),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")

      curr_vars = self._gen_curr_vars(x_final)
      loss_final = make_with_custom_variables(self._loss_func, curr_vars)

    with tf.name_scope("fx"):
      fx_array = fx_array.write(unroll_len, loss_final)

    if not self._opt_last:
      loss_final = tf.reduce_sum(fx_array.stack(), name="loss")

    update_ops = []
    optimizer = tf.train.AdamOptimizer(self._learning_rate)
    step = optimizer.minimize(loss_final)
    if self._train_opt:
      update_ops.append(step)

    var_len = len(x_final)

    for i in range(var_len):
      var = x[i]
      updated_var = x_final[i]

      state = initial_states[i]
      updated_state = s_final[i]

      update_ops.append(tf.assign_add(var, updated_var-var))
      update_ops.append(tf.assign(state, updated_state))

    return update_ops

  def _get_update_ops(self, loss, unroll_len):
      if unroll_len <= 1:
        return self._simple_update(loss)
      else:
        return self._rnn_update(loss, unroll_len)

  def minimize(self, loss, unroll_len=1, global_step=None):
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
    update_ops = self._get_update_ops(loss, unroll_len)

    if global_step is None:
      apply_updates = tf.group(*update_ops)
    else:
      with tf.control_dependencies(update_ops):
        with tf.colocate_with(global_step):
          apply_updates = tf.assign_add(global_step, 1).op
    return apply_updates

