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
from preprocess2 import LogAndSign
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
               corr_smooth=0.999, optimizer_ckpt=None, name="L2L"):
    super(L2LOptimizer, self).__init__(False, name)
    self._internal_optimizer = internal_optimizer
    self._loss_func = loss_func
    self._original_vars = None
    self._opt_vars = None
    self._lstm_units = lstm_units
    with tf.variable_scope('', reuse=True):
      self._original_vars, constants = get_created_variables(loss_func)

    self._slot_map = {}
    self._cells = [tf.contrib.rnn.BasicLSTMCell(lstm_units, state_is_tuple=False, activation=tf.nn.relu, name='lstm_optimizer_layer_%d' % (i)) for i in range(rnn_layer_cnt)]
    self._cell = tf.contrib.rnn.MultiRNNCell(self._cells, state_is_tuple=False)
    self._preprocess = LogAndSign(10)
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

  def _create_slot(self):
    i = 0

    print('total variables in graph:')
    print(self._original_vars)

    opt_vars = []
    for i in range(len(self._original_vars)):
      v = self._original_vars[i]
      if isinstance(v, variables.PartitionedVariable) or ((self._opt_vars is not None) and (v not in self._opt_vars)):
        self._omitted_items.add(i)
        continue
      else:
        opt_vars.append(v)

      with ops.colocate_with(v):
        dtype = v.dtype.base_dtype
        init = init_ops.constant_initializer(0.0,
                                             dtype=dtype)

      shape = v.get_shape().as_list()
      shape.append(self._lstm_units * 2 * self._rnn_layer_cnt)

      slot = self._get_or_make_slot_with_initializer(v, init, tensor_shape.as_shape(shape), dtype,
                                                     "state", self._name)
      self._slot_map[v] = slot

    self._opt_vars = opt_vars #list(self._slot_map.keys())
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
    inputs = self._preprocess(inputs)
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
    for v in self._original_vars:
      if v not in gradient_map:
        updated_vars.append(v)
      else:
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
          updated_vars.append(delta + v)
          #updated_vars.append(delta + tf.stop_gradient(v))
          state_update_op = tf.assign(self._slot_map[v], state, use_locking=True)
          var_update_op = tf.assign_add(v, delta * self._update_ratio, use_locking=True)
          vars_assign.append(var_update_op)
          states_assign.append(state_update_op)

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
    corr_var_updates = []
    def _update(fx, x, state):
      """Parameter and RNN state update."""
      with tf.name_scope("gradients"):
        gradients = tf.gradients(fx, x)
        gradients = [tf.stop_gradient(g) for g in gradients]
        #gradients = [tf.stop_gradient(g) if g is not None else None for g in gradients]

      with tf.name_scope("deltas"):
        #deltas, state_next = zip(*[self._get_prediction(g, s) for g, s in zip(gradients, state)])
        deltas = []
        state_next = []
        for g, s, v in zip(gradients, state, x):
          with ops.colocate_with(s):
            output, state = self._get_prediction(g, s)

            delta_dot = tf.sqrt(tf.reduce_sum(output * output))
            grad_dot = tf.sqrt(tf.reduce_sum(g * g))

            ratio = tf.cond(grad_dot > 0,
                            lambda: delta_dot/grad_dot,
                            lambda: ops.convert_to_tensor(0.0))

            final_output = tf.cond(ratio > self._delta_ratio,
                             lambda: output * self._delta_ratio / ratio,
                             lambda: output)

            deltas.append(final_output)
            state_next.append(state)

            if not self._dynamic_unroll:
              denominator = grad_dot * delta_dot
              correlation = tf.cond(denominator > 0,
                                    lambda: tf.reduce_sum(g * output) / denominator,
                                    lambda: ops.convert_to_tensor(0.0))

              correlation_var = tf.Variable(0.0, trainable=False)
              smoothed_correlation = correlation_var * self._corr_smooth + correlation * (1 - self._corr_smooth)
              corr_assign = tf.assign(correlation_var, smoothed_correlation, use_locking=True)
              corr_var_updates.append(corr_assign)
              summary.scalar(v.name+"_Gradient/dir correlation", correlation)
              summary.scalar(v.name+"_Gradient/dir smoothed correlation", smoothed_correlation)
              summary.scalar(v.name+"_grad_dot", grad_dot)
              summary.scalar(v.name+"_delta_dot", delta_dot)
              summary.scalar(v.name+"_delta_grad_ratio", ratio)


        state_next = list(state_next)

      return deltas, state_next

    def _step(t, fx_array, fx, x, state):
      x_next = []
      with tf.name_scope("fx"):
        fx_array = fx_array.write(t, fx)

      with tf.name_scope("dx"):
        deltas, state_next = _update(fx, x, state)

        for j in range(len(deltas)):
          with ops.colocate_with(x[j]):
            value = x[j] + deltas[j] * self._delta_ratio
            x_next.append(value)

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
          with ops.colocate_with(x[j]):
            value = x[j] + deltas[j] * self._delta_ratio
            x_next.append(value)

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
    optimizer = self._internal_optimizer

    var_list = self._optimizer_vars
    if self._co_opt:
      var_list = self._opt_vars + self._optimizer_vars

    step = optimizer.minimize(loss_final, var_list=var_list)

    if self._train_opt:
      update_ops.append(step)

    var_len = len(x_final)

    for i in range(var_len):
      var = x[i]
      updated_var = x_final[i]

      state = initial_states[i]
      updated_state = s_final[i]

      update_ops.append(tf.assign_add(var, (updated_var-var) * self._update_ratio, use_locking=True))
      update_ops.append(tf.assign(state, updated_state, use_locking=True))

    return update_ops + corr_var_updates

  def _get_update_ops(self, loss, unroll_len):
      if unroll_len <= 1:
        return self._simple_update(loss)
      else:
        return self._rnn_update(loss, unroll_len)

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
        with ops.colocate_with(global_step):
          apply_updates = tf.assign_add(global_step, 1, use_locking=True).op
    return apply_updates

