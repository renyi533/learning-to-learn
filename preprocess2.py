from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Preprocessor:
  def __init__(self, name):
    self._name = name

class Clamp(Preprocessor):

  def __init__(self, min_value=None, max_value=None, name="Clamp"):
    super(Clamp, self).__init__(name=name)
    self._min = min_value
    self._max = max_value

  def __call__(self, inputs):
    output = inputs
    if self._min is not None:
      output = tf.maximum(output, self._min)
    if self._max is not None:
      output = tf.minimum(output, self._max)
    return output


class LogAndSign(Preprocessor):
  """Log and sign preprocessing.

  As described in https://arxiv.org/pdf/1606.04474v1.pdf (Appendix A).
  """

  def __init__(self, k, name="LogAndSign"):
    super(LogAndSign, self).__init__(name=name)
    self._k = k

  def __call__(self, gradients):
    """Connects the LogAndSign module into the graph.

    Args:
      gradients: `Tensor` of gradients with shape `[d_1, ..., d_n]`.

    Returns:
      `Tensor` with shape `[d_1, ..., d_n-1, 2 * d_n]`. The first `d_n` elements
      along the nth dimension correspond to the log output and the remaining
      `d_n` elements to the sign output.
    """
    eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
    ndims = gradients.get_shape().ndims

    log = tf.log(tf.abs(gradients) + eps)
    clamped_log = Clamp(min_value=-1.0)(log / self._k)  # pylint: disable=not-callable
    sign = Clamp(min_value=-1.0, max_value=1.0)(gradients * np.exp(self._k))  # pylint: disable=not-callable

    return tf.concat([clamped_log, sign], ndims - 1)

