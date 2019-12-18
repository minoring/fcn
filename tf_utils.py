"""Utility functions for tensorflow"""
import tensorflow as tf

def get_variable(weights, name):
  init = tf.compat.v1.constant_initializer(weights, dtype=tf.float32)
  var = tf.compat.v1.get_variable(name=name,
                                  initializer=init,
                                  shape=weights.shape)
  return var


def weight_variable(shape, stddev=0.2, name=None):
  initial = tf.random.truncated_normal(shape, stddev=stddev)
  if name is None:
    return tf.Variable(initial)
  return tf.compat.v1.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
  initial = tf.compat.v1.constant(0.0, shape=shape)
  if name is None:
    return tf.compat.v1.Variable(initial)
  return tf.compat.v1.get_variable(name, initializer=initial)


def conv2d(x, W, bias):
  conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  return tf.nn.bias_add(conv, bias)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
  if output_shape is None:
    output_shape = x.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[2] *= 2
    output_shape[3] = W.get_shape().as_list()[2]
  conv = tf.nn.conv2d_transpose(x,
                                W,
                                output_shape,
                                strides=[1, stride, stride, 1],
                                padding='SAME')
  return tf.nn.bias_add(conv, b)


def avg_pool_2x2(x):
  return tf.nn.avg_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')


def add_to_regulariation_and_summary(var):
  if var is not None:
    tf.compat.v1.summary.histogram(var.op.name, var)
    tf.compat.v1.add_to_collection('reg_loss', tf.nn.l2_loss(var))


def add_activation_summary(var):
  if var is not None:
    tf.compat.v1.summary.histogram(var.op.name + '/activation', var)
    tf.compat.v1.summary.scalar(var.op.name + '/sparsity',
                                tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
  if grad is not None:
    tf.compat.v1.summary.histogram(var.op.name + '/gradient', grad)
