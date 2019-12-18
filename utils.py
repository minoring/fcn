"""Utility functions for FNC"""
import os
import sys
import urllib
import tarfile
import zipfile
import imageio

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import scipy.io as sio


def get_model_data(model_dir, model_URL):
  """Get model data"""
  download_and_extract(model_dir, model_URL)
  filename = model_URL.split('/')[-1]
  filepath = os.path.join(model_dir, filename)
  if not os.path.exists(filepath):
    raise IOError('VGG model not found!')
  data = sio.loadmat(filepath)
  return data


def download_and_extract(dir_path, URL, is_tarfile=False, is_zipfile=False):
  """Download file on URL and extract all if it is tarfile or zipfile"""
  if not os.path.exists(dir_path):
    os.mkdir(dir_path)
  filename = URL.split('/')[-1]
  filepath = os.path.join(dir_path, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write(
          '\r>> Downloading %s %.1f%%' %
          (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(URL,
                                             filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    if is_tarfile:
      tarfile.open(filepath, 'r:gz').extractall(dir_path)
    elif is_zipfile:
      with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dir_path)


def save_image(image, save_dir, name, mean=None):
  if mean:
    image = unprocess_image(image, mean)
  imageio.imwrite(os.path.join(save_dir, name + '.png'), image)


def process_image(image, mean_pixel):
  return image - mean_pixel


def unprocess_image(image, mean_pixel):
  return image + mean_pixel


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
  else:
    return tf.compat.v1.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
  initial = tf.compat.v1.constant(0.0, shape=shape)
  if name is None:
    return tf.compat.v1.Variable(initial)
  else:
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
