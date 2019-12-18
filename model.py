"""Define model FCN for Semantic Segmentation"""
import tensorflow as tf
import numpy as np

from tf_utils import get_variable
from tf_utils import conv2d
from tf_utils import add_activation_summary
from tf_utils import avg_pool_2x2
from tf_utils import max_pool_2x2
from tf_utils import weight_variable
from tf_utils import bias_variable
from tf_utils import conv2d_transpose_strided

tf.compat.v1.disable_eager_execution()
NUM_CLASSES = 151


def vgg_net(weights, x, debug=False):
  layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
            'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
            'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4')

  vgg = {}
  for i, name in enumerate(layers):
    layer = name[:4]
    if layer == 'conv':
      kernels, bias = weights[i][0][0][0][0]
      # matconvnet: weights are [width, height, in_channels, out_channels]
      # tensorflow: weights are [height, width, in_channels, out_channels]
      kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)),
                             name=name + "_w")
      bias = get_variable(bias.reshape(-1), name=name + "_b")
      x = conv2d(x, kernels, bias)
    elif layer == 'relu':
      x = tf.nn.relu(x, name=name)
      if debug:
        add_activation_summary(x)
    elif layer == 'pool':
      x = avg_pool_2x2(x)
    vgg[name] = x

  return vgg


def fcn(weights, x, keep_prob, debug=False):
  """Define Fully Convolutional Networks

  Args:
    x (tf.Tensor): input image shape of (None, IMG_SIZE, IMAGE_SIZE, 3).
    keep_prob(tf.float32): Keep prob for dropout
  Returns:
    Tuple of anotated prediction and logits.
  """
  with tf.compat.v1.variable_scope('inference'):
    vgg = vgg_net(weights, x)
    conv5_3 = vgg['conv5_3']
    pool5 = max_pool_2x2(conv5_3)

    W6 = weight_variable([7, 7, 512, 4096], name='W6')
    b6 = bias_variable([4096], name='b6')
    conv6 = conv2d(pool5, W6, b6)
    relu6 = tf.nn.relu(conv6, name='relu6')
    if debug:
      add_activation_summary(relu6)
    dropout6 = tf.nn.dropout(relu6, rate=(1.0 - keep_prob))

    W7 = weight_variable([1, 1, 4096, 4096], name='W7')
    b7 = bias_variable([4096], name='b7')
    conv7 = conv2d(dropout6, W7, b7)
    relu7 = tf.nn.relu(conv7, name='relu7')
    if debug:
      add_activation_summary(relu7)
    dropout7 = tf.nn.dropout(relu7, rate=(1.0 - keep_prob))

    W8 = weight_variable([1, 1, 4096, NUM_CLASSES], name='W8')
    b8 = bias_variable([NUM_CLASSES], name='b8')
    conv8 = conv2d(dropout7, W8, b8)
    # annotation_pred1 = tf.math.argmax(conv8, axis=3, name='prediction1')

    # Upsample to the image size
    conv_trans_shape1 = vgg['pool4'].get_shape()
    W_t1 = weight_variable([4, 4, conv_trans_shape1[3], NUM_CLASSES],
                           name='W_t1')
    b_t1 = bias_variable([conv_trans_shape1[3]], name='b_t1')
    conv_t1 = conv2d_transpose_strided(conv8,
                                       W_t1,
                                       b_t1,
                                       output_shape=tf.shape(vgg['pool4']))
    fuse_1 = tf.add(conv_t1, vgg['pool4'], name='fuse_1')

    conv_trans_shape2 = vgg['pool3'].get_shape()
    W_t2 = weight_variable(
        [4, 4, conv_trans_shape2[3], conv_trans_shape1[3]],
        name='W_t2')
    b_t2 = bias_variable([conv_trans_shape2[3]], name='b_t2')
    conv_t2 = conv2d_transpose_strided(fuse_1,
                                       W_t2,
                                       b_t2,
                                       output_shape=tf.shape(vgg['pool3']))
    fuse_2 = tf.add(conv_t2, vgg['pool3'], name='fuse_2')

    shape = tf.shape(x)
    conv_trans_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_CLASSES])
    W_t3 = weight_variable([16, 16, NUM_CLASSES, conv_trans_shape3[3]],
                           name='W_t3')
    b_t3 = bias_variable([NUM_CLASSES], name='b_t3')
    conv_t3 = conv2d_transpose_strided(fuse_2,
                                       W_t3,
                                       b_t3,
                                       output_shape=conv_trans_shape3,
                                       stride=8)
    annotation_pred = tf.argmax(conv_t3, axis=3, name='prediction')

  return tf.expand_dims(annotation_pred, axis=3), conv_t3
