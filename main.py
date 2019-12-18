"""Run Fully Convolutional Networks"""
import datetime

import numpy as np
from absl import app
from absl import flags
import tensorflow as tf

from flags import define_flags
from model import fcn
from utils import get_model_data
from utils import process_image
from utils import save_image
from dataset import read_files
from dataset import BatchDataset
from tf_utils import add_gradient_summary
from tf_utils import add_to_regulariation_and_summary

tf.compat.v1.disable_eager_execution()
FLAGS = flags.FLAGS


def train(loss, var):
  optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate)
  grads = optimizer.compute_gradients(loss, var_list=var)
  if FLAGS.debug:
    for grad, var in grads:
      add_gradient_summary(grad, var)

  return optimizer.apply_gradients(grads)


def main(_):
  keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
  image = tf.compat.v1.placeholder(
      tf.float32,
      shape=[None, FLAGS.image_size, FLAGS.image_size, 3],
      name='input_image')
  annotation = tf.compat.v1.placeholder(
      tf.int32,
      shape=[None, FLAGS.image_size, FLAGS.image_size, 1],
      name='annotation')

  print('Setting up vgg initialized conv layers...')
  model_data = get_model_data(FLAGS.model_dir, FLAGS.model_URL)
  mean = model_data['normalization'][0][0][0]
  mean_pixel = np.mean(mean, axis=(0, 1))
  processed_image = process_image(image, mean_pixel)

  weights = np.squeeze(model_data['layers'])

  pred_annotation, logits = fcn(weights,
                                processed_image,
                                keep_prob,
                                debug=FLAGS.debug)
  tf.compat.v1.summary.image('input_image', image, max_outputs=2)
  tf.compat.v1.summary.image('ground_truth',
                             tf.cast(annotation, tf.uint8),
                             max_outputs=2)
  tf.compat.v1.summary.image('pred_annotation',
                             tf.cast(pred_annotation, tf.uint8),
                             max_outputs=2)
  loss = tf.math.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=tf.squeeze(annotation, axis=3), name='entroy')))
  loss_summary = tf.summary.scalar('entropy', loss)

  trainable_var = tf.compat.v1.trainable_variables()
  if FLAGS.debug:
    for var in trainable_var:
      add_to_regulariation_and_summary(var)

  train_op = train(loss, trainable_var)

  print('Setting up summary op...')
  summary_op = tf.compat.v1.summary.merge_all()

  print('Setting up image reader...')
  train_files, valid_files = read_files(FLAGS.data_dir)
  print('Training file length ', len(train_files))
  print('Validation file length', len(valid_files))

  print('Setting up dataset reader')
  image_options = {'resize': True, 'resize_size': FLAGS.image_size}
  if FLAGS.mode == 'train':
    train_dataset_reader = BatchDataset(train_files, image_options)
  validation_dataset_reader = BatchDataset(valid_files, image_options)

  sess = tf.compat.v1.Session()

  print('Setting up Saver...')
  saver = tf.compat.v1.train.Saver()

  # Create two summary writers to show training loss and validation loss in
  # the same graph need to create two folders 'train' and 'validation'
  # inside FLAGS.logs_dir
  train_writer = tf.compat.v1.summary.FileWriter(FLAGS.logs_dir + '/train',
                                                 sess.graph)
  validation_writer = tf.compat.v1.summary.FileWriter(FLAGS.logs_dir +
                                                      '/validation')
  sess.run(tf.compat.v1.global_variables_initializer())
  ckpt = tf.compat.v1.train.get_checkpoint_state(FLAGS.logs_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Model restored...')

  if FLAGS.mode == 'train':
    for step in range(FLAGS.num_steps):
      train_images, train_annotations = train_dataset_reader.next_batch(
          FLAGS.batch_size)
      feed_dict = {
          image: train_images,
          annotation: train_annotations,
          keep_prob: 0.8
      }
      sess.run(train_op, feed_dict=feed_dict)

      if step % 10 == 0:
        train_loss, summary_str = sess.run([loss, loss_summary],
                                           feed_dict=feed_dict)
        print('Step: %d, Train loss: %g' % (step, train_loss))
        train_writer.add_summary(summary_str, step)

      if step % 500 == 0:
        valid_images, valid_annotations = validation_dataset_reader.next_batch(
            FLAGS.batch_size)
        valid_loss, summary_sva = sess.run([loss, loss_summary],
                                           feed_dict={
                                               image: valid_images,
                                               annotation: valid_annotations,
                                               keep_prob: 1.0
                                           })
        print('%s ---> Validation loss %g' %
              (datetime.datetime.now(), valid_loss))

        # Add validation loss to TensorBoard
        validation_writer.add_summary(summary_sva, step)
        saver.save(sess, FLAGS.logs_dir + 'model.ckpt', step)

  elif FLAGS.mode == 'visualize':
    valid_images, valid_annotations = validation_dataset_reader.get_random_batch(
        FLAGS.batch_size)
    pred = sess.run(pred_annotation,
                    feed_dict={
                        image: valid_images,
                        annotation: valid_annotations,
                        keep_prob: 1.0
                    })

    for i in range(FLAGS.batch_size):
      save_image(valid_files[i].astype(np.uint8),
                 FLAGS.logs_dir,
                 name='input_' + str(5 + i))
      save_image(valid_annotations[i].astype(np.uint8),
                 FLAGS.logs_dir,
                 name='label_' + str(5 + i))
      save_image(pred[i].astype(np.uint8),
                 FLAGS.logs_dir,
                 name='pred_' + str(5 + i))
      print('Saved image: %d' % i)


if __name__ == '__main__':
  define_flags()
  app.run(main)
