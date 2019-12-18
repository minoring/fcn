import os
import glob
import random
import pickle
import imageio

import numpy as np
import tensorflow as tf
import scipy.misc

from utils import download_and_extract

DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


class BatchDataset(object):

  def __init__(self, files, image_options):
    print('Initializing Batch dataset reader...')
    print(image_options)
    self.files = files
    self.image_options = image_options
    self._read_images()
    self.batch_offset = 0
    self.epochs_completed = 0

  def _read_images(self):
    self.__channels = True
    self.images = np.array([
        self._transform(self._load(filename['image']))
        for filename in self.files
    ])
    self.__channels = False
    self.annotations = np.array([
        np.expand_dims(self._transform(self._load(filename['annotation'])),
                       axis=3) for filename in self.files
    ])
    print('Read images, {} shape', self.images.shape)
    print('Read annotations, {} shape', self.annotations.shape)

  def _load(self, filename):
    return tf.convert_to_tensor(imageio.imread(filename), dtype=tf.float32)

  def _transform(self, image):
    if self.image_options.get('resize', False) and self.image_options['resize']:
      resize_shape = (int(self.image_options['resize_size']),
                      int(self.image_options['resize_size']))
      image = tf.image.resize(
          image,
          resize_shape,
      )

    return image

  def next_batch(self, batch_size):
    start = self.batch_offset
    self.batch_offset += batch_size
    if self.batch_offset > self.images.shape[0]:
      # Finished epoch
      self.epochs_completed += 1
      print("****************** Epochs completed: " +
            str(self.epochs_completed) + "******************")
      # Shuffle the data
      perm = np.arange(self.images.shape[0])
      np.random.shuffle(perm)
      self.images = self.images[perm]
      self.annotations = self.annotations[perm]
      # Start next epoch
      start = 0
      self.batch_offset = batch_size
    end = self.batch_offset

    return self.images[start:end], self.annotations[start:end]

  def get_random_batch(self, batch_size):
    indices = np.random.randint(0, self.images.shape[0],
                                size=batch_size).tolist()
    return self.images[indices], self.annotations[indices]


def read_files(data_dir):
  pickle_filename = 'MITSceneParsing.pickle'
  pickle_filepath = os.path.join(data_dir, pickle_filename)
  if not os.path.exists(pickle_filepath):
    download_and_extract(data_dir, DATA_URL, is_zipfile=True)
    scene_parsing_folder = os.path.splitext(DATA_URL.split('/')[-1])[0]
    result = create_image_lists(os.path.join(data_dir, scene_parsing_folder))
    print('Pickling...')
    with open(pickle_filepath, 'wb') as f:
      pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
  else:
    print('Found pickle file!')

  with open(pickle_filepath, 'rb') as f:
    result = pickle.load(f)
    training_records = result['training']
    validation_records = result['validation']

  return training_records, validation_records


def create_image_lists(image_dir):
  if not tf.io.gfile.exists(image_dir):
    print('Image directory' + image_dir + 'not found.')
    return None
  images = {}

  for directory in ['training', 'validation']:
    images[directory] = []
    file_glob = os.path.join(image_dir, 'images', directory, '*.' + 'jpg')
    file_list = glob.glob(file_glob)

    if not file_list:
      print('No files found')
    else:
      for f in file_list:
        filename = os.path.splitext(f.split('/')[-1])[0]
        annotation_file = os.path.join(image_dir, "annotations", directory,
                                       filename + '.png')
        if os.path.exists(annotation_file):
          record = {
              'image': f,
              'annotation': annotation_file,
              'filename': filename
          }
          images[directory].append(record)
        else:
          print('Annotaiton file not found for %s - Skipping' % filename)

    random.shuffle(images[directory])
    num_images = len(images[directory])
    print('Number of %s files: %d' % (directory, num_images))

  return images
