"""Utility functions"""
import os
import sys
import urllib
import tarfile
import zipfile
import imageio
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
