from absl import flags


def define_flags():
  flags.DEFINE_integer('batch_size', 2, 'Batch size for training')
  flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
  flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
  flags.DEFINE_string('model_dir', 'models/', 'Path to model')
  flags.DEFINE_string(
      'model_URL',
      'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat',
      'VGG Model URL')
  flags.DEFINE_float('learning_rate', '1e-4',
                     'Learning rate for Adam optimizer')
  flags.DEFINE_bool('debug', 'False', 'Debug mode: [True, False]')
  flags.DEFINE_string('mode', 'train', 'Mode [train, test, visualize]')
  flags.DEFINE_integer('image_size', 224, 'Size of input image')
  flags.DEFINE_integer('num_steps', 1000, 'Number of training steps')
