import tensorflow as tf
import os
import numpy as np

IMAGE_SIZE=300

def _generate_training_batch(left, disp, conf, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of tuples in the form of (left, disp, conf).
    Args:
        left: 3-D Tensor of [height, width, 3] of type.uint8
        disp: 3-D Tensor of [height, width, 1] of type.uint8
        conf: 3-D Tensor of [height, width, 1] of type.float
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        lefts: Left images. 4D tensor of [batch_size, height, width, 3] size.
        disps: Disp images. 4D tensor of [batch_size, height, width, 1] size.
        confs: Conf images. 4D tensor of [batch_size, height, width, 1] size.
    """

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        lefts, disps, confs = tf.train.shuffle_batch(
            [left, disp, conf],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        lefts, disps, confs = tf.train.batch(
            [left, disp, conf],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('lefts', lefts)
    tf.summary.image('disps', disps)
    tf.summary.image('confs', confs)

    return lefts, disps, confs

def distorted_inputs(data_dir, batch_size):
    """
    Construct random HEIGHTxWIDTH crops of the images in data_dir/train.tfrecord

    Args:
        data_dir: Name of the dataset. It will be searched inside the data/processed
        batch_size: Number of samples per batch.
    Returns:
        lefts: Left images. 4D tensor of [batch_size, height, width, 3] size.
        disps: Disp images. 4D tensor of [batch_size, height, width, 1] size.
        confs: Conf images. 4D tensor of [batch_size, height, width, 1] size.
    """
    
    #todo change
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '../../data/processed', data_dir, 'train.tfrecords')

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer([filename])

    # Read examples from files in the filename queue.
    left, disp, conf = read_and_decode(filename_queue)

    left = tf.cast(left, tf.float32)
    disp = tf.cast(disp, tf.float32)
    conf = tf.cast(conf, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. 
    # We apply random 'in-place' distorsion to the left images
    # Translations/flips/rotations would require a careful implementation
    # to modify the disparity and confidence maps correctly

    # We apply the same random crop to all the images
    # Randomly crop a [height, width] section of the image.
    offset_height = tf.random_uniform([], minval=0, maxval=360-height, dtype=tf.int32)
    offset_width = tf.random_uniform([], minval=0, maxval=1200-width, dtype=tf.int32)

    resized_left = tf.image.crop_to_bounding_box(left, offset_height, offset_width, height, width)
    resized_disp = tf.image.crop_to_bounding_box(disp, offset_height, offset_width, height, width)
    resized_conf = tf.image.crop_to_bounding_box(conf, offset_height, offset_width, height, width)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_left = resized_left


#    distorted_left = tf.image.random_brightness(distorted_left,
#					       max_delta=63)
#    distorted_left = tf.image.random_contrast(distorted_left,
#					     lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
#    distorted_left = tf.image.per_image_standardization(distorted_left)


    # Ensure that the random shuffling has good mixing properties.
    min_queue_examples = 100
    print ('Filling queue with %d CIFAR images before starting to train. '
	 'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_training_batch(distorted_left, resized_disp, resized_conf,
					 min_queue_examples, batch_size,
					 shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the directory containing the tfrecord files.
        batch_size: Number of images per batch.

    Returns:
        lefts: Left Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        disps: Disparity Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        confs: . 1D tensor of [batch_size] size.
    """

    if not eval_data:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '../../data/processed', data_dir, 'test.tfrecords')
    else:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '../../data/processed', data_dir, 'train.tfrecords')

    filename_queue = tf.train.string_input_producer([tfname])

    left, disp, conf = read_and_decode(filename_queue)
    left = tf.cast(left, tf.float32)
    disp = tf.cast(disp, tf.float32)
    conf = tf.cast(conf, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    offset_height = tf.random_uniform([], minval=0, maxval=60, dtype=tf.int32)
    offset_width = tf.random_uniform([], minval=0, maxval=900, dtype=tf.int32)

    resized_left = tf.image.crop_to_bounding_box(left, offset_height, offset_width, height, width)
    resized_disp = tf.image.crop_to_bounding_box(disp, offset_height, offset_width, height, width)
    resized_conf = tf.image.crop_to_bounding_box(conf, offset_height, offset_width, height, width)

    # Ensure that the random shuffling has good mixing properties.
    min_queue_examples = 100
    print ('Filling queue with %d CIFAR images before starting to train. '
	 'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_training_batch(distorted_left, resized_disp, resized_conf,
					 min_queue_examples, batch_size,
					 shuffle=True)


def read_and_decode(filename_queue):

    # Reader 
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'left_raw': tf.FixedLenFeature([], tf.string),
        'disp_raw': tf.FixedLenFeature([], tf.string),
        'conf_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    # Get original height and width, needed to reconstruct the image
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    # Convert from a scalar string tensor to a uint8
    # tensor with the original shape
    left = tf.decode_raw(features['left_raw'], tf.uint8)
    disp = tf.decode_raw(features['disp_raw'], tf.uint8)

    # Confidence map is made of floats
    conf = tf.decode_raw(features['conf_raw'], tf.float32)

    # Reshape to original size.
    # Disp and gt will have a third dimension to allow future stacking.
    left = tf.reshape(left, tf.stack([height, width, 3]))
    disp = tf.reshape(disp, tf.stack([height, width, 1]))
    conf = tf.reshape(conf, tf.stack([height, width, 1]))

    return left, disp, conf

    # Resize to constant size
    # resized_left = tf.image.resize_image_with_crop_or_pad(image=left, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    # resized_disp = tf.image.resize_image_with_crop_or_pad(image=disp, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    # resized_gt = tf.image.resize_image_with_crop_or_pad(image=gt, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)

    # return _generate_training_batch(resized_left, resized_disp, resized_gt, 100, 100)
