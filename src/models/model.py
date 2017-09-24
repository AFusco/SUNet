import tensorflow as tf
import re
import model_input
import os
import argparse
from utils import layers

parser = argparse.ArgumentParser()

dir_path = os.path.dirname(os.path.realpath(__file__))

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str,
                    default=os.path.join(dir_path, '../../data/kitti'),
                    help='Path to the data directory.')

parser.add_argument('--data_name', type=str,
                    default='kitti',
                    help='Path to the data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='Train the model using fp16.') # Non funziona

parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='The optimization learning rate') # Non funziona

#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

FLAGS = parser.parse_args()

def distorted_inputs():
    lefts, disps, confs = model_input.distorted_inputs(data_dir=FLAGS.data_name,
                                                    batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        lefts = tf.cast(lefts, tf.float16)
        disps = tf.cast(disps, tf.float16)
        confs = tf.cast(confs, tf.float16)

    return lefts, disps, confs


def inputs(eval_data):
    lefts, disps, confs = model_input.inputs(eval_data=eval_data,
                                        data_dir=FLAGS.data_name,
                                        batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        lefts = tf.cast(lefts, tf.float16)
        disps = tf.cast(disps, tf.float16)
        confs = tf.cast(confs, tf.float16)

    return lefts, disps, confs

def inference(lefts, disps):

    batch_size = int(lefts.shape[0])

    print("Batch size: ", batch_size)

    with tf.variable_scope('left') as scope:
        left = layers.conv2d_relu('conv1_a', lefts, [3,3,3,64], [1,1,1,1])
        left = layers.conv2d_relu('conv1_b', left, [3,3,64,64], [1,1,1,1])
        left = layers.conv2d_relu('conv1_c', left, [3,3,64,64], [1,1,1,1])
        left_pool = tf.contrib.layers.batch_norm(left)
        left_pool = tf.nn.max_pool(left_pool, ksize=[1,2,2,1], strides=[1,2,2,1],
                padding='SAME', name='pool1')


    with tf.variable_scope('disp') as scope:
        disp = layers.conv2d_relu('conv1_a', disps, [3,3,1,64], [1,1,1,1])
        disp = layers.conv2d_relu('conv1_b', disp, [3,3,64,64], [1,1,1,1])
        disp = layers.conv2d_relu('conv1_c', disp, [3,3,64,64], [1,1,1,1])
        disp_pool = tf.contrib.layers.batch_norm(disp)
        disp_pool = tf.nn.max_pool(disp_pool, ksize=[1,2,2,1], strides=[1,2,2,1],
                padding='SAME', name='pool1')

    net = tf.concat([left_pool, disp_pool], axis=3)

    net = layers.conv2d_relu('conv3', net, [3,3,128,128], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = layers.conv2d_relu('conv4', net, [3,3,128,128], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    net = layers.conv2d_relu('conv5', net, [3,3,128,256], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = layers.conv2d_relu('conv6', net, [3,3,256,256], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')

    net = layers.conv2d_relu('conv7', net, [3,3,256,512], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = layers.conv2d_relu('conv8', net, [3,3,512,512], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool4')

    net = layers.conv2d_relu('conv9', net, [3,3,512,1024], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = layers.conv2d_relu('conv10', net, [3,3,1024,1024], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)

    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool5')

    net = layers.conv2d_relu('conv11', net, [3,3,1024,1024], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = layers.conv2d_relu('conv12', net, [3,3,1024,1024], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)

    net = layers.bilinear_upsample('unpool1', net, 4)

    net = layers.conv2d_relu('conv13', net, [3,3,1024,512], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = layers.conv2d_relu('conv14', net, [3,3,512,256], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)

    net = layers.bilinear_upsample('unpool2', net, 4)

    net = layers.conv2d_relu('conv15', net, [3,3,256,64], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    net = layers.conv2d_relu('conv16', net, [3,3,64,8], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)

    net = layers.bilinear_upsample('unpool3', net, 2)

    net = layers.conv2d_relu('conv17', net, [3,3,8,1], [1,1,1,1])

    net = tf.sigmoid(net)

    return net

def loss(disp, disp_gt):
    # Mask out the error corresponding to unknown pixels
    mask = tf.where(disp_gt == -1, tf.ones_like(disp_gt), tf.zeros_like(disp_gt))

    # Calculate MSE
    mse = tf.losses.mean_squared_error(
        disp,
        disp_gt,
        weights=mask
    )

    tf.summary.scalar('Mse loss', mse)

    return mse


def train(total_loss, global_step):

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    #Compute gradients.
    grads = optimizer.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op
