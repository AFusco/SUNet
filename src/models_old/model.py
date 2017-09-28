import tensorflow as tf
import re
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

import model_input
import argparse
from utils import layers


parser = model_input.parser

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

parser.add_argument('-f', default='no')

#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

FLAGS = []

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

    bs = int(lefts.shape[0])

    print("Batch size: ", bs)

    #256*512
    # ->conv1->
    #128*256
    # ->conv2->
    #64*128
    # ->conv3->
    #32*64
    # ->conv4->
    #16*32
    # ->layers.conv_transpose4->
    #32*64
    # ->layers.conv_transpose5->
    #64*128
    # ->layers.conv_transpose6->
    #128*256
    # ->layers.conv_transpose7->
    #256*512


    #256x512x3 -> 256x512x32
    with tf.variable_scope('left') as scope:
        left_feat = layers.conv2d_relu('conv1', lefts, [3,3,3,32], [1,1,1,1])

    #256x512x1 -> 256x512x32
    with tf.variable_scope('disp') as scope:
        disp_feat = layers.conv2d_relu('conv1', disps, [3,3,1,32], [1,1,1,1])

    #x32 + x32 -> 256x512x64
    concat = tf.concat([left_feat, disp_feat], axis=3)

    #256x512x64 -> 256x512x64
    net = layers.conv2d_relu('conv1_concat', concat, [3,3,64,64], [1,1,1,1])
    scale1 = net
    net = tf.contrib.layers.batch_norm(net)
    #256x512x64 -> 128x256x64
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')


    def encoder_unit(name, batch, kernel_shape, strides, padding='SAME',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    bias_initializer=tf.constant_initializer(0.0)):

        with tf.variable_scope('encoder' + number):
            with tf.variable_scope('conv'):
                kernel = _variable_on_cpu('weights',
                                           shape=kernel_shape,
                                           initializer=kernel_initializer)

                conv = tf.nn.conv2d(batch, kernel, strides, padding=padding)
                biases = _variable_on_cpu('biases', 
                        [kernel_shape[3]], bias_initializer)

                pre_activation = tf.nn.bias_add(conv, biases)





    print(net.name, net.shape)


    # 128x256x64 -> 128x256x128
    net = layers.conv2d_relu('conv2', net, [3,3,64,128], [1,1,1,1])
    scale2 = net
    net = tf.contrib.layers.batch_norm(net)
    print(net.name, net.shape)
    #128x256x128 -> 64x128x128
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')
    print(net.name, net.shape)



    #64x128x128 ->64x128x128
    net = layers.conv2d_relu('conv3', net, [3,3,128,256], [1,1,1,1])
    scale3 = net
    net = tf.contrib.layers.batch_norm(net)
    print(net.name, net.shape)
    #64x128x128 -> 32x64x256
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')
    print("scale3-->", net.name, net.shape)


    #32x64x256 -> 32x64x512
    net = layers.conv2d_relu('conv4', net, [3,3,256,512], [1,1,1,1])
    net = tf.contrib.layers.batch_norm(net)
    print(net.name, net.shape)
    #32x64x512 -> 16x32x512
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool4')
    print(net.name, net.shape)


    #16x32x512 -> 32x64x256
    up = layers.conv_transpose_2x_relu('transp1', net, [3,3,256,512], [1,2,2,1])
    tf.add(up, scale3)
    print(up.name, up.shape)

    #32x64x256 -> 64x128x128
    up = layers.conv_transpose_2x_relu('transp2', up, [3,3,128,256], [1,2,2,1])
    tf.add(up, scale2)
    print(up.name, up.shape)

    #64x128x128 -> 128x256x64
    up = layers.conv_transpose_2x_relu('transp3', up, [3,3,64,128], [1,2,2,1])
    tf.add(up, scale1)
    print(up.name, up.shape)

    #128x256x64 -> 256x512x32
    up = layers.conv_transpose_2x_relu('transp4', up, [3,3,32,64], [1,2,2,1])
    tf.add(up, left_feat)
    tf.add(up, disp_feat)
    print(up.name, up.shape)

    up = layers.conv2d('conv5', up, [3,3,32,1], [1,1,1,1])
    print(up.name, up.shape)

    return up

def loss(disp, disp_gt):
    # Mask out the error corresponding to unknown pixels
    mask = tf.where(disp_gt == -1, tf.ones_like(disp_gt), tf.zeros_like(disp_gt))


    # Calculate MSE
    mse = tf.losses.sigmoid_cross_entropy(
        disp,
        disp_gt,
        weights=mask
    )

    tf.summary.scalar('Cross_entropy', mse)

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

if __name__ == "__main__":
    FLAGS = parser.parse_args()

