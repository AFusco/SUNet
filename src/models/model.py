import tensorflow as tf
import re
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

import model_input
import argparse
from utils import layers

BATCH_SIZE=128
DATA_DIR='../../data/kitti'
DATA_NAME='kitti'
USE_FP16=False
LEARNING_RATE=0.01

#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

FLAGS = []

def distorted_inputs():
    lefts, disps, confs = model_input.distorted_inputs(data_dir=FLAGS.data_name,
                                                    batch_size=FLAGS.batch_size)

    if USE_FP16:
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
    """ Generate logits for a confidence map for left and disp """

    def encoding_unit(name, inputs, num_outputs, batch_norm=True):
        """
        input -> conv(3x3) -> norm -> relu -> maxpool2

        return maxpool2, conv3x3

        """

        with tf.variable_scope('encoding' + str(name)):

            # Apply 3x3 convolution
            # Don't apply Relu now because this activation 
            # will be reused for feed forwarding
            conv = tf.contrib.layers.conv2d( 
                        inputs=inputs,
                        num_outputs=num_outputs,
                        kernel_size=3,
                        activation_fn=None
                    )

            # Normalize batch
            if batch_norm:
                conv = tf.contrib.layers.batch_norm(conv)
            # Apply relu
            relu = tf.nn.relu(conv)
            # Maxpool of a factor of 2 and default stride 2
            pool = tf.contrib.layers.max_pool2d(relu, 2)

        forward = conv
        return pool, forward

    def decoding_unit(number, inputs, num_outputs, forwards=None, batch_norm=True):
        """
        input -> conv_transpose(3x3s2) -> add(forwards) -> batch_norm -> relu

        return relu
        """

        with tf.variable_scope('decoding' + number):

            conv_transpose = tf.contrib.layers.conv2d_transpose(
                        inputs=inputs,
                        num_outputs=num_outputs*2,
                        kernel_size=3,
                        stride=2,
                        activation_fn=None
                    )

            if forwards != None:
                if isinstance(forwards, (list, tuple)):
                    for f in forwards:
                        conv_transpose = tf.concat([conv_transpose, f], axis=3)
                else:
                    conv_transpose = tf.concat([conv_transpose, forwards], axis=3)

            # Reduce depth
            conv = tf.contrib.layers.conv2d( 
                        inputs=inputs,
                        num_outputs=num_outputs,
                        kernel_size=3,
                        activation_fn=None
                    )

            if batch_norm:
                conv = tf.contrib.layers.batch_norm(conv)

            relu = tf.nn.relu(conv)

        return conv_transpose


    ######################
    # DEFINE THE NETWORK #
    ######################

    # Left image input
    #256x512x3 -> 256x512x32
    with tf.variable_scope('left') as scope:
        #conv3x3->relu
        left_feat = tf.contrib.layers.conv2d( 
                        inputs=lefts,
                        num_outputs=32,
                        kernel_size=3
                    )

    # disp image input
    #256x512x1 -> 256x512x32
    with tf.variable_scope('disp') as scope:
        #conv3x3->relu
        disp_feat = tf.contrib.layers.conv2d( 
                        inputs=lefts,
                        num_outputs=32,
                        kernel_size=3
                    )

    # Concatenation of left + disp activation maps
    # x32 + x32 -> 256x512x64
    concat = tf.concat([left_feat, disp_feat], axis=3)

    net, scale1 = encoding_unit('1', concat, num_outputs=64)
    print('Net shape: ', net.shape, ' Scale shape ', scale1.shape)
    net, scale2 = encoding_unit('2', net, num_outputs=128)
    print('Net shape: ', net.shape, ' Scale shape ', scale2.shape)
    net, scale3 = encoding_unit('3', net, num_outputs=256)
    print('Net shape: ', net.shape, ' Scale shape ', scale3.shape)
    net, scale4 = encoding_unit('4', net, num_outputs=512)
    print('Net shape: ', net.shape, ' Scale shape ', scale4.shape)

    net = decoding_unit('4', net, num_outputs=256, forwards=scale4)
    print('Net shape: ', net.shape)
    net = decoding_unit('3', net, num_outputs=128, forwards=scale3)
    print('Net shape: ', net.shape)
    net = decoding_unit('2', net, num_outputs=64,  forwards=scale2)
    print('Net shape: ', net.shape)

    # Use only left feat 
    net = decoding_unit('1', net, num_outputs=32, forwards=concat, batch_norm=False)

    disp_logits = tf.contrib.layers.conv2d( 
                inputs=net,
                num_outputs=1,
                kernel_size=3,
                activation_fn=None
            )
    print('Net shape: ', disp_logits.shape)
    disparity = tf.nn.sigmoid(disp_logits)

    return disp_logits, disparity

def loss(disp_logits, disp_gt):
    # Mask out the error corresponding to unknown pixels
    mask = tf.where(disp_gt == -1, tf.ones_like(disp_gt), tf.zeros_like(disp_gt))

    # Calculate cross entropy
    xentr = tf.losses.sigmoid_cross_entropy(
        disp_gt,
        logits=disp_logits,
        weights=mask
    )

    tf.summary.scalar('Cross_entropy', xentr)

    return xentr

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
