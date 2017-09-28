import tensorflow as tf
import re

#FIXME
use_fp16=False
TOWER_NAME = 'tower'

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    #with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float32 #tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv2d_relu(name, batch, kernel_shape, strides, padding='SAME',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.0)):

    with tf.variable_scope(name) as scope:

        kernel = _variable_on_cpu('weights',
                                   shape=kernel_shape,
                                   initializer=kernel_initializer)

        conv = tf.nn.conv2d(batch, kernel, strides, padding=padding)
        biases = _variable_on_cpu('biases', [kernel_shape[3]], bias_initializer)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv)

    return conv

def conv2d(name, batch, kernel_shape, strides, padding='SAME',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.0)):

    with tf.variable_scope(name) as scope:

        kernel = _variable_on_cpu('weights',
                                   shape=kernel_shape,
                                   initializer=kernel_initializer)

        conv = tf.nn.conv2d(batch, kernel, strides, padding=padding)
        biases = _variable_on_cpu('biases', [kernel_shape[3]], bias_initializer)
        pre_activation = tf.nn.bias_add(conv, biases)

    return pre_activation

def relu(batch):
    out = tf.nn.relu(batch, name=batch.name)
    return out


def conv2d_sigmoid(name, batch, kernel_shape, strides, padding='SAME',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.0)):

    with tf.variable_scope(name) as scope:

        kernel = _variable_on_cpu('weights',
                                   shape=kernel_shape,
                                   initializer=kernel_initializer)

        conv = tf.nn.conv2d(batch, kernel, strides, padding=padding)
        biases = _variable_on_cpu('biases', [kernel_shape[3]], bias_initializer)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.sigmoid(pre_activation, name=scope.name)
        _activation_summary(conv)

    return conv

def calculate_upscale_shape(batch, kernel_shape, scale_factor):
    """ Return an op that calculates sizes """

    batch_size = tf.shape(batch)[0]
    new_height = tf.multiply(tf.shape(batch)[1], scale_factor)
    new_width = tf.multiply(tf.shape(batch)[2], scale_factor)
    depth = kernel_shape[2] # format for conv_transpose is [h,w, output_depth, input_depth]

    return tf.stack([batch_size, new_height, new_width, depth])

def conv_transpose_2x_relu(name, batch, kernel_shape, strides, padding='SAME',
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.constant_initializer(0.0)):

    with tf.variable_scope(name) as scope:

        output_shape = calculate_upscale_shape(batch, kernel_shape, 2)
        kernel = _variable_on_cpu('weights',
                                   shape=kernel_shape,
                                   initializer=kernel_initializer)
        conv = tf.nn.conv2d_transpose(batch, kernel, output_shape, strides)
        biases = _variable_on_cpu('biases', [kernel_shape[2]], bias_initializer)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)

    return conv


def bilinear_kernel_initializer(shape):
    if shape[0] != shape[1]:
        raise Exception('deconv2d_bilinear_upsampling_initializer' +
                        'only supports symmetrical filter sizes')

    if shape[3] < shape [2]:
        raise Exception('deconv2d_bilinear_upsampling_initializer' +
                'behaviour is not defined for num_in_channels < num_out_channels')

    filter_size = shape[0]
    num_out_channels = shape[2]
    num_in_channels = shape[3]

    #Create bilinear filter kernel as numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                   (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    #assign numpy array to constant_initalizer and pass to get_variable
    bilinear_weights_init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return bilinear_weights_init

def upsample_layer(name, batch, scale):
    with tf.variable_scope(name) as scope:

        kernel_shape = (2 * rescale_factor - rescale_factor % 2)

        kernel = _variable_on_cpu('weights',
                                   shape=kernel_shape,
                                   initializer=bilinear_kernel_initializer)

        conv = tf.nn.conv2d(batch, kernel, strides, padding=padding)
        biases = _variable_on_cpu('biases', [kernel_shape[3]], bias_initializer)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv)

    return conv



def bilinear_upsample(name, batch, scale):
    with tf.variable_scope(name) as scope:
        new_height = int(batch.get_shape().as_list()[1] * scale)
        new_width = int(batch.get_shape().as_list()[2] * scale)
        resized = tf.image.resize_images(batch, [new_height, new_width])

        return resized

