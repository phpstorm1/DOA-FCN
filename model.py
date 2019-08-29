import numpy as np
import tensorflow as tf


def create_model_settings(dim_direction_label,
                          sample_rate,
                          win_len,
                          win_shift,
                          nDFT,
                          context_window_width):

    model_settings = {
        'num_class': dim_direction_label,
        'sample_rate': sample_rate,
        'win_len': win_len,
        'win_shift': win_shift,
        'nDFT': nDFT,
        'win_fun': np.hamming,
        'context_window_width': context_window_width,
    }

    return model_settings


def load_variables_from_checkpoint(sess, start_checkpoint, var_list=None):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
      var_list: List of variables to load from checkpoint. If None, tf.global_variables() will be used.
    """
    if var_list is None:
        var_list = tf.global_variables()
    saver = tf.train.Saver(var_list)
    saver.restore(sess, start_checkpoint)


def doa_cnn(phase_specs, model_settings, is_training=False):

    # output = tf.layers.batch_normalization(phase_specs, training=is_training)
    output = phase_specs
    num_batch = tf.shape(output)[0]

    # with tf.variable_scope('conv_0'):
    #     output = tf.layers.conv2d(output, 2, (model_settings['context_window_width'], 1), padding='valid')
    #     output = tf.nn.relu(output)
    with tf.variable_scope('conv_1'):
        output = tf.layers.conv2d(output, 8, (1, 7), dilation_rate=(1, 1), padding='same')
        output = tf.nn.leaky_relu(output)
    with tf.variable_scope('conv_2'):
        output = tf.layers.conv2d(output, 64, (1, 7), dilation_rate=(1, 2), padding='same')
        output = tf.nn.relu(output)
    # with tf.variable_scope('conv_3'):
    #     output = tf.layers.conv2d(output, 16, (1, 5), dilation_rate=(1, 4), padding='same')
    #     output = tf.nn.relu(output)
    # with tf.variable_scope('conv_4'):
    #     output = tf.layers.conv2d(output, 32, (1, 5), dilation_rate=(1, 8), padding='same')
    #     output = tf.nn.relu(output)
    # with tf.variable_scope('conv_5'):
    #     output = tf.layers.conv2d(output, 64, (1, 5), dilation_rate=(1, 8), padding='same')
    #     output = tf.nn.relu(output)
    with tf.variable_scope('conv_3'):
        output = tf.layers.conv2d(output, 128, (1, 7), dilation_rate=(1, 4), padding='same')
        output = tf.nn.relu(output)
    with tf.variable_scope('reduce_sum'):
        output = tf.reduce_sum(output, [1, 2])
        output = tf.reshape(output, shape=[num_batch, 1, 1, 128])
    # with tf.variable_scope('reshape'):
    #     output = tf.layers.flatten(output)
    # with tf.variable_scope('dense_1'):
    #     output = tf.layers.dense(output, 128)
    #     output = tf.nn.relu(output)
    #     # output = tf.layers.dropout(output, training=is_training)
    # with tf.variable_scope('dense_2'):
    #     output = tf.layers.dense(output, model_settings['num_class'])
    with tf.variable_scope('1_by_1_conv_1'):
        output = tf.layers.conv2d(output, 128, (1, 1), padding='same')
        output = tf.nn.relu(output)
    with tf.variable_scope('1_by_1_conv_2'):
        output = tf.layers.conv2d(output, model_settings['num_class'], (1, 1), padding='same')
    with tf.variable_scope('reshape'):
        output = tf.layers.flatten(output)

    return output
