"""Create a Generator and Discriminator for a GAN network.

Assume inputs will be NCHW-formatted.
"""
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


def dense_layer(
        inputs,
        size_out,
        variables_collections=None,
        trainable=None):
    """Apply a dense layer to 2D inputs."""
    size_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
        "weights",
        shape=(size_in, size_out),
        initializer=initializers.xavier_initializer(uniform=False),
        trainable=trainable)
    tf.add_to_collection(variables_collections, weights)
    bias = tf.get_variable(
        "bias",
        shape=(size_out),
        initializer=initializers.xavier_initializer(uniform=False),
        trainable=trainable)
    tf.add_to_collection(variables_collections, bias)
    return tf.matmul(inputs, weights) + bias


def conv2d(inputs,
           features_out,
           kernel_size=(5, 5),
           stride=(2, 2),
           padding="SAME",
           variables_collections=None,
           trainable=None):
    """Convolve inputs (NCHW) and add bias."""
    kernel = tf.get_variable(
        "kernel",
        shape=(kernel_size[0],
               kernel_size[1],
               inputs.get_shape().as_list()[1],
               features_out),
        initializer=initializers.xavier_initializer(uniform=False),
        trainable=trainable)
    tf.add_to_collection(variables_collections, kernel)
    bias = tf.get_variable(
        "bias",
        shape=(1, features_out, 1, 1),
        initializer=tf.zeros_initializer(),
        trainable=trainable)
    tf.add_to_collection(variables_collections, bias)
    evidence = tf.nn.conv2d(inputs,
                            kernel,
                            strides=(1, 1, stride[0], stride[1]),
                            padding=padding,
                            data_format="NCHW")
    return evidence + bias


def conv2d_T(inputs,
             features_out,
             kernel_size=(5, 5),
             stride=(2, 2),
             padding="SAME",
             variables_collections=None,
             trainable=None):
    """Conv transpose the inputs (NCHW) and add bias."""
    _, in_depth, in_rows, in_cols = inputs.get_shape().as_list()
    in_batch = tf.shape(inputs)[0]
    kernel = tf.get_variable(
        "kernel",
        shape=(kernel_size[0],
               kernel_size[1],
               features_out,
               in_depth),
        initializer=initializers.xavier_initializer(uniform=False),
        trainable=trainable)
    tf.add_to_collection(variables_collections, kernel)
    bias = tf.get_variable(
        "bias",
        shape=(1, features_out, 1, 1),
        initializer=tf.zeros_initializer(),
        trainable=trainable)
    tf.add_to_collection(variables_collections, bias)
    output_shape = tf.stack(
                    [in_batch,
                     features_out,
                     int(in_rows * stride[0]),
                     int(in_cols * stride[1])])

    evidence = tf.nn.conv2d_transpose(inputs,
                                      kernel,
                                      output_shape=output_shape,
                                      strides=(1, 1, stride[0], stride[1]),
                                      padding=padding,
                                      data_format="NCHW")
    return evidence + bias


def leakyReLU(inputs, leak=0.2, name="leakyReLU"):
    """Leaky ReLU operation.

    Normal ReLU is defined as max(inputs, 0). Leaky ReLU is max(inputs, leak).

    """
    with tf.name_scope(name):
        return tf.maximum(inputs, leak*inputs)


def generator(anno, noise, training, reuse=None):
    """Build a generator using prior_space.

    Randomly generate `z`, the noisy input. Concatenate with labels `c`.
    Use a dense layer to transform the latent space into desired shape.

    Args:
        anno: An iterable of given labels.
        training: A Python boolean or TensorFlow boolean tensor-like, such
            as ``placeholder``. This tells ``batch_norm`` when to set
            moving averages.
        batch_size: An integer; used to create random noise.

    Keyword Args:
        prior_dim (100): An integer for the length of random noise.
        reuse (None): Variable scope control.

    Returns:
        The generated image tensor.

    Implicit Returns:
        Batch norm update ops are added to ``tf.GraphKeys.UPDATE_OPS``.
        All generator variables are added to collection ``"G_theta"``.

    Examples:
        Example of using the update ops ::

            batch_norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(batch_norm_update_ops):
                train_op = tf.train.AdamOptimizer(1e-4).minimize(
                    loss, var_list="G_theta")


    """
    gen_vars = "G_theta" if not reuse else None
    with tf.variable_scope("Generator", reuse=reuse):
        latent_space = tf.concat([noise, anno], 1, name="latent_space")

        with tf.variable_scope("Dense1"):
            h_dense = dense_layer(
                latent_space,
                4*4*1024,
                variables_collections=gen_vars)
        h_dense = tf.reshape(h_dense, [-1, 1024, 4, 4])
        h_dense = tf.layers.batch_normalization(
            h_dense,
            axis=1,
            training=training,
            name="batch_norm1")
        a_dense = tf.nn.relu(h_dense)

        with tf.variable_scope("conv2d_T2"):
            conv2 = conv2d_T(
                a_dense,
                512,
                variables_collections=gen_vars)
        conv2 = tf.layers.batch_normalization(
            conv2,
            axis=1,
            training=training,
            name="batch_norm2")
        a_conv2 = tf.nn.relu(conv2)

        with tf.variable_scope("conv2d_T3"):
            conv3 = conv2d_T(
                a_conv2,
                256,
                variables_collections=gen_vars)
        conv3 = tf.layers.batch_normalization(
            conv3,
            axis=1,
            training=training,
            name="batch_norm3")
        a_conv3 = tf.nn.relu(conv3)

        with tf.variable_scope("conv2d_T4"):
            conv4 = conv2d_T(
                a_conv3,
                128,
                variables_collections=gen_vars)
        conv4 = tf.layers.batch_normalization(
            conv4,
            axis=1,
            training=training,
            name="batch_norm4")
        a_conv4 = tf.nn.relu(conv4)

        with tf.variable_scope("conv2d_T5"):
            conv5 = conv2d_T(
                a_conv4,
                3,
                variables_collections=gen_vars)
        image = tf.nn.tanh(conv5)

    return image


def discriminator(image, anno, training, reuse=None):
    """Build a discriminator to verify ``image``.

    Perform usual convolutions on ``image``. Flatten and concatenate with
    the label tensor ``c``. Then use a fully-connected layer to reduce to
    a true/false prediction.

    Args:
        image: The input Tensor.
        anno: The label Tensor.
        training: A Python boolean or TensorFlow boolean tensor-like, such
            as ``placeholder``. This tells ``batch_norm`` when to set
            moving averages.

    Keyword Args:
        reuse: Variable scope control.

    Returns:
        A ``batch_size x 1`` Tensor indicating true/false, the image is real.

    Implicit Returns:
        All discriminator variables are added to collection ``"D_theta"``.

    """
    dis_vars = "D_theta" if not reuse else None
    with tf.variable_scope("Discriminator", reuse=reuse):
        with tf.variable_scope("conv2d1"):
            conv1 = conv2d(
                image,
                128,
                variables_collections=dis_vars)
        a_conv1 = leakyReLU(conv1)
        with tf.variable_scope("conv2d2"):
            conv2 = conv2d(
                a_conv1,
                256,
                variables_collections=dis_vars)
        a_conv2 = leakyReLU(conv2)
        with tf.variable_scope("conv2d3"):
            conv3 = conv2d(
                a_conv2,
                512,
                variables_collections=dis_vars)
        a_conv3 = leakyReLU(conv3)
        with tf.variable_scope("conv2d4"):
            conv4 = conv2d(
                a_conv3,
                1024,
                variables_collections=dis_vars)
        a_conv4 = leakyReLU(conv4)
        a_conv4 = tf.reshape(a_conv4, [-1, 1024*4*4])
        with tf.name_scope("Dense5"):
            logprob = dense_layer(
                tf.concat([a_conv4, anno], 1, name="concat_c"),
                1,
                variables_collections=dis_vars)
        prob = tf.nn.sigmoid(logprob)
    return prob
