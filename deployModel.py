#!/usr/bin/env python
import tensorflow as tf
import numpy as np


batch_size = 1

def noisy_space(batch_size, noise_size=100):
    """Create noise input for generator."""
    return np.random.uniform(
        size=[batch_size, noise_size], low=-1, high=1)


sess = tf.Session()
signature_key = (tf.saved_model.signature_constants.
                 DEFAULT_SERVING_SIGNATURE_DEF_KEY)
input_key = "anno"
istrain_key = "is_train"
noise_key = "noise"
output_key = "g_output"

export_path = "./savedmodel"
meta_graph_def = tf.saved_model.loader.load(
    sess,
    [tf.saved_model.tag_constants.SERVING],
    export_path)
signature = meta_graph_def.signature_def

x_name = signature[signature_key].inputs[input_key].name
y_name = signature[signature_key].outputs[output_key].name
p_name = signature[signature_key].inputs[istrain_key].name
n_name = signature[signature_key].inputs[noise_key].name

c = sess.graph.get_tensor_by_name(x_name)
g_img = sess.graph.get_tensor_by_name(y_name)
is_train = sess.graph.get_tensor_by_name(p_name)
noise = sess.graph.get_tensor_by_name(n_name)

y_out = sess.run(
    g_img,
    feed_dict={
        c: np.random.randint(0, 2, [batch_size, 40]),
        noise: noisy_space(batch_size),
        is_train: False})
