#!/usr/bin/env python
import json
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from utils import plot


BATCH_SIZE = 16


def noisy_space(batch_size, noise_size=100):
    """Create noise input for generator."""
    return np.random.uniform(
        size=[batch_size, noise_size], low=-1, high=1)


def main(_):
    desired_label = -np.ones([BATCH_SIZE, len(options)])
    for i in range(len(options)):
        if options[i] in classes:
            desired_label[:, i] = True

    signature_key = (tf.saved_model.signature_constants.
                     DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    input_key = "anno"
    istrain_key = "is_train"
    noise_key = "noise"
    output_key = "g_output"


    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            FLAGS.model_dir)
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
            feed_dict={c: desired_label,
                       noise: noisy_space(BATCH_SIZE),
                       is_train: False})

    plot(y_out, BATCH_SIZE, None, save=False, show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.abspath("./savedmodel"),
        help="Folder containing the model.")
    FLAGS, unparsed = parser.parse_known_args()
    configured_classes = json.load(
        open(os.path.join(FLAGS.model_dir, "classes.json"), "r"))
    options = [key for key, val in configured_classes.items() if val]
    classes = input(("Enter your choices of the following "
                     "separated with spaces [{}]: \n").format(
                         " ".join(options)))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
