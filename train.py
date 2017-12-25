#!/usr/bin/env python
"""Train a Classifier Deep Convolutional GAN on the Celeba dataset"""
import os
import time
import argparse
import sys
import logging
import tensorflow as tf
from model import generator, discriminator
import utils


DSET_SIZE = 202599


def goodfellow_loss(d_real, d_fake):
    """Ian Goodfellow loss for GAN with small epsilon for stability."""
    epsilon = 1.e-10
    with tf.name_scope("D_Eval"):
        d_loss = -tf.reduce_mean(
            tf.log(d_real+epsilon) + tf.log(1. - d_fake+epsilon))
        tf.summary.scalar("D_loss", d_loss)
    with tf.name_scope("G_Eval"):
        g_loss = -tf.reduce_mean(tf.log(d_fake+epsilon))
        tf.summary.scalar("G_loss", g_loss)
    return d_loss, g_loss


def train(init_rate, loss, var_list, beta1=0.5, name="Train"):
    """Use adam to train."""
    with tf.name_scope(name):
        train_op = tf.train.AdamOptimizer(init_rate, beta1=beta1).minimize(
            loss, var_list=var_list)
    return train_op


def init_logger(name):
    """Initialize a logger with nice formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def inputs(is_training):
    """Construct input pipeline.

    Use command line args to set up dataset iterators with appropriate batch
    sizes and epoch counts pointing to the correct tfr path. Then use
    is_training to return the contents of the desired iterator.
    """
    with tf.name_scope("inputs"):
        with tf.name_scope("train_data"):
            train_dataset = utils.celeba_input(
                FLAGS.tfr_tr_path, FLAGS.batch_size, FLAGS.epochs)
            train_iterator = train_dataset.make_initializable_iterator()
        with tf.name_scope("test_data"):
            test_dataset = utils.celeba_input(
                FLAGS.tfr_te_path,
                FLAGS.batch_size,
                int(FLAGS.epochs / FLAGS.eval_freq))
            test_iterator = test_dataset.make_initializable_iterator()
        image, anno = tf.cond(
            is_training,
            lambda: train_iterator.get_next(),
            lambda: test_iterator.get_next(),
            name="choose_data")
        iterators = [train_iterator, test_iterator]
    return iterators, image, anno


def run_training():
    """Run training."""
    # Initialize logging.
    trlogger = init_logger("[TR]")
    telogger = init_logger("[TE]")
    runlogger = init_logger("run")
    # Build the graph.
    is_train = tf.placeholder(tf.bool, shape=[], name="is_training")
    iterators, image, anno = inputs(is_train)
    g_sample = generator(anno, is_train, FLAGS.batch_size)
    tf.summary.image(
        "GENERATED_IMG",
        tf.transpose(g_sample, [0, 2, 3, 1], name="FORMAT"),
        1)
    with tf.variable_scope("Discriminator"):
        d_real = discriminator(
            image, anno, is_train)
        with tf.variable_scope("ProbSummary"):
            utils.prob_scale(d_real, "DISC_PROB_REAL", 1)
        d_fake = discriminator(
            g_sample, anno, is_train, reuse=True)
        with tf.variable_scope("ProbSummary"):
            utils.prob_scale(d_fake, "DISC_PROB_FAKE", 1)

        tf.summary.scalar("AvgReal", tf.reduce_mean(d_real))
        tf.summary.scalar("AvgFake", tf.reduce_mean(d_fake))
    d_loss, g_loss = goodfellow_loss(d_real, d_fake)
    d_train = train(FLAGS.D_init_rate,
                    d_loss,
                    tf.get_collection("D_theta"),
                    beta1=FLAGS.beta_1,
                    name="D_train")
    g_batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(g_batch_norm_ops):
        g_train = train(FLAGS.G_init_rate,
                        g_loss,
                        tf.get_collection("G_theta"),
                        beta1=FLAGS.beta_1,
                        name="G_train")
    summary_op = tf.summary.merge_all()
    # Loop variables
    epochs_per_dset = int(DSET_SIZE * (1-FLAGS.test_size) / FLAGS.batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("\n    $ tensorboard --logdir={}\n".format(FLAGS.tb_dir))
        train_writer = tf.summary.FileWriter(FLAGS.tb_tr_path, sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.tb_te_path, sess.graph)
        sess.run([iterator.initializer for iterator in iterators])
        start = time.time()

        for epoch in range(FLAGS.epochs):
            # Iteration actions
            test_and_print = epoch % FLAGS.eval_freq == 0
            add_train_summary = epoch % 500 == 0
            train_discriminator = epoch % 1 == 0

            if test_and_print:
                samples, d_loss_test, g_loss_test, summary = sess.run(
                    [g_sample, d_loss, g_loss, summary_op],
                    feed_dict={is_train: False})
                runlogger.info("epoch [{}], during est. pass [{}]".format(
                    epoch, epoch // epochs_per_dset))
                telogger.info("D_loss {:.4}, G_loss {:.4}".format(
                    d_loss_test, g_loss_test))
                plot_save_file = FLAGS.save_folder + "/{}.png".format(
                    str(epoch//FLAGS.eval_freq).zfill(3))
                utils.plot(samples, FLAGS.num_plot, plot_save_file)
                test_writer.add_summary(summary, epoch)

            if train_discriminator:
                d_loss_curr, _, summary = sess.run(
                    [d_loss, d_train, summary_op],
                    feed_dict={is_train: True})

            g_loss_curr, _, summary = sess.run(
                [g_loss, g_train, summary_op], feed_dict={is_train: True})

            if add_train_summary:
                train_writer.add_summary(summary, epoch)

            if test_and_print:
                trlogger.info("D_loss {:.4}, G_loss {:.4}".format(
                    d_loss_curr, g_loss_curr))
                msg = utils.time_update(start, time.time())
                runlogger.info(msg)
                start = time.time()

        train_writer.close()
        test_writer.close()


def main(_):
    """Prepare everything and run training.

    Write tfrecords if necessary. Set up TensorBoard dir and image saving
    folder. Write hyperparameters to file. Run training.
    """
    if FLAGS.write_tfr:
        utils.write_tfrecords(
            FLAGS.data_dir,
            FLAGS.anno_path,
            FLAGS.tfr_tr_path,
            FLAGS.tfr_te_path,
            test_size=FLAGS.test_size)
    utils.setup_directories(
        FLAGS.tb_dir, FLAGS.tb_tr_path, FLAGS.tb_te_path, FLAGS.save_folder)
    utils.hparam_file(FLAGS.save_folder, vars(FLAGS))
    run_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-e", "--epochs", type=int, default=100000,
        help="Number of steps to train.")
    parser.add_argument(
        "-d", "--D_init_rate", type=float, default=.00002,
        help="Initial learning rate for discriminator ADAM.")
    parser.add_argument(
        "-g", "--G_init_rate", type=float, default=.00002,
        help="Initial learning rate for generator ADAM.")
    parser.add_argument(
        "-zdim", "--prior_dim", type=int, default=100,
        help="Size of random-uniform prior space.")
    parser.add_argument(
        "-b1", "--beta_1", type=float, default=0.5,
        help="Training hyperparameter.")
    parser.add_argument(
        "--batch_size", type=int, default=128)
    parser.add_argument(
        "-w", "--write_tfr", action='store_true', default=False,
        help="Rewrite tfrecords.")
    parser.add_argument(
        "-ts", "--test_size", type=float, default=0.1,
        help="Test split size.")
    parser.add_argument(
        "-f", "--eval_freq", type=int, default=1000,
        help="How many steps for an evaluation on test.")
    parser.add_argument(
        "-p", "--num_plot", type=int, default=30,
        help="Number of images to save to save_folder in eval")
    parser.add_argument(
        "-tfr", "--tfr_dir", type=str,
        default="./TFR",
        help="Path for folder containing .tfrecords")
    parser.add_argument(
        "--tfr_train", type=str, default="train",
        help="Filename for train .tfrecords from --tfr_dir.")
    parser.add_argument(
        "--tfr_test", type=str, default="test",
        help="Filename for test .tfrecords from --tfr_dir.")
    parser.add_argument(
        "-tb", "--tb_dir", type=str,
        default="./TB",
        help="Path for folder containing TensorBoard data.")
    parser.add_argument(
        "--tb_train", type=str, default="train",
        help="TensorBoard extension for training data.")
    parser.add_argument(
        "--tb_test", type=str, default="test",
        help="TensorBoard extension for validation data.")
    parser.add_argument(
        "--data_dir", type=str,
        default="./CelebaData",
        help="Path to folder of images.")
    parser.add_argument(
        "--anno_path", type=str,
        default="./list_attr_celeba.txt",
        help="Path to folder of images.")
    parser.add_argument(
        "--save_folder", type=str, default="./out",
        help="Path to visualization folder.")
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.tfr_tr_path = os.path.join(FLAGS.tfr_dir, FLAGS.tfr_train)
    FLAGS.tfr_te_path = os.path.join(FLAGS.tfr_dir, FLAGS.tfr_test)
    FLAGS.tb_tr_path = os.path.join(FLAGS.tb_dir, FLAGS.tb_train)
    FLAGS.tb_te_path = os.path.join(FLAGS.tb_dir, FLAGS.tb_test)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
