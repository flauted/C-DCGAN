#!/usr/bin/env python
"""Train a Classifier Deep Convolutional GAN on the Celeba dataset."""
import os
import time
import argparse
import logging
import sys
import tensorflow as tf
from tensorflow import saved_model as tfsm
from model import generator, discriminator
import utils
from utils import HEADER


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(levelname)s: %(name)s: %(asctime)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def goodfellow_loss(d_real, d_fake, eps=1.e-10):
    """Ian Goodfellow loss for GAN with small epsilon for stability."""
    with tf.name_scope("D_Eval"):
        d_loss = -tf.reduce_mean(tf.log(d_real+eps) + tf.log(1.-d_fake+eps))
        tf.summary.scalar("D_loss", d_loss)
    with tf.name_scope("G_Eval"):
        g_loss = -tf.reduce_mean(tf.log(d_fake+eps))
        tf.summary.scalar("G_loss", g_loss)
    return d_loss, g_loss


def train(init_rate, loss, var_list, beta1=0.5, name="Train"):
    """Use adam to train."""
    with tf.name_scope(name):
        train_op = tf.train.AdamOptimizer(init_rate, beta1=beta1).minimize(
            loss, var_list=var_list)
    return train_op


def inputs(is_training):
    """Construct input pipeline.

    Use command line args to set up dataset iterators with appropriate batch
    sizes and epoch counts pointing to the correct tfr path. Then use
    is_training to return the contents of the desired iterator.
    """
    with tf.name_scope("inputs"):
        with tf.name_scope("train_data"):
            train_dataset = utils.celeba_input(
                FLAGS.tfr_tr_path, FLAGS.batch_size, FLAGS.classes)
            train_iterator = train_dataset.make_initializable_iterator()
        with tf.name_scope("test_data"):
            test_dataset = utils.celeba_input(
                FLAGS.tfr_te_path, FLAGS.batch_size, FLAGS.classes)
            test_iterator = test_dataset.make_initializable_iterator()
        image, anno = tf.cond(
            is_training,
            lambda: train_iterator.get_next(),
            lambda: test_iterator.get_next(),
            name="choose_data")
    return (train_iterator, test_iterator), image, anno


def build_graph(image, anno, is_train, noise):
    """Construct the GAN graph."""
    g_img = generator(anno, noise, is_train)
    tf.summary.image("G_IMG", tf.transpose(g_img, [0, 2, 3, 1]), 1)

    with tf.variable_scope("Discriminator"):
        d_real = discriminator(image, anno, is_train)
        with tf.variable_scope("ProbSummary"):
            utils.prob_scale(d_real, "DISC_PROB_REAL", 1)
        d_fake = discriminator(g_img, anno, is_train, reuse=True)
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
    return g_img, d_loss, g_loss, d_train, g_train


def run_training():
    """Run training."""
    is_train = tf.placeholder(tf.bool, shape=[], name="is_training")
    iterators, image, anno = inputs(is_train)
    noise = tf.random_uniform(
        tf.stack([tf.shape(anno)[0], FLAGS.prior_dim]), minval=-1, maxval=1)
    g_img, d_loss, g_loss, d_train, g_train = build_graph(
        image, anno, is_train, noise)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("\n    $ tensorboard --logdir={}\n".format(FLAGS.tb_dir))
        train_writer = tf.summary.FileWriter(FLAGS.tb_tr_path, sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.tb_te_path, sess.graph)
        sess.run([itr.initializer for itr in iterators])
        start = time.time()

        epoch = 1
        minibatch_count = 0
        logger.info("[TR] Beginning training.")
        while epoch <= FLAGS.epochs:
            try:
                tr_d_loss, tr_g_loss, _, _, summary = sess.run(
                    [d_loss, g_loss, g_train, d_train, summary_op],
                    {is_train: True})
                minibatch_count += 1

            except tf.errors.OutOfRangeError:
                # End of training epoch.
                logger.info("[TR] Finished epoch %g, minibatch count at %g" %
                            (epoch, minibatch_count))
                logger.info("[TR] Last disc loss: %g, last gen loss: %g" %
                            (tr_d_loss, tr_g_loss))
                logger.info("[TR] Writing train summary.")
                train_writer.add_summary(summary, epoch)

                # Minibatch over the test set once, then break. Collect data.
                running_d = 0
                running_g = 0
                ctr = 0
                while True:
                    try:
                        ctr += 1
                        samples, te_d_loss, te_g_loss, summary = sess.run(
                            [g_img, d_loss, g_loss, summary_op],
                            {is_train: False})
                        running_d += te_d_loss
                        running_g += te_g_loss
                        # Save imgs after first test minibatch.
                        if ctr == 1:
                            logger.info("[TE] Writing images.")
                            plot_save_file = os.path.join(
                                FLAGS.save_folder,
                                "{}.png".format(str(epoch).zfill(3)))
                            utils.plot(samples, FLAGS.num_plot, plot_save_file)
                            logger.info("[TE] Writing test summary.")
                            test_writer.add_summary(summary, epoch)

                    except tf.errors.OutOfRangeError:
                        logger.info(
                            "[TE] Test completed in %g batches." % ctr)
                        logger.info(
                            "[TE] Est. cum.  disc loss: %g, gen loss: %g" %
                            (running_d/ctr, running_g/ctr))
                        sess.run([itr.initializer for itr in iterators])
                        break  # back to ``while epoch``
                logger.info(utils.time_update(start, time.time()))
                logger.info("[TR] Finished test after epoch %g" % epoch)
                epoch += 1
                start = time.time()

        if not FLAGS.no_save:
            builder = tfsm.builder.SavedModelBuilder(FLAGS.export_dir)
            sig = tfsm.signature_def_utils.build_signature_def(
                    inputs={"anno": tfsm.utils.build_tensor_info(anno),
                            "noise": tfsm.utils.build_tensor_info(noise),
                            "is_train": tfsm.utils.build_tensor_info(is_train)
                            },
                    outputs={"g_output": tfsm.utils.build_tensor_info(g_img)},
                    method_name=tfsm.signature_constants.PREDICT_METHOD_NAME)

            builder.add_meta_graph_and_variables(
                sess,
                [tfsm.tag_constants.SERVING],
                signature_def_map={
                    tfsm.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    sig})
            builder.save()
            logger.info("Saving model in {}\n\n".format(
                FLAGS.export_dir))

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
    utils.setup_tb_dir(
        FLAGS.tb_dir,
        FLAGS.tb_tr_path,
        FLAGS.tb_te_path)
    utils.output_overwrite_control(FLAGS.save_folder)
    if not FLAGS.no_save:
        # then make sure export_dir does NOT exist
        utils.export_dir_overwrite_control(FLAGS.export_dir)
    utils.hparam_file(FLAGS.save_folder, vars(FLAGS))
    run_training()  # will save model if appropriate, creating export_dir
    if not FLAGS.no_save:
        # then save classes list inside export_dir
        utils.save_config(
            FLAGS.classes,
            filename=os.path.join(FLAGS.export_dir, "classes.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-e", "--epochs", type=int, default=50,
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
        default=os.path.abspath("./TFR"),
        help="Path for folder containing .tfrecords")
    parser.add_argument(
        "--tfr_train", type=str, default="train",
        help="Filename for train .tfrecords from --tfr_dir.")
    parser.add_argument(
        "--tfr_test", type=str, default="test",
        help="Filename for test .tfrecords from --tfr_dir.")
    parser.add_argument(
        "-tb", "--tb_dir", type=str,
        default=os.path.abspath("./TB"),
        help="Path for folder containing TensorBoard data.")
    parser.add_argument(
        "--tb_train", type=str, default="train",
        help="TensorBoard extension for training data.")
    parser.add_argument(
        "--tb_test", type=str, default="test",
        help="TensorBoard extension for validation data.")
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.abspath("./CelebaData"),
        help="Path to folder of images.")
    parser.add_argument(
        "--anno_path", type=str,
        default=os.path.abspath("./list_attr_celeba.txt"),
        help="Path to folder of images.")
    parser.add_argument(
        "--save_folder", type=str, default="./out",
        help="Path to visualization folder.")
    parser.add_argument(
        "--no_save", action="store_true", default=False,
        help="Do not save model.")
    parser.add_argument(
        "--export_dir",
        type=str,
        default=os.path.abspath("./savedmodel"),
        help="Location to save .ckpt on testing improvement.")
    parser.add_argument(
        "--classes",
        type=str,
        default="Male Blond_Hair",
        help="One string of options separated with spaces: " + HEADER)
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.tfr_tr_path = os.path.join(FLAGS.tfr_dir, FLAGS.tfr_train)
    FLAGS.tfr_te_path = os.path.join(FLAGS.tfr_dir, FLAGS.tfr_test)
    FLAGS.tb_tr_path = os.path.join(FLAGS.tb_dir, FLAGS.tb_train)
    FLAGS.tb_te_path = os.path.join(FLAGS.tb_dir, FLAGS.tb_test)
    FLAGS.classes = FLAGS.classes.split(" ")
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
