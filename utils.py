"""Utilities for classifier GAN."""
import os
import sys
import math
import itertools
import time
import tensorflow as tf
import numpy as np
import sklearn.model_selection as sk
from cv2 import resize
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from settings import TFR_ROWS, TFR_COLS, TFR_CHANNELS
from settings import HWC_INPUT_SHAPE, CHW_INPUT_SHAPE


def _get_image_paths(data_dir):
    """Read off the paths inside the image dir."""
    if os.path.isdir(data_dir):
        images = os.listdir(data_dir)
        image_paths = [os.path.join(data_dir, image) for image in images]
    else:
        raise ValueError("data_dir is invalid.")
    return image_paths


def _get_image_annos(anno_file):
    """Return a list of annotation lists.

    Open the annotation .txt. Read all lines. Iterate through the read lines,
    skipping two header rows. Split the line after the image extension.
    Split the annotation string on spaces. Since annotations are double-spaced,
    this adds null characters "" into the list. Add to the running anno list
    the current list, converted to int and without null characters.

    NOTE: The anno_file is sequential by filename, so we preserve ordering
    when we read filenames with ``os.listdir`` rather than from the ``.txt``.
    """
    with open(anno_file, "r") as f:
        file_annos = f.read().splitlines()
    annos = []
    for anno_and_filename in file_annos[2:]:  # Skip 2 headers
        anno_string = anno_and_filename.split(".jpg")[1]
        anno_strs_and_nulls = anno_string.split(" ")
        annos.append([
            int(anno) for anno in anno_strs_and_nulls if anno is not ""])
    return annos


def _preprocess(img):
    """Augment images before writing tfrecords."""
    img_crop = img[20:198, :]
    img_resized = resize(img_crop, (TFR_COLS, TFR_ROWS))
    img_chw = np.transpose(img_resized, [2, 0, 1])
    return img_chw


def _write_data_tfr(image_paths, annos, tfr_dir):
    """Write loaded disk_data into tfr_dir."""
    def _int64_feature(value):
        """Convert (single) value into tf.train.Feature dtype int64."""
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        """Convert (single) value into tf.train.Feature dtype bytes."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value]))

    def _int64_list_feature(value):
        """Convert list of ints into tf.train.Feature dtype int64."""
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))

    writer = tf.python_io.TFRecordWriter(tfr_dir)
    for idx, (path, anno) in enumerate(zip(image_paths, annos)):
        print("Record number: {}\r".format(idx), end="")
        disk_im = imread(path)
        preproc_im = _preprocess(disk_im)
        raw_im = preproc_im.tostring()
        # construct the Example proto-obj
        example = tf.train.Example(
            # example contains a Features proto-obj
            features=tf.train.Features(
                feature={
                    'height': _int64_feature(TFR_ROWS),
                    'width': _int64_feature(TFR_COLS),
                    'depth': _int64_feature(TFR_CHANNELS),
                    'image_raw': _bytes_feature(raw_im),
                    'anno': _int64_list_feature(anno)
                }))
        # use the proto-obj to serialize the example to a string
        serialized = example.SerializeToString()
        writer.write(serialized)


def write_tfrecords(
        data_dir, anno_path, train_tfr_dir, test_tfr_dir, test_size):
    """Write two tfrecords binary format files.

    Shuffle and split are handled internally.

    Args:
        data_dir: Path to the folder containing images.
        anno_path: Path to the annotation folder.
        train_tfr_dir: Path to the folder where training TFR will be written.
        test_tfr_dir: Path to the folder where testing TFR will be written.
        test_size: Number or fraction of images for testing.

    Implicit returns:
        Two  tfrecords file at respective *_tfr_dir. Format is ::

            features={"height", "width", "depth",
                      "image_raw", "anno"}

    """
    image_paths = _get_image_paths(data_dir)
    image_annos = _get_image_annos(anno_path)
    train_paths, test_paths, train_annos, test_annos = sk.train_test_split(
        image_paths, image_annos, test_size=test_size)
    print("Writing training records.")
    _write_data_tfr(train_paths, train_annos, train_tfr_dir)
    print("Writing testing records.")
    _write_data_tfr(test_paths, test_annos, test_tfr_dir)


def celeba_input(tfr_file, batch_size, epochs):
    """Create an iterator object to return a shuffled batch of inputs.

    Args:
        tfr_file: A string placeholder for the tfrecords filenames to use that
            iteration.
        batch_size: Used to determine size of dataset internal queue.
        epochs: Used to determine size of dataset internal queue.

    Returns:
        iterator: A ``TFRecordDataset`` object full of shuffled batches
            of images and feature annotations.

    """
    dataset = tf.data.TFRecordDataset(tfr_file)

    def parse_protocol_buffer(example_proto):
        """Map tfrecords format proto to Python format."""
        features = {'image_raw': tf.FixedLenFeature((), tf.string),
                    'anno': tf.FixedLenFeature((40), tf.int64)}
        parsed_features = tf.parse_single_example(
            example_proto, features)
        return parsed_features['image_raw'], parsed_features["anno"]

    dataset = dataset.map(parse_protocol_buffer)

    def parsed_proto_to_model_input(image_string, anno):
        """Map parsed protos to intended format."""
        image_decoded = tf.decode_raw(image_string, tf.uint8)
        image_resized = tf.reshape(
            image_decoded,
            CHW_INPUT_SHAPE)
        image = tf.cast(image_resized, tf.float32) * (2. / 255) - 1
        anno = tf.reshape(anno, [40])
        anno = tf.cast(anno, tf.float32)
        return image, anno

    dataset = dataset.map(parsed_proto_to_model_input)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat(batch_size * epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    return dataset


def factor_int(n):
    """Find integer factors closest to square."""
    max_candidate = math.ceil(math.sqrt(n))
    for candidate in range(max_candidate, 0, -1):
        if n % candidate == 0:
            break
    # n>=1 implies max_candidate>=1, so candidate is defined
    factor1, factor2 = [candidate, int(n / candidate)]
    return factor1, factor2


def prob_scale(Prob, img_name, num):
    """Add an image summary of prob; black and white keep prob from scaling."""
    prob = tf.reshape(Prob, [-1, 1, 1, 1])
    black = tf.zeros_like(prob)
    white = tf.ones_like(prob)
    scale_img = tf.concat([black, prob, white], 2, name="Probscale")
    tf.summary.image(img_name, scale_img, num)


def plot(samples, num_to_plot, save_file):
    """Make a matplotlib plot of eval images."""
    samples = np.transpose(samples, [0, 2, 3, 1])
    samples = (samples + 1)/2
    size1, size2 = factor_int(num_to_plot)
    fig = plt.figure(figsize=(size1, size2))
    gs = gridspec.GridSpec(size1, size2)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(itertools.islice(samples, num_to_plot)):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(HWC_INPUT_SHAPE))
    plt.savefig(save_file, bbox_inches="tight")
    plt.close(fig)


def setup_directories(tb_dir, train_path, test_path, save_folder):
    """Possibly overwrite and create TensorBoard directory."""
    # TensorBoard overwrite control.
    while tf.gfile.Exists(train_path) or tf.gfile.Exists(test_path):
        print("\nTensorBoard directory is already written. Overwrite?")
        print("[y] proceed, [<enter>] cancel: ")
        proc = input()
        if proc == 'y':
            try:
                tf.gfile.DeleteRecursively(tb_dir)
            except Exception as e:
                raise PermissionError(
                    "Cannot overwrite tb_dir. TensorBoard (Localhost:6006) "
                    "might be active in another terminal.")
        elif proc == "":
            sys.exit()
        else:
            print("Invalid input. Try again: ")
    tf.gfile.MakeDirs(tb_dir)
    # save folder and hparam overwrite control.
    if os.path.exists(save_folder):
        print("Directory for saving generated images is already written. "
              "Filenames of pattern ###.png will overwrite and hparam.txt "
              "will be deleted. Continue?")
        print("[y] proceed, [<enter>] cancel: ")
        proc = input()
        if proc != "y":
            sys.exit()
        elif os.path.exists(save_folder + "/hparam.txt"):
            os.remove(save_folder + "/hparam.txt")
    else:
        os.makedirs(save_folder)


def time_update(start, end):
    """Print a nice stopwatch."""
    minutes = int((end-start) // 60)
    seconds = (end-start) % 60
    msg = "Time elapsed: {}:{:05.2f}\n".format(minutes, seconds)
    return msg


def _dev_spec(dev_type):
    """Unused. Kept because it may be useful someday."""
    d_spec = tf.DeviceSpec(
        job="localhost",
        replica=0,
        task=0,
        device_type=dev_type,
        device_index=0)
    return d_spec


def hparam_file(save_folder, hparam_dict):
    """Log hyperparameters to a file in save folder."""
    with open(save_folder + "/hparam.txt", "a") as f:
        f.write("\n\n" + time.asctime())
        for key, val in hparam_dict.items():
            f.write("\n" + str(key) + ": " + str(val))