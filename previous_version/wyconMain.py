from wyconSRCNN import srcnn

import tensorflow.compat.v1 as tf
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 50000, "Number of epoch [6000]")
flags.DEFINE_integer('img_size', 33, "Size of the images.")
flags.DEFINE_integer('lbl_size', 21, "Size of the images.")
flags.DEFINE_integer('num_chnl', 1, "Size of the images.")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "wyconcheckpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "wyconsample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
FLAGS = flags.FLAGS


def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    srcnn()



if __name__ == '__main__':
    tf.app.run()
