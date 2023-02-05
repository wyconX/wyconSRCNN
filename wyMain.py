import wySR
import tensorflow.compat.v1 as tf
import os
from tensorflow.compat.v1 import ConfigProto

def createDir(checkpoint_dir, sample_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

def run(FLAGS,config):

    createDir(FLAGS.checkpoint_dir,FLAGS.sample_dir)
    sess = tf.InteractiveSession(config=config)

    sr = wySR.wySR(sess,FLAGS.num_chnl,FLAGS.scale,FLAGS.batch_size,FLAGS.is_train,FLAGS.cifar,FLAGS.stride,FLAGS.epoch)
    sr.runModel()

    sess.close()

if __name__ == "__main__":

    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 50000, "Number of epoch [6000]")
    flags.DEFINE_integer('img_size', 33, "Size of the images.")
    flags.DEFINE_integer('lbl_size', 21, "Size of the images.")
    flags.DEFINE_integer('num_chnl', 1, "Size of the images.")
    flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
    flags.DEFINE_string("checkpoint_dir", "wycheckpoint", "Name of checkpoint directory [checkpoint]")
    flags.DEFINE_string("sample_dir", "wysample", "Name of sample directory [sample]")
    flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
    flags.DEFINE_boolean("cifar", False, "True for upscaling cifar10, False for normal [False]")
    flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
    flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
    FLAGS = flags.FLAGS

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    config = ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True

    run(FLAGS,config)

