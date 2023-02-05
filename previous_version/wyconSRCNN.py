
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import os
import cv2
import time
import scipy.misc
import imageio

from PIL import Image as im
from tensorflow.compat.v1 import ConfigProto


from six.moves import xrange
from editData import(
    inputGen,
    readH5,
    loadCkpt,
    saveCkpt,
    merge
)
FLAGS = tf.app.flags.FLAGS


def weight_variable(shape, n):
    initial = tf.random_normal(shape, dtype=tf.float32, stddev=1e-3, name=n)
    return tf.Variable(initial)


def bias_variable(shape, n):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32, name=n)
    return tf.Variable(initial)


def conv2d(x, W):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    return conv

def predict(img):
    # conv1
    w1 = weight_variable([9, 9, FLAGS.num_chnl, 64],'w1')
    b1 = bias_variable([64],'b1')
    conv1 = tf.nn.relu(conv2d(img, w1) + b1)

    # conv2
    w2 = weight_variable([1, 1, 64, 32],'w2')
    b2 = bias_variable([32],'b2')
    conv2 = tf.nn.relu(conv2d(conv1, w2) + b2)

    # conv3
    w3 = weight_variable([5, 5, 32, 1],'w3')
    b3 = bias_variable([1],'b3')
    conv3 = conv2d(conv2, w3) + b3

    return conv3

def train():

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 选择哪一块gpu
    config = ConfigProto()
    config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True  # 按需分配显存，这个比较重要'
    sess = tf.InteractiveSession(config=config)

    img = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, FLAGS.num_chnl], name="input")
    lbl = tf.placeholder(tf.float32, shape=[None, FLAGS.lbl_size, FLAGS.lbl_size, FLAGS.num_chnl], name="label")

    pred = predict(img)
    loss = tf.reduce_mean(tf.square(lbl - pred))

    inputGen(sess, FLAGS)
    dir = os.path.join('./{}'.format("wyconcheckpoint"), "wytrain.h5")

    trainData, trainLabel = readH5(dir)

    var_list1 = tf.trainable_variables()[0:4]
    var_list2 = tf.trainable_variables()[4:]
    opt1 = tf.train.GradientDescentOptimizer(1e-4)
    opt2 = tf.train.GradientDescentOptimizer(1e-5)
    grads = tf.gradients(loss, var_list1+var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))

    train_op = tf.group(train_op1, train_op2)

    saver = tf.train.Saver()

    tf.initialize_all_variables().run()

    iteration = 0
    start_time = time.time()

    ckptPath = loadCkpt(FLAGS.checkpoint_dir)
    if len(ckptPath) == 0:
        print(" [!] Load failed...")
    else:
        saver.restore(sess, ckptPath)
        print(" [*] Load SUCCESS")

    print("Training...")

    for itr in xrange(FLAGS.epoch):
        batches = len(trainData) // FLAGS.batch_size
        for batch in xrange(0, batches):
            batchImages = trainData[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
            batchLabels = trainLabel[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]

            iteration += 1
            _, tLoss = sess.run([train_op, loss], feed_dict={img: batchImages, lbl: batchLabels})

            if iteration % 10 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % (
                (itr + 1), iteration, time.time() - start_time, tLoss))

            if iteration % 500 == 0:
                ckptPath = saveCkpt(FLAGS.checkpoint_dir)
                saver.save(sess, ckptPath, global_step=iteration)

    sess.close()



def test():

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 选择哪一块gpu
    config = ConfigProto()
    config.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    config.gpu_options.allow_growth = True  # 按需分配显存，这个比较重要'
    sess = tf.InteractiveSession(config=config)

    img = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, FLAGS.num_chnl], name="input")
    lbl = tf.placeholder(tf.float32, shape=[None, FLAGS.lbl_size, FLAGS.lbl_size, FLAGS.num_chnl], name="label")

    nx, ny = inputGen(sess, FLAGS)

    pred = predict(img)

    dir = os.path.join('./{}'.format("wyconcheckpoint"), "wytest.h5")
    trainData, trainLabel = readH5(dir)

    saver = tf.train.Saver()

    ckptPath = loadCkpt(FLAGS.checkpoint_dir)
    if len(ckptPath) == 0:
        print(" [!] Load failed...")
    else:
        saver.restore(sess, ckptPath)
        print(" [*] Load SUCCESS")

    print("Testing...")

    output = pred.eval({img: trainData, lbl: trainLabel})
    output = merge(output, [nx, ny])
    output = output.squeeze()

    imagePath = os.path.join(os.getcwd(), FLAGS.sample_dir)
    imagePath = os.path.join(imagePath, "wycontemp2.png")
    imageio.imwrite(imagePath, output)

    sess.close()

def srcnn():
    if FLAGS.is_train:
        train()
    else:
        test()
