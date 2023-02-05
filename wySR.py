import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import os
import time
import cv2
import numpy as np

from six.moves import xrange
from wyTool import(
    inputGen,
    readH5,
    merge,
    loadCkpt,
    saveCkpt,
    writeLoss
)

class wySR:

    def __init__(self, sess, num_chnl=1, scale=3, batch_size=128, is_train=True, cifar=False, stride=14, epoch=50000):
        self.img_size = 33
        self.lbl_size = 21

        self.sess = sess
        self.num_chnl = num_chnl
        self.scale = scale
        self.batch_size = batch_size
        self.is_train = is_train
        self.cifar = cifar
        self.stride = stride
        self.epoch = epoch
        self.model = None

        self.train_path = os.path.join(os.getcwd(), "Train")
        self.test_path = os.path.join(os.sep, (os.path.join(os.getcwd(), "Test")), "Set5")
        self.checkpoint_dir = "wycheckpoint"
        self.sample_dir = "wysample"

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lr1 = tf.Variable(1e-4, trainable=False, name='learning_rate1')
        self.lr2 = tf.Variable(1e-5, trainable=False, name='learning_rate2')


    def pred(self,img):
        # conv1
        w1 = tf.Variable(tf.random.normal([9, 9, self.num_chnl, 64], dtype=tf.float32, stddev=1e-3, name='w1'))
        b1 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32, name='b1'))
        conv1 = tf.nn.relu(tf.nn.conv2d(img, w1, strides=[1, 1, 1, 1], padding='VALID') + b1)

        # conv2
        w2 = tf.Variable(tf.random.normal([1, 1, 64, 32], dtype=tf.float32, stddev=1e-3, name='w2'))
        b2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32, name='b2'))
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='VALID') + b2)

        # conv3
        w3 = tf.Variable(tf.random.normal([5, 5, 32, self.num_chnl], dtype=tf.float32, stddev=1e-3, name='w3'))
        b3 = tf.Variable(tf.constant(0.0, shape=[self.num_chnl], dtype=tf.float32, name='b3'))
        conv3 = tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1], padding='VALID') + b3

        return conv3

    def runModel(self):
        img = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_chnl], name="input")
        lbl = tf.placeholder(tf.float32, shape=[None, self.lbl_size, self.lbl_size, self.num_chnl], name="label")

        if self.is_train:
            inputGen(self.is_train,self.scale,self.num_chnl,self.cifar,self.img_size,self.lbl_size,self.stride)
        else:
            nx, ny = inputGen(self.is_train,self.scale,self.num_chnl,self.cifar,self.img_size,self.lbl_size,self.stride)
        if self.is_train:
            dir = os.path.join('./{}'.format("wycheckpoint"), "wytrain.h5")
        else:
            dir = os.path.join('./{}'.format("wycheckpoint"), "wytest.h5")

        trainData, trainLabel = readH5(dir)

        self.model = self.pred(img)
        self.saver = tf.train.Saver(max_to_keep=5)

        tf.global_variables_initializer().run()

        ckptPath = loadCkpt(self.checkpoint_dir)
        if len(ckptPath) == 0:
            print(" [!] Load failed...")
            iteration = 0
        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(ckptPath))
            iteration = self.global_step.eval()
            print(" [*] Load SUCCESS", iteration)

        if self.is_train:
            print("Training...")

            self.loss = tf.reduce_mean(tf.square(lbl - self.model))

            '''lr1 = tf.train.exponential_decay(self.lr1, global_step=self.global_step, decay_steps=100000,
                                                  decay_rate=0.99,
                                                  staircase=True)
            lr2 = tf.train.exponential_decay(self.lr2, global_step=self.global_step, decay_steps=100000,
                                                  decay_rate=0.99,
                                                  staircase=True)'''
            var_list1 = tf.trainable_variables()[0:4]
            var_list2 = tf.trainable_variables()[4:]
            opt1 = tf.train.GradientDescentOptimizer(self.lr1)
            opt2 = tf.train.GradientDescentOptimizer(self.lr2)
            grads = tf.gradients(self.loss, var_list1 + var_list2)
            grads1 = grads[:len(var_list1)]
            grads2 = grads[len(var_list1):]
            train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
            train_op2 = opt2.apply_gradients(zip(grads2, var_list2))

            self.train_op = tf.group(train_op1, train_op2)

            start_time = time.time()
            batches = len(trainData) // self.batch_size

            for itr in xrange(self.epoch):
                ll = []

                if itr % 20 == 0:
                    state = np.random.get_state()
                    np.random.shuffle(trainData)
                    np.random.set_state(state)
                    np.random.shuffle(trainLabel)

                for batch in xrange(0,batches):
                    batchImages = trainData[batch * self.batch_size: (batch + 1) * self.batch_size]
                    batchLabels = trainLabel[batch * self.batch_size: (batch + 1) * self.batch_size]

                    iteration += 1

                    tLoss,_ = self.sess.run([self.loss,self.train_op], feed_dict={img: batchImages, lbl: batchLabels})
                    ll.append(tLoss)
                    if iteration % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % (
                            (itr + 1), iteration, time.time() - start_time, tLoss))

                    if iteration % 10000 == 0:
                        global_steps_record = tf.assign(self.global_step, iteration)
                        self.sess.run(global_steps_record)

                    '''if iteration % 100000 == 0:
                        self.lr1, self.lr2 = self.sess.run([lr1,lr2])'''

                    if iteration % 500 == 0:
                        ckptPath = saveCkpt(self.checkpoint_dir, self.num_chnl)
                        self.saver.save(self.sess, ckptPath, global_step=iteration)

                writeLoss(sum(ll)/batches)

        else:
            print("Testing...")
            output = self.model.eval({img: trainData, lbl: trainLabel})

            output = merge(output, [nx, ny])*255.
            output = output.squeeze()
            imagePath = os.path.join(os.getcwd(), self.sample_dir)
            imagePath = os.path.join(imagePath, "wycontest.png")
            cv2.imwrite(imagePath, output)

