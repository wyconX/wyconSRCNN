import os
import glob
import h5py
import scipy.misc
import imageio
import scipy.ndimage
import numpy as np
import tensorflow.compat.v1 as tf
import cv2


from six.moves import xrange
FLAGS = tf.app.flags.FLAGS


def modcrop(image, scale=3):

  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def createH5(input, label):

  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'wyconcheckpoint/wytrain.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'wyconcheckpoint/wytest.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=input)
    hf.create_dataset('label', data=label)

def readH5(path):

  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def loadCkpt(checkpoint_dir):
    print(" [*] Reading checkpoints...")
    modelDir = "%s_%s" % ("wyconsrcnn.1", FLAGS.lbl_size)
    checkpoint_dir = os.path.join(checkpoint_dir, modelDir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        return os.path.join(checkpoint_dir, ckpt_name)
    else:
        return ""

def saveCkpt(checkpoint_dir):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("wyconsrcnn.1", FLAGS.lbl_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ckptPath = os.path.join(checkpoint_dir, model_name)

    return ckptPath

def merge(images, size):

  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img

def inputGen(sess, FLAGS):

    if FLAGS.is_train:
        data_dir = os.path.join(os.getcwd(), "Train")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), "Test")), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(FLAGS.img_size - FLAGS.lbl_size) / 2 # 6

    if FLAGS.is_train:

        for i in xrange(len(data)):
            img = imageio.imread(data[i], as_gray=True, pilmode='YCbCr').astype(np.float)
            img = modcrop(img, FLAGS.scale)
            #Normalize
            lbl = img / 255.
            #bicubic
            input = scipy.ndimage.interpolation.zoom(lbl, (1./FLAGS.scale), prefilter=False)
            input = scipy.ndimage.interpolation.zoom(input, (FLAGS.scale/1.), prefilter=False)

            if len(input.shape) == 3:
                h, w, _ = input.shape
            else:
                h, w = input.shape

            for x in range(0, h - FLAGS.img_size + 1, FLAGS.stride):
                for y in range(0, w - FLAGS.img_size + 1, FLAGS.stride):
                    sub_input = input[x:x + FLAGS.img_size, y:y + FLAGS.img_size]  # [33 x 33]
                    sub_label = lbl[x + int(padding):x + int(padding) + FLAGS.lbl_size, y + int(padding):y + int(padding) + FLAGS.lbl_size]  # [21 x 21]

                    # Make channel value
                    sub_input = sub_input.reshape([FLAGS.img_size, FLAGS.img_size, 1])
                    sub_label = sub_label.reshape([FLAGS.lbl_size, FLAGS.lbl_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

    else:
        img = imageio.imread(data[2], as_gray=True, pilmode='YCbCr').astype(np.float)
        img = modcrop(img, FLAGS.scale)
        #Normalize
        #img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        lbl = img / 255.

        #bicubic
        input = scipy.ndimage.interpolation.zoom(lbl, (1./FLAGS.scale), prefilter=False)
        input = scipy.ndimage.interpolation.zoom(input, (FLAGS.scale/1.), prefilter=False)
        #input = scipy.ndimage.interpolation.zoom(lbl, (6/1.), prefilter=False)
        #lbl=input

        if len(input.shape) == 3:
            h, w, _ = input.shape
        else:
            h, w = input.shape

        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0
        for x in range(0, h-FLAGS.img_size+1, FLAGS.stride):
            nx += 1; ny = 0
            for y in range(0, w-FLAGS.img_size+1, FLAGS.stride):
                ny += 1
                sub_input = input[x:x+FLAGS.img_size, y:y+FLAGS.img_size] # [33 x 33]
                sub_label = lbl[x+int(padding):x+int(padding)+FLAGS.lbl_size, y+int(padding):y+int(padding)+FLAGS.lbl_size] # [21 x 21]

                sub_input = sub_input.reshape([FLAGS.img_size, FLAGS.img_size, 1])
                sub_label = sub_label.reshape([FLAGS.lbl_size, FLAGS.lbl_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    inputarray = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    labelarray = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

    createH5(inputarray, labelarray)

    if not FLAGS.is_train:
        return nx, ny
