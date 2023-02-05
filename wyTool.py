import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import os
import glob
import h5py

from six.moves import xrange

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

def createH5(is_train, input, label):

  if is_train:
    savepath = os.path.join(os.getcwd(), 'wycheckpoint/wytrain.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'wycheckpoint/wytest.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=input)
    hf.create_dataset('label', data=label)

def readH5(path):

  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def loadCkpt(checkpoint_dir, num_chnl=1):
    print(" [*] Reading checkpoints...")
    modelDir = "%s_%s" % ("wyconsrcnn.4", num_chnl)
    checkpoint_dir = os.path.join(checkpoint_dir, modelDir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      #ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      return checkpoint_dir
      #return os.path.join(checkpoint_dir, ckpt_name)
    else:
      return ""

def saveCkpt(checkpoint_dir, chnl_num=1):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("wyconsrcnn.4", chnl_num)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ckptPath = os.path.join(checkpoint_dir, model_name)

    return ckptPath

def imread(path, num_chnl):
  if num_chnl == 1:
    img = cv2.imread(path,0)

    return img
  else:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img

def makepath(is_train, cifar=False):
    if is_train:
        path = os.path.join(os.getcwd(), "Train")
    else:
        if cifar:
            path = os.path.join(os.sep, (os.path.join(os.getcwd(), "Test")), "cifar")
        else:
            path = os.path.join(os.sep, (os.path.join(os.getcwd(), "Test")), "Set5")
    return path

def makedata(path,cifar=False):
  if cifar:
    data = glob.glob(os.path.join(path, "*.png"))
  else:
    data = glob.glob(os.path.join(path, "*.bmp"))
  return data

def preprocess(path, scale,num_chnl, cifar=False):
    img = imread(path,num_chnl)

    if cifar:
        if num_chnl == 1:
            img = cv2.fastNlMeansDenoising(img,None,7,3,21).astype(np.float32)
        else:
            img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 3, 21).astype(np.float32)
        lbl = img / 255.

        lbl = modcrop(lbl, scale)

        input = cv2.resize(lbl,(0,0),fx=6/1.,fy=6/1.,interpolation=cv2.INTER_CUBIC)
        lbl = input
    else:
        lbl = img.astype(np.float32) / 255.
        img = cv2.GaussianBlur(img,(7,7),0).astype(np.float32) / 255.
        lbl = modcrop(lbl, scale)

        input = cv2.resize(img,(0,0),fx=1./3,fy=1./3,interpolation=cv2.INTER_CUBIC)
        input = cv2.resize(input,(0,0),fx=3/1.,fy=3/1.,interpolation=cv2.INTER_CUBIC)

    return input, lbl

def inputGen(is_train, scale, num_chnl, cifar=False, img_size=33, lbl_size=21, stride=14):

    path = makepath(is_train, cifar)
    data = makedata(path, cifar)

    sub_input = []
    sub_label = []
    padding = abs(img_size - lbl_size) / 2

    if is_train:
        for i in xrange(len(data)):
            input, label = preprocess(data[i], scale, num_chnl)
            if num_chnl == 3:
                h, w, _ = input.shape
            else:
                h, w = input.shape
            for x in range(0, h - img_size+1, stride):
                for y in range(0, w - img_size+1, stride):
                    subinput = input[x:x + img_size, y:y + img_size]
                    sublabel = label[x + int(padding):x + int(padding) + lbl_size,
                               y + int(padding):y + int(padding) + lbl_size]

                    subinput = subinput.reshape([img_size, img_size, 1])
                    sublabel = sublabel.reshape([lbl_size, lbl_size, 1])

                    sub_input.append(subinput)
                    sub_label.append(sublabel)

    else:
        input, label = preprocess(data[4], scale, num_chnl, cifar)

        if num_chnl == 3:
            h, w, _ = input.shape
        else:
            h, w = input.shape

        nx = ny = 0
        for x in range(0, h - img_size+1, stride):
            nx+=1
            ny=0
            for y in range(0, w - img_size+1, stride):
                ny+=1

                subinput = input[x:x + img_size, y:y + img_size]
                sublabel = label[x + int(padding):x + int(padding) + lbl_size, y + int(padding):y + int(padding) + lbl_size]

                subinput = subinput.reshape([img_size, img_size, 1])
                sublabel = sublabel.reshape([lbl_size, lbl_size, 1])

                sub_input.append(subinput)
                sub_label.append(sublabel)

    inputarray = np.asarray(sub_input)
    labelarray = np.asarray(sub_label)

    createH5(is_train, inputarray, labelarray)

    if not is_train:
        return nx,ny

def merge(patches, size):
    h, w = patches.shape[1], patches.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))

    for idx, image in enumerate(patches):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img

def writeLoss(loss):
    path = "wycheckpoint/wyconLoss.txt"
    path = os.path.join(os.getcwd(), path)
    loss = "%s" % (loss)

    with open(path, "a") as f:
        f.writelines(loss)
        f.writelines('\n')
