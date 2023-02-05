import numpy as np
import math
import scipy.misc
import imageio
import scipy.ndimage
import cv2


from editData import(
    modcrop
)


ori = cv2.imread("./wysample/bird_GT.bmp",0).astype(np.float)
ori = modcrop(ori, 3)
con = cv2.imread("./wysample/wycontest2.png",0).astype(np.float)

input = cv2.resize(ori,(0,0),fx=1./3,fy=1./3,interpolation=cv2.INTER_CUBIC)
input = cv2.resize(input,(0,0),fx=3/1.,fy=3/1.,interpolation=cv2.INTER_CUBIC)


def psnr(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)

    img1 = img1[3:-3, 3:-3]
    img2 = img2[3:-3, 3:-3]

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    d1 = int((h1 - h2)*0.5)
    d2 = h1-h2-d1
    d3 = int((w1 - w2)*0.5)
    d4 = w1-w2-d3
    if d1 == 0:
        img1 = img1[:, :]
    else:
        img1 = img1[d1:-d2, :]
    if d2 == 0:
        img1 = img1[:, :]
    else:
        img1 = img1[:, d3:-d4]

    diff = img1 - img2
    diff = diff.flatten('C')
    mse = np.mean(diff ** 2)
    if mse ==0:
        return 100
    PIXEL_MAX = 255.0
    return 20*math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == "__main__":
    d = psnr(ori, con)
    dd = psnr(ori, input)
    print(d)
    print(dd)