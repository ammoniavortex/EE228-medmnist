import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pdb
import cv2
# from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
import PIL
print("Using Pillow version {}. Upgrade using 'pip install Pillow -U'".format(PIL.__version__))

from PIL import Image
import numpy as np

# path="../input/breastmnist.npz"
path="./input/breastmnist.npz"
data=np.load(path)
print(data.files)
train = data["train_images"]
val = data["val_images"]
test = data["test_images"]
trainlabel = data["train_labels"]
valabel = data["val_labels"]
testlabel = data["test_labels"]


# Shifting Left
def shfleft(img,distance):
    HEIGHT = WIDTH = 28
    for i in range(HEIGHT, 1, -1):
        for j in range(WIDTH):
            if (i < HEIGHT-distance):
                img[j][i] = img[j][i-distance]
            elif (i < HEIGHT-1):
                img[j][i] = 0
    return img
# Shifting right
def shfright(img,distance):
    HEIGHT = WIDTH = 28
    for j in range(WIDTH):
        for i in range(HEIGHT):
            if (i < HEIGHT-distance):
                img[j][i] = img[j][i+distance]
    return img

# Shifting Up
def shfup(img,distance):
    HEIGHT = WIDTH = 28
    for j in range(WIDTH):
        for i in range(HEIGHT):
            if (j < WIDTH - distance and j > distance):
                img[j][i] = img[j + distance][i]
            else:
                img[j][i] = 0
    return img
#Shifting Down
def shfdown(img,distance):
    HEIGHT = WIDTH = 28
    for j in range(WIDTH, 1):
        for i in range(HEIGHT):
            if (j < WIDTH - distance and j > distance):
                img[j][i] = img[j - distance][i]
            else:
                img[j][i] = 0
    return img


def rotate_img(img, angle):
    (height, width) = img.shape[:2]
    center = (height//2, width//2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 旋转图像
    rotate_img = cv2.warpAffine(img, matrix, (width,height))
    return rotate_img

def gnoise(img):
    HEIGHT = WIDTH = 28
    # DEPTH = img.shape[2]
    noise = np.random.randint(5, size=img.shape, dtype='uint8')

    for i in range(WIDTH):
        for j in range(HEIGHT):
            # for k in range(DEPTH):
                if (img[i][j] != 255):
                    img[i][j] += noise[i][j]
    return img

trainimgs = []
for i in range(train.shape[0]):
    # print(train[i].shape)
    img = train[i]
    flp = np.fliplr(img)
    left = shfleft(img,3)
    right = shfright(img,3)
    up = shfup(img,3)
    # print(img.shape)
    down = shfdown(img,3)
    # print(img.shape)
    r1 = rotate_img(img,30)
    r2 = rotate_img(img,110)
    g = gnoise(img)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(down)
    # plt.show()
    # plt.imshow(r1)
    # plt.show()
    # plt.imshow(g)
    # plt.show()
    trainimgs.append(img)
    trainimgs.append(flp)
    trainimgs.append(left)
    trainimgs.append(right)
    trainimgs.append(up)
    trainimgs.append(r1)
    trainimgs.append(r2)
    trainimgs.append(g)
    # print(trainimgs)

# valimgs = []
# # print(valimgs)
# for i in range(val.shape[0]):
#     img = val[i]
#     flp = np.fliplr(img)
#     left = shfleft(img,3)
#     right = shfright(img,3)
#     up = shfup(img,3)
#     # print(img.shape)
#     down = shfdown(img,3)
#     # print(img.shape)
#     r1 = rotate_img(img,30)
#     r2 = rotate_img(img,110)
#     g = gnoise(img)
#     valimgs.append(img)
#     valimgs.append(flp)
#     valimgs.append(left)
#     valimgs.append(right)
#     valimgs.append(up)
#     valimgs.append(r1)
#     valimgs.append(r2)
#     valimgs.append(g)

# testimgs = []
# for i in range(test.shape[0]):
#     img = test[i]
#     flp = np.fliplr(img)
#     left = shfleft(img,3)
#     right = shfright(img,3)
#     up = shfup(img,3)
#     # print(img.shape)
#     down = shfdown(img,3)
#     # print(img.shape)
#     r1 = rotate_img(img,30)
#     r2 = rotate_img(img,110)
#     g = gnoise(img)
#     testimgs.append(img)
#     testimgs.append(flp)
#     testimgs.append(left)
#     testimgs.append(right)
#     testimgs.append(up)
#     testimgs.append(r1)
#     testimgs.append(r2)
#     testimgs.append(g)

ntrainlabel = np.zeros(8*trainlabel.shape[0],)

#标签部分
print(train.shape[0],trainlabel.shape[0])
tmparr = []
for i in range(trainlabel.shape[0]):
    for j in range(8):
        # print(trainlabel[i])
        tmparr.append(int(trainlabel[i]))
        #tmparr.append(trainlabel[i])
# print(tmparr)
ntrainlabel = np.array(tmparr)
# print(trainlabel.shape,ntrainlabel.shape)

# nvalabel = np.zeros(8*valabel.shape[0],)

# #标签部分
# tmparr = []
# for i in range(valabel.shape[0]):
#     for j in range(8):
#         tmparr.append(int(valabel[i]))
# # print(tmparr)
# nvalabel = np.array(tmparr)
#
# ntestlabel = np.zeros(8*testlabel.shape[0],)
#
# #标签部分
# tmparr = []
# for i in range(testlabel.shape[0]):
#     for j in range(8):
#         tmparr.append(int(testlabel[i]))
# # print(tmparr)
# ntestlabel = np.array(tmparr)

np.savez("./newoutput/breastmnist.npz",
         train_images=trainimgs,
         val_images=val,test_images=test,
         train_labels=ntrainlabel,val_labels=valabel,test_labels=testlabel)
