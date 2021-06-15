import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pdb
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
import PIL
print("Using Pillow version {}. Upgrade using 'pip install Pillow -U'".format(PIL.__version__))

from PIL import Image
import numpy as np

path="C:/Users/13595/Desktop/机器学习/AutoAugment-master/retinamnist.npz"
data=np.load(path)
print(data.files)
train = data["train_images"]
val = data["val_images"]
test = data["test_images"]
trainlabel = data["train_labels"]
valabel = data["val_labels"]
testlabel = data["test_labels"]
# print(train.shape[0],trainlabel.shape)
print(train.shape[0]+val.shape[0]+test.shape[0])
policy = ImageNetPolicy()
trainimgs = []
for i in range(train.shape[0]):
    print(train.shape[0]+val.shape[0]+test.shape[0])
    img = Image.fromarray(train[i])
    lst = []
    for _ in range(8):
        lst.append(policy(img))
        # print(policy(img))
    for k in range(8):
        ndtmp = np.array(lst[k])
        # print(type(ndtmp))
        trainimgs.append(ndtmp)
# valimgs = []
# for i in range(val.shape[0]):
#     for _ in range(8):
#         tmp = policy(Image.fromarray(val[i]))
#         ndtmp = np.array(tmp)
#         valimgs.append(ndtmp)
# testimgs = []
# for i in range(test.shape[0]):
#     for _ in range(8):
#         tmp = policy(Image.fromarray(test[i]))
#         ndtmp = np.array(tmp)
#         testimgs.append(ndtmp)
# im1 = Image.fromarray(train[0])
ntrainlabel = np.zeros(8*trainlabel.shape[0],)

#标签部分
tmparr = []
for i in range(trainlabel.shape[0]):
    for j in range(8):
        tmparr.append(int(trainlabel[i]))
# print(tmparr)
ntrainlabel = np.array(tmparr)
# print(trainlabel.shape,ntrainlabel.shape)

# nvalabel = np.zeros(8*valabel.shape[0],)

#标签部分
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
#
# t1 = Image.fromarray(np.uint8(im1))
# print(train[0]-t1)
# print(type(train[0]),type(trainimgs[6]))
# plt.imshow(train[0])
# plt.show()
# plt.imshow(trainimgs[6])
# plt.show()
# trainimgs=trainimgs,valimgs=valimgs,testimgs=testimgs,
np.savez("C:/Users/13595/Desktop/机器学习/AutoAugment-master/iretinamnist.npz",train_images=trainimgs,
         val_images=val,test_images=test,train_labels=ntrainlabel,val_labels=valabel,test_labels=testlabel)

# def show_sixteen(images, titles=0):
#     f, axarr = plt.subplots(4, 4, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
#     for idx, ax in enumerate(f.axes):
#         ax.imshow(images[idx])
#         ax.axis("off")
#         if titles: ax.set_title(titles[idx])
#     plt.show()
#
# im1 = Image.fromarray(train[10])
# im2 = Image.fromarray(train[6])
#
# imgs = []
# for _ in range(8): imgs.append(policy(im1))
# for _ in range(8): imgs.append(policy(im2))
# show_sixteen(imgs)
