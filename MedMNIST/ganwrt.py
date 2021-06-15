import cv2
import argparse
import os
import numpy as np
import math
from PIL import Image
path="./input/breastmnist.npz"
data=np.load(path)
print(data.files)
train = data["train_images"]
val = data["val_images"]
test = data["test_images"]
trainlabel = data["train_labels"]
valabel = data["val_labels"]
testlabel = data["test_labels"]
print(train.shape)
ntrain = train.tolist()



for i in range(300):
    total_pic_path = ('./breast-0-images' + '/' + str(i) +'.png')
    img = Image.open(total_pic_path)  # 打开图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    img_array = np.array(reIm.convert('L'))
    ntrain.append(img_array)





for i in range(750):
    total_pic_path = ('./breast-1-images' + '/' + str(i) +'.png')
    img = Image.open(total_pic_path)  # 打开图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    img_array = np.array(reIm.convert('L'))
    ntrain.append(img_array)

ntrainlabel = np.zeros(trainlabel.shape[0]+1050,)
#标签部分
tmparr = []
for i in range(trainlabel.shape[0]):
        tmparr.append(int(trainlabel[i]))
for i in range(300):
    tmparr.append(0)
for i in range(750):
    tmparr.append(1)
# print(tmparr)
ntrainlabel = np.array(tmparr)

train = np.array(ntrain)


print(train.shape)
print(ntrainlabel)

np.savez("./ganoutput/breastmnist.npz",
         train_images=train,
         val_images=val,test_images=test,
         train_labels=ntrainlabel,val_labels=valabel,test_labels=testlabel)