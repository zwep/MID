# encoding: utf-8

"""
Here we are trying the simple-unet architecture proposed by

https://github.com/mirzaevinom/prostate_segmentation

where we have downloaded his weights...
C:\Users\C35612\data\mid\simple_unet_weights.h5

# Amazing source for dicom images
https://dicom.innolitics.com/ciods
"""

import os
import numpy as np
import pydicom

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage.transform
from models.model_unet import simple_unet, actual_unet
from skimage.exposure import equalize_hist

from PIL import Image
import cv2
from keras import backend as K


def pred_model(input_img, n_x=96, n_y=96):
    img_resized = skimage.transform.resize(input_img, (n_x, n_y))
    img_equalized = equalize_hist(img_resized)
    c1 = np.reshape(img_equalized, (1, n_x, n_y, 1))
    res_c1 = model.predict(c1)
    res_d1 = np.reshape(res_c1, (n_x, n_y))
    return res_d1


dir_img = r'C:\Users\C35612\data\mid\PROSTATEx'
# Load model
model = simple_unet(96, 96)
model.load_weights(r'C:\Users\C35612\data\mid\simple_unet_weights.h5')


# Get certain images...
res_info = []
max_count = 5
count = 0
for i in os.walk(dir_img):
    if len(i[2]):
        if 't2tsetra' in i[0]:
            count += 1
            temp = []
            for i_file in i[2]:
                img_path = os.path.join(i[0], i_file)
                print(count, img_path)
                i_img = pydicom.read_file(img_path)
                temp.append(i_img)
            res_info.append(temp)
    if count > max_count:
        break

plt.show()

z = res_info[0]

# Trying out the model with their example...
github_example = Image.open(r'C:\Users\C35612\data\mid\mirzaevinom_example.png').convert('LA')
github_example = np.array(github_example)[:,:,0]
res = pred_model(github_example)
plt.imshow(res, cmap=plt.cm.gray)
plt.show()

# ... and my own example as well
my_example = Image.open(r'C:\Users\C35612\data\mid\my_example.jpg').convert('LA')
my_example = np.array(my_example)[:,:,0]
res = pred_model(my_example)
plt.imshow(res, cmap=plt.cm.gray)
plt.show()


# Looks good, let's continue to score the batch of images we just got

print(len(res_info))

res_img = [x.pixel_array for x in res_info[0]]
pred_img = [pred_model(x) for x in res_img]

for i, x in enumerate(res_img):
    try:
        plt.figure(i)
        plt.subplot(1,2,1)
        plt.imshow(x)
        plt.subplot(1,2,2)
        plt.imshow(pred_img[i])
        plt.show()
        plt.pause(0.05)
        input()
    except KeyboardInterrupt:
        break


# Analyze the hidden layers of the u-net model...
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function

layer_outs = functor([c1, 1.])
print(len(layer_outs))

# This is the input itself..
x1 = layer_outs[0]
print(x1)
print(x1.shape)

def fun_x(x, n=96, l=8):
    x2 = np.reshape(x1, (n, n, l))
    for i in range(l):
        plt.figure(i)
        plt.imshow(x2[:,:,i], cmap=plt.cm.gray)
        plt.show()

# This is after the first layer..
# i = 1,2 are the same...
x1 = layer_outs[1]
print(x1)
print(x1.shape)
fun_x(x1)

x1 = layer_outs[3]
print(x1.shape)
fun_x(x1, 48)

x1 = layer_outs[-2]
print(x1.shape)
fun_x(x1, 96)

x1 = layer_outs[-4]
print(x1.shape)
fun_x(x1, 96, 16)

x1 = layer_outs[-5]
print(x1.shape)
fun_x(x1, 96, 8)

x1 = layer_outs[-6]
print(x1.shape)
fun_x(x1, 96, 16)
