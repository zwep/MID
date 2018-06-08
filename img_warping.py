# encoding: utf-8

"""

Doing some elastic transformation on images with user-defined functions

"""
import cv2
import numpy as np
import math
import pydicom


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

img = cv2.imread('images/input.jpg', cv2.IMREAD_GRAYSCALE)

z = r'C:\Users\C35612\data\mid\PROSTATEx\ProstateX-0001\07-08-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-95738\8-ep2ddifftraDYNDISTMIXADC-33954\000011.dcm'
value_img = pydicom.read_file(z)
img = value_img.pixel_array
rows, cols = img.shape

#####################
# Vertical wave

img_output = np.zeros(img.shape, dtype=img.dtype)

for i in range(rows):
    for j in range(cols):
        offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
        offset_y = 0
        if j+offset_x < rows:
            img_output[i,j] = img[i,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0

plt.figure(1)
plt.imshow(img, cmap=plt.cm.gray)
plt.figure(2)
plt.imshow(img_output, cmap=plt.cm.gray)
plt.show()


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


plt.imshow(img, cmap=plt.cm.gray)
plt.show()
count = 0
for i_alpha in np.arange(0, 10, 1):
    for i_sgm in np.arange(0, 2, 0.2):
        count += 1
        plt.figure(count)
        img_elas = elastic_transform(img, 1, 0.2)
        plt.imshow(img_elas, cmap=plt.cm.gray)
        plt.title('alpha {0} sigm {1}'.format(i_alpha, i_sgm))
        plt.show()


test = np.zeros((300,300))
for i in np.arange(0,100,10):
    test[:, i] = 1
    test[:, i+1] = 1
for i in np.arange(5,100,10):
        test[i, :] = 1
        test[i+1, :] = 1

plt.imshow(test)
plt.show()

count = 0
for i_alpha in np.arange(1, 5, 1):
    for i_sgm in np.arange(1.2, 3, 0.2):
        count += 1
        plt.figure(count)
        img_elas = elastic_transform(test, i_alpha, i_sgm)
        plt.imshow(img_elas, cmap=plt.cm.gray)
        plt.title('alpha {0} sigm {1}'.format(i_alpha, i_sgm))
        plt.show()


img_elas = elastic_transform(test, test.shape[1] * 2, test.shape[1] * 0.08)
plt.figure(99)
plt.imshow(img_elas, cmap=plt.cm.gray)
plt.title('alpha {0} sigm {1}'.format(test.shape[1] * 2, test.shape[1] * 0.08))
plt.show()

