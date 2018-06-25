# encoding: utf-8

"""
Based on.. https://github.com/mirzaevinom/prostate_segmentation/blob/master/codes/augmenters.py
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center


def elastic_transform(image, alpha=100, sigma=20, random_state=None):
    """
    Elastic deformation of images as described in [Simard2003] Simard, Steinkraus and Platt,
    "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis",
    in Proc. of the International Conference on Document Analysis and Recognition, 2003.

    When shit is more than two dimensions, we gonna apply the same transformation to all dimensions
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[0:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    if len(image.shape) == 3:
        for i_dim in range(image.shape[2]):
            temp_transformed = map_coordinates(image[:, :, i_dim], indices, order=1).reshape(shape)
            image[:, :, i_dim] = temp_transformed
    else:
        image = map_coordinates(image, indices, order=1).reshape(shape)

    return image


def random_rotation(x, y, rg=15, row_index=0, col_index=1, channel_index=2, fill_mode='nearest', cval=0.):
    """

    :param x:
    :param y:
    :param rg:
    :param row_index:
    :param col_index:
    :param channel_index:
    :param fill_mode:
    :param cval:
    :return:
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y
