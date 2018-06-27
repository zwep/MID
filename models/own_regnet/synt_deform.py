# encoding: utf-8

import pydicom
import numpy as np
import os

import os, time, sys, os.path
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
import submodules.RegNet.Functions.PyFunctions as PF

from settings.config import *


def create_dvf(input_shape, dim, border, max_deform, n_p):
    """
    Here we create a random deformation field... based on the shape and input parameters

    :param input_shape:
    :param dim:
    :param border:
    :param max_deform:
    :param n_p:
    :return:
    """

    # CREATE THE DVF FIELD
    border_mask = np.zeros(input_shape)
    if dim == 2:
        border_mask[border:input_shape[0]-border+1,border:input_shape[1]-border+1] = 1
    elif dim == 3:
        border_mask[border:input_shape[0]-border+1,border:input_shape[1]-border+1,border:input_shape[2]-border+1] = 1

    # Declare deformation field for three dimensions...
    dvf = [np.zeros(input_shape, dtype=np.float64) for x in range(3)]
    i = 0
    index_edge = np.where((border_mask > 0) )

    # Dont understand this critera yet...
    while (len(index_edge[0]) > 4) & (i < n_p):
        selectVoxel = int(np.random.randint(0, len(index_edge[0]) - 1, 1, dtype=np.int64))

        # I dont understnad why this is reversed...
        z = index_edge[0][selectVoxel]
        y = index_edge[1][selectVoxel]
        x = index_edge[2][selectVoxel]

        # Generate deformation parameters
        # Could add an if-statement to set the z-axis
        d_deform = [((np.random.ranf([1]))[0] - 0.5) * max_deform * 2 for x in range(3)]

        # Insert the deformation parameters
        for i_x, i_deform in zip(dvf, d_deform):
            i_x[z, y, x] = i_deform

        i += 1

    return dvf


def distort_dvf(input_dvf, sigma_b):
    """
    Here we modify the DVF by... applying a Gaussian filter and a normalization step

    :param input_dvf: input distortion field
    :param sigma_b: sigma parameter of gaussian filter
    :return:
    """

    dvf_b = [gaussian_filter(x, sigma=sigma_b) for x in input_dvf]
    index_p = [np.where(x > 0) for x in dvf_b]
    index_n = [np.where(x < 0) for x in dvf_b]

    # In the following code, we linearly normalize the DVF for negative and positive values.
    # Please note that if normalization is done for all values, then a shift can occur which leads
    # too many nonzero values
    for xb, i_p, i_n in zip(dvf_b, index_p, index_n):
        xb[i_p] = ((np.max(xb) - 0) / (np.max(xb[i_p]) - np.min(xb[i_p])) * (xb[i_p] - np.min(xb[i_p])) + 0)
        xb[i_n] = ((0 - np.min(xb[i_n])) / (0 - np.min(xb[i_n])) * (xb[i_n] - np.min(xb[i_n])) + np.min(xb[i_n]))

    return dvf_b


def create_dvf_area(input_shape, dim, border, max_deform, n_p, distance_deform):
    """

    :param input_shape: the shape.. (x, y)
    :param dim: dimension of the image (2D or 3D)
    :param border: border of the image
    :param max_deform: max deform region
    :param n_p: amount of pixels that we are going to move
    :param distance_deform:
    :return:
    """

    border_mask = np.zeros(input_shape)
    if dim == 2:
        border_mask[border:input_shape[0]-border+1,border:input_shape[1]-border+1] = 1
    elif dim == 3:
        border_mask[border:input_shape[0]-border+1,border:input_shape[1]-border+1,border:input_shape[2]-border+1] = 1

    # Declare deformation field for three dimensions...
    dvf = [np.zeros(input_shape, dtype=np.float64) for x in range(3)]
    i = 0
    index_edge = np.where((border_mask > 0) )

    while (len(index_edge[0]) > 4) & (i < n_p):
        selectVoxel = int(np.random.randint(0, len(index_edge[0]) - 1, 1, dtype=np.int64))

        # I dont understnad why this is reversed...
        z = index_edge[0][selectVoxel]
        y = index_edge[1][selectVoxel]
        x = index_edge[2][selectVoxel]
        zyx = [z, y, z]

        # Generate deformation parameters
        # Could add an if-statement to set the z-axis
        d_deform = [((np.random.ranf([1]))[0] - 0.5) * max_deform * 2 for x in range(3)]

        # Insert the deformation parameters
        for i_x, i_deform in zip(dvf, d_deform):
            i_x[z, y, x] = i_deform

        zyx_d_min = [max(i - distance_deform, 0) for i in zyx]
        zyx_d_max = [min(i + distance_deform, input_shape[i] - 1) for i, x in enumerate(zyx)]

        # Here we update the border_mask and index_edge in the while loop itself.
        border_mask[zyx_d_min[0]:zyx_d_max[0], zyx_d_min[1]:zyx_d_max[1], zyx_d_min[2]:zyx_d_max[2]] = 0
        index_edge = np.where(border_mask > 0)
        i += 1

    return dvf


def deform_image(input_path, dim=2, border=33, max_deform=40, n_p=100, sigma_b=35, id_dvf_area=False, distance_deform=None):
    """
    Here we can deform an image... which is being read.. deformed.. and written again...

    But I dont understand how the Area is used at this moment...
    :return:
    """
    # Read the image
    input_image = sitk.ReadImage(input_path)
    # Get the values..
    temp_value = sitk.GetArrayFromImage(input_image)
    temp_value = temp_value[0, :, :]

    # No idea how to handle the multi dimensional thing...
    temp_value = np.reshape(temp_value, temp_value.shape + tuple({1}))
    temp_shape = temp_value.shape

    # Here we create a DVF and DVFb....
    if id_dvf_area:
        dvf = create_dvf_area(temp_shape, dim, border, max_deform, n_p, distance_deform)
    else:
        dvf = create_dvf(temp_shape, dim, border, max_deform, n_p)

    dvfb = distort_dvf(dvf, sigma_b)

    return dvfb


def write_images(input_path, dvfb, sigma_n=5):
    """
    We create a separate function to handle all the writing...

    :return:
    """
    input_image = sitk.ReadImage(input_path)

    idx = re.findall('(ProstateX-[0-9]{4})', input_path)[0]
    id_seriex = re.findall('([0-9]{6})\.dcm', input_path)[0]

    # File operational stuff...
    file_dvf = os.path.join(os.path.dirname(input_path), 'DVF' + id_seriex + '.dcm')
    file_mvd = os.path.join(os.path.dirname(input_path), 'MVD' + id_seriex + '.dcm')

    temp_orig = input_image.GetOrigin()

    sitk_dvfb = sitk.GetImageFromArray(dvfb, isVector=True)
    # Not sure what this exactly does
    sitk_dvfb.SetOrigin(temp_orig)

    # Write image...
    sitk.WriteImage(sitk.Cast(sitk_dvfb, sitk.sitkVectorFloat32), file_dvf)

    # ! After this line you cannot save DeformedDVF any more
    dvf_t = sitk.DisplacementFieldTransform(np.array(sitk_dvfb))



    # This is the clean version of the deformed image. Intensity noise should be added to this image
    deformed_image = sitk.Resample(input_image, dvf_t)
    deformed_image = sitk.AdditiveGaussianNoise(deformed_image, sigma_n, 0, 0)
    deformed_image = sitk.GetArrayFromImage(deformed_image)
    sitk.WriteImage(sitk.Cast(deformed_image, sitk.sitkInt16), file_mvd)

