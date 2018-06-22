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
    while ((len(index_edge[0]) > 4) & (i < n_p)):
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
    Here we modify the DVF by... applying a Gaussian filter and a normalization
    :param input_dvf:
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

def create_dva(self):

    # CREATE THE DVA
    border_mask = np.zeros(input_shape)
    if dim == 2:
        border_mask[border:input_shape[0]-border+1,border:input_shape[1]-border+1] = 1
    elif dim == 3:
        border_mask[border:input_shape[0]-border+1,border:input_shape[1]-border+1,border:input_shape[2]-border+1] = 1

    # Declare deformation field for three dimensions...
    dva = np.zeros(input_shape)
    i = 0
    index_edge = np.where((border_mask > 0) )

    # Dont understand this critera yet...
    while ((len(index_edge[0]) > 4) & (i < n_p)):
        selectVoxel = int(np.random.randint(0, len(index_edge[0]) - 1, 1, dtype=np.int64))

        # I dont understnad why this is reversed...
        z = index_edge[0][selectVoxel]
        y = index_edge[1][selectVoxel]
        x = index_edge[2][selectVoxel]
        zyx = [z, y, z]

        zyx_d_min = [max(i - DistanceDeform, 0) for i in zyx]
        zyx_d_max = [min(i + DistanceDeform, input_shape[i] - 1) for i, x in enumerate(zyx)]

        zyx_A_min = [max(i - DistanceArea, 0) for i in zyx]
        zyx_A_max = [min(i + DistanceArea, input_shape[i] - 1) for i, x in enumerate(zyx)]

        # Okay so he makes an exception here for a 2D case... remember this for later..
        # if (Dim == '2D'):
        #     zminA = z - DistanceArea
        #     zmaxA = z + DistanceArea
        # else:
        #     zminA = z - 1
        #     zmaxA = z + 2  # This is exclusively for 2D !!!!

        border_mask[zyx_d_min[0]:zyx_d_max[0], zyx_d_min[1]:zyx_d_max[1], zyx_d_min[2]:zyx_d_max[2]] = 0
        dva[zyx_A_min[0]:zyx_A_max[0], zyx_A_min[1]:zyx_A_max[1], zyx_A_min[2]:zyx_A_max[2]] = 1
        index_edge = np.where(border_mask > 0)
        i += 1

    return dva

def deform_image(input_path):
    """
    Here we can deform an image... which is being read.. deformed.. and written again...

    But I dont understand how the Area is used at this moment...
    :return:
    """
    max_deform = 0.5
    n_p = 100
    sigma_b = 3
    sigma_n = 2  # no idea...
    border = 10
    dim = 2

    # File operational stuff...
    file_dvf = os.path.join(os.path.dirname(input_path), 'DVF' + idx + '.mha')
    file_mvd = os.path.join(os.path.dirname(input_path), 'MVD' + idx + '.mha')

    input_image = sitk.ReadImage(input_path)
    temp_orig = input_image.GetOrigin()

    temp_value = sitk.GetArrayFromImage(input_image)
    temp_value = temp_value[0, :, :]

    # No idea how to handle the multi dimensional thing...
    temp_value = np.reshape(temp_value, temp_value.shape + tuple({1}))
    temp_shape = temp_value.shape

    # Here we create a DVF and DVFb....
    dvf = create_dvf(temp_shape, dim, border, max_deform, n_p)
    dvfb = distort_dvf(dvf, sigma_b)

    sitk_dvfb = sitk.GetImageFromArray(dvfb, isVector=True)
    # Not sure what this exactly does
    sitk_dvfb.SetOrigin(temp_orig)

    # Write image...
    sitk.WriteImage(sitk.Cast(sitk_dvfb, sitk.sitkVectorFloat32), file_dvf)

    # ! After this line you cannot save DeformedDVF any more
    dvf_t = sitk.DisplacementFieldTransform(sitk_dvfb)

    # This is the clean version of the deformed image. Intensity noise should be added to this image
    deformed_image = sitk.Resample(input_image, dvf_t)
    deformed_image = sitk.AdditiveGaussianNoise(deformed_image, sigma_n, 0, 0)
    deformed_image = sitk.GetArrayFromImage(deformed_image)
    sitk.WriteImage(sitk.Cast(deformed_image, sitk.sitkInt16), file_mvd)

    # Here we create some Deformed Area... not sure what this hsould be...
    if False:
        dva = create_dva()
        dvf = create_dvf(temp_shape, dim, border, max_deform, n_p)
        dvfb = distort_dvf(dvf, sigma_b)
        DeformedArea = sitk.GetImageFromArray(dva)
        DeformedArea.SetOrigin(input_im.GetOrigin())
        DeformedArea.SetSpacing(input_im.GetSpacing())
        sitk.WriteImage(DeformedArea, 'somename.mha')
    return 1

