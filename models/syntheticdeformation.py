# encoding: utf-8

import pydicom
import numpy as np
import os

import os, time, sys, os.path
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
import submodules.RegNet.Functions.PyFunctions as PF

from settings.config import *

idx = 'ProstateX-0169'
i_path = os.path.join(DIR_IMG, idx)
i_image_series = []
for i_dir, _, i_file in os.walk(os.path.join(DIR_IMG, idx)):
    temp_list = []
    if len(i_file):
        temp_list = [os.path.join(i_dir, i) for i in i_file]
        i_image_series.append(temp_list)

temp = i_image_series[0][0]



# Now create two paths where there DVF and the changed thing can reside...

self._DeformedDVF_ = np.zeros([self._FixedIm_.shape[0], self._FixedIm_.shape[1], self._FixedIm_.shape[2], 3], dtype=np.float64)

DeformedDVF = sitk.GetImageFromArray(self._DeformedDVF_, isVector=True)
DeformedDVF.SetOrigin(self._FixedIm.GetOrigin())
sitk.WriteImage(sitk.Cast(DeformedDVF, sitk.sitkVectorFloat32), DeforemdDVFAddress)
DVF_T = sitk.DisplacementFieldTransform(DeformedDVF)    # After this line you cannot save DeformedDVF any more !!!!!!!!!
DefomedImClean = sitk.Resample(self._FixedIm, DVF_T)          # This is the clean version of the deformed image. Intensity noise should be added to this image
DefomedIm = sitk.AdditiveGaussianNoise(DefomedImClean, self._setting['sigmaN'], 0, 0)
self._DefomedIm_=sitk.GetArrayFromImage(DefomedIm)
sitk.WriteImage(sitk.Cast(DefomedIm, sitk.sitkInt16), DeforemdImAddress)

# perform smooth() or blob()

def create_dvf(input_shape, dim, border, max_deform, n_p):
    """
    Here we create a random deformation field...
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


def create_dva(input_shape, dim, border, max_deform, n_p):
    """
    Creates a deformation vector area...

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


def deform_image(input_path):
    """
    Here we can deform an image... which is being read.. deformed.. and written again
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

    return 1


def blob(self):
    Dsmooth = self._Dsmooth
    D = self._D
    DeformTotNames=[self._DeformName+'F',self._DeformName+'M']
    typeTot=['Fixed','Moving']
    DeformFolder=DeformTotNames[self._ImageType]
    DeformPath=self._setting['DLFolder']+'Elastix/'+DeformFolder+'/'
    ExpN = self._Ginfo['ExpN'] + str(self._IN)


    MaxDeform = self._setting['MaxDeform'][D]
    Np = self._setting['Np'][D]
    sigmaB = self._setting['sigmaB'][D]
    Border = self._setting['Border']
    Dim = self._setting['Dim']
    DistanceDeform = self._setting['DistanceDeform']
    DistanceArea = self._setting['DistanceArea']
    if self._Ini > 0:
        Dfolder = DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/DIni' + str(self._Ini) + '/'
    else:
        Dfolder = DeformPath + ExpN + '/Dsmooth' + str(Dsmooth) + '/D' + str(D) + '/'
    if not os.path.exists(Dfolder):
        os.makedirs(Dfolder)

    DVFX = np.zeros(self._FixedIm_.shape, dtype=np.float64)
    DVFY = np.zeros(self._FixedIm_.shape, dtype=np.float64)
    DVFZ = np.zeros(self._FixedIm_.shape, dtype=np.float64)
    DeformedArea_ = np.zeros(self._FixedIm_.shape)
    border_mask = np.zeros(self._FixedIm_.shape)
    border_mask[Border:self._FixedIm_.shape[0] - Border + 1, Border:self._FixedIm_.shape[1] - Border + 1, Border:self._FixedIm_.shape[2] - Border + 1] = 1

    i = 0;
    index_edge = np.where(border_mask > 0) # Previously, we only selected voxels on the edges (CannyEdgeDetection), but now we use all voxels.
    if (len(index_edge[0]) == 0):
        print('SyntheticDeformation: We are out of points. Plz change the threshold value of Canny method!!!!! ') # Old method. only edges!

    while ((len(index_edge[0]) > 4) & (i < Np)): # index_edge will change at the end of this while loop!
        if sys.version_info[0] < 3:
            selectVoxel = long(np.random.randint(0, len(index_edge[0]) - 1, 1, dtype=np.int64))
        else:
            selectVoxel = int(np.random.randint(0, len(index_edge[0]) - 1, 1, dtype=np.int64))
        z = index_edge[0][selectVoxel]
        y = index_edge[1][selectVoxel]
        x = index_edge[2][selectVoxel]
        if i < 2:  # We like to include zero deformation in our training set.
            Dx = 0
            Dy = 0
            Dz = 0
        else:
            Dx = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
            Dy = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2
            Dz = ((np.random.ranf([1]))[0] - 0.5) * MaxDeform * 2

        DVFX[z, y, x] = Dx
        DVFY[z, y, x] = Dy
        DVFZ[z, y, x] = Dz

        xminD = x - DistanceDeform
        xmaxD = x + DistanceDeform
        yminD = y - DistanceDeform
        ymaxD = y + DistanceDeform
        zminD = z - DistanceDeform
        zmaxD = z + DistanceDeform

        if zmaxD > (self._FixedIm_.shape[0] - 1): zmaxD = (self._FixedIm_.shape[0] - 1)
        if ymaxD > (self._FixedIm_.shape[1] - 1): ymaxD = (self._FixedIm_.shape[1] - 1)
        if xmaxD > (self._FixedIm_.shape[2] - 1): xmaxD = (self._FixedIm_.shape[2] - 1)
        if zminD < 0: zminD = 0
        if yminD < 0: yminD = 0
        if xminD < 0: xminD = 0
        xminA = x - DistanceArea
        xmaxA = x + DistanceArea
        yminA = y - DistanceArea
        ymaxA = y + DistanceArea
        if (Dim == '3D'):
            zminA = z - DistanceArea
            zmaxA = z + DistanceArea
        else:
            zminA = z - 1
            zmaxA = z + 2  # This is exclusively for 2D !!!!

        if zmaxA > (self._FixedIm_.shape[0] - 1): zmaxA = (self._FixedIm_.shape[0] - 1)
        if ymaxA > (self._FixedIm_.shape[1] - 1): ymaxA = (self._FixedIm_.shape[1] - 1)
        if xmaxA > (self._FixedIm_.shape[2] - 1): xmaxA = (self._FixedIm_.shape[2] - 1)
        if zminA < 0: zminA = 0
        if yminA < 0: yminA = 0
        if xminA < 0: xminA = 0

        border_mask[zminD:zmaxD, yminD:ymaxD, xminD:xmaxD] = 0
        DeformedArea_[zminA:zmaxA, yminA:ymaxA, xminA:xmaxA] = 1
        index_edge = np.where(border_mask > 0)
        i += 1


    del border_mask

    DeformedArea = sitk.GetImageFromArray(DeformedArea_)
    DeformedArea.SetOrigin(self._FixedIm.GetOrigin())
    DeformedArea.SetSpacing(self._FixedIm.GetSpacing())
    sitk.WriteImage(DeformedArea, Dfolder + 'DeformedArea.mha')

    DVFXb = gaussian_filter(DVFX, sigma=sigmaB)
    DVFYb = gaussian_filter(DVFY, sigma=sigmaB)
    DVFZb = gaussian_filter(DVFZ, sigma=sigmaB)

    IXp = np.where(DVFXb > 0)
    IXn = np.where(DVFXb < 0)
    IYp = np.where(DVFYb > 0)
    IYn = np.where(DVFYb < 0)
    IZp = np.where(DVFZb > 0)
    IZn = np.where(DVFZb < 0)

    DVFXb[IXp] = ((np.max(DVFX) - 0) / (np.max(DVFXb[IXp]) - np.min(DVFXb[IXp])) * (DVFXb[IXp] - np.min(DVFXb[IXp])) + 0)
    DVFXb[IXn] = ((0 - np.min(DVFX[IXn])) / (0 - np.min(DVFXb[IXn])) * (DVFXb[IXn] - np.min(DVFXb[IXn])) + np.min(DVFX[IXn]))
    DVFYb[IYp] = ((np.max(DVFY) - 0) / (np.max(DVFYb[IYp]) - np.min(DVFYb[IYp])) * (DVFYb[IYp] - np.min(DVFYb[IYp])) + 0)
    DVFYb[IYn] = ((0 - np.min(DVFY[IYn])) / (0 - np.min(DVFYb[IYn])) * (DVFYb[IYn] - np.min(DVFYb[IYn])) + np.min(DVFY[IYn]))

    _DeformedDVF_[:, :, :, 0] = DVFXb
    _DeformedDVF_[:, :, :, 1] = DVFYb

    if (Dim == '3D'):
        DVFZb[IZp] = ((np.max(DVFZ) - 0) / (np.max(DVFZb[IZp]) - np.min(DVFZb[IZp])) * (DVFZb[IZp] - np.min(DVFZb[IZp])) + 0)
        DVFZb[IZn] = ((0 - np.min(DVFZ[IZn])) / (0 - np.min(DVFZb[IZn])) * (DVFZb[IZn] - np.min(DVFZb[IZn])) + np.min(DVFZ[IZn]))
        self._DeformedDVF_[:, :, :, 2] = DVFZb

