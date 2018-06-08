# encoding: utf-8

"""

Examples from the medpy packages
http://loli.github.io/medpy/filter.html

Tried to use itk, or pipypye or pipy.. but all of those didnt work as easy as medpy.
Simple filters and transformations.. amazing
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import itk
import numpy as np
import pydicom
from scipy import ndimage, misc

z = r'C:\Users\C35612\data\mid\PROSTATEx\ProstateX-0001\07-08-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-95738\8-ep2ddifftraDYNDISTMIXADC-33954\000011.dcm'
value_img = pydicom.read_file(z)
Z = value_img.pixel_array

import medpy.filter.smoothing as medpysmooth
import medpy.filter as medpyfilter
import medpy.filter.noise as medpynoise

Z_smooth = medpysmooth.anisotropic_diffusion(Z, 1000, 100, 0.1)
result = ndimage.sobel(Z)
result_smooth = ndimage.sobel(Z_smooth)

medpynoise.immerkaer(Z_smooth)
medpynoise.immerkaer_local(Z_smooth)

Z_bin_comp = medpyfilter.binary.largest_connected_component(Z)

def normie(x):
    xnormie = (x-np.min(x))/(np.max(x) - np.min(x))

    return xnormie

plot_list = [Z,Z_smooth, result, result_smooth, Z_bin_comp]
plot_list_new = []
for i, x in enumerate(plot_list):
    plt.figure(i)
    # plot_list_new.append(normie(x))
    # plt.imshow(normie(x), cmap=plt.cm.gray)
    plt.imshow(x, cmap=plt.cm.gray)
plt.show()

w = r'C:\Users\C35612\data\mid\ProstateX-Screenshots-Train\ProstateX-0000-Finding1-t2_tse_sag0.jpg'
value_img = itk.imread(w)

median = itk.MedianImageFilter.New(value_img.pixel_array, Radius = 2)

help(itk.MedianImageFilter)
value_img.pixel_values