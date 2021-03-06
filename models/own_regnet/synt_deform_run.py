# encoding: utf-8

"""
Here we run the deformaiton algorithm...
"""

import os
from settings.config import *
import models.own_regnet.synt_deform as synt_deform
import numpy as np
import importlib
importlib.reload(synt_deform)
import SimpleITK as sitk

# TODO Okay, I have now recreated the Synt Deform myself. I need to check whether Im getting the same results with my thing as with theirs..

for i_idx in os.listdir(DIR_IMG):
    i_idx = 'ProstateX-0015'
    i_path = os.path.join(DIR_IMG, i_idx)
    i_image_series = []
    for i_dir, _, i_file in os.walk(os.path.join(DIR_IMG, i_idx)):
        temp_list = []
        if len(i_file) and 'tsetra' in i_dir:
            temp_list = [os.path.join(i_dir, i) for i in i_file]
            i_image_series.append(temp_list)

    if len(i_image_series) == 1:
        temp_imag_series = i_image_series[0]
    else:
        print('we found multiple tsetra series', i_idx)

    for i_file in temp_imag_series:
        i_file = temp_imag_series[0]
        dvfb = synt_deform.deform_image(i_file, dim=2, border=50, max_deform=30, n_p=30, sigma_b=2, id_dvf_area=True, distance_deform=1)
        synt_deform.write_image(i_file, dvfb)
