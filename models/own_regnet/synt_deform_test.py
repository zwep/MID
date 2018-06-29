# encoding: utf-8

"""
Here we test the properties of syntdeform and different parameters to see if it really works...
"""

from settings.config import *
import models.own_regnet.synt_deform as synt_deform
import numpy as np
import SimpleITK as sitk


# Make sure that we can get the created images..
def plot_difference(input_path):
    id_seriex = re.findall('([0-9]{6})\.dcm', input_path)[0]
    file_mvd = os.path.join(os.path.dirname(input_path), 'MVD' + id_seriex + '.dcm')

    deformed_image_array = sitk.GetArrayFromImage(sitk.ReadImage(file_mvd))
    input_image_array = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
    z = (input_image_array - deformed_image_array).transpose()
    plt.imshow(z[:, :, 0], cmap='gray')


# Define some settings...
border_set = [10, 50, 150]
max_deform_set = [10, 30, 100]
n_p_set = [10, 30, 100]
sigma_b_set = [2, 10, 20]
id_dvf_area_set = [False, False, False]
dist_deform_set = [1, 25, 50]
list_prop = [border_set, max_deform_set, n_p_set, sigma_b_set, id_dvf_area_set, dist_deform_set]

# Put them in a dictionary
full_set_dict = {}
prop_names = ['border', 'max_deform', 'n_p', 'sigma_b', 'id_dvf_area', 'distance_deform']
for i, j in zip(prop_names, list_prop):
    full_set_dict[i] = j

# Define some image paths...
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


i_file = temp_imag_series[0]

# Plot the deformation field
for i in range(len(border_set)):
    print(i, '-------')

    temp_set = {k: v[i] for k, v in full_set_dict.items()}
    print(temp_set)
    temp_dvfb = synt_deform.deform_image(i_file, **temp_set)
    synt_deform.write_image(i_file, temp_dvfb)

    dvfb_stack = np.stack(temp_dvfb, axis=3)
    plt.figure(i)
    plt.imshow(dvfb_stack[:, :, 0, 0])

    plt.figure(99 - i)
    plot_difference(i_file)


sitk_dvfb = sitk.GetImageFromArray(dvfb_stack, isVector=False)
dvf_t = sitk.DisplacementFieldTransform(sitk_dvfb)
z = sitk.GetArrayFromImage(dvf_t)

# This is the clean version of the deformed image. Intensity noise should be added to this image
deformed_image = sitk.Resample(input_image, dvf_t)
deformed_image = sitk.AdditiveGaussianNoise(deformed_image, sigma_n, 0, 0)

