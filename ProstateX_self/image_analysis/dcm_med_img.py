# encoding: utf-8

"""
Usingt dcom files and reading them

This is where I got my data...
https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#3da478cb8e54404d8eb76db063a7b113



"""

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import helper.misc as misc

dir_img = r'C:\Users\C35612\data\mid\PROSTATEx\ProstateX-0000\07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711\3-t2tsesag-87368'
im_path = r'C:\Users\C35612\data\mid\ProstateXKtrains-train-fixed\ProstateX-0000\ProstateX-0000-Ktrans.mhd'
dir_name, subdir_list, file_list = tuple(*os.walk(dir_img))

file_dir_dcm = [os.path.join(dir_name, i_file) for i_file in file_list if i_file.endswith('.dcm')]
# Get ref file
temp_file = pydicom.read_file(file_dir_dcm[0])

plt.imshow(temp_file.pixel_array)
plt.show()
dir(temp_file)
temp_file.BodyPartExamined
temp_file.PatientSex
temp_file.PatientSize
temp_file.PatientWeight
temp_file.PatientAge

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
dim_pixels = (int(temp_file.Rows),
              int(temp_file.Columns),
              len(file_dir_dcm))

# Load spacing values (in mm)
spacing_pixels = (float(temp_file.PixelSpacing[0]),
                     float(temp_file.PixelSpacing[1]),
                     float(temp_file.SliceThickness))

x = np.arange(0.0, (dim_pixels[0]+1)*spacing_pixels[0], spacing_pixels[0])
y = np.arange(0.0, (dim_pixels[1]+1)*spacing_pixels[1], spacing_pixels[1])
z = np.arange(0.0, (dim_pixels[2]+1)*spacing_pixels[2], spacing_pixels[2])

# The array is sized based on 'ConstPixelDims'
temp_img = np.zeros(dim_pixels, dtype=temp_file.pixel_array.dtype)

# loop through all the DICOM files
for i_file in file_dir_dcm:
    # read the file
    value_img = pydicom.read_file(i_file)
    # store the raw image data
    temp_img[:, :, file_dir_dcm.index(i_file)] = value_img.pixel_array


plt.figure(dpi=300)
plt.axes().set_aspect('equal', 'datalim')
plt.set_cmap(plt.gray())
plt.pcolormesh(x, y, np.flipud(temp_img[:, :, 8]))

one_img = temp_img[:, :, 1]
nx_slice = 50
size_img = one_img.shape
factor = int(size_img[0] / nx_slice)
if size_img[0] % nx_slice != 0:
    print('alarm')


ugly_input_set = []
for i in range(factor):
    for j in range(factor):
        idx = j*nx_slice + np.array(range(0, nx_slice))
        z = one_img[(i*nx_slice + 0):((i+1)*nx_slice + nx_slice), (j*nx_slice + 0):((j+1)*nx_slice + nx_slice)]
        z_shape = np.array(z.shape)
        padded_z = np.zeros((2*nx_slice, 2*nx_slice))
        if any(z_shape != 2*nx_slice):
            print('noo')
            padded_z[0:z_shape[0], 0:z_shape[1]] = z
            ugly_input_set.append(padded_z)
        else:
            ugly_input_set.append(z)


z = misc.load_itk(im_path)
z