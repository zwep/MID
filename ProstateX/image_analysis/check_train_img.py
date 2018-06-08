# encoding: utf-8

"""
here we check whether the images shown in

C:\Users\C35612\data\mid\ProstateX-Screenshots-Train

are found in the data given by

C:\Users\C35612\data\mid\PROSTATEx

And also show that none of this is equal to what is found in

C:\Users\C35612\data\mid\ProstateX-TrainingLesionInformationv2\ProstateX-Images-Train.csv

An amazing source for DICOM images
https://dicom.innolitics.com/ciods
"""

# Load in necessary libraries
from ProstateX.config import *
from helper.misc import prep_splitcol

import numpy as np
from PIL import Image
import pandas as pd
import glob

import pydicom


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# New column names used to split columns
pos_cols = ['pos_x', 'pos_y', 'pos_z']
ijk_cols = ['i', 'j', 'k']

# Load target data
pd_dat_train = pd.read_csv(os.path.join(dir_xls, file_name_fid_train))
pd_dat_train = prep_splitcol(pd_dat_train, pos_cols, 'pos')

# Load info about the images
pd_dat_img = pd.read_csv(os.path.join(dir_xls, file_name_img_train))
pd_dat_img_ind = pd_dat_img.set_index(['ProxID', 'Name'])
# List of images that can be used to validate
list_valid_img = glob.glob(dir_valid + '\*.bmp')


pd_img_info = []
col_names = ['PatientID', 'SeriesName', 'MySerNum', 'i_x', 'i_y', 'n_x', 'n_y']
pd_img_info.append(col_names)

# Here we search for the matching image...
for i, i_row in pd_dat_img.iterrows():
    # Load info from the row data..
    i_name = i_row['Name']
    id = i_row['ProxID']
    img_nr = i_row['DCMSerNum']
    i_folder = i_row['DCMSerDescr'].replace('_', '')  # Remove these to find the files...
    print(id, i_row['Name'])

    # Recursively find all the images belonging to this patient and the mentioned DCM Series Description.
    i_path = os.path.join(dir_img, id) + '\*\*\*'
    i_id_files = glob.glob(i_path, recursive=True)
    file_dir_dcm = [x for x in i_id_files if i_folder in x]

    # Load all those series images
    try:
        # This should be the file that is linked to the res_img..
        i_file = file_dir_dcm[img_nr]
    except IndexError:
        print('We have an index error', id, i_name, img_nr)
        i_file = file_dir_dcm[0]

    temp_file = pydicom.read_file(i_file)
    max_x, max_y = (temp_file.Rows, temp_file.Columns)
    n_files = len(file_dir_dcm)
    dim_pixels = (int(max_x), int(max_y), n_files)
    # array to load in the whole series
    temp_img = np.zeros(dim_pixels, dtype=temp_file.pixel_array.dtype)

    # fill the predefined array with info
    for i_file in file_dir_dcm:
        # read the file
        value_img = pydicom.read_file(i_file)
        # store the raw image data
        temp_img[:, :, file_dir_dcm.index(i_file)] = np.array(value_img.pixel_array)

    # Get the image that we want to check
    res_check = [x for x in list_valid_img if (id in x) and (i_name in x)]
    if len(res_check) == 1:
        res_img = Image.open(res_check[0])
        res_img = np.array(res_img)
        # Normalize
        res_img = res_img/np.max(res_img)
    else:
        print('more ref-img found. Taking first', id, i_name, 'length of img found', len(res_check))
        res_img = Image.open(res_check[0])
        res_img = np.array(res_img)
        res_img = res_img/np.max(res_img)

    # Here we are going to subset the original picture with the same size as res_img.
    # Then for each patch we are going to compare how well the fit was
    # Resulting, finally, in a position with the best fit.. which is the actual location of res_img compared to the
    # series data presented by the patient

    n_x, n_y = res_img.shape
    nstep_x = max_x - n_x
    nstep_y = max_y - n_y

    final_dist = []
    final_xy = []
    for i_img in range(n_files):
        count = 0
        temp_linalg = []
        temp_xy = []

        derp_img = temp_img[:, :, i_img]
        derp_img = derp_img/np.max(derp_img)

        for i in np.arange(0, nstep_x, 1):
            for j in np.arange(0, nstep_y, 1):
                temp_xy.append((i, j))
                z = derp_img[(i + 0):(i + n_x),
                             (j + 0):(j + n_y)]
                count += 1
                temp_dist = np.linalg.norm(z - res_img)
                temp_linalg.append(temp_dist)
        final_dist.append(temp_linalg)
        final_xy.append(temp_xy)

    Z = np.reshape(final_dist, (n_files, nstep_x, nstep_y))
    chosen_id, i_ding, j_ding = np.unravel_index(np.argmin(Z, axis=None), Z.shape)

    info_dict = [id, i_name, chosen_id, i_ding, j_ding, n_x, n_y]
    pd_img_info.append(info_dict)

A = pd.DataFrame(pd_img_info)

file_name = 'loc_of_real_img.csv'
if os.path.isfile(os.path.join(dir_mid, file_name)):
    print('filename is already present')
    print('Overwite?')
    bool_ovwrt = input()
    if 'y' in bool_ovwrt:
        print('overwite. ok.')
        A.to_csv(os.path.join(dir_mid, file_name), index=False)
    else:
        if re.search('\([0-9]\)\.', file_name):
            x = int(re.findall('\(([0-9])\)\.', file_name)[0])
            x += 1
            file_name_new = re.sub('\([0-9]+\)\.', '({0}).'.format(x), file_name)
        else:
            file_name_new = re.sub('\.', '(1).', file_name)
        print('new filename: ', file_name_new)
        A.to_csv(os.path.join(dir_mid, file_name_new), index=False)



# Example of recovery
A = pd.read_csv(os.path.join(dir_mid, 'loc_of_real_img.csv'))
A.columns = ['index', 'PatientID', 'SeriesName', 'MySerNum', 'i_xy', 'n_xy']

# This works.. a bit.. not amazing though...
# Needs a lot of debugging for it to run smoothly
test = list()
for i, i_row in A.iterrows():
    try:
        # Retrieve info from the row...
        i_id = i_row['PatientID']
        i_name0 = re.sub('[0-9]+$','', i_row['SeriesName'])
        i_name = i_name0.replace('_', '')
        i_nr = i_row['MySerNum']
        i_x, i_y = map(int, re.sub('\)|\(', '', i_row['i_xy']).split(','))
        n_x, n_y = map(int, re.sub('\)|\(', '', i_row['n_xy']).split(','))

        #
        img_csv_nr = pd_dat_img[(pd_dat_img['ProxID'] == i_id) & (pd_dat_img['DCMSerDescr'] == i_name0)]['DCMSerNum']
        img_csv_nr = np.array(img_csv_nr)[0]

        # loop through all the image of i_id
        for x in os.walk(os.path.join(dir_mid, r'PROSTATEx\\' + i_id)):
            if i_name in x[0]:
                i_files = os.path.join(x[0], x[2][i_nr])
                csv_files = os.path.join(x[0], x[2][img_csv_nr])

        # Get the actual image with which we have compared stuff
        list_valid_img = glob.glob(dir_valid + '\*.bmp')
        res_check = [x for x in list_valid_img if (i_id in x) and (i_name0 in x)]

        if len(res_check) == 1:
            print('ok')
        else:
            print('more findings found')

        value_img = pydicom.read_file(i_files)
        csv_img = pydicom.read_file(csv_files)
        s_plot = 2
        res_img_list = [np.array(Image.open(x)) for x in res_check]
        n_fid = len(res_img_list)
        s_plot += n_fid
        res_img_list.append(value_img.pixel_array)
        res_img_list.append(csv_img.pixel_array)

        title_list = ['fid_{0}'.format(x) for x in range(n_fid)]
        title_list.append('my_image')
        title_list.append('csv_image')

        import pylab

        plt.figure(1)
        plt.imshow(res_img_list[i], cmap=plt.cm.gray)
        plt.show()
        for i,v in enumerate(res_img_list):
            ax1 = pylab.subplot(s_plot, 1,i+1)
            ax1.imshow(res_img_list[i], cmap=plt.cm.gray)
            plt.title(title_list[i])

        plt.pause(0.05)
        plt.show()

        print('y or n. If Y then we are correct, if N, then we are not correct')
        x = input()
        test.append(x)
    except IndexError:
        print('wooops')
        continue
    except KeyboardInterrupt:
        print('Cancelled by user')
        break