# encoding: utf-8

"""
In check_trian_img we have created a table where we figured out some mapping... between the screenshots and the true data

Here we are going to compare many different slices of the input data in order to understand the training data...
With the help of an external program that helps to visualize the DICOM images
"""

from ProstateX.config import *

import pydicom
import numpy as np
from PIL import Image
import pandas as pd
from helper.misc import prep_splitcol

screenshot_img = glob.glob(dir_valid + '\*.bmp')

# New column names used to split columns
pos_cols = ['pos_x', 'pos_y', 'pos_z']
ijk_cols = ['i', 'j', 'k']

pd_dat_train = pd.read_csv(os.path.join(dir_xls, file_name_fid_train))
pd_dat_train = prep_splitcol(pd_dat_train, pos_cols, 'pos')

pd_dat_img = pd.read_csv(os.path.join(dir_xls, file_name_img_train))
pd_dat_img_ind = pd_dat_img.set_index(['ProxID', 'Name'])

# Self created table...
A = pd.read_csv(os.path.join(dir_mid, 'loc_of_real_img.csv'))
A.columns = ['index', 'ProxID', 'Name', 'MySerNum', 'i_xy', 'n_xy']
A = A.set_index(['ProxID', 'Name'])


# Now also check the pos in train.. and the pos in the DICOM images...
i_id = 'ProstateX-0001'

for i in os.walk(os.path.join(dir_img, i_id)):
    #if re.findall(i_re_type, i[0]):
    if i_re_type in i[0]:
        img_list = os.listdir(i[0])
        csv_img = pydicom.read_file(os.path.join(i[0], img_list[SerNum])).pixel_array
        my_img = pydicom.read_file(os.path.join(i[0], img_list[min(len(img_list)-1, MySerNum)])).pixel_array
        ijk_img = pydicom.read_file(os.path.join(i[0], img_list[ijkNum])).pixel_array
        break


# (This step should actually be done later... or in a different file... Here we recover
for i_img in screenshot_img:
    res_img = Image.open(i_img)
    res_img = np.array(res_img)
    # Normalize
    res_img = res_img/np.max(res_img)

    i_id = re.findall('.*(ProstateX-[0-9]{4}).*', i_img)[0]
    i_id_nr = int(re.findall('[0-9]+', i_id)[0])
    if i_id_nr > 50:
        print('We have our max..')
        break
    i_fnd = int(re.findall('.*Finding([0-9]).*', i_img)[0])
    i_type = re.findall('.*Finding[0-9]-(.*).bmp', i_img)[0]
    if i_type.startswith('tfl'):
        continue
    i_re_type = re.sub('[0-9]+$', '', re.sub('_','', i_type)) + '-'
    pd_subset = pd_dat_img_ind.loc[(i_id, i_type)]
    A_subset = A.loc[(i_id, i_type)]
    SerNum = pd_subset['DCMSerNum'].values[i_fnd-1]
    ijkNum = int(pd_subset['ijk'].values[0].split()[-1])
    MySerNum = A_subset['MySerNum'].values[i_fnd-1]
    #pd_subset['ijk'].values
    print('-----', i_id)
    for i in os.walk(os.path.join(dir_img, i_id)):
        #if re.findall(i_re_type, i[0]):
        if i_re_type in i[0]:
            img_list = os.listdir(i[0])
            csv_img = pydicom.read_file(os.path.join(i[0], img_list[SerNum])).pixel_array
            my_img = pydicom.read_file(os.path.join(i[0], img_list[min(len(img_list)-1, MySerNum)])).pixel_array
            ijk_img = pydicom.read_file(os.path.join(i[0], img_list[ijkNum])).pixel_array
            break

    plt.figure(1)
    plt.subplot(2,4,2)
    plt.title(i_type )
    plt.imshow(res_img, cmap=plt.cm.gray)

    plt.subplot(2,4,5)
    plt.title('csv index ' + img_list[SerNum])
    plt.imshow(csv_img, cmap=plt.cm.gray)
    plt.subplot(2,4,6)
    plt.title('my index ' + img_list[min(len(img_list)-1, MySerNum)])
    plt.imshow(my_img, cmap=plt.cm.gray)
    plt.subplot(2,4,7)
    plt.title('ijk index ' + img_list[ijkNum])
    plt.imshow(ijk_img, cmap=plt.cm.gray)
    plt.pause(0.05)
    plt.show()

    try:
        input()
    except KeyboardInterrupt:
        print('Intterupted by used..')
        break
