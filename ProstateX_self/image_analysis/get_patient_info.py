# encoding: utf-8

"""
Here we will retrieve some patient info using the dicom images

"""

import numpy as np
import os
from PIL import Image
import pandas as pd
import glob

import pydicom
import SimpleITK as sitk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


dir_img_train = r'C:\Users\C35612\data\mid\PROSTATEx'
dir_img_test = r'C:\Users\C35612\data\mid\PROSTATEx_test\PROSTATEx'
dir_mid = r'C:\Users\C35612\data\mid'

get_info = ['BirthDate', 'ID', 'Name', 'Position', 'Sex', 'Size', 'Weight']
get_info_patient = ['Patient' + x for x in get_info]

res_info = []
for i in os.walk(dir_img_test):
    if len(i[2]):
        for i_file in i[2]:
            img_path = os.path.join(i[0], i_file)
            i_img = pydicom.read_file(img_path)
            i_info = [getattr(i_img, x_attr, -1) for x_attr in get_info_patient]
            res_info.append(i_info)

final_ding = []
prev = res_info[0]
for i in res_info[1:]:
    if i != prev:
        final_ding.append(prev)
        prev = i

A = pd.DataFrame(final_ding)
A.head()
A.columns = ['index', 'ID', 'ID1', 'POS', 'SEX', 'H', 'KG']

plt.hist(A['KG'])
plt.show()

plt.hist(A['H'])
plt.show()

# Here we have some DQ issue.. since sometimes the length is in M and sometimes in CM..
length_patient = list(A['H'])
length_patient = [x*100 if x < 3 else x for x in length_patient]
A['H'] = length_patient
plt.hist(length_patient)
plt.show()

A['POS'].value_counts()

file_name = 'patient_info_train.csv'
file_name = 'patient_info_test.csv'
A.to_csv(os.path.join(dir_mid, file_name), index=False)
