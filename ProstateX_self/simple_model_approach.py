# encoding: utf-8

"""
Here we build a simple model based on the train and test data...
"""

# Load in necessary libraries
import os
import re
import numpy as np
from PIL import Image
import pandas as pd
import glob

import pydicom
import SimpleITK as sitk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Define paths to certain info...
dir_data = r'C:\Users\C35612\data\mid\ProstateX-TrainingLesionInformationv2'
dir_data_test = r'C:\Users\C35612\data\mid\PROSTATEx_test\ProstateX-TestLesionInformation'

dir_img = r'C:\Users\C35612\data\mid\PROSTATEx'
dir_img_test = r'C:\Users\C35612\data\mid\PROSTATEx_test\PROSTATEx'

dir_mid = r'C:\Users\C35612\data\mid'

file_name_fid_train = 'ProstateX-Findings-Train.csv'
file_name_img_train = 'ProstateX-Images-Train.csv'
file_name_info_train = 'patient_info_train.csv'

file_name_fid_test = 'ProstateX-Findings-Test.csv'
file_name_img_test = 'ProstateX-Images-Test.csv'
file_name_info_test = 'patient_info_test.csv'

# New column names used to split columns
pos_cols = ['pos_x', 'pos_y', 'pos_z']
ijk_cols = ['i', 'j', 'k']


def prep_splitcol(pd_x, new_col, orig_col, splt_key=r'\s'):
    # Simple prep for a specific column..

    pd_x[new_col] = pd_x[orig_col].str.strip().str.split(splt_key, expand=True)
    pd_x[new_col] = pd_x[new_col].apply(pd.to_numeric, errors='coerce', axis=1)
    return pd_x

# Load target data
pd_dat_train = pd.read_csv(os.path.join(dir_data, file_name_fid_train))
pd_dat_train = prep_splitcol(pd_dat_train, pos_cols, 'pos')
pd_info_train = pd.read_csv(os.path.join(dir_mid, file_name_info_train))
pd_dat_train.shape
pd_info_train.shape
A = pd.merge(pd_dat_train, pd_info_train, left_on='ProxID', right_on='ID')
subset_target = ['ProxID', 'ClinSig']
subset_ftr = ['ProxID', 'pos_x', 'pos_y', 'pos_z', 'H', 'KG']

A_target = A[subset_target]
A_ftr = A[subset_ftr]


pd_dat_test = pd.read_csv(os.path.join(dir_data_test, file_name_fid_test))
pd_dat_test = prep_splitcol(pd_dat_test, pos_cols, 'pos')
pd_info_test = pd.read_csv(os.path.join(dir_mid, file_name_info_test))
B = pd.merge(pd_dat_test, pd_info_test, left_on='ProxID', right_on='ID')
B_ftr = B[subset_ftr]
B_target = B[subset_target]
#
from sklearn.preprocessing import LabelBinarizer, CategoricalEncoder

X = A_ftr.loc[:, A_ftr.columns !='ProxID']
n_sample =  X.shape[0]
n_train = np.random.choice(n_sample, int(n_sample*0.7), replace=False)
n_test = [x for x in range(n_sample) if x not in n_train]

X_train = X.iloc[n_train]
X_test = X.iloc[n_test]
y_train = A_target.loc[n_train, A_target.columns !='ProxID']
y_test = A_target.loc[n_test, A_target.columns !='ProxID']

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, np.array(y_train).ravel())
clf.score(X_test, y_test)
clf.score(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, np.array(y_train).ravel())
clf.score(X_test, y_test)
clf.score(X_train, y_train)


z_train = [x[0] for x in y_test.values]
train_list = list(collections.Counter(z_train).values())
train_list[0]/sum(train_list)
train_list[1]/sum(train_list)