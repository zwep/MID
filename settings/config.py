# encoding: utf-8

"""
Module to set certain parameters...
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

PLT_INT = True
plt.interactive(PLT_INT)

print('Interactive mode is:', PLT_INT)

import re
import os
import glob

TRAIN_ID = True

if TRAIN_ID:
    ID0 = 'Train'
    ID1 = 'train'
    ID2 = 'Training'
else:
    ID0 = 'Test'
    ID1 = 'test'
    ID2 = 'Test'

# Define paths to certain info...
DIR_MID = r'C:\Users\C35612\data\mid\PROSTATEx_2016'
DIR_IMG = os.path.join(DIR_MID, r'PROSTATEx_{0}\PROSTATEx'.format(ID1))
DIR_SS = os.path.join(DIR_MID, r'PROSTATEx_{0}\ProstateX-Screenshots-{1}'.format(ID1, ID0))
DIR_XLS = os.path.join(DIR_MID, r'PROSTATEx_{0}\ProstateX-{1}LesionInformation'.format(ID1, ID2))
DIR_KTRANS = os.path.join(DIR_MID, r'PROSTATEx_{0}\ProstateXKtrans-{1}-fixed'.format(ID1, ID1))

FIND_CSV = 'ProstateX-Findings-{0}.csv'.format(ID0)
KTRANS_CSV = 'ProstateX-Images-KTrans-{0}.csv'.format(ID0)
IMG_CSV = 'ProstateX-Images-{0}.csv'.format(ID0)

DIR_CIFAR = r'C:\Users\C35612\data\cifar'