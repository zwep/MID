# encoding: utf-8

"""
Module to set certain parameters...
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import re, os
import glob

# Define paths to certain info...
dir_mid = r'C:\Users\C35612\data\mid\PROSTATEx_2016'
dir_img = r'C:\Users\C35612\data\mid\PROSTATEx_2016\PROSTATEx_train\PROSTATEx'
dir_valid = r'C:\Users\C35612\data\mid\PROSTATEx_2016\PROSTATEx_train\ProstateX-Screenshots-Train'
dir_xls = r'C:\Users\C35612\data\mid\PROSTATEx_2016\PROSTATEx_train\ProstateX-TrainingLesionInformation'
dir_ktrans = r'C:\Users\C35612\data\mid\PROSTATEx_2016\PROSTATEx_train\ProstateXKtrains-train-fixed'

file_name_fid_train = 'ProstateX-Findings-Train.csv'
file_name_ktrans_train = 'ProstateX-Images-KTrans-Train.csv'
file_name_img_train = 'ProstateX-Images-Train.csv'
