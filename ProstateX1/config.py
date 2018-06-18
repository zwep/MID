# encoding: utf-8

"""
Module to set certain parameters...
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import re, os
import glob

TRAIN_ID = True

# Define paths to certain info...
DIR_MID = r'C:\Users\C35612\data\mid\PROSTATEx_2016'
DIR_IMG = r'C:\Users\C35612\data\mid\PROSTATEx_2016\PROSTATEx_train\PROSTATEx'
DIR_SS = r'C:\Users\C35612\data\mid\PROSTATEx_2016\PROSTATEx_train\ProstateX-Screenshots-Train'
DIR_XLS = r'C:\Users\C35612\data\mid\PROSTATEx_2016\PROSTATEx_train\ProstateX-TrainingLesionInformation'
DIR_KTRANS = r'C:\Users\C35612\data\mid\PROSTATEx_2016\PROSTATEx_train\ProstateXKtrains-train-fixed'

FIND_CSV = 'ProstateX-Findings-Train.csv'
KTRANS_CSV = 'ProstateX-Images-KTrans-Train.csv'
IMG_CSV = 'ProstateX-Images-Train.csv'
