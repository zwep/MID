# encoding: utf-8

"""
Here we analyze the output scores using numpy and pandas

"""

import os
from pandas import read_csv, to_numeric

dir_data = r'C:\Users\C35612\data\mid\ProstateX-TrainingLesionInformationv2'

file_name = 'ProstateX-Findings-Train.csv'
file_path = os.path.join(dir_data, file_name)
if os.path.isfile(os.path.join(dir_data, file_name)):
    pd_dat = read_csv(file_path)

pos_cols = ['pos_x', 'pos_y', 'pos_z']
pd_dat[pos_cols] = pd_dat['pos'].str.strip().str.split('\s', expand=True)
pd_dat[pos_cols] = pd_dat[pos_cols].apply(to_numeric, errors='coerce', axis=1)

# Getting an idea how frequent the ClinSig value is
pd_dat['ClinSig'].value_counts()
pd_dat['fid'].value_counts()
pd_dat['zone'].value_counts()
pd_dat['pos_x'].describe()
pd_dat['pos_y'].describe()
pd_dat['pos_z'].describe()

pd_dat.groupby(['ClinSig', 'fid']).count()['ProxID']
