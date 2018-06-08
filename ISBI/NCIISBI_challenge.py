# encoding: utf-8

"""
Here we are going to try to train a simple unet model with the

NCI-ISBI challenge data. To be found here
https://wiki.cancerimagingarchive.net/display/DOI/NCI-ISBI+2013+Challenge%3A+Automated+Segmentation+of+Prostate+Structures

aaand go
"""

import re
import numpy as np
import nrrd
import pydicom
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


dir_target_train = r'C:\Users\C35612\data\mid\NCIISBI\Training'
dir_diag_train = r'C:\Users\C35612\data\mid\NCIISBI\PROSTATE-DIAGNOSIS_train'
dir_prost_train = r'C:\Users\C35612\data\mid\NCIISBI\Prostate-3T_train'


def fucking_fuck_load_data(input_loc, id):
    """
    Loads the fucking prostate Dx or 3T data.

    Abso-fucking-lutely gets the nrrd data as well

    :param input_loc:
    :param id: id of the patient
    :return:
    """

    id = str(id).zfill(4)

    if 'Prostate-3T' in input_loc:
        file_spec = 'Prostate3T'
    elif 'PROSTATE-DIAGNOSIS' in input_loc:
        file_spec = 'ProstateDx'
    else:
        print('Error0')
        file_spec = 'Unknown'

    if re.findall('_train$', input_loc):
        prefix = 'Training'
    elif re.findall('_test$', input_loc):
        prefix = 'Test'
    else:
        print('Error1')
        prefix = 'Unkown'

    base_dir = os.path.dirname(input_loc)
    temp_dir = '{path}\{file_spec}-01-{id}'.format(path=input_loc, file_spec=file_spec, id=id)
    target_file = '{path}\{prefix}\{file_spec}-01-{id}.nrrd'.format(path=base_dir,
                                                                    prefix=prefix,
                                                                    file_spec=file_spec,
                                                                    id=id)

    counter = 0
    dat_files = []
    for x in os.walk(temp_dir):
        if len(x[2]):
            dat_files = [pydicom.read_file(os.path.join(x[0], i)).pixel_array.T for i in x[2]]

            counter += 1
            if counter > 1:
                print('Watch it', input_loc, file_spec,  id)

    readdata, options = nrrd.read(target_file)
    readdata_rolled = np.rollaxis(readdata, 2)

    return dat_files, readdata_rolled

fucking_img_data, target_fucking_data = fucking_fuck_load_data(dir_diag_train, 5)

plt.imshow(fucking_img_data[0], colormap=plt.cm.gray)
plt.figure(2)
plt.imshow(target_fucking_data[0])
plt.show()
#wooo
