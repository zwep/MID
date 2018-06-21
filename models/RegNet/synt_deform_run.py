# encoding: utf-8

"""
Here we run the deformaiton algorithm...
"""

import os


idx = 'ProstateX-0169'
i_path = os.path.join(DIR_IMG, idx)
i_image_series = []
for i_dir, _, i_file in os.walk(os.path.join(DIR_IMG, idx)):
    temp_list = []
    if len(i_file):
        temp_list = [os.path.join(i_dir, i) for i in i_file]
        i_image_series.append(temp_list)

temp = i_image_series[0][0]

deform_image(temp)

