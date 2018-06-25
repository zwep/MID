# encoding: utf-8

"""
Here we are going to build a simple model.. generate some data.. and see if the CNN can catch augmentations..
"""

from settings.config import *
from data_augmentation.elastic import elastic_transform, random_rotation

import numpy as np

def unpickle(file):
    """
    Used now for specific CIFAR data...
    :param file:
    :return:
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Get some data
list_batch = [x for x in os.listdir(DIR_CIFAR) if x.startswith('data_batch')]
i_batch = list_batch[0]
temp_dict = unpickle(os.path.join(DIR_CIFAR, i_batch))
temp_dict.keys()

counter = 0
for i_image in temp_dict[list(temp_dict.keys())[2]]:
    if counter > 5:
        break
    counter += 1
    A = np.reshape(i_image, (32, 3*32))
    B = np.reshape(A, (3, 32, 32))
    C = np.swapaxes(B, 0, 2)
    C = random_rotation(C)
    D = elastic_transform(C)


# Augment it

# Put it in a CNN thing