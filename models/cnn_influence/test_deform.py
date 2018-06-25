# encoding: utf-8

"""
Here we are going to build a simple model.. generate some data.. and see if the CNN can catch augmentations..
"""

from settings.config import *

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

A = np.reshape(temp_dict[list(temp_dict.keys())[2]][np.random.randint(0,100)], (32, 3*32))
B = np.reshape(A, (3, 32, 32))
plt.imshow(np.swapaxes(B, 0,2))

# Augment it

# Put it in a CNN thing