# encoding: utf-8

"""
Here we are going to build a simple model.. generate some data.. and see if the CNN can catch augmentations..
"""

from settings.config import *
from data_augmentation.elastic import elastic_transform, random_rotation
from models.cnn_influence.define_model import unet_ae
import numpy as np
from skimage.exposure import equalize_adapthist, equalize_hist
from keras import backend as K


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

batch_train = [0, 1, 2, 3]
batch_test = [4]
n_epoch = 10
weight_list = []
loss_list = []

# Define the model..
model = unet_ae(2)
run_counter = 0

for i_epoch in range(n_epoch):
    for i_batch in batch_train:
        run_counter += 1
        i_path_batch = list_batch[i_batch]
        temp_dict = unpickle(os.path.join(DIR_CIFAR, i_path_batch))

        counter = 0
        count_max = 10000
        count_split = int(0.2*count_max)
        orig_imag = []
        target_imag = []
        N_epoch = 10

        # Get images.. create target images..
        for i_image in temp_dict[list(temp_dict.keys())[2]]:
            if counter > count_max:
                break
            counter += 1
            A = np.reshape(i_image, (32, 3*32))
            B = np.reshape(A, (3, 32, 32))
            C = np.swapaxes(B, 0, 2)
            C_rot, _ = random_rotation(C, C)
            D = elastic_transform(C_rot)

            C_eq = equalize_hist(C)
            D_eq = equalize_hist(D)

            orig_imag.append(C_eq)
            target_imag.append(D_eq)

        orig_imag = np.array(orig_imag)
        target_imag = np.array(target_imag)

        for i in range(0, count_max, count_split):
            i_orig_imag = orig_imag[i: (i+count_split), :, :, :]
            i_target_imag = target_imag[i: (i+count_split), :, :, :]
            A = model.fit(i_orig_imag, i_target_imag)

            intm_weights = model.layers[1].get_weights()[0]
            weight_list.append(intm_weights)
            loss_list.append(A.history)


plt.plot([x['loss'][0] for x in loss_list])

test = model.predict(C[np.newaxis])

plt.imshow(C)
plt.imshow(test[0,:,:,:])

z_test = model.layers[1].get_weights()[0]
z_test.shape

from scipy.ndimage.filters import convolve

for i_fig in range(20):
    temp_weight = weight_list[i_fig]
    counter = 1
    plt.figure(i_fig)
    for i in range(3):
        for j in range(4):
            print(i,j)
            print(temp_weight[:, :, i, j])
            print(counter)
            plt.figure(1)
            plt.subplot(3, 4, counter)
            plt.imshow(temp_weight[:, :, i, j], cmap='gray')

            plt.figure(i_fig)
            plt.subplot(3, 4, counter)
            plt.imshow(convolve(C_eq[:,:,0], temp_weight[:, :, i, j]), cmap='gray')
            counter += 1

convolve(fidelity_list[0], z_test[:, :, i, j], mode='same')
fidelity_list[0].shape
# with a Sequential model
test_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output = test_output([C[np.newaxis]])
plt.imshow(fidelity_list[0][:,:,0:3])

fidelity_list