# encoding: utf-8

from keras.models import Model
from keras.layers import Input, concatenate, Deconv2D, Conv2D, MaxPooling2D, Conv2DTranspose, Conv3D
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D


def size_cnn(width, filter, padding, stride):
    return (width - filter + 2*padding)/(stride) + 1

from keras.layers import Input, Dense
from keras.models import Model

def get_model(N):
    """
    Create a model...
    :param N:
    :return:
    """

    # This returns a tensor
    inputs = Input(shape=(32, 32, 3))
    conv1 = Conv2D(2**N, kernel_size=3, activation='relu', padding='same')(inputs)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pred = Deconv2D(1, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(pool2)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def unet_ae(N):
    """
    Create a model... that acts like an autoencoder.. by using convolutions
    :param N:
    :return:
    """

    # This returns a tensor
    inputs = Input(shape=(32, 32, 3))
    conv1 = Conv2D(2**N, kernel_size=3, activation='relu', padding='same')(inputs)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pred = Deconv2D(1, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(pool2)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model.fit(data, labels)  # starts training
