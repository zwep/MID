# encoding:utf-8

"""
Define the model...
"""


import numpy as np
import os, time , datetime, sys
import tensorflow as tf


def regnet_model(learning_rate = 1E-4 , max_steps = 1000):

    with tf.name_scope('inputs') as scope:
        x = tf.placeholder(tf.float32, shape=[None, 29, 29 , 2], name="x")
        # x = tf.placeholder(tf.float32, shape=[None, None, None, 2], name="x")
        xLow = tf.placeholder(tf.float32, shape=[None, 27, 27, 2], name="xLow")
        # tf.summary.image('input', x_image, 3)
        y = tf.placeholder(tf.float32, shape=[None, 1, 1, 2], name="labels")
        # y = tf.placeholder(tf.float32, shape=[None, None, None, 2], name="labels")
        bn_training = tf.placeholder(tf.bool, name='bn_training')
        mseTrainAverage_net = tf.placeholder(tf.float32 , shape = []  )
        y_dirX=y[0,np.newaxis,:,:,0,np.newaxis]
        y_dirY=y[0,np.newaxis,:,:,1,np.newaxis]
        x_Fixed=x[0,np.newaxis,:,:,0,np.newaxis]
        x_Deformed = x[0, np.newaxis, :, :, 1, np.newaxis]
        tf.summary.image('y_dir', tf.concat((y_dirX,y_dirY), 0),2)
        tf.summary.image('Images', tf.concat((x_Fixed,x_Deformed),0), 2)
        # tf.summary.image('x_Deformed', x_Deformed, 1)

    with tf.name_scope('lateFusion') as scope:
        conv1F = tf.layers.conv2d(inputs=x[:,:,:,0,np.newaxis],filters=16,kernel_size=[3, 3], padding="valid", activation=None ,name='conv1F')
        conv1F = tf.layers.batch_normalization(conv1F, training=bn_training,  name='bn1F', scale=True)
        conv1F = tf.nn.relu(conv1F)

        conv1M = tf.layers.conv2d(inputs=x[:,:,:,1,np.newaxis],filters=16,kernel_size=[3, 3], padding="valid", activation=None ,name='conv1M')
        conv1M = tf.layers.batch_normalization(conv1M, training=bn_training,  name='bn1M', scale=True)
        conv1M = tf.nn.relu(conv1M)

        conv1FLow = tf.layers.conv2d(inputs=xLow[:,:,:,0,np.newaxis],filters=16,kernel_size=[3, 3], padding="valid", activation=None ,name='conv1FLow')
        conv1FLow = tf.layers.batch_normalization(conv1FLow, training=bn_training,  name='bn1FLow', scale=True)
        conv1FLow = tf.nn.relu(conv1FLow)

        conv1MLow = tf.layers.conv2d(inputs=xLow[:,:,:,1,np.newaxis],filters=16,kernel_size=[3, 3], padding="valid", activation=None ,name='conv1MLow')
        conv1MLow = tf.layers.batch_normalization(conv1MLow, training=bn_training,  name='bn1MLow', scale=True)
        conv1MLow = tf.nn.relu(conv1MLow)

        for i in range (2,4):
            conv1F = tf.layers.conv2d(conv1F, 16 , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'F')
            conv1F = tf.layers.batch_normalization(conv1F, training=bn_training)
            conv1F = tf.nn.relu(conv1F)

        for i in range(2, 4):
            conv1M = tf.layers.conv2d(conv1M, 16 , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'M')
            conv1M = tf.layers.batch_normalization(conv1M, training=bn_training)
            conv1M = tf.nn.relu(conv1M)

        for i in range (2,4):
            conv1FLow = tf.layers.conv2d(conv1FLow, 16 , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'FLow')
            conv1FLow = tf.layers.batch_normalization(conv1FLow, training=bn_training)
            conv1FLow = tf.nn.relu(conv1FLow)

        for i in range (2,4):
            conv1MLow = tf.layers.conv2d(conv1MLow, 16 , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'MLow')
            conv1MLow = tf.layers.batch_normalization(conv1MLow, training=bn_training)
            conv1MLow = tf.nn.relu(conv1MLow)

    with tf.name_scope('MergeFixedMoving') as scope:
        conv2 = tf.concat([conv1F, conv1M], 3)
        conv2Low = tf.concat([conv1FLow, conv1MLow], 3)

    numberOfFeatures = [25,25,25,30,30,30]
    for i in range (4,10):
        conv2Low = tf.layers.conv2d(conv2Low, numberOfFeatures[i-4] , [3, 3],  padding="valid", activation=None, name='conv'+str(i)+'Low')
        conv2Low = tf.layers.batch_normalization(conv2Low, training=bn_training)
        conv2Low = tf.nn.relu(conv2Low)

    numberOfFeatures = [25,30]
    for i in range(4, 6):
        conv2 = tf.layers.conv2d(conv2, numberOfFeatures[i-4] , [3, 3],  padding="valid", activation=None, name='conv'+str(i))
        conv2 = tf.layers.batch_normalization(conv2, training=bn_training)
        conv2 = tf.nn.relu(conv2)

    conv2 = tf.layers.max_pooling2d(conv2 , [2,2], 2, name='conv6')

    conv3 = tf.concat([conv2, conv2Low], 3)

    numberOfFeatures = [60, 70, 75, 150]
    for i in range(1, 5):
        conv3 = tf.layers.conv2d(conv3, numberOfFeatures[i-1] , [3, 3],  padding="valid", activation=None, name='convFullyConnected'+str(i))
        conv3 = tf.layers.batch_normalization(conv3, training=bn_training)
        conv3 = tf.nn.relu(conv3)

    conv4 = tf.layers.conv2d(conv3, numberOfFeatures[i-1] , [1, 1],  padding="valid", activation=None, name='convFullyConnected'+str(5))
    conv4 = tf.layers.batch_normalization(conv4, training=bn_training)
    conv4 = tf.nn.relu(conv4)

    yHat = tf.layers.conv2d(conv4, 2, [1, 1], padding="valid", activation=None, dilation_rate=(1, 1))
    mse = (tf.losses.huber_loss(y, yHat, weights=1))

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)

    print('mse shape %s ' % (mse.get_shape()))
    print('y shape %s ' % (y.get_shape()))
    yHat_dirX=yHat[0,np.newaxis,:,:,0,np.newaxis]
    yHat_dirY=yHat[0,np.newaxis,:,:,1,np.newaxis]
    tf.summary.image('yHat_dir', tf.concat((yHat_dirX, yHat_dirY), 0), 2)
    tf.summary.scalar("mse", mseTrainAverage_net)

    summ = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOGDIR + '/train'+Exp, sess.graph)
    test_writer = tf.summary.FileWriter(LOGDIR + '/test' + Exp, sess.graph)
    # tf.global_variables_initializer().run() #Otherwise you encounter this error : Attempting to use uninitialized value conv2d/kerne
    print(' total numbe of variables %s' %(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

