# encoding: utf-8

"""
Here we run the model...
"""

import os

# Load data... ?
load_data()

# how to define batches...

# Create model
tf_graph = regnet_model()

# Run session
with tf.Session(tf_graph) as sess:
    tf.initialize()



