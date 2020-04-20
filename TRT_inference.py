import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import pandas as pd
import h5py

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants

# Dataset reading
pixel_data = np.array(pd.read_hdf('pixel_only_data_test.h5', key='data'))
pixel_labels = np.array(pd.read_hdf('pixel_only_data_test.h5', key='labels'))

pixel_data = np.transpose(np.reshape(pixel_data, [-1,20,16,16]), (0,2,3,1))

pixel_train_data = pixel_data[:19000, ]
pixel_test_data = pixel_data[19000:, ]
pixel_train_labels = pixel_labels[:19000, ]
pixel_test_labels = pixel_labels[19000:, ]

input_data = tf.constant(pixel_test_data, dtype=tf.float32)

# Trt Model Loading
saved_model_loaded = tf.saved_model.load(
    'testtrt_out')
graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)

# Trt Inference
output = frozen_func(input_data)[0].numpy()

# Output printing
with open('out.txt', 'w') as f:
    print(output, file=f)
