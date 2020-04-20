import sys
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants

sys.stderr = open('errors.txt', 'w')

#Conversion Parameters Definition
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(
    max_workspace_size_bytes=(1 << 32))
conversion_params = conversion_params._replace(precision_mode="FP32")
# conversion_params = conversion_params._replace(maximum_cached_engiens=100)

# Converter Definition
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='pixel_only_model',
    conversion_params=conversion_params)
converter.convert()


# Input Function (necessary for conversion in Trt)
num_rounds = 10
def my_input_fn():
    for _ in range(num_rounds):
        inp1 = np.random.normal(size=(1, 1, 16, 16, 20)).astype(np.float32)
        yield inp1


converter.build(input_fn=my_input_fn)
converter.save('testtrt_out')
