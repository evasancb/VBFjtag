#!/usr/bin/env python3

# Optimise and convert an VBFjTag model in SavedModel format into a TensorFlow frozen graph

import tensorflow as tf
# tfv1 = tf.compat.v1
# tfv1.disable_eager_execution()
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

import sys


model = tf.saved_model.load(sys.argv[1])

print(model.signatures["serving_default"].inputs)
print(model.signatures["serving_default"].outputs)


# here the input size needs to be specified. 1 (always) x (max nb of jets) x (nb of features)
@tf.function(input_signature=[tf.TensorSpec([1, 10, 18], tf.float32, name="MyInput")], jit_compile=True)
def infer_no_batch(x):
    # x: [1, 10, 15]
    # x_b = tf.expand_dims(x, axis=0)              # [1, 10, 15]
    y_b = model(x, training=False)             # inference path (BN in inference mode)
    # If output is batch-first and you want to remove it:
    # y = tf.nest.map_structure(lambda t: tf.squeeze(t, axis=0), y_b)
    return tf.identity(y_b, name="MyOutput")


import numpy as np
print(infer_no_batch.get_concrete_function()(np.random.rand(1, 10, 18)))


# convert_variables_to_constants_v2 is important otherwise the "resources" are not saved (and then )
pre_opti_inference = convert_variables_to_constants_v2(infer_no_batch.get_concrete_function(), aggressive_inlining=False).graph.as_graph_def(add_shapes=True)
opt_graph = optimize_for_inference_lib.optimize_for_inference(pre_opti_inference, 
                                                              ["MyInput"],
                                                              ["Identity"],
                                                              #['serving_default'], 
                                                              #[t.name.split(":")[0] for t in frozen_func.outputs], 
                                                             dtypes.float32.as_datatype_enum)


tf.io.write_graph(opt_graph, logdir=".", name="frozen_optimized.pb", as_text=False)

