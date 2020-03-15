# -*- coding:utf-8 -*-
import tensorflow as tf

in_path = "./models/DenseNet/1/"
out_path = "./models/output.tflite"
# out_path = "./model/quantize_frozen_graph.tflite"

# 模型输入节点
# input_tensor_name = ["input"]
input_tensor_name = ["conv2d"]
input_tensor_shape = {"input": [224,224,3]}
# 模型输出节点
classes_tensor_name = ["classification_layer"]

# converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path,
#                                             input_tensor_name, classes_tensor_name,
#                                             input_shapes = input_tensor_shape)
converter = tf.lite.TFLiteConverter.from_saved_model(in_path)
# converter.post_training_quantize = True
converter.experimental_new_converter = True
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)

