#import keras_applications.densenet as DenseNet
import os
import tensorflow as tf
import numpy as np
from tensorflow_core.python.keras import applications
from tensorflow_core.python import keras
from tensorflow_core.python.platform import tf_logging as logging
import IPython.display as display
from data_loader import *

logging.set_verbosity(logging.DEBUG)


# def read_tfRecord(file_tfRecord):
#     queue = tf.train.string_input_producer([file_tfRecord])
#     reader = tf.data.TFRecordDataset(file_tfRecord) # reader = tf.TFRecordReader()
#     _,serialized_example = reader.read(queue)
#
    # features = {
    #     'image_raw': tf.FixedLenFeature([], tf.string),
    #     'height': tf.FixedLenFeature([], tf.int64),
    #     'width': tf.FixedLenFeature([], tf.int64),
    #     'depth': tf.FixedLenFeature([], tf.int64),
    #     'label': tf.FixedLenFeature([], tf.int64)
    # }
#
#     def _parse_function(exam_proto):  # 映射函数，用于解析一条example
#         return tf.io.parse_single_example(exam_proto, features)
#
#     reader = reader.map(_parse_function)
#
#     image = tf.decode_raw(features['image_raw'],tf.uint8)
#     tf_height = features['height']
#     tf_width = features['width']
#     tf_depth = features['depth']
#     #height = tf.cast(features['height'], tf.int64)
#     #width = tf.cast(features['width'], tf.int64)
#     image = tf.reshape(image,[448,448,3])
#     image = tf.cast(image, tf.float32)
#     image = tf.image.per_image_standardization(image)
#     label = tf.cast(features['label'], tf.int32)
#     print(image,label)
#     return image,label


filepath = 'D:/Programming/tensorflow/data/train/chicken'

model =  applications.DenseNet121(weights=None,
                                   input_tensor=keras.layers.Input(shape=(224,224,3),dtype =tf.float32),
                                   pooling=None,
                                   input_shape=(224, 224, 3),
                                   classes=5)
X_train,y_train=data_loadder(filepath)
#model = DenseNet.DenseNet(blocks=[6, 12, 24, 16],weights='imagenet',input_tensor=X_train,pooling='avg',input_shape=(224, 224, 3),classes=2)
model.compile(loss=keras.losses.CategoricalCrossentropy(),
               optimizer=keras.optimizers.Adam(lr=0.01),
               metrics=['sparse_categorical_accuracy'])
model.summary()
model.fit(X_train,y_train,
           batch_size = 32,
           epochs =100 ,
           verbose = 2,
           steps_per_epoch =None)

model.save('path_to_saved_model',save_format ='tf')

#
# file_tfname = os.path.join(os.getcwd(), '/data/train224.tfrecords')
#
# def read_tfRecord(file_tfRecord):
#     reader = tf.data.TFRecordDataset(file_tfname)  # 打开一个TFrecord
#
#     features = {
#         'image_raw': tf.io.FixedLenFeature([], tf.string),
#         'height': tf.io.FixedLenFeature([], tf.int64),
#         'width': tf.io.FixedLenFeature([], tf.int64),
#         'depth': tf.io.FixedLenFeature([], tf.int64,default_value=3),
#         'label': tf.io.FixedLenFeature([], tf.int64)
#     }
#
#
#     def _parse_function(exam_proto):  # 映射函数，用于解析一条example
#         return tf.io.parse_single_example(exam_proto, features)
#
#     # reader = reader.repeat (1) # 读取数据的重复次数为：1次，这个相当于epoch
#     reader = reader.shuffle (buffer_size = 2000) # 在缓冲区中随机打乱数据
#
#     reader = reader.map (_parse_function) # 解析数据
#     # image = tf.io.decode_raw(features['image_raw'],tf.float64)
#     # tf_height = features['height']
#     # tf_width = features['width']
#     # tf_depth = features['depth']
#     #
#     # image = tf.reshape(image,[tf_height,tf_width,tf_depth])
#     # label = tf.cast(features['label'], tf.int32)
#     for image_feature in reader.take(10):
#         image_raw = np.frombuffer(image_feature['image_raw'].numpy(), dtype=np.uint8)
#         display.display(display.Image(data= image_raw))
#     # batch  = reader.batch (batch_size = 32) # 每10条数据为一个batch，生成一个新的Dataset
#
#     # shape = []
#     # batch_data_x, batch_data_y = np.array([]), np.array([])
#     # for item in batch.take(1): # 测试，只取1个batch
#     #     shape = item['shape'][0].numpy()
#     #     for data in item['data']: # 一个item就是一个batch
#     #         img_data = np.frombuffer(data.numpy(), dtype=np.uint8)
#     #         batch_data_x = np.append (batch_data_x, img_data)
#     #     for label in item ['label']:
#     #         batch_data_y = np.append (batch_data_y, label.numpy())
#     #
#     # batch_data_x = batch_data_x.reshape ([-1, shape[0], shape[1], shape[2]])
#     #print (image._shape,label) # = (10, 480, 640, 3) (10,)
#     # return image,label
#     return reader




