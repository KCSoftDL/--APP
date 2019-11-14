
import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers import Conv2d
from tensorlayer.layers import DenseLayer
from tensorlayer.layers import FlattenLayer
from tensorlayer.layers import InputLayer
from tensorlayer.layers import MaxPool2d

from tensorlayer import logging

__all__ = [
    'VGG16',
]

class VGG16(object):
    def vgg16_net(net_in,alpha1,alpha2):
        with tf.name_scope('preprocess') as scope:
            # 减去全局均值，做归一化
            net_in.outputs = net_in.outputs * 255.0
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            net_in.outputs = net_in.outputs - mean
        """conv1"""
        network= Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network= Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network= MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """conv2"""
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv2_1')
        network_1 = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv2_2')
        network = MaxPool2d(network_1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """conv3"""
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv3_3')
        """特征谱融合模块区——底层特征融合"""
        network_1=tf.nn.dilation2d(network_1,filter=256,strides=[1,1,1,1],rates=[1,3,3,1],padding='SAME',name='dilation1')
        network_1=MaxPool2d(network_1,filter_size=(2,2),strides=(2,2),padding='SAME',name='Pool6_1')
        #代替caffe框架中的scale层
        network_1=alpha1*tf.divide(network_1,tf.norm(network_1,ord='euclidean'))
        #主分支的特征加低层特征处理后的特征谱图作为下一层输入
        network=tf.add(network,network_1,name='Eltwise1')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """conv4"""
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv4_2')
        network_1 = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv4_3')
        network = MaxPool2d(network_1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')
        """conv5"""
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',name='conv5_3')
        """特征谱融合模块区——高层特征融合"""
        network_1= tf.nn.dilation2d(network_1, filter=512, strides=[1, 1, 1, 1], rates=[1, 3, 3, 1], padding='SAME',name='dilation2')

        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')
        """fc_layer"""
        network=FlattenLayer(network,name='flatten')
        network=DenseLayer(network,n_units=4096,act=tf.nn.relu,name='fc1_relu')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc3_relu')
        return network


   # def fc_layer(net):
   #     net=FlattenLayer(net,name='flatten')
   #     net=DenseLayer(net,n_units=4096,act=tf.nn.relu,name='fc1')
   #     net = DenseLayer(net, n_units=4096, act=tf.nn.relu, name='fc2')
   #     net = DenseLayer(net, n_units=1000, act=tf.identity, name='fc3')
   #     return net


    # def restore_params(self, sess):
    #     logging.info("重写预训练模型参数")
    #     maybe_download_and_extract(
    #         'vgg16_weights.npz', 'models', 'http://www.cs.toronto.edu/~frossard/vgg16/', expected_bytes=553436134
    #     )
    #     npz = np.load(os.path.join('models', 'vgg16_weights.npz'))
    #
    #     params = []
    #     for val in sorted(npz.items()):
    #         logging.info("  Loading params %s" % str(val[1].shape))
    #         params.append(val[1])
    #         if len(self.all_params) == len(params):
    #             break
    #     tl.files.assign_params(sess, params, self.net)
    #     del params


    def __init__(self, x,reuse=None):
        with tf.variable_scope("vgg16", reuse=reuse):
            scope_name = tf.get_variable_scope().name
            self.name = scope_name + '/vgg16' if scope_name else '/vgg16'

            net_in = InputLayer(x, name='input')
            self.net=VGG16.vgg16_net
            #self.network=VGG16.fc_layer(self.net)
            self.outputs = self.net.outputs

            self.all_params = list(self.net.all_params)
            self.all_layers = list(self.net.all_layers)
            self.all_drop = dict(self.net.all_drop)

            self.print_layers = self.net.print_layers
            self.print_params = self.net.print_params
        print("Success init")
