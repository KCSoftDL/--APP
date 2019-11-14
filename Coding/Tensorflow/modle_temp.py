import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import tf_logging as logging

from tensorlayer import logging
from tensorlayer.layers import Conv2d
from tensorlayer.layers import DenseLayer
from tensorlayer.layers import FlattenLayer
from tensorlayer.layers import InputLayer
from tensorlayer.layers import MaxPool2d
from tensorlayer.layers import LayerNormLayer


class Model_base(object):

    """L2正则化源码，由于tensorflow中取消了这一个，又没找到可替换的，就直接套用"""
    def l2_regularizer(scale, scope=None):
        """Returns a function that can be used to apply L2 regularization to weights.
        Small values of L2 can help prevent overfitting the training data.
        Args:
          scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
          scope: An optional scope name.
        Returns:
          A function with signature `l2(weights)` that applies L2 regularization.
        Raises:
          ValueError: If scale is negative or if scale is not a float.
        """
        if isinstance(scale, numbers.Integral):
            raise ValueError('scale cannot be an integer: %s' % (scale,))
        if isinstance(scale, numbers.Real):
            if scale < 0.:
                raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                                 scale)
            if scale == 0.:
                logging.info('Scale of 0 disables regularizer.')
                return lambda _: None

        def l2(weights):
            """Applies l2 regularization to weights."""
            with ops.name_scope(scope, 'l2_regularizer', [weights]) as name:
                my_scale = ops.convert_to_tensor(scale,
                                                 dtype=weights.dtype.base_dtype,
                                                 name='scale')
                return standard_ops.multiply(my_scale, nn.l2_loss(weights), name=name)

        return l2

    def fun(x,y):
        return tf.math.multiply(x,y)

    #框架主体
    def partnetwork(net_in):
        #con1、2、3均为VGG16原结构不变，该框架主要是取con2最后一个卷积层结果与con3最后进行融合，然后再作为输入继续VGG16的卷积，
        # 后续同理取最后卷积结果和中层进行融合作为最终结果输出
        with tf.name_scope('preprocess') as scope:
            # 减去全局均值，做归一化
            net_in=tf.cast(net_in, tf.float32)
            net_in = net_in * 255.0
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            net_in = net_in - mean
        alpha1=tf.constant(0.01, shape=[1,1,1,3],dtype=tf.float32, name='alpha1')
        alpha2 = tf.constant(0.0001, dtype=tf.float32, name='alpha2')
        net_in= InputLayer(net_in,name='input' )
        """conv1"""
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """conv2"""
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv2_1')
        network_1 = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                           name='conv2_2')
        network = MaxPool2d(network_1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """conv3"""
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                         name='conv3_3')
        """低层特征融合"""
        # network_1 = tf.nn.dilation2d(network_1, filter=256, strides=[1, 1, 1, 1], rates=[1, 3, 3, 1], padding='SAME',
        #                              name='dilation1')
        network_1=Conv2d(network_1,n_filter=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu, padding='SAME',
                         name='conv6_1')

        network_1 = MaxPool2d(network_1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='Pool6_1')
        # 代替caffe框架中的scale层
        #network_1=My_ScaleLayer.myscale(trainable=True,name='scale_1')
        network_1 = LayerNormLayer(network_1,scale=True,trainable=True,name="scale_1")
        network_1 = tl.layers.LambdaLayer(Model_base.fun(network_1,alpha1),name='lambda_1') #tf.multiply(network_1,alpha1)
        # 主分支的特征加低层特征处理后的特征谱图作为下一层输入
        network = tf.add(network, network_1, name="Eltwise1")
        #network = tl.layers.ElementwiseLayer(combine_fn=tf.add,name="Eltwise1")([network, network_1])
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')

        if tf.TensorShape(net_in)==[None, 224, 224, 3]:
            """conv4"""
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             name='conv4_1')
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             name='conv4_2')
            network_1 = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,padding='SAME',
                             name='conv4_3')
            network = MaxPool2d(network_1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')
            """conv5"""
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             name='conv5_1')
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             name='conv5_2')
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             name='conv5_3')
            """高层特征谱图融合"""
            network_1= MaxPool2d(network_1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool7')
            # 代替caffe框架中的scale层
            #network_1 = My_ScaleLayer.myscale(trainable=True, name='scale_2')
            network_1 = LayerNormLayer(network_1, scale=True, trainable=True, name="scale_1")
            network_1 = tl.layers.LambdaLayer(Model_base.fun(network_1, alpha2), name='lambda_2')#network_1 = tf.multiply(network_1, alpha2)
            # 主分支的特征加低层特征处理后的特征谱图作为下一层输入
            network = tf.add(network, network_1, name="Eltwise2")
            network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')

        elif tf.TensorShape(net_in)==[None, 448, 448, 3]:
            """conv4"""
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             name='conv4_1')
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             name='conv4_2')
            #network_1=tf.nn.atrous_conv2d(network,filters=256,rate=1,padding='SAME',name='dilation1')
            network_1 = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             dilation_rate=1,name='conv4_3')
            network = MaxPool2d(network_1, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')
            """conv5"""
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             dilation_rate=1,name='conv5_1')
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             dilation_rate=1,name='conv5_2')
            network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                             dilation_rate=1,name='conv5_3')
            """高层特征谱图融合"""
            pad2d = tl.layers.ZeroPad2d(padding=( 4 , 4 ))(network_1)
            network_1=Conv2d(pad2d,n_filter=512,filter_size=(3,3),strides=(1,1),padding='VALID')
            network_1= MaxPool2d(network_1, filter_size=(3, 3), strides=(3, 3), padding='SAME', name='pool7')
            # 代替caffe框架中的scale层
            #network_1 = My_ScaleLayer.myscale(trainable=True, name='scale_2')
            network_1 = LayerNormLayer(network_1, scale=True, trainable=True, name="scale_2")
            network_1 = tl.layers.LambdaLayer(Model_base.fun(network_1, alpha2), name='lambda_1')#network_1 = tf.multiply(network_1, alpha2)
            # 主分支的特征加低层特征处理后的特征谱图作为下一层输入
            network = tf.add(network, network_1, name="Eltwise2")
            network = MaxPool2d(network, filter_size=(3, 3), strides=(3, 3), padding='SAME', name='pool5')
        else:
            try:
                if tf.TensorShape(net_in) != [None, 448, 448, 3] or [None, 224, 224, 3]:
                    raise Exception(" input Error !");
            except Exception as e:
                print(e)
        return network

    #多尺度融合应用+全连接层
    def my_net(net_in,y_,reuse,is_train):

        x1 = tf.image.central_crop(net_in, 0.5)
        x2 = net_in

        # x2 = imresize(x2, (448, 448))
        # x2 = tf.cast(net_in, tf.uint8)
        # x2 = tf.reshape(x2,[448,448,3])
        # x2 = tf.cast(x2, tf.float32)

        network1 = Model_base.partnetwork(x1)
        network2 = Model_base.partnetwork(x2)

        network = tf.add(network1,network2,name='Eltwise3')
        """fc_layer"""
        network = FlattenLayer(network,name='flatten')
        network = DenseLayer(network,n_units=4096,act=tf.nn.relu,name='fc1_relu')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
        network = DenseLayer(network, n_units=5, act=tf.identity, name='fc3_relu')

        # network.partnet1=network1
        # network.partnet2=network2
        # network.all_layers=list(network.all_layers)+list(network.partnet1.all_layers)

        y=network.outputs

        ce=tl.cost.cross_entropy(y,y_,name='cost')
        L2=0
        for p in tl.layers.get_variables_with_name('relu/W',True,True):
            L2+=Model_base.l2_regularizer(0.004)(p)
        cost=ce+L2

        correct=tf.equal(tf.cast(tf.arg_max(y,1),tf.int32),y_)
        acc=tf.reduce_mean(tf.cast(correct,tf.float32))

        return network,cost,acc

    def __init__(self,x,y,reuse=None):
        with tf.variable_scope("mynet", reuse=reuse):
            scope_name = tf.get_variable_scope().name
            self.name = scope_name + '/mynet' if scope_name else '/mynet'

            net_in = InputLayer(x, name='input')

            self.net,self.cost,self.acc=Model_base.my_net(net_in)

            self.outputs = self.net.outputs

            self.all_params = list(self.net.all_params)
            self.all_layers = list(self.net.all_layers)
            self.all_drop = dict(self.net.all_drop)

            self.print_layers = self.net.print_layers
            self.print_params = self.net.print_params
