import os
import sys
import ssl
import time
import numpy as np
from scipy.misc import imread,imresize

import tensorflow as tf
import tensorlayer as tl

from data.imagenet_classes import *
import MyMeasure_API

"""
tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

vgg=tl.models.VGG16(x)
sess=tf.InteractiveSession(config=config)
vgg.restore_params(sess)
probs=tf.nn.softmax(vgg.outputs)

vgg.print_params(False)
vgg.print_layers()

img_text=imread('data\img7.jpg',mode='RGB')
img_text=imresize(img_text,(224,224))
img_text=img_text/255.0
if ((0 <= img_text).all() and (img_text <= 1.0).all()) is False:
    raise Exception("image value should be [0, 1]")


# 输出概率最高的5各种类及其概率
starttime=time.time()
prob=sess.run(probs,feed_dict={x:[img_text]})[0]
print("花费时间: %.5s" % (time.time() - starttime))
preds =(np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p],prob[p])

"""

MyMeasure_API.Measure_Calculation()