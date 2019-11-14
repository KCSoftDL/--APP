import os,io
import sys
import ssl
import time
import numpy as np
#from scipy.misc import imread,imresize
import tensorlayer as tl
import tensorflow as tf
from tensorlayer import logging
from tensorlayer.layers import *
from PIL import Image
import tensorboard as tb
import modle_temp

# 读取1000各种类的英文列表
from data.imagenet_classes import *

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)



if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 448, 448, 3])
    batch_size=128
    model_file_name="MyVGG16net.ckpt"
    train_file_tfRecord ='data/train224.tfrecords'
    test_file_tfRecord = 'data/test224.tfrecords'
    with tf.device('/cpu:0'):
        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess=tf.Session(config = config )
        x_train,y_train=read_tfRecord(train_file_tfRecord)
        x_test, y_test = read_tfRecord(test_file_tfRecord)
        #训练时，打乱顺序
        x_train_batch,y_train_batch=tf.train.shuffle_batch([x_train,y_train],batch_size=batch_size,capacity=2000,
                                                           min_after_dequeue=1000,num_threads=32)
        #测试不需要乱序
        # x_test_batch, y_test_batch = tf.train.batch([x_train, y_train], batch_size=batch_size, capacity=2000,
        #                                                       min_after_dequeue=1000, num_threads=32)
        with tf.device('/gpu:0'):
            network,cost,acc = modle_temp.Model_base.my_net(x_train_batch,y_train_batch,None,is_train=True)

        n_epoch = 1000
        learning_rate = 0.01
        print_freq = 1
        n_step_epoch = int(len(y_train)/batch_size)
        n_step = n_epoch * n_step_epoch
        with tf.device('/gpu:0'):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        tl.layers.initialize_global_variables(sess)

        #启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step=0
        for epoch in range(n_epoch):
            statr_time=time.time()

            #训练一个Epoch
            train_loss , train_acc , n_batch = 0 , 0 , 0
            for s in range(n_step_epoch):
                err , ac ,_ = sess.run([cost,acc,train_op]) #,feed_dict={ x:[x_train_batch] }
                step +=1; train_loss += err; train_acc += ac; n_batch +=1

            if epoch+1 ==1 or (epoch+1) % (print_freq) ==0 :

                print("epoch %d: Step %d-%d of %d took %fs " %
                      (epoch,step,step +n_step_epoch ,n_step,time.time() -statr_time))
                print(" train loss: %f" % (train_loss/n_batch ))
                print(" train acc: %f" % (train_acc / n_batch))

                # test_loss, test_acc, n_batch = 0, 0, 0
                # for _ in range(int(len(y_test)/batch_size)):
                #     err , ac =sess.run([cost_test,acc_test])
                #     test_loss += err; test_acc += ac; n_batch+=1
                # print(" test loss: %f" % (test_loss/n_batch ))
                # print(" test acc: %f" % (test_acc / n_batch))

            #保存模型为ckpt格式
            if (epoch+1) % (print_freq*50) == 0:
                print("Save model " + "!"*10)
                saver = tf.train.Saver()
                save_path = saver.save( sess,model_file_name )
        #退出多线程
        coord.request_stop()
        coord.join(threads)
        sess.close()

"""  初期代码部分
# 占位符，用于图片输入
x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# 调用完整的预训练的VGG16模型
vgg_cnn=demo.VGG16(x)
#vgg_cnn=tl.models.VGG16(x)

# 采用GPU加速，同时让TensorFlow按需分配显存。
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 初始化参数
sess=tf.InteractiveSession(config=config)
#vgg_cnn.restore_params(sess)
y = vgg_cnn.outputs
probs = tf.nn.softmax(y)
vgg_cnn.print_params(False)
vgg_cnn.print_layers()

# 载入训练好的参数，减少训练所需要的庞大时间
# tl.files.save_npz(vgg_cnn.all_params , name='model.npz')

npz = np.load(os.path.join('models', 'vgg16_weights.npz'))

params=[]

for val in sorted(npz.items()):
    logging.info("  Loading params %s" % str(val[1].shape))
    params.append(val[1])
    if len(vgg_cnn.all_params) == len(params):
        break

tl.files.assign_params(sess,params,vgg_cnn.net)


#tl.files.load_and_assign_npz(sess=sess,name='vgg16_weights.npz',network=vgg_cnn)


print("success")

# 读取测试图片
img_text=imread('data\img15.jpg',mode='RGB')
img_text=imresize(img_text,(224,224))
img_text=img_text/255.0
if ((0 <= img_text).all() and (img_text <= 1.0).all()) is False:
    raise Exception("image value should be [0, 1]")


# 输出概率最高的5个种类及其概率
start_time=time.time()
prob=sess.run(probs,feed_dict={x:[img_text]})[0]
print("花费时间: %.5s" % (time.time() - start_time))
preds =(np.argsort(prob)[::-1])[0:5]
#print('pearmain',1-prob[1])
for p in preds:
    print(class_names[p],prob[p])
"""