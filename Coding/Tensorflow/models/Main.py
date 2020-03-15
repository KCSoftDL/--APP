import os
import tensorflow as tf
import numpy as np

from tensorflow.keras import applications
# from tensorflow_core.python import keras
from tensorflow import keras
from tensorflow import data
from data_loader import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from models.Dense_model import DenseNet

# logging.set_verbosity(logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def train_model(train_filepath, test_filepath):

    import json
    with open("config.json", "r") as f:
        config = json.load(f)
    # model = applications.DenseNet121(weights=None,
    #                                input_tensor=keras.layers.Input(shape=(224,224,3),dtype =tf.float32),
    #                                pooling=None,
    #                                input_shape=(224, 224, 3),
    #                                classes=2)
    model = DenseNet(config)
    model.build((config["trainer"]["batch_size"], 224, 224, 3))
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


    # model.load_weights('path_to_saved_model')


    X_train = data_loadder(train_filepath)
    X_test = data_loadder(test_filepath)

    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # 数据预处理
    def progress(x, y):
        x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
        x = tf.reshape(x, [ 224 , 224 , 3])
        y = tf.one_hot(y, depth=10, dtype=tf.int32)
        return x, y

    # 构建dataset对象 方便对数据的管理
    # x_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000)
    # x_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    #
    # train_db = x_train.map(progress).batch(32)
    # test_db = x_test.map(progress).batch(32)
    # x_train = x_train.reshape( 224 , 224 , 3 ).astype('float32') / 255
    # x_test = x_test.reshape( 224 , 224 , 3 ).astype('float32') / 255
    #model = DenseNet.DenseNet(blocks=[6, 12, 24, 16],weights='imagenet',input_tensor=X_train,pooling='avg',input_shape=(224, 224, 3),classes=2)

    model.compile(loss=loss_object,
               optimizer=optimizer,
               metrics=['accuracy'])
    model.summary()
    history = model.fit( X_train,
           # batch_size = BATCH_SIZE,
           epochs =1000 ,
           verbose = 2,
           steps_per_epoch = 32,
           # validation_data = test_db,
           # validation_freq = 2
    )

    """
    手动定义的fit方法
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

        train_accuracy(labels, predictions)

    for epoch in range(config["trainer"]["epochs"]):
        for step, (images, labels) in tqdm(enumerate(train_loader),
                                           total=int(len(data) / config["trainer"]["batch_size"])):
            # for step, (images, labels) in tqdm(data.read_tfRecord('D:/Programming/tensorflow/data/train224.tfrecords'), total=int(len(data) / config["trainer"]["batch_size"])):
            train_step(images, labels)
        template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100
                              )
              )
        # train_accuracy.reset_states()
    """
    print('history dict:', history.history)

    test_scores = model.evaluate(X_test, verbose=2, steps= 32)
    # print('Test loss:', test_scores[0])
    # print('Test accuracy:', test_scores[1])

    savepath = "./models/DenseNet/1/"
    tf.saved_model.save(model, savepath)

    print("Success Save Model!")

    # model = model.load_weights('model.h5')
    #
    # model.save('saved_model',save_format ='tf')
    # model = model.load_weights('saved_model')

    # new_predictions = model.predict(X_test)
    # print(new_predictions)
    model.evaluate(X_test , verbose=1, steps= 32)

    imported = tf.saved_model.load(savepath)

    return model
#
# file_tfname = os.path.join(os.getcwd(), '/data/train224.tfrecords')
#
def read_tfRecord(file_tfRecord):
    '''
    该函数暂未测试
    :param file_tfRecord:  文件路径
    :return:
    '''
    reader = tf.data.TFRecordDataset(file_tfRecord)  # 打开一个TFrecord

    features = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64,default_value=3),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }


    def _parse_function(exam_proto):  # 映射函数，用于解析一条example
        return tf.io.parse_single_example(exam_proto, features)

    # reader = reader.repeat (1) # 读取数据的重复次数为：1次，这个相当于epoch
    reader = reader.shuffle (buffer_size = 2000) # 在缓冲区中随机打乱数据

    reader = reader.map (_parse_function) # 解析数据
    # image = tf.io.decode_raw(features['image_raw'],tf.float64)
    # tf_height = features['height']
    # tf_width = features['width']
    # tf_depth = features['depth']
    #
    # image = tf.reshape(image,[tf_height,tf_width,tf_depth])
    # label = tf.cast(features['label'], tf.int32)
    for image_feature in reader.take(10):
        image_raw = np.frombuffer(image_feature['image_raw'].numpy(), dtype=np.uint8)
        display.display(display.Image(data= image_raw))
    # batch  = reader.batch (batch_size = 32) # 每10条数据为一个batch，生成一个新的Dataset

    # shape = []
    # batch_data_x, batch_data_y = np.array([]), np.array([])
    # for item in batch.take(1): # 测试，只取1个batch
    #     shape = item['shape'][0].numpy()
    #     for data in item['data']: # 一个item就是一个batch
    #         img_data = np.frombuffer(data.numpy(), dtype=np.uint8)
    #         batch_data_x = np.append (batch_data_x, img_data)
    #     for label in item ['label']:
    #         batch_data_y = np.append (batch_data_y, label.numpy())
    #
    # batch_data_x = batch_data_x.reshape ([-1, shape[0], shape[1], shape[2]])
    #print (image._shape,label) # = (10, 480, 640, 3) (10,)
    # return image,label
    return reader

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" # 选择哪一块gpu, 如果是 - 1，就是调用cpu
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7 # 分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
    session = InteractiveSession(config=config)


    train_filepath = 'D:/Programming/tensorflow/models/data/train'
    test_filepath = 'D:/Programming/tensorflow/models/data/test'
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # model=train_model(train_filepath=test_filepath,test_filepath=test_filepath)


    # model = applications.DenseNet121(weights=None,
    #                                input_tensor=keras.layers.Input(shape=(224,224,3),dtype =tf.float32),
    #                                pooling=None,
    #                                input_shape=(224, 224, 3),
    #                                classes=2)
    # model.compile(loss=keras.losses.CategoricalCrossentropy(),
    #            optimizer=keras.optimizers.Adam(lr=0.01),
    #            metrics=['accuracy'])
    # x_test = data_loadder(test_filepath)

    model = train_model(train_filepath,test_filepath)

    # predict = model.predict(x_test,verbose=1)
    # print(model.predict(x_test))

    # new_model=keras.models.load_model('saved_model')

    # new_predictions = new_model.predict(x_test)
    # print(new_predictions)
    #
    # test_filepath = 'E:/DL/data/test'
    # x_test = data_loadder(test_filepath)
    # new_predictions = model.predict(x_test)

    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # x_train= x_train[0]
    # plt.imshow(x_train)
    # plt.grid(False)
    # plt.xlabel(y_train)
    # plt.show()