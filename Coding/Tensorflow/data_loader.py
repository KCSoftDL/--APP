import tensorflow as tf
import pathlib
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mp

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
# filepath = 'D:/Programming/tensorflow/data/train/chicken'

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def preprocess_image(image):
#   image = tf.image.decode_jpeg(image, channels=3)
#   # image = tf.image.decode_gif(image)
#   image = tf.image.convert_image_dtype(image, tf.float32)   # normalize to [0,1] range 即 image /= 255.0
#   image = tf.image.resize(image, [224, 224] )
#   return image

def preprocess_image(image):
    # # cast image to float32 type
    # image = tf.cast(image, tf.float32)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)   # normalize to [0,1] range 即 image /= 255.0
    # resize images
    image = tf.image.resize(image, [224,224])
    # normalize according to training data stats
    image = (image - [125.307, 122.950, 113.865]) / [62.993,62.088,66.705]
    # data augmentation
    image = augment(image)

    return image

def augment(image):
    """ Image augmentation """
    image = _random_flip(image)
    return image


def _random_flip(image):
    """ Flip augmentation randomly"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

def data_loadder(filepath):
  # filepath=os.path.join(os.getcwd(),'/data/train/chicken')

  data_root = pathlib.Path(filepath)
  # for item in data_root.iterdir():
  #     print(item)

  all_image_paths = list(data_root.glob('*/*'))
  all_image_paths = [str(path) for path in all_image_paths]
  print(all_image_paths)
  # random.shuffle(all_image_paths)

  image_count = len(all_image_paths)
  print(image_count)

  label_names = sorted(item.name for item in data_root.glob('*') if item.is_dir())
  # label_names = ['potato_slips']
  print("label_names:",label_names)

  label_to_index = dict((name, index) for index, name in enumerate(label_names))
  print("label_to_index:",label_to_index)
  all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

  # all_image_labels =[]
  # for i in range(image_count):
  #   all_image_labels.append(1)
  print("First labels indices: ", all_image_labels[:-1])

  print(len(all_image_labels))
  # attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
  # attributions = [line.split(' CC-BY') for line in attributions]
  # attributions = dict(attributions)
  #
  # def caption_image(image_path):
  #     image_rel = pathlib.Path(image_path).relative_to(data_root)
  #     return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])

  # for n in range(3):
  #   image_path = random.choice(all_image_paths)
  #   display.display(display.Image(image_path))


  # label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

  img_path = all_image_paths[0]
  img_raw = tf.io.read_file(img_path)


  image_path = all_image_paths[0]
  label = all_image_labels[0]

  # plt.imshow( load_and_preprocess_image(img_path) )
  # plt.grid(False)
  # plt.xlabel("1")
  # plt.show()
  # print()

  path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)

  image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

  label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
  print(label_ds)

  """
  # 将数据集和标签打包
  image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
  print(image_label_ds)
  """

  ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

  # 元组被解压缩到映射函数的位置参数中
  def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

  image_label_ds = ds.map(load_and_preprocess_from_path_label)
  print(image_label_ds)

  # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据被充分打乱。
  image_label_ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
  # image_ds = image_ds.shuffle(buffer_size = image_count )
  # image_ds = image_ds.repeat()

  image_label_ds = image_label_ds.batch(BATCH_SIZE)
  # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch
  image_label_ds =image_label_ds.prefetch(buffer_size= AUTOTUNE )
  print(image_label_ds)

  return image_label_ds

def read_tfRecord(file_tfRecord):
    '''
    读取TfRecord文件
    :param file_tfRecord:
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
        # display.display(display.Image(data= image_raw))

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

def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecords(image_ds,label_ds):
  # 从原始图片数据中建造出一个TFRecord文件：
  tfrec = tf.data.experimental.TFRecordWriter('tf2_train_new.tfrec')
  tfrec.write(image_ds)

  # 建立一个从TFRecord文件读取的数据集，并使用之前定义的preprocess_image函数对图像进行解码/重新格式化：
  image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)

  # 压缩该数据集和你之前的定义的标签数据集以得到期望的(图片,标签)对：
  ds = tf.data.Dataset.zip( (image_ds, label_ds) )
  ds = ds.apply( tf.data.experimental.shuffle_and_repeat(buffer_size=image_count) )
  ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

  # 制作数据集
  paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
  image_ds = paths_ds.map(load_and_preprocess_image)

if __name__ == "__main__":
    filepath ="D:/Programming/tensorflow/models\data/train/xihongshichaodan/xhs (6).jpg"

    img =tf.io.read_file(filepath)

    img_data_jpg = tf.image.decode_jpeg(img, channels=3)

    plt.figure(1)
    plt.imshow(img_data_jpg)

    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)

    # resize images
    img_data_jpg = tf.image.resize(img_data_jpg, [224, 224])

    # 垂直翻转图片
    img_1 = tf.image.flip_up_down(img_data_jpg)
    # 水平翻转图片
    img_2 = tf.image.flip_left_right(img_data_jpg)
    # img_3 = tf.image.transpose_image(img_data_jpg)


    # 随机设置图片的对比度
    random_contrast = tf.image.random_contrast(img_data_jpg, lower=0.8, upper=1.5)

    # 随机设置图片的饱和度
    random_satu = tf.image.random_saturation(img_data_jpg, lower=0.8, upper=1.5)

    #随机设置图片的亮度
    # random_brightness = tf.image.random_brightness(img_data_jpg,max_delta=1.3)
    random_brightness = tf.image.adjust_brightness(img_data_jpg,delta=0.2)
    random_brightness2 = tf.image.adjust_brightness(img_data_jpg, delta=-0.2)

    # plt.figure(2)
    # plt.imshow(random_contrast)
    # plt.figure(3)
    # plt.imshow(random_satu)
    # plt.show()
    # plt.figure(4)
    # plt.imshow(random_brightness2)
    # plt.show()

    mp.imsave("D:/Programming/tensorflow/models\data/train/xihongshichaodan/xhs (6)_random_brightness2.jpg",random_brightness2)

