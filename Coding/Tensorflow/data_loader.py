import tensorflow as tf
import pathlib
import random
import os
import IPython.display as display

AUTOTUNE = tf.data.experimental.AUTOTUNE
# filepath = 'D:/Programming/tensorflow/data/train/chicken'

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

def data_loadder(filepath):
  # filepath=os.path.join(os.getcwd(),'/data/train/chicken')
  BATCH_SIZE =32


  data_root = pathlib.Path(filepath)
  # for item in data_root.iterdir():
  #     print(item)

  all_image_paths = list(data_root.glob('*'))
  all_image_paths = [str(path) for path in all_image_paths]
  # print(all_image_paths)
  random.shuffle(all_image_paths)

  image_count = len(all_image_paths)
  print(image_count)

  label_names = sorted(item.name for item in data_root.glob('*/*') if item.is_dir())
  label_names = ['chicken']
  # print(label_names)

  label_to_index = dict((name, index) for index, name in enumerate(label_names))
  # print(label_to_index)
  all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

  print("First labels indices: ", all_image_labels[:-1])

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


  label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

  img_path = all_image_paths[0]
  img_raw = tf.io.read_file(img_path)


  image_path = all_image_paths[0]
  label = all_image_labels[0]

  path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)

  image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
  print(image_ds)

  label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
  print(label_ds)
  return image_ds,label_ds

"""
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
  
"""