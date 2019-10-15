import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

cwd = "E:/Desktop/data/"
classes = {'cat', 'dog'}  # 预先自己定义的类别
writer = tf.python_io.TFRecordWriter('tfrecord/train.tfrecords')  # 输出成tfrecord文件
# writer = tf.python_io.TFRecordWriter('test.tfrecords')  # 输出成tfrecord文件


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


for index, name in enumerate(classes):  # 这里index就没用了，并没有dog和cat文件夹
    class_path = cwd
    # class_path = cwd + name + '\\'
    print(class_path)
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  # 每个图片的地址
        # print(img_path)  # 可以查看每张图片
        img = Image.open(img_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": _int64_feature(index),
            "img_raw": _bytes_feature(img_raw),
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()
print("writed OK")


# 生成tfrecord文件后，下次可以不用再执行这段代码！！！


def read_and_decode(filename, batch_size):  # read train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [128, 128, 3])  # reshape image to 208*208*3
    # 据说下面这行多余
    # img = tf.cast(img,tf.float32)*(1./255)-0.5
    label = tf.cast(features['label'], tf.int64)

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    num_threads=8,
                                                    capacity=100,
                                                    min_after_dequeue=60, )
    return img_batch, tf.reshape(label_batch, [batch_size])


filename = 'tfrecord/train.tfrecords'
image_batch, label_batch = read_and_decode(filename, 4) #这里应该是列表长度/4 拿到所有批次

with tf.Session() as sess:
    # print(image_batch)
    image_batch = tf.cast(image_batch, tf.uint8)  # 这里image_batch和下边的jpeg编码参数都可以是numpy格式的数据
    # print(type(image_batch))
    print(image_batch)
    print("this-->",image_batch[0, :, :, :])
    # exit()
    for i in range(4):
        print(i)
        img = tf.image.encode_jpeg(image_batch[i, :, :, :])
        print(img)  # 这里img的shape显示空()
        print(img.shape)
        exit()
        print(i)
        tf.gfile.GFile('dog_and_cat/cat_encode' + str(i) + '.jpg', 'wb').write(img.eval())  #卡在这里不往下执行了,.eval()无法执行
