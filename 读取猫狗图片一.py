import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


# os模块包含操作系统相关的功能，
# 可以处理文件和目录这些我们日常手动需要做的操作。因为我们需要获取test目录下的文件，所以要导入os模块。
#
# 数据构成，在训练数据中，There are 12500 cat,There are 12500 dogs,共25000张
# 获取文件路径和标签
def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 乱序后的图片和标签

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        # name的形式为['dog', '9981', 'jpg']
        # os.listdir将名字转换为列表表达
        if name[0] == 'cat':
            cats.append(file_dir + file)
            # 注意文件路径和名字之间要加分隔符，不然后面查找图片会提示找不到图片
            # 或者在后面传路径的时候末尾加两//  'D:/Python/neural network/Cats_vs_Dogs/data/train//'
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
        # 猫为0，狗为1

    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))

    # 打乱文件顺序
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    # np.hstack()方法将猫和狗图片和标签整合到一起,标签也整合到一起

    temp = np.array([image_list, label_list])
    # 这里的数组出来的是2行10列，第一行是image_list的数据，第二行是label_list的数据
    temp = temp.transpose()  # 转置
    # 将其转换为10行2列，第一列是image_list的数据，第二列是label_list的数据
    np.random.shuffle(temp)
    # 对应的打乱顺序
    image_list = list(temp[:, 0])  # 取所有行的第0列数据
    label_list = list(temp[:, 1])  # 取所有行的第1列数据，并转换为int
    # print(type(label_list[0])) # 这里标签中的数据类型是<class 'numpy.str_'>
    label_list = [int(i) for i in label_list]

    return image_list, label_list


# 生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将原来的python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)  # 强制类型转换
    label = tf.cast(label, tf.int32)

    # 生成队列。我们使用slice_input_producer()来建立一个队列，将image和label放入一个list中当做参数传给该函数
    input_queue = tf.train.slice_input_producer([image, label])
    image_contents = tf.read_file(input_queue[0])
    # 按队列读数据和标签
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 要按照图片格式进行解码。本例程中训练数据是jpg格式的，所以使用decode_jpeg()解码器，
    # 如果是其他格式，就要用其他geshi具体可以从官方API中查询。
    # 注意decode出来的数据类型是uint8，之后模型卷积层里面conv2d()要求输入数据为float32类型

    # 统一图片大小
    # 通过裁剪统一,包括裁剪和扩充
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 我的方法，通过缩小图片，采用NEAREST_NEIGHBOR插值方法
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                   align_corners=False)
    image = tf.cast(image, tf.float32)
    # 因为没有标准化，所以需要转换类型
    # image = tf.image.per_image_standardization(image)   # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,  # 线程
                                              capacity=capacity)
    # print("label_batch",label_batch)
    # print("label_batch_shape",label_batch.shape)
    # image_batch是一个4D的tensor，[batch, width, height, channels]，
    # label_batch是一个1D的tensor，[batch]。
    # 这行多余？  确实是多余，前后是一样的
    label_batch = tf.reshape(label_batch, [batch_size])
    # print("label_batch", label_batch)
    # print("label_batch_shape", label_batch.shape)
    # exit()

    return image_batch, label_batch


'''
下面代码为查看图片效果，主要用于观察图片是否打乱，你会可能会发现，图片显示出来的是一堆乱点，不用担心，这是因为你对图片的每一个像素进行了强制类型转化为了tf.float32,使像素值介于-1~1之间，若想看原图，可使用tf.uint8，像素介于0~255
'''

# print("yes")
image_list,label_list = get_files("E:/Desktop/data/")
image_batch,label_batch = train_batch,train_label_batch = get_batch(image_list,label_list,128,128,4,256)

# print("ok")
#
k = 0
for i in range(int(len(image_list) / 4)): # 这个循环是打印多少批次，如果注释掉，那么只会打印一批次，也就是四张
    with tf.Session()  as sess:
        i = 0
        coord = tf.train.Coordinator() # 线程协调器
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1: #这个while确保下边每批次只打印一次，不会重复输出某张图片
                # just plot one batch size
                image, label = sess.run([image_batch, label_batch])
                for j in np.arange(4):
                    k += 1
                    print('label: %d' % label[j])
                    img = tf.image.encode_jpeg(image[j, :, :, :])
                    tf.gfile.GFile('dog_and_cat/cat_encode' + str(k) + '.jpg', 'wb').write(img.eval())

                    # plt.imshow(image[j, :, :, :])
                    # plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
# for i in range(4):
#     sess = tf.Session()
#     image,label = sess.run([image_batch,label_batch])
#     for j in range(4):
#         print('label:%d' % label[j])
#         plt.imshow(image[j, :, :, :])
#         plt.show()
#     sess.close()