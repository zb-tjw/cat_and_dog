import os
import numpy as np
import tensorflow as tf
# import test
import 读取猫狗图片一 as test
# import model
import LeNet5猫狗识别模型 as model
import time

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000  # 队列中元素个数
MAX_STEP = 8000
learning_rate = 0.0001  # 小于0.001

print("I'm OK")
train_dir = 'E:/Desktop/data/'  # 训练图片文件夹
logs_train_dir = 'animal_recognize'  # 保存训练结果文件夹

train, train_label = test.get_files(train_dir)

train_batch, train_label_batch = test.get_batch(train,
                                                train_label,
                                                IMG_W,
                                                IMG_H,
                                                BATCH_SIZE,
                                                CAPACITY)

# 训练操作定义
sess = tf.Session()

train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.trainning(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)

# train_label_batch = tf.one_hot(train_label_batch,2,1,0)
# 测试操作定义


summary_op = tf.summary.merge_all()

# 产生一个writer来写log文件
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 加入队列，很重要

tra_loss = .0
tra_acc = .0
# val_loss = .0
# val_acc = .0

try:
    start = time.clock()  # 计算每一个step所花的时间
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss_, tra_acc_ = sess.run([train_op, train_loss, train_acc])
        # val_loss_, val_acc_ = sess.run([test_loss, test_acc])
        # 下面这一段为我为了打印神经网络最后一层变化写的，可以不要
        '''
        train,label = sess.run([train_logits,train_label_batch])
        #print(train)
        L = []
        for i in train:
            max_ = np.argmax(i)
            L.append(max_)
        print(L)
        print(label)
        '''
        tra_loss = tra_loss + tra_loss_
        tra_acc = tra_acc + tra_acc_
        # val_loss = val_loss+val_loss_
        # val_acc = val_acc+val_acc_

        if (step + 1) % 50 == 0 and step != 0:
            end = time.clock()
            print(
                'Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step + 1, tra_loss / 50, tra_acc * 100.0 / 50))
            # print('Step %d, val loss = %.2f, val accuracy = %.2f%%' % (step, val_loss/50,val_acc*100.0/50))
            print(str(end - start))
            tra_loss = .0
            tra_acc = .0
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)

            start = time.clock()

        # 每隔2000步，保存一次训练好的模型
        if step % 2000 == 0 or step == MAX_STEP - 1:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()

coord.join(threads)
sess.close()