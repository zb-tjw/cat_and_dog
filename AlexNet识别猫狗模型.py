import tensorflow as tf
import numpy as np


def AlexNet(X, KEEP_PROB, NUM_CLASSES):
    """Create the network graph."""
    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    conv1 = conv(X, [5, 5, 3, 64], [64], 1, 1, name='conv1')
    norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 2, 2, 2, 2, name='pool1')  ##64*64*64
    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv(pool1, [5, 5, 64, 128], [128], 1, 1, name='conv2')
    norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 2, 2, 2, 2, name='pool2')  ##32*32*128
    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, [3, 3, 128, 256], [256], 1, 1, name='conv3')
    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, [3, 3, 256, 512], [512], 1, 1, name='conv4')
    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, [3, 3, 512, 512], [512], 1, 1, name='conv5')
    pool5 = max_pool(conv5, 2, 2, 2, 2, name='pool5')
    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 16 * 16 * 512])
    fc6 = fc(flattened, [16 * 16 * 512, 1024], [1024], name='fc6')
    fc6 = tf.nn.relu(fc6)
    dropout6 = dropout(fc6, KEEP_PROB)
    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, [1024, 2048], [2048], name='fc7')
    fc7 = tf.nn.relu(fc7)
    dropout7 = dropout(fc7, KEEP_PROB)
    # 8th Layer: FC and return unscaled activations
    fc8 = fc(dropout7, [2048, NUM_CLASSES], [NUM_CLASSES], name='fc8')
    return fc8


def conv(x, kernel_size, bias_size, stride_y, stride_x, name):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                  shape=kernel_size,
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=bias_size,
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(x, weights, strides=[1, stride_y, stride_x, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases, name=scope.name)
    return pre_activation


def fc(x, kernel_size, bias_size, name):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights',
                                  shape=kernel_size,
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=bias_size,
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(x, weights), biases, name=scope.name)
    return softmax_linear


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""

    return tf.nn.dropout(x, keep_prob)


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# --------------------------------------------------------------------------
# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# -----------------------------------------------------------------------

# 评价/准确率计算

# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。

# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy