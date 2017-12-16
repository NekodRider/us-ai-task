import tensorflow as tf
import numpy as np
import time


def conv(input, name, kh, kw, n_out, dh, dw, p):
    n_in = input.get_shape()[-1].value  # 通道数

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        con = tf.nn.conv2d(input, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        bias = tf.Variable(bias_init, trainable=True, name='b')
        activation = tf.nn.relu(tf.nn.bias_add(con, bias), name=scope)
        p += [kernel, bias]
        return activation


def full_connect(input, name, n_out, p):
    n_in = input.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')  # 避免死亡节点
        activation = tf.nn.relu_layer(input, kernel, bias, name=scope)
        p += [kernel, bias]
        return activation


def vgg19(input_op, keep_prob):
    param = []

    # 第一段
    conv1_1 = conv(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=param)
    conv1_2 = conv(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=param)
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

    # 第二段
    conv2_1 = conv(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=param)
    conv2_2 = conv(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=param)
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")

    # 第三段
    conv3_1 = conv(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=param)
    conv3_2 = conv(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=param)
    conv3_3 = conv(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=param)
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")

    # 第四段
    conv4_1 = conv(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=param)
    conv4_2 = conv(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=param)
    conv4_3 = conv(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=param)
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")

    # 第五段
    conv5_1 = conv(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=param)
    conv5_2 = conv(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=param)
    conv5_3 = conv(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=param)
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool5")

    # 三个全连接层 = pool5.get_shape()
    shp = pool5.get_shape()
    flattended_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattended_shape], name="resh1")

    fc6 = full_connect(resh1, name="fc6", n_out=4096, p=param)
    fc6_do = tf.nn.dropout(fc6, keep_prob, name="fc6_do")

    fc7 = full_connect(fc6_do, name="fc7", n_out=4096, p=param)
    fc7_do = tf.nn.dropout(fc7, keep_prob, name="fc7_do")

    fc8 = full_connect(fc7_do, name="fc8", n_out=1000, p=param)
    softmax = tf.nn.softmax(fc8)
    predict = tf.argmax(softmax, 1)
    return predict,  param