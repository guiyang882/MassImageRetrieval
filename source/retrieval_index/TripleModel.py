#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, copy
proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])
sys.path.insert(0, proj_dir)

import importlib
importlib.reload(sys)

import tensorflow as tf


class TripleModel:
    def __init__(self):
        self.anchor_input = None
        self.positive_input = None
        self.negative_input = None

        self._loss1 = None
        self._loss2 = None
        self._total_loss = None
        self._anchor_out = None
        self._accuracy = None

        # 配置各个任务的输出参数
        self.num_hashbit = 12
        self.num_cluster = 2
        self.num_classify = 10
        self.BATCH_SIZE = 2000

    @property
    def anchor_out(self):
        return self._anchor_out

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def triple_loss_val(self):
        return self._loss1

    @property
    def classify_loss_val(self):
        return self._loss2

    @property
    def hash_loss_val(self):
        return self._loss3

    @property
    def accuracy(self):
        return self._accuracy

    def shared_network(self, input_tensor):
        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=32,
                                 kernel_size=(3, 3), strides=(2, 2),
                                 padding="same", activation=tf.nn.relu,
                                 name="conv1")
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=(2, 2), strides=(2, 2), name="pool1")
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(3, 3),
                                 strides=(2, 2), padding="same",
                                 activation=tf.nn.relu, name="conv2")
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=(2, 2), strides=(2, 2), name="pool2")
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=(3, 3),
                                 strides=(1, 1), padding="same",
                                 activation=tf.nn.relu, name="conv3")
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3, pool_size=(2, 2), strides=(2, 2), name="pool3")
        flatten1 = tf.layers.flatten(inputs=pool3, name="flatten1")
        dense1 = tf.layers.dense(
            inputs=flatten1, units=48, activation=tf.nn.relu, name="dense1")
        dense2 = tf.layers.dense(
            inputs=dense1, units=24, activation=tf.nn.relu, name="dense2")

        classify_tensor = tf.layers.dense(
            inputs=dense2, units=self.num_classify, activation=tf.nn.softmax,
            name="classify_tensor")
        cluster_tensor = tf.layers.dense(
            inputs=dense1, units=self.num_cluster, activation=None,
            name="cluster_tensor")
        hashbit_tensor = tf.layers.dense(
            inputs=dense1, units=self.num_hashbit, activation=None,
            name="hashbit_tensor")
        return cluster_tensor, classify_tensor, hashbit_tensor

    def build_model(self):
        self.anchor_input = tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name="anchor_input")
        self.positive_input = tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name="positive_input")
        self.negative_input = tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32,name="negative_input")
        self.all_y_true_label = tf.placeholder(shape=(None, 10), dtype=tf.float32, name="y_true_label")

        with tf.variable_scope("triple"):
            self._anchor_out, classify_anchor, hash_anchor = self.shared_network(self.anchor_input)
            tf.get_variable_scope().reuse_variables()
            positive_out, classify_pos, hash_pos = self.shared_network(self.positive_input)
            tf.get_variable_scope().reuse_variables()
            negative_out, classify_neg, hash_neg = self.shared_network(self.negative_input)

        cluster_outs = [self._anchor_out, positive_out, negative_out]
        classify_outs = [classify_anchor, classify_pos, classify_neg]
        hash_outs = [hash_anchor, hash_pos, hash_neg]

        self._total_loss = self.get_total_loss(cluster_outs, classify_outs, hash_outs)
        self._accuracy = self.get_classify_accuracy(classify_outs)

    def get_total_loss(self, cluster_outs, classify_outs, hash_outs):
        loss1 = self.triplet_loss_tf(inputs=cluster_outs, margin="maxplus_closest")

        y_pred = tf.concat(classify_outs, axis=0)
        loss2 = self.classify_loss_tf(y_pred=y_pred, y_true=self.all_y_true_label)

        hash_pred = tf.concat(hash_outs, axis=0)
        loss3 = self.hash_loss_tf(hash_pred, self.all_y_true_label)

        self._loss1 = tf.reduce_mean(loss1)
        self._loss2 = tf.reduce_mean(loss2)
        self._loss3 = loss3
        return self._loss1 + self._loss2 + self._loss3

    def triplet_loss_tf(self, inputs, dist='sqeuclidean', margin='maxplus', margin_value=500):
        anchor, positive, negative = inputs
        positive_distance = tf.square(anchor - positive)
        negative_distance = tf.square(anchor - negative)
        if dist == 'euclidean':
            positive_distance = tf.sqrt(
                tf.reduce_sum(positive_distance, axis=-1, keep_dims=True))
            negative_distance = tf.sqrt(
                tf.reduce_sum(negative_distance, axis=-1, keep_dims=True))
        elif dist == 'sqeuclidean':
            positive_distance = tf.reduce_mean(positive_distance, axis=-1,
                                               keep_dims=True)
            negative_distance = tf.reduce_mean(negative_distance, axis=-1,
                                               keep_dims=True)
        pn_distance = positive_distance - negative_distance
        if margin == 'maxplus':
            loss = tf.maximum(0.0, margin_value + pn_distance)
        elif margin == "maxplus_closest":
            loss = tf.maximum(0.0, margin_value + pn_distance) + positive_distance
        elif margin == 'softplus':
            loss = tf.log(margin_value + tf.exp(pn_distance))
        elif margin == "lgy_maxplus":
            loss = tf.keras.backend.switch(
                tf.greater(pn_distance, margin_value),
                tf.square(pn_distance),
                tf.keras.backend.switch(
                    tf.greater(pn_distance, -1.0 * margin_value),
                    5 * (margin_value + pn_distance),
                    tf.maximum(0.0, tf.abs(pn_distance))))
        return tf.reduce_mean(loss)

    def classify_loss_tf(self, y_pred, y_true):
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    def hash_loss_tf(self, hash_pred, y_true, alpha=0.01):
        m = tf.constant(self.num_hashbit * 2, dtype=tf.float32, name='bi_margin')
        w_label = tf.matmul(y_true, y_true, False, True)
        r = tf.reshape(tf.reduce_sum(hash_pred * hash_pred, 1), [-1, 1])
        b = r - 2 * tf.matmul(hash_pred, hash_pred, False, True)
        p2_distance = r - 2 * tf.matmul(hash_pred, hash_pred, False,
                                        True) + tf.transpose(r)
        temp = w_label * p2_distance + (1 - w_label) * tf.maximum(m - p2_distance, 0)
        regularizer = tf.reduce_sum(tf.abs(tf.abs(hash_pred) - 1))
        d_loss = tf.reduce_sum(temp) / (self.BATCH_SIZE * (self.BATCH_SIZE - 1)) + alpha * regularizer / self.BATCH_SIZE
        return d_loss

    def get_classify_accuracy(self, classify_outs):
        with tf.name_scope("accuracy"):
            y_pred = tf.concat(classify_outs, axis=0)
            correct_prediction = tf.equal(tf.argmax(y_pred, axis=1),
                                          tf.argmax(self.all_y_true_label, axis=1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)
