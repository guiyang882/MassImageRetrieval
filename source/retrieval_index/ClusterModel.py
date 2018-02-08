# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 先实现对训练数据的聚类操作，并在二维平面上进行可视化操作

import tensorflow as tf


from source.retrieval_index.BaseModel import BaseModel


class ClusterModel(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)
        with tf.name_scope("input"):
            self.input_images = tf.placeholder(tf.float32,
                                               shape=(None, 28, 28, 1),
                                               name="input_names")
            self.labels = tf.placeholder(tf.int64, shape=(None), name="labels")

        self.ratio = 0.5
        self.alpha = 0.5
        self.num_classes = 10

    def get_center_loss_tf(self, features):
        """获取center loss及center的更新op

            Arguments:
                features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
                labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
                alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
                num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

            Return：
                loss: Tensor,可与softmax loss相加作为总的loss进行优化.
                centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
                centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        """
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]
        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        centers = tf.get_variable('centers',
                                  shape=[self.num_classes, len_features],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(self.labels, [-1])

        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss = tf.nn.l2_loss(features - centers_batch)

        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = self.alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)

        return loss, centers, centers_update_op

    @staticmethod
    def inference(input_images):
        with tf.name_scope("modelbody"):
            conv1 = tf.layers.conv2d(inputs=input_images, filters=32,
                                     kernel_size=(3, 3), strides=(2, 2),
                                     padding="same", activation=tf.nn.relu,
                                     name="conv1")
            pool1 = tf.layers.max_pooling2d(
                inputs=conv1, pool_size=(2, 2), strides=(2, 2), name="pool1")
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64,
                                     kernel_size=(3, 3),
                                     strides=(2, 2), padding="same",
                                     activation=tf.nn.relu, name="conv2")
            pool2 = tf.layers.max_pooling2d(
                inputs=conv2, pool_size=(2, 2), strides=(2, 2), name="pool2")
            conv3 = tf.layers.conv2d(inputs=pool2, filters=64,
                                     kernel_size=(3, 3),
                                     strides=(1, 1), padding="same",
                                     activation=tf.nn.relu, name="conv3")
            pool3 = tf.layers.max_pooling2d(
                inputs=conv3, pool_size=(2, 2), strides=(2, 2), name="pool3")
            flatten = tf.layers.flatten(inputs=pool3, name="flatten")

            cluster_tensor = tf.layers.dense(
                inputs=flatten, units=2, activation=None, name="cluster_tensor")

            fea_activation = tf.nn.relu(cluster_tensor)
            classify_tensor = tf.layers.dense(
                inputs=fea_activation, units=10, activation=tf.nn.softmax,
                name="classify_tensor")

            return cluster_tensor, classify_tensor

    def build_model(self):
        cluster_tensor, classify_tensor = self.inference(self.input_images)

        with tf.name_scope('loss'):
            with tf.name_scope('center_loss'):
                center_loss, centers, centers_update_op = self.get_center_loss_tf(cluster_tensor)
            with tf.name_scope('softmax_loss'):
                softmax_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.labels, logits=classify_tensor))
            with tf.name_scope('total_loss'):
                total_loss = softmax_loss + self.ratio * center_loss

        with tf.name_scope('acc'):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(classify_tensor, 1), self.labels), tf.float32))

        with tf.name_scope('loss/'):
            tf.summary.scalar('CenterLoss', center_loss)
            tf.summary.scalar('SoftmaxLoss', softmax_loss)
            tf.summary.scalar('TotalLoss', total_loss)

        model_param = {
            "input_images": self.input_images,
            "labels": self.labels,
            "logits": classify_tensor,
            "cluster": cluster_tensor,
            "total_loss": total_loss,
            "acc": accuracy,
            "centers_update_op": centers_update_op
        }
        return model_param


if __name__ == '__main__':
    pass
