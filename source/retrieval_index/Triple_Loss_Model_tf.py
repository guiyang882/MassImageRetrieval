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

import keras
import numpy as np
from collections import defaultdict
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from source.retrieval_index.utils import show_array, build_rainbow
from source.retrieval_index.utils import plot_origin_images, plot_images


# load MNIST
mnist = fetch_mldata('MNIST original')
x = mnist.data.reshape(-1, 28, 28, 1).astype(np.float32) / 255.
y = mnist.target.astype(np.int32)

# shuffle images and labels
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

grouped = defaultdict(list)
for i, label in enumerate(y):
    grouped[label].append(i)

for class_id in grouped.keys():
    grouped[class_id] = np.array(grouped[class_id])

# verify the data is formatted correctly
print(x.dtype, x.min(), x.max(), x.shape)
print(y.dtype, y.min(), y.max(), y.shape)
# print(keras.utils.to_categorical(y, 10))

# build colored versions
colors = build_rainbow(len(np.unique(y)))
colored_x = np.asarray([colors[cur_y] * cur_x for cur_x, cur_y in zip(x, y)])

# sys.exit(0)

class DataGenerator:

    def __init__(self, x, y, grouped):
        self.x = copy.deepcopy(x)
        self.y = keras.utils.to_categorical(y, 10)
        self.y = self.y.astype(np.float32)
        self.grouped = copy.deepcopy(grouped)
        self.num_classes = len(grouped)
        self.update_pos_neg_grouped = copy.deepcopy(grouped)
        self.anchor_grouped = copy.deepcopy(grouped)
        self.transformed_value = copy.deepcopy(x)

    def get_triples_data(self, batch_size, is_update=False, is_sample_cosine=True):
        if is_update:
            indices = self.get_triples_indices_with_strategy(batch_size)
        elif is_sample_cosine:
            indices = self.get_triples_indices_with_cosine(batch_size)
        else:
            indices = self.get_triples_indices(batch_size)
        return self.x[indices[:,0]], self.x[indices[:,1]], self.x[indices[:,2]], self.y[indices[:, 0]], self.y[indices[:, 1]], self.y[indices[:, 2]]

    def get_triples_indices(self, batch_size):
        positive_labels = np.random.randint(0, self.num_classes, size=batch_size)
        negative_labels = (np.random.randint(1, self.num_classes, size=batch_size) + positive_labels) % self.num_classes
        triples_indices = []
        for positive_label, negative_label in zip(positive_labels, negative_labels):
            negative = np.random.choice(self.grouped[negative_label])
            positive_group = self.grouped[positive_label]
            m = len(positive_group)
            anchor_j = np.random.randint(0, m)
            anchor = positive_group[anchor_j]
            positive_j = (np.random.randint(1, m) + anchor_j) % m
            positive = positive_group[positive_j]
            triples_indices.append([anchor, positive, negative])
        return np.asarray(triples_indices)

    def __calc_apn_cosine(self, anchor, pos, neg):
        na = neg - anchor
        pa = pos - anchor
        na = na.reshape(-1)
        pa = pa.reshape(-1)
        Lna = np.sqrt(na.dot(na))
        Lpa = np.sqrt(pa.dot(pa))
        cos_angle = na.dot(pa) / (Lna * Lpa)
        if cos_angle > 0 and cos_angle < 1:
            return True
        return False

    # 根据采样出来的三元组样本，x_ap 与 x_an 之间的夹角余弦值
    def get_triples_indices_with_cosine(self, batch_size):
        positive_labels = np.random.randint(0, self.num_classes, size=batch_size)
        negative_labels = (np.random.randint(1, self.num_classes, size=batch_size) + positive_labels) % self.num_classes
        triples_indices = []
        for positive_label, negative_label in zip(positive_labels, negative_labels):
            negative = np.random.choice(self.grouped[negative_label])
            positive_group = self.grouped[positive_label]
            m = len(positive_group)
            anchor_j = np.random.randint(0, m)
            anchor = positive_group[anchor_j]
            sample_a = self.transformed_value[anchor]
            sample_n = self.transformed_value[negative]

            # select_positive_samples = list()
            # for pos_idx in positive_group:
            #     if pos_idx == anchor:
            #         continue
            #     sample_p = self.transformed_value[pos_idx]
            #     if not self.__calc_apn_cosine(sample_a, sample_p, sample_n):
            #         select_positive_samples.append(pos_idx)
            # if len(select_positive_samples) == 0:
            #     positive_j = (np.random.randint(1, m) + anchor_j) % m
            #     positive = positive_group[positive_j]
            # else:
            #     positive = np.random.choice(select_positive_samples)

            cnt_select = 0
            while True:
                cnt_select += 1
                positive_j = (np.random.randint(1, m) + anchor_j) % m
                positive = positive_group[positive_j]
                sample_p = self.transformed_value[positive]
                if self.__calc_apn_cosine(sample_a, sample_p, sample_n):
                    break
                if cnt_select >= 50:
                    break
            
            triples_indices.append([anchor, positive, negative])
        # print("\nget_triples_indices_with_cosine 采样数据集结束")
        return np.asarray(triples_indices)

    # 设计一个全局都可以访问的随机选择的数据集合
    # 当启用策略时，可以将通过回调函数更新数据集中图像对应的索引
    # 因此在设计时需要设一个全局一致的idx
    def get_triples_indices_with_strategy(self, batch_size):
        positive_labels = np.random.randint(0, self.num_classes, size=batch_size)
        negative_labels = (np.random.randint(1, self.num_classes, size=batch_size) + positive_labels) % self.num_classes
        triples_indices = []
        for positive_label, negative_label in zip(positive_labels, negative_labels):
            negative = np.random.choice(self.update_pos_neg_grouped[negative_label])
            positive = np.random.choice(self.update_pos_neg_grouped[positive_label])
            anchor = np.random.choice(self.anchor_grouped[positive_label])
            triples_indices.append([anchor, positive, negative])
        return np.asarray(triples_indices)

    def cb_update_random_selected(self, class_id, anchor_idx, remote_pn_idx):
        print("更新了候选集合")
        self.update_pos_neg_grouped[class_id] = remote_pn_idx
        self.anchor_grouped[class_id] = anchor_idx

    def cb_update_total_predict_values(self, predict_values):
        self.transformed_value = predict_values


class TripleModel:
    def __init__(self, x, images, y, grouped):
        self.x = x
        self.images = images
        self.y = y
        self.grouped = grouped

        self.anchor_input = tf.placeholder(
            shape=(None, 28, 28, 1), dtype=tf.float32, name="anchor_input")
        self.positive_input = tf.placeholder(
            shape=(None, 28, 28, 1), dtype=tf.float32, name="positive_input")
        self.negative_input = tf.placeholder(
            shape=(None, 28, 28, 1), dtype=tf.float32, name="negative_input")
        self.all_y_true_label = tf.placeholder(
            shape=(None, 10), dtype=tf.float32, name="y_true_label")

        self._loss1 = None
        self._loss2 = None
        self._total_loss = None
        self._anchor_out = None

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def loss1(self):
        return self._loss1

    @property
    def loss2(self):
        return self._loss2

    def shared_network(self, input_tensor, mode=tf.estimator.ModeKeys.TRAIN):
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
        drop1 = tf.layers.dropout(
            inputs=pool2, rate=0.25, training=(mode == tf.estimator.ModeKeys.TRAIN))
        flatten1 = tf.layers.flatten(inputs=drop1, name="flatten1")
        dense1 = tf.layers.dense(
            inputs=flatten1, units=32, activation=tf.nn.relu, name="dense1")
        classify_tensor = tf.layers.dense(
            inputs=dense1, units=10, activation=tf.nn.softmax, name="classify_tensor")
        cluster_tensor = tf.layers.dense(
            inputs=classify_tensor, units=2, activation=None, name="cluster_tensor")
        return cluster_tensor, classify_tensor

    def build_model(self):
        with tf.variable_scope("triple") as scope:
            self._anchor_out, classify_anchor = self.shared_network(self.anchor_input)
            scope.reuse_variables()
            positive_out, classify_pos = self.shared_network(self.positive_input)
            scope.reuse_variables()
            negative_out, classify_neg = self.shared_network(self.negative_input)

            cluster_outs = [self._anchor_out, positive_out, negative_out]
            classify_outs = [classify_anchor, classify_pos, classify_neg]
        self._total_loss = self.get_total_loss(cluster_outs, classify_outs)

    def get_total_loss(self, cluster_outs, classify_outs):
        loss1 = self.triplet_loss_tf(inputs=cluster_outs)
        y_pred = tf.concat(classify_outs, axis=0)
        loss2 = self.classify_loss_tf(y_pred=y_pred, y_true=self.all_y_true_label)
        # print(loss1.get_shape())
        # print(loss2.get_shape())
        self._loss1 = tf.reduce_mean(loss1)
        self._loss2 = tf.reduce_mean(loss2)
        return self._loss1 + self._loss2

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
        # print(y_pred.get_shape())
        # print(y_true.get_shape())
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    def get_cluster_resuls(self, session, plot_size, epoch):
        xy = session.run(self._anchor_out, feed_dict={
            self.anchor_input: self.x[:plot_size]
        })
        fp = "../../experiment/triple_loss/triple_loss_result_{}.png".format(epoch)
        show_array(255 - plot_images(self.images[:plot_size].squeeze(), xy), filename=fp)
        file_name = "../../experiment/triple_loss/origin_tl_{}.png".format(epoch)
        plot_origin_images(xy, y[:plot_size], colors, file_name)

    def cluster_one_class(self, data_sampler_obj, class_id, selected_xy, one_label_image_idx):
        # kmeans_model = KMeans(n_clusters=1, init="k-means++", n_jobs=1, max_iter=1)
        # kmeans_model.fit(selected_xy)
        # t_cluster_center = kmeans_model.cluster_centers_

        # 获取每个点到聚类中心的距离，并且按照距离中心点的欧式距离从小到大排序
        t_cluster_center = np.mean(selected_xy, axis=0)
        total_len = len(selected_xy)
        diff_xy = selected_xy - t_cluster_center
        dist_xy = diff_xy[:, 0] ** 2 + diff_xy[:, 1] ** 2
        dist_xy_idx_sort = dist_xy.argsort()
        outline_idx = dist_xy_idx_sort[int(total_len * 0.8):]
        anchor_idx = dist_xy_idx_sort[:int(total_len * 0.2)]

        # print(t_cluster_center)
        # plt.scatter(selected_xy[:, 0], selected_xy[:, 1])
        # plt.scatter(t_cluster_center[:, 0], t_cluster_center[:, 1])
        # plt.show()
        # print(outline_idx)
        data_sampler_obj.cb_update_random_selected(
            class_id,
            one_label_image_idx[anchor_idx],
            one_label_image_idx[outline_idx])
        # sys.exit(0)

    def cb_update_selected_index(self, session, data_sampler_obj=None, is_update=False):
        if data_sampler_obj == None:
            return
        xy = session.run(self._anchor_out, feed_dict={
            self.anchor_input: self.x
        })
        data_sampler_obj.cb_update_total_predict_values(xy)

        if not is_update:
            return

        print("total images shape is ", xy.shape)
        for class_id in range(len(self.grouped)):
            one_label_image_idx = self.grouped[class_id]
            selected_xy = xy[one_label_image_idx]
            self.cluster_one_class(data_sampler_obj, class_id, selected_xy, one_label_image_idx)

def demo_train():
    batch_size = 2000
    epochs = 200
    plot_size = 10000
    is_update = True

    sample_train = DataGenerator(x, y, grouped)
    sess = tf.InteractiveSession()

    main_model = TripleModel(x, colored_x, y, grouped)
    main_model.build_model()
    total_loss = main_model.total_loss
    train_step = tf.train.AdamOptimizer(0.01).minimize(total_loss)
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    if os.path.exists("./log/checkpoint"):
        saver.restore(sess, "./log/model.ckpt")

    try:
        for epoch_id in range(0, epochs):
            epoch_loss_vals = list()
            for iter in range(0, len(y)//batch_size):
                x_a, x_p, x_n, y_a, y_p, y_n = sample_train.get_triples_data(
                    batch_size, is_update=is_update)
                y_label = np.concatenate([y_a, y_p, y_n])
                _, loss_v, loss1, loss2 = sess.run(
                    [train_step, total_loss, main_model.loss1, main_model.loss2],
                    feed_dict={
                        main_model.anchor_input: x_a,
                        main_model.positive_input: x_p,
                        main_model.negative_input: x_n,
                        main_model.all_y_true_label: y_label
                    })
                epoch_loss_vals.append(loss_v)
                if iter % 50 == 0:
                    print("\t{} epoch, mean loss {}".format(epoch_id, np.mean(epoch_loss_vals)))
                    print("\tloss1: {}, loss2: {}".format(loss1, loss2))
            print("{} epoch, mean loss {}".format(epoch_id, np.mean(epoch_loss_vals)))
            # predict and show the results
            main_model.get_cluster_resuls(sess, plot_size, epoch_id)
            main_model.cb_update_selected_index(sess, sample_train, is_update)
            if not os.path.isdir("./log/"):
                os.makedirs("./log/")
            saver.save(sess, "./log/model.ckpt", global_step=epoch_id)
    except KeyboardInterrupt:
        if not os.path.isdir("./log/"):
            os.makedirs("./log/")
        saver.save(sess, "./log/model.ckpt")

demo_train()
