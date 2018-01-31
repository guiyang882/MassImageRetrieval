# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import pickle
import random
import numpy as np
from collections import defaultdict

import keras
from keras.datasets import mnist
from sklearn.cluster import KMeans
from source.retrieval_index.utils import show_array, build_rainbow
from source.retrieval_index.utils import plot_origin_images, plot_images, show_array
from source.retrieval_index.SamplerBase import AvgSampler


dataset_dir = "/Volumes/projects/ImageRetireval/dataset/"


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []  # 一会儿一对对的样本要放在这里
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        # 对第d类抽取正负样本
        for i in range(n):
            # 遍历d类的样本，取临近的两个样本为正样本对
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            # randrange会产生1~9之间的随机数，含1和9
            inc = random.randrange(1, 10)
            # (d+inc)%10一定不是d，用来保证负样本对的图片绝不会来自同一个类
            dn = (d + inc) % 10
            # 在d类和dn类中分别取i样本构成负样本对
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            # 添加正负样本标签
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def mnist_dataset_reader():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255  # 归一化
    X_test /= 255

    digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    tr_pairs, tr_y = create_pairs(X_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(X_test, digit_indices)

    input_dim = 784

    return input_dim, tr_pairs, tr_y, te_pairs, te_y


def cifar100_dataset_reader():
    feature_file_path = dataset_dir + "CIFAR-100/src/nn_features.pkl"
    cifar100_features = pickle.load(open(feature_file_path, "rb"))
    print(cifar100_features)
    print(cifar100_features.keys())


# cifar100_dataset_reader()

class DataGenerator:
    def __init__(self, dataset_name="mnist"):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.grouped = None
        self.num_classes = None
        self.train_colors = None
        self.train_colored_x = None
        self.test_colors = None
        self.test_colored_x = None
        self.epoch_id = 0

        self.m_AvgSampler = None

        if dataset_name == "mnist":
            self.num_classes = 10
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X_train = X_train.reshape(60000, 28, 28, 1)
            X_test = X_test.reshape(10000, 28, 28, 1)
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            y_train = y_train.astype("int32")
            y_test = y_test.astype("int32")
            X_train /= 255  # 归一化
            X_test /= 255
            self.X_train, self.y_train = X_train, keras.utils.to_categorical(y_train, self.num_classes)
            self.X_test, self.y_test = X_test, keras.utils.to_categorical(y_test, self.num_classes)
            self.y_train = self.y_train.astype("int32")
            self.y_test = self.y_test.astype("int32")
            print(self.X_train.shape, self.X_train.dtype)
            print(self.y_train.shape, self.y_train.dtype)

            self.shuffle_train_samples()

    @property
    def train_sample_length(self):
        return len(self.X_train)

    @property
    def test_sample_length(self):
        return len(self.X_test)
    
    def shuffle_train_samples(self):
        train_indices = np.arange(len(self.X_train))
        np.random.shuffle(train_indices)
        self.X_train = self.X_train[train_indices]
        self.y_train = self.y_train[train_indices]

        self.grouped = defaultdict(list)
        for i, label in enumerate(np.argmax(self.y_train, axis=1)):
            self.grouped[label].append(i)

        for class_id in self.grouped.keys():
            self.grouped[class_id] = np.array(self.grouped[class_id])

        self.update_pos_neg_grouped = copy.deepcopy(self.grouped)
        self.anchor_grouped = copy.deepcopy(self.grouped)
        self.transformed_value = copy.deepcopy(self.X_train)

        self.train_colors = build_rainbow(self.num_classes)
        self.train_colored_x = np.asarray(
            [self.train_colors[cur_y] * cur_x for cur_x, cur_y in
             zip(self.X_train, np.argmax(self.y_train, axis=1).T)])

        self.test_colors = build_rainbow(self.num_classes)
        self.test_colored_x = np.asarray(
            [self.test_colors[cur_y] * cur_x for cur_x, cur_y in
             zip(self.X_test, np.argmax(self.y_test, axis=1))])

    def total_random_sampling(self, batch_size):
        # 全量随机采样策略：在每轮采样中使得每个样本有且被采样一次
        if self.m_AvgSampler is None:
            self.m_AvgSampler = AvgSampler(self.grouped)
        triples_indices = self.m_AvgSampler.fetch_batch(batch_size)
        return triples_indices

    def get_triples_data(self, batch_size, is_update=False, is_sample_cosine=True):
        indices = self.total_random_sampling(batch_size)

        # if is_update and is_sample_cosine:
        #     indices = self.get_triples_indices_with_cosine(batch_size, is_update)
        # elif is_update:
        #     indices = self.get_triples_indices_with_strategy(batch_size)
        # elif is_sample_cosine:
        #     indices = self.get_triples_indices_with_cosine(batch_size)
        # else:
        #     indices = self.get_triples_indices(batch_size)
        return self.X_train[indices[:, 0]], self.X_train[indices[:, 1]], self.X_train[
            indices[:, 2]], self.y_train[indices[:, 0]], self.y_train[indices[:, 1]], \
               self.y_train[indices[:, 2]]

    def get_triples_indices(self, batch_size):
        positive_labels = np.random.randint(0, self.num_classes,
                                            size=batch_size)
        negative_labels = (np.random.randint(1, self.num_classes,
                                             size=batch_size) + positive_labels) % self.num_classes
        triples_indices = []
        for positive_label, negative_label in zip(positive_labels,
                                                  negative_labels):
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
    def get_triples_indices_with_cosine(self, batch_size, is_update=False):
        positive_labels = np.random.randint(0, self.num_classes,
                                            size=batch_size)
        negative_labels = (np.random.randint(1, self.num_classes,
                                             size=batch_size) + positive_labels) % self.num_classes
        triples_indices = []
        for positive_label, negative_label in zip(positive_labels,
                                                  negative_labels):
            if is_update:
                negative = np.random.choice(self.update_pos_neg_grouped[negative_label])
                positive_group = self.update_pos_neg_grouped[positive_label]
            else:
                negative = np.random.choice(self.grouped[negative_label])
                positive_group = self.grouped[positive_label]

            m = len(positive_group)
            anchor_j = np.random.randint(0, m)
            anchor = positive_group[anchor_j]
            sample_a = self.transformed_value[anchor]
            sample_n = self.transformed_value[negative]

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
        positive_labels = np.random.randint(0, self.num_classes,
                                            size=batch_size)
        negative_labels = (np.random.randint(1, self.num_classes,
                                             size=batch_size) + positive_labels) % self.num_classes
        triples_indices = []
        for positive_label, negative_label in zip(positive_labels,
                                                  negative_labels):
            negative = np.random.choice(
                self.update_pos_neg_grouped[negative_label])
            positive = np.random.choice(
                self.update_pos_neg_grouped[positive_label])
            anchor = np.random.choice(self.anchor_grouped[positive_label])
            triples_indices.append([anchor, positive, negative])
        return np.asarray(triples_indices)

    def cb_update_random_selected(self, class_id, anchor_idx, remote_pn_idx):
        self.update_pos_neg_grouped[class_id] = remote_pn_idx
        self.anchor_grouped[class_id] = anchor_idx

    def cb_update_total_predict_values(self, predict_values):
        print("将训练样本全量测试了一次，数据更新回样本管理空间")
        self.transformed_value = predict_values

        for class_id in range(self.num_classes):
            one_label_image_idx = self.grouped[class_id]
            self.cluster_one_class(class_id, one_label_image_idx)

    def cluster_one_class(self, class_id, one_label_image_idx):
        # 通过计算在训练数据集合中每个label的predict的分布情况，改进下次采样的策略
        # 假设数据的分布为一个高斯分布
        selected_xy = self.transformed_value[one_label_image_idx]
        # t_kmeans_center = KMeans(n_clusters=1, max_iter=100, n_jobs=-1).fit(selected_xy).cluster_centers_
        t_center_mean = np.mean(selected_xy, axis=0)
        # print("kmeans center: {}, mean center: {}".format(t_kmeans_center, t_center_mean))
        # print("class id: {}, mean center: {}".format(class_id, t_center_mean))

        diff_xy = selected_xy - t_center_mean
        dist_xy = diff_xy[:, 0] ** 2 + diff_xy[:, 1] ** 2
        # dist_xy = np.sqrt(dist_xy)
        # t_mean = np.mean(dist_xy, axis=0)
        # t_std = np.std(dist_xy)
        # t_gaussian = np.exp(-1.0 * (dist_xy - t_mean) ** 2 / (2 * (t_std ** 2))) / (t_std * np.sqrt(2 * np.pi))

        # 选择下次的正负样本的索引
        # t_down_limit = t_mean - 2 * t_std
        # t_up_limit = t_mean + 2 * t_std
        # b1 = t_gaussian > t_up_limit
        # b2 = t_gaussian < t_down_limit
        # t_select_idx = b1 | b2

        # 选择下次anchor的索引
        total_len = len(selected_xy)
        dist_xy_idx_sort = dist_xy.argsort()
        t_anchor_idx = dist_xy_idx_sort[:int(0.2*total_len)]
        t_outline_idx = dist_xy_idx_sort[int(0.75*total_len):]

        self.cb_update_random_selected(
            class_id,
            one_label_image_idx[t_anchor_idx],
            one_label_image_idx[t_outline_idx])
        # sys.exit(0)

    def show_predict_result(self, plot_size=10000, is_save_predict=False):
        self.epoch_id += 1
        save_prefix = "../../experiment/triple_loss/"
        file_name = save_prefix + "triple_loss_result_{}.png".format(self.epoch_id)
        show_array(255 - plot_images(self.train_colored_x[:plot_size].squeeze(), self.transformed_value), filename=file_name)
        # file_name = save_prefix + "origin_tl_{}.png".format(self.epoch_id)
        # plot_origin_images(self.transformed_value, np.argmax(self.y_train), self.num_classes, file_name)
        if is_save_predict:
            save_prefix = "../../experiment/pred_results/"
            if not os.path.isdir(save_prefix):
                os.makedirs(save_prefix)
            with open(save_prefix + "pred_results_{}.csv".format(self.epoch_id), "w") as writer:
                for idx in range(0, len(self.transformed_value)):
                    x = self.transformed_value[idx][0]
                    y = self.transformed_value[idx][1]
                    label = np.argmax(self.y_train[idx])
                    writer.write("{},{},{}\n".format(x, y, label))


if __name__ == '__main__':
    sample_obj = DataGenerator(dataset_name="mnist")
    cnt = 0
    while True:
        cnt += 1
        sample_obj.total_random_sampling(batch_size=2000)
        print("Sample Cnt is ", cnt)
