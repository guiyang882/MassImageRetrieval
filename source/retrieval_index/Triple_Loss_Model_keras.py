#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, copy
proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])
sys.path.insert(0, proj_dir)

import importlib
importlib.reload(sys)

import gc

import keras
from keras.losses import categorical_crossentropy
from keras.models import Model, load_model
from keras.layers import concatenate, Reshape
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from source.retrieval_index.utils import show_array, build_rainbow, plot_images

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
        print(self.y.dtype)
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
                if cnt_select >= 100:
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


def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus', margin_value=100):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.mean(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.mean(negative_distance, axis=-1, keepdims=True)
    pn_distance = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, margin_value + pn_distance)
    elif margin == 'softplus':
        loss = K.log(margin_value + K.exp(pn_distance))
    elif margin == "lgy_maxplus":
        loss = K.switch(
            K.greater(pn_distance, margin_value), 
            K.square(pn_distance),
            K.switch(
                K.greater(pn_distance, -1.0 * margin_value),
                5 * (margin_value + pn_distance),
                K.maximum(0.0, K.abs(pn_distance))))
    return K.mean(loss)


def classify_loss(inputs, y_true):
    anchor_classify, positive_classify, negative_classify = inputs
    y_pred = concatenate([anchor_classify, positive_classify, negative_classify], axis=0)
    print(y_pred.get_shape())
    print(y_true.get_shape())
    return categorical_crossentropy(y_true, y_pred)


def build_model(input_shape, label_dim):
    base_input = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(base_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    loss_classify_layer = Dense(10, activation='softmax')(x)
    x = Dense(2, activation='linear')(loss_classify_layer)
    # x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # force the embedding onto the surface of an n-sphere
    embedding_model = Model(base_input, x, name='embedding')
    classify_model = Model(base_input, loss_classify_layer, name="classify")
    
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    y_gt_label = K.placeholder(shape=(None, label_dim), name="y_gt_label")
    # anchor_input_label = K.placeholder(shape=(None, label_dim), name="anchor_input_label")
    # positive_input_label = K.placeholder(shape=(None, label_dim), name="positive_input_label")
    # negative_input_label = K.placeholder(shape=(None, label_dim), name="negative_input_label")

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    anchor_classify = classify_model(anchor_input)
    positive_classify = classify_model(positive_input)
    negative_classify = classify_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding, 
               anchor_classify, positive_classify, negative_classify]
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(
        K.mean(triplet_loss(outputs[0:3], dist='sqeuclidean', margin='maxplus')) +
        classify_loss(outputs[3:6], y_gt_label))
    # triplet_model.add_loss(
    #     classify_loss(outputs[3:6], y_gt_label))
    triplet_model.compile(loss=None, optimizer='adam')

    return embedding_model, triplet_model


class Plotter(keras.callbacks.Callback):
    def __init__(self, embedding_model, x, images, plot_size):
        self.embedding_model = embedding_model
        self.x = x
        self.images = images
        self.plot_size = plot_size
    
    def on_epoch_end(self, epoch, logs={}):
        xy = self.embedding_model.predict(self.x[:self.plot_size])
        show_array(255-plot_images(self.images[:self.plot_size].squeeze(), xy), filename="../../experiment/triple_loss/triple_loss_result_{}.png".format(epoch))


class ReCluster(keras.callbacks.Callback):
    def __init__(self, embedding_model, x, images, grouped, data_sampler_obj=None, is_update=False):
        self.embedding_model = embedding_model
        self.x = x
        self.images = images
        self.grouped = grouped
        # 该参数负责在聚类之后更新随机选择数据的候选集合
        self.data_sampler_obj = data_sampler_obj
        self.is_update = is_update

    def cluster_one_class(self, class_id, selected_xy, one_label_image_idx):
        # print(selected_xy)
        # print(selected_xy.shape)
        # print(selected_xy.dtype)
        
        kmeans_model = KMeans(n_clusters=1, init="k-means++", n_jobs=1, max_iter=1)
        kmeans_model.fit(selected_xy)
        t_cluster_center = kmeans_model.cluster_centers_
        # 获取每个点到聚类中心的距离，并且按照距离中心点的欧式距离从小到大排序
        total_len = len(selected_xy)
        diff_xy = selected_xy - t_cluster_center
        dist_xy = diff_xy[:, 0] ** 2 + diff_xy[:, 1] ** 2
        dist_xy_idx_sort = dist_xy.argsort()
        outline_idx = dist_xy_idx_sort[int(total_len*0.8):]
        anchor_idx = dist_xy_idx_sort[:int(total_len*0.2)]

        # print(t_cluster_center)
        # plt.scatter(selected_xy[:, 0], selected_xy[:, 1])
        # plt.scatter(t_cluster_center[:, 0], t_cluster_center[:, 1])
        # plt.show()
        # print(outline_idx)
        self.data_sampler_obj.cb_update_random_selected(
            class_id, 
            one_label_image_idx[anchor_idx], 
            one_label_image_idx[outline_idx])
        # sys.exit(0)

    def on_epoch_end(self, epoch, logs={}):
        xy = self.embedding_model.predict(self.x)
        self.data_sampler_obj.cb_update_total_predict_values(xy)

        if (epoch + 1) % 5 != 0 or not self.is_update:
            return
        print("total images shape is ", xy.shape)
        for class_id in range(len(self.grouped)):
            one_label_image_idx = self.grouped[class_id]
            selected_xy = xy[one_label_image_idx]
            self.cluster_one_class(class_id, selected_xy, one_label_image_idx)

batch_size = 1000
epochs = 100
plot_size = 5000
is_update = False
input_shape = (28, 28, 1)
label_dim = 10
sample_train = DataGenerator(x, y, grouped)
embedding_model, triplet_model = build_model(input_shape, label_dim)

plotter = Plotter(embedding_model, x, colored_x, plot_size)
recluster = ReCluster(embedding_model, x, colored_x, grouped, sample_train, is_update=is_update)
def triplet_generator(x, y, batch_size):
    while True:
        x_anchor, x_positive, x_negative, y_a, y_p, y_n = sample_train.get_triples_data(batch_size, is_update=is_update)
        y_label = np.concatenate([y_a, y_p, y_n])
        # yield (
        #     {
        #         'anchor_input': x_anchor,
        #         'positive_input': x_positive,
        #         'negative_input': x_negative
        #     }, 
        #     {
        #         'anchor_input_label': y_a,
        #         'positive_input_label': y_p,
        #         'negative_input_label': y_n
        #     })
        yield (
            {
                'anchor_input': x_anchor,
                'positive_input': x_positive,
                'negative_input': x_negative
            }, 
            {
                'y_gt_label': y_label
            })

try:
    history = triplet_model.fit_generator(
        generator=triplet_generator(x, y, batch_size),
        steps_per_epoch=len(y) // batch_size,
        epochs=epochs,
        verbose=True,
        callbacks=[plotter, recluster])
except KeyboardInterrupt:
    triplet_model.save("triplet_model.h5")
    gc.collect()
    pass
triplet_model.save("triplet_model.h5")
gc.collect()


