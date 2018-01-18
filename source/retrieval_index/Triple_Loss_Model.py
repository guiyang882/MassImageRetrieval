#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, copy
proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])
sys.path.insert(0, proj_dir)

import importlib
importlib.reload(sys)

import keras
from keras.models import Model
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

# build colored versions
colors = build_rainbow(len(np.unique(y)))
colored_x = np.asarray([colors[cur_y] * cur_x for cur_x, cur_y in zip(x, y)])

class DataGenerator:

    def __init__(self, x, y, grouped):
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.grouped = copy.deepcopy(grouped)
        self.num_classes = len(grouped)
        self.update_pos_neg_grouped = copy.deepcopy(grouped)
        self.anchor_grouped = copy.deepcopy(grouped)

    def get_triples_data(self, batch_size, is_update=False):
        if is_update:
            indices = self.get_triples_indices_with_strategy(batch_size)
        else:
            indices = self.get_triples_indices(batch_size)
        return self.x[indices[:,0]], self.x[indices[:,1]], self.x[indices[:,2]]

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


def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus', margin_value=10):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.mean(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.mean(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, margin_value + loss)
    elif margin == 'softplus':
        loss = K.log(margin_value + K.exp(loss))
    return K.mean(loss)

def build_model(input_shape):
    base_input = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(base_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(2, activation='linear')(x)
#     x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x) # force the embedding onto the surface of an n-sphere
    embedding_model = Model(base_input, x, name='embedding')
    
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
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
        if (epoch + 1) % 5 != 0 or not self.is_update:
            return
        xy = self.embedding_model.predict(self.x)
        print("total images shape is ", xy.shape)
        for class_id in range(len(self.grouped)):
            one_label_image_idx = self.grouped[class_id]
            selected_xy = xy[one_label_image_idx]
            self.cluster_one_class(class_id, selected_xy, one_label_image_idx)

batch_size = 100
epochs = 30
plot_size = 5000
is_update = False
sample_train = DataGenerator(x, y, grouped)
embedding_model, triplet_model = build_model((28, 28, 1))
plotter = Plotter(embedding_model, x, colored_x, plot_size)
recluster = ReCluster(embedding_model, x, colored_x, grouped, sample_train, is_update=is_update)
def triplet_generator(x, y, batch_size):
    while True:
        x_anchor, x_positive, x_negative = sample_train.get_triples_data(batch_size, is_update=is_update)
        yield ({
            'anchor_input': x_anchor,
            'positive_input': x_positive,
            'negative_input': x_negative
            }, None)

try:
    history = triplet_model.fit_generator(
        generator=triplet_generator(x, y, batch_size),
        steps_per_epoch=len(y) // batch_size,
        epochs=epochs,
        verbose=True,
        callbacks=[plotter, recluster])
except KeyboardInterrupt:
    triplet_model.save("triplet_model.h5")
    pass
triplet_model.save("triplet_model.h5")


