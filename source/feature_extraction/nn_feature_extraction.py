#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import numpy as np

from keras.models import Model
from keras.preprocessing import image
from keras.layers import Flatten, Input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.utils.vis_utils import plot_model
import gc


res50_base_model = ResNet50(weights='imagenet', pooling=max, include_top=False)
image_input = Input(shape=(224, 224, 3), name="image_input")
x = res50_base_model(image_input)
x = Flatten()(x)
res50_model = Model(inputs=image_input, outputs=x)
res50_model.summary()
# plot_model(res50_base_model, to_file='model.png', show_shapes=True)

batch_size = 1000
# image file abs_path, label
image_dir = "/home/ai-i-liuguiyang/ImageRetireval/dataset/OxBuild/src/"
image_list_file_path = "/home/ai-i-liuguiyang/ImageRetireval/dataset/OxBuild/src/index_file.csv"
nn_feature_save_dir = "/home/ai-i-liuguiyang/ImageRetireval/dataset/OxBuild/src/"

with open(image_list_file_path, "r") as fl_reader:

    image_nn_feature_dict = dict()

    def __fetch_nn_feature(batch_image, batch_file_name):
        batch_image = np.concatenate(batch_image, axis=0)
        x = preprocess_input(batch_image)

        features = res50_model.predict(x)
        features_reduce = features.squeeze()
        for idx in range(len(batch_file_name)):
            image_nn_feature_dict[batch_file_name[idx]] = features_reduce[idx]
        # print(features_reduce)
        print(features_reduce.shape)

    batch_image, batch_file_name = [], []
    for line in fl_reader.readlines():
        image_name, image_label = line.strip().split(",")
        image_path = image_dir + image_name
        if not os.path.exists(image_path):
            print("{} not found !".format(image_path))
            continue
        image_data = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(image_data)
        x = np.expand_dims(x, axis=0)
        batch_image.append(x)
        batch_file_name.append(image_name)
        if len(batch_image) == batch_size:
            __fetch_nn_feature(batch_image, batch_file_name)
            batch_image = list()
            batch_file_name = list()

    if len(batch_image):
        __fetch_nn_feature(batch_image, batch_file_name)

    print("Before dump the data !")
    pickle.dump(image_nn_feature_dict, open(nn_feature_save_dir+"nn_features.pkl", "wb"), True)
    print("After dump the data !")

gc.collect()

