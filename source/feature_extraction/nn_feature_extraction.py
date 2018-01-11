#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

from keras.models import Model
from keras.preprocessing import image
from keras.layers import Flatten, Input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


res50_base_model = ResNet50(weights='imagenet', pooling=max, include_top=False)
image_input = Input(shape=(224, 224, 3), name="image_input")
x = res50_base_model(image_input)
x = Flatten()(x)
res50_model = Model(inputs=image_input, outputs=x)

X_train = []
Y_train = [], []
# image file abs_path, label
image_list_file_path = ""
with open(image_list_file_path, "r") as fl_reader:
	for line in fl_reader.readlines():
		image_path, image_label = line.strip().split(",")
		image_data = image.load_img(image_path, target_size=(224, 224))
		x = image.img_to_array(image_data)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		features = res50_model.predict(x)
		features_reduce = features.squeeze()
		X_train.append(features_reduce)
		Y_train.append(image_label)
		sys.exit(0)

