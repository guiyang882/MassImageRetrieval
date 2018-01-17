#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])
sys.path.insert(0, proj_dir)

import importlib
importlib.reload(sys)

import numpy as np

from keras import backend as K
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Dropout, Reshape, Flatten
from keras.layers import merge, Conv2D, MaxPooling2D, Lambda
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

from source.retrieval_index.sample_pipline import mnist_dataset_reader


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def get_Shared_Model(input_dim):
    sharedNet = Sequential()
    sharedNet.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    sharedNet.add(Dropout(0.1))
    sharedNet.add(Dense(128, activation='relu'))
    sharedNet.add(Dropout(0.1))
    sharedNet.add(Dense(128, activation='relu'))
    # sharedNet.add(Dropout(0.1))
    # sharedNet.add(Dense(3, activation='relu'))
    # sharedNet = Sequential()
    # sharedNet.add(Dense(4096, activation="tanh", kernel_regularizer=l2(2e-3)))
    # sharedNet.add(Reshape(target_shape=(64, 64, 1)))
    # sharedNet.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding="same", activation="relu", kernel_regularizer=l2(1e-3)))
    # sharedNet.add(MaxPooling2D())
    # sharedNet.add(Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding="same", activation="relu", kernel_regularizer=l2(1e-3)))
    # sharedNet.add(MaxPooling2D())
    # sharedNet.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same", activation="relu", kernel_regularizer=l2(1e-3)))
    # sharedNet.add(Flatten())
    # sharedNet.add(Dense(1024, activation="sigmoid", kernel_regularizer=l2(1e-3)))
    return sharedNet

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

input_dim = 784
input_shape = (input_dim, )
left_input = Input(input_shape)
right_input = Input(input_shape)

sharedNet = get_Shared_Model(input_dim)
left_output = sharedNet(left_input)
right_output = sharedNet(right_input)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([left_output, right_output])
siamese_model = Model(inputs=[left_input, right_input], outputs=distance)
siamese_model.summary()
plot_model(siamese_model, to_file='siamese_model.png', show_shapes=True)

_, tr_pairs, tr_y, te_pairs, te_y = mnist_dataset_reader()

nb_epoch = 5
rms = RMSprop(lr=0.001, rho=0.9)
siamese_model.compile(loss=contrastive_loss, optimizer=rms)
siamese_model.fit(
    [tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
    validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
    batch_size=128,
    epochs=nb_epoch)

# compute final accuracy on training and test sets
pred = siamese_model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = siamese_model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.4f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.4f%%' % (100 * te_acc))

siamese_model.get_layer()