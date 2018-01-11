#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import cv2
import numpy as np


def unpickle(file_path):
    fo = open(file_path, "rb")
    data = pickle.load(fo, encoding="iso-8859-1")
    fo.close()
    return data


def foramt_CIFAR100(step="train"):
    cifar100_save_dir = "/Volumes/projects/ImageRetireval/dataset/CIFAR-100/src/"
    cifar100_save_dir = "/home/ai-i-liuguiyang/ImageRetireval/dataset/CIFAR-100/src/"
    cifar100_dir = "/Volumes/projects/ImageRetireval/dataset/CIFAR-100/cifar-100-python/"
    cifar100_dir = "/home/ai-i-liuguiyang/ImageRetireval/dataset/CIFAR-100/cifar-100-python/"
    cifar100_meta = unpickle(cifar100_dir + "meta")
    cifar100_file_path = cifar100_dir + step
    cifar100_data = unpickle(cifar100_file_path)

    # ['filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data']
    print(cifar100_data.keys())
    print(len(cifar100_data["filenames"]))
    print(cifar100_data["data"].shape)

    image_datas = cifar100_data["data"]
    image_names = cifar100_data["filenames"]

    cifar100_index_file = cifar100_save_dir + "index_file.csv"
    index_writer = open(cifar100_index_file, "a+")

    n_images = image_datas.shape[0]
    for idx in range(n_images):
        single_image = image_datas[idx]
        r = single_image[:1024].reshape(32, 32)
        r = np.expand_dims(r, axis=3)
        g = single_image[1024:1024*2].reshape(32, 32)
        g = np.expand_dims(g, axis=3)
        b = single_image[1024*2:].reshape(32, 32)
        b = np.expand_dims(b, axis=3)
        image = np.concatenate((b, g, r), axis=2)
        resize_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image_label = step + "_" + image_names[idx][:-3]+"jpg"
        cv2.imwrite(cifar100_save_dir + image_label, resize_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        fine_label_id = cifar100_data["fine_labels"][idx]
        fine_label = cifar100_meta["fine_label_names"][fine_label_id]
        index_writer.write("{},{}\n".format(image_label, fine_label))
    index_writer.close()


foramt_CIFAR100("train")
foramt_CIFAR100("test")