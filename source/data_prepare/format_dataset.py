#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import cv2
import numpy as np
import scipy.io as sio

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


# foramt_CIFAR100("train")
# foramt_CIFAR100("test")

def format_Caltech_101():
    src_image_save_dir = "/Volumes/projects/ImageRetireval/dataset/Caltech_101/src/"
    caltech_101_index_file = src_image_save_dir + "index_file.csv"
    index_writer = open(caltech_101_index_file, "w")

    src_dataset_dir = "/Volumes/projects/ImageRetireval/dataset/Caltech_101/"
    src_image_dir = src_dataset_dir + "101_ObjectCategories/"
    tpl_src_annotation_dir = src_dataset_dir + "Annotations/" + "{}/annotation_{}.mat"
    categories = os.listdir(src_image_dir)
    for class_name in categories:
        if class_name.startswith("."):
            continue
        src_class_image_dir = src_image_dir + class_name + "/"
        image_name_list = os.listdir(src_class_image_dir)
        for image_name in image_name_list:
            if image_name.startswith("."):
                continue
            src_image_file_path = src_class_image_dir + image_name
            image_data = cv2.imread(src_image_file_path)
            resize_image_data = cv2.resize(image_data, (224, 224), interpolation=cv2.INTER_CUBIC)
            new_image_save_path = src_image_save_dir + class_name + "_" + image_name
            cv2.imwrite(new_image_save_path, resize_image_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            index_writer.write("{},{}\n".format(class_name + "_" + image_name, class_name))
            # image_id = ".".join(image_name.split(".")[:-1]).split("_")[-1]
            # image_anno_file_path = tpl_src_annotation_dir.format(class_name, image_id)
            # print(image_anno_file_path)
            # data = sio.loadmat(image_anno_file_path)
            # print(data)
            # sys.exit(0)

# format_Caltech_101()

def format_OxBuild():
    src_image_save_dir = "/Volumes/projects/ImageRetireval/dataset/OxBuild/src/"
    oxbuild_index_file = src_image_save_dir + "index_file.csv"
    index_writer = open(oxbuild_index_file, "w")

    src_image_dir = "/Volumes/projects/ImageRetireval/dataset/OxBuild/not_deal_src/"
    for image_name in os.listdir(src_image_dir):
        if image_name.startswith("."):
            continue
        class_name = image_name.split("_")[0]
        image_data = cv2.imread(src_image_dir + image_name)
        resize_image_data = cv2.resize(image_data, (224, 224), interpolation=cv2.INTER_CUBIC)
        new_image_save_path = src_image_save_dir + image_name
        cv2.imwrite(new_image_save_path, resize_image_data, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        index_writer.write("{},{}\n".format(image_name, class_name))

# format_OxBuild()
