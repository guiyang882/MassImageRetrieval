# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/1/31

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np


class SamplerBase:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def fetch_batch(self, batch_size):
        raise NotImplementedError


class AvgSampler(SamplerBase):

    def __init__(self, grouped=None):
        SamplerBase.__init__(self)
        if grouped is None:
            raise ValueError("Should Give a Grouped Value, Type is Dict !")
        self.grouped = deepcopy(grouped)
        self.unique_grouped = deepcopy(self.grouped)
        self.labels = np.array([key for key, _ in grouped.items()])
        self.unique_labels = deepcopy(self.labels)

    def step_batch(self, batch_size):
        positive_labels_idx = np.random.randint(0, len(self.unique_labels), size=batch_size)
        negative_labels_idx = np.random.randint(1, len(self.unique_labels), size=batch_size)
        negative_labels_idx += positive_labels_idx
        negative_labels_idx %= len(self.unique_labels)

        positive_labels = self.unique_labels[negative_labels_idx]
        negative_labels = self.unique_labels[positive_labels_idx]

        triples_indices = []
        remove_labels = set()
        for pos_label, neg_label in zip(positive_labels, negative_labels):
            if len(self.unique_grouped[neg_label]) == 0:
                remove_labels.add(neg_label)
                continue
            if len(self.unique_grouped[pos_label]) < 2:
                if len(self.unique_grouped[pos_label]) == 0:
                    remove_labels.add(pos_label)
                continue
            negative = np.random.choice(self.unique_grouped[neg_label])
            positive_group = self.unique_grouped[pos_label]
            m = len(positive_group)
            anchor_j = np.random.randint(0, m)
            anchor = positive_group[anchor_j]
            positive_j = (np.random.randint(1, m) + anchor_j) % m
            positive = positive_group[positive_j]
            triples_indices.append([anchor, positive, negative])

            # 将选中的数据从unique_grouped中去掉
            negative_idx = np.where(self.unique_grouped[neg_label] == negative)[0]
            self.unique_grouped[neg_label] = np.delete(self.unique_grouped[neg_label], negative_idx)
            self.unique_grouped[pos_label] = np.delete(self.unique_grouped[pos_label], [anchor_j, positive_j])

        # 在下一次采样中把已经全部使用的类别的样本的id移除出去
        remove_label_idx = np.where(self.unique_labels == remove_labels)[0]
        self.unique_labels = np.delete(self.unique_labels, remove_label_idx)

        return triples_indices

    def fetch_batch(self, batch_size):
        src_batch_size = batch_size
        total_triples_indices = list()
        cnt = 0
        while batch_size:
            cnt += 1
            triples_indices = self.step_batch(batch_size)
            batch_size -= len(triples_indices)
            total_triples_indices.extend(triples_indices)
            if cnt >= 3:
                break

        while True:
            diff_size = src_batch_size - len(total_triples_indices)
            if diff_size == 0:
                break
            supplement = total_triples_indices[:diff_size]
            total_triples_indices.extend(supplement)

        # 将更新采样的状态量，进行下一次的采样
        self.unique_labels = deepcopy(self.labels)
        self.unique_grouped = deepcopy(self.grouped)

        return np.array(total_triples_indices)
