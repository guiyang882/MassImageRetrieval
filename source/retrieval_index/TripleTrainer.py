#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, copy
proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])
sys.path.insert(0, proj_dir)

import importlib
importlib.reload(sys)

import numpy as np
import tensorflow as tf

from source.retrieval_index.TripleModel import TripleModel
from source.retrieval_index.sample_pipline import DataGenerator
from source.retrieval_index.utils import plot_origin_images, plot_images, show_array


class TripleTrainer:
    def __init__(self, sample_creator=None, triple_model=None):
        self.sample_creator = sample_creator
        self.triple_model = triple_model

        self.batch_size = 1000
        self.epochs = 200
        self.plot_size = 10000
        self.is_update = False

        self.sess = tf.InteractiveSession()
        self.saver = None
        self.log_save_dir = "./log/"

    def reload_model(self):
        if os.path.exists(self.log_save_dir + "checkpoint"):
            ckpt = tf.train.get_checkpoint_state(self.log_save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def start_train(self):
        self.triple_model.build_model()
        total_loss = self.triple_model.total_loss
        train_step = tf.train.AdamOptimizer(0.01).minimize(total_loss)
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

        self.reload_model()

        for epoch_id in range(0, self.epochs):
            epoch_loss_vals = list()
            for iter in range(0, self.sample_creator.train_sample_length//self.batch_size):
                x_a, x_p, x_n, y_a, y_p, y_n = self.sample_creator.get_triples_data(self.batch_size, is_update=self.is_update)
                y_label = np.concatenate([y_a, y_p, y_n])
                _, loss_v, loss1, loss2 = self.sess.run(
                    [train_step, total_loss, self.triple_model.loss1, self.triple_model.loss2],
                    feed_dict={
                        self.triple_model.anchor_input: x_a,
                        self.triple_model.positive_input: x_p,
                        self.triple_model.negative_input: x_n,
                        self.triple_model.all_y_true_label: y_label
                    })
                epoch_loss_vals.append(loss_v)
                if iter % 50 == 0:
                    print("\t{} epoch, mean loss {}".format(epoch_id, np.mean(epoch_loss_vals)))
                    print("\tloss1: {}, loss2: {}".format(loss1, loss2))
            print("{} epoch, mean loss {}".format(epoch_id, np.mean(epoch_loss_vals)))

            # predict and show the results
            # self.show_model_resuls(epoch_id)
            self.cb_update_selected_index()
            self.save_model_log(epoch_id)

    def save_model_log(self, epoch_id):
        if not os.path.isdir(self.log_save_dir):
            os.makedirs(self.log_save_dir)
        self.saver.save(self.sess, self.log_save_dir + "model.ckpt", global_step=epoch_id)

    def show_model_resuls(self, epoch):
        xy = self.sess.run(self.triple_model.anchor_out, feed_dict={
            self.triple_model.anchor_input: self.sample_creator.X_train[:self.plot_size]
        })
        # fp = "../../experiment/triple_loss/triple_loss_result_{}.png".format(epoch)
        # show_array(255 - plot_images(self.images[:self.plot_size].squeeze(), xy), filename=fp)
        # file_name = "../../experiment/triple_loss/origin_tl_{}.png".format(epoch)
        # plot_origin_images(xy, y[:plot_size], colors, file_name)

    def cluster_one_class(self, class_id, selected_xy, one_label_image_idx):
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
        self.sample_creator.cb_update_random_selected(
            class_id,
            one_label_image_idx[anchor_idx],
            one_label_image_idx[outline_idx])
        # sys.exit(0)

    def cb_update_selected_index(self):
        xy = self.sess.run(self.triple_model.anchor_out, feed_dict={
            self.triple_model.anchor_input: self.sample_creator.X_train
        })
        self.sample_creator.cb_update_total_predict_values(xy)

        if not self.is_update:
            return

        print("total images shape is ", xy.shape)
        for class_id in range(self.sample_creator.num_classes):
            one_label_image_idx = self.sample_creator.grouped[class_id]
            selected_xy = xy[one_label_image_idx]
            self.cluster_one_class(class_id, selected_xy, one_label_image_idx)


if __name__ == '__main__':
    sample_creator = DataGenerator(dataset_name="mnist")
    sample_creator.shuffle_train_samples()

    triple_model = TripleModel()

    triple_trainer = TripleTrainer(sample_creator, triple_model)
    triple_trainer.start_train()
