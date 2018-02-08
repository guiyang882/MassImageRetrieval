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
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from source.retrieval_index.ClusterModel import ClusterModel
from source.retrieval_index.DataSampler import DataGenerator


mnist = input_data.read_data_sets(
            "/Volumes/projects/DataSets/basic_dataset/mnist", reshape=False)
mean_data = np.mean(mnist.train.images, axis=0)


class ClusterTrainer:

    def __init__(self, sample_creator=None, train_model=None):
        self.sample_creator = sample_creator
        self.train_model = train_model

        self.batch_size = 256
        self.epochs = 100
        self.plot_size = 60000
        self.is_update = True
        self.lr = 0.001

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.model_param = None
        self.sess = tf.InteractiveSession()
        self.saver = None
        self.log_save_dir = "./log/cluster/"

    def reload_model(self):
        if os.path.exists(self.log_save_dir + "checkpoint"):
            ckpt = tf.train.get_checkpoint_state(self.log_save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def start_train(self):
        self.model_param = self.train_model.build_model()

        optimizer = tf.train.AdamOptimizer(self.lr)

        with tf.control_dependencies([self.model_param["centers_update_op"]]):
            train_op = optimizer.minimize(self.model_param["total_loss"], global_step=self.global_step)

        summary_op = tf.summary.merge_all()

        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter(self.log_save_dir, self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=3)

        step = self.sess.run(self.global_step)
        N_step_pre_Epoch = len(mnist.train.images) // self.batch_size
        while step <= self.epochs * N_step_pre_Epoch:
            batch_images, batch_labels = mnist.train.next_batch(self.batch_size)
            _, summary_str, train_acc = self.sess.run(
                [train_op, summary_op, self.model_param["acc"]],
                feed_dict={
                    self.model_param["input_images"]: batch_images - mean_data,
                    self.model_param["labels"]: batch_labels,
                })
            step += 1

            writer.add_summary(summary_str, global_step=step)

            if (step+1) % 200 == 0:
                vali_image = mnist.validation.images - mean_data
                vali_acc = self.sess.run(
                    self.model_param["acc"],
                    feed_dict={
                        self.model_param["input_images"]: vali_image,
                        self.model_param["labels"]: mnist.validation.labels
                    })
                print(("step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".format(step, train_acc, vali_acc)))

            if (step+1) % N_step_pre_Epoch == 0:
                self.save_model_log(step // N_step_pre_Epoch)
                self.visualize_results(step)

    def save_model_log(self, epoch_id=None):
        if not os.path.isdir(self.log_save_dir):
            os.makedirs(self.log_save_dir)
        if epoch_id is not None:
            self.saver.save(self.sess, self.log_save_dir + "model.ckpt",
                            global_step=epoch_id)
        else:
            self.saver.save(self.sess, self.log_save_dir + "model.ckpt")

    def visualize_results(self, step):
        feat = self.sess.run(
            self.model_param["cluster"],
            feed_dict={
                self.model_param["input_images"]: mnist.train.images[:10000] - mean_data
            })
        labels = mnist.train.labels[:10000]

        f = plt.figure(figsize=(16, 9))
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        for i in range(10):
            plt.plot(feat[labels == i, 0].flatten(),
                     feat[labels == i, 1].flatten(), '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.grid()
        plt.savefig("../../experiment/cluster/" + str(step) + "_vis.png")


if __name__ == '__main__':

    cluster_model = ClusterModel()

    triple_trainer = ClusterTrainer(sample_creator=None, train_model=cluster_model)
    triple_trainer.start_train()
