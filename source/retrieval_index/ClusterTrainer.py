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
from tensorflow.examples.tutorials.mnist import input_data

from source.retrieval_index.ClusterModel import ClusterModel
from source.retrieval_index.DataSampler import DataGenerator


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

        self.sess = tf.InteractiveSession()
        self.saver = None
        self.log_save_dir = "./log/"

    def reload_model(self):
        if os.path.exists(self.log_save_dir + "checkpoint"):
            ckpt = tf.train.get_checkpoint_state(self.log_save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def start_train(self):
        mnist = input_data.read_data_sets("/Volumes/projects/DataSets/basic_dataset/", reshape=False)
        mean_data = np.mean(mnist.train.images, axis=0)

        model_param = self.train_model.build_model()

        optimizer = tf.train.AdamOptimizer(self.lr)

        with tf.control_dependencies([model_param["centers_update_op"]]):
            train_op = optimizer.minimize(model_param["total_loss"], global_step=self.global_step)

        summary_op = tf.summary.merge_all()

        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter(self.log_save_dir, self.sess.graph)

        step = self.sess.run(self.global_step)
        while step <= self.epochs * len(mnist.train.images) // self.batch_size:
            batch_images, batch_labels = mnist.train.next_batch(self.batch_size)
            _, summary_str, train_acc = self.sess.run(
                [train_op, summary_op, model_param["acc"]],
                feed_dict={
                    model_param["input_images"]: batch_images - mean_data,
                    model_param["labels"]: batch_labels,
                })
            step += 1

            writer.add_summary(summary_str, global_step=step)

            if step % 200 == 0:
                vali_image = mnist.validation.images - mean_data
                vali_acc = self.sess.run(
                    model_param["acc"],
                    feed_dict={
                        model_param["input_images"]: vali_image,
                        model_param["labels"]: mnist.validation.labels
                    })
                print(("step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".format(step, train_acc, vali_acc)))

    def save_model_log(self, epoch_id=None):
        if not os.path.isdir(self.log_save_dir):
            os.makedirs(self.log_save_dir)
        if epoch_id is not None:
            self.saver.save(self.sess, self.log_save_dir + "model.ckpt",
                            global_step=epoch_id)
        else:
            self.saver.save(self.sess, self.log_save_dir + "model.ckpt")


if __name__ == '__main__':

    cluster_model = ClusterModel()

    triple_trainer = ClusterTrainer(sample_creator=None, train_model=cluster_model)
    triple_trainer.start_train()
