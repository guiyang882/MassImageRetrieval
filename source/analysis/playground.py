# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_multi_gaussian():
    sampleNo = 10000
    mu = [0, 10]
    cov = [[1, 0], [0, 100]]
    x, y = np.random.multivariate_normal(mu, cov, sampleNo).T
    plt.plot(x, y, '+')
    plt.show()

def show_real_plane():
    prefix_dir = "/Users/liuguiyang/Documents/CodeProj/PyProj/experiment/pred_results/"
    for filename in os.listdir(prefix_dir):
        if filename.startswith("."):
            continue
        column_name = ["x", "y", "label"]
        pd_data = pd.read_csv(prefix_dir + filename, header=-1)
        pd_data.columns = column_name
        print(pd_data.shape)
        print(pd_data.dtypes)
        plt.scatter(pd_data["x"], pd_data["y"], pd_data["label"], c=pd_data["label"])
        plt.show()
        return

def multi_gaussian_model():
    from sklearn.mixture import GaussianMixture
    prefix_dir = "/Users/liuguiyang/Documents/CodeProj/PyProj/experiment/pred_results/"
    for filename in os.listdir(prefix_dir):
        if filename.startswith("."):
            continue
        column_name = ["x", "y", "label"]
        pd_data = pd.read_csv(prefix_dir + filename, header=-1)
        pd_data.columns = column_name
        print(pd_data.shape)
        print(pd_data.dtypes)
        gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=0)
        gmm.fit(pd_data[["x", "y"]])

        clusters = gmm.means_
        point_list = list()
        for i in range(0, len(clusters)):
            theta_min = np.pi
            point_idx = i
            for j in range(0, len(clusters)):
                if i == j: continue
                diff = clusters[j] - clusters[i]
                if diff[0] != 0.0:
                    theta = np.arctan(diff[1] / diff[0])
                elif diff[1] == 0.0:
                    theta = 0
                else:
                    theta = np.pi / 2.0

                while theta < 0:
                    theta += np.pi

                if theta < theta_min:
                    theta_min = theta
                    point_idx = j
            point_list.append((i, point_idx))
        print(point_list)

        plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1])
        for idx in range(0, len(point_list)):
            p1, p2 = point_list[idx]
            t = clusters[[p1, p2]]
            plt.plot(t[:, 0], t[:, 1])
        plt.show()
        return


def show_density():

    def __calc_density(points, radius=20):
        dest_points = copy.deepcopy(points)
        dest_points["neighbors"] = 0

        labels = pd.unique(dest_points["label"])
        for label in labels:
            class_idx = dest_points["label"] == label
            # print(class_idx)
            class_points = dest_points[["x", "y"]][class_idx].values
            # center = np.mean(class_points, axis=0)
            # print(center)
            n_samples = len(class_points)
            n_neighbors = np.zeros(n_samples)
            dist_matrix = np.zeros((n_samples, n_samples))
            for i in range(0, n_samples):
                p0 = class_points[i]
                d_vals = np.sqrt(np.sum((class_points - p0) ** 2, axis=1))
                dist_matrix[i] = d_vals
                n_ok = len(d_vals[d_vals <= radius])
                n_neighbors[i] = n_ok
            # a_max = np.max(dist_matrix, axis=1)
            # a_min = np.min(dist_matrix, axis=1)
            # a_mean = np.mean(dist_matrix, axis=1)
            # print(a_max, a_min, a_mean)
            # print(n_neighbors)
            dest_points["neighbors"][class_idx] = n_neighbors / n_samples
        return dest_points

    prefix_dir = "/Users/liuguiyang/Documents/CodeProj/PyProj/experiment/pred_results/"
    for filename in os.listdir(prefix_dir):
        if filename.startswith("."):
            continue
        column_name = ["x", "y", "label"]
        pd_data = pd.read_csv(prefix_dir + filename, header=-1)
        pd_data.columns = column_name
        pd_density = __calc_density(pd_data)
        print(pd_density)
        fig = plt.figure()
        ax = Axes3D(fig)
        # X = pd_density["x"]
        # Y = pd_density["y"]
        # X, Y = np.meshgrid(X, Y)
        ax.plot_trisurf(
            pd_density['x'], pd_density['y'], pd_density['neighbors'],
            cmap=cm.jet, linewidth=0.2)
        # ax.plot_trisurf(X, Y, pd_density['neighbors'])
        plt.show()
        return


if __name__ == '__main__':
    show_density()
