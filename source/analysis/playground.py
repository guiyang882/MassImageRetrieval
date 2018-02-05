# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
        print(gmm.means_)
        labels_hat = gmm.predict(pd_data[["x", "y"]])
        print(labels_hat)
        plt.scatter(pd_data["x"], pd_data["y"], labels_hat,
                    c=pd_data["label"])
        plt.show()
        return


if __name__ == '__main__':
    multi_gaussian_model()
