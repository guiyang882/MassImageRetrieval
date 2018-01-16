#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# image file abs_path, label
dataset_prefix_dir = "/Volumes/projects/ImageRetireval/dataset/"
# dataset_prefix_dir = "/home/ai-i-liuguiyang/ImageRetireval/dataset/"
dataset_name = "Caltech_101"
image_dir = dataset_prefix_dir + dataset_name + "/src/"
image_list_file_path = dataset_prefix_dir + dataset_name + "/src/index_file.csv"
nn_feature_file_path = dataset_prefix_dir + dataset_name + "/src/nn_features.pkl"

nn_features = pickle.load(open(nn_feature_file_path, "rb"))
labels, features = [], []
for file_name, file_feature in nn_features.items():
	t_labels = "_".join(file_name.split("_")[:-2])
	labels.append(t_labels)
	features.append(file_feature)
labels_idx = list(set(labels))
np_features = np.array(features, dtype=np.float64)
np_feature_labels = list()
for i in range(len(labels)):
	np_feature_labels.append(labels_idx.index(labels[i]))
np_feature_labels = np.array(np_feature_labels)
print(np_features.shape)
print(np_features.dtype)
print(len(labels_idx))


def analysis_KMeans():
	mean_distortions = []
	K = len(labels_idx)
	K_range = range(320, 1000)
	for k in K_range:
		print("Cluster k is {}".format(k))
		kmeans_model = KMeans(n_clusters=k, init="k-means++", n_jobs=-1)
		kmeans_model.fit(np_features)
		t_distortions = sum(
			np.min(cdist(np_features, kmeans_model.cluster_centers_, 'euclidean'), axis=1)) / np_features.shape[0]
		mean_distortions.append(t_distortions)

	with open("./kmeans_cluster.csv", "a+") as wh:
		for idx in range(len(K_range)):
			wh.write("{},{}\n".format(K_range[idx], mean_distortions[idx]))

	# plt.plot(K_range, mean_distortions, 'bx-')
	# plt.xlabel('k')
	# plt.ylabel(u'Avgerage distortion degree')
	# plt.title(u'Elbows rule to select the best K value')
	# plt.savefig("kmeans_cluster.png")


def analysis_PCA(is_show=True):
	'''
	[0.050000000000000003, 0.10000000000000001, 0.15000000000000002, 0.20000000000000001, 0.25, 0.30000000000000004, 0.35000000000000003, 0.40000000000000002, 0.45000000000000001, 0.5, 0.55000000000000004, 0.60000000000000009, 0.65000000000000002, 0.70000000000000007, 0.75, 0.80000000000000004, 0.85000000000000009, 0.90000000000000002, 0.95000000000000007] 
	[1, 2, 3, 5, 8, 11, 14, 19, 26, 34, 44, 58, 77, 104, 143, 202, 298, 471, 824]

	[0.94999999999999996, 0.95999999999999996, 0.96999999999999997, 0.97999999999999998, 0.98999999999999999] 
	[824, 940, 1085, 1274, 1542]
	'''
	if is_show:
		x = [0.050000000000000003, 0.10000000000000001, 0.15000000000000002, 0.20000000000000001, 0.25, 0.30000000000000004, 0.35000000000000003, 0.40000000000000002, 0.45000000000000001, 0.5, 0.55000000000000004, 0.60000000000000009, 0.65000000000000002, 0.70000000000000007, 0.75, 0.80000000000000004, 0.85000000000000009, 0.90000000000000002, 0.95000000000000007]
		x.extend([0.94999999999999996, 0.95999999999999996, 0.96999999999999997, 0.97999999999999998, 0.98999999999999999])
		y = [1, 2, 3, 5, 8, 11, 14, 19, 26, 34, 44, 58, 77, 104, 143, 202, 298, 471, 824]
		y.extend([824, 940, 1085, 1274, 1542])

		x.insert(0, 0.0)
		y.insert(0, 0.0)
		x.append(1.0)
		y.append(2048)
		print(x, y)
		plt.scatter(x, y)
		plt.show()
		sys.exit(0)

	x, y = list(), list()
	for prob in np.linspace(0.95, 1, 6):
		if prob == 0.0 or prob == 1.0:
			continue
		model_pca = PCA(n_components=prob)
		model_pca.fit(np_features)
		# print(model_pca.explained_variance_ratio_) # 投影后不同特征维度的方差比例
		# print(model_pca.explained_variance_)
		num_components = len(model_pca.explained_variance_)
		x.append(prob)
		y.append(num_components)
		print(num_components)
		# new_np_feature = model_pca.transform(np_features)
		# print(new_np_feature)
		# fig = plt.figure()
		# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
		# plt.scatter(new_np_feature[:, 0], new_np_feature[:, 1], new_np_feature[:, 2], marker='o')
		# plt.savefig("pca_vis.png")
		# plt.show()

def analysis_LDA():
	x, y = list(), list()
	for prob in np.linspace(0, 1, 21):
		if prob == 0.0 or prob == 1.0:
			continue
		model_lda = LDA(n_components=prob)
		model_lda.fit(np_features, labels)
		num_components = len(model_lda.explained_variance_ratio_)
		x.append(prob)
		y.append(num_components)
	print(x)
	print(y)


def analysis_Pearsonr():
	x = list()
	y_score, y_p_value = list(), list()
	for idx in range(len(np_features.T)):
		score, p_value = pearsonr(np_features.T[idx,:], np_feature_labels)
		# print(score, p_value)
		x.append(idx)
		y_score.append(score)
		y_p_value.append(p_value)
	plt.scatter(x, y_score)
	# plt.scatter(x, y_p_value)
	plt.show()


def analysis_Cluster_Results():

	pass

analysis_Pearsonr()
# analysis_KMeans()
# with open("./kmeans_cluster.csv", "r") as h:
# 	x_list, y_list = list(), list()
# 	for line in h:
# 		x, y = line.strip().split(",")
# 		x_list.append(float(x))
# 		y_list.append(float(y))
# 	plt.scatter(x_list, y_list)
# 	plt.show()


