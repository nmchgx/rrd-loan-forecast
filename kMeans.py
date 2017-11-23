#! python2
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('apply_info_samp.csv')

estimator = KMeans(n_clusters=3)
estimator.fit(df.ix[:,5:])
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和