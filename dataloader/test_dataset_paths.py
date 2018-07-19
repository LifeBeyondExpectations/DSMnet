#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from dataset_paths import dataset_paths

# sceneflow
rootpath = '/media/qjc/D/data/sceneflow'
dataset = dataset_paths(flag_dataset="monkaa", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path) 

dataset = dataset_paths(flag_dataset="driving", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path) 

dataset = dataset_paths(flag_dataset="FlyingThings3D-tr", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path) 

dataset = dataset_paths(flag_dataset="FlyingThings3D-te", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path) 

# kitti
rootpath = '/media/qjc/D/data/kitti'
dataset = dataset_paths(flag_dataset="kitti2015-tr", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path)

dataset = dataset_paths(flag_dataset="kitti2015-te", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path) 

dataset = dataset_paths(flag_dataset="kitti2012-tr", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path) 

dataset = dataset_paths(flag_dataset="kitti2012-te", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path) 

dataset = dataset_paths(flag_dataset="kitti-raw", rootpath=rootpath)
print("%s: %d" % (dataset.flag_dataset, dataset.count))
for path in dataset.get_paths_idx(dataset.count//2): print(path) 

