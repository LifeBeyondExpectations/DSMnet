#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from dataset_check import dataset_check

#
rootpath = '/media/qjc/D/data/sceneflow'
checkpath = dataset_check("monkaa", rootpath)
checkpath.checkandsavepath()
checkpath = dataset_check("driving", rootpath)
checkpath.checkandsavepath()
#
rootpath = '/media/qjc/D/data/sceneflow'
checkpath = dataset_check("flyingthings3d-tr", rootpath)
checkpath.checkandsavepath()
checkpath = dataset_check("flyingthings3d-te", rootpath)
checkpath.checkandsavepath()
#
rootpath = '/media/qjc/D/data/kitti'
checkpath = dataset_check("kitti2015-tr", rootpath)
checkpath.checkandsavepath()
checkpath = dataset_check("kitti2015-te", rootpath)
checkpath.checkandsavepath()
#
rootpath = '/media/qjc/D/data/kitti'
checkpath = dataset_check("kitti2012-tr", rootpath)
checkpath.checkandsavepath()
checkpath = dataset_check("kitti2012-te", rootpath)
checkpath.checkandsavepath()
#
rootpath = '/media/qjc/D/data/kitti'
checkpath = dataset_check("kitti-raw", rootpath)
checkpath.checkandsavepath()


