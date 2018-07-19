#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
stereo dataloader
"""

import numpy as np
from dataset_check import dataset_check
from imagerw import load_image, load_disp
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

def myDataLoader(training, dataset="kitti2015-tr", root='./kitti', batch_size=1, num_workers=1, flag_disp=True):
    paths, size = dataset_merge(dataset, root)
    if(batch_size == 1): size = None
    assert len(paths) >= 2
    if(len(paths) > 2) and flag_disp:
        dataset = myDataset(paths[0], paths[1], paths[2], None, training=training, size=size)
    else:
        dataset = myDataset(paths[0], paths[1], None, None, training=training, size=size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=num_workers, drop_last=False)

def dataset_merge(datasets="kitti2015-tr_kitti2012-tr", root='./kitti'):
    datasets = datasets.split("_")
    paths_all, size_all = dataset_check(datasets[0], root).getpaths()
    for i in range(1, len(datasets)):
        dataset = datasets[i]
        paths, size = dataset_check(dataset, root).getpaths()
        logging.debug("len(paths): %d , len(paths_all): %d " % (len(paths), len(paths_all)))
        assert len(paths) == len(paths_all)
        for i in range(len(paths)):
            paths_all[i].extend(paths[i])
        size_all[0] = min(size_all[0], size[0])
        size_all[1] = min(size_all[1], size[1])
    return paths_all, size_all

class myDataset(Dataset):
    def __init__(self, left, right, disp_left, disp_right, training, size=None, loader_img=load_image, loader_disp= load_disp):
 
        self.left = left
        self.right = right
        self.disp_left = disp_left
        self.disp_right = disp_right
        self.loader_img = loader_img
        self.loader_disp = loader_disp
        self.preprocess = transforms.Compose([transforms.ToTensor(),])
        self.training = training
        if(size is not None):
            self.h = size[0]
            self.w = size[1]
        else:
            self.h = 100000
            self.w = 100000

    def __getitem__(self, index):
        
        # load image and disp
        img_left = None
        img_right = None
        disp_left = None
        disp_right = None
        while(True):
            try:
                # load image
                img_left = self.loader_img(self.left[index])[-self.h:, :self.w, :]
                img_right = self.loader_img(self.right[index])[-self.h:, :self.w, :]
                disp_left = None
                disp_right = None
                
                # load disp
                if(self.disp_left is not None):
                    disp_left= self.loader_disp(self.disp_left[index])[None, -self.h:, :self.w].copy()
                    if(self.disp_right is not None):
                        disp_right = self.loader_disp(self.disp_right[index])[None, -self.h:, :self.w].copy()
            except Exception as err:
                msg = "a error occurred when loadering img(%s) \nerror info: %s " % (self.left[index], str(err))
                logging.error(msg)
                if(index > 10): index -= np.random.randint(index//2, index)
                else: index += np.random.randint(10, 20)
                index = min(index, len(self.left))
            else:
                break

        # preprocess
        processed = self.preprocess
        img_left   = processed(img_left)
        img_right  = processed(img_right)

        if(disp_right is not None):
            return img_left, img_right, disp_left, disp_right
        elif(disp_left is not None):
            return img_left, img_right, disp_left
        else:
            return img_left, img_right

    def __len__(self):
        return len(self.left)
