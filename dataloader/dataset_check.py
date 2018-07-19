#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
check and save paths of stereo dataset
"""

import os, sys
import pickle
from imagerw import load_image, load_disp
from dataset_paths import dataset_paths
import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

dirpath_file = sys.path[0]

class dataset_check():
    """dataset_check"""
    def __init__(self, name="kitti2015-tr", root='./kitti'):
        self.name = name
        self.root = root
        self.paths_good = None
        self.h = 100000
        self.w = 100000
        if(not os.path.exists(self.root)):
            return
        self.dataset = dataset_paths(self.name, self.root)
        self.idxs = range(self.dataset.count)
        logging.debug("dataset count: %d" % len(self.idxs))

    def checkdisp(self, disp):
        assert len(disp.shape) == 2
        # remove extreme case
        th = disp.shape[1]/3
        mask = (disp>th)
        if(mask.mean() > 0.2):
            return False
        return True
        
    def checkgroup(self, paths_group):
        assert type(paths_group) == list
        n = len(paths_group)
        try:
            for j in range(n):
                path = paths_group[j]
                assert os.path.exists(path)
                tmp = None
                if(j < 2):
                    tmp = load_image(path)
                else:
                    tmp = load_disp(path)
                    if(not self.checkdisp(tmp)):
                        return False
                assert tmp is not None
                assert len(tmp.shape) >= 2
                if(j == 1):
                    h, w, c = tmp.shape
                    if(self.h > h): self.h = h
                    if(self.w > w): self.w = w
            return True
        except Exception as err:
            msg = "a error occurred when checking img(%s) \nerror info: %s " % (paths_group[0], str(err))
            logging.error(msg)
            return False
    
    def checkpaths_file0(self, savepath, savepath_bad):
        if(os.path.exists(savepath) and os.path.exists(savepath_bad) ):
            f1=file(savepath,'rb')
            paths_good= pickle.load(f1)
            f1.close()
            f1=file(savepath_bad,'rb')
            paths_bad= pickle.load(f1)
            f1.close()
            print("{} has been existed! ".format(savepath))
            print('Basic Info | dataset Name: {}, Count_good: {}, Count_bad: {}'.format(self.name, len(paths_good[0]), len(paths_bad[0])))
            print("Check regain(y/n)?")
            res = raw_input()
            if(res.lower() != u"y"):
                print("passed")
                return [paths_good, paths_bad]
        return None
        
    def checkpaths_file(self, savepath, savepath_bad):
        if(os.path.exists(savepath) and os.path.exists(savepath_bad) ):
            f1=file(savepath,'rb')
            paths_good, self.h, self.w = pickle.load(f1)
            f1.close()
            f1=file(savepath_bad,'rb')
            paths_bad= pickle.load(f1)
            f1.close()
            return [paths_good, paths_bad]
        return None
        
    def checkpaths(self):
        # checking
        logging.info("checking stereo dataset: " + self.name)
        if(len(self.idxs) < 1):
            return None
        paths_good=[]
        paths_bad=[]
        n = len(self.dataset.get_paths_idx(0))
        for i in range(n):
            paths_good.append([])
            paths_bad.append([])
        for idx in self.idxs:
            paths_group = self.dataset.get_paths_idx(idx)
            if(self.checkgroup(paths_group)):
                for j in range(n):
                    paths_good[j].append(paths_group[j])
            else:
                for j in range(n):
                    paths_bad[j].append(paths_group[j])
        return [paths_good, paths_bad]
        
    def checkandsavepath(self):
        if(not os.path.exists(self.root)):
            return
        # create dirpath
        root = self.root
        dirpath = os.path.join(root, "paths")
        if(not os.path.exists(dirpath)):
            os.mkdir(dirpath)
        savepath = os.path.join(dirpath, "{}.pkl".format(self.name))
        savepath_bad = savepath.replace(".pkl", "_bad.pkl")

        # check paths from file
        paths = self.checkpaths_file(savepath, savepath_bad)
        if(paths is not None and len(paths)==2 ):
            paths_good, paths_bad = paths
        else:
            # checking
            paths_good, paths_bad = self.checkpaths()
            # save result
            with open(savepath, 'wb') as f:
                pickle.dump([paths_good, self.h, self.w], f, True)
            with open(savepath_bad, 'wb') as f:
                pickle.dump(paths_bad, f, True)
            with open(savepath_bad.replace(".pkl", ".txt"), 'wb') as f:
                f.write("Count_bad: {} \n".format(len(paths_bad[0])))
                for path in paths_bad[0]:
                    f.write(path + "\n")
        msg = "dataset Name: {}, Count_good: {}, Count_bad: {}".format(self.name, len(paths_good[0]), len(paths_bad[0]))
        logging.info(msg)
        self.paths_good = paths_good

    def getpaths(self):
        # check exist file
        if(self.paths_good is None):
            self.checkandsavepath()
        return self.paths_good, [self.h, self.w]

