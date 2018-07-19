#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
from dataloader import myDataLoader

def stereoplot_batch(imL_batch, imR_batch):
    print imL_batch.shape
    n = min(4, imL_batch.shape[0])
    for i in range(n):
        ax = plt.subplot(n, 2, 2*i + 1)
        plt.sca(ax); 
        image = imL_batch[i].transpose(1, 2, 0)
        plt.imshow(image)
        ax = plt.subplot(n, 2, 2*i + 2)
        plt.sca(ax); 
        image = imR_batch[i].transpose(1, 2, 0)
        plt.imshow(image)
    plt.show()

def stereoplot_batch0(batch):
    print [tmp.shape for tmp in batch]
    ncol = len(batch)
    assert ncol >= 2
    nrow = min(4, batch[0].shape[0])
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(nrow, ncol, ncol*i + j + 1)
            image = batch[j][i].numpy().transpose(1, 2, 0).squeeze()
            #print image.shape
            plt.sca(ax); plt.imshow(image)
#    plt.pause(1)
    plt.show()


root = "/media/qjc/E/data/sceneflow"
dataset = "flyingthings3d-tr"
dataset = "flyingthings3d-te"

root = "/media/qjc/D/data/kitti"
dataset = "kitti2015-tr"
dataset = "kitti2015-te"

root = "/media/qjc/D/data/kitti"
dataset = "kitti2012-tr"
dataset = "kitti2012-te"

root = "/media/qjc/D/data/kitti"
dataset = "kitti-raw"

root = "/media/qjc/D/data/kitti"
dataset = "kitti2012-tr_kitti2015-tr"
dataloader = myDataLoader(training=True, dataset=dataset, root=root, batch_size=4, num_workers=4, flag_disp=True)

for batch_idx, batch in enumerate(dataloader):
    print(batch_idx)
    stereoplot_batch0(batch)
    if(batch_idx>5): break

