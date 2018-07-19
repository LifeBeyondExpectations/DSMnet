#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
from imwrap import imwrap_BCHW
from util import to_tensor

def evaluate(imL, imR, dispL, dispL_gt=None):
    # epe, d1
    if(dispL_gt is None):
        epe, d1 = -1, -1
    else:
        mask_disp = dispL_gt > 0
        disp_diff = np.abs(dispL_gt - dispL)
        # epe
        epe = disp_diff[mask_disp].mean()
        # d1
        pixels_good = np.logical_or(disp_diff[mask_disp] <= 3, (disp_diff[mask_disp] / dispL_gt[mask_disp]) <= 0.05)
        d1 = 100 - 100.0 * pixels_good.sum() / mask_disp.sum()
    
    # pixelerror of imwrap 
    imL = to_tensor(imL)
    imR = to_tensor(imR)
    dispL = to_tensor(dispL)
    imL_w = imwrap_BCHW(imR, -dispL)
    im_diff = torch.abs(imL - imL_w)
    mask_im = imL_w.sum(dim=1, keepdim=True)>0
    tmp = im_diff[mask_im] 
    tmp = tmp if(len(tmp) > 0) else im_diff
    #print len(tmp)
    pixelerror = tmp.mean().data[0]*255
    
    return d1, epe, pixelerror

def imwrap_pixel_errors(imL, imR, dispL):
    imL = to_tensor(imL)
    imR = to_tensor(imR)
    dispL = to_tensor(dispL)
    imL_w = imwrap_BCHW(imR, -dispL)
    #print imR.shape, dispL.shape, imL_w.shape
    mask = imL_w>0
    im_diff = torch.abs(imL - imL_w)[mask]
    return im_diff.mean().data[0]*255

def compute_errors(gt, pred):
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]
    eps = 1e-6
    disp_diff = np.abs(gt - pred)
    
    # a1, a2, a3
    thresh = np.maximum((gt / (pred + eps)), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    # d1
    bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt) >= 0.05)
    d1 = 100.0 * bad_pixels.sum() / mask.sum()

    # rmse, rmse_log
    rmse = disp_diff ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred + eps)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # abs_rel, sq_rel
    abs_rel = np.mean(disp_diff / gt)
    sq_rel = np.mean((disp_diff**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, d1, a1, a2, a3


