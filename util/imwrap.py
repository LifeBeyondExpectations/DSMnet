#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def imswrap(im_src, disps, scale_disps, fliplr=False, LeftTop=[0, 0]):
    assert type(disps) is list or type(scale_disps) is list
    assert len(disps) == len(scale_disps)
    count = len(scale_disps)
    ims_wrap =[]
    for i in range(count):
        scale_factor = 2**scale_disps[i]
        im = imwrap_BCHW(im_src, -disps[i], fliplr, LeftTop, scale_factor)
        ims_wrap.append(im)
    return ims_wrap

def imwrap_pyramid(im_src, disps_pyramid, fliplr=False, LeftTop=[0, 0]):
    assert type(disps_pyramid) is list
    levels = len(disps_pyramid)
    ims_wrap =[]
    scale_factor = 1
    for i in range(levels):
        im = imwrap_BCHW(im_src, -disps_pyramid[i], fliplr, LeftTop, scale_factor)
        ims_wrap.append(im)
        scale_factor = scale_factor*2
    return ims_wrap

def imwrap_BCHW(im_src, disp, fliplr=False, LeftTop=[0, 0], scale_factor=1):
    '''
    the shape of im_src should be (bn, c , h0, w0)
    the shape of disp should be (bn, 1 , h, w)
    fliplr is the flag of flip im_src horizontally
    LeftTop is the imwrap's left top position in im_src_fliplr
    scale_factor is rate of scale between imwrap and im_src
    '''
    # imwrap
    bn, _, h0, w0 = im_src.shape
    bn, c, h, w = disp.shape
    assert c == 1 and min(h, w, h0, w0)>1
    # ------------------------compute area(x, x1, y, y1)------------------------------
    x, y = LeftTop
    x = x*2.0/(w0 - 1) - 1 # use (w0-1) because the boundary is the center of pixel
    y = y*2.0/(h0 - 1) - 1
    x1 = x + (w - 1)*scale_factor*2.0/(w0 - 1)
    y1 = y + (h - 1)*scale_factor*2.0/(h0 - 1)
    #print x, x1, y, y1
    # ---------------------------create sample grid-------------------------------------
    row = torch.linspace(x, x1, w)
    col = torch.linspace(y, y1, h)
    grid = torch.zeros(bn, h, w, 2)
    for n in range(bn):
        for i in range(h):
            grid[n, i, :, 0] = row
        for i in range(w):
            grid[n, :, i, 1] = col
    grid = Variable(grid, requires_grad=False).type_as(im_src)
    k = -1.0 if fliplr else 1
    grid[:, :, :, 0] = k*(grid[:, :, :, 0] + disp.squeeze(1)*2.0/(w0 - 1))
    #print grid.shape, type(grid), type(im_src)
    # ---------------------------sample image by grid-----------------------------------
    delt = 1e-4*(torch.rand(1)[0] + 0.1)
    im_wrap = F.grid_sample(im_src + delt, grid)
    return im_wrap

def imwrap_BCHW0(im_src, disp):
    # imwrap
    bn, c, h, w = im_src.shape
    row = torch.linspace(-1, 1, w)
    col = torch.linspace(-1, 1, h)
    grid = torch.zeros(bn, h, w, 2)
    for n in range(bn):
        for i in range(h):
            grid[n, i, :, 0] = row
        for i in range(w):
            grid[n, :, i, 1] = col
    grid = Variable(grid, requires_grad=True).type_as(im_src)
    grid[:, :, :, 0] = grid[:, :, :, 0] + disp.squeeze(1)*2/w
    #print disp[-1, -1, -1], grid[-1, -1, -1, 0]
    im_src.clamp(min=1e-6)
    im_wrap = F.grid_sample(im_src, grid)
    return im_wrap

