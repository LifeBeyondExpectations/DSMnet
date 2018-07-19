#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch.autograd import Variable
from dispnet import dispnet
from dispnetcorr import dispnetcorr
from iresnet import iresnet
from gcnet import gcnet
from psmnet.stackhourglass import PSMNet as psmnet

if __name__ == "__main__":
    models = [dispnet(), dispnetcorr(), iresnet() , gcnet(), psmnet()]
    models = [model.eval().cuda() for model in models]
    img_shape = [1, 3, 257, 513]
    imL = Variable(torch.randn(img_shape), volatile=True).cuda()
    imR = Variable(torch.randn(img_shape), volatile=True).cuda()
    
    for model in models:
        print("model: %s" % model.name)
        _, disps = model(imL, imR, mode="test")
        print("input: %s \noutput: %s \nPassed! \n" % (str(imL.shape), str(disps[0].shape)))

