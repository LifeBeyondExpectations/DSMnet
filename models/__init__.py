#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from dispnet import dispnet
from dispnetcorr import dispnetcorr
from iresnet import iresnet
from gcnet import gcnet
from psmnet.stackhourglass import PSMNet as psmnet
#from gcnetm import gcnetm
#from gcnetm1 import gcnetm1
#from msfnet import msfnet

def model_create_by_name(name_model, maxdisparity=192):
    model = None
    if name_model == 'dispnet':
        model = dispnet(maxdisparity).cuda()
    elif name_model == 'dispnetcorr':
        model = dispnetcorr(maxdisparity).cuda()
    elif name_model == 'iresnet':
        model = iresnet(maxdisparity).cuda()
    elif name_model == 'gcnet':
        model = gcnet(maxdisparity).cuda()
    elif name_model == 'psmnet':
        model = psmnet(maxdisparity).cuda()
#    elif name_model == 'gcnetm':
#        model = gcnetm(maxdisparity).cuda()
#    elif name_model == 'gcnetm1':
#        model = gcnetm1(maxdisparity).cuda()
#    elif name_model == 'msfnet':
#        model = msfnet(maxdisparity).cuda()
    
    assert model is not None
    return model


