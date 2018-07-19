#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import torch
from torch.autograd import Variable
from dispnet import dispnet
from dispnetcorr import dispnetcorr
from iresnet import iresnet
from gcnet import gcnet
from psmnet.stackhourglass import PSMNet as psmnet

def test(model_id, count, img_shape):
    assert len(img_shape) == 4
    if(model_id == 0):
        model = dispnet()
    elif(model_id == 1):
        model = dispnetcorr()
    elif(model_id == 2):
        model = iresnet()
    elif(model_id == 3):
        model = gcnet()
    elif(model_id == 4):
        model = psmnet()
    else:
        return
    model = model.eval().cuda()
    imL_rand = torch.rand(img_shape)
    imR_rand = torch.rand(img_shape)
    n = 2
    for i in range(count + n):
        if(i==n): time_start = time.time() # 排除初次测试启动时间较长的部分
        imL = imL_rand * (0.8 + 0.4*torch.rand(1))
        imR = imR_rand * (0.8 + 0.4*torch.rand(1))
        imL = Variable(imL, volatile=True).cuda()
        imR = Variable(imR, volatile=True).cuda()
        model(imL, imR, mode="test")
        del imL, imR
    time_all = time.time() - time_start
    msg_test = ("model: %12s , full time: %.3f(s), mean time: %.3f(s)" % 
                (model.name, time_all, time_all/count))
    print(msg_test)
    return model.name, time_all, time_all/count

if __name__ == "__main__":
    img_shape = [1, 3, 320, 1242] # [1, 3, 375, 1242] # [1, 3, 480, 960] #
    count = 100
    print("---Overview--- test count: %d --- image shape: %s ---" % (count, str(img_shape)))
    res = []
    for i in range(5):
        try:
            res.append(test(i, count, img_shape))
        except Exception as err:
            print(str(err))
    for tmp in res:
        print("%12s\t%.3f\t%.3f" %(tmp[0], tmp[1], tmp[2]))
