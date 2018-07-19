#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from util_conv import net_init, conv2d_bn, deconv4x4_bn, Corr1d, mycat

flag_bn = False
activefun_default = nn.ReLU(inplace=True)

class dispnetcorr(nn.Module):
    def __init__(self, maxdisparity=192):
        super(dispnetcorr, self).__init__()
        self.name = "dispnetcorr"
        #self.D = maxdisparity
        self.delt = 1e-6

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # 卷积层
        self.conv1 = conv2d_bn(3, 64, kernel_size=7, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, bn=flag_bn, activefun=activefun_default)
        self.corr = Corr1d(kernel_size=1, stride=1, D=41, simfun=None)
        self.redir = conv2d_bn(128, 64, kernel_size=1, stride=1, bn=flag_bn, activefun=activefun_default)        
        self.conv3a = conv2d_bn(64 + 41, 256, kernel_size=5, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv3b = conv2d_bn(256, 256, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.conv4a = conv2d_bn(256, 512, kernel_size=3, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv4b = conv2d_bn(512, 512, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.conv5a = conv2d_bn(512, 512, kernel_size=3, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv5b = conv2d_bn(512, 512, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.conv6a = conv2d_bn(512, 1024, kernel_size=3, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv6b = conv2d_bn(1024, 1024, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        
        # 解卷积层和视差预测层
        self.pr6 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv5 = deconv4x4_bn(1024, 512, bn=flag_bn, activefun=activefun_default)
        self.iconv5 = conv2d_bn(1025, 512, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv4 = deconv4x4_bn(512, 256, bn=flag_bn, activefun=activefun_default)
        self.iconv4 = conv2d_bn(769, 256, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv3 = deconv4x4_bn(256, 128,  bn=flag_bn, activefun=activefun_default)
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv2 = deconv4x4_bn(128, 64, bn=flag_bn, activefun=activefun_default)
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv1 = deconv4x4_bn(64, 32, bn=flag_bn, activefun=activefun_default)
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        
        # 权重初始化
        net_init(self)
#        for m in [self.pr6, self.pr5, self.pr4, self.pr3, self.pr2, self.pr1]:
#            m.weight.data = m.weight.data*1e-1

    def forward(self, imL, imR, mode="train"):
        assert imL.shape == imR.shape
        #　设置最大视差
        self.D = imL.shape[-1]

        #　编码阶段
        conv1L = self.conv1(imL)
        conv1R = self.conv1(imR)
        conv2L = self.conv2(conv1L)
        conv2R = self.conv2(conv1R)
        corr = self.corr(conv2L, conv2R)
        redir = self.redir(conv2L)
        conv3a = self.conv3a(torch.cat([corr, redir], dim=1))
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)
        conv5a = self.conv5a(conv4b)
        conv5b = self.conv5b(conv5a)
        conv6a = self.conv6a(conv5b)
        conv6b = self.conv6b(conv6a)
        
        #　解码和预测阶段
        out = []
        out_scale = []
        pr6 = self.pr6(conv6b)
        if(mode == "test"): pr6 = pr6.clamp(self.delt, self.D)
        out.insert(0, pr6)
        out_scale.insert(0, 6)
        
        deconv5 = self.deconv5(conv6b)
        iconv5 = self.iconv5(mycat([deconv5, self.upsample(pr6), conv5b, ], dim=1))
        pr5 = self.pr5(iconv5)
        if(mode == "test"): pr5 = pr5.clamp(self.delt, self.D)
        out.insert(0, pr5)
        out_scale.insert(0, 5)

        deconv4 = self.deconv4(iconv5)
        iconv4 = self.iconv4(mycat([deconv4, self.upsample(pr5), conv4b, ], dim=1))
        pr4 = self.pr4(iconv4)
        if(mode == "test"): pr4= pr4.clamp(self.delt, self.D)
        out.insert(0, pr4)
        out_scale.insert(0, 4)

        deconv3 = self.deconv3(iconv4)
        iconv3 = self.iconv3(mycat([deconv3, self.upsample(pr4), conv3b, ], dim=1))
        pr3 = self.pr3(iconv3)
        if(mode == "test"): pr3 = pr3.clamp(self.delt, self.D)
        out.insert(0, pr3)
        out_scale.insert(0, 3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(mycat([deconv2, self.upsample(pr3), conv2L, ], dim=1))
        pr2 = self.pr2(iconv2)
        if(mode == "test"): pr2 = pr2.clamp(self.delt, self.D)
        out.insert(0, pr2)
        out_scale.insert(0, 2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(mycat([deconv1, self.upsample(pr2), conv1L, ], dim=1))
        pr1 = self.pr1(iconv1)
        if(mode == "test"): pr1 = pr1.clamp(self.delt, self.D)
        out.insert(0, pr1)
        out_scale.insert(0, 1)

        pr0 = self.upsample(pr1)
        pr0 = pr0[:, :, :imL.shape[-2], :imL.shape[-1]]
        if(mode == "test"): pr0 = pr0.clamp(self.delt, self.D)
        out.insert(0, pr0)
        out_scale.insert(0, 0)

        return out_scale, out

