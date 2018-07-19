#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append("../")
import torch
import torch.nn as nn
from util.imwrap import imwrap_BCHW
from util_conv import net_init, conv2d_bn, deconv4x4_bn, deconv8x8_bn, Corr1d, mycat

flag_bn = False
activefun_default = nn.ReLU(inplace=True)

class iresnet(nn.Module):
    def __init__(self, maxdisparity=192):
        super(iresnet, self).__init__()
        self.name = "iresnet"
        #self.D = maxdisparity
        self.delt = 1e-6

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # Stem Block for Multi-scale Shared Features Extraction
        self.conv1 = conv2d_bn(3, 64, kernel_size=7, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, bn=flag_bn, activefun=activefun_default)
        self.deconv1_s = deconv4x4_bn(64, 32, bn=flag_bn, activefun=activefun_default) # stride=2
        self.deconv2_s = deconv8x8_bn(128, 32, bn=flag_bn, activefun=activefun_default) # stride=4
        self.conv_de1_de2 = conv2d_bn(64, 32, kernel_size=1, stride=1, bn=flag_bn, activefun=activefun_default)
        
        # Initial Disparity Estimation Sub-network
        self.corr = Corr1d(kernel_size=1, stride=1, D=81, simfun=None)
        self.redir = conv2d_bn(128, 64, kernel_size=1, stride=1, bn=flag_bn, activefun=activefun_default)
        self.conv3 = conv2d_bn(81 + 64, 256, kernel_size=3, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv3_1 = conv2d_bn(256, 256, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.conv4 = conv2d_bn(256, 512, kernel_size=3, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv4_1 = conv2d_bn(512, 512, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.conv5 = conv2d_bn(512, 512, kernel_size=3, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv5_1 = conv2d_bn(512, 512, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.conv6 = conv2d_bn(512, 1024, kernel_size=3, stride=2, bn=flag_bn, activefun=activefun_default)
        self.conv6_1 = conv2d_bn(1024, 1024, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr6 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        self.deconv5 = deconv4x4_bn(1024, 512, bn=flag_bn, activefun=activefun_default) # stride=2
        self.iconv5 = conv2d_bn(1025, 512, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.deconv4 = deconv4x4_bn(512, 256, bn=flag_bn, activefun=activefun_default) # stride=2
        self.iconv4 = conv2d_bn(769, 256, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.deconv3 = deconv4x4_bn(256, 128,  bn=flag_bn, activefun=activefun_default) # stride=2
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.deconv2 = deconv4x4_bn(128, 64, bn=flag_bn, activefun=activefun_default) # stride=2
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.deconv1 = deconv4x4_bn(64, 32, bn=flag_bn, activefun=activefun_default) # stride=2
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.deconv0 = deconv4x4_bn(32, 32, bn=flag_bn, activefun=activefun_default) # stride=2
        self.iconv0 = conv2d_bn(65, 32, kernel_size=3, stride=1, bn=flag_bn, activefun=activefun_default)
        self.pr0 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # Disparity Refinement Sub-network
        #imwrap up_conv1b2b
        self.r_conv0 = conv2d_bn(65, 32, kernel_size=3, stride=1, bn=flag_bn)
        self.r_conv1 = conv2d_bn(32, 64, kernel_size=3, stride=2, bn=flag_bn)
        self.c_conv1 = conv2d_bn(64, 64, kernel_size=3, stride=1, bn=flag_bn)
        self.r_corr = Corr1d(kernel_size=3, stride=2, D=41, simfun=None)
        self.r_conv1_1 = conv2d_bn(105, 64, kernel_size=3, stride=1, bn=flag_bn)
        self.r_conv2 = conv2d_bn(64, 128, kernel_size=3, stride=2, bn=flag_bn)
        self.r_conv2_1 = conv2d_bn(128, 128, kernel_size=3, stride=1, bn=flag_bn)
        self.r_res2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.r_deconv1 = deconv4x4_bn(128, 64, bn=flag_bn) # stride=2
        self.r_iconv1 = conv2d_bn(129, 64, kernel_size=3, stride=1, bn=flag_bn)
        self.r_res1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.r_deconv0 = deconv4x4_bn(64, 32, bn=flag_bn) # stride=2
        self.r_iconv0 = conv2d_bn(65, 32, kernel_size=3, stride=1, bn=flag_bn)
        self.r_res0 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # init weight
        net_init(self)
        for m in [self.pr6, self.pr5, self.pr4, self.pr3, self.pr2, self.pr1, self.r_res2, self.r_res1, self.r_res0]:
            m.weight.data = m.weight.data*1e-2
        
    def forward(self, imL, imR, mode="train"):
        #
        self.D = imL.shape[-1]
        if(mode == "train"):
            self.train()
        else:
            self.eval()
        #
        assert imL.shape == imR.shape
        out = []
        out_scale =[]
        
        # Multi-scale Shared Features Extraction
        conv1L = self.conv1(imL)
        conv1R = self.conv1(imR)
        conv2L = self.conv2(conv1L)
        conv2R = self.conv2(conv1R)
        deconv1L = self.deconv1_s(conv1L)
        deconv1R = self.deconv1_s(conv1R)
        deconv1L = deconv1L[:, :, :imL.shape[-2], :imL.shape[-1]]
        deconv1R = deconv1R[:, :, :imL.shape[-2], :imL.shape[-1]]
        deconv2L = self.deconv2_s(conv2L)
        deconv2R = self.deconv2_s(conv2R)
        deconv1L2L = self.conv_de1_de2(mycat([deconv1L, deconv2L, ], dim=1))
        deconv1R2R = self.conv_de1_de2(mycat([deconv1R, deconv2R, ], dim=1))
        
        # Initial Disparity
        corr = self.corr(conv2L, conv2R)
        redir = self.redir(conv2L)
        conv3 = self.conv3(torch.cat([corr, redir], dim=1))
        conv3_1 = self.conv3_1(conv3)
        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)
        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)
        conv6 = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(conv6)
        
        pr6 = self.pr6(conv6_1)
        if(mode == "test"): pr6 = pr6.clamp(self.delt, self.D)
        out.insert(0, pr6)
        out_scale.insert(0, 6)
        
        deconv5 = self.deconv5(conv6_1)
        iconv5 = self.iconv5(mycat([deconv5, self.upsample(pr6), conv5_1, ], dim=1))
        pr5 = self.pr5(iconv5)
        if(mode == "test"): pr5 = pr5.clamp(self.delt, self.D)
        out.insert(0, pr5)
        out_scale.insert(0, 5)

        deconv4 = self.deconv4(iconv5)
        iconv4 = self.iconv4(mycat([deconv4, self.upsample(pr5), conv4_1, ], dim=1))
        pr4 = self.pr4(iconv4)
        if(mode == "test"): pr4 = pr4.clamp(self.delt, self.D)
        out.insert(0, pr4)
        out_scale.insert(0, 4)

        deconv3 = self.deconv3(iconv4)
        iconv3 = self.iconv3(mycat([deconv3, self.upsample(pr4), conv3_1, ], dim=1))
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

        deconv0 = self.deconv0(iconv1)
        iconv0 = self.iconv0(mycat([deconv0, self.upsample(pr1), deconv1L2L, ], dim=1))
        pr0 = self.pr0(iconv0)
        if(mode == "test"): pr0 = pr0.clamp(self.delt, self.D)
        out.insert(0, pr0)
        out_scale.insert(0, 0)
        
        # Disparity Refinement 
        w_deconv1L2L = imwrap_BCHW(deconv1R2R, -pr0)
        reconerror = torch.abs(deconv1L2L - w_deconv1L2L)
        r_conv0 = self.r_conv0(mycat([reconerror, pr0, deconv1L2L, ], dim=1))
        r_conv1 = self.r_conv1(r_conv0)
        c_conv1L = self.c_conv1(conv1L)
        c_conv1R = self.c_conv1(conv1R)
        r_corr = self.r_corr(c_conv1L, c_conv1R)
        r_conv1_1 = self.r_conv1_1(mycat([r_conv1, r_corr, ], dim=1))
        r_conv2 = self.r_conv2(r_conv1_1)
        r_conv2_1 = self.r_conv2_1(r_conv2)
        
        r_res2 = self.r_res2(r_conv2_1)
        r_pr2 = pr2 + r_res2
        if(mode == "test"): r_pr2 = r_pr2.clamp(self.delt, self.D)
        out.insert(0, r_pr2)
        out_scale.insert(0, 2)
        
        r_deconv1 = self.r_deconv1(r_conv2_1)
        r_iconv1 = self.r_iconv1(mycat([r_deconv1, self.upsample(r_res2), r_conv1_1, ], dim=1))
        r_res1 = self.r_res1(r_iconv1)
        r_pr1 = pr1 + r_res1
        if(mode == "test"): r_pr1 = r_pr1.clamp(self.delt, self.D)
        out.insert(0, r_pr1)
        out_scale.insert(0, 1)
        
        r_deconv0 = self.r_deconv0(r_iconv1)
        r_iconv0 = self.r_iconv0(mycat([r_deconv0, self.upsample(r_res1), r_conv0, ], dim=1))
        r_res0 = self.r_res0(r_iconv0)
        r_pr0 = pr0 + r_res0
        if(mode == "test"): r_pr0 = r_pr0.clamp(self.delt, self.D)
        out.insert(0, r_pr0)
        out_scale.insert(0, 0)
        
        return out_scale, out

