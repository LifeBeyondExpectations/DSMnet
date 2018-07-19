import sys
sys.path.append("../")
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from SSIM import SSIM
from util.imwrap import imswrap
from util.util import imsplot_tensor

import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

flag_test = False
flag_imshow = False

def create_impyramid(im, levels):
    impyramid = [im]
    # pyramid
    for i in range(1, levels):
        impyramid.append(impyramid[-1][:, :, ::2, ::2])
    return impyramid

class loss_stereo(torch.nn.Module):
    def __init__(self):
        super(loss_stereo, self).__init__()
        self.w_ap = 1.0
        self.w_ds = 0.001
        self.w_lr = 0.001
        self.w_m = 0.0001
        self.ssim = SSIM()
    
    def wfun(self, similarity):
        return max(0, similarity - 0.75)/2 + 0.001

    def diff1_dx(self, img):
        assert len(img.shape) == 4
        diff1 = img[:,:,:,1:] - img[:,:,:,:-1]
        return F.pad(diff1, [0,1,0,0])
        
    def diff1_dy(self, img):
        assert len(img.shape) == 4
        diff1 = img[:,:,1:] - img[:,:,:-1]
        return F.pad(diff1, [0,0,0,1])
        
    def diff2_dx(self, img):
        assert len(img.shape) == 4
        diff2 = img[:,:,:,2:] + img[:,:,:,:-2] - img[:,:,:,1:-1] - img[:,:,:,1:-1]
        return F.pad(diff2, [1,1,0,0])

    def diff2_dy(self, img):
        assert len(img.shape) == 4
        diff2 = img[:,:,2:] + img[:,:,:-2] - img[:,:,1:-1] - img[:,:,1:-1]
        return F.pad(diff2, [0,0,1,1])

    def diff_z_dx(self, disp):
        assert len(disp.shape) == 4
        diff_p = (disp[:,:,:,1:-1]/disp[:,:,:,2:]) + (disp[:,:,:,1:-1]/disp[:,:,:,:-2]) - 2
        return F.pad(diff_p, [1,1,0,0])

    def diff_z_dy(self, disp):
        assert len(disp.shape) == 4
        diff_p = (disp[:,:,1:-1]/disp[:,:,2:]) + (disp[:,:,1:-1]/disp[:,:,:-2]) - 2
        return F.pad(diff_p, [0,0,1,1])

    def C_imdiff1(self, img, img_wrap):
        L1_dx = torch.abs(self.diff1_dx(img) - self.diff1_dx(img_wrap))
        L1_dy = torch.abs(self.diff1_dy(img) - self.diff1_dy(img_wrap))
        return L1_dx + L1_dy
    
    def C_ds1(self, img, disp):
        disp_dx = torch.abs(self.diff1_dx(disp))
        disp_dy = torch.abs(self.diff1_dy(disp))

        image_dx = torch.abs(self.diff1_dx(img))
        image_dy = torch.abs(self.diff1_dy(img))
        weights_x = torch.exp(-torch.sum(image_dx, dim=1, keepdim=True))
        weights_y = torch.exp(-torch.sum(image_dy, dim=1, keepdim=True))
    
        #print weights_x.shape, disp_gradients_x.shape
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        return smoothness_x + smoothness_y
    
    def C_ds2(self, img, disp):
        disp_dx = torch.abs(self.diff2_dx(disp))
        disp_dy = torch.abs(self.diff2_dy(disp))

        image_dx = torch.abs(self.diff2_dx(img))
        image_dy = torch.abs(self.diff2_dy(img))
        weights_x = torch.exp(-torch.sum(image_dx, dim=1, keepdim=True))
        weights_y = torch.exp(-torch.sum(image_dy, dim=1, keepdim=True))
    
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        return smoothness_x + smoothness_y

    def C_ds3(self, img, disp):
        disp = torch.abs(disp) + 1
        disp_dx = torch.abs(self.diff_z_dx(disp)).clamp(0, 10)
        disp_dy = torch.abs(self.diff_z_dy(disp)).clamp(0, 10)

        image_dx = torch.abs(self.diff1_dx(img))
        image_dy = torch.abs(self.diff1_dy(img))
        mImage_dx = image_dx.mean(-1,True).mean(-2,True).mean(-3,True)
        mImage_dy = image_dy.mean(-1,True).mean(-2,True).mean(-3,True)
        weights_x = torch.exp(-torch.max(image_dx, dim=1, keepdim=True)[0]/(0.5*mImage_dx))
        weights_y = torch.exp(-torch.max(image_dy, dim=1, keepdim=True)[0]/(0.5*mImage_dy))
    
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        return smoothness_x + smoothness_y

class lossfun(loss_stereo):
    def __init__(self, loss_name):
        super(lossfun, self).__init__()
        self.loss_name = loss_name
    
    def loss_common(self, im, im_wrap, disp, disp_wrap, factor=1.0, weight_common=None):
        # ----------------set w_ds and w_lr---------------------
        img_ssim = self.ssim(im, im_wrap)
        mask_ap = (im_wrap[:, :1] != 0).detach()
        simlary = img_ssim[mask_ap].mean().data[0]
        w = self.wfun(simlary)
        self.w_ds = w
        self.w_lr = w
        
        # ----------------set C_ap and C_lr---------------------
        C_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*(torch.abs(im - im_wrap)) # + self.C_imdiff1(im, im_wrap))
        C_lr = torch.abs(disp - disp_wrap)

        # ---------------------set mask------------------------
        if(weight_common is not None):
            mask_im = ((disp_wrap==0) + mask_ap).detach() > 1
            mask_lr = (disp_wrap==0).detach()
            weight_im = weight_common.clone()
            weight_im[mask_im] = 1.0
            weight_lr = weight_common.clone()
            weight_lr[mask_lr] = 0
            C_ap = C_ap * weight_im
            C_lr = C_lr * weight_lr
            msg = "weight_im maxV: %f, minV: %f ;" % (weight_im.max().data[0], weight_im.min().data[0])
            msg += "weight_lr maxV: %f, minV: %f " % (weight_lr.max().data[0], weight_lr.min().data[0])
            logging.debug(msg)
            

        # ----------------------C_all----------------------------
        C_ap = C_ap.mean()
        C_ds = self.C_ds3(im, disp).mean()
        C_lr = C_lr.mean()
        C = C_ap*self.w_ap + C_ds*(self.w_ds/factor) + C_lr*self.w_lr
        
        # show in screen
        if(flag_test):
            print self.w_ap, self.w_ds, self.w_lr, simlary
            print C.data[0], C_ap.data[0]*self.w_ap, C_ds.data[0]*self.w_ds, C_lr.data[0]*self.w_lr
        return C

    def loss_depthmono(self, im, im_wrap, disp, disp_wrap, factor=1.0, weight_common=None):
        # ----------------set w_ds and w_lr---------------------
        img_ssim = self.ssim(im, im_wrap)
        mask_ap = (im_wrap[:, :1] != 0).detach()
        simlary = img_ssim[mask_ap].mean().data[0]
        w = self.wfun(simlary)
        self.w_ds = w
        self.w_lr = w
        
        # ----------------set C_ap and C_lr---------------------
        C_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*torch.abs(im - im_wrap)
        C_lr = torch.abs(disp - disp_wrap)

        # ---------------------set mask------------------------
        if(weight_common is not None):
            mask_im = ((disp_wrap==0) + mask_ap).detach() > 1
            mask_lr = (disp_wrap==0)
            weight_im = weight_common.clone()
            weight_im[mask_im] = 1.0
            weight_lr = weight_common.clone()
            weight_lr[mask_lr] = 0
            C_ap = C_ap * weight_im
            C_lr = C_lr * weight_lr
            msg = "weight_im maxV: %f, minV: %f ;" % (weight_im.max().data[0], weight_im.min().data[0])
            msg += "weight_lr maxV: %f, minV: %f " % (weight_lr.max().data[0], weight_lr.min().data[0])
            logging.debug(msg)
            

        # ----------------------C_all----------------------------
        C_ap = C_ap.mean()
        C_ds = self.C_ds1(im, disp).mean()
        C_lr = C_lr.mean()
        C = C_ap*self.w_ap + C_ds*(self.w_ds/factor) + C_lr*self.w_lr
        
        # show in screen
        if(flag_test):
            print self.w_ap, self.w_ds, self.w_lr, simlary
            print C.data[0], C_ap.data[0]*self.w_ap, C_ds.data[0]*self.w_ds, C_lr.data[0]*self.w_lr
        return C

    def loss_Cap_ds_lr(self, im, im_wrap, disp, disp_wrap, factor=1.0, weight_common=None):
        # ----------------set w_ds and w_lr---------------------
        img_ssim = self.ssim(im, im_wrap)
        mask_ap = (im_wrap[:, :1] != 0).detach()
        simlary = img_ssim[mask_ap].mean().data[0]
        w = self.wfun(simlary)
        self.w_ds = w
        self.w_lr = w
        
        # ----------------set C_ap and C_lr---------------------
        C_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*torch.abs(im - im_wrap)
        C_lr = torch.abs(disp - disp_wrap)

        # ---------------------set mask------------------------
        if(weight_common is not None):
            mask_im = ((disp_wrap==0) + mask_ap).detach() > 1
            mask_lr = (disp_wrap==0)
            weight_im = weight_common.clone()
            weight_im[mask_im] = 1.0
            weight_lr = weight_common.clone()
            weight_lr[mask_lr] = 0
            C_ap = C_ap * weight_im
            C_lr = C_lr * weight_lr
            msg = "weight_im maxV: %f, minV: %f ;" % (weight_im.max().data[0], weight_im.min().data[0])
            msg += "weight_lr maxV: %f, minV: %f " % (weight_lr.max().data[0], weight_lr.min().data[0])
            logging.debug(msg)

        # ----------------------C_ap----------------------------
        C_ap = C_ap.mean()
        C = C_ap * self.w_ap

        # ----------------------C_ds----------------------------
        if("ds" in self.loss_name):
            C_ds = self.C_ds1(im, disp)
            C += C_ds * (self.w_ds/factor)

        # ----------------------C_lr----------------------------
        if("lr" in self.loss_name):
            C_lr = C_lr.mean()
            C += C_lr * self.w_lr
        
        # show in screen
        if(flag_test):
            print self.w_ap, self.w_ds, self.w_lr, simlary
            print C.data[0], C_ap.data[0]*self.w_ap, C_ds.data[0]*self.w_ds, C_lr.data[0]*self.w_lr
        return C

    def loss_SsSMnet(self, im, im_wrap, im_wrap1, disp, factor=1.0, weight_common=None):
        # ----------------set w_ds and w_lr---------------------
        img_ssim = self.ssim(im, im_wrap)
        mask_ap = (im_wrap[:, :1] != 0).detach()
        simlary = img_ssim[mask_ap].mean().data[0]
        w = self.wfun(simlary)
        self.w_ds = w
        self.w_lr = w
        
        # ----------------set C_ap and C_lr---------------------
        C_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*(torch.abs(im - im_wrap) + self.C_imdiff1(im, im_wrap))
        C_lr = torch.abs(im - im_wrap1)

        # ---------------------set mask------------------------
        if(weight_common is not None):
            mask_im = ((im_wrap1[:, :1] == 0) + mask_ap).detach() > 1
            mask_lr = (im_wrap1[:, :1] == 0)
            weight_im = weight_common.clone()
            weight_im[mask_im] = 1.0
            weight_lr = weight_common.clone()
            weight_lr[mask_lr] = 0
            C_ap = C_ap * weight_im
            C_lr = C_lr * weight_lr
            msg = "weight_im maxV: %f, minV: %f ;" % (weight_im.max().data[0], weight_im.min().data[0])
            msg += "weight_lr maxV: %f, minV: %f " % (weight_lr.max().data[0], weight_lr.min().data[0])
            logging.debug(msg)
            

        # ----------------------C_all----------------------------
        C_ap = C_ap.mean()
        C_ds = self.C_ds2(im, disp).mean()
        C_lr = C_lr.mean()
        C_mdh = torch.abs(disp).mean()
        C = C_ap*self.w_ap + C_ds*(self.w_ds/factor) + C_lr*self.w_lr + C_mdh*self.w_m
        
        # show in screen
        if(flag_test):
            print self.w_ap, self.w_ds, self.w_lr, simlary
            print C.data[0], C_ap.data[0]*self.w_ap, C_ds.data[0]*self.w_ds, C_lr.data[0]*self.w_lr
        return C

    def loss_supervised(self, disp_gt, disp, flag_smooth=False, factor=1.0):
        mask = disp_gt>0
        C_disp = torch.abs(disp_gt - disp)[mask].mean()
        if(flag_smooth):
            disp_dx = self.diff1_dx(disp)
            disp_dy = self.diff1_dy(disp)
            disp_dxdy = (torch.abs(disp_dx) + torch.abs(disp_dy))/factor
            C_smooth = (disp_dxdy[mask] - 0.05).clamp(0, 1).mean()
            C_disp = C_disp + C_smooth
        
        return C_disp
    
    
class losses(lossfun):
    def __init__(self, loss_name="supervised"):
        
        # loss_name parse
        self.flag_mask = ("mask" in loss_name)
        loss_name = loss_name.split("-")[0].lower()
        self.loss_names = ["supervised", "depthmono", "SsSMnet".lower(), "Cap_ds_lr".lower(), "common"]
        assert loss_name in self.loss_names or "Cap".lower() in loss_name
        
        #  set lossfun and lossesfun
        super(losses, self).__init__(loss_name)
        self.lossfun = None
        self.lossesfun = None
        self.setlossfun(loss_name)
    
    def setlossfun(self, loss_name):
        if(self.loss_names[0] in loss_name):
            self.lossfun = self.loss_supervised
            self.lossesfun = self.losses_pyramid0
        elif(self.loss_names[1] in loss_name):
            self.lossfun = self.loss_depthmono
            self.lossesfun = self.losses_pyramid1
        elif(self.loss_names[2] in loss_name):
            self.lossfun = self.loss_SsSMnet
            self.lossesfun = self.losses_pyramid2
        elif("Cap".lower() in loss_name):
            self.lossfun = self.loss_Cap_ds_lr
            self.lossesfun = self.losses_pyramid1
        elif(self.loss_names[4] in loss_name):
            self.lossfun = self.loss_common
            self.lossesfun = self.losses_pyramid1

    def weight_common(self, disp, disp_wrap, factor=1.0):
        disp_delt = torch.abs(disp - disp_wrap).detach()/factor
        weight = Variable(torch.zeros(disp_delt.shape), requires_grad=False).type_as(disp_delt)
        mask1 = disp_delt<1
        mask2 = (disp_delt<3) - mask1
        mask3 = disp_delt >= 3
        weight[mask1] = 1.0
        weight[mask2] = 1.0 - (disp_delt[mask2] - 1)*(0.99/2)
        weight[mask3] = 0.01
        msg = "weight maxV: %f, minV: %f" % (weight.max().data[0], weight.min().data[0])
        logging.debug(msg)
        return weight
        
    # losses for loss_supervised
    def losses_pyramid0(self, disp_gt, disps, scale_disps, flag_smooth=False):
        count = len(scale_disps)
        levels = max(scale_disps) + 1
        disps_gt = create_impyramid(disp_gt, levels)
        loss = 0
        for i in range(0,  count):
            level = scale_disps[i]
            gt = disps_gt[level]
            pred = disps[level]
            loss = loss + self.lossfun(gt, pred, flag_smooth, factor=(2**level)) / (4**level)
        return loss

    # losses for loss_depthmono
    def losses_pyramid1(self, imR_src, imL, dispLs, scale_dispLs, LeftTop, imR1_src, imL1, dispL1s, scale_dispL1s, LeftTop1):
        imLs_wrap = imswrap(imR_src, dispLs, scale_dispLs, fliplr=False, LeftTop=LeftTop)
        imL1s_wrap = imswrap(imR1_src, dispL1s, scale_dispL1s, fliplr=False, LeftTop=LeftTop1)
        dispLs_wrap = imswrap(dispL1s[0], dispLs, scale_dispLs, fliplr=True, LeftTop=[0, 0])
        dispL1s_wrap = imswrap(dispLs[0], dispL1s, scale_dispL1s, fliplr=True, LeftTop=[0, 0])
        # compute loss
        count = len(scale_dispLs)
        levels = max(scale_dispLs) + 1
        imLs = create_impyramid(imL, levels)
        imL1s = create_impyramid(imL1, levels)
        loss = 0
        for i in range(0,  count):
            level = scale_dispLs[i]
            weight_common = self.weight_common(dispLs[i], dispLs_wrap[i], factor=(2**level)) if self.flag_mask else None
            weight_common1 = self.weight_common(dispL1s[i], dispL1s_wrap[i], factor=(2**level)) if self.flag_mask else None
            tmp = self.lossfun(imLs[level], imLs_wrap[i], dispLs[i], dispLs_wrap[i], factor=(2**level), weight_common=weight_common)
            tmp1 = self.lossfun(imL1s[level], imL1s_wrap[i], dispL1s[i], dispL1s_wrap[i], factor=(2**level), weight_common=weight_common1)
            loss = loss + (tmp + tmp1)/(4**level)
        # imshow
        if(flag_imshow):
            imsplot_tensor(imL, imL1, imLs_wrap[0], imL1s_wrap[0], 
                           dispLs[0], dispL1s[0], dispLs_wrap[0], dispL1s_wrap[0])
            import matplotlib.pyplot as plt
            plt.savefig("tmp_check.png")
        
        return loss
    
    # losses for loss_SsSMnet
    def losses_pyramid2(self, imR_src, imL, dispLs, scale_dispLs, LeftTop, imR1_src, imL1, dispL1s, scale_dispL1s, LeftTop1):
        imLs_wrap = imswrap(imR_src, dispLs, scale_dispLs, fliplr=False, LeftTop=LeftTop)
        imL1s_wrap = imswrap(imR1_src, dispL1s, scale_dispL1s, fliplr=False, LeftTop=LeftTop1)
        imLs_wrap1 = imswrap(imL1s_wrap[0], dispLs, scale_dispLs, fliplr=True, LeftTop=[0, 0])
        imL1s_wrap1 = imswrap(imLs_wrap[0], dispL1s, scale_dispL1s, fliplr=True, LeftTop=[0, 0])
        if(self.flag_mask):
            dispLs_wrap = imswrap(dispL1s[0], dispLs, scale_dispLs, fliplr=True, LeftTop=[0, 0])            
            dispL1s_wrap = imswrap(dispLs[0], dispL1s, scale_dispL1s, fliplr=True, LeftTop=[0, 0])
        # compute loss
        count = len(scale_dispLs)
        levels = max(scale_dispLs) + 1
        imLs = create_impyramid(imL, levels)
        imL1s = create_impyramid(imL1, levels)
        loss = 0
        for i in range(0,  count):
            level = scale_dispLs[i]
            weight_common = self.weight_common(dispLs[i], dispLs_wrap[i], factor=(2**level)) if self.flag_mask else None
            weight_common1 = self.weight_common(dispL1s[i], dispL1s_wrap[i], factor=(2**level)) if self.flag_mask else None
            tmp = self.lossfun(imLs[level], imLs_wrap[i], imLs_wrap1[i], dispLs[i], factor=(2**level), weight_common=weight_common)
            tmp1 = self.lossfun(imL1s[level], imL1s_wrap[i], imL1s_wrap1[i], dispL1s[i], factor=(2**level), weight_common=weight_common1)
            loss = loss + (tmp + tmp1)/(4**level)
        # imshow
        if(flag_imshow):
            imsplot_tensor(imL, imL1, imLs_wrap[0], imL1s_wrap[0], 
                            imLs_wrap1[0], imL1s_wrap1[0], dispLs[0], dispL1s[0])
            import matplotlib.pyplot as plt
            plt.savefig("tmp_check.png")
        return loss
    
    def forward(self, args):
        return self.lossesfun(**args)

    def computeloss(self, **args):
        return self.lossesfun(**args)

