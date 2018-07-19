import os
import torch
import numpy as np
from util.util import mycrop, to_tensor, to_numpy
from models import model_create_by_name as model_create
from losses.loss import losses

class model_stereo(object):

    def __init__(self, args):
        super(model_stereo, self).__init__()
        self.updated = 0
        self.size_crop = [512, 256] # [1024, 512] # [256, 128] # 
        self.lr = args.lr
        self.alpha = args.alpha
        self.betas = (args.beta1, args.beta2)
        self.lr_adjust_start = args.lr_adjust_start
        self.lr_adjust_stride = args.lr_adjust_stride
        self.flag_lr_adjust = False
    
    def save_weight(self, flag_postfix=False):
        postfix = ''
        if(flag_postfix):
            postfix = '_{:06d}'.format(self.updated)
        filename = 'model{}.pkl'.format(postfix)
        path_model = os.path.join(self.dirpath, filename)
        if(not os.path.exists(self.dirpath)):
            os.makedirs(self.dirpath)
        data = {"state_dict": self.model.state_dict(), 
                'num_update': self.updated, 
                }
        # make sure saving complete weight 
        path_model_tmp = path_model + ".tmp"
        torch.save(data, path_model_tmp)
        if(os.path.exists(path_model)): 
            os.unlink(path_model)
        os.rename(path_model_tmp, path_model)
    
    def load_weight(self, postfix='', path_pretrained=""):
        filename = 'model{}.pkl'.format(postfix)
        path_model = os.path.join(self.dirpath, filename)
        if(os.path.exists(path_model)):
            data = torch.load(path_model)
            self.model.load_state_dict(data['state_dict'])
            self.updated = data['num_update']
        elif(os.path.exists(path_pretrained)):
            data = torch.load(path_pretrained)
            self.model.load_state_dict(data['state_dict'])
            self.updated = 0
    
    def optim_create(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas)
        #return torch.optim.RMSprop(model.parameters(), lr=self.lr, alpha=self.alpha)
    
    def lr_adjust(self):
        lr = self.lr
        start, stride = self.lr_adjust_start, self.lr_adjust_stride
        if(self.updated < start):
            return
        if 0 == ((self.updated - start)%stride):
            self.flag_lr_adjust = False
        if(self.flag_lr_adjust): return
        n = 1 + (self.updated - start)//stride
        lr = self.lr /(2**n)
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
            self.flag_lr_adjust = True
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def estimate(self, imL, imR):
        imL = to_tensor(imL, volatile=True, requires_grad=False)
        imR = to_tensor(imR, volatile=True, requires_grad=False)
        _, dispLs = self.model(imL, imR)
        return to_numpy(dispLs[0])

    def augment_color_pair(self, left_image, right_image):
        # randomly augment image
        bn, c = left_image.shape[:2]
        # randomly shift gamma
        do_augment = torch.rand(1)[0]
        if(do_augment > 0.5):
            random_gamma = to_tensor(torch.rand(bn)*0.4 + 0.8)
            left_image  = (left_image.transpose(0,-1)  ** random_gamma).transpose(0,-1)
            random_gamma = to_tensor(torch.rand(bn)*0.4 + 0.8)
            right_image = (right_image.transpose(0,-1) ** random_gamma).transpose(0,-1)

        # randomly shift brightness
        do_augment  = torch.rand(1)[0]
        if(do_augment > 0.5):
            random_brightness =  to_tensor(torch.rand(bn)*1.5 + 0.5)
            left_image  =  (left_image.transpose(0,-1) * random_brightness).transpose(0,-1)
            random_brightness =  to_tensor(torch.rand(bn)*1.5 + 0.5)
            right_image = (right_image.transpose(0,-1) * random_brightness).transpose(0,-1)

        # randomly shift color
        do_augment  = torch.rand(1)[0]
        if(do_augment > 0.5):
            random_colors = torch.rand(bn, c)*0.4 + 0.8
            color_image = (torch.ones(left_image.shape).permute(2,3,0,1)*random_colors).permute(2, 3, 0, 1)
            left_image  = left_image * to_tensor(color_image)
            random_colors = torch.rand(bn, c)*0.4 + 0.8
            color_image = (torch.ones(left_image.shape).permute(2,3,0,1)*random_colors).permute(2, 3, 0, 1)
            right_image = right_image * to_tensor(color_image)

        # saturate
        do_augment  = torch.rand(1)[0]
        if(do_augment > 0.5):
            left_image  = left_image.clamp(0, 1)
            right_image = right_image.clamp(0, 1)

        return left_image, right_image

    def predict(self, imL, imR, augment=True):
        if(augment):
            imL, imR = self.augment_color_pair(imL, imR)
        return self.model(imL, imR)
        
    def update_selfsupervised(self, imL_src, imR_src):
        assert len(imL_src.shape) == 4
        if(torch.is_tensor(imL_src)):
            imL_src = imL_src.numpy()
            imR_src = imR_src.numpy()
        bn, c, h0, w0 = imL_src.shape
        sw, sh, ew, eh = mycrop(w0, h0, self.size_crop, size_cell=64)
        # image prepare
        imL = imL_src[:, :, sh:eh, sw:ew]
        imR = imR_src[:, :, sh:eh, sw:ew]
        imL1 = np.flip(imR, axis=-1).copy()
        imR1 = np.flip(imL, axis=-1).copy()
        imL = to_tensor(imL, volatile=False, requires_grad=False)
        imR = to_tensor(imR, volatile=False, requires_grad=False)
        imL1 = to_tensor(imL1, volatile=False, requires_grad=False)
        imR1 = to_tensor(imR1, volatile=False, requires_grad=False)
        # disp estimate
        scale_dispLs, dispLs = self.predict(imL, imR)
        scale_dispL1s, dispL1s = self.predict(imL1, imR1)
        # compute loss and backward
        if(self.lossfun.flag_mask):
            imR_src = to_tensor(imR_src, requires_grad=False)
            imR1_src = to_tensor(np.flip(imL_src, axis=-1).copy(), requires_grad=False)
            LeftTop = [sw, sh]
            LeftTop1 = [w0-ew, sh]
        else:
            imR_src = imR
            imR1_src = imR1
            LeftTop = [0, 0]
            LeftTop1 = [0, 0]
        args = {
            "imR_src": imR_src, "imL": imL, "dispLs": dispLs, "scale_dispLs": scale_dispLs, "LeftTop": LeftTop, 
            "imR1_src": imR1_src, "imL1": imL1, "dispL1s": dispL1s, "scale_dispL1s": scale_dispL1s, "LeftTop1": LeftTop1, 
            }
        loss = self.lossfun(args)
        assert loss.data[0]>0
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.updated += 1
        self.lr_adjust()
        return loss.data[0]

    def update_supervised(self, imL_src, imR_src, dispL):
        assert len(imL_src.shape) == 4
        bn, c, h0, w0 = imL_src.shape
        sw, sh, ew, eh = mycrop(w0, h0, self.size_crop, size_cell=64)
        # image prepare
        imL = imL_src[:, :, sh:eh, sw:ew]
        imR = imR_src[:, :, sh:eh, sw:ew]
        dispL = dispL[:, :, sh:eh, sw:ew]
        imL = to_tensor(imL, volatile=False, requires_grad=False)
        imR = to_tensor(imR, volatile=False, requires_grad=False)
        dispL = to_tensor(dispL, volatile=False, requires_grad=False)
        # disp estimate
        scale_dispLs, dispLs = self.predict(imL, imR)
        # compute loss and backward
        args = {
            "disp_gt": dispL, "disps": dispLs, "scale_disps": scale_dispLs, 
            "flag_smooth": True, 
            }
        loss = self.lossfun(args)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.updated += 1
        self.lr_adjust()
        return loss.data[0]

    def update_self(self, imL_src, imR_src):
        assert len(imL_src.shape) == 4
        bn, c, h0, w0 = imL_src.shape
        sw, sh, ew, eh = mycrop(w0, h0, self.size_crop, size_cell=64)
        # image prepare
        imL = imL_src[:, :, sh:eh, sw:ew]
        imR = imR_src[:, :, sh:eh, sw:ew]
        imL = to_tensor(imL, volatile=False, requires_grad=False)
        imR = to_tensor(imR, volatile=False, requires_grad=False)
        # disp estimate
        dispLs = self.model(imL, imL)
        dispRs = self.model(imR, imR)
        # compute loss and backward
        loss = 0
        for disp in dispLs:
            loss = loss + disp.mean()
        for disp in dispRs:
            loss = loss + disp.mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

class models_stereo(model_stereo):

    def __init__(self, args):
        super(models_stereo, self).__init__(args)
        self.name = args.net
        # dirpath of saving weight
        self.dirpath = os.path.join(args.output, "%s_%s_%s_%s" % (args.mode, args.dataset, args.net, args.loss_name))
        # model, optim and lossfun
        self.model = model_create(self.name, args.maxdisparity)
        self.optim = self.optim_create(self.model)
        self.lossfun = losses(loss_name=args.loss_name)
    
