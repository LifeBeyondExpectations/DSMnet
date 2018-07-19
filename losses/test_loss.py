import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from util.util import to_tensor, imsplot_tensor
from dataloader.imagerw import load_image, load_disp #, save_image
from loss import losses

if __name__ == "__main__":
    dirpath = r"/media/qjc/D/data/testimgs_stereo/"
#    img1 = load_image(dirpath + "im0.png").transpose(2, 0, 1)[None]/255.0
#    img2 = load_image(dirpath + "im1.png").transpose(2, 0, 1)[None]/255.0
#    disp1 = load_disp(dirpath + "disp0.pfm")[None, None]/1.0
#    disp2 = load_disp(dirpath + "disp1.pfm")[None, None]/1.0
    img1 = load_image(dirpath + "im2.ppm").transpose(2, 0, 1)[None]/255.0
    img2 = load_image(dirpath + "im6.ppm").transpose(2, 0, 1)[None]/255.0
    disp1 = load_disp(dirpath + "disp2.pgm")[None, None]/8.0
    disp2 = load_disp(dirpath + "disp6.pgm")[None, None]/8.0
    img1t = to_tensor(np.flip(img2, axis=-1).copy())
    img2t = to_tensor(np.flip(img1, axis=-1).copy())
    img1 = to_tensor(img1, requires_grad=True)
    img2 = to_tensor(img2, requires_grad=True)
    disp1 = to_tensor(disp1, requires_grad=True)
    disp1t = to_tensor(np.flip(disp2, axis=-1).copy(), requires_grad=True)
   
    lossfun = losses(loss_name="depthmono")
    "imR_src, imL, dispLs, scale_dispLs, LeftTop, imR1_src, imL1, dispL1s, scale_dispL1s, LeftTop1"
    args = {
        "imR_src": img2, "imL": img1, "dispLs": [disp1], "scale_dispLs": [0], "LeftTop": [0, 0], 
        "imR1_src": img2t, "imL1": img1t, "dispL1s": [disp1t], "scale_dispL1s": [0], "LeftTop1": [0, 0], 
        }
    loss = lossfun(args)
    loss.backward(retain_graph=True)
    print loss.data[0]

    lossfun = losses(loss_name="depthmono-mask")
    "imR_src, imL, dispLs, scale_dispLs, LeftTop, imR1_src, imL1, dispL1s, scale_dispL1s, LeftTop1"
    args = {
        "imR_src": img2, "imL": img1, "dispLs": [disp1], "scale_dispLs": [0], "LeftTop": [0, 0], 
        "imR1_src": img2t, "imL1": img1t, "dispL1s": [disp1t], "scale_dispL1s": [0], "LeftTop1": [0, 0], 
        }
    loss = lossfun(args)
    loss.backward(retain_graph=True)
    print loss.data[0]

    lossfun = losses(loss_name="Cap_ds_lr-mask")
    "imR_src, imL, dispLs, scale_dispLs, LeftTop, imR1_src, imL1, dispL1s, scale_dispL1s, LeftTop1"
    args = {
        "imR_src": img2, "imL": img1, "dispLs": [disp1], "scale_dispLs": [0], "LeftTop": [0, 0], 
        "imR1_src": img2t, "imL1": img1t, "dispL1s": [disp1t], "scale_dispL1s": [0], "LeftTop1": [0, 0], 
        }
    loss = lossfun(args)
    loss.backward(retain_graph=True)
    print loss.data[0]

    lossfun = losses(loss_name="SsSMnet-mask")
    "imR_src, imL, dispLs, scale_dispLs, LeftTop, imR1_src, imL1, dispL1s, scale_dispL1s, LeftTop1"
    args = {
        "imR_src": img2, "imL": img1, "dispLs": [disp1], "scale_dispLs": [0], "LeftTop": [0, 0], 
        "imR1_src": img2t, "imL1": img1t, "dispL1s": [disp1t], "scale_dispL1s": [0], "LeftTop1": [0, 0], 
        }
    loss = lossfun(args)
    loss.backward(retain_graph=True)
    print loss.data[0]

    if(True):
        imsplot_tensor(img1, img1t, disp1, disp1t)
        plt.show()
    
