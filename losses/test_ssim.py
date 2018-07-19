import sys
sys.path.append("../")
import torch
import matplotlib.pyplot as plt
from util.util import to_tensor, imsplot_tensor
from util.imwrap import imwrap_BCHW # as imwrap
from dataloader.imagerw import load_image, load_disp #, save_image
from SSIM import SSIM

def implot(im1, im2, im3, im4, im5, im6, im7, im8):
    m = 4
    n = 2
    ims = [im1, im2, im3, im4, im5, im6, im7, im8]
    for i in range(m*n):
        ax = plt.subplot(m, n, i+1)
        plt.sca(ax)
        plt.imshow(ims[i])

def test():
    dirpath = r"/media/qjc/D/data/testimgs_stereo/"
#    img1 = load_image(dirpath + "im0.png").transpose(2, 0, 1)[None]/255.0
#    img2 = load_image(dirpath + "im1.png").transpose(2, 0, 1)[None]/255.0
#    disp1 = load_disp(dirpath + "disp0.pfm")[None, None]/1.0
#    disp2 = load_disp(dirpath + "disp1.pfm")[None, None]/1.0
    img1 = load_image(dirpath + "im2.ppm").transpose(2, 0, 1)[None]/255.0
    img2 = load_image(dirpath + "im6.ppm").transpose(2, 0, 1)[None]/255.0
    disp1 = load_disp(dirpath + "disp2.pgm")[None, None]/8.0
    disp2 = load_disp(dirpath + "disp6.pgm")[None, None]/8.0
#    img1 = load_image(dirpath + "10L.png").transpose(2, 0, 1)[None]/255.0
#    img2 = load_image(dirpath + "10R.png").transpose(2, 0, 1)[None]/255.0
#    disp1 = load_disp(dirpath + "disp10L.png")[None, None]
#    disp2 = load_disp(dirpath + "disp10R.png")[None, None]
    im1 = to_tensor(img1)
    im2 = to_tensor(img2)
    d1 = to_tensor(disp1)
    d2 = to_tensor(disp2)
    im1t = imwrap_BCHW(im2, -d1)
    im2t = imwrap_BCHW(im1, d2)
    ssim = SSIM(window_size = 11)
    ssim1 = ssim(im1, im1t)
    ssim2 = ssim(im2, im2t)
    ssim3 = ssim(im1, im2)
    abs1 = torch.abs(im1 - im1t).sum(dim=1, keepdim=True)
    abs2 = torch.abs(im2 - im2t).sum(dim=1, keepdim=True)
    print ssim1.shape, ssim2.shape
    print ssim1.mean().data[0], ssim2.mean().data[0], ssim3.mean().data[0]
    imsplot_tensor(im1, im2, im1t, im2t, 1-ssim1, 1-ssim2, abs1, abs2)
    plt.show()

test()
