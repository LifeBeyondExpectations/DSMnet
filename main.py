#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import os
import time
import numpy as np
import torch
#import copy
import cv2
import matplotlib.pyplot as plt
from stereo import models_stereo
from dataloader.dataloader import myDataLoader
from util.evaluate import evaluate
import traceback
import logging
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of program')

def train(dataloader, dataloader_val, model, args):
    if(model.updated >= args.updates):
        print("Training Finished!")
        return
    #------------------------------------------prepare----------------------------------------------
    mlosses = []
    merrors = []
    path_loss = os.path.join(model.dirpath, 'loss.pkl')
    if(os.path.exists(path_loss)):
        data = torch.load(path_loss)
        mlosses.extend(data['loss'])
        merrors.extend(data['error'])
    count = len(dataloader)
    epoches = (args.updates + count/2)//count
    tepoch = (model.updated)//count
    time_start_full = time.time()
    flag_val = False
    plt.figure()
    for epoch in range(tepoch+1, epoches+1):
        msg = "This is %d-th epoch, updated: %6d, lr: %f " %(epoch, model.updated, model.optim.param_groups[0]['lr'])
        logging.info(msg)
        #----------------------------------Training--------------------------------------------
        model.train()
        loss_train_total = 0
        for batch_idx, batch_stereo in enumerate(dataloader):
            time_start = time.time()
            try:
                if(args.loss_name == "supervised"):
                    loss = model.update_supervised(*batch_stereo[:3])
                else:
                    loss = model.update_selfsupervised(*batch_stereo[:2])
            except Exception as err:
                logging.error(err)
                logging.error(traceback.format_exc())
                continue;
            if(batch_idx%20==0):
                msg = "Iter: %d, training loss: %.3f, time: %.2f" %(batch_idx, loss, time.time()-time_start)
                logging.info(msg)
            loss_train_total += loss
            if(model.updated%1000 == 1): flag_val = True
            if(model.updated%100000 == 0): model.save_weight(flag_postfix=True)
        loss_mean = loss_train_total/len(dataloader)
        mlosses.append([epoch, loss_mean])
        model.save_weight(flag_postfix=False)
        time_full_h = (time.time() - time_start_full)/3600
        msg = "**Train phase** Total mean training loss: %.3f, full training time: %.2f HR " % (loss_mean, time_full_h)
        logging.info(msg)
        #----------------------------------validate--------------------------------------------
        if(flag_val):
            flag_val = False
            model_val = model
            model_val.eval()
            terrors = []
            for batch_idx, (img1, img2, disp1) in enumerate(dataloader_val):
                if(batch_idx > 50): break
                # visual check
                dispL = model_val.estimate(img1, img2)
                error = evaluate(img1, img2, dispL, disp1.numpy())
                terrors.append(error)
            merror = np.mean(terrors, axis=0)
            merrors.append([epoch, merror[0], merror[1], merror[2]])
            #----------------------------------visual check--------------------------------------------
            torch.save({'loss': mlosses, 'error': merrors}, path_loss)
            msg = "**Validate phase** epoch: %d, d1: %.2f, EPE: %.2f, pixelerror: %.2f" % (epoch, merror[0], merror[1], merror[2])
            logging.info(msg)
            
            mlosses_np = np.array(mlosses)
            merrors_np = np.array(merrors)
            msg = str(mlosses_np.shape) + str(merrors_np.shape)
            logging.debug(msg)
            plt.subplot(221); plt.cla(); plt.plot(mlosses_np[:, 0], mlosses_np[:, 1]); plt.xlabel("epoch"); plt.ylabel("mean train loss")
            plt.subplot(222); plt.cla(); plt.plot(merrors_np[:,0], merrors_np[:, 1]); plt.xlabel("epoch"); plt.ylabel("D1")
            plt.subplot(223); plt.cla(); plt.plot(merrors_np[:,0], merrors_np[:, 2]); plt.xlabel("epoch"); plt.ylabel("EPE")
            plt.subplot(224); plt.cla(); plt.plot(merrors_np[:,0], merrors_np[:, 3]); plt.xlabel("epoch"); plt.ylabel("RPE")
            plt.savefig("check_%s_%s_%s_%s.png" % (args.mode, args.dataset, args.net, args.loss_name))
    
    print("Training Finish!")

def test(dataloader, model, args):
    # testing
    times = []
    errors = []
    dir_testInfor = "testInfo"
    if(not os.path.exists(dir_testInfor)):
        os.makedirs(dir_testInfor)
    path_testInfo = os.path.join(dir_testInfor, "%s_%s.pkl" % (args.dataset, args.flag_model))
    if(os.path.exists(path_testInfo)):
        data = torch.load(path_testInfo)
        times.extend(data['time'])
        errors.extend(data['err'])
        for i in range(len(times)):
            print("test: %4d | time: %6.3f, d1: %6.3f, EPE: %6.3f, pixelerror: %6.3f" %
                   (i, times[i], errors[i][0], errors[i][1], errors[i][2]))
    model.eval()
    for batch_idx, (img1, img2, dispL_gt) in enumerate(dataloader):
        if(batch_idx < len(errors)):
            continue
        stime = time.time()
        dispL = model.estimate(img1, img2)
        times.append(time.time() - stime)
        error = evaluate(img1, img2, dispL, dispL_gt.numpy())
        errors.append(error)
        print("test: %4d | time: %6.3f, d1: %6.3f, EPE: %6.3f, pixelerror: %6.3f" %
               (batch_idx, times[-1], error[0], error[1], error[2]))
        if(args.flag_save):
            dirpath_save = os.path.join(dir_testInfor, "%s_%s" % (args.dataset, args.flag_model))
            if(not os.path.exists(dirpath_save)):
                os.makedirs(dirpath_save)
            path_file = os.path.join(dirpath_save, "{:06d}.png".format(batch_idx))
            cv2.imwrite(path_file, dispL[0, 0])
#        cv2.imwrite("estimate/{:06d}_gt.png".format(step), dispL_gt[0])
        if(batch_idx%100==0):
            torch.save({"time":times, 'err':errors}, path_testInfo)
    torch.save({"time":times, 'err':errors}, path_testInfo)
    print np.mean(times), np.mean(errors, axis=0)

def submit(dataloader, model, args):
    # testing
    times = []
    errors = []
    dir_testInfor = "submit"
    if(not os.path.exists(dir_testInfor)):
        os.makedirs(dir_testInfor)
    path_testInfo = os.path.join(dir_testInfor, "%s_%s.pkl" % (args.dataset, args.flag_model))
    if(os.path.exists(path_testInfo)):
        data = torch.load(path_testInfo)
        times.extend(data['time'])
        for i in range(len(times)):
            print("test: %4d | time: %6.3f" % (i, times[i]))
    model.eval()
    for batch_idx, (img1, img2) in enumerate(dataloader):
        if(batch_idx < len(errors)):
            continue
        stime = time.time()
        dispL = model.estimate(img1, img2)
        times.append(time.time() - stime)
        print("test: %4d | time: %6.3f" % (batch_idx, times[-1]))
        if(True):
            dirpath_save = os.path.join(dir_testInfor, "%s_%s" % (args.dataset, args.flag_model))
            if(not os.path.exists(dirpath_save)):
                os.makedirs(dirpath_save)
            path_file = os.path.join(dirpath_save, "{:06d}.png".format(batch_idx))
            cv2.imwrite(path_file, dispL[0, 0])
        if(batch_idx%100==0):
            torch.save({"time":times}, path_testInfo)
    torch.save({"time":times}, path_testInfo)
    print np.mean(times)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch on stereo matching with Multi-model')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/finetune/test/submit')
    parser.add_argument('--dataset', default='kitti2015-tr', type=str, help='support option: [kitti2015/kitti2012/flyingthings3d/middlebury]-[tr/te]')
    parser.add_argument('--root', default='./kitti', type=str, help='root path of dataset')
    parser.add_argument('--dataset_val', default='kitti2015-tr', type=str, help='support option: [kitti2015/kitti2012/flyingthings3d/middlebury]-[tr/te]')
    parser.add_argument('--root_val', default='./kitti', type=str, help='root path of dataset_val')
    parser.add_argument('--path_model', default="", type=str, help='state dict of model for test or finetune')
    parser.add_argument('--flag_model', default='dispnetcorr', type=str, help='flag of model for test')
    parser.add_argument('--flag_save', default='false', type=bool, help='flag of saving test result')
    parser.add_argument('--updates', default=200000, type=int, help='max number of update to train')
    parser.add_argument('--display', default=100, type=int, help='frequent of display current result')
    parser.add_argument('--loss_name', default="supervised", type=str, help='support option: supervised/(depthmono/SsSMnet/Cap_ds_lr)[-mask]')
    parser.add_argument('--net', default='dispnet', type=str, help='support option: dispnet/dispnetcorr/iresnet/gcnet/...')
    parser.add_argument('--maxdisparity', default=192, type=int, help='')
    parser.add_argument('--batchsize', default=1, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for Adam optim')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta1 for Adam optim')
    parser.add_argument('--alpha', default=0.9, type=float, help='alpha for RMSprop optim')
    parser.add_argument('--lr_adjust_start', default=200000, type=float, help='start number of update to adjust learning rate')
    parser.add_argument('--lr_adjust_stride', default=100000, type=float, help='stride of update to adjust learning rate')
    parser.add_argument('--output', default='output', type=str, help='dirpath for save model and training losses')
    args = parser.parse_args()
    
    # stereo model and load weight
    model = models_stereo(args)
    model.load_weight(postfix='', path_pretrained=args.path_model)

    # train or test
    args.mode = args.mode.lower()
    if args.mode == 'test':
        # dataloader
        flag_disp = True # (args.losstype == "supervised" or args.mode == "test")
        dataloader = myDataLoader(training=False, dataset=args.dataset, root=args.root, batch_size=args.batchsize, num_workers=4, flag_disp=flag_disp)
        msg = "Test dataset: %s , model name: %s " % (args.dataset, args.net)
        logging.info(msg)
        test(dataloader, model, args)
    elif args.mode == 'submit' :
        # dataloader
        flag_disp = False # (args.losstype == "supervised" or args.mode == "test")
        dataloader = myDataLoader(training=False, dataset=args.dataset, root=args.root, batch_size=args.batchsize, num_workers=4, flag_disp=flag_disp)
        msg = "Submit dataset: %s , model name: %s " % (args.dataset, args.net)
        logging.info(msg)
        submit(dataloader, model, args)
    elif args.mode == 'train' or args.mode == 'finetune':
        # dataloader
        flag_disp = True # (args.losstype == "supervised" or args.mode == "test")
        dataloader = myDataLoader(training=True, dataset=args.dataset, root=args.root, batch_size=args.batchsize, num_workers=4, flag_disp=flag_disp)
        dataloader_val = myDataLoader(training=False, dataset=args.dataset_val, root=args.root_val, batch_size=args.batchsize, num_workers=4, flag_disp=flag_disp)
        msg = "Train dataset: %s , val dataset: %s " % (args.dataset, args.dataset_val)
        logging.info(msg)
        msg = "Model name: %s , loss name: %s , updated: %d" % (args.net, args.loss_name, model.updated)
        logging.info(msg)
        train(dataloader, dataloader_val, model, args)

