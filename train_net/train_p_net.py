#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:48:03 2018

@author: wujiyang
"""

import sys
sys.path.append("/home/wujiyang/FaceProjects/MTCNN_TRAIN")

import os
import argparse
import datetime
import torch
import config
from tools.image_reader import TrainImageReader
from train_net.models import PNet, LossFn
from train_net.models import compute_accuracy
import tools.image_tools as image_tools
from tools.imagedb import ImageDB


def train_p_net(annotation_file, model_store_path, end_epoch=50, frequent=200, base_lr=0.01, batch_size=256, use_cuda=True):
    
    # initialize the PNet ,loss function and set optimization for this network
    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)
    net = PNet(is_train=True, use_cuda=use_cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_cuda:
        net.to(device)
    lossfn = LossFn()
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 40], gamma=0.1)
    # load training image
    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb)
    train_data = TrainImageReader(gt_imdb, 12, batch_size, shuffle=True)
    
    # train net 
    net.train()
    for cur_epoch in range(end_epoch):
        scheduler.step()
        train_data.reset() # shuffle the data for this epoch
        for batch_idx, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):
            im_tensor = [image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0])]
            im_tensor = torch.stack(im_tensor)
            
            gt_label = torch.from_numpy(gt_label).float()
            gt_bbox = torch.from_numpy(gt_bbox).float()
            # gt_landmark = torch.from_numpy(gt_landmark).float()
            if use_cuda:
                im_tensor = im_tensor.to(device)
                gt_label = gt_label.to(device)
                gt_bbox = gt_bbox.to(device)
            
            cls_pred, box_offset_pred = net(im_tensor)
            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)
            all_loss = cls_loss * 1.0 + box_offset_loss * 0.5
            
            if batch_idx % frequent == 0:
                accuracy = compute_accuracy(cls_pred, gt_label)
                print("[%s, Epoch: %d, Step: %d] accuracy: %.6f, all_loss: %.6f, cls_loss: %.6f, bbox_reg_loss: %.6f, lr: %.6f" % 
                      (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cur_epoch + 1, batch_idx, accuracy.data.tolist(), 
                       all_loss.data.tolist(), cls_loss.data.tolist(), box_offset_loss.data.tolist(), scheduler.get_lr()[0]))
            
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
        
        # TODO: add validation set for trained model   
        
        if (cur_epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(model_store_path,"pnet_model_epoch_%d.pt" % (cur_epoch + 1)))

    torch.save(net.state_dict(), os.path.join(model_store_path, 'pnet_nodel_final.pt'))




def parse_args():
    parser = argparse.ArgumentParser(description='Train PNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--anno_file', dest='annotation_file', help='training data annotation file', 
                        default=os.path.join(config.ANNO_STORE_DIR,config.PNET_TRAIN_IMGLIST_FILENAME), type=str)
    parser.add_argument('--model_path', dest='model_store_path', help='training model store directory',
                        default=config.MODLE_STORE_DIR, type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=config.END_EPOCH, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--base_lr', dest='base_lr', help='learning rate',
                        default=config.TRAIN_LR, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=config.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=config.USE_CUDA, type=bool)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # print('train Pnet argument:')
    # print(args)

    train_p_net(annotation_file=args.annotation_file, model_store_path=args.model_store_path,
                end_epoch=args.end_epoch, frequent=args.frequent, base_lr=args.base_lr, batch_size=args.batch_size, use_cuda=args.use_cuda)
