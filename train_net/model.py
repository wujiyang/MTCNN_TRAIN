#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:41:52 2018

@author: wujiyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)
        
        
class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        self.loss_cls = nn.BCELoss()
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()
        
    def cls_loss(self, gt_label, pred_label):
        gt_label = torch.squeeze(gt_label)
        pred_label = torch.squeeze(pred_label)
        # only use negative samples and positive samples for classification which labels 0 and 1
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.loss_cls(valid_gt_label, valid_pred_label) * self.cls_factor
    
    def box_loss(self, gt_label, gt_offset, pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        # only use positive samples and partface samples for bounding box regression which labels 1 and -1
        unmask = torch.eq(gt_label,0)
        mask = torch.eq(unmask,0)
        #convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        #only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]
        return self.loss_box(valid_pred_offset,valid_gt_offset)*self.box_factor
    
    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        # only CelebA data been used in landmark regression
        mask = torch.eq(gt_label, -2)
        
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark) * self.land_factor
    
    

class PNet(nn.Module):
    '''PNet'''
    def __init__(self, is_train=Fasle, use_cuda=True):
        super(PNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU()
        )
        
        # face classification
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regression
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark regression
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
        
        # weight initializaation with xavier
        self.apply(weight_init)
        
    def forward(self, x):
        x = self.pre_layer(x)
        label = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        # landmark = self.conv4_3(x)
        
        if self.is_train is True:
            return label, offset
        
        return label, offset
        