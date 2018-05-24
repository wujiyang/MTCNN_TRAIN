#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:41:52 2018

@author: wujiyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)
        

def compute_accuracy(prob_cls, gt_cls):
    '''return a tensor which contains predicted accuracy'''
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)
    
    # only positive and negative examples has the classification loss which labels 1 and 0
    mask = torch.ge(gt_cls, 0)
    valid_gt_cls = torch.masked_select(gt_cls,mask)
    valid_prob_cls = torch.masked_select(prob_cls,mask)
    # computer predicted accuracy
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls,0.6).float()
    right_ones = torch.eq(prob_ones,valid_gt_cls).float()

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))

        
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
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # only use negative samples and positive samples for classification which labels 0 and 1
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.loss_cls(valid_pred_label, valid_gt_label) * self.cls_factor
    
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
        valid_gt_offset = gt_offset[chose_index, :]
        valid_pred_offset = pred_offset[chose_index, :]
        return self.loss_box(valid_pred_offset, valid_gt_offset) * self.box_factor
    
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
        return self.loss_landmark(valid_pred_landmark,valid_gt_landmark) * self.land_factor
    
    

class PNet(nn.Module):
    '''PNet'''
    def __init__(self, is_train=False, use_cuda=True):
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
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

        # weight initiation with xavier
        self.apply(weights_init)
        
    def forward(self, x):
        x = self.pre_layer(x)
        label = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        # landmark = self.conv4_3(x)

        if self.is_train is True:
            return label,offset
        
        return label, offset
        
    
class RNet(nn.Module):
    ''' RNet '''

    def __init__(self,is_train=False, use_cuda=True):
        super(RNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(28, 48, kernel_size=3, stride=1), 
            nn.PReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(48, 64, kernel_size=2, stride=1), 
            nn.PReLU()  

        )
        # this is little different from MTCNN paper, cause in pytroch, pooliing is calculated by floor()
        self.conv4 = nn.Linear(64*2*2, 128)  
        self.prelu4 = nn.PReLU() 
        # face calssification
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        #x = x.view(-1, x.size(0))
        x = x.view(-1, 64 * 2 * 2)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)

        if self.is_train is True:
            return det, box
        
        return det, box
    

class ONet(nn.Module):
    ''' ONet '''
    def __init__(self, is_train=False, use_cuda=True):
        super(ONet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), 
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), 
            nn.Conv2d(64,128,kernel_size=2,stride=1), 
            nn.PReLU() 
        )
        self.conv5 = nn.Linear(128*2*2, 256) 
        self.prelu5 = nn.PReLU() 
        # face classification
        self.conv6_1 = nn.Linear(256, 1)
        # bounding box regression
        self.conv6_2 = nn.Linear(256, 4)
        # lanbmark localization
        self.conv6_3 = nn.Linear(256, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(-1, 128*2*2)
        x = self.conv5(x)
        x = self.prelu5(x)
        # detection
        det = torch.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        if self.is_train is True:
            return det, box, landmark
        
        return det, box, landmark
    
        
        