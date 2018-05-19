#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 20:55:10 2018

@author: wujiyang
"""

import sys
sys.path.append("/home/wujiyang/MTCNN_TRAIN")

import cv2
import time
import numpy as np
import torch
from train_net.models import PNet, RNet, ONet
import tools.utils as utils
import tools.image_tools as image_tools


def create_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None, use_cuda=True):
    
    pnet, rnet, onet = None, None, None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if p_model_path is not None:
        pnet = PNet(use_cuda=use_cuda)
        pnet.load_state_dict(torch.load(p_model_path))
        if(use_cuda):
            pnet.to(device)
        
        pnet.eval()
    
    if r_model_path is not None:
        rnet = RNet(use_cuda=use_cuda)
        rnet.load_state_dict(torch.load(r_model_path))
        if(use_cuda):
            rnet.to(device)
        
        rnet.eval()
        
    if o_net_path is not None:
        onet = ONet(use_cuda=use_cuda)
        onet.load_state_dict(torch.load(o_model_path))
        if(use_cuda):
            onet.to(device)
        
        onet.eval()
        
    return pnet, rnet, onet

