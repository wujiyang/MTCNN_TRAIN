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
        
    if o_model_path is not None:
        onet = ONet(use_cuda=use_cuda)
        onet.load_state_dict(torch.load(o_model_path))
        if(use_cuda):
            onet.to(device)
        
        onet.eval()
        
    return pnet, rnet, onet



class MtcnnDetector(object):
    ''' P, R, O net for face detection and landmark alignment'''
    def __init__(self, 
                 pnet=None, 
                 rnet=None, 
                 onet=None, 
                 min_face_size=12, 
                 stride=2, 
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709):
        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride=stride
        self.thresh = threshold
        self.scale_factor = scale_factor
    
    def unique_image_format(self, im):
         if not isinstance(im,np.ndarray):
            if im.mode == 'I':
                im = np.array(im, np.int32, copy=False)
            elif im.mode == 'I;16':
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.asarray(im)
                
         return im
    
    def square_box(self, bbox):
        '''
        convert bbox to square
        Parameters:
            bbox: numpy array, shape n x m
        Returns:
            square bbox
        '''
        square_bbox = bbox.copy()
        h = bbox[:, 3] - bbox[:, 1]
        w = bbox[:, 2] - bbox[:, 0]
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side
        square_bbox[:, 3] = square_bbox[:, 1] + max_side
    
        return square_bbox
    
    
    def generate_bounding_box(self, map, reg, scale, threshold):
        """ TODO： 这个函数没看懂 """
        '''
        generate bbox from feature map
        for PNet, there exists no fc layer, only convolution layer ,so feature map n x m x 1/4
        Parameters:
            map: numpy array , n x m x 1, detect score for each position
            reg: numpy array , n x m x 4, bbox
            scale: float number, scale of this detection
            threshold: float number, detect threshold
        Returns:
            bbox array
        '''
        stride = 2
        cellsize = 12

        t_index = np.where(map > threshold)
        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])
        
        score = map[t_index[0], t_index[1], 0]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg,
                                 # landmarks
                                 ])

        return boundingbox.T
    
    
    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel,input image, channels in BGR order here
            scale: float number, scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        
        return img_resized
    
    def pad(self, bboxes, w, h):
        """
            pad the the boxes
        Parameters:
        ----------
            bboxes: numpy array, n x 5, input bboxes
            w: float number, width of the input image
            h: float number, height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1, start point of the bbox in target image
            edy, edx : numpy array, n x 1, end point of the bbox in target image
            y, x : numpy array, n x 1, start point of the bbox in original image
            ey, ex : numpy array, n x 1, end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1, height and width of the bbox
        """
        
        tmpw = (bboxes[:, 2] - bboxes[:, 0]).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1]).astype(np.int32)
        numbox = bboxes.shape[0]

        dx = np.zeros((numbox, ))
        dy = np.zeros((numbox, ))
        edx, edy  = tmpw.copy(), tmph.copy()
        
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        
        tmp_index = np.where(ex > w)
        edx[tmp_index] = tmpw[tmp_index] + w - 1 - ex[tmp_index]
        ex[tmp_index] = w

        tmp_index = np.where(ey > h)
        edy[tmp_index] = tmph[tmp_index] + h - 1 - ey[tmp_index]
        ey[tmp_index] = h

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list
        
    
    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array, input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        
    




































































