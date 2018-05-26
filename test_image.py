#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:57:59 2018

@author: wujiyang
"""

import sys
sys.path.append("/home/wujiyang/FaceProjects/MTCNN_TRAIN")

import cv2
from tools.detect import create_mtcnn_net, MtcnnDetector
import tools.vision as vision


if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./model_store/pnet_model_final.pt",
                                        r_model_path="./model_store/rnet_model_final.pt",
                                        o_model_path="./model_store/onet_model_final.pt", 
                                        use_cuda=False)
    
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    
    #print bboxs.shape[0]
    #print landmarks.shape[0]

    vision.vis_face(img, bboxs, landmarks)