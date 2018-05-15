#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:37:59 2018

@author: wujiyang
"""

import numpy as np

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    
    # compute the width and height of the inter box 
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    
    inter = w * h
    ovr = np.true_divide(inter, (box_area + area - inter))
    
    
    return ovr