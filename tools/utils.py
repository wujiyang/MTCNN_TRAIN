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


def convert_to_square(bbox):
    ''' Convert bbox to a square which it can include the bbox
    Parameters:
        bbox: numpy array, shape n x 5
        
    returns:
        square box
    '''
    
    square_bbox = box.copy()
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    
    return square_bbox


def nms(dets, thresh, mode='Union'):
    ''' greedily select bboxes with high confidence,if an box overlap with the highest score box > thres, rule it out
    
    params:
        dets: [[x1, y1, x2, y2, score]]
        thresh: retain overlap <= thresh
    return:
        indexes to keep
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1] # the index of scores by desc
    
    keep = []
    while order.size() > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
            
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    
    return keep
        
        


