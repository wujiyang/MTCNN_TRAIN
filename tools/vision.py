#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:14:18 2018

@author: wujiyang
"""
import matplotlib.pyplot as plt

def vis_two(im_array, dest1, dest2, thresh=0.9):
    """Visualize detection results before and after calibration
    
    Parameters:
    ----------
    im_array: numpy.ndarray, shape(1, c, h, w)
        test image in rgb
    dets1: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results before calibration
    dets2: numpy.ndarray([[x1 y1 x2 y2 score]])
        detection results after calibration
    thresh: float
        boxes with scores > thresh will be drawn in red 

    Returns:
    -------
    """

    figure = plt.figure()
    plt.subplot(121)
    plt.imshow(im_array)
    for i in range(dest1.shape[0]):
        bbox = dest1[i, 0:4]
        score = dest1[i, 4]
        landmarks = dest1[i, 5:]
        if score > thresh:
            rect = plt.Rectangle((bbox[0], bbox[1]), 
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor='red', linewidth=0.7)
            plt.gca().add_patch(rect) # get current Axes and do some modification on it
            landmarks = landmarks.reshape((5, 2))
            for j in range(5):
                plt.scatter(landmarks[j, 0], landmarks[j, 1], c='yellow', linewidth=1, marker='x', s = 20)
                
    plt.subplot(122)
    plt.imshow(im_array)
    for i in range(dest2.shape[0]):
        bbox = dest2[i, 0:4]
        score = dest2[i, 4]
        landmarks = dest2[i, 5:]
        if score > thresh:
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor='red', linewidth=0.7)
            plt.gca().add_patch(rect)

            landmarks = landmarks.reshape((5, 2))
            for j in range(5):
                plt.scatter(landmarks[j, 0], landmarks[j, 1], c='yellow', linewidths=1, marker='x', s=20)
    
    plt.show()
    
    

def vis_face(im_array, dets, landmarks=None):
    """Visualize detection results of an image

    Parameters:
    ----------
    im_array: numpy.ndarray, shape(1, c, h, w)
        test image in rgb
    dets: numpy.ndarray([[x1 y1 x2 y2 score landmarks]])
        detection results before calibration
    landmarks: numpy.ndarray([landmarks for five facial landmarks])

    Returns:
    -------
    """
    figure = plt.figure()
    plt.show(im_array)
    plt.set_title('Face Detector', fontsize=12, color='r')
    
    for i in range(dets.shape[0]):
        bbox = dets[i, 0:4]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='yellow', linewidth=0.9)
        plt.gca().add_patch(rect)
        
    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):
                plt.scatter(landmarks[j, 0], landmarks[j, 1], c='red',linewidths=0.1, marker='x', s=5)
    
    plt.show()
       