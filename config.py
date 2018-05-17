#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:37:39 2018

@author: wujiyang
"""

import os


MODLE_STORE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/MTCNN_TRAIN/model_store"

ANNO_STORE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/MTCNN_TRAIN/anno_store"

TRAIN_DATA_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/MTCNN_TRAIN/training_data"

LOG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/MTCNN_TRAIN/log"

USE_CUDA = True

TRAIN_BATCH_SIZE = 1024

TRAIN_LR = 0.01

END_EPOCH = 50

PNET_POSTIVE_ANNO_FILENAME = "pos_12.txt"
PNET_NEGATIVE_ANNO_FILENAME = "neg_12.txt"
PNET_PART_ANNO_FILENAME = "part_12.txt"
PNET_LANDMARK_ANNO_FILENAME = "landmark_12.txt"


RNET_POSTIVE_ANNO_FILENAME = "pos_24.txt"
RNET_NEGATIVE_ANNO_FILENAME = "neg_24.txt"
RNET_PART_ANNO_FILENAME = "part_24.txt"
RNET_LANDMARK_ANNO_FILENAME = "landmark_24.txt"


ONET_POSTIVE_ANNO_FILENAME = "pos_48.txt"
ONET_NEGATIVE_ANNO_FILENAME = "neg_48.txt"
ONET_PART_ANNO_FILENAME = "part_48.txt"
ONET_LANDMARK_ANNO_FILENAME = "landmark_48.txt"

PNET_TRAIN_IMGLIST_FILENAME = "imglist_anno_12.txt"
RNET_TRAIN_IMGLIST_FILENAME = "imglist_anno_24.txt"
ONET_TRAIN_IMGLIST_FILENAME = "imglist_anno_48.txt"