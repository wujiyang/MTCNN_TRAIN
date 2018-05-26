#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 12:21:58 2018

@author: wujiyang
"""


import sys
sys.path.append("/home/wujiyang/FaceProjects/MTCNN_TRAIN")

import os
import config
import prepare_data.assemble as assemble


if __name__ == '__main__':

    anno_list = []

    net_landmark_file = os.path.join(config.ANNO_STORE_DIR,config.ONET_LANDMARK_ANNO_FILENAME)
    net_postive_file = os.path.join(config.ANNO_STORE_DIR,config.ONET_POSTIVE_ANNO_FILENAME)
    net_part_file = os.path.join(config.ANNO_STORE_DIR,config.ONET_PART_ANNO_FILENAME)
    net_neg_file = os.path.join(config.ANNO_STORE_DIR,config.ONET_NEGATIVE_ANNO_FILENAME)

    anno_list.append(net_postive_file)
    anno_list.append(net_part_file)
    anno_list.append(net_neg_file)
    anno_list.append(net_landmark_file)

    imglist_file = os.path.join(config.ANNO_STORE_DIR, config.ONET_TRAIN_IMGLIST_FILENAME)

    chose_count = assemble.assemble_data(imglist_file ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_file)
