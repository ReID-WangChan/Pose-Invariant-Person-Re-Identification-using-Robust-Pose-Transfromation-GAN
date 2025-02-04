# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:42:00 2019

@author: VR LAB PC3
"""

from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

# Other options
__C.NET = 'GAN'
__C.GPU_ID = '0'
__C.NUM_CLASS = 751 
__C.DEBUG = False
__C.FILE_PATH = './'   # working directory

# Train options
__C.TRAIN = edict()
# __C.TRAIN.imgs_path = 'Datasets/Market-1501-v15.09.15/bounding_box_train'
__C.TRAIN.imgs_path = 'Datasets/DukeMTMC-reID/bounding_box_train'
# __C.TRAIN.pose_path = 'Datasets/Market-1501-v15.09.15/openpose_train'
__C.TRAIN.pose_path = 'Datasets/DukeMTMC-reID/openpose_train'
# __C.TRAIN.idx_path = 'Datasets/Market-1501-v15.09.15/train_idx.txt'
__C.TRAIN.idx_path = 'Datasets/DukeMTMC-reID/train_idx.txt'
__C.TRAIN.LR = 0.0002
__C.TRAIN.LR_DECAY = 10
__C.TRAIN.MAX_EPOCH = 20

__C.TRAIN.DISPLAY = 100
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.NUM_WORKERS = 0

# GAN options
__C.TRAIN.ngf = 64
__C.TRAIN.ndf = 64
__C.TRAIN.num_resblock = 9
__C.TRAIN.lambda_idt = 10
__C.TRAIN.lambda_att = 1


# Test options
__C.TEST = edict()
# __C.TEST.imgs_path = 'Datasets/Market-1501-v15.09.15/bounding_box_test'
__C.TEST.imgs_path = 'Datasets/DukeMTMC-reID/bounding_box_test'
# __C.TEST.pose_path = 'Datasets/Market-1501-v15.09.15/openpose_test'
# __C.TEST.pose_path = 'Datasets/Market-1501-v15.09.15/12_gmm_pose'
__C.TEST.pose_path = 'Datasets/DukeMTMC-reID/openpose_test'
# __C.TEST.idx_path = 'Datasets/Market-1501-v15.09.15/test_idx.txt'
__C.TEST.idx_path = 'Datasets/DukeMTMC-reID/test_idx.txt'
__C.TEST.BATCH_SIZE = 1
__C.TEST.GPU_ID = '0'
