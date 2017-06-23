"""
This scripts demos how to do single video classification using the framework
Before using this scripts, please download the model files using

bash models/get_reference_models.sh

Usage:

python classify_video.py <video name>
"""

import os
anet_home = os.environ['ANET_HOME']
import sys
sys.path.append(anet_home)

from pyActionRec.action_classifier import ActionClassifier
from pyActionRec.anet_db import ANetDB
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("--use_flow", action="store_true", default=False)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

USE_FLOW = args.use_flow
GPU=args.gpu

models=[]

models = [('models/resnet200_anet_2016_deploy.prototxt',
           'models/resnet200_anet_2016.caffemodel',
           1.0, 0, True, 224)]


if USE_FLOW:
    models.append(('models/bn_inception_anet_2016_temporal_deploy.prototxt',
                   'models/bn_inception_anet_2016_temporal.caffemodel.v5',
                   0.2, 1, False, 224))

cls = ActionClassifier(models, dev_id=GPU)

process_list = {}
counter = 0
for act in os.listdir(args.data_path):
    print 'Processing videos in directory: ', act
    act_path = os.path.join(args.data_path, act)
    for vid in os.listdir(act_path):
        if vid not in process_list:
            if vid[-4:] == '.mp4' or vid[-4:] == '.mkv' or vid[-4:] == 'webm':
                print 'Processing video: ', vid
                vid_path = os.path.join(act_path, vid)
                rst = cls.classify(vid_path)
                counter += 1
                process_list[vid] = counter
                print 'NO. ', counter
