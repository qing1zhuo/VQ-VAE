

import os
import os.path as path
import json
import torch
import torch.utils.data as data
import numpy as np
import random
from PIL import Image
import pdb
import csv

        
# set the paths of the csv files
train_csv = os.path.join('./', 'train.csv')
val_csv = os.path.join('./', 'val.csv')
test_csv = os.path.join('./', 'test.csv')


pdb.set_trace()
# construct the positive and negative pairs
class_img_list={}

with open(train_csv) as f_csv:
    f_train = csv.reader(f_csv, delimiter=',')
    for row in f_train:
        if f_train.line_num == 1:
            continue
        img_name, img_class = row
        if class_img_list.__contains__(img_class):
            class_img_list[img_class].append(img_name)
        else:
            class_img_list[img_class]=[]
            class_img_list[img_class].append(img_name)