#!/usr/bin/env python

from __future__ import print_function, division
import os
import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as data

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from skimage import io, transform
from random import shuffle


cuda = torch.cuda.is_available() 
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class SIH(Dataset):
    def __init__(self, root_file, transform=None):
		idx = 0
		file_img = open(root_file, 'r')
		self.img_anno = {}
		for line in file_img:
			self.img_anno[idx] = line[0:-1]
			idx = idx + 1
		shuffle(self.img_anno)
		#print(self.img_anno)
		sih_csv = pd.read_csv('Dataset_1/Dataset_1.csv',skiprows=1,nrows=50)
		sih_csv = np.array(sih_csv,dtype=np.str)
		self.sih_target = []
		for i in range(len(sih_csv)):
			self.sih_target.append(sih_csv[i][1:])
			#print(sih_csv[i][1:])
		self.sih_target = np.array(self.sih_target,dtype=np.float32)
		#print(self.sih_target[5])
		self.transform = transform

    def __getitem__(self, index):
        tar_num = self.img_anno[index]
        tar_num = int(tar_num[17:19])-1
        measure = self.sih_target[tar_num]
        #print(tar_num,self.sih_target[tar_num])
        _img_temp = cv2.imread(self.img_anno[index] + '.jpeg')
        #print(self.img_anno[index] + '.jpeg')
        _img_temp = cv2.resize(_img_temp,(600,600))

        _img = torch.from_numpy(np.array(_img_temp).transpose(2, 0, 1)).float() 
        _target = torch.from_numpy(np.array(measure))

        return _img,_target

    def __len__(self):
        return len(self.img_anno)

THREADS = 1
USE_CUDA = False
batch_size = 1

sih_dataset = SIH(root_file="sih_train.txt")
train_loader = data.DataLoader(sih_dataset, batch_size,num_workers = THREADS,pin_memory= USE_CUDA)

for batch_i, (imgs, targets) in enumerate(train_loader):
    imgs = Variable(imgs.type(Tensor))
    targets = Variable(targets.type(Tensor), requires_grad=False)

