#!/usr/bin/env python
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import cv2
import argparse

from models import *
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

device = torch.device('cpu')
cuda = torch.cuda.is_available() 
print(torch.cuda.is_available())
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class SIH(Dataset):
	def __init__(self, root_file, transform=None):
		idx=0
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
batch_size = 2
sih_dataset = SIH(root_file="sih_train.txt")
train_loader = data.DataLoader(sih_dataset, batch_size,num_workers = THREADS,pin_memory= USE_CUDA)

# Model
model = ResNet18()
model = model.to(device)

num_epochs = 50
learning_rate = 0.0001

criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

PATH="checkpoint/27_resnet.pt"
#checkpoint=torch.load(PATH)
model.load_state_dict(torch.load(PATH))
# Training
train_loss,next_batch = 0,0
for epoch in range(27,num_epochs):
	next_batch,train_loss,valid_loss = 0,0,0
	for batch_i, (imgs, targets) in enumerate(train_loader):
		imgs = imgs.to(device)
		targets = targets.to(device)
		#print(targets,targets.size())
		outputs = model(imgs)
		#print(outputs.size(),targets.size())
		loss = criterion(outputs,targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		next_batch = next_batch+1
	print("epoch:",epoch,"train_loss:",train_loss)
	torch.save(model.state_dict(), 'checkpoint/%s_resnet.pt'%(epoch))
