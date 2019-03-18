#!/usr/bin/env python
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import cv2
import argparse
import sys
from models import *
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error

dirname = os.path.dirname(__file__)

device = torch.device('cpu')
cuda = torch.cuda.is_available()
#print(torch.cuda.is_available())
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class SIH(Dataset):
	def __init__(self, root_file, transform=None):
		idx=0
		self.transform = transform
	def __getitem__(self, index):

		self.img_anno=sys.argv[1]
		_img_temp = cv2.imread(self.img_anno)
		#print(self.img_anno[index])
		#print(self.img_anno[index] + '.jpeg')
		_img_temp = cv2.resize(_img_temp,(600,600))

		_img = torch.from_numpy(np.array(_img_temp).transpose(2, 0, 1)).float() 
		#_target = torch.from_numpy(np.array(measure))

		return _img,0

	def __len__(self):
		return (1)

THREADS = 1
USE_CUDA = False
batch_size = 1
sih_dataset = SIH(root_file="sih_train.txt")
train_loader = data.DataLoader(sih_dataset, batch_size,num_workers = THREADS,pin_memory= USE_CUDA)

# Model
model=ResNet18()
model=model.to(device)

num_epochs = 50
learning_rate = 0.0001

criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

def cal_accuracy(targets,outputs):
	accuracy_value=[]
	for i in range(len(targets)):
		accuracy_value.append(((targets[i]-outputs[i])/targets[i])*100)
	return accuracy_value

# Testing
for j in range(0,1):
	model.load_state_dict(torch.load(os.path.join(dirname, 'checkpoint/%s_resnet.pt'%(j))))
	model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
	#print("\n")
	#print("models round")
	with torch.no_grad():
		correct = 0
		total = 0
		for batch_i, (imgs, targets) in enumerate(train_loader):
			imgs = imgs.to(device)
			targets = targets.to(device)
			#print(imgs)
			outputs = model(imgs)

			outputs2=np.array(outputs)
			thisdict={"shirtHeight":str(outputs2[0][0]),"shoulder":str(outputs2[0][1]),"hand":str(outputs2[0][2]),"collar":str(outputs2[0][3]),
			"chest":str(outputs2[0][4]),"stomach":str(outputs2[0][5]),"hip":str(outputs2[0][6])}
			#print("targets")
			#print(targets)
			print(json.dumps(thisdict))
			#targets,outputs = targets.cpu().numpy(),outputs.cpu().numpy()
			#accuracy = cal_accuracy(targets,outputs)
			#print(mean_absolute_error(targets,outputs))
			#print(accuracy)
		    
