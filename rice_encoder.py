#!/usr/bin/python3.8

import numpy as np
import torch as tc
import torchvision as tcv
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
import torchvision.transforms.functional as vF
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


import sys
import os

import warnings
warnings.filterwarnings("ignore")

from torchvision.transforms import Compose, ToTensor
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

import umap
import umap.plot

#===============================================================

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, recompute_scale_factor=False, mode=self.mode, align_corners=False)
        return x
        
#===============================================================

class Autoencoder(nn.Module):
        def __init__(self,n_latent=10):
                super(Autoencoder,self).__init__()
                
                n_features = 50 * 50
                n_latent   = n_latent
                
                self.enc1 = nn.Sequential(
                                nn.Conv2d(3,10,11),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.3)
                                )
                self.enc2 = nn.Sequential(
                                nn.Conv2d(10,20,11),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.3)
                                )
                self.enc3 = nn.Sequential(
                                nn.Linear(20*30*30,n_latent),
                                )
                                
                self.dec3 = nn.Sequential(
                                nn.Linear(n_latent,20*30*30)
                                )
                self.dec2 = nn.Sequential(
                                Interpolate(50,'bilinear'),
                                nn.Conv2d(20,10,11),
                                nn.LeakyReLU(0.2)
                                )
                self.dec1 = nn.Sequential(
                                Interpolate(60,'bilinear'),
                                nn.Conv2d(10,3,11),
                                nn.LeakyReLU(0.2)
                                )
        
        def forward(self,x):
                x = self.enc1(x)        # 40
                x = self.enc2(x)        # 30
                x = x.view(-1,20*30*30)    
                x = self.enc3(x)        # z
                y = tc.clone(x)
                x = self.dec3(x)
                x = x.view(-1,20,30,30) # 30
                x = self.dec2(x)        # 40
                x = self.dec1(x)        # 50
                
                return x,y
                
#==========================================================================

def train_autoencoder(autoencoder,optimizer,loss,batch):
        
        optimizer.zero_grad()
        
        fake   = autoencoder(batch)
        error  = loss(fake,batch)
        
        error.backward()
        
        optimizer.step()
        
        return error

def rice_data():

    compose = Compose([ ToTensor(), Resize(50) ])
    return ImageFolder("Rice_Image_Dataset", transform = compose )       

#==========================================================================

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = vF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def plot_batch(N=10):
        
        K = 90
        
        examples = enumerate(DataLoader(rice_data(),batch_size=N,shuffle=True))
        batch_idx, example = next(examples)
        
        grid = []
        
        for i in range(N):
            grid += [ example[0][i] ]
            
        grid = make_grid(grid)
        show(grid)
        
def plot_recon(ae, N=10):
        
        K = 90
        
        examples = enumerate(DataLoader(rice_data(),batch_size=N,shuffle=True))
        batch_idx, example = next(examples)
        
        grid = []
        grid_= []
        
        with tc.no_grad():
            output = ae(example[0])
        
        for i in range(N):
            grid += [ example[0][i] ]
            grid_+= [ output[i] ]
        
        grid = make_grid(grid)
        show(grid)
        grid_= make_grid(grid_)
        show(grid_)

#==========================================================================

data = rice_data()
dataloader = DataLoader(data,batch_size=3000,shuffle=True)

batch_idx, sample = next(enumerate(dataloader))

##################################### SIMPLE ENCODER
# this encoder has been trained on the original data, where
# all rice is rotated by a random angle

autoencoder = Autoencoder(2)
autoencoder.load_state_dict(tc.load('autoencoder.pt'))

with tc.no_grad():
    rec, lat = autoencoder(sample[0])
    
    lat = lat.detach().numpy()
    labs= sample[1].detach().numpy()

print(lat.shape)
print(labs.shape)

x = np.zeros((3,lat.shape[0]))

x[0] = lat[:,0]
x[1] = lat[:,1]
x[2] = labs
print(x.shape)

plt.scatter(x[0],x[1],c=x[2],cmap='Spectral')
plt.show()

r = np.sqrt(x[0]*x[0] + x[1]*x[1])
t = np.arccos(x[0]/r)

plt.scatter(r,t,c=x[2],cmap='Spectral')
plt.show()

##################################### ROTATIONAL ENCODER
# this encoder has been trained to reproduce a dataset
# where the original image has been rotated so the major
# axis is tilted by a 45° angle. The autoencoder starts
# from a rice seed rotated by a random angle and tries to
# reproduce the corresponding 45° oriented seed.

autoencoder = Autoencoder(2)
autoencoder.load_state_dict(tc.load('autoencoder_.pt'))

with tc.no_grad():
    rec, lat = autoencoder(sample[0])
    
    lat = lat.detach().numpy()
    labs= sample[1].detach().numpy()

print(lat.shape)
print(labs.shape)

x = np.zeros((3,lat.shape[0]))

x[0] = lat[:,0]
x[1] = lat[:,1]
x[2] = labs
print(x.shape)

plt.scatter(x[0],x[1],c=x[2],cmap='Spectral')
plt.show()

r = np.sqrt(x[0]*x[0] + x[1]*x[1])
t = np.arccos(x[0]/r)

plt.scatter(r,t,c=x[2],cmap='Spectral')
plt.show()

