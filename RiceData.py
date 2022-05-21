#!/usr/bin/python3.8


import numpy as np
import torch as tc
import torchvision as tcv
import matplotlib.pyplot as plt

from torch import functional as F
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader

import sys
import os

import warnings
warnings.filterwarnings("ignore")

#==========================================================================================


class RiceDataset(Dataset):
        
        def __init__(self,transform=None):
                
                self.root_dir = "Rice_Image_Dataset/"
                self.root_rec = "Rice_Roted_Dataset/"
                 
                self.length = 0
                self.lookup = []
                
                for t in ["Arborio", "basmati", "Ipsala", "Jasmine", "Karacadag"]:
                    for x in os.listdir("%s/%s/" % (self.root_dir, t) ):
                            self.lookup += ["%s/%s" % (t,x)]
                            self.length += 1
                        
                self.lookup = np.array(self.lookup)
                
                self.transform = transform
                
        def __len__(self):
                
                return self.length                

        def __getitem__(self,idx):
        
                if tc.is_tensor(idx):
                        idx = idx.tolist()

                file_name = os.path.join(self.root_dir, self.lookup[idx])
                file_recs = os.path.join(self.root_rec, self.lookup[idx])
                
                x = tcv.io.read_image(file_name).float()/255.
                y = tcv.io.read_image(file_recs).float()/255.
                
                if self.transform:
                        x = self.transform(x)
                        y = self.transform(y)
                        
                return x,y
                
                
                
                
                
                
                
                
                
