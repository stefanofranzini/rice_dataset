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

import sys
import os

import warnings

warnings.filterwarnings("ignore")

from RiceData import RiceDataset

from torchvision.transforms import Compose, ToTensor
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

# ===============================================================

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(
            x,
            size=self.size,
            recompute_scale_factor=False,
            mode=self.mode,
            align_corners=False,
        )
        return x


# ===============================================================


class Autoencoder(nn.Module):
    def __init__(self, n_latent=10):
        super(Autoencoder, self).__init__()

        n_features = 50 * 50
        n_latent = n_latent

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 10, 11), nn.LeakyReLU(0.2), nn.Dropout2d(0.3)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(10, 20, 11), nn.LeakyReLU(0.2), nn.Dropout2d(0.3)
        )
        self.enc3 = nn.Sequential(
            nn.Linear(20 * 30 * 30, n_latent),
        )

        self.dec3 = nn.Sequential(nn.Linear(n_latent, 20 * 30 * 30))
        self.dec2 = nn.Sequential(
            Interpolate(50, "bilinear"), nn.Conv2d(20, 10, 11), nn.LeakyReLU(0.2)
        )
        self.dec1 = nn.Sequential(
            Interpolate(60, "bilinear"), nn.Conv2d(10, 3, 11), nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.enc1(x)  # 40
        x = self.enc2(x)  # 30
        x = x.view(-1, 20 * 30 * 30)
        x = self.enc3(x)  # z
        x = self.dec3(x)
        x = x.view(-1, 20, 30, 30)  # 30
        x = self.dec2(x)  # 40
        x = self.dec1(x)  # 50

        return x


# ==========================================================================


def train_autoencoder(autoencoder, optimizer, loss, batch, batch_):

    optimizer.zero_grad()

    fake = autoencoder(batch)
    error = loss(fake, batch_)

    error.backward()

    optimizer.step()

    return error


def rice_data():

    compose = Compose([Resize(50)])
    return RiceDataset(transform=compose)


# ==========================================================================


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

    examples = enumerate(DataLoader(rice_data(), batch_size=N, shuffle=True))
    batch_idx, example = next(examples)

    grid = []

    for i in range(N):
        grid += [example[0][i]]
    for i in range(N):
        grid += [example[1][i]]

    grid = make_grid(grid)
    show(grid)


def plot_recon(ae, N=10):

    examples = enumerate(DataLoader(rice_data(), batch_size=N, shuffle=True))
    batch_idx, example = next(examples)

    grid = []

    with tc.no_grad():
        output = ae(example[0])

    for i in range(N):
        grid += [example[1][i]]
    for i in range(N):
        grid += [output[i]]

    grid = make_grid(grid)
    show(grid)


# ==========================================================================

# 0) load the data and show a small batch

data = rice_data()
dataloader = DataLoader(data, batch_size=128, shuffle=True)

plot_batch(32)
plt.show()

# ---------------------------------------------------------------------------

# 1) train a simple autoencoder with n_latent = 2

autoencoder = Autoencoder(2)

optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)

scheduler = tc.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

loss = nn.MSELoss()

for epoch in range(5):

    for n_batch, real_batch in enumerate(dataloader):

        error = train_autoencoder(
            autoencoder, optimizer, loss, real_batch[0], real_batch[0]
        )

        print(epoch, n_batch, error.detach().numpy())

        loss_file = open("loss_.txt", "a")
        loss_file.write("%d\t%d\t%f\n" % (epoch, n_batch, error.detach().numpy()))
        loss_file.close()

        if n_batch % 10 == 0:
            tc.save(autoencoder.state_dict(), "autoencoder.pt")

            plot_recon(autoencoder, 32)
            plt.savefig("last_plot.png", transparent=True)

    scheduler.step()

tc.save(autoencoder.state_dict(), "autoencoder.pt")
loss_file.close()

# ---------------------------------------------------------------------------

# 2) train an autoencoder with n_latent = 2 which rotates the rice seed to 45 degrees

autoencoder = Autoencoder(2)

optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)

scheduler = tc.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

loss = nn.MSELoss()

for epoch in range(5):

    for n_batch, real_batch in enumerate(dataloader):

        error = train_autoencoder(
            autoencoder, optimizer, loss, real_batch[0], real_batch[1]
        )

        print(epoch, n_batch, error.detach().numpy())

        loss_file = open("loss_.txt", "a")
        loss_file.write("%d\t%d\t%f\n" % (epoch, n_batch, error.detach().numpy()))
        loss_file.close()

        if n_batch % 10 == 0:
            tc.save(autoencoder.state_dict(), "autoencoder_.pt")

            plot_recon(autoencoder, 32)
            plt.savefig("last_plot_.png", transparent=True)

    scheduler.step()

tc.save(autoencoder.state_dict(), "autoencoder_.pt")
loss_file.close()
