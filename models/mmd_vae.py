
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

import matplotlib.pyplot as plt

from skimage import io, transform
import numpy as np


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

class Encoder(nn.Module):
    def __init__(self, capacity, window_dim=64, hdim=None, stride=2, kernel_size=4, extra_layer=False, depth_input=False):

        super(Encoder, self).__init__()
        self.hdim = hdim
        self.capacity = capacity
        self.window_dim = window_dim
        self.extra_layer = extra_layer
        self.convolutions = [
            nn.Sequential(
                nn.Conv2d(in_channels=self.hdim[i], out_channels=self.hdim[i + 1], kernel_size=kernel_size, stride=stride, padding=1),

                nn.ReLU()
            ) 
            for i in range(len(self.hdim) - 1)
        ]
        self.latent_in_dim = (int(window_dim/((len(self.hdim) - 1)**2)) ** 2)*self.hdim[-1]

        

        
        
        
        if extra_layer:
            middle_fc_dim = 512
            self.final_layer = nn.Sequential(nn.Linear(in_features=self.latent_in_dim, out_features = middle_fc_dim), nn.ReLU(), nn.Linear(in_features=middle_fc_dim, out_features=self.capacity))
        else:
            self.final_layer = nn.Sequential(nn.Linear(in_features=self.latent_in_dim, out_features=self.capacity))

        self.encoder = nn.Sequential(*self.convolutions, Flatten(), self.final_layer)

        #self.fc_logvar = nn.Linear(in_features=self.latent_in_dim, out_features=self.capacity)
            
    def forward(self, x):
        z = self.encoder(x)
        return z, z

class Decoder(nn.Module):
    def __init__(self, capacity, recon_loss="BCE", window_dim=64, hdim=None, stride=2, kernel_size=4, extra_layer=False, depth_input=False):
        super(Decoder, self).__init__()

        self.window_dim=window_dim
        self.hdim = hdim
        self.reverse_hdim = list(reversed(self.hdim))
        self.recon_loss = recon_loss
        self.capacity = capacity
        self.extra_layer = extra_layer

        latent_out_dim = (int(window_dim/((len(self.hdim) - 1)**2)) ** 2)*self.hdim[-1]
        self.latent_window = int(window_dim/((len(self.hdim) - 1)**2))
        middle_fc_dim = 512
        
        if self.extra_layer:
            self.fc = nn.Sequential(nn.Linear(in_features=self.capacity, out_features=middle_fc_dim), nn.Linear(in_features=middle_fc_dim, out_features=latent_out_dim))
        else:
            self.fc = nn.Linear(in_features=self.capacity, out_features=latent_out_dim)
        self.convolutions = [
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.reverse_hdim[i], out_channels=self.reverse_hdim[i + 1], kernel_size=kernel_size, stride=stride, padding=1, output_padding=0),
                #nn.BatchNorm2d(self.reverse_hdim[i + 1]), 
                nn.ReLU()
            )
            for i in range(len(self.reverse_hdim) - 2)
        ]
        self.final_layer = nn.Sequential()  
        if recon_loss == "BCE":
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.reverse_hdim[-2], out_channels=self.reverse_hdim[-1], kernel_size=kernel_size, stride=stride, padding=1, output_padding=0),
                #nn.BatchNorm2d(self.reverse_hdim[-1]),
                nn.Sigmoid()
            )
        elif recon_loss == "MSE":
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.reverse_hdim[-2], out_channels=self.reverse_hdim[-1], kernel_size=kernel_size, stride=stride, padding=1, output_padding=0),
                #nn.BatchNorm2d(self.reverse_hdim[-1]),
                nn.sigmoid()
            )
        self.decoder = nn.Sequential(*self.convolutions, self.final_layer)
        # self.conv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        # self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.reverse_hdim[0], self.latent_window, self.latent_window) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        
        x = self.decoder(x)


        return x
    
class InfoVAE(nn.Module):
    def __init__(self, capacity = 2, recon_loss="BCE", window_dim=64, channels=4, stride=2, kernel_size=4, extra_layer = False, **kwargs):
        super(InfoVAE, self).__init__()
        self.channels = channels
        self.capacity = capacity
        self.hdim = [self.channels, 32, 64, 128, 256, 512]
        self.encoder = Encoder(capacity = capacity, window_dim=window_dim, hdim=self.hdim, stride=stride, kernel_size=kernel_size, extra_layer=extra_layer)
        self.decoder = Decoder(capacity = capacity, recon_loss=recon_loss, window_dim=window_dim, hdim=self.hdim, stride=stride, kernel_size=kernel_size, extra_layer=extra_layer)
    
    def forward(self, x):
        z, _ = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, z
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()

            return eps.mul(std).add_(mu)
        else:
            return mu


    def compute_kernel(self, x, y):
        # x: dim X img_s 
        # y: dim X latent size
        x_size = x.size(1) 
        y_size = y.size(1)
        dim = x.size(0)
        first_term = -2 * x.T.matmul(y)
        second_term = (x.pow(2)).sum(axis=0).unsqueeze(0).T
        third_term = (y.pow(2)).sum(axis=0)
        kernel_input = (first_term + second_term + third_term)
        #print(kernel_input.shape)
        return torch.exp(-kernel_input/float(dim)) # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        #print(x_kernel.shape, y_kernel.shape, xy_kernel.shape)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

    def sample(self, num_samples: int, current_device: int):
        z = torch.randn(num_samples, self.capacity)
        z = z.to(current_device)
        return self.decoder(z)

    def representation(self, x):
        z = self.encoder(x)

        return z
    
    def loss(self, recon_x, x , z, _, variational_beta, recon_loss="MSE", window_dim=64, channels=4):
        # recon_x is the probability of a multivariate Bernoulli distribution p.
        # -log(p(x)) is then the pixel-wise binary cross-entropy.
        # Averaging or not averaging the binary cross-entropy over all pixels here
        # is a subtle detail with big effect on training, since it changes the weight
        # we need to pick for the other loss term by several orders of magnitude.
        # Not averaging is the direct implementation of the negative log likelihood,
        # but averaging makes the weight of the other loss term independent of the image resolution.


        if recon_loss == "BCE":
            recon_loss = F.binary_cross_entropy(recon_x.view(-1, channels * window_dim ** 2), x.view(-1, channels * window_dim ** 2), reduction="mean")
        elif recon_loss == "MSE":
            recon_loss = F.mse_loss(recon_x.view(-1, channels * window_dim ** 2), x.view(-1, channels * window_dim ** 2), reduction="mean")
        
        mmd = self.compute_mmd(x.reshape(-1, window_dim * channels *  window_dim), z)
        nll = recon_loss
        loss = nll + mmd
        return loss, nll, mmd


vae_models = {
    "info_vae": InfoVAE,
}