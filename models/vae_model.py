
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

    
class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class Encoder(nn.Module):
    def __init__(self, capacity, window_dim=64, hdim=None, stride=2, kernel_size=4, extra_layer=False, depth_input=False, alt_model=False):

        super(Encoder, self).__init__()
        self.alt_model = alt_model
        self.hdim = hdim
        self.capacity = capacity
        self.window_dim = window_dim
        self.extra_layer = extra_layer
        self.convolutions = [
            nn.Sequential(
                nn.Conv2d(in_channels=self.hdim[i], out_channels=self.hdim[i + 1], kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm2d(self.hdim[i + 1]),
                nn.ReLU()
            ) 
            for i in range(len(self.hdim) - 1)
        ]
        self.latent_in_dim = (int(window_dim/((len(self.hdim) - 1)**2)) ** 2)*self.hdim[-1]

        middle_fc_dim = 512

        self.middle_layer = nn.Sequential(nn.Linear(in_features=self.latent_in_dim, out_features = middle_fc_dim), nn.ReLU())

        self.encoder = nn.Sequential(*self.convolutions)

        
        if extra_layer:
            self.latent_in_dim = middle_fc_dim

        

        self.fc_mu = nn.Linear(in_features=self.latent_in_dim, out_features=self.capacity)
        self.fc_logvar = nn.Linear(in_features=self.latent_in_dim, out_features=self.capacity)


        if self.alt_model:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=self.hdim[0], out_channels=20,kernel_size=3, stride=1, padding=1),
                nn.Dropout(p=0.2),
                nn.BatchNorm2d(20),
                nn.ReLU(),
                #nn.Conv2d(in_channels=20, out_channels=20,kernel_size=3, stride=2, padding=1),
                #nn.ReLU(),
                nn.Conv2d(in_channels=20, out_channels=40,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(40),
                nn.ReLU(),
                #nn.Conv2d(in_channels=20, out_channels=40,kernel_size=3, stride=2, padding=1),
                #nn.ReLU(),
                nn.Conv2d(in_channels=40, out_channels=60,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(60),
                nn.ReLU(),
                #nn.Conv2d(in_channels=60, out_channels=60,kernel_size=3, stride=2, padding=1),
                #nn.ReLU(),
            )
            self.middle_layer = nn.Sequential(nn.Linear(in_features=self.window_dim*self.window_dim*60, out_features = self.latent_in_dim), nn.BatchNorm1d(self.latent_in_dim), nn.ReLU())

            self.fc_mu = nn.Sequential(nn.Linear(in_features=self.latent_in_dim, out_features=self.capacity), nn.BatchNorm1d(self.capacity))
            self.fc_logvar = nn.Sequential(nn.Linear(in_features=self.latent_in_dim, out_features=self.capacity), nn.BatchNorm1d(self.capacity))
            
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        if self.extra_layer:
            x = self.middle_layer(x)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, capacity, recon_loss="BCE", window_dim=64, hdim=None, stride=2, kernel_size=4, extra_layer=False, depth_input=False, alt_model=False):
        super(Decoder, self).__init__()

        self.alt_model = alt_model
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
            self.fc = nn.Sequential(nn.Linear(in_features=self.capacity, out_features=middle_fc_dim), nn.ReLU(), nn.Linear(in_features=middle_fc_dim, out_features=latent_out_dim), nn.ReLU())
        else:
            self.fc = nn.Sequential(nn.Linear(in_features=self.capacity, out_features=latent_out_dim), nn.ReLU())
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
                nn.Tanh()
            )
        self.decoder = nn.Sequential(*self.convolutions, self.final_layer)

        if self.alt_model:
            self.fc = nn.Sequential(
                nn.Linear(out_features=256, in_features=self.capacity),
                nn.ReLU(),
                nn.Linear(out_features=self.window_dim * self.window_dim *60, in_features=256),
                Reshape((60, self.window_dim, self.window_dim))
            )
            self.decoder = nn.Sequential(
                nn.ReLU(),
                #nn.ConvTranspose2d(out_channels=60, in_channels=60,kernel_size=3, stride=2, padding=1, output_padding=1),
                #nn.ReLU(),
                nn.ConvTranspose2d(out_channels=40, in_channels=60,kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                #nn.ConvTranspose2d(out_channels=20, in_channels=40,kernel_size=3, stride=2, padding=1, output_padding=1),
                #nn.ReLU(),
                nn.ConvTranspose2d(out_channels=20, in_channels=40,kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                #nn.ConvTranspose2d(out_channels=20, in_channels=20,kernel_size=3, stride=2, padding=1),
                #nn.ReLU(),
                nn.ConvTranspose2d(out_channels=self.hdim[0], in_channels=20,kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        # self.conv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        # self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        # self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)

        
        x = self.decoder(x)


        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, capacity = 2, recon_loss="BCE", window_dim=64, channels=4, stride=2, kernel_size=3, extra_layer = False, alt_model=True, **kwargs):

        super(VariationalAutoencoder, self).__init__()
        self.channels = channels
        self.capacity = capacity
        self.hdim = [self.channels, 32, 64, 128, 256, 512]
        
        self.encoder = Encoder(capacity = capacity, window_dim=window_dim, hdim=self.hdim, stride=stride, kernel_size=kernel_size, extra_layer=extra_layer, alt_model=alt_model)
        self.decoder = Decoder(capacity = capacity, recon_loss=recon_loss, window_dim=window_dim, hdim=self.hdim, stride=stride, kernel_size=kernel_size, extra_layer=extra_layer, alt_model=alt_model)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()

            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def __str__(self):
        return f"VAE (cap= {self.capacity} )"

    def sample(self, num_samples: int, current_device: int):
        z = torch.randn(num_samples, self.capacity)
        z = z.to(current_device)
        return self.decoder(z)

    def representation(self, x):
        latent_mu, latent_logvar = self.encoder(x)

        return latent_mu
    
    def loss(self, recon_x, x, mu, logvar, variational_beta, recon_loss="BCE", window_dim=64, channels=4):
        # recon_x is the probability of a multivariate Bernoulli distribution p.
        # -log(p(x)) is then the pixel-wise binary cross-entropy.
        # Averaging or not averaging the binary cross-entropy over all pixels here
        # is a subtle detail with big effect on training, since it changes the weight
        # we need to pick for the other loss term by several orders of magnitude.
        # Not averaging is the direct implementation of the negative log likelihood,
        # but averaging makes the weight of the other loss term independent of the image resolution.
        if recon_loss == "BCE":
            recon_loss = F.binary_cross_entropy(recon_x.view(-1, channels * window_dim ** 2), x.view(-1, channels * window_dim ** 2), reduction="sum")
        elif recon_loss == "MSE":
            recon_loss = F.mse_loss(recon_x.view(-1, channels * window_dim ** 2), x.view(-1, channels * window_dim ** 2), reduction="sum")

        n_samples = x.shape[0]
        # KL-divergence between the prior distribution over latent vectors
        # (the one we are going to sample from when generating new images)
        # and the distribution estimated by the generator for the given image.
        kldivergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return recon_loss / n_samples + variational_beta * kldivergence, recon_loss, variational_beta * kldivergence

def datapVAE(*args, **kwargs):
    return nn.DataParallel(VariationalAutoencoder(*args, **kwargs))
    
vae_models = {
    "datap_vae": datapVAE,
    "vanilla_vae": VariationalAutoencoder
}