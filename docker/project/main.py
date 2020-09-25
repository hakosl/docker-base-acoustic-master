
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

import acoustic_private.data.echogram as echogram

tensorcrop = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((352, 352))
])

class AcousticDataset(Dataset):
    def __init__(self, root_folder, files, transform, shape=[256, 256]):

        self.frequencies = [18, 38, 120, 200]
        self.window_size = shape
        self.shape = shape
        self.sonar_data = []
        self.data_lengths = []
        self.dataset_length = 0
        self.transform = transform

        echograms = echogram.get_echograms(frequencies=self.frequencies, minimum_shape= self.shape)

        for eg in echograms:
            eg_np = eg.data_numpy()
            print(eg_np.shape)
            self.sonar_data.append(eg_np)
            self.data_lengths.append(eg_np.shape[1])

            self.dataset_length += eg_np.shape[1] % shape[1]

        self.data = []

        for record in self.sonar_data:
            i = 0

            while (i + 1) * self.shape[1] < record.shape[1]:
                d1, d2 = self.shape
                self.data.append(F.interpolate(record[:d1, d2 * (i):d2*(i + 1), :4], (352, 352)))
                i += 1

        self.data = torch.Tensor(self.data)
        self.data = self.data.view(-1, 5, self.shape[0], self.shape[1])

        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 2-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)
latent_dims = 2
num_epochs = 100
batch_size = 100
capacity = 64
learning_rate = 1e-3
variational_beta = 1
use_gpu = True

# # 10-d latent space, for comparison with non-variational auto-encoder
# latent_dims = 10
# num_epochs = 100
# batch_size = 128
# capacity = 64
# learning_rate = 1e-3
# variational_beta = 1
# use_gpu = True


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(in_features=c*2*22*22, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*22*22, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*22*22)
        self.conv4 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=5, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity*2, 22, 22) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
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
    
def vae_loss(recon_x, x, mu, logvar):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.

    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 5 * 352 * 352), x.view(-1, 5 * 352*352), reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence
    
    
vae = VariationalAutoencoder()

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print(device)
vae = vae.to(device)

num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

# set to training mode
vae.train()

train_loss_avg = []
#mnist = datasets.MNIST("/acoustic", download=True,transform=transforms.ToTensor())

files = os.listdir("/memmap")[:100]

eg_dataset = AcousticDataset("/memmap", files, tensorcrop)

train_dataloader = DataLoader(eg_dataset, batch_size=batch_size, shuffle=True)
print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    
    for image_batch in train_dataloader:
        
        image_batch = image_batch.to(device)
        # vae reconstruction
        image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
        
        # reconstruction error
        loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        
        train_loss_avg[-1] += loss.item()
        num_batches += 1
        
    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))


#path = "/memmap/2013842-D20130501-T130913"

#eg = echogram.Echogram(path)
#print(eg.data_numpy()[:, :352].shape)
