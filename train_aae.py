# code adapted from paper https://arxiv.org/pdf/1511.05644.pdf
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae
# 

import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import ptvsd
from hdbscan import HDBSCAN

from visualization.validate_clustering import validate_clustering
from utils.data_utils import get_datasets

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=4, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=300, help="interval between image sampling")
parser.add_argument("--layer_size", type=int, default=2048, help="interval between image sampling")
parser.add_argument("--debugger", "-d", 
    dest="debug", 
    action="store_true",
    help="run the program with debugger server running")

opt = parser.parse_args()
print(opt)


if opt.debug:
    try:
        ptvsd.enable_attach()
        ptvsd.wait_for_attach()
    except OSError as exc:
        print(exc)

os.makedirs("output/aae", exist_ok=True)
os.makedirs("output/aae/samples", exist_ok=True)
os.makedirs("output/aae/reconstruction", exist_ok=True)
os.makedirs("output/aae/clustering", exist_ok=True)
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

device = torch.device(0 if torch.cuda.is_available() else "cpu")

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


    
class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=opt.channels, out_channels=20,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(in_channels=20, out_channels=20,kernel_size=3, stride=2, padding=1),
            #nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=40,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(in_channels=20, out_channels=40,kernel_size=3, stride=2, padding=1),
            #nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=60,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(in_channels=60, out_channels=60,kernel_size=3, stride=2, padding=1),
            #nn.ReLU(),
            nn.Flatten()
        )
        self.middle_layer = nn.Sequential(nn.Linear(in_features=opt.img_size*opt.img_size*60, out_features =256), nn.ReLU())

        self.mu = nn.Linear(in_features=256, out_features=opt.latent_dim)


    def forward(self, img):
        #img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        return mu


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(out_features=256, in_features=opt.latent_dim),
            nn.ReLU(),
            nn.Linear(out_features=opt.img_size * opt.img_size *60, in_features=256),
            Reshape((60, opt.img_size, opt.img_size))
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
            nn.ConvTranspose2d(out_channels=opt.channels, in_channels=20,kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.fc(z)
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, opt.layer_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.layer_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

(dataloader_train, dataloader_test, 
dataset_train, dataset_test, 
echograms_train, echograms_test) = get_datasets(
    frequencies=[18, 38, 120, 200], 
    iterations=3000, 
    num_workers=0, 
    window_dim=opt.img_size,
    include_depthmap=False)


_, (inputs_test, labels_test, si_test) = next(enumerate(dataloader_test))
inputs_test = inputs_test.float().to(device)
labels_test = labels_test.long().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    save_image(gen_imgs[:, 0].reshape(-1, 1, 32, 32).data, "output/aae/samples/%d.png" % batches_done, nrow=10, normalize=True)

def save_recon(original_image, reconstruction, batches_done, nrow):
    save_image(reconstruction[:, 0].reshape(-1, 1, 32, 32).data, "output/aae/reconstruction/%d.png" % batches_done, nrow=10, normalize=True)
# ----------
#  Training
# ----------

class AAE:
    def __init__(self):
        
        
        self.decoder = decoder
        self.discriminator = discriminator
    def encoder(self, x):
        z = encoder(x)
        return z, z

#validate_clustering(AAE(), HDBSCAN(prediction_data=True), inputs_test, si_test, dataset_test.samplers, device, opt.latent_dim, 0, fig_path="output/aae/clustering/hdbscan.png")
for i, (imgs, imgs_train, si) in enumerate(dataloader_train):
    # Adversarial ground truths
    valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_imgs = Variable(imgs.type(Tensor))

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    encoded_imgs = encoder(real_imgs)
    decoded_imgs = decoder(encoded_imgs)

    # Loss measures generator's ability to fool the discriminator
    g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
        decoded_imgs, real_imgs
    )

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(z), valid)
    fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
    d_loss = 0.5 * (real_loss + fake_loss)

    d_loss.backward()
    optimizer_D.step()

    print(
        "[Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (i, len(dataloader_train), d_loss.item(), g_loss.item())
    )

    batches_done = i

    #sample_image(n_row=10, batches_done=batches_done)
    if batches_done % opt.sample_interval == 0:
        sample_image(n_row=10, batches_done=batches_done)
        save_recon(real_imgs, decoded_imgs, batches_done=batches_done, nrow=10)

validate_clustering(AAE(), HDBSCAN(prediction_data=True), inputs_test, si_test, dataset_test.samplers, device, opt.latent_dim, 0, fig_path="output/aae/clustering/hdbscan.png")

"""
X * X * X * *
* *       *     
X   X   X     
*     *       
X   X   X    
* *       * 
*  

*---*---*---X
| \       /     
*   *   *     
|     X       
*   *   *    
| /       \
X           \
"""