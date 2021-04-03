# code adapted from https://github.com/shaharazulay/adversarial-autoencoder-classifier
import argparse
import os
import numpy as np
import math
import itertools

from time import time
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
from itertools import cycle
import json
from visualization.validate_clustering import validate_clustering
from utils.data_utils import get_datasets
from torch.utils.tensorboard import SummaryWriter

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--n_categories", type=int, default=4, help="number of classes")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=4, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=300, help="interval between image sampling")
parser.add_argument("--layer_size", type=int, default=2048, help="interval between image sampling")
parser.add_argument("--device", type=int, default=1, help="the gpu device to use")
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
def main(latent_dim = 10):
    os.makedirs("output/semi_aae", exist_ok=True)
    os.makedirs("output/semi_aae/samples", exist_ok=True)
    os.makedirs("output/semi_aae/reconstruction", exist_ok=True)
    os.makedirs("output/semi_aae/clustering", exist_ok=True)
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    cuda = True if torch.cuda.is_available() else False

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    def reparameterization(mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
        z = sampled_z * std + mu
        return z

    encoder_model_path = "/acoustic/aae_semi_supervised.pt"

    class Reshape(nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.shape = args

        def forward(self, x):
            return x.view(self.shape)

    class Encoder(nn.Module):
        def __init__(self, img_shape, layer_size, n_categories, latent_dim):
            super(Encoder, self).__init__()

            
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=opt.channels, out_channels=20,kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.2)
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

            self.mu = nn.Linear(in_features=256, out_features=latent_dim)

            self.cat = nn.Sequential(
                nn.Linear(256, n_categories),
                nn.Softmax(dim=1)
            )

        def forward(self, img):
            x = self.model(img)
            x = self.middle_layer(x)
            z = self.mu(x)
            y = self.cat(x)
            return z, y

        def save(self, path):
            torch.save(self.state_dict(), path)

        def load(self, path):
            self.load_state_dict(torch.load(path))


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()

            self.fc = nn.Sequential(
                nn.Linear(out_features=256, in_features=opt.latent_dim + opt.n_categories),
                nn.ReLU(),
                nn.Linear(out_features=opt.img_size * opt.img_size *60, in_features=256),
                Reshape(-1, 60, opt.img_size, opt.img_size)
            )

            self.model = nn.Sequential(
                self.fc,
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
                nn.Tanh(),
            )

        def forward(self, z):
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

    class Discriminator_cat(nn.Module):
        def __init__(self):
            super(Discriminator_cat, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(opt.n_categories, opt.layer_size),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(opt.layer_size, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        def forward(self, z):
            return self.model(z)


    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    encoder = Encoder(img_shape, opt.layer_size, opt.n_categories, opt.latent_dim)
    decoder = Decoder()
    discriminator = Discriminator()
    discriminator_cat = Discriminator_cat()

    if cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        discriminator_cat.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    start_time = time()

    (dataloader_train, dataloader_test, dataloader_val,
    dataset_train, dataset_test, dataset_val,
    echograms_train, echograms_test, echograms_val) = get_datasets(
        frequencies=[18, 38, 120, 200], 
        iterations=1000, 
        batch_size=64,
        num_workers=0, 
        include_depthmap=False,
        test_size=400)
    end_time = time()

    print(f"time taken to load dataset {end_time - start_time}")

    _, (inputs_test, labels_test, si_test) = next(enumerate(dataloader_test))
    inputs_test = inputs_test.float().to(device)
    labels_test = labels_test.long().to(device)

    _, (inputs_tr, labels_tr, si_tr) = next(enumerate(dataloader_train))
    inputs_tr = inputs_tr.float().to(device)
    labels_tr = labels_tr.long().to(device)
    lab_dataloader = DataLoader(list(zip(inputs_tr, labels_tr, si_tr)), 64)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        itertools.chain(discriminator.parameters(), discriminator_cat.parameters()), 
        lr=opt.lr, 
        betas=(opt.b1, opt.b2)
    )
    optimizier_classifier = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def sample_image(n_row, batches_done):
        """Saves a grid of generated digits"""
        # Sample noise
        z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
        gen_imgs = decoder(z)
        save_image(gen_imgs[:, 0].reshape(-1, 1, 32, 32).data, f"output/semi_aae/samples/{batches_done}_c_{latent_dim}.png", nrow=10, normalize=True)

    def save_recon(original_image, reconstruction, batches_done, nrow):
        save_image(reconstruction[:, 0].reshape(-1, 1, 32, 32).data, f"output/semi_aae/reconstruction/{batches_done}_c_{latent_dim}.png", nrow=10, normalize=True)

    # ----------
    #  Training
    # ----------
    def sample_categorical(batch_size, n_classes=4, p=None):
        '''
        Sample from a categorical distribution
        of size batch_size and # of classes n_classes
        In case stated, a sampling probability given by p is used.
        return: torch.autograd.Variable with the sample
        '''
        #cat = np.random.randint(0, n_classes, batch_size)
        cat = np.random.choice(range(n_classes), size=batch_size, p=p)
        cat = np.eye(n_classes)[cat].astype('float32')
        cat = torch.from_numpy(cat)
        return Variable(cat.type(Tensor))


    class AAESS(nn.Module):
        def __init__(self, encoder, decoder):
            super(AAESS, self).__init__()
            self.encoder_obj = encoder
            self.decoder = decoder
            
            
        def encoder(self, x):
            z, y = self.encoder_obj(x)
            return z, z

        def classify(self, x):
            z, y = self.encoder_obj(x)
            return torch.argmax(y, dim=1)


    i = 0
    
    writer = SummaryWriter(comment=f" AAESS ( cap= {latent_dim} )")

    for ((imgs, l, _), (imgs_labeled, l, si)) in zip(dataloader_train, cycle(lab_dataloader)):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        z, y = encoder(real_imgs)
        decoded_imgs = decoder(torch.cat((z, y), 1))

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(z), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()
        optimizer_G.zero_grad()
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z_r = Variable(Tensor(np.random.normal(0, 5, (imgs.shape[0], opt.latent_dim))))
        y_r = sample_categorical(imgs.shape[0], opt.n_categories)
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z_r), valid)
        fake_loss = adversarial_loss(discriminator(z.detach()), fake)
        real_loss_cat = adversarial_loss(discriminator_cat(y_r), valid)
        fake_loss_cat = adversarial_loss(discriminator_cat(y.detach()), fake)

        d_loss = 0.25 * (real_loss + fake_loss + real_loss_cat + fake_loss_cat)

        d_loss.backward()
        optimizer_D.step()
        optimizer_D.zero_grad()


        #
        # semi supervised phase
        #

        optimizier_classifier.zero_grad()


        imgs_labeled_c = Variable(imgs_labeled.type(Tensor)).cuda()
        si = Variable(si.type(Tensor)).cuda().long()

        _, pred = encoder(imgs_labeled_c)
        classifier_loss = F.cross_entropy(pred, si)
        classifier_loss.backward()
        optimizier_classifier.step()

        writer.add_scalar("Loss/discriminator/train", d_loss.item(), i)
        writer.add_scalar("Loss/generator/train", g_loss.item(), i)
        writer.add_scalar("Loss/classifier/train", classifier_loss.item(), i)

        optimizier_classifier.zero_grad()
        if i % 10 == 0:
            print(
                f"[Batch {i}/{len(dataloader_train)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [C loss: {classifier_loss.item()}]"
            )

        batches_done = i

        if batches_done % opt.sample_interval == 0:
            #sample_image(n_row=10, batches_done=batches_done)
            writer.add_images("Reconstruction", decoded_imgs[:10, 0:1], i)
            save_recon(real_imgs, decoded_imgs, batches_done=batches_done, nrow=10)
        i += 1

    r = validate_clustering(AAESS(encoder, decoder), HDBSCAN(prediction_data=True), inputs_test, si_test, dataset_test.samplers, device, opt.latent_dim, 0, fig_path="output/aae/clustering/hdbscan_c_{latent_dim}.png", i=i, writer=writer)
    (r_score, cm, clf, clf_score) = r
    writer.add_scalar("Accuracy/svm", clf_score, i)
    writer.add_hparams({"model_name": "AAE", "capacity": latent_dim, "variational_beta": 0}, {"svm_accuracy": clf_score, "adjusted_rand_score": r_score, "loss_train": g_loss.item()})
    encoder.save(encoder_model_path)
    return r_score, clf_score
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

if __name__ == "__main__":
    latent_dim_options = [10]
    results = {}
    for i in latent_dim_options:
        r = main(i)
        results[i] = r

    with open(f"output/semi_aae/res.json", "w") as jf:
            json.dump(results, jf, sort_keys=True, indent=2)