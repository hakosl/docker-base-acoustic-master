# code adapted from https://arxiv.org/pdf/2007.09990.pdf
#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from data.echogram import get_echograms
from batch.dataset import Dataset
from torch.utils.data import DataLoader
from data.normalization import db
from datetime import datetime
from batch.samplers.background import Background
from batch.samplers.seabed import Seabed
from batch.samplers.shool import Shool
from batch.samplers.shool_seabed import ShoolSeabed
from batch.augmentation.flip_x_axis import flip_x_axis
from batch.augmentation.add_noise import add_noise
import os
from hdbscan import HDBSCAN

from data.echogram import get_echograms


from batch.data_transform_functions.remove_nan_inf import remove_nan_inf
from batch.data_transform_functions.db_with_limits import db_with_limits
from batch.label_transform_functions.index_0_1_27 import index_0_1_27
from batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from batch.combine_functions import CombineFunctions

import itertools
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, 
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
# parser.add_argument('--input', metavar='FILENAME',
#                     help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=5.0, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                    help='step size for scribble loss')
args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def partition_data(echograms, partition='random', portion_train=0.85):
    # Choose partitioning of data by specifying 'partition' == 'random' OR 'year'

    if partition == 'random':
        # Random partition of all echograms

        # Set random seed to get the same partition every time
        np.random.seed(seed=10)
        np.random.shuffle(echograms)
        train = echograms[:int(portion_train * len(echograms))]
        test = echograms[int(portion_train * len(echograms)):]
        val = []

        # Reset random seed to generate random crops during training
        np.random.seed(seed=None)


    elif partition == 'year':
        # Partition by year of echogram
        train = list(filter(lambda x: any(
            [year in x.name for year in
             ['D2011', 'D2012', 'D2013', 'D2014', 'D2015','D2016']]), echograms))
        test = list(filter(lambda x: any([year in x.name for year in ['D2017']]), echograms))
        val = []

    else:
        print(f"Parameter 'partition' must equal 'random' or 'year' {partition}")

    print('Train:', len(train), ' Test:', len(test), ' Val:', len(val))

    return train, test, val


# load image
window_dim=256
num_workers=0
batch_size=1
iterations=1
partition='year'
frequencies = [18, 38, 120, 200]
window_size = [window_dim, window_dim]

# Load echograms and create partition for train/test/val
echograms = get_echograms(frequencies=frequencies, minimum_shape=window_dim)

echograms_train, echograms_test, echograms_val = partition_data(echograms, partition=partition, portion_train=0.85)

sample_options = {
    "window_size": window_size,
    "random_sample": False,
}

sample_options_train = {
    "echograms": echograms_train,
    **sample_options
}
sample_options_test = {
    "echograms": echograms_test,
    **sample_options
}


sampler_probs = [1, 5, 5, 5, 5, 5]
label_types = [1, 27]

samplers_train = [
    #Background(**sample_options_train, sample_probs=sampler_probs[0]),
    #Seabed(**sample_options_train, sample_probs=sampler_probs[1]),
    Shool(**sample_options_train, fish_type=27, sample_probs=sampler_probs[2]),
    Shool(**sample_options_train, fish_type=1, sample_probs=sampler_probs[3]),    
    ShoolSeabed(**sample_options_train, fish_type=27, sample_probs=sampler_probs[4]),
    ShoolSeabed(**sample_options_train, fish_type=1, sample_probs=sampler_probs[5])
]
samplers_test = [
    Background(**sample_options_test, sample_probs=sampler_probs[0]),
    Seabed(**sample_options_test, sample_probs=sampler_probs[1]),
    Shool(**sample_options_test, fish_type=27, sample_probs=sampler_probs[2]),
    Shool(**sample_options_test, fish_type=1, sample_probs=sampler_probs[3]),    
    ShoolSeabed(**sample_options_test, fish_type=27, sample_probs=sampler_probs[4]),
    ShoolSeabed(**sample_options_test, fish_type=1, sample_probs=sampler_probs[5])
]

augmentation = CombineFunctions([add_noise, flip_x_axis])
label_transform = CombineFunctions([index_0_1_27, relabel_with_threshold_morph_close])
data_transform = CombineFunctions([remove_nan_inf])

dataset_train = Dataset(
    samplers_train,
    window_size,
    frequencies,
    batch_size * iterations,
    sampler_probs,
    augmentation_function=augmentation,
    label_transform_function=label_transform,
    data_transform_function=data_transform,
    si=True)

dataloader_train = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed)

i, (input_t, label_t, sample_index) = next(enumerate(dataloader_train))
data = db(input_t.float())

label_img = label_t.permute(1, 2, 0)
im = data.permute(2, 3, 1, 0)
print(data.shape, im.shape)
if use_cuda:
    data = data.cuda()
data = Variable(data)

# load scribble
if args.scribble:
    mask = cv2.imread(args.input.replace('.'+args.input.split('.')[-1],'_scribble.png'),-1)
    mask = mask.reshape(-1)
    mask_inds = np.unique(mask)
    mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )
    inds_sim = torch.from_numpy( np.where( mask == 255 )[ 0 ] )
    inds_scr = torch.from_numpy( np.where( mask != 255 )[ 0 ] )
    target_scr = torch.from_numpy( mask.astype(np.int) )
    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()
    target_scr = Variable( target_scr )
    # set minLabels
    args.minLabels = len(mask_inds)

# train
model = MyNet( data.size(1) )
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel)
HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel)
if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))

for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

    outputHP = output.reshape( (im.shape[0], im.shape[1], args.nChannel) )
    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy,HPy_target)
    lhpz = loss_hpz(HPz,HPz_target)

    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( [256, 256, 3] ).astype( np.uint8 )
        #cv2.imshow( "output1.png", im_target_rgb )
        #cv2.waitKey(10)

    # loss 
    if args.scribble:
        loss = args.stepsize_sim * loss_fn(output[ inds_sim ], target[ inds_sim ]) + args.stepsize_scr * loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ]) + args.stepsize_con * (lhpy + lhpz)
    else:
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
        
    loss.backward()
    optimizer.step()

    print (" ", batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item(), "\r", end="")

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )


inp = im[:, :, 1:, 0].data.cpu().numpy().astype(np.uint8)
label = label_img.data.cpu().numpy().astype(np.uint8)
concatted = np.vstack((inp, im_target_rgb))
print()
print(output.shape, im_target_rgb.shape, inp.shape, im.shape, concatted.shape)
timestamp = datetime.now().strftime("%H.%M.%S.%B.%d.%Y")
foldername = f"{timestamp}.{sampler_names[sample_index]}"
os.mkdir(f"output/{foldername}")

cv2.imwrite( f"output/{foldername}/input.png", inp )
cv2.imwrite( f"output/{foldername}/output.png", im_target_rgb )
cv2.imwrite( f"output/{foldername}/combined.png", concatted )
cv2.imwrite(f"output/{foldername}/labels.png", label)