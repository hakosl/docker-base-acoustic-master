# -*- coding: utf-8 -*-

# Run Nils Olavs pixel bases segmentation based on U-net

import os
import numpy as np
import sys
import json
import matplotlib.pyplot as plt

from data.echogram import Echogram
from models.keras.unet import Umodel
from models.keras.unet import jaccard_distance_loss

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

# import pdb

# Get and set up environment
json_data = open('./data_preprocessing/setpyenv.json').read()
localenv = json.loads(json_data)
sys.path.append(os.path.join(localenv["syspath"], "utils"))

modeldir = os.path.join(localenv["scratch"], 'model')
logdir = os.path.join(localenv["scratch"], 'logs')
if not os.path.isdir(modeldir):
    os.makedirs(modeldir)
if not os.path.isdir(logdir):
    os.makedirs(logdir)

path_to_echogram = os.path.join(
    localenv["scratch"],
    'DataOverview_North Sea NOR Sandeel cruise in Apr_May',
    'memmap')

# Set up Keras

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
tensorboard = TensorBoard(log_dir=logdir, write_graph=True)

# Parameters
freqs = [18, 38, 200, 333]  # select frequencies
nepo = 50  # Number of epochs

# window_size = {'vsize': 24, 'hsize': 48, 'vstep': 24, 'hstep': 48}
model_size = {'vsize': 64, 'hsize': 64}  # 'vstep': 64, 'hstep': 64}
window_size = {'vsize': 64, 'hsize': 64, 'vstep': 32, 'hstep': 32}

# Function definitions


def test_batch(data, labels, predictions=None):
    '''Plot a test image for the data and label'''

    # Initialize plot
    plt.tight_layout()
    plt.clf()
    # Remove the first dimension:
    labels = np.squeeze(labels)

    # Number of subplots
    n_plts = data.shape[0] + 1
    if predictions is not None:
        n_plts += 1

    # Channels
    for i in range(data.shape[0]):
        if i == 0:
            main_ax = plt.subplot(n_plts, 1, i+1)
        else:
            plt.subplot(n_plts, 1, i + 1, sharex=main_ax, sharey=main_ax)
        plt.imshow(10*np.log(data[i, :, :]), cmap='jet', aspect='auto')

    # Plot labels

    plt.subplot(n_plts, 1, i+2, sharex=main_ax, sharey=main_ax)
    plt.imshow(labels, aspect='auto')

    # Plot predictions
    if predictions is not None:
        predictions = np.squeeze(predictions)
        loss = jaccard_distance_loss(
            K.variable(labels),
            K.variable(predictions)).eval(session=K.get_session())

        plt.subplot(n_plts, 1, n_plts, sharex=main_ax, sharey=main_ax)
        plt.imshow(predictions, aspect='auto')  # , vmin=0, vmax=1)
        plt.title(loss)
    plt.show()


def get_batch(echogram, window_size, freqs):
    '''This function get batch data for  training'''

    # Get the data and labels
    data = echogram.data_numpy(frequencies=freqs)
    label = echogram.label_numpy()

    # Initialize data output arrays
    ds = data.shape
    nv = (ds[0]-window_size['vsize'])//window_size['vstep'] + 1
    nh = (ds[1]-window_size['hsize'])//window_size['hstep'] + 1
    batch_im = np.empty([nv*nh, len(freqs), window_size['vsize'],
                         window_size['hsize']])
    label_im = np.empty([nv*nh,  window_size['vsize'], window_size['hsize']])
    # Add data slice
    for i in np.arange(nv).astype('int'):
        for j in np.arange(nh).astype('int'):
            v0 = i*window_size['vstep']
            v1 = i*window_size['vstep']+window_size['vsize']
            h0 = j*window_size['hstep']
            h1 = j*window_size['hstep']+window_size['hsize']
            # batch_im[i*nv+j, :, :, :] =
            batch_im[i*nv+j, :, :, :] = np.transpose(
                data[v0:v1, h0:h1, :], (2, 0, 1))
            label_im[i*nv+j, :, :] = label[v0:v1, h0:h1]
    # Add empty dim for the frequency dimension in the labels
    label_im = label_im[:, np.newaxis, :, :]

    # label_im_cat = to_categorical(label_im, num_classes=None)

    return batch_im, label_im


def select_batch(batch_im, label_im):
    '''This function selects the batch data to get a more balanced data set'''

    # the total number of sub-images in the batch
    nim = batch_im.shape[0]
    # count the number of frames that has an acoustic category
    ind = np.full(nim, False, dtype=bool)
    for i in np.arange(nim):
        if np.sum(label_im[i, :].flatten()) > 0:  # This needs attention
            ind[i] = True
    batch_im_out = batch_im[ind, :, :, :]
    label_im_out = label_im[ind, :, :, :]

    return batch_im_out, label_im_out

# ############ Script - start ##############


# Get the echogram objects
echogram_names = os.listdir(path_to_echogram)
echogram = [
    Echogram(os.path.join(path_to_echogram, s)) for s in echogram_names]

# Only use echogram with selected frequencies
echogram = [s for s in echogram if all([f in s.frequencies for f in freqs])]

# Get the model
K.clear_session()
model, predict_model = Umodel(len(freqs), model_size['hsize'],
                              model_size['vsize'])

# Do da shit
data = np.empty([0, len(freqs), window_size['vsize'],
                 window_size['hsize']], dtype=float)
label = np.empty([0, 1, window_size['vsize'],
                  window_size['hsize']], dtype=float)
weight = np.empty([0, 1, window_size['vsize'],
                 window_size['hsize']], dtype=float)
# Extract data from the echogram objects
for s in echogram:
    # try:
    print('Loading    : '+os.path.split(s.name)[1])
    # Read batch from memmaps
    batch_im_full, batch_id_full = get_batch(s, window_size, freqs)
    # Select only images with both classes (fish/no fish)
    batch_im, batch_id = select_batch(batch_im_full, batch_id_full)
    data = np.append(data, batch_im, axis=0)
    label = np.append(label, batch_id, axis=0)
    batch_weight = np.expand_dims(batch_im[:, 3, :, :], axis=1)
    weight = np.append(weight, batch_weight, axis=0)

# Relabel the data to "fish" "no fish": NB må vera ulikl null
label[label > 0] = 1

# Try transformation on the input data. Usually a lower threshold
# of -75dB would be ok. An upper threshold of 0dB.
# The advantage of a hard treshold is that
# the values will continue to have a meaning in terms of energy.
data_db = 10*np.log10(data)
# ma = np.max(data_db)
# mi = np.min(data_db)
ma = 0  # dB
mi = -75  # dB

# Rescale
data_db = (data_db-mi)/(ma-mi)
data_db[data_db > 1] = 1
data_db[data_db < 0] = 0

# Try a small subset to see if I can overfit
# data_db = data_db[0:32, :, :]
# label = label[0:32, :, :]

print('Processing')

# https://machinelearningmastery.com/check-point-deep-learning-models-keras/
modelfile_final = os.path.join(modeldir, 'modelweights.hdf5')
modelfile = os.path.join(modeldir,
                         'modelweights-{epoch:02d}-{val_loss:.2f}.hdf5')

checkpoint = ModelCheckpoint(modelfile, monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
callbacks_list = [tensorboard, checkpoint]

# Fit the model
history_callback = model.fit([data_db, label, weight], verbose=1,
                             epochs=nepo,
                             validation_split=0.1,
                             callbacks=callbacks_list,
                             batch_size=32)

# Get the model and weights
dat = model.get_weights()
mod = model.to_json()

# Save the final weights as numpy after each epoch
modelfile = os.path.join(
    modeldir, 'modelweights.npz')
np.savez(modelfile_final, dat=dat, mod=mod)
