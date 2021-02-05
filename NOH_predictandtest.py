# -*- coding: utf-8 -*-

# Visualize Nils Olav's pixel bases segmentation based on U-net.
# Run NOH_main.py prior to this script

import numpy as np
import os
import sys
from data.echogram import get_echograms
from data.normalization import db
from models.keras.unet import Umodel
from predict.segmentation import segmentation
from keras import backend as K
# from data.frequencyresponse import plotfrequencyresponse
import json

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

# Parameters
freqs = [18, 38, 200, 333]  # select frequencies
model_size = {'vsize': 512, 'hsize': 512}

all_echograms = get_echograms(years=2012, frequencies=freqs)
all_echograms = all_echograms[0:20]
# plotfrequencyresponse(all_echograms, freqs)

# Parameters
freqs = [18, 38, 200, 333]  # select frequencies
model_size = {'vsize': 512, 'hsize': 512}

# Define model
model, predict_model = Umodel(len(freqs), model_size['hsize'],
                              model_size['vsize'])

# Read and set weights from file
json_data = open('./data_preprocessing/setpyenv.json').read()
localenv = json.loads(json_data)
sys.path.append(os.path.join(localenv["syspath"], "utils"))
modeldir = os.path.join(localenv["scratch"], 'model')
modelfile = os.path.join(modeldir, 'modelweights.hdf5.npz')
saved_model = np.load(modelfile)
predict_model.set_weights(saved_model['dat'])

# Run predictions
window_dim_init = [512, 512]
trim_edge = [50, 50]
data = all_echograms[0].data_numpy(freqs)

data_db = 10*np.log10(data)
# ma = np.max(data_db)
# mi = np.min(data_db)
ma = 0  # dB
mi = -75  # dB

# Rescale
data_db = (data_db-mi)/(ma-mi)
data_db[data_db > 1] = 1
data_db[data_db < 0] = 0

# Run predictions
prediction = segmentation(predict_model, data_db, window_dim_init, trim_edge)

# Vizuliase results
len(all_echograms)
all_echograms[1].visualize(predictions=prediction.squeeze(), frequencies=[200])

for ec in all_echograms:
    ec.visualize(predictions=prediction.squeeze(), frequencies=[200])
