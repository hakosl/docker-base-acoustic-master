import numpy as np

#Example on drawing samples
from batch.augmentation.flip_x_axis import flip_x_axis
from batch.data_transform_functions.db_with_limits import db_with_limits
from batch.dataset import Dataset
from batch.label_transform_functions.is_sandeel import is_sandeel
from batch.samplers.background import Background
from batch.samplers.background_seabed import BackgroundSeabed
from batch.samplers.shool import Shool
from batch.samplers.shool_seabed import ShoolSeabed
from data.echogram import get_echograms

#Get echograms for training
echograms = get_echograms(list(range(2005,2017))) #All years but 2017
window_size = [54,54]
frequencies = [200]

training_ds = Dataset(

    #We will draw from four classes:
    [ Background(echograms, window_size),
      BackgroundSeabed(echograms, window_size),
      Shool(echograms, 'all'),
      ShoolSeabed(echograms, window_size[0]//2, 'all')],

    window_size, frequencies,
    sampler_probs=[.25,.25,.25,.25], #Probabilities of using the different samplers
    #label_transform_function=is_sandeel, #Convert label image
    data_transform_function=db_with_limits, #Apply db-transform and limit between -75 and 0
    augmentation_function=flip_x_axis, #Random flipping of axis
)


for i in range(100):
    sample = training_ds[i]
    print(np.min(sample[0]),np.max(sample[0]), np.unique(sample[1]))
