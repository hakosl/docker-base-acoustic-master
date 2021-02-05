import numpy as np
import torch
from scipy.interpolate import interpn
from torch.autograd import Variable
import torch.nn.functional as F








# Create heatmap: pixel-wise prediction of echogram from trained patch classifier
from predict._frameworks import get_prediction_function


def heatmap(model, data, window_size, resolution, normalization=None):
    """

    :param model: (torch.nn.Model object): classifier to evaluate each patch of the echogram
    :param echogram: (Echogram object): echogram to predict
    :param resolution: (int): distance between center pixels of two neighbouring patches
    :param frequencies:
    :param gpu_no:
    :param patch_normalization:
    :return:
    """

    pred_func = get_prediction_function(model)

    grid_y = np.arange(0, data.shape[0] - window_size, resolution)
    grid_x = np.arange(0, data.shape[1] - window_size, resolution)
    data = np.rollaxis(data, 2)
    data = np.expand_dims(data, 0)
    prediction = np.zeros((len(grid_y), len(grid_x)))


    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):

            patch = data[:,:, y:y+window_size, x:x+window_size].copy()

            # Normalize patch to [-1, 1] (if all values in patch are equal, values become equal)
            if normalization is not None:
                patch = normalization(patch)
            prediction = pred_func(model, data)[1]

            prediction[i, j] = F.softmax(model(patch)[0], dim=0).cpu().numpy()[1]

    coord = (np.arange(0, prediction.shape[0])*resolution + window_size // 2,
              np.arange(0, prediction.shape[1])*resolution + window_size // 2)
    value = np.power(prediction, 3)
    grid = np.asarray(np.meshgrid(np.arange(data.shape[0]), np.arange(data..shape[1]), indexing='ij'))
    grid = np.moveaxis(grid, 0, 2).reshape(-1, 2)

    hmap = interpn(coord, value, grid, method='linear', bounds_error=False, fill_value=0)
    hmap = hmap.reshape(data.shape[0], data.shape[1])

    return hmap
