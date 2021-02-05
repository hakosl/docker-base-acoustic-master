import numpy as np
import torch
from scipy.interpolate import interpn
from torch.autograd import Variable
import torch.nn.functional as F

# Create heatmap from patch classifier by directly evaluating the classifier on the full image
#from predict._frameworks import predict_with_pytorch, get_prediction_function
from predict._frameworks_Olav import get_prediction_function


def heatmap_yolo(model, data, device):

    input_data_shape = data.shape

    pred_func = get_prediction_function(model)


    data[np.invert(np.isfinite(data))] = 0
    data = np.rollaxis(data, 2)
    data = np.expand_dims(data, 0)

    def new_spatial_dims_for_input(in_dim):
        if (in_dim - model.window_dim) % 8 == 0:
            return 0
        else:
            return 8 - (in_dim - model.window_dim) % 8

    def pad_to_shift_prediction_center_pixel(data, n):
        return np.pad(data, ((0, 0), (0, 0), (n, 0), (n, 0)), mode='constant')

    def pad_to_get_correct_dimensions(data, n_y, n_x):
        return np.pad(data, ((0, 0), (0, 0), (0, n_y), (0, n_x)), mode='constant')

    data = pad_to_shift_prediction_center_pixel(data, n=model.window_dim//2)
    data = pad_to_get_correct_dimensions(
        data=data,
        n_y=new_spatial_dims_for_input(data.shape[2]),
        n_x=new_spatial_dims_for_input(data.shape[3])
    )
    prediction = pred_func(model, data, device)[0, 1, :, :]

    coord = (
        np.arange(prediction.shape[0]) * 8,
        np.arange(prediction.shape[1]) * 8
    )

    grid = np.asarray(np.meshgrid(np.arange(input_data_shape[0]), np.arange(input_data_shape[1]), indexing='ij'))
    grid = np.moveaxis(grid, 0, 2).reshape(-1, 2)

    prediction = interpn(coord, prediction, grid, method='linear', bounds_error=False, fill_value=0)
    prediction = prediction.reshape(input_data_shape[0], input_data_shape[1])

    return prediction

if __name__ == "__main__":
    from models.simple_net_51_2 import Net3_new
    from data.echogram import get_echograms
    from model_train import partition_data
    import utils.plotting
    plt = utils.plotting.setup_matplotlib()  # Returns import matplotlib.pyplot as plt

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    path_model_params = \
        '/nr/project/bild/Cogmar/usr/obr/model/simple_net_54_Net3_new_sigmoid_log_sandeel.pt'
    echograms = partition_data(get_echograms())[1]  # Validation set
    freqs = [18, 38, 120, 200]

    model = Net3_new()
    model.load_state_dict(torch.load(path_model_params))
    model.to(device)
    model.eval()

    for ech in echograms:
        with torch.no_grad():
            data = ech.data_numpy(frequencies=[18, 38, 120, 200])
            prediction = heatmap_yolo(model, data, device)
            prediction = np.power(prediction, 10)
            ech.visualize(predictions=prediction)
