import numpy as np

# Create segmentation from trained segmentation model (e.g. U-Net)
from predict._frameworks_Olav import get_prediction_function


def segmentation(model, data, patch_size, patch_overlap, device):
    """
    Due to memory restrictions on device, echogram is divided into patches.
    Each patch is segmented, and then stacked to create segmentation of the full echogram
    :param model:(torch.nn.Model object): segmentation model
    :param echogram:(Echogram object): echogram to predict
    :param window_dim_init: (positive int): initial window dimension of each patch (cropped by trim_edge parameter after prediction)
    :param trim_edge: (positive int): each predicted patch is cropped by a frame of trim_edge number of pixels
    :return:
    """

    pred_func = get_prediction_function(model)


    # Functions to convert between B x C x H x W format and W x H x C format
    def hwc_to_bchw(x):
        return np.expand_dims(np.moveaxis(x, -1, 0), 0)

    def bcwh_to_hwc(x):
        return np.moveaxis(x.squeeze(0), 0, -1)

    if type(patch_size) == int:
        patch_size = [patch_size, patch_size]

    if type(patch_overlap) == int:
        patch_overlap = [patch_overlap, patch_overlap]

    if len(data.shape) == 2:
        data = np.expand_dims(data, -1)

    # Add padding to avoid trouble when removing the overlap later
    data = np.pad(data, [[patch_overlap[0], patch_overlap[0]], [patch_overlap[1], patch_overlap[1]], [0, 0]],
                  'constant')

    # Loop through patches identified by upper-left pixel
    upper_left_x0 = np.arange(0, data.shape[0] - patch_overlap[0], patch_size[0] - patch_overlap[0] * 2)
    upper_left_x1 = np.arange(0, data.shape[1] - patch_overlap[1], patch_size[1] - patch_overlap[1] * 2)

    predictions = []
    for x0 in upper_left_x0:
        for x1 in upper_left_x1:
            # Cut out a small patch of the data
            data_patch = data[x0:x0 + patch_size[0], x1:x1 + patch_size[1], :]

            # Pad with zeros if we are at the edges
            pad_val_0 = patch_size[0] - data_patch.shape[0]
            pad_val_1 = patch_size[1] - data_patch.shape[1]

            if pad_val_0 > 0:
                data_patch = np.pad(data_patch, [[0, pad_val_0], [0, 0], [0, 0]], 'constant')

            if pad_val_1 > 0:
                data_patch = np.pad(data_patch, [[0, 0], [0, pad_val_1], [0, 0]], 'constant')

            # Run it through model
            out_patch = pred_func(model, hwc_to_bchw(data_patch), device)
            out_patch = bcwh_to_hwc(out_patch)


            # Make output array (We do this here since it will then be agnostic to the number of output channels)
            if len(predictions) == 0:
                predictions = np.concatenate(
                    [data[:-(patch_overlap[0] * 2), :-(patch_overlap[1] * 2), 0:1] * 0] * out_patch.shape[2], -1)

            # Remove potential padding related to edges
            out_patch = out_patch[0:patch_size[0] - pad_val_0, 0:patch_size[1] - pad_val_1, :]

            # Remove potential padding related to overlap between data_patches
            out_patch = out_patch[patch_overlap[0]:-patch_overlap[0], patch_overlap[1]:-patch_overlap[1], :]

            # Insert output_patch in out array
            predictions[x0:x0 + out_patch.shape[0], x1:x1 + out_patch.shape[1], :] = out_patch

    return predictions

if __name__ == "__main__":

    import torch

    from models.unet_bn_sequential import UNet
    from data.echogram import get_echograms
    from data.normalization import sigmoid_log_visualize_echogram
    from model_train_unet import partition_data
    from batch.label_transform_functions.is_fish import is_fish
    from batch.label_transform_functions.relabel_with_threshold import relabel_with_threshold
    import utils.plotting
    plt = utils.plotting.setup_matplotlib()  # Returns import matplotlib.pyplot as plt

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # (str) Path to directory containing file with model parameters
    path_model_params_dir_name = # Insert path

    # (str) Name of file with model parameters
    path_model_params_file_name = 'unet_bn_sequential' + '_binary_fish_background' + '.pt'

    # Create path to load model parameters
    if not path_model_params_dir_name[-1] == '/':
        path_model_params_dir_name = path_model_params_dir_name + '/'
    path_model_params = path_model_params_dir_name + path_model_params_file_name

    echograms = partition_data(get_echograms())[1]  # Validation set
    freqs = [18, 38, 120, 200]
    patch_size = 2048
    patch_overlap = 20

    model = UNet(n_classes=2, in_channels=4)
    model.to(device)
    model.load_state_dict(torch.load(path_model_params))
    model.eval()

    for ech in echograms:
        if ech.n_objects != 0:
            with torch.no_grad():
                data = ech.data_numpy(frequencies=freqs)
                data[np.invert(np.isfinite(data))] = 0
                seg = segmentation(model, data, patch_size, patch_overlap, device)[:, :, 1]

                # Get refined labels for plot
                labels = ech.label_numpy()
                relabel = is_fish(data, labels, ech)[1]
                relabel = relabel_with_threshold(np.rollaxis(data, -1, 0), relabel, ech, ignore_zero_inside_bbox=False)[1]

                # Visualize echogram with original/refined labels and prediction
                ech.visualize(frequencies=freqs, pred_contrast=30.0, predictions=[relabel, seg], show_labels=True, show_grid=True, data_transform=sigmoid_log_visualize_echogram)