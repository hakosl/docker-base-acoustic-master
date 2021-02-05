import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interpn
import copy

from echogram import *
from _not_in_use.train_model_old import partition_data
from models.simple_net_51_2 import Net3, Net3_new

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Create heatmap: pixel-wise prediction of echogram from trained patch classifier
def heatmap(model, echogram, resolution):

    ### Input parameters
    # model (torch.nn.Model object): classifier to evaluate each patch of the echogram
    # echogram (Echogram object): echogram to predict
    # resolution (int): distance between center pixels of two neighbouring patches
    ###

    window_size = model.window_dim
    data = echogram.data_numpy(frequencies=[18, 38, 120, 200])
    data[np.invert(np.isfinite(data))] = 0

    grid_y = np.arange(0, data.shape[0] - window_size, resolution)
    grid_x = np.arange(0, data.shape[1] - window_size, resolution)
    data = np.rollaxis(data, 2)
    data = np.expand_dims(data, 0)
    prediction = np.zeros((len(grid_y), len(grid_x)))

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for i, y in enumerate(grid_y):
        for j, x in enumerate(grid_x):

            patch = data[:,:, y:y+window_size, x:x+window_size].copy()

            # Normalize patch to [-1, 1] (if all values in patch are equal, values become equal)
            for f in range(patch.shape[1]):
                patch[:,f,:,:] -= np.min(patch[:,f,:,:])
                patch[:,f,:,:] = np.log(patch[:,f,:,:] + 1e-10)
                patch[:,f,:,:] -= np.mean(patch[:,f,:,:])
                patch[:,f,:,:] *= 1 / (np.std(patch[:,f,:,:]) + 1e-10)

            with torch.no_grad():
                patch = torch.Tensor(patch).float()
                patch = patch.to(device)
                prediction[i, j] = F.softmax(model(patch)[0], dim=0).cpu().numpy()[1]

    coord = (np.arange(0, prediction.shape[0])*resolution + window_size // 2,
              np.arange(0, prediction.shape[1])*resolution + window_size // 2)
    value = np.power(prediction, 3)
    grid = np.asarray(np.meshgrid(np.arange(echogram.shape[0]), np.arange(echogram.shape[1]), indexing='ij'))
    grid = np.moveaxis(grid, 0, 2).reshape(-1, 2)

    hmap = interpn(coord, value, grid, method='linear', bounds_error=False, fill_value=0)
    hmap = hmap.reshape(echogram.shape[0], echogram.shape[1])

    return hmap


# Save heatmap as numpy file
def save_heatmap(model, echogram, resolution, save_path):

    ### Input parameters
    # model: (torch.nn.Model object), classifier to evaluate each patch of the echogram
    # echogram: (Echogram object), echogram to predict
    # resolution: (int), number of pixels between center pixel of neighbouring patches
    # save_path: (string), path directory to save file
    ###

    hmap = heatmap(model=model, echogram=echogram, resolution=resolution)
    np.save(save_path, hmap)

    return None


def create_and_visualize_heatmap(model, echogram, resolution):

    path_to_trained_model_parameters = '/nr/project/bild/Cogmar/usr/obr/model/simple_net_54_Net3_balanced_seabed.pt'
    #hmap_file_name = 'hmap_echo_2_ws_52_0_res_5.npy'
    #hmap_save_path = os.path.join(os.getcwd(), hmap_file_name)

    model.load_state_dict(torch.load(path_to_trained_model_parameters))
    model.to(device)
    hmap = heatmap(model, echogram, resolution)
    echogram.visualize(predictions=hmap, pred_contrast=5)

    return None

# Create heatmap from patch classifier by directly evaluating the classifier on the full image
def heatmap_yolo(model, path_model_params, echogram, div_std):

    data = echogram.data_numpy(frequencies=[18, 38, 120, 200])

    data[np.invert(np.isfinite(data))] = 0
    data = np.rollaxis(data, 2)
    data = np.expand_dims(data, 0)

    if div_std:
        data -= np.min(data, axis=(2, 3), keepdims=True)
        data = np.log(data + 1e-10)
        data -= np.mean(data, axis=(2, 3), keepdims=True)
        data *= 1 / (np.std(data, axis=(2, 3), keepdims=True) + 1e-10)


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

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path_model_params))
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = torch.Tensor(data).float()
        data = data.to(device)
        prediction = model(data)
        prediction = F.softmax(prediction, dim=1)[0, 1, :, :]
        prediction = prediction.cpu().numpy()

    coord = (
        np.arange(prediction.shape[0]) * 8,
        np.arange(prediction.shape[1]) * 8
    )

    grid = np.asarray(np.meshgrid(np.arange(echogram.shape[0]), np.arange(echogram.shape[1]), indexing='ij'))
    grid = np.moveaxis(grid, 0, 2).reshape(-1, 2)

    prediction = interpn(coord, prediction, grid, method='linear', bounds_error=False, fill_value=0)
    prediction = prediction.reshape(echogram.shape[0], echogram.shape[1])

    return prediction


# Create heatmap from patch classifier by directly evaluating the classifier on the full image
def heatmap_yolo_multiple_echograms(model, path_model_params, echograms, div_std):

    def new_spatial_dims_for_input(in_dim):
        if (in_dim - model.window_dim) % 8 == 0:
            return 0
        else:
            return 8 - (in_dim - model.window_dim) % 8

    def pad_to_shift_prediction_center_pixel(data, n):
        return np.pad(data, ((0, 0), (0, 0), (n, 0), (n, 0)), mode='constant')

    def pad_to_get_correct_dimensions(data, n_y, n_x):
        return np.pad(data, ((0, 0), (0, 0), (0, n_y), (0, n_x)), mode='constant')

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path_model_params))
    model.to(device)
    model.eval()

    predictions = []

    for echogram in echograms:
        with torch.no_grad():

            data = echogram.data_numpy(frequencies=[18, 38, 120, 200])

            data[np.invert(np.isfinite(data))] = 0
            data = np.rollaxis(data, 2)
            data = np.expand_dims(data, 0)

            if div_std:
                data -= np.min(data, axis=(2, 3), keepdims=True)
                data = np.log(data + 1e-10)
                data -= np.mean(data, axis=(2, 3), keepdims=True)
                data *= 1 / (np.std(data, axis=(2, 3), keepdims=True) + 1e-10)

            data = pad_to_shift_prediction_center_pixel(data, n=model.window_dim // 2)
            data = pad_to_get_correct_dimensions(
                data=data,
                n_y=new_spatial_dims_for_input(data.shape[2]),
                n_x=new_spatial_dims_for_input(data.shape[3])
            )

            data = torch.Tensor(data).float()
            data = data.to(device)
            prediction = model(data)
            prediction = F.softmax(prediction, dim=1)[0, 1, :, :]
            prediction = prediction.cpu().numpy()

        coord = (
            np.arange(prediction.shape[0]) * 8,
            np.arange(prediction.shape[1]) * 8
        )

        grid = np.asarray(np.meshgrid(np.arange(echogram.shape[0]), np.arange(echogram.shape[1]), indexing='ij'))
        grid = np.moveaxis(grid, 0, 2).reshape(-1, 2)

        prediction = interpn(coord, prediction, grid, method='linear', bounds_error=False, fill_value=0)
        #prediction = prediction.reshape(echogram.shape[0], echogram.shape[1])
        prediction = list(prediction.reshape(-1))

        predictions += prediction

    return predictions

# Create segmentation from trained segmentation model (e.g. U-Net)
def segmentation(model, echogram, window_dim_init, trim_edge):

    # Due to memory restrictions on device, echogram is divided into patches.
    # Each patch is segmented, and then stacked to create segmentation of the full echogram

    ### Input parameters
    # model (torch.nn.Model object): segmentation model
    # echogram (Echogram object): echogram to predict
    # window_dim_init (positive int): initial window dimension of each patch (cropped by trim_edge parameter after prediction)
    # trim_edge (positive int): each predicted patch is cropped by a frame of trim_edge number of pixels
    ###

    #device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    window_dim_reduced = window_dim_init - 2*trim_edge

    data = echogram.data_numpy(frequencies=[18, 38, 120, 200])
    data = np.rollaxis(data, 2)
    data = np.expand_dims(data, 0)
    print(data.shape)

    # Set non-finite values (nan, positive inf, negative inf) to zero
    if np.any(np.invert(np.isfinite(data))):
        data[np.invert(np.isfinite(data))] = 0

    dim_x_data = data.shape[3]
    dim_y_data = data.shape[2]

    segmentation = np.zeros((dim_y_data, dim_x_data))
    print(segmentation.shape)

    # Returns patch, zero-padded if necessary
    def patch_zero_pad(data, y_start, y_stop, x_start, x_stop):

        if y_start < 0:
            data = np.pad(data, pad_width=((0, 0), (0, 0), (-y_start, 0), (0, 0)), mode='constant')
            y_stop -= y_start
            y_start = 0
        if y_stop > dim_y_data:
            data = np.pad(data, pad_width=((0, 0), (0, 0), (0, y_stop - dim_y_data), (0, 0)), mode='constant')
        if x_start < 0:
            data = np.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (-x_start, 0)), mode='constant')
            x_stop -= x_start
            x_start = 0
        if x_stop > dim_x_data:
            data = np.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, x_stop - dim_x_data)), mode='constant')

        return data[:, :, y_start:y_stop, x_start:x_stop]


    pos_y = -trim_edge
    while pos_y + trim_edge < dim_y_data:
        pos_x = -trim_edge
        while pos_x + trim_edge < dim_x_data:
            patch = patch_zero_pad(data, y_start = pos_y, y_stop = pos_y + window_dim_init, x_start = pos_x, x_stop = pos_x + window_dim_init)


            # Log-transform and normalize data (each batch item and frequency separately)
            patch -= np.min(patch, axis=(2, 3), keepdims=True)
            patch = np.log(patch + 1e-10)
            patch -= np.mean(patch, axis=(2, 3), keepdims=True)
            patch *= 1 / (np.std(patch, axis=(2, 3), keepdims=True) + 1e-10)

            with torch.no_grad():
                patch = torch.Tensor(patch).to(device)
                patch_segmentation = model(patch)
                #print(patch_segmentation.size())
                patch_segmentation = F.softmax(patch_segmentation, dim=1)
                #print(patch_segmentation.size())
                patch_segmentation = patch_segmentation.cpu().numpy()

            patch_segmentation = np.squeeze(patch_segmentation[:, 1, :, :])
            #print(patch_segmentation.shape)

            patch_stop_y = trim_edge + window_dim_reduced
            patch_stop_x = trim_edge + window_dim_reduced
            if pos_y + window_dim_reduced > dim_y_data:
                patch_stop_y = trim_edge + dim_y_data % window_dim_reduced
            if pos_x + window_dim_reduced > dim_x_data:
                patch_stop_x = trim_edge + dim_x_data % window_dim_reduced

            segmentation[(pos_y + trim_edge):(pos_y + trim_edge + window_dim_reduced), (pos_x + trim_edge):(pos_x + trim_edge + window_dim_reduced)]\
                = patch_segmentation[trim_edge:patch_stop_y, trim_edge:patch_stop_x]

            pos_x += window_dim_reduced

        pos_y += window_dim_reduced

    return segmentation


def plot_numpy(data):
    #import utils.plotting
    #plt = utils.plotting.setup_matplotlib()  # Returns import matplotlib.pyplot as plt

    plt.gcf()
    plt.imshow(data)
    plt.show()


def prediction_measures(model, path_model_params, echogram, div_std, pred_contrast=1.0):

    def measures(prediction, target):
        tp = np.sum(prediction * target)  # True positives
        fp = np.sum(prediction * (1 - target))  # False positives
        fn = np.sum((1 - prediction) * target)  # False negatives

        precision = tp / (tp + fp) if tp + fp != 0 else 1.0
        recall = tp / (tp + fn) if tp + fn != 0 else 1.0
        jaccard = tp / (tp + fp + fn) if tp + fp + fn != 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
        # f1 == 1 / (1 + (fp + fn) / (2 * tp))

        return dict({
            'precision': np.array(precision),
            'recall': np.array(recall),
            'jaccard': np.array(jaccard),
            'f1': np.array(f1)
        })

    _prediction = heatmap_yolo(model, path_model_params, echogram, div_std)
    _prediction = np.power(_prediction, pred_contrast)

    target = echogram.label_numpy()
    target[target != 0] = 1

    n_tresholds = 100

    precision = []
    recall = []
    jaccard = []
    f1 = []

    for pred_treshold in np.linspace(0, 1, n_tresholds, endpoint=False):

        prediction = copy.copy(_prediction)
        prediction[prediction < pred_treshold] = 0
        prediction[prediction >= pred_treshold] = 1

        measures_dict = measures(prediction, target)

        precision.append(measures_dict['precision'])
        recall.append(measures_dict['recall'])
        jaccard.append(measures_dict['jaccard'])
        f1.append(measures_dict['f1'])

    return dict({
        'prediction': _prediction,
        'precision': np.array(precision),
        'recall': np.array(recall),
        'jaccard': np.array(jaccard),
        'f1': np.array(f1)
    })



def prediction_measures_multiple_echograms(model, path_model_params, echograms, div_std, pred_contrast=1.0):

    def measures(prediction, target):
        tp = np.sum(prediction * target)        # True positives
        fp = np.sum(prediction * (1 - target))  # False positives
        fn = np.sum((1 - prediction) * target)  # False negatives

        precision = tp / (tp + fp) if tp + fp != 0 else 1.0
        recall = tp / (tp + fn) if tp + fn != 0 else 1.0
        jaccard = tp / (tp + fp + fn) if tp + fp + fn != 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
        # f1 == 1 / (1 + (fp + fn) / (2 * tp))

        return dict({
            'precision': np.array(precision),
            'recall': np.array(recall),
            'jaccard': np.array(jaccard),
            'f1': np.array(f1)
        })

    _predictions = heatmap_yolo_multiple_echograms(model, path_model_params, echograms, div_std)
    _predictions = np.array(_predictions)
    _predictions = np.power(_predictions, pred_contrast)

    targets = []
    for echogram in echograms:
        target = echogram.label_numpy().reshape(-1)
        target[np.invert(np.isfinite(target))] = 0
        target[target != 0] = 1
        targets += list(target)
    targets = np.array(targets)

    n_tresholds = 100

    precision = []
    recall = []
    jaccard = []
    f1 = []

    for pred_treshold in np.linspace(0, 1, n_tresholds, endpoint=False):

        predictions = copy.copy(_predictions)
        predictions[predictions < pred_treshold] = 0
        predictions[predictions >= pred_treshold] = 1

        measures_dict = measures(predictions, targets)

        precision.append(measures_dict['precision'])
        recall.append(measures_dict['recall'])
        jaccard.append(measures_dict['jaccard'])
        f1.append(measures_dict['f1'])

    return dict({
        'prediction': _predictions,
        'precision': np.array(precision),
        'recall': np.array(recall),
        'jaccard': np.array(jaccard),
        'f1': np.array(f1)
    })


def plot_predictions_and_prediction_measures(echogram, model_1, model_2, path_model_params_1, path_model_params_2, div_std_1, div_std_2):

    if echogram.n_objects == 0:
        print('Warning: Echogram contains no labeled objects')

    pred_measure_true = prediction_measures(
        model=model_1,
        path_model_params=path_model_params_1,
        echogram=echogram,
        div_std=div_std_1,
        pred_contrast=60
    )
    pred_measure_false = prediction_measures(
        model=model_2,
        path_model_params=path_model_params_2,
        echogram=echogram,
        div_std=div_std_2,
        pred_contrast=20
    )

    prediction_true = pred_measure_true['prediction']
    prediction_false = pred_measure_false['prediction']
    freqs = [18, 38, 120, 200]
    echogram.visualize(predictions=[prediction_true, prediction_false], frequencies=freqs, draw_seabed=True)

    precision_true = pred_measure_true['precision']
    precision_false = pred_measure_false['precision']
    recall_true = pred_measure_true['recall']
    recall_false = pred_measure_false['recall']
    jaccard_true = pred_measure_true['jaccard']
    jaccard_false = pred_measure_false['jaccard']

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(recall_true)
    axs[0, 0].plot(precision_true)
    axs[0, 0].plot(jaccard_true)
    axs[0, 0].legend(('Recall', 'Precision', 'Jaccard'))
    axs[0, 0].set_title('Balanced Seabed: True')

    axs[1, 0].plot(recall_false)
    axs[1, 0].plot(precision_false)
    axs[1, 0].plot(jaccard_false)
    axs[1, 0].legend(('Recall', 'Precision', 'Jaccard'))
    axs[1, 0].set_title('Balanced Seabed: False')

    axs[0, 1].scatter(recall_true, precision_true)
    axs[0, 1].scatter(recall_false, precision_false)
    axs[0, 1].legend(('Balanced Seabed: True', 'Balanced Seabed: False'))
    axs[0, 1].set_xlabel('Recall')
    axs[0, 1].set_ylabel('Precision')

    plt.show()



def plot_prediction_measures_multiple_echograms(echograms, model_1, model_2, path_model_params_1, path_model_params_2, div_std_1, div_std_2):

    pred_measure_true = prediction_measures_multiple_echograms(
        model=model_1,
        path_model_params=path_model_params_1,
        echograms=echograms,
        div_std=div_std_1,
        pred_contrast=100
    )
    pred_measure_false = prediction_measures_multiple_echograms(
        model=model_2,
        path_model_params=path_model_params_2,
        echograms=echograms,
        div_std=div_std_2,
        pred_contrast=20
    )

    freqs = [18, 38, 120, 200]

    precision_true = pred_measure_true['precision']
    precision_false = pred_measure_false['precision']
    recall_true = pred_measure_true['recall']
    recall_false = pred_measure_false['recall']
    jaccard_true = pred_measure_true['jaccard']
    jaccard_false = pred_measure_false['jaccard']

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(recall_true)
    axs[0, 0].plot(precision_true)
    axs[0, 0].plot(jaccard_true)
    axs[0, 0].legend(('Recall', 'Precision', 'Jaccard'))
    axs[0, 0].set_title('Balanced Seabed: True')

    axs[1, 0].plot(recall_false)
    axs[1, 0].plot(precision_false)
    axs[1, 0].plot(jaccard_false)
    axs[1, 0].legend(('Recall', 'Precision', 'Jaccard'))
    axs[1, 0].set_title('Balanced Seabed: False')

    axs[0, 1].scatter(recall_true, precision_true)
    axs[0, 1].scatter(recall_false, precision_false)
    axs[0, 1].legend(('Balanced Seabed: True', 'Balanced Seabed: False'))
    axs[0, 1].set_xlabel('Recall')
    axs[0, 1].set_ylabel('Precision')

    plt.show()


if __name__ == '__main__':

    # Plot prediction measures - multiple echograms
    '''
    model_1 = Net3_new()
    model_2 = Net3()
    path_model_params_1 = \
        '/nr/project/bild/Cogmar/usr/obr/model/simple_net_54_Net3_new_balanced_seabed_True_sigmoid_log_separate__fc_prelu__final.pt'
    path_model_params_2 = \
        '/nr/project/bild/Cogmar/usr/obr/model/simple_net_54_Net3_balanced_seabed_True_div_patch_by_std__fc_prelu__final.pt'

    echograms = partition_data(get_echograms())[1] #Validation set
    echograms = echograms[:50]
    #freqs = [18, 38, 120, 200]

    plot_prediction_measures_multiple_echograms(
        echograms=echograms,
        model_1=model_1,
        model_2=model_2,
        path_model_params_1=path_model_params_1,
        path_model_params_2=path_model_params_2,
        div_std_1=False,
        div_std_2=True
    )
    '''



    # Plot prediction measures - one echogram at the time
    #'''
    model_1 = Net3_new()
    model_2 = Net3()
    path_model_params_1 = \
        '/nr/project/bild/Cogmar/usr/obr/model/simple_net_54_Net3_new_balanced_seabed_True_sigmoid_log_separate__fc_prelu__final.pt'
    path_model_params_2 = \
        '/nr/project/bild/Cogmar/usr/obr/model/simple_net_54_Net3_balanced_seabed_True_div_patch_by_std__fc_prelu__final.pt'

    echograms = partition_data(get_echograms())[1] #Validation set
    freqs = [18, 38, 120, 200]

    for num, echogram in enumerate(echograms):
        if echogram.n_objects > 0:

            print(num)

            plot_predictions_and_prediction_measures(
                echogram=echogram,
                model_1=model_1,
                model_2=model_2,
                path_model_params_1=path_model_params_1,
                path_model_params_2=path_model_params_2,
                div_std_1=False,
                div_std_2=True
            )
    #'''


    #hmap_file_name = 'hmap_echo_2_ws_52_0_res_5.npy'
    #hmap_save_path = os.path.join(os.getcwd(), hmap_file_name)

    #create_and_visualize_heatmap(model=Net3(), echogram=partition_data(get_echograms())[1][20], resolution=20)
    #save_heatmap(model=model, echogram=echogram, resolution=resolution, save_path=save_path)
    #ech.visualize(predictions=np.load(hmap_save_path), pred_contrast=5)

