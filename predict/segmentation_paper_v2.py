import numpy as np
import torch
import random
import pickle
import os
#from matplotlib.font_manager import FontProperties

from models.unet_bn_sequential_db import UNet
from data.echogram import get_echograms
from batch.label_transform_functions.index_0_1_27 import index_0_1_27
from batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from batch.data_transform_functions.db_with_limits import db_with_limits

from data_preprocessing.generate_maskfromJson_python import get_korona_list_from_json
from data_preprocessing.generate_maskfromJson_python import get_korona_labels
from paths import path_to_korona_data
from predict._frameworks_Olav import get_prediction_function
import utils.plotting


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


def post_processing(seg, ech):

    """ Set all predictions below seabed to zero. """
    seabed = ech.get_seabed().copy()
    seabed += 10
    assert seabed.shape[0] == seg.shape[1]
    for x, y in enumerate(seabed):
        seg[y:, x] = 0
    return seg


def get_segmentation_sandeel(model, ech, freqs):

    patch_size = 256
    patch_overlap = 20

    data = ech.data_numpy(frequencies=freqs)
    data[np.invert(np.isfinite(data))] = 0

    # Get modified labels
    labels = ech.label_numpy()
    relabel = index_0_1_27(data, labels, ech)[1]
    relabel_morph_close = relabel_with_threshold_morph_close(np.moveaxis(data, -1, 0), relabel, ech)[1]
    relabel_morph_close[relabel_morph_close == -100] = -1

    data = db_with_limits(np.moveaxis(data, -1, 0), None, None, None)[0]
    data = np.moveaxis(data, 0, -1)

    # Get segmentation
    seg = segmentation(model, data, patch_size, patch_overlap, device)[:, :, 1]

    # Remove sandeel predictions 10 pixels below seabed and down
    seg = post_processing(seg, ech)

    return seg, relabel_morph_close


def get_extended_label_mask_for_echogram(ech, extend_size):

    fish_types = [1, 27]
    extension = np.array([-extend_size, extend_size, -extend_size, extend_size])
    eval_mask = np.zeros(shape=ech.shape, dtype=np.bool)

    for obj in ech.objects:

        obj_type = obj["fish_type_index"]
        if obj_type not in fish_types:
            continue
        bbox = np.array(obj["bounding_box"])

        # Extend bounding box
        bbox += extension

        # Crop extended bounding box if outside of echogram boundaries
        bbox[bbox < 0] = 0
        bbox[1] = np.minimum(bbox[1], ech.shape[0])
        bbox[3] = np.minimum(bbox[3], ech.shape[1])

        # Add extended bounding box to evaluation mask
        eval_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]] = True

    return eval_mask


def get_sandeel_probs_object_pathces(model, echs, freqs, eval_mode, n_echs, extend_size, korona_list=None):
    '''Get sandeel predictions for all labeled schools (sandeel, other) with surrounding region'''

    if korona_list is None:
        _korona_preds = None
    else:
        _korona_preds = {'tp': 0, 'fp': 0, 'fn': 0}

    _sandeel_probs = {0: [], 1: []}
    pixel_counts_year = np.zeros(3)

    for i, ech in enumerate(echs):

        if n_echs is not None:
            assert isinstance(n_echs, int)
            if i >= n_echs:
                break

        # Get pixel probability of sandeel, and ground truth labels (-1=ignore, 0=background, 1=sandeel, 2=other)
        seg, labels = get_segmentation_sandeel(model, ech, freqs)

        # Get Korona labels (values: 0 or 1)
        if korona_list is not None:
            korona_labels = get_korona_labels(ech, korona_list)
            assert korona_labels.shape == labels.shape

        # 'Region' evaluation mode: Exclude all pixels not in a neigborhood of labeled shool ('sandeel' or 'other')
        if eval_mode == 'region':
            # Get evaluation mask, i.e. the pixels to be evaluated
            eval_mask = get_extended_label_mask_for_echogram(ech, extend_size)
            # Set labels to -1 if not included in evaluation mask
            labels[eval_mask != True] = -1

        # 'Fish' evaluation mode: Set all background pixels to 'ignore', i.e. only evaluate the discrimination on species
        if eval_mode == 'fish':
            labels[labels == 0] = -1

        # Store sandeel predictions for positive labels ("sandeel")
        _sandeel_probs[1].extend(list(seg[labels == 1]))

        # Store sandeel predictions for negative labels ("other" and "background", background excluded outside of evaluation mask)
        _sandeel_probs[0].extend(list(seg[(labels == 0) | (labels == 2)]))

        # Store korona predictions (true positives, false positives, false negatives)
        if korona_list is not None:
            _korona_preds['tp'] += np.sum((korona_labels == 1) & (labels == 1))
            _korona_preds['fp'] += np.sum((korona_labels == 1) & ((labels == 0) | (labels == 2)))
            _korona_preds['fn'] += np.sum((korona_labels == 0) & (labels == 1))

        pixel_counts_year += np.array([np.sum(labels == 0), np.sum(labels == 1), np.sum(labels == 2)])

    # From {list, list} to {ndarray, ndarray}
    _sandeel_probs[0] = np.array(_sandeel_probs[0])
    _sandeel_probs[1] = np.array(_sandeel_probs[1])

    return _sandeel_probs, _korona_preds, pixel_counts_year


def print_f1_from_p_r(precision, recall, positives):

    precision_inv = 1 / precision - 1
    recall_inv = 1 / recall - 1
    tp = positives / (recall_inv + 1)
    fn = positives * recall_inv / (recall_inv + 1)
    fp = positives * precision_inv / (recall_inv + 1)

    tp_total = np.sum(tp)
    fn_total = np.sum(fn)
    fp_total = np.sum(fp)

    precision_total = tp_total / (tp_total + fp_total)
    recall_total = tp_total / (tp_total + fn_total)
    f1 = 2 * precision_total * recall_total / (precision_total + recall_total)

    print('F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}'.format(f1, precision_total, recall_total))


def get_sandeel_probs(model, echs, freqs, mode, n_echs):
    '''

    :param model:
    :param echs:
    :param freqs:
    :param mode: (str) "all" compares sandeel (pos) to other+background (neg). "fish" compares sandeel (pos) to other (neg).
    :param n_echs: (int) number of echograms (upper limit) per year
    :return:
    '''

    assert mode in ["all", "fish"]

    #random.shuffle(echs)
    _sandeel_probs = {0: [], 1: []}

    for i, ech in enumerate(echs):

        if i >= n_echs:
            break

        # Get binary segmentation (probability of sandeel) and labels (-1=ignore, 0=background, 1=sandeel, 2=other)
        seg, labels = get_segmentation_sandeel(model, ech, freqs)

        # Store sandeel predictions per pixel in list [0] (negatives) or [1] (positives) based on label.
        if mode == "all":
            # Negatives: Labels == "background" + "other" ("ignore" is excluded)
            _sandeel_probs[0].extend(list(seg[(labels == 0) | (labels == 2)]))
        elif mode == "fish":
            # Negatives: Labels == "other" ("background" + "ignore" is excluded)
            _sandeel_probs[0].extend(list(seg[labels == 2]))
        # Positives: Labels "sandeel"
        _sandeel_probs[1].extend(list(seg[labels == 1]))

    # From {list, list} to {ndarray, ndarray}
    _sandeel_probs[0] = np.array(_sandeel_probs[0])
    _sandeel_probs[1] = np.array(_sandeel_probs[1])

    return _sandeel_probs


def get_precision_and_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp != 0 else 1.0
    recall = tp / (tp + fn) if tp + fn != 0 else 1.0
    return precision, recall


def get_F1_model(precision, recall, threshold):

    assert precision.shape == recall.shape, 'Size mismatch: precision {}, recall {}'.format(precision.shape, recall.shape)
    assert precision.shape == threshold.shape, 'Size mismatch: precision {}, threshold {}'.format(precision.shape, threshold.shape)

    F1 = 2 * precision * recall / (precision + recall)
    # Set F1 to zero where precision and recall both equal to zero (appear as np.nan or np.inf in F1).
    F1[np.invert(np.isfinite(F1))] = 0
    idx = np.argmax(F1)
    out = {'F1': F1[idx], 'precision': precision[idx], 'recall': recall[idx], 'threshold': threshold[idx]}
    return out

def get_F1_korona(precision, recall):

    assert isinstance(precision, float)
    assert isinstance(recall, float)

    F1 = 2 * precision * recall / (precision + recall)
    out = {'F1': F1, 'precision': precision, 'recall': recall}
    return out


def get_pr_curve_and_thresholds(sandeel_probs, n_thresholds=200):

    # Get list of threshold values to compute p/r, adjusted to give evenly-ish distributed points on the p/r curve
    val_range = np.linspace(-20, 20, n_thresholds, endpoint=False)
    val_range = 1 / (1 + np.exp(-0.4 * (val_range + 3)))
    assert (np.min(val_range) >= 0) and (np.max(val_range) <= 1)

    pr_curve = []
    for value in val_range:

        tp = np.sum(sandeel_probs[1] >= value)
        fp = np.sum(sandeel_probs[0] >= value)
        fn = np.sum(sandeel_probs[1] < value)
        #tn = np.sum(sandeel_probs[0] < value)

        precision, recall = get_precision_and_recall(tp=tp, fp=fp, fn=fn)
        pr_curve.append([recall, precision, value])

    pr_curve = np.array(pr_curve)
    out = {'recall': pr_curve[:, 0], 'precision': pr_curve[:, 1], 'threshold': pr_curve[:, 2]}
    return out


def plot_echograms_with_sandeel_prediction(year, device, path_model_params, ignore_mode='normal'):

    # ignore_mode == 'normal': difference between original and modified labels are changed to 'ignore'
    # ignore_mode == 'region': in addition to 'normal' mode, label 'background' is changed to 'ignore' outside of region around labeled schools

    assert ignore_mode in ['normal', 'region']

    freqs = [18, 38, 120, 200]
    echograms_all = get_echograms(frequencies=freqs, minimum_shape=256)
    years_all = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
    echograms_year = {y: [ech for ech in echograms_all if ech.year == y] for y in years_all}
    echs = echograms_year[year]

    with torch.no_grad():

        model = UNet(n_classes=3, in_channels=4)
        model.to(device)
        model.load_state_dict(torch.load(path_model_params, map_location=device))
        model.eval()

        for i, ech in enumerate(echs):
            print(i, ech.name)

            # Get binary segmentation (probability of sandeel) and labels (-1=ignore, 0=background, 1=sandeel, 2=other)
            seg, labels = get_segmentation_sandeel(model, ech, freqs)

            if ignore_mode == 'region':
                # Get evaluation mask, i.e. the pixels to be evaluated
                eval_mask = get_extended_label_mask_for_echogram(ech, extend_size=20)
                # Set labels to -1 if not included in evaluation mask
                labels[eval_mask != True] = -1

            # Add two zero-channels to plot img as (R, G, B) = (p_sandeel, 0, 0)
            seg = np.expand_dims(seg, 2)
            seg = np.concatenate((seg, np.zeros((seg.shape[0], seg.shape[1], 2))), axis=2)

            # Visualize echogram with predictions
            ech.visualize(
                frequencies=[200],
                # frequencies=freqs,
                pred_contrast=5.0,
                # labels_original=relabel,
                labels_refined=labels,
                predictions=seg,
                draw_seabed=True,
                show_labels=False,
                show_object_labels=False,
                show_grid=False,
                show_name=False,
                show_freqs=True
            )


def plot_echograms_with_sandeel_prediction_and_korona_labels(year, device, path_model_params, dir_savefig, ignore_mode='normal'):

    # ignore_mode == 'normal': difference between original and modified labels are changed to 'ignore'
    # ignore_mode == 'region': in addition to 'normal' mode, label 'background' is changed to 'ignore' outside of region around labeled schools

    assert ignore_mode in ['normal', 'region']

    freqs = [18, 38, 120, 200]
    echograms_all = get_echograms(frequencies=freqs, minimum_shape=256)
    years_all = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
    years_korona = [2007, 2008, 2009, 2010, 2017, 2018]
    echograms_year = {y: [ech for ech in echograms_all if ech.year == y] for y in years_all}
    echs = echograms_year[year]
    random.shuffle(echs)

    do_korona = year in years_korona

    if do_korona:
        root_json_korona = path_to_korona_data()
        path_json_korona = root_json_korona + "korona_" + str(year) + ".json"
        korona_list = get_korona_list_from_json(path_json_korona=path_json_korona)

    with torch.no_grad():

        model = UNet(n_classes=3, in_channels=4)
        model.to(device)
        model.load_state_dict(torch.load(path_model_params, map_location=device))
        model.eval()

        for i, ech in enumerate(echs):

            if ech.name != '2017843-D20170509-T085907':
                continue

            n_1 = np.sum([obj['fish_type_index'] == 1 for obj in ech.objects])
            n_27 = np.sum([obj['fish_type_index'] == 27 for obj in ech.objects])
            n_other = np.sum([obj['fish_type_index'] not in [0, 1, 27] for obj in ech.objects])
            if (n_1 < 5) or (n_27 < 5) or (n_other < 3):
                continue

            print(i, ech.name)

            # Get binary segmentation (probability of sandeel) and labels (-1=ignore, 0=background, 1=sandeel, 2=other)
            seg, labels = get_segmentation_sandeel(model, ech, freqs)

            if ignore_mode == 'region':
                # Get evaluation mask, i.e. the pixels to be evaluated
                eval_mask = get_extended_label_mask_for_echogram(ech, extend_size=20)
                # Set labels to -1 if not included in evaluation mask
                labels[eval_mask != True] = -1

            # Add two zero-channels to plot img as (R, G, B) = (p_sandeel, 0, 0)
            seg = np.expand_dims(seg, 2)
            seg = np.concatenate((seg, np.zeros((seg.shape[0], seg.shape[1], 2))), axis=2)

            if do_korona:
                labels_korona = get_korona_labels(echogram=ech, korona_list=korona_list)
            else:
                labels_korona = None

            labels_original = index_0_1_27(None, ech.label_numpy(), None)[1]

            # Visualize echogram with predictions
            #dpi = 400
            dpi = 1000
            mm_to_inch = 1 / 25.4
            figsize_x = 170.0
            figsize_y = 1.2 * figsize_x
            name_savefig = dir_savefig + 'echogram_1000dpi'
            fig = plt.figure(figsize=(mm_to_inch * figsize_x, mm_to_inch * figsize_y), dpi=dpi)
            plt.tight_layout()
            ech.visualize(
                frequencies=freqs,
                pred_contrast=5.0,
                labels_original=labels_original,
                labels_korona=labels_korona,
                labels_refined=labels,
                predictions=seg,
                draw_seabed=True,
                show_labels=True,
                show_object_labels=False,
                show_grid=True,
                show_name=False,
                show_freqs=True,
                return_fig=True,
                paper_print=True,
                figure=fig
            )

            plt.gcf()
            plt.savefig(fname=name_savefig + '.png', dpi=dpi)
            plt.show()


def plot_pr_curves(device, path_model_params, dir_savefig, eval_mode, n_max_echs_per_year=None):

    n_echs_print = 'All' if (n_max_echs_per_year == None) else str(n_max_echs_per_year)
    print('\nNumber of echograms per year: ' + n_echs_print)
    print('Evaluation mode: ', eval_mode)

    assert eval_mode in ['all', 'fish', 'region']
    assert os.path.isdir(dir_savefig)

    freqs = [18, 38, 120, 200]
    echograms_all = get_echograms(frequencies=freqs, minimum_shape=256)

    years_test = [2007, 2008, 2009, 2010, 2017, 2018]
    years_all = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016, 2017, 2018]
    years_korona = [2007, 2008, 2009, 2010, 2017, 2018]
    assert set(years_test) <= set(years_all)
    assert set(years_korona) <= set(years_all)

    echograms_year = {y: [ech for ech in echograms_all if ech.year == y] for y in years_all}

    color_year = dict(zip(
        years_all,
        ["blue", "blue", "blue", "blue", "red", "red", "red", "red", "red", "blue", "blue"])
    )

    # Initialize plot
    dpi = 400
    mm_to_inch = 1 / 25.4
    figsize_x = 170.0
    figsize_y = 0.8 * figsize_x
    fig = plt.figure(figsize=(mm_to_inch * figsize_x, mm_to_inch * figsize_y), dpi=dpi)

    with torch.no_grad():

        model = UNet(n_classes=3, in_channels=4)
        model.to(device)
        model.load_state_dict(torch.load(path_model_params, map_location=device))
        model.eval()

        pixel_counts = np.zeros((len(years_all), 3))

        sandeel_probs_test_years = {0: [], 1:[]}
        model_f1 = {}
        korona_f1 = {}

        for j, year in enumerate(years_all):

            print(year)

            # Set boolean variable for Korona evaluation for current year
            do_korona = year in years_korona

            # Get automated labels from LSSS (Korona)
            if do_korona:
                root_json_korona = path_to_korona_data()
                path_json_korona = root_json_korona + "korona_" + str(year) + ".json"
                korona_list = get_korona_list_from_json(path_json_korona=path_json_korona)
            else:
                korona_list = None

            echs = echograms_year[year]
            assert np.all([e.year == year for e in echs])
            if n_max_echs_per_year is not None:
                random.shuffle(echs)

            # Get sandeel probabilities for all echograms
            # sandeel_probs = get_sandeel_probs(model, echs, freqs, mode="all", n_echs=n_ech_per_year)
            sandeel_probs, korona_preds, pixel_counts_year = \
                get_sandeel_probs_object_pathces(model, echs, freqs, eval_mode=eval_mode, n_echs=n_max_echs_per_year, extend_size=20, korona_list=korona_list)

            if year in years_test:
                sandeel_probs_test_years[0].extend(list(sandeel_probs[0]))
                sandeel_probs_test_years[1].extend(list(sandeel_probs[1]))

            if do_korona:
                sandeel_probs_test_years[0].extend(list(sandeel_probs[0]))
                sandeel_probs_test_years[1].extend(list(sandeel_probs[1]))

            pixel_counts[j, :] = pixel_counts_year

            # Compute model precision/recall values
            pr_curve_and_thresholds_model = get_pr_curve_and_thresholds(sandeel_probs, n_thresholds=200)
            model_precision = pr_curve_and_thresholds_model['precision']
            model_recall = pr_curve_and_thresholds_model['recall']
            model_threshold = pr_curve_and_thresholds_model['threshold']

            # Compute F1 score: model
            model_f1[year] = get_F1_model(precision=model_precision, recall=model_recall, threshold=model_threshold)

            # Compute precision/recall point from Korona labels
            if do_korona:
                korona_precision, korona_recall = \
                    get_precision_and_recall(tp=korona_preds['tp'], fp=korona_preds['fp'], fn=korona_preds['fn'])
                #korona_pr.append([year, korona_recall, korona_precision])
                korona_f1[year] = get_F1_korona(precision=korona_precision, recall=korona_recall)

            # Plot
            ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ticks_empty = [''] * len(ticks)
            ax = fig.add_subplot(3, 4, 1 + j, xlim=(0, 1), ylim=(0, 1))
            ax.set_title(year, fontsize=8)  # , pad=5)
            if j % 4 == 0:
                ax.set_ylabel("Precision", fontsize=8)  # , labelpad=-25)
                ax.set_yticks(ticks)
            else:
                ax.set_yticklabels(ticks_empty)
            if 1 + j >= 8:
                ax.set_xlabel("Recall", fontsize=8)  # , labelpad=-20)
                ax.set_xticks(ticks)
            else:
                ax.set_xticklabels(ticks_empty)
            #if 1 + j == 8:
            #    ax.set_xticks(ticks)
            ax.tick_params(labelsize=6)
            ax.scatter(model_recall, model_precision, s=2, c=color_year[year])
            if do_korona:
                ax.scatter(korona_recall, korona_precision, s=3, c='green', marker="D")

        # From {list, list} to {ndarray, ndarray}
        sandeel_probs_test_years[0] = np.array(sandeel_probs_test_years[0])
        sandeel_probs_test_years[1] = np.array(sandeel_probs_test_years[1])

        # Compute precision/recall for test years
        pr_curve_and_thresholds_model_test_years = get_pr_curve_and_thresholds(sandeel_probs_test_years, n_thresholds=200)
        model_f1_test_years = get_F1_model(
            precision=pr_curve_and_thresholds_model_test_years['precision'],
            recall=pr_curve_and_thresholds_model_test_years['recall'],
            threshold=pr_curve_and_thresholds_model_test_years['threshold']
        )

        # Print model F1, pr, thresholds for test years
        print('\n### Model - Total on all test years {} ###'.format(years_test))
        print('F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, Threshold: {:.3f}'.format(
            model_f1_test_years['F1'],
            model_f1_test_years['precision'],
            model_f1_test_years['recall'],
            model_f1_test_years['threshold']))

        # Print model F1/precision/recall/threshold per year
        print('\n### Model ###')
        for year in years_all:
            print(year, 'F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, Threshold: {:.3f}'.format(
                model_f1[year]['F1'],
                model_f1[year]['precision'],
                model_f1[year]['recall'],
                model_f1[year]['threshold']
            ))

        # Print Korona F1/precision/recall per year
        print('\n### Korona ###')
        for year in years_korona:
            print(year, 'F1: {:.3f}, Precision: {:.3f}, Recall: {:.3f}'.format(
                korona_f1[year]['F1'],
                korona_f1[year]['precision'],
                korona_f1[year]['recall']
            ))

        print('\n### Pixel count all years ###')
        print(np.sum(pixel_counts, axis=0))

        # Print pixel count statistics
        print('\n### Pixel count per year ###')
        print(pixel_counts)

        print('\n### Pixel distribution all years ###')
        print(np.sum(pixel_counts, axis=0) / np.sum(pixel_counts))

        print('\n### Pixel distribution per year ###')
        print(pixel_counts / np.sum(pixel_counts, axis=1, keepdims=True))

        plt.tight_layout()
        name_savefig = dir_savefig + 'pr_' + eval_mode
        plt.savefig(fname=name_savefig + '.png', dpi=dpi)
        with open(name_savefig + '.pkl', "wb") as file:
            pickle.dump(fig, file)
        plt.show()


def plot_patches(dir_savefig):

    import matplotlib.colors as mcolors
    from batch.data_transform_functions.db_with_limits import db_with_limits

    color_seabed = {'seabed': 'white'}
    lw = {'seabed': 0.4}
    cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
    boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
    norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

    echs = get_echograms()
    random.shuffle(echs)

    for ech in echs:

        if ech.n_objects == 0:
         continue
        print(ech.name)
        data = ech.data_numpy(frequencies=[18, 38, 120, 200])
        data = np.moveaxis(data, 2, 0)
        data[np.invert(np.isfinite(data))] = 0
        labels_original = index_0_1_27(None, ech.label_numpy(), None)[1]
        labels_refined = relabel_with_threshold_morph_close(data, labels_original, ech)[1]
        data = db_with_limits(data, None, None, None)[0]

        for obj in ech.objects:

            if obj['fish_type_index'] != 27:
                continue

            bbox = obj['bounding_box']
            y_0 = bbox[0]
            y_1 = bbox[1]
            x_0 = bbox[2]
            x_1 = bbox[3]
            n = 20

            if obj['n_pixels'] < 500:
                continue

            patch_data = data[3, (y_0 - n):(y_1 + n), (x_0 - n):(x_1 + n)]
            patch_labels_original = labels_original[(y_0 - n):(y_1 + n), (x_0 - n):(x_1 + n)]
            patch_labels_refined = labels_refined[(y_0 - n):(y_1 + n), (x_0 - n):(x_1 + n)]

            # Visualize echogram with predictions
            dpi = 400
            mm_to_inch = 1 / 25.4
            figsize_x = 170.0 / 2
            figsize_y = 1.2 * figsize_x
            name_savefig = dir_savefig + 'modified_annotations'

            plt.figure(figsize=(mm_to_inch * figsize_x, mm_to_inch * figsize_y), dpi=dpi)
            plt.tight_layout()

            plt.subplot(3, 1, 1)
            plt.imshow(patch_data, cmap='jet', aspect='auto')
            plt.title('200 kHz', fontsize=8, pad=2)
            plt.axis('off')

            plt.subplot(3, 1, 2)
            plt.imshow(patch_labels_original, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            plt.title('Annotations (original)', fontsize=8, pad=2)
            plt.axis('off')

            plt.subplot(3, 1, 3)
            plt.imshow(patch_labels_refined, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            plt.title('Annotations (modified)', fontsize=8, pad=2)
            plt.axis('off')

            plt.savefig(fname=name_savefig + '.png', dpi=dpi)
            plt.show()


def plot_area_pr_evaluation(device, path_model_params, dir_savefig):

    import matplotlib.colors as mcolors
    from batch.data_transform_functions.db_with_limits import db_with_limits

    cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
    boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
    norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

    echs = get_echograms()
    random.shuffle(echs)

    with torch.no_grad():
        model = UNet(n_classes=3, in_channels=4)
        model.to(device)
        model.load_state_dict(torch.load(path_model_params, map_location=device))
        model.eval()

        for ech in echs:

            if ech.n_objects == 0:
             continue
            print(ech.name)
            data = ech.data_numpy(frequencies=[18, 38, 120, 200])
            data = np.moveaxis(data, 2, 0)
            data[np.invert(np.isfinite(data))] = 0
            labels_original = index_0_1_27(None, ech.label_numpy(), None)[1]
            labels_refined = relabel_with_threshold_morph_close(data, labels_original, ech)[1]
            data = db_with_limits(data, None, None, None)[0]
            eval_mask = get_extended_label_mask_for_echogram(ech, 20)
            labels_refined[eval_mask != True] = -1

            seg = get_segmentation_sandeel(model, ech, [18, 38, 120, 200])[0]
            print(seg.shape)

            for obj in ech.objects:

                if obj['fish_type_index'] != 27:
                    continue

                bbox = obj['bounding_box']
                y_0 = bbox[0]
                y_1 = bbox[1]
                x_0 = bbox[2]
                x_1 = bbox[3]
                n_y = 40
                n_x = 200

                if obj['n_pixels'] < 200:
                    continue

                patch_data = data[3, (y_0 - n_y):(y_1 + n_y), (x_0 - n_x):(x_1 + n_x)]
                patch_labels_refined = labels_refined[(y_0 - n_y):(y_1 + n_y), (x_0 - n_x):(x_1 + n_x)]
                patch_seg = seg[(y_0 - n_y):(y_1 + n_y), (x_0 - n_x):(x_1 + n_x)]

                # Visualize echogram with predictions
                dpi = 400
                mm_to_inch = 1 / 25.4
                figsize_x = 170.0
                figsize_y = 0.5 * figsize_x
                name_savefig = dir_savefig + 'area_pr_evaluation'

                plt.figure(figsize=(mm_to_inch * figsize_x, mm_to_inch * figsize_y), dpi=dpi)
                plt.tight_layout()

                plt.subplot(3, 1, 1)
                plt.imshow(patch_data, cmap='jet', aspect='auto')
                plt.title('200 kHz', fontsize=8, pad=2)
                plt.axis('off')

                plt.subplot(3, 1, 2)
                plt.imshow(patch_labels_refined, aspect='auto', cmap=cmap_labels, norm=norm_labels)
                plt.title('Annotations (modified)', fontsize=8, pad=2)
                plt.axis('off')

                plt.subplot(3, 1, 3)
                plt.imshow(patch_seg, aspect='auto', cmap=cmap_labels, norm=norm_labels)
                plt.title('Predictions', fontsize=8, pad=2)
                plt.axis('off')

                #plt.tight_layout()
                plt.savefig(fname=name_savefig + '.png', dpi=dpi)
                plt.show()


def plot_missing_annotations(device, path_model_params, dir_savefig):

    import matplotlib.colors as mcolors
    from batch.data_transform_functions.db_with_limits import db_with_limits

    cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
    boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
    norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

    echs = get_echograms()
    random.shuffle(echs)

    with torch.no_grad():
        model = UNet(n_classes=3, in_channels=4)
        model.to(device)
        model.load_state_dict(torch.load(path_model_params, map_location=device))
        model.eval()

        for ech in echs:

            if ech.name != '2011206-D20110501-T112240':
                continue

            if ech.n_objects == 0:
             continue
            print(ech.name)
            data = ech.data_numpy(frequencies=[18, 38, 120, 200])
            data = np.moveaxis(data, 2, 0)
            data[np.invert(np.isfinite(data))] = 0
            labels_original = index_0_1_27(None, ech.label_numpy(), None)[1]
            labels_refined = relabel_with_threshold_morph_close(data, labels_original, ech)[1]
            data = db_with_limits(data, None, None, None)[0]

            seg = get_segmentation_sandeel(model, ech, [18, 38, 120, 200])[0]
            print(seg.shape)

            for obj in ech.objects:

                if obj['fish_type_index'] != 27:
                    continue

                bbox = obj['bounding_box']
                y_0 = bbox[0]
                y_1 = bbox[1]
                x_0 = bbox[2]
                x_1 = bbox[3]
                n_y = 50
                n_x = 1000

                #if obj['n_pixels'] < 200:
                #    continue

                patch_data = data[3, (y_0 - n_y):(y_1 + n_y), (x_0 - 5):(x_1 + n_x)]
                patch_labels_refined = labels_refined[(y_0 - n_y):(y_1 + n_y), (x_0 - 5):(x_1 + n_x)]
                patch_seg = seg[(y_0 - n_y):(y_1 + n_y), (x_0 - 5):(x_1 + n_x)]

                if not np.all(patch_labels_refined[:, patch_labels_refined.shape[1]//2:] == 0):
                    continue

                # Visualize echogram with predictions
                dpi = 400
                mm_to_inch = 1 / 25.4
                figsize_x = 160.0 / 2
                figsize_y = 0.8 * figsize_x
                name_savefig = dir_savefig + 'missing_annotations'

                plt.figure(figsize=(mm_to_inch * figsize_x, mm_to_inch * figsize_y), dpi=dpi)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

                plt.subplot(3, 1, 1)
                plt.imshow(patch_data, cmap='jet', aspect='auto')
                plt.title('200 kHz', fontsize=8, pad=2)
                plt.axis('off')

                plt.subplot(3, 1, 2)
                plt.imshow(patch_labels_refined, aspect='auto', cmap=cmap_labels, norm=norm_labels)
                plt.title('Annotations (modified)', fontsize=8, pad=2)
                plt.axis('off')

                plt.subplot(3, 1, 3)
                plt.imshow(patch_seg, aspect='auto', cmap=cmap_labels, norm=norm_labels)
                plt.title('Predictions', fontsize=8, pad=2)
                plt.axis('off')

                plt.savefig(fname=name_savefig + '.png', dpi=dpi)
                plt.show()

                break


def plot_false_positives(device, path_model_params, dir_savefig):

    import matplotlib.colors as mcolors
    from batch.data_transform_functions.db_with_limits import db_with_limits

    cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
    boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
    norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

    echs = get_echograms()
    random.shuffle(echs)

    with torch.no_grad():
        model = UNet(n_classes=3, in_channels=4)
        model.to(device)
        model.load_state_dict(torch.load(path_model_params, map_location=device))
        model.eval()

        for ech in echs:

            if ech.name != '2015837-D20150428-T170615':
                pass

            print(ech.name)
            data = ech.data_numpy(frequencies=[18, 38, 120, 200])
            data = np.moveaxis(data, 2, 0)
            data[np.invert(np.isfinite(data))] = 0
            labels_original = index_0_1_27(None, ech.label_numpy(), None)[1]
            labels_refined = relabel_with_threshold_morph_close(data, labels_original, ech)[1]
            data = db_with_limits(data, None, None, None)[0]

            seg = get_segmentation_sandeel(model, ech, [18, 38, 120, 200])[0]
            print(seg.shape)


            y_0 = 0
            y_1 = 100
            x_0 = 0
            x_1 = 1000

            patch_data = data[3, y_0:y_1, x_0:x_1]
            patch_labels_refined = labels_refined[y_0:y_1, x_0:x_1]
            patch_seg = seg[y_0:y_1, x_0:x_1]

            if not np.all(patch_labels_refined == 0):
                continue

            if not np.mean(patch_seg) > 0.01:
                continue

            # Visualize echogram with predictions
            dpi = 400
            mm_to_inch = 1 / 25.4
            figsize_x = 160.0 / 2
            figsize_y = 0.8 * figsize_x
            name_savefig = dir_savefig + 'false_positives'

            plt.figure(figsize=(mm_to_inch * figsize_x, mm_to_inch * figsize_y), dpi=dpi)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

            plt.subplot(3, 1, 1)
            plt.imshow(patch_data, cmap='jet', aspect='auto')
            plt.title('200 kHz', fontsize=8, pad=2)
            plt.axis('off')

            plt.subplot(3, 1, 2)
            plt.imshow(patch_labels_refined, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            plt.title('Annotations (modified)', fontsize=8, pad=2)
            plt.axis('off')

            plt.subplot(3, 1, 3)
            plt.imshow(patch_seg, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            plt.title('Predictions', fontsize=8, pad=2)
            plt.axis('off')

            plt.savefig(fname=name_savefig + '.png', dpi=dpi)
            plt.show()



if __name__ == "__main__":

    plt = utils.plotting.setup_matplotlib()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    path_model_params = '' # Insert path
    dir_savefig = '' # Insert path
    eval_mode = 'region'

    #plot_false_positives(device=device, path_model_params=path_model_params, dir_savefig=dir_savefig)
    #plot_missing_annotations(device=device, path_model_params=path_model_params, dir_savefig=dir_savefig)
    #plot_area_pr_evaluation(device=device, path_model_params=path_model_params, dir_savefig=dir_savefig)
    #plot_patches(dir_savefig=dir_savefig)
    #plot_echograms_with_sandeel_prediction_and_korona_labels(2017, device, path_model_params, dir_savefig=dir_savefig)
    #plot_pr_curves(device=device, path_model_params=path_model_params, dir_savefig=dir_savefig, eval_mode=eval_mode, n_max_echs_per_year=None)
