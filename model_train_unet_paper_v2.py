import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.echogram import get_echograms
from batch.dataset import Dataset
from batch.samplers.background import Background
from batch.samplers.seabed import Seabed
from batch.samplers.shool import Shool
from batch.samplers.shool_seabed import ShoolSeabed
from batch.augmentation.flip_x_axis import flip_x_axis
from batch.augmentation.add_noise import add_noise
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf
from batch.data_transform_functions.db_with_limits import db_with_limits
from batch.label_transform_functions.index_0_1_27 import index_0_1_27
from batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from batch.combine_functions import CombineFunctions

from torch.utils.data import DataLoader
from utils.logger import TensorboardLogger
import models.unet_bn_sequential_db as models


# Partition data into train, test, val
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
        print("Parameter 'partition' must equal 'random' or 'year'")

    print('Train:', len(train), ' Test:', len(test), ' Val:', len(val))

    return train, test, val


def get_validation_set_paths(validation_set):
    return [ech.name for ech in validation_set[1]]


# Train model
def train_model(
        dev,
        model,
        window_dim,
        batch_size,
        lr,
        lr_reduction,
        momentum,
        test_iter,
        log_step,
        lr_step,
        iterations,
        path_model_params_load,
        path_model_params_save=None,
        partition='random',
        load_pre_trained=False,
        save_model_params=False
):
    '''

    :param dev: (device) Cuda device
    :param model: (torch.nn.module) Model
    :param window_dim: (int) Dimension of window size for training data
    :param batch_size: (int) Batch size
    :param lr: (float) Learning rate, general
    :param lr_reduction: (float) Learning rates are changed every 'log_step' number of training iteration by multiplication with this factor
    :param momentum: (float) Momentum factor in updating model parameters at each iteration
    :param test_iter: (int) Number of iterations in each validation
    :param log_step: (int) Number of training iterations between each validation (and subsequently logging of measured values)
    :param lr_step: (int) Number of training iterations between updating the learning rates
    :param iterations: (int) Total number of training iterations
    :param path_model_params_load: (str) Path to load pre-trained model parameters
    :param path_model_params_save: (str) Path to save model parameters after training
    :param partition: (str) Mode for how to partition the data into training/validation/test sets
    :param load_pre_trained: (bool) Initialize model with pre-trained parameters from file before training
    :param save_model_params: (bool) Save model parameters to file after training is complete
    :return: None
    '''

    frequencies = [18, 38, 120, 200]
    window_size = [window_dim, window_dim]

    # Load echograms and create partition for train/test/val
    echograms = get_echograms(frequencies=frequencies, minimum_shape=window_dim)
    echograms_train, echograms_test, echograms_val = partition_data(echograms, partition, portion_train=0.85)

    # Set device
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    print(device)

    model.to(device)

    if load_pre_trained:
        model.load_state_dict(torch.load(path_model_params_load, map_location=device))

    logger = TensorboardLogger('cogmar_test')

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([10, 300, 250]).to(device))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_reduction)
    scheduler.step()

    num_workers = 10
    print('num_workers: ', num_workers)

    samplers_train = [
        Background(echograms_train, window_size),
        Seabed(echograms_train, window_size),
        Shool(echograms_train, 27),
        Shool(echograms_train, 1),
        ShoolSeabed(echograms_train, window_dim // 2, 27),
        ShoolSeabed(echograms_train, window_dim // 2, 1)
    ]
    samplers_test = [
        Background(echograms_test, window_size),
        Seabed(echograms_test, window_size),
        Shool(echograms_test, 27),
        Shool(echograms_test, 1),
        ShoolSeabed(echograms_test, window_dim // 2, 27),
        ShoolSeabed(echograms_test, window_dim // 2, 1)
    ]

    sampler_probs = [1, 5, 5, 5, 5, 5]
    label_types = [1, 27]

    augmentation = CombineFunctions([add_noise, flip_x_axis])
    label_transform = CombineFunctions([index_0_1_27, relabel_with_threshold_morph_close])
    data_transform = CombineFunctions([remove_nan_inf, db_with_limits])

    dataset_train = Dataset(
        samplers_train,
        window_size,
        frequencies,
        batch_size * iterations,
        sampler_probs,
        augmentation_function=augmentation,
        label_transform_function=label_transform,
        data_transform_function=data_transform)

    dataset_test = Dataset(
        samplers_test,
        window_size,
        frequencies,
        batch_size * test_iter,
        sampler_probs,
        augmentation_function=None,
        label_transform_function=label_transform,
        data_transform_function=data_transform)

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed)
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=np.random.seed)

    running_loss_train = 0.0
    running_loss_test = 0.0

    # Train model
    for i, (inputs_train, labels_train) in enumerate(dataloader_train):

        # Load train data and transfer from numpy to pytorch
        inputs_train = inputs_train.float().to(device)
        labels_train = labels_train.long().to(device)

        # Forward + backward + optimize
        model.train()
        optimizer.zero_grad()
        outputs_train = model(inputs_train)
        loss_train = criterion(outputs_train, labels_train)
        loss_train.backward()
        optimizer.step()

        # Update loss count for train set
        running_loss_train += loss_train.item()

        # Log loss and accuracy
        if (i + 1) % log_step == 0:
            model.eval()
            with torch.no_grad():
                labels_true = []
                labels_correct = []
                labels_predict = []
                for inputs_test, labels_test in dataloader_test:

                    # Load test data and transfer from numpy to pytorch
                    inputs_test = inputs_test.float().to(device)
                    labels_test = labels_test.long().to(device)

                    # Evaluate test data
                    outputs_test = model(inputs_test)
                    loss_test = criterion(outputs_test, labels_test)

                    # Update loss count for test set
                    running_loss_test += loss_test.item()

                    predicted_classes_test = np.argmax(F.softmax(outputs_test, dim=1).cpu().numpy(), axis=1).reshape(-1)
                    labels_test = labels_test.cpu().numpy().reshape(-1)

                    # Add correctly predicted classes for calculating accuracy
                    labels_true += list(labels_test)
                    labels_correct += list(labels_test[predicted_classes_test == labels_test])
                    labels_predict += list(predicted_classes_test)

                labels_true = np.array(labels_true)
                labels_correct = np.array(labels_correct)
                labels_predict = np.array(labels_predict)

                confusion = np.zeros((1 + len(label_types), 1 + len(label_types)))
                for p in range(1 + len(label_types)):
                    for t in range(1 + len(label_types)):
                        confusion[p, t] = np.sum((labels_predict == p) & (labels_true == t))
                confusion = confusion + 1 # Avoid division by zero

                confusion_portion_of_true = confusion / np.sum(confusion, axis=0, keepdims=True)
                confusion_portion_of_pred = confusion / np.sum(confusion, axis=1, keepdims=True)

                #'''
                classes_all = np.array([np.sum(labels_true == c) for c in range(len([0] + label_types))], dtype=np.float32)
                classes_correct = np.array([np.sum(labels_correct == c) for c in range(len([0] + label_types))], dtype=np.float32)
                classes_all += 1e-10
                classes_accuracy = classes_correct / classes_all
                #'''

                # Save values to tensorboard logger
                logger.log_scalar('loss train', running_loss_train / log_step, i + 1)
                logger.log_scalar('loss test', running_loss_test / test_iter, i + 1)
                logger.log_scalar('accuracy background', classes_accuracy[0], i + 1)
                logger.log_scalar('accuracy sandeel', classes_accuracy[1], i + 1)
                logger.log_scalar('accuracy other', classes_accuracy[2], i + 1)

                print('{:>6} {:>2} {:4.3f} {:4.3f}'.format(
                    i + 1,
                    '  ',
                    running_loss_train / log_step,
                    running_loss_test / test_iter,
                ))
                np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=800)
                print(classes_accuracy)
                print(confusion_portion_of_true)
                print(confusion_portion_of_pred)

                # Reset counts to zero
                running_loss_train = 0.0
                running_loss_test = 0.0

        # Update learning rate every 'lr_step' number of batches
        if (i + 1) % lr_step == 0:
            print(i + 1)
            scheduler.step()
            print('lr:', [group['lr'] for group in optimizer.param_groups])

    print('Training complete')

    # Save model parameters to file after training
    if save_model_params:
        if path_model_params_save == None:
            path_model_params_save = path_model_params_load
        #torch.cuda.empty_cache()
        #torch.save(model.to('cpu').state_dict(), path_model_params)
        torch.save(model.state_dict(), path_model_params_save)
        print('Trained model parameters saved to file: ' + path_model_params_save)

    return None


if __name__ == '__main__':

    train_model(
        dev=2, # Set Cuda device number here, e.g. "dev=2".
        model=models.UNet(n_classes=3, in_channels=4),
        window_dim=256,
        batch_size=16,
        lr=0.01,
        lr_reduction=0.5,
        momentum=0.95,
        test_iter=20,
        log_step=100,
        lr_step=1000,
        iterations=10000,
        path_model_params_load="", # Set path to load pre-trained model params (only required if load_pre_trained==True).
        path_model_params_save="", # Insert path, e.g. "/nr/project/bild/Cogmar/usr/obr/model/paper_v2_heave_2.pt",
        partition='year',
        load_pre_trained=False,
        save_model_params=True,
    )