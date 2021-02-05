import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from data.echogram import get_echograms
from batch.dataset import Dataset
from batch.samplers.background import Background
from batch.samplers.background_seabed import BackgroundSeabed
from batch.samplers.shool import Shool
from batch.samplers.shool_seabed import ShoolSeabed
from batch.augmentation.flip_x_axis import flip_x_axis
from batch.augmentation.add_noise import add_noise
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf
from batch.data_transform_functions.sigmoid_log import sigmoid_log
from batch.label_transform_functions.index import index
from batch.label_transform_functions.is_fish import is_fish
from batch.label_transform_functions.is_sandeel import is_sandeel
from batch.label_transform_functions.relabel_with_threshold import relabel_with_threshold
from batch.label_transform_functions.binary_classification import binary_classification
from batch.label_transform_functions.binary_classification import binary_classification_with_ignore_values
from batch.combine_functions import CombineFunctions

from torch.utils.data import DataLoader
from utils.logger import TensorboardLogger
import models.unet_bn_sequential as models


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
             ['D2008', 'D2009', 'D2010', 'D2011', 'D2012', 'D2013', 'D2014', 'D2015', '2016']]), echograms))
        test = list(filter(lambda x: any([year in x.name for year in ['D2007']]), echograms))
        val = list(filter(lambda x: any([year in x.name for year in ['D2017']]), echograms))

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
        path_model_params_dir_name,
        path_model_params_file_name,
        window_dim,
        batch_size,
        lr,
        lr_k,
        lr_a,
        lr_reduction,
        momentum,
        test_iter,
        log_step,
        lr_step,
        iterations,
        partition='random',
        load_pre_trained=False,
        save_model_params=False
):
    '''
    Train model for binary classification.

    :param dev: (device) Cuda device
    :param model: (torch.nn.module) Model
    :param path_model_params_dir_name: (str) Path to directory to save/load model parameters
    :param path_model_params_file_name: (str) Name of file to save/load model parameters
    :param window_dim: (int) Dimension of window size for training data
    :param batch_size: (int) Batch size
    :param lr: (float) Learning rate, general
    :param lr_k: (float) Learning rate for preprocessing parameters 'k'
    :param lr_a: (float) Learning rate for preprocessing parameters 'a'
    :param lr_reduction: (float) Learning rates are changed every 'log_step' number of training iteration by multiplication with this factor
    :param momentum: (float) Momentum factor in updating model parameters at each iteration
    :param test_iter: (int) Number of iterations in each validation
    :param log_step: (int) Number of training iterations between each validation (and subsequently logging of measured values)
    :param lr_step: (int) Number of training iterations between updating the learning rates
    :param iterations: (int) Total number of training iterations
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

    # Create path to save/load model parameters
    if not path_model_params_dir_name[-1] == '/':
        path_model_params_dir_name = path_model_params_dir_name + '/'
    path_model_params = path_model_params_dir_name + path_model_params_file_name

    if load_pre_trained:
        model_state_dict = torch.load(path_model_params)
        model.load_state_dict(model_state_dict)

    logger = TensorboardLogger('cogmar_test')

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 100]).to(device))

    # Define optimizer (with separate learning rates for preprocessing parameters 'k' and 'a')
    optimizer = optim.SGD(
        [
            {'params': model.preprocess.k, 'lr': lr_k, 'momentum': 0.95},
            {'params': model.preprocess.a, 'lr': lr_a, 'momentum': 0.95},
            {'params': [param for name, param in model.named_parameters() if 'preprocess' not in name]}
        ],
        lr=lr, momentum=momentum
    )
    print('lr:', [group['lr'] for group in optimizer.param_groups])

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_reduction)
    scheduler.step()

    num_workers = 10
    print('num_workers: ', num_workers)

    samplers_train = [
        Background(echograms_train, window_size),
        BackgroundSeabed(echograms_train, window_size),
        Shool(echograms_train, 'all'),
        ShoolSeabed(echograms_train, window_dim // 2, 'all')
    ]
    samplers_test = [
        Background(echograms_test, window_size),
        BackgroundSeabed(echograms_test, window_size),
        Shool(echograms_test, 'all'),
        ShoolSeabed(echograms_test, window_dim // 2, 'all')
    ]

    sampler_probs = [1, 1, 5, 5]

    augmentation = CombineFunctions([add_noise, flip_x_axis])
    data_transform = CombineFunctions([remove_nan_inf])
    label_transform = CombineFunctions([is_fish, relabel_with_threshold])

    dataset_train = Dataset(
        samplers_train,
        window_size,
        frequencies,
        batch_size * iterations,
        sampler_probs,
        augmentation_function=augmentation,
        data_transform_function=data_transform,
        label_transform_function=label_transform)

    dataset_test = Dataset(
        samplers_test,
        window_size,
        frequencies,
        batch_size * test_iter,
        sampler_probs,
        augmentation_function=None,
        data_transform_function=data_transform,
        label_transform_function=label_transform)

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

        # Forward + backward
        model.train()
        optimizer.zero_grad()
        outputs_train = model(inputs_train)
        loss_train = criterion(outputs_train, labels_train)
        loss_train.backward()

        # Optimize
        optimizer.step()

        # Update loss count for train set
        running_loss_train += loss_train.item()

        # Log loss and accuracy
        if (i + 1) % log_step == 0:
            model.eval()
            with torch.no_grad():
                for inputs_test, labels_test in dataloader_test:

                    # Load test data and transfer from numpy to pytorch
                    inputs_test = inputs_test.float().to(device)
                    labels_test = labels_test.long().to(device)

                    # Evaluate test data
                    outputs_test = model(inputs_test)
                    loss_test = criterion(outputs_test, labels_test)

                    # Update loss count for test set
                    running_loss_test += loss_test.item()

                logger.log_scalar('loss train', running_loss_train / log_step, i + 1)
                logger.log_scalar('loss test', running_loss_test / test_iter, i + 1)

                for j in range(4):
                    logger.log_scalar('preprocess weight k_' + str(j), model.preprocess.k.data[0, j, 0, 0].item(), i + 1)
                    logger.log_scalar('preprocess weight a_' + str(j), model.preprocess.a.data[0, j, 0, 0].item(), i + 1)

                print('{:>6} {:>2} {:4.3f} {:4.3f} [{:4.3e}, {:4.3e}, {:4.3e}, {:4.3e}] [{:4.3f}, {:4.3f}, {:4.3f}, {:4.3f}]'.format(
                    i + 1,
                    '  ',
                    running_loss_train / log_step,
                    running_loss_test / test_iter,
                    model.preprocess.k.data[0, 0, 0, 0].item(),
                    model.preprocess.k.data[0, 1, 0, 0].item(),
                    model.preprocess.k.data[0, 2, 0, 0].item(),
                    model.preprocess.k.data[0, 3, 0, 0].item(),
                    model.preprocess.a.data[0, 0, 0, 0].item(),
                    model.preprocess.a.data[0, 1, 0, 0].item(),
                    model.preprocess.a.data[0, 2, 0, 0].item(),
                    model.preprocess.a.data[0, 3, 0, 0].item(),
                ))

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
        torch.save(model.state_dict(), path_model_params)
        print('Trained model parameters saved to file: ' + path_model_params)

    return None


if __name__ == '__main__':

    train_model(
        dev='cuda:1',
        model=models.UNet(n_classes=2, in_channels=4),
        path_model_params_dir_name=, # (str) Path to directory to save/load model parameters
        path_model_params_file_name='unet_bn_sequential' + '_binary_fish_background' + '.pt', # (str) Name of file to save/load model parameters
        window_dim=256,
        batch_size=16,
        lr=0.01,
        lr_k=1.e-10,
        lr_a=1.e-05,
        lr_reduction=0.5,
        momentum=0.99,
        test_iter=10,
        log_step=50,
        lr_step=1000,
        iterations=5000,
        partition='random',
        load_pre_trained=False,
        save_model_params=False
    )