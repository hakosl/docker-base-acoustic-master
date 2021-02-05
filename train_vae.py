import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import argparse
import matplotlib.pyplot as plt
import time
import itertools
import os

from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

from torch import nn
from torch.utils.data import DataLoader

from data.echogram import get_echograms
from batch.dataset import Dataset
from batch.samplers import Background, Seabed, Shool, ShoolSeabed
from batch.augmentation.flip_x_axis import flip_x_axis
from batch.augmentation.add_noise import add_noise
from batch.data_transform_functions.log import log
from batch.data_transform_functions.remove_nan_inf import remove_nan_inf
#from batch.data_transform_functions.db_with_limits import db_with_limits
from batch.data_transform_functions.db_ import db
from batch.data_transform_functions.db_with_limits_norm import db_with_limits_norm
from batch.label_transform_functions.index_0_1_27 import index_0_1_27
from batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from batch.combine_functions import CombineFunctions
from vae_model import VariationalAutoencoder, vae_loss, datapVAE, vae_models
from visualization.vae_loss_vis import vae_loss_visualization
from utils.logger import TensorboardLogger
import models.unet_bn_sequential_db as models


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='config/vae_train.yaml')

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


def validate_clustering(model, cm, clustering_params, dataloader_train, dataloader_test, samplers_train, samplers_test, device, capacity, vb, fig_path="output/clustering.png", n_visualize=250):
    enc = model.encoder

    
    latent_mus = []
    latent_logvars = []
    sample_indexes = []

    for i, (inputs_train, labels_train, si) in itertools.islice(enumerate(dataloader_train), clustering_params["n_batches"]):    
        inputs_train = inputs_train.float().to(device)
        labels_train = labels_train.long().to(device)
    
        latent_mu, latent_logvar = enc(inputs_train)
        latent_mus.append(latent_mu.data.cpu().numpy())
        latent_logvars.append(latent_logvar.data.cpu().numpy())
        sample_indexes.append(si.data.cpu().numpy())


    

    latent_mus = np.array(latent_mus).reshape((-1, capacity))
    latent_logvars = np.array(latent_logvars).reshape((-1, capacity))
    sample_indexes = np.array(sample_indexes).reshape(-1)

    #me = PCA(n_components=3, random_state = 42).fit_transform(latent_mus)
    clusterer = cm()
    clusterer.fit(latent_mus)
    best_labels = clusterer.labels_

    best_r_score = adjusted_rand_score(best_labels, sample_indexes)

    me = PCA(n_components=2, random_state = 42).fit_transform(latent_mus)   

    fig, ax = plt.subplots(1, 2, figsize=(9, 5))

    ax[0].scatter(me[:n_visualize][:, 0], me[:n_visualize][:, 1], c=best_labels[:n_visualize])
    ax[0].set_title("DBSCAN clusters")

    colors = ["r", "g", "b", "tab:orange", "purple", "cyan"]
    for si in np.unique(sample_indexes[:n_visualize]):
        sm = sample_indexes[:n_visualize] == si
        ax[1].scatter(me[:n_visualize][sm, 0], me[:n_visualize][sm, 1], alpha=0.4, c=colors[si], label=str(samplers_train[:n_visualize][si]))
        ax[1].set_title("original labels")

    ax[1].legend()
    fig.suptitle(f"cap: {capacity} beta: {vb} r_score: {best_r_score}")
    fig.savefig(fig_path)
    plt.close(fig)

    X_train, X_val, y_train, y_val = train_test_split(latent_mus, sample_indexes, test_size=0.2)

    clf = LogisticRegression(random_state=0, multi_class="auto")
    #clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    clf.fit(X_train, y_train)

    clf_predictions = clf.predict(X_val)
    clf_acc = accuracy_score(y_val, clf_predictions)

    print(f"classifier accuracy: {clf_acc}")


    return best_r_score





# Train model
def train_model(
        model,
        clustering_params,
        dev=0,
        window_dim=64,
        batch_size=36,
        lr=0.0001,
        log_step=100,
        iterations=10000,
        path_model_params_load="/acoustic/vae_trained.pt",
        verbose=True,
        frequencies=[38],
        variational_beta=1.0,
        path_model_params_save=None,
        partition='year',
        load_pre_trained=False,
        save_model_params=False,
        recon_criterion="MSE",
        base_figure_dir="output/",
        num_workers=1
        
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

    window_size = [window_dim, window_dim]

    # Load echograms and create partition for train/test/val
    echograms = get_echograms(frequencies=frequencies, minimum_shape=window_dim)

    echograms_train, echograms_test, echograms_val = partition_data(echograms, partition, portion_train=0.85)

    # Set device
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"device: {device}")

    model.to(device)
    if verbose:
        print(model)
    if load_pre_trained:
        model.load_state_dict(torch.load(path_model_params_load, map_location=device))

    logger = TensorboardLogger('cogmar_test')

    criterion = vae_loss
    optimizer = optim.Adam(model.parameters())

    num_workers = 0 
    if verbose:
        print('num_workers: ', num_workers)

    sample_options = {
        "window_size": window_size,
        "random_sample": False,
    }

    sample_options_train = {
        "echograms": echograms_train,
        **sample_options
    }
    sample_options_test = {
        "echograms": echograms_test,
        **sample_options
    }

    
    sampler_probs = [1, 5, 5, 5, 5, 5]
    label_types = [1, 27]

    samplers_train = [
        Background(**sample_options_train, sample_probs=sampler_probs[0]),
        Seabed(**sample_options_train, sample_probs=sampler_probs[1]),
        Shool(**sample_options_train, fish_type=27, sample_probs=sampler_probs[2]),
        Shool(**sample_options_train, fish_type=1, sample_probs=sampler_probs[3]),    
        ShoolSeabed(**sample_options_train, fish_type=27, sample_probs=sampler_probs[4]),
        ShoolSeabed(**sample_options_train, fish_type=1, sample_probs=sampler_probs[5])
    ]
    samplers_test = [
        Background(**sample_options_test, sample_probs=sampler_probs[0]),
        Seabed(**sample_options_test, sample_probs=sampler_probs[1]),
        Shool(**sample_options_test, fish_type=27, sample_probs=sampler_probs[2]),
        Shool(**sample_options_test, fish_type=1, sample_probs=sampler_probs[3]),    
        ShoolSeabed(**sample_options_test, fish_type=27, sample_probs=sampler_probs[4]),
        ShoolSeabed(**sample_options_test, fish_type=1, sample_probs=sampler_probs[5])
    ]


    augmentation = CombineFunctions([])
    label_transform = CombineFunctions([index_0_1_27, relabel_with_threshold_morph_close])
    data_transform = CombineFunctions([remove_nan_inf, db_with_limits_norm])

    transform_functions = {
        "augmentation_function": augmentation,
        "label_transform_function": label_transform,
        "data_transform_function": data_transform
    }

    dataset_arguments = [window_size, frequencies, batch_size * iterations]

    dataset_train = Dataset(samplers_train, *dataset_arguments, si=True, **transform_functions, include_depthmap=True)
    dataset_test = Dataset(samplers_test, *dataset_arguments, si=True, **transform_functions, include_depthmap=True)

    dataloader_arguments = {
        "batch_size": batch_size, 
        "shuffle": False, 
        "num_workers":num_workers, 
        "worker_init_fn": np.random.seed
    }

    dataloader_train = DataLoader(dataset_train, **dataloader_arguments)
    dataloader_test = DataLoader(dataset_test, **dataloader_arguments)

    start_time = time.time()


    running_loss_train = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    running_loss_test = 0.0

    
    # Train model
    losses = []
    recon_losses = []
    kl_losses = []
    iteration = []
    for i, (inputs_train, labels_train, si) in enumerate(dataloader_train):
        # Load train data and transfer from numpy to pytorch
        #print(inputs_train)
        inputs_train = inputs_train.float().to(device)
        labels_train = labels_train.long().to(device)


        # Forward + backward + optimize
        model.train()
        optimizer.zero_grad()
        x_recon, mu, logvar = model(inputs_train)

        loss_train, recon_loss, kl_loss = criterion(
            x_recon, 
            inputs_train, 
            mu, 
            logvar, 
            variational_beta, 
            recon_criterion, 
            window_dim=window_dim,
            channels=model.channels)
        loss_train.backward()


        optimizer.step()

        # Update loss count for train set
        running_loss_train += loss_train.item()
        running_recon_loss += recon_loss.item()
        running_kl_loss += kl_loss.item()
        

        # Log loss and accuracy
        if (i + 1) % log_step == 0:
            current_time = time.time()

            delta = current_time - start_time
            est_remaining = (delta / i) * (iterations - i) 
            if verbose:
                print(
                    f"{'-'*5} " 
                    f"progress: {int(((i + 1) / iterations) * 100):3}% "
                    f"loss: {(running_loss_train / log_step):8.2e} "
                    f"recon: {running_recon_loss / log_step:8.2e} "
                    f"kl: {running_kl_loss / log_step:5.2e} "
                    f"time_elapsed: {delta:7.2f} "
                    f"remaining: {est_remaining:7.2f} "
                    f"{'-'*5}"
                )
            
            losses.append(running_loss_train / log_step)
            recon_losses.append(running_recon_loss / log_step)
            kl_losses.append(running_kl_loss / log_step)
            iteration.append(i)

            running_loss_train = 0.0       
            running_recon_loss = 0.0
            running_kl_loss = 0.0
            logger.log_scalar('loss train', running_loss_train / log_step, i + 1)
    if verbose:
        print()
        print('Training complete')

    # Save model parameters to file after training
    if save_model_params:
        if path_model_params_save == None:
            path_model_params_save = path_model_params_load
        #torch.cuda.empty_cache()
        #torch.save(model.to('cpu').state_dict(), path_model_params)
        torch.save(model.state_dict(), path_model_params_save)
        print('Trained model parameters saved to file: ' + path_model_params_save)


    os.makedirs(base_figure_dir, exist_ok=True)
    os.makedirs(f"{base_figure_dir}/loss_graph", exist_ok=True)
    os.makedirs(f"{base_figure_dir}/clustering", exist_ok=True)


    fig = vae_loss_visualization(iteration, losses, recon_losses, kl_losses)
    loss_fig_path = f"{base_figure_dir}/loss_graph/c{model.capacity}:b:{variational_beta}.png"
    fig.savefig(loss_fig_path)
    plt.close(fig)
    print(f"loss graph saved to {loss_fig_path}")


    fig_path = f"{base_figure_dir}/clustering/c{model.capacity}:b:{variational_beta}.png"
    best_r_score = validate_clustering(model, HDBSCAN, clustering_params, dataloader_train, dataloader_test, samplers_train, samplers_test, device, model.capacity, variational_beta, fig_path=fig_path)
    print(f"clustering figure saved to: {fig_path}")
    print(f"r_score: {best_r_score}")
    return best_r_score


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model_cfg = config["model"]
    data_params = config["data_params"]

    parameters_to_search = {"capacity": model_cfg["capacity"], "variational_beta": data_params["variational_beta"]}
    for current_param in ParameterGrid(parameters_to_search):
        np.random.seed(42)
        model_cfg = {**model_cfg, "capacity": current_param["capacity"]}
        path_model_params_save = f"/acoustic/vae_trained:c{current_param['capacity']}:b:{current_param['variational_beta']}.pt"
        data_params = {**data_params, "path_model_params_save": path_model_params_save, "variational_beta": current_param["variational_beta"]} 
        model_cfg["window_dim"] = config["data_params"]["window_dim"]
        model = vae_models[config["model_name"]](**model_cfg)

        r_score = train_model(
            model=model,
            **data_params
        )
        print(f"r score: {r_score} model_params: {current_param}")
   