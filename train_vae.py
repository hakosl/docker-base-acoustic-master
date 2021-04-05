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
import ptvsd
import json

import seaborn
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from visualization.tile_image import tile_recon, tile_sampling
from visualization.make_grid import make_grid
from utils.data_utils import partition_data, get_validation_set_paths

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
from batch.data_transform_functions.db_with_limits_norm import db_with_limits_norm, db_with_limits_norm_MSE
from batch.label_transform_functions.index_0_1_27 import index_0_1_27
from batch.label_transform_functions.relabel_with_threshold_morph_close import relabel_with_threshold_morph_close
from batch.combine_functions import CombineFunctions
from models.vae_models import vae_models
from visualization.vae_loss_vis import vae_loss_visualization
import models.unet_bn_sequential_db as models
from visualization.plot_latent_space import plot_latent_space, plot_latent_space_gif
from visualization.validate_clustering import validate_clustering, compute_DCI, compute_mean_auc, label_efficiency
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import get_datasets

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='config/vae_train.yaml')

parser.add_argument("--debugger", "-d", 
    dest="debug", 
    action="store_true",
    help="run the program with debugger server running")

def save_recon(
    model, 
    inputs_train, 
    x_recon, 
    labels_train, 
    dev, 
    base_figure_dir, 
    recon_criterion, 
    variational_beta,
    i,
    writer
    ):
    samples = model.sample(3, dev)

    img = tile_sampling(samples.detach(), 3)
    writer.add_image("samples", img, i, dataformats="HW")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img, aspect="auto", cmap="gray")
    recon_image_fn = f"{base_figure_dir}/samples/r{recon_criterion}:i:{i}c:{model.capacity}b:{variational_beta}.png"
    fig.suptitle(recon_image_fn)
    fig.savefig(recon_image_fn)
    plt.close(fig)

    img = tile_recon(inputs_train, x_recon, labels_train, 3)
    writer.add_image("reconstruction", img, i, dataformats="HW")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img, aspect="auto", cmap="gray")
    recon_image_fn = f"{base_figure_dir}/recon/r{recon_criterion}:i:{i}c:{model.capacity}b:{variational_beta}.png"
    fig.suptitle(recon_image_fn)
    fig.savefig(recon_image_fn)
    plt.close(fig)

def print_loss_save_latent(
    model, 
    inputs_test, 
    start_time, 
    i, 
    iterations, 
    base_figure_dir, 
    recon_criterion, 
    variational_beta, 
    running_loss_train, 
    running_recon_loss, 
    running_kl_loss, 
    log_step,
    losses,
    recon_losses,
    kl_losses,
    iteration,
    device,
    si_test,
    latent_image_fns,
    verbose,
    writer=None):
    x_recon_test, mu_test, logvar_test = model(inputs_test.float().to(device))
    latent_image_fn = f"{base_figure_dir}/latent/r{recon_criterion}:i:{i}c:{model.capacity}b:{variational_beta}.png"

    plot_latent_space(mu_test, logvar_test, si_test, latent_image_fn, writer)
    latent_image_fns.append(latent_image_fn)

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

    return delta
    
def label_nr_to_string(nr):
    labels = np.array(["background", "seabed", "other", "sandeel"])

    return labels[np.array(nr)] 
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
        num_workers=1,
        early_stopping=True,
        patience=200
        
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

    os.makedirs(base_figure_dir, exist_ok=True)
    os.makedirs(f"{base_figure_dir}/loss_graph", exist_ok=True)
    os.makedirs(f"{base_figure_dir}/clustering", exist_ok=True)
    os.makedirs(f"{base_figure_dir}/latent", exist_ok=True)
    os.makedirs(f"{base_figure_dir}/samples", exist_ok=True)
    os.makedirs(f"{base_figure_dir}/recon", exist_ok=True)
    os.makedirs(f"{base_figure_dir}/latent", exist_ok=True)
    window_size = [window_dim, window_dim]

    writer = SummaryWriter(comment=f"/ {model} variationalbeta {variational_beta}")
    # Set device
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"device: {device}")


    model.to(device)
    if verbose:
        print(model)
    if load_pre_trained:
        model.load_state_dict(torch.load(path_model_params_load, map_location=device))


    criterion = model.loss
    optimizer = optim.Adam(model.parameters())

    num_workers = 0 
    if verbose:
        print('num_workers: ', num_workers)

    dataloader_train, dataloader_test, dataloader_val, dataset_train, dataset_test, dataset_val, ehograms_train, echograms_test, echograms_val = get_datasets(
        frequencies=frequencies, 
        window_dim=window_dim, 
        partition=partition, 
        batch_size=batch_size, 
        iterations=iterations, 
        num_workers=num_workers,
        include_depthmap=True
    )


    data_transform = CombineFunctions([remove_nan_inf, db_with_limits_norm])

    
    _, (inputs_test, labels_test, si_test) = next(enumerate(dataloader_test))
    inputs_test = inputs_test.float().to(device)
    labels_test = labels_test.long().to(device)

    
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
    latent_image_fns = []

    best_val_loss = -1
    best_val_iteration = 0

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
        with_stack=True
        ) as profiler:
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
                recon_loss=recon_criterion, 
                window_dim=window_dim,
                channels=model.channels)
            loss_train.backward()


            optimizer.step()
            # Update loss count for train set
            writer.add_scalar("Loss/train", loss_train.item(), i)
            writer.add_scalar("Loss/train_kl", kl_loss.item(), i)
            writer.add_scalar("Loss/train_recon", recon_loss.item(), i)


            running_loss_train += loss_train.item()
            running_recon_loss += recon_loss.item()
            running_kl_loss += kl_loss.item()
            model.zero_grad()
            # Log loss and accuracy
            if (i) % (log_step*10) == 0:
                save_recon(model, inputs_train, x_recon, labels_train, dev, base_figure_dir, recon_criterion, variational_beta, i, writer)
                #best_r_score, cm, clf, clf_acc = validate_clustering(model, HDBSCAN(prediction_data=True), inputs_test, si_test, dataset_test.samplers, device, model.capacity, variational_beta, fig_path=None, i=i, save_plot=False, writer=writer)

                #writer.add_scalar("Accuracy/svm", clf_acc, i)
                #writer.add_scalar("R_score/HDBSCAN", best_r_score, i)
                explicitness = compute_mean_auc(model, dataloader_val)
                
                print(f"explicitness: {explicitness}")
                writer.add_scalar("f-stat/explicitness", explicitness, i)

                if best_val_loss < explicitness:
                    best_val_loss = explicitness
                    
                    best_val_iteration = i
                    # Save model parameters to file after training
                    if save_model_params:
                        if path_model_params_save == None:
                            path_model_params_save = path_model_params_load
                        #torch.cuda.empty_cache()
                        #torch.save(model.to('cpu').state_dict(), path_model_params)
                        torch.save(model.state_dict(), path_model_params_save)
                        print('Trained model parameters saved to file: ' + path_model_params_save)

                else:
                    save_recon(model, inputs_train, x_recon, labels_train, dev, base_figure_dir, recon_criterion, variational_beta, i, writer)
                    break

            if (i + 1) % log_step == 0:
                delta = print_loss_save_latent(model, inputs_test, start_time, i, iterations, base_figure_dir, recon_criterion, variational_beta, running_loss_train, running_recon_loss, running_kl_loss, log_step, losses, recon_losses, kl_losses, iteration, device, si_test, latent_image_fns, verbose)
                if best_val_loss > running_loss_train:
                    best_val_loss = running_loss_train
                    
                    best_val_iteration = i

                    # Save model parameters to file after training
                    if save_model_params:
                        if path_model_params_save == None:
                            path_model_params_save = path_model_params_load
                        #torch.cuda.empty_cache()
                        #torch.save(model.to('cpu').state_dict(), path_model_params)
                        torch.save(model.state_dict(), path_model_params_save)
                        print('Trained model parameters saved to file: ' + path_model_params_save)

                else:
                    if i > best_val_iteration + patience:
                        save_recon(model, inputs_train, x_recon, labels_train, dev, base_figure_dir, recon_criterion, variational_beta, i, writer)
                        break


                running_loss_train = 0.0       
                running_recon_loss = 0.0
                running_kl_loss = 0.0
                #profiler.step()
    if verbose:
        print()
        print('Training complete')


    fig = vae_loss_visualization(iteration, losses, recon_losses, kl_losses)
    loss_fig_path = f"{base_figure_dir}/loss_graph/r{recon_criterion}c{model.capacity}:b:{variational_beta}.png"
    fig.savefig(loss_fig_path)
    plt.close(fig)
    plot_latent_space_gif(latent_image_fns, f"{base_figure_dir}/latent/c{model.capacity}b:{variational_beta}.gif")
    print(f"loss graph saved to {loss_fig_path}", )

    modularity, explicitness, information, ind_modu, compactness = compute_DCI(model, dataloader_train, dataloader_val, dataloader_test, writer)
    n, accs = label_efficiency(model, dataloader_train, dataloader_val, dataloader_test)
    print(n, accs)

    fig_path = f"{base_figure_dir}/clustering/r{recon_criterion}c:{model.capacity}:b:{variational_beta}.png"
    best_r_score, cm, clf, clf_acc = validate_clustering(model, HDBSCAN(prediction_data=True), dataloader_val, dataloader_test, dataset_test.samplers, device, model.capacity, variational_beta, fig_path=fig_path, i=i, writer=writer)

    grid_fig_path = f"{base_figure_dir}/grid_r{recon_criterion}c:{model.capacity}:b:{variational_beta}.png"
    make_grid(echograms_test[5], model, cm, clf, device, data_transform, window_dim, path=grid_fig_path)

    print(f"clustering figure saved to: {fig_path}")
    print(f"r_score: {best_r_score}")

    writer.add_hparams({"model_name": model.__class__.__name__, "capacity": model.capacity, "variational_beta": variational_beta, "classifier": str(clf)}, {"svm_accuracy": clf_acc, "adjusted_rand_score": best_r_score, "loss_train": loss_train.item(), "explicitness": explicitness, "modularity": modularity, "information": information})
    writer.add_graph(model, inputs_test[:10])
    writer.add_embedding(model.encoder(inputs_test)[0], metadata=label_nr_to_string(si_test), label_img=inputs_test[:, 0:1])

    writer.add_scalar("Accuracy/svm", clf_acc, i)
    writer.add_scalar("R_score/HDBSCAN", best_r_score, i)

    writer.flush()
    writer.close()
    return best_r_score, clf_acc, cm, clf, delta, i


if __name__ == '__main__':
    
    args = parser.parse_args()
    if args.debug:
        try:
            ptvsd.enable_attach()
            ptvsd.wait_for_attach()
        except OSError as exc:
            print(exc)

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    


    model_cfg = config["model"]
    data_params = config["data_params"]
    data_params["base_figure_dir"] = f"{data_params['base_figure_dir']}/{config['model_name']}"

    parameters_to_search = {"capacity": model_cfg["capacity"], "variational_beta": data_params["variational_beta"]}
    score_grid = {}
    for current_param in ParameterGrid(parameters_to_search):
        np.random.seed(42)

        model_cfg = {**model_cfg, "capacity": current_param["capacity"]}
        path_model_params_save = f"/acoustic/{config['model_name']}_trained:c{current_param['capacity']}:b:{current_param['variational_beta']}.pt"
        data_params = {**data_params, "path_model_params_save": path_model_params_save, "variational_beta": current_param["variational_beta"], } 
        model_cfg["window_dim"] = config["data_params"]["window_dim"]
        model = vae_models[config["model_name"]](**model_cfg)

        r_score, clf_acc, clusterer, classifier, tr_time, iterations = train_model(
            model=model,
            **data_params
        )
        score_grid[f"type:{config['model_name']};capacity:{current_param['capacity']};variational_beta:{current_param['variational_beta']}"] = {
            "clustering_score": r_score, 
            "clf_accuracy": clf_acc, 
            "clustering_params": str(clusterer).replace("\n    ", " "), 
            "classifier params": str(classifier).replace("\n    ", " "), 
            "training_time": tr_time,
            "iterations": iterations
        }
        print(f"r score: {r_score} model_params: {current_param}")

        with open(f"{data_params['base_figure_dir']}/res.json", "w") as jf:
            json.dump(score_grid, jf, sort_keys=True, indent=2)
    