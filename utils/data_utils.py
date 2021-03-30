import numpy as np

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
             ['D2011', 'D2012', 'D2013', 'D2014', 'D2015']]), echograms))
        test = list(filter(lambda x: any([year in x.name for year in ['D2017']]), echograms))
        val = list(filter(lambda x: any([year in x.name for year in ['D2016']]), echograms))

    else:
        print("Parameter 'partition' must equal 'random' or 'year'")

    print('Train:', len(train), ' Test:', len(test), ' Val:', len(val))

    return train, test, val


def get_validation_set_paths(validation_set):
    return [ech.name for ech in validation_set[1]]

def get_datasets(frequencies=[18, 38, 120, 200], window_dim=64, partition="random", batch_size=64, iterations=1000, num_workers=0, include_depthmap=True, test_size=1000):
    # Load echograms and create partition for train/test/val
    window_size = [window_dim, window_dim]
    echograms = get_echograms(frequencies=frequencies, minimum_shape=window_dim)

    echograms_train, echograms_test, echograms_val = partition_data(echograms, partition, portion_train=0.85)


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

    sample_options_val = {
        "echograms": echograms_val,
        **sample_options
    }

    
    sampler_probs = [2, 2, 1, 1, 1, 1]
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

    samplers_val = [
        Background(**sample_options_val, sample_probs=sampler_probs[0]),
        Seabed(**sample_options_val, sample_probs=sampler_probs[1]),
        Shool(**sample_options_val, fish_type=27, sample_probs=sampler_probs[2]),
        Shool(**sample_options_val, fish_type=1, sample_probs=sampler_probs[3]),    
        ShoolSeabed(**sample_options_val, fish_type=27, sample_probs=sampler_probs[4]),
        ShoolSeabed(**sample_options_val, fish_type=1, sample_probs=sampler_probs[5])
    ]


    augmentation = CombineFunctions([])
    label_transform = CombineFunctions([index_0_1_27, relabel_with_threshold_morph_close])
    #if recon_criterion == "MSE":
    #    # if recon criterion is mse we want the domain to be -1 - 1
    #    data_transform = CombineFunctions([remove_nan_inf, db_with_limits_norm_MSE])
    #else:
    data_transform = CombineFunctions([remove_nan_inf, db_with_limits_norm])

    transform_functions = {
        "augmentation_function": augmentation,
        "label_transform_function": label_transform,
        "data_transform_function": data_transform
    }

    dataset_arguments = [window_size, frequencies, batch_size * iterations]
    test_dataset_arguments = [window_size, frequencies, 1000]

    dataset_train = Dataset(samplers_train, *dataset_arguments, si=True, **transform_functions, include_depthmap=include_depthmap)
    dataset_test = Dataset(samplers_test, *test_dataset_arguments, si=True, **transform_functions, include_depthmap=include_depthmap)
    dataset_val = Dataset(samplers_val, *dataset_arguments, si=True, **transform_functions, include_depthmap=include_depthmap)
    dataloader_arguments = {
        "batch_size": batch_size, 
        "shuffle": False, 
        "num_workers":num_workers, 
        "worker_init_fn": np.random.seed
    }

    test_dataloader_arguments = {
        **dataloader_arguments,
        "batch_size": test_size
    }

    val_dataloader_arguments = {
        **dataloader_arguments
    }

    dataloader_train = DataLoader(dataset_train, **dataloader_arguments)
    dataloader_test = DataLoader(dataset_test, **test_dataloader_arguments)
    dataloader_val = DataLoader(dataset_val, **dataloader_arguments)

    return dataloader_train, dataloader_test, dataloader_val, dataset_train, dataset_test, dataset_val, echograms_train, echograms_test, echograms_val

