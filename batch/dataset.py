import numpy as np

from utils.np import getGrid, getGridalt, linear_interpolation, nearest_interpolation

class Dataset():

    def __init__(self, samplers, window_size, frequencies,
                 n_samples = 1000,
                 sampler_probs=None,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None,
                 include_depthmap = False,
                 depthmap_transform=True,
                 si=False,
                 loss="BCE",
                 full_labels=False,
                 stratified_dataloader=False):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """
        self.include_depthmap = include_depthmap
        self.samplers = samplers
        self.window_size = window_size
        self.n_samples = n_samples
        self.frequencies = frequencies
        self.sampler_probs = [s.get_sample_probability() for s in self.samplers]
        self.augmentation_function = augmentation_function
        self.label_transform_function = label_transform_function
        self.data_transform_function = data_transform_function
        self.si = si
        self.depthmap_transform = depthmap_transform
        # Normalize sampling probabillities
        if self.sampler_probs is None:
            self.sampler_probs = np.ones(len(samplers))
        self.sampler_probs = np.array(self.sampler_probs)
        self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
        if not stratified_dataloader:
            self.sampler_probs /= np.max(self.sampler_probs)
        self.loss = loss
        self.full_labels = full_labels
        self.stratified_dataloader = stratified_dataloader

    def __getitem__(self, index):
        #Select which sampler to use
        if self.stratified_dataloader:
            sampler_index = np.where((index % self.sampler_probs[-1]) < self.sampler_probs)[0][0]
            sampler = self.samplers[sampler_index]
        else:
            i = np.random.rand()
            sample_index = np.where(i < self.sampler_probs)[0][0]
            sampler = self.samplers[np.where(i < self.sampler_probs)[0][0]]

        #Draw coordinate and echogram with sampler

        center_location, echogram = sampler.get_sample()

        # Adjust coordinate by random shift in y and x direction


        #Get data/labels-patches
        if sampler.random_sample:
            center_location[0] += np.random.randint(-self.window_size[0]//2, self.window_size[0]//2 + 1)
            center_location[1] += np.random.randint(-self.window_size[1]//2, self.window_size[1]//2 + 1)

            center_location[0] = max(min(self.window_size[0]//2, center_location[0]), echogram.shape[0] - self.window_size[0]//2)
            center_location[1] = max(min(self.window_size[1]//2, center_location[1]), echogram.shape[1] - self.window_size[1]//2)
            data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies, sampler.random_sample)
        else:
            data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies, sampler.random_sample)
        

        
        if self.include_depthmap:
            min_depth, max_depth = echogram.range_vector[0], echogram.range_vector[-1]
            y_len = echogram.shape[0]
            if not sampler.random_sample:
                [(ymin, ymax), _] = center_location
            else:
                [ycenter, _] = center_location
                ymin, ymax = ycenter - self.window_size[0]//2, ycenter + self.window_size[0]//2

            
            eg_min_depth, eg_max_depth = (ymin / y_len) * max_depth + min_depth, (ymax / y_len) * max_depth + min_depth 

            depthmap = np.repeat(np.linspace(eg_min_depth, eg_max_depth, data.shape[1])[np.newaxis], data.shape[2], axis=0)[np.newaxis].transpose((0, 2, 1))
            if self.depthmap_transform:
                depthmap = np.tanh(depthmap/100)
            data = np.vstack((data, depthmap))
    
        # Apply augmentation
        if self.augmentation_function is not None:
            data, labels, echogram = self.augmentation_function(data, labels, echogram)

        # Apply label-transform-function
        if self.label_transform_function is not None:
            data, labels, echogram = self.label_transform_function(data, labels, echogram)

        # Apply data-transform-function
        if self.data_transform_function is not None:
            data, labels, echogram, frequencies = self.data_transform_function(data, labels, echogram, self.frequencies)
        
        

        labels = labels.astype('int16')
        if self.si:
            if self.full_labels:
                return data, labels, sample_index
            else:
                return data, labels, sampler.get_label()
        else:
            return data, labels

    def __len__(self):
        return self.n_samples


def get_crop(echogram, center_location, window_size, freqs, random_sample):
    """
    Returns a crop of data around the pixels specified in the center_location.

    """
    # Get grid sampled around center_location
    
    
    if random_sample:
        grid = getGrid(window_size)
        grid[0] += center_location[0]
        grid[1] += center_location[1]
    else:
        gridinput = [[center_location[i][0], center_location[i][1], window_size[i]] for i in range(len(center_location))]
        grid = getGridalt(gridinput)# + np.expand_dims(np.expand_dims(center_location, 1), 1)

    channels = []
    for f in freqs:

        # Interpolate data onto grid
        memmap = echogram.data_memmaps(f)[0]
        data = linear_interpolation(memmap, grid, boundary_val=0, out_shape=window_size)
        del memmap

        # Set non-finite values (nan, positive inf, negative inf) to zero
        if np.any(np.invert(np.isfinite(data))):
            data[np.invert(np.isfinite(data))] = 0

        channels.append(np.expand_dims(data, 0))
    channels = np.concatenate(channels, 0)

    labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)

    return channels, labels
