from utils.np import getGrid, linear_interpolation
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Loader(Dataset):

    def __init__(self, return_len, func, *args, **kwargs):
        self.return_len = return_len
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, index):
        x = self.func(*self.args, **self.kwargs)
        x = [x[0][0], x[1][0]]
        return x

    def __len__(self):
        return self.return_len



# Function to cut out crop
def get_crop(echogram, center_location, window_size, freqs, mirror_crops, add_noise, return_labels=True):
    # Get grid sampled around center_location
    grid = getGrid(window_size) + np.expand_dims(np.expand_dims(center_location, 1), 1)

    # Mirror crop about vertical axis?
    mirror = bool(np.random.binomial(1, 0.5))

    # Interpolate data onto grid
    channels = []
    for f in freqs:

        data = linear_interpolation(echogram.data_memmaps(f)[0], grid, boundary_val=0, out_shape=window_size)

        # Set non-finite values (nan, positive inf, negative inf) to zero
        if np.any(np.invert(np.isfinite(data))):
            data[np.invert(np.isfinite(data))] = 0



        # Log-transform and normalize data (each frequency separately)
        '''
        data -= np.min(data)
        data = np.log(data + 1e-10)
        data -= np.mean(data)
        data *= 1 / (np.std(data) + 1e-10)
        '''

        # Add random noise (boolean function argument)
        '''
        # In a portion of the crops, a portion of the pixel values are increased by random uniformly distributed values
        if add_noise:
            data += \
                np.random.binomial(1, 0.5) * \
                np.multiply(
                    np.random.binomial(1, 0.9, data.shape),
                    np.random.uniform(-0.2, 0.2, data.shape)
                )
            # Re-normalize data
            data -= np.mean(data)
            data *= 1 / (np.std(data) + 1e-10)
        '''

        # Mirror crop
        if mirror_crops and mirror:
            data = np.flip(data, axis=1)

        data = np.expand_dims(data, 0)
        channels.append(data)
    channels = np.concatenate(channels, 0)

    if return_labels:
        labels = linear_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)

        # Mirror labels about vertical axis
        if mirror:
            labels = np.flip(labels, axis=1)

        return channels, labels

    else:
        return channels


# Get training batches to classify image as either fish or no fish
# Controlling the portion of fish crops from seabed vs. not seabed
def get_batch(batch_size, window_size, freqs, echograms, class_balance_seabed=True, mirror_crops=True, add_noise=False, segmentation=False):

    ### Input parameters
    # batch_size (int): number of crops in one batch
    # window_size (int): size of crop, number of pixels in each spatial dimension
    # freqs (list of ints): frequency channels that are included
    # echograms (list of Echogram objects): echograms that are included
    # class_balace_seabed (bool):
    #   When 'True', a portion of the crops will be located at the seabed.
    #   When 'False', all crop are located randomly with respect to seabed.
    ###

    if type(window_size) == int:
        window_size = np.array([window_size, window_size])

    # Returns crop with fish, with or without seabed
    # seabed==True: crop contains seabed
    # seabed==False: crop does not contain seabed
    def crop_with_fish(seabed):
        fish = True
        crop_with_fish_not_found = True

        while crop_with_fish_not_found:
            echogram_no = np.random.randint(0, len(echograms))
            while len(echograms[echogram_no].objects) == 0:
                echogram_no = np.random.randint(0, len(echograms))
            echogram = echograms[echogram_no]

            for obj in np.random.permutation(echogram.objects):

                bbox = obj['bounding_box']
                bbox_x_mean = int((bbox[2] + bbox[3]) / 2)

                # Class balancing of patches with respect to seabed:
                # class_balance_seabed == True: A portion of the crops are drawn close to seabed
                # class_balance_seabed == False: Crops are randomly selected, i.e. not controlling closeness to seabed
                if class_balance_seabed:
                    if seabed:
                        # Check if bounding box is a likely candidate for getting a crop that contains seabed
                        seabed_condition_is_met = bool(np.abs(echogram.seabed[bbox_x_mean] - bbox[1]) < window_size[0]/2)
                    else:
                        # Check if bounding box is a likely candidate for getting a crop that does not contain seabed
                        seabed_condition_is_met = bool(np.abs(echogram.seabed[bbox_x_mean] - bbox[1]) > window_size[0])
                else:
                    seabed_condition_is_met = True

                if seabed_condition_is_met:
                    # Crop with center coordinate at random location close to or within the bounding box
                    x = int(np.random.random_integers(bbox[2], bbox[3]) + np.random.normal(0, 0.1) * window_size[1]/2)
                    y = int(np.random.random_integers(bbox[0], bbox[1]) + np.random.normal(0, 0.1) * window_size[0]/2)
                    d, l = get_crop(echogram, [y, x], window_size, freqs, mirror_crops, add_noise, return_labels=True)

                    if np.sum(l != 0) - np.sum(np.isnan(l)) == 0:
                        fish = False  # Relabel crop as 'no fish' if no pixels are labeled 'fish'
                    crop_with_fish_not_found = False
                    break

        return d, l, fish

    data = []
    labels_by_pixel = []
    labels = []
    sb_labels = []

    while len(data) < batch_size:

        # Decide if crop shall contain fish/seabed, respectively
        # The resulting distribution should be approximately 0.25 for each combination of fish/seabed
        fish = bool(np.random.binomial(1, 0.5))  # True: fish, False: no fish
        seabed = bool(np.random.binomial(1, 0.5))  # True: seabed, False: no seabed

        # Find crop that contains fish
        if fish:
            # Find crop that contains fish with or without seabed
            # Might return crop without fish - in that case, crop is relabeled as 'not fish'
            d, l, fish = crop_with_fish(seabed)

        # Find crop that does not contain any fish
        else:

            crop_coordinates_not_found = True
            while crop_coordinates_not_found:

                while True:
                    echogram_no = np.random.randint(0, len(echograms))
                    echogram = echograms[echogram_no]
                    if window_size[1] // 2 < echogram.shape[1] - window_size[1] // 2 + 1:
                        break

                if class_balance_seabed:

                    # Select random coordinates close to seabed (seabed line with random vertical shift)
                    if seabed:
                            x = np.random.randint(window_size[1]//2, echogram.shape[1] - window_size[1]//2 + 1)
                            y = echogram.seabed[x] + np.random.randint(-window_size[0]//2, window_size[0]//2 + 1)
                            crop_coordinates_not_found = False

                    # Select random coordinates off the seabed
                    else:
                        # The seabed is close to the surface for some of the echograms.
                        # In such cases, it will not be possible to fit a crop between the surface and the seabed without including any seabed.
                        # For a given echogram, if 10 random x-coordinates does not produce a valid crop, we break and pick a new echogram.
                        num_tries = 0
                        while num_tries < 10:
                            num_tries += 1
                            x = np.random.randint(window_size[1]//2, echogram.shape[1] - window_size[1]//2 + 1)
                            if window_size[0]//2 < echogram.seabed[x] - window_size[0]:
                                # Select random y coordinate sufficiently far from, and above, seabed
                                y = np.random.randint(window_size[0]//2, echogram.seabed[x] - window_size[0])
                                crop_coordinates_not_found = False
                                break

                # class_balance_seabed == False: crop from random location above seabed
                else:
                    num_tries = 0
                    while num_tries < 10:
                        num_tries += 1
                        x = np.random.randint(window_size[1]//2, echogram.shape[1] - window_size[1]//2 + 1)
                        # Make sure that a crop can fit between the surface and the seabed
                        if window_size[0]//2 < echogram.seabed[x]:
                            y = np.random.randint(window_size[0]//2, echogram.seabed[x])
                            crop_coordinates_not_found = False
                            break

            d, l = get_crop(echogram, [y, x], window_size, freqs, mirror_crops, add_noise, return_labels=True)

            # Relabel crop as 'fish' if it contains any pixels labeled as 'fish'
            if np.sum(l != 0) - np.sum(np.isnan(l)) != 0:
                fish = True

        if np.any(np.isnan(d)):
            d[np.isnan(d)] = 0
        data.append(np.expand_dims(d, 0))

        #Return either label crop or segmentation
        if not segmentation:
            labels.append(np.array([int(fish)]))
        else:
            #labels.append(np.expand_dims(np.expand_dims(l,0),0).astype('int'))
            labels.append(np.expand_dims(l, 0).astype('int'))

        labels_by_pixel.append(np.expand_dims(np.expand_dims(l, 0), 0).astype('int'))
        sb_labels.append(np.array([int(seabed)]))

    # Concat to make N x C x H x W batch
    data = np.concatenate(data, 0)
    labels = np.concatenate(labels, 0)
    labels_by_pixel = np.concatenate(labels_by_pixel, 0)
    sb_labels = np.concatenate(sb_labels, 0)

    return data, labels, labels_by_pixel, sb_labels


# Example of use
if __name__ == '__main__':

    dataset = Loader(2*3, get_batch, 1, 2, [18, 38], echogram.get_echograms())
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, worker_init_fn=np.random.seed)

    for i, (inputs, labels) in enumerate(dataloader):
        # Do something with inputs and labels
        pass
    '''
    data, labels, labels_by_pixel, sb_labels = get_batch(32, 52, freqs=[18, 38, 120, 200], echograms=echogram.get_echograms(), segmentation=True)
    print(data.shape)
    print(labels.shape)
    print(labels_by_pixel.shape)
    print(sb_labels.shape)
    print(sb_labels)
    print(labels)
    '''


