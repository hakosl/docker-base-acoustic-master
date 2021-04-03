import torch
from utils.data_utils import get_datasets
from data.echogram import get_echograms
print("this is before loading data")

dataloader_train, dataloader_test, dataloader_val, dataset_train, dataset_test, dataset_val, echograms_train, echograms_test, echograms_val = get_datasets(include_depthmap=False)

cnt = 0
fst_moment = torch.empty(5)
snd_moment = torch.empty(5)
print("computing_mean")
for (data, labels, si) in dataloader_train:
    b, c, h, w = data.shape
    nb_pixels = b * h * w
    sum_ = torch.sum(data, dim=[0, 2, 3])
    sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
    fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
    snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

    cnt += nb_pixels

print(fst_moment, torch.sqrt(snd_moment - fst_moment ** 2))