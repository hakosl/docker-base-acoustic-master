import torch
from utils.data_utils import get_datasets
from data.echogram import get_echograms
import numpy as np
print("this is before loading data")

dataloader_train, dataloader_test, dataloader_val, dataset_train, dataset_test, dataset_val, echograms_train, echograms_test, echograms_val = get_datasets(batch_size = 1000, iterations=100, depthmap_transform=False)

imgs = np.array((1000* 100, 5, 32, 32))
labels = np.array((1000 * 100, 1, 32, 32))
si = np.array((1000 * 100,))

for i, (data, l, s) in enumerate(dataloader_train):
    b, c, h, w = data.shape
    imgs[100 * i: 100* (i + 1)] = data
    labels[100 * i: 100*(i + 1)] = l
    si[100 * i: 100*(i + 1)] = s

print(imgs.shape)

with open("./data/train.npy", "wb") as f:
    np.save(f, imgs)
    np.save(f, labels)
    np.save(f, si)

imgs = np.array((1000* 100, 5, 32, 32))
labels = np.array((1000 * 100, 1, 32, 32))
si = np.array((1000 * 100,))

for i, (data, l, s) in enumerate(dataloader_test):
    b, c, h, w = data.shape
    imgs[100 * i: 100* (i + 1)] = data
    labels[100 * i: 100*(i + 1)] = l
    si[100 * i: 100*(i + 1)] = s

with open("./data/test.npy", "wb") as f:
    np.save(f, imgs)
    np.save(f, labels)
    np.save(f, si)

