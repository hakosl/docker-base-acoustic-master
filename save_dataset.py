import torch
from utils.data_utils import get_datasets
from data.echogram import get_echograms
import numpy as np
print("this is before loading data")

dataloader_train, dataloader_test, dataloader_val, dataset_train, dataset_test, dataset_val, echograms_train, echograms_test, echograms_val = get_datasets(window_dim=64, batch_size = 100, iterations=1000, depthmap_transform=False)

imgs = np.zeros((1000* 100, 5, 64, 64))
labels = np.zeros((1000 * 100, 64, 64))
si = np.zeros((1000 * 100,))
i = 0
for data, l, s in dataloader_train:
    b, c, h, w = data.shape
    print(data.shape)
    imgs[100 * i: 100* (i + 1)] = data
    labels[100 * i: 100*(i + 1)] = l
    si[100 * i: 100*(i + 1)] = s
    i += 1

print(imgs.shape)

with open("./data/train.npy", "wb") as f:
    np.save(f, imgs)
    np.save(f, labels)
    np.save(f, si)

imgs = np.zeros((100* 100, 5, 64, 64))
labels = np.zeros((100 * 100, 64, 64))
si = np.zeros((100 * 100,))

i = 0
for data, l, s in dataloader_test:
    b, c, h, w = data.shape
    imgs[100 * i: 100* (i + 1)] = data
    labels[100 * i: 100*(i + 1)] = l
    si[100 * i: 100*(i + 1)] = s
    i += 1


with open("./data/test.npy", "wb") as f:
    np.save(f, imgs)
    np.save(f, labels)
    np.save(f, si)

