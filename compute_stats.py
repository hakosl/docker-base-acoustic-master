import torch
from utils.data_utils import get_datasets
print("this is before loading data")
dataloader_train, dataloader_test, dataset_train, dataset_test, echograms_train, echograms_test = get_datasets(include_depthmap=False)

sums = torch.Tensor(4)
sumsq = torch.Tensor(4)
print("reached for loop")
for (x, _, _) in dataloader_train:
    x = x.float()
    sums += torch.sum(x.transpose(0, 1).reshape(4, -1), axis=1)
    sumsq += torch.sum(x.transpose(0, 1).reshape(4, -1).pow(2), axis=1)

dlen = len(dataloader_train) * 64 * 64
mean = sums / dlen
std = torch.sqrt(sumsq - torch.div(sums * sums, dlen))/(dlen - 1)
print(mean.item(), std.item())
