import torch
from utils.data_utils import get_datasets
from data.echogram import get_echograms
print("this is before loading data")

dataloader_train, dataloader_test, dataloader_val, dataset_train, dataset_test, dataset_val, echograms_train, echograms_test, echograms_val = get_datasets(include_depthmap=False)

def find_eg_cap(echograms):
    sandeel_eg = [e for e in echograms if 27 in [o["fish_type_index"] for o in e.objects]]
    other_eg = [e for e in echograms if 1 in [o["fish_type_index"] for o in e.objects]]
    none_eg = [e for e in echograms if 27 and 1 not in [o["fish_type_index"] for o in e.objects]]
    print("echograms with sandeel", len(sandeel_eg))
    print("echograms containing other class", len(other_eg))
    print("empty echograms without sandeel or other", len(none_eg))
    return len(sandeel_eg), len(other_eg), len(none_eg)

print("training set")
train = find_eg_cap(echograms_train)
print("test set")
test = find_eg_cap(echograms_test)

classes = ["sandeel", "other", "empty"]
with open("output/class_dist.csv", "w") as f:
    f.write("class,train,test\n")
    for (cla, tes, tra) in zip(classes, train, test):
        f.write(f"{cla},{tes},{tra}\n")