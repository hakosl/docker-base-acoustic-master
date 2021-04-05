import torch
from utils.data_utils import get_datasets
from data.echogram import get_echograms
print("this is before loading data")

dataloader_train, dataloader_test, dataloader_val, dataset_train, dataset_test, dataset_val, echograms_train, echograms_test, echograms_val = get_datasets(include_depthmap=False)

def find_eg_cap(echograms):
    sandeel_eg = [sum([1 for o in e.objects if o["fish_type_index"] == 27]) for e in echograms]
    other_eg =  [sum([1 for o in e.objects if o["fish_type_index"] == 1]) for e in echograms]
    none_eg = [e for e in echograms if (27 not in [o["fish_type_index"] for o in e.objects]) and (1 not in [o["fish_type_index"] for o in e.objects])]
    print("echograms with sandeel", sum(sandeel_eg))
    print("echograms containing other class", sum(other_eg))
    print("empty echograms without sandeel or other", len(none_eg))
    return sum(sandeel_eg), sum(other_eg), len(none_eg)

print("training set")
train = find_eg_cap(echograms_train)
print("test set")
test = find_eg_cap(echograms_test)

classes = ["sandeel", "other", "empty"]
with open("output/class_total.csv", "w") as f:
    f.write("class,train,test\n")
    for (cla, tes, tra) in zip(classes, train, test):
        f.write(f"{cla},{tes},{tra}\n")