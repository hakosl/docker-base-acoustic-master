
from torch.utils.data import Dataset, DataLoader
import numpy as np

class acoustics_dataset(Dataset):
    def __init__(self, file, transform=None):
        self.file = file
        with open(file, "rb") as f:
            self.imgs = np.load(f)
            self.labels = np.load(f)
            self.si = np.load(f)

        self.si = self.si.astype(int)

        self.transform = transform
        self.label_names = ["background", "seabed", "other", "sandeel"]

        self.length = self.si.shape[0]
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx], self.si[idx]

    def get_label_names(self):
        return label_names

