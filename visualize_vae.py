import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from vae_model import VariationalAutoencoder

dev = 0
device = torch.device(dev if torch.cuda.is_available() else "cpu")

model = VariationalAutoencoder()
checkpoint = "/acoustic/vae_trained.pt"
#model = nn.DataParallel(model)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.to(device)
