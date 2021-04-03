import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

class SequentialNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, vae, device= 0, verbose = False, epochs=100):
        super(SequentialNetwork, self).__init__()
        self.verbose = verbose
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.losses = []
        self.device = device
        self.epochs = epochs
        self.network = nn.Sequential(
            nn.Linear(in_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, int(out_dim)),
            nn.Softmax(dim=1)
            )
        print(self.network)
        self.network.to(device)

        self._estimator_type = "classifier"


        self.optimizer = optim.Adam(self.network.parameters())


    def __str__(self):
        return f"Sequential ( in = {self.in_dim} out = {self.out_dim} )"
    def forward(self, X):
        #_, X, _ = self.vae(X)
        return self.network(X)

    def loss(self, X, y):
        return F.mse_loss(self.forward(X), F.one_hot(y).float())


    def fit(self, X, y):
        best_epoch_loss = float("inf")
        for j in range(self.epochs):
            cur_epoch_loss = 0.0
            for i, (inputs_train, si) in enumerate(DataLoader(list(zip(X, y)), batch_size=64)):

                inputs_train = inputs_train.float().to(self.device)
                si = si.to(self.device)
                self.network.train()
                self.optimizer.zero_grad()

                self.optimizer.zero_grad()
                l = self.loss(inputs_train, si)
                cur_epoch_loss += l.item()
                l.backward()
                self.optimizer.step()
                self.losses.append(l.item())

                if i % 100 == 0 and self.verbose:
                    print(f"i: {i} loss: {self.losses[-1]}")

            if cur_epoch_loss < best_epoch_loss:
                print(f"finished after {j} epochs")
                return
        print("finished after all epochs")
    
    def predict(self, data):
        data = torch.tensor(data).float().to(self.device)
        probabilities = self.network(data)
        pred = torch.argmax(probabilities, dim=1)
        return pred.detach().cpu().numpy()
