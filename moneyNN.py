import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import linecache


class ToTensor():
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class CSVDataset(Dataset):

    def __init__(self, csv_file, transform=None, n=None):
        self.csv_file = csv_file
        self.transform = transform
        with open(csv_file, 'r') as f:
            self.length = sum(1 for _ in f)
        self.length = min(n, self.length) if n is not None else self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx, columns=[1, 2, 3, 5, 7, 8, 9, 10]):
        line = linecache.getline(self.csv_file, idx + 2).strip()
        fields = line.split(',')
        try:
            sample = tuple(float(fields[c]) / (float(fields[1])/(1 + float(fields[11])/100)) for c in [1, 2, 3, 5]) + tuple(float(fields[c]) for c in [7, 8, 9, 10])
            #print(sample)
            label = float(fields[76])
        except (ValueError, IndexError, ZeroDivisionError, RuntimeError):
            return self.__getitem__((idx + 2) % len(self), columns)
        label = float(fields[76])
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor([label], dtype=torch.float32)

dataset = CSVDataset("nasdaq100_earnings_events.csv", transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Payday(nn.Module):
    def __init__(self, num_columns):
        super(Payday, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_columns, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Payday(num_columns=8).to(device)
    if os.path.exists("model_weights.pth"):
        print("Existing money model found, continuing training.")
        net.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    else:
        print("We are lowkey moneyless. Creating new money model.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(1000):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch: %d, total loss: %.3f" % (epoch, running_loss))

    print('Finished Training')

    torch.save(net.state_dict(), "model_weights.pth")