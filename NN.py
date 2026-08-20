import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import os

# Source - https://stackoverflow.com/a/59941305
# Posted by bashBedlam, modified by community. See post "Timeline" for change history
# Retrieved 2026-08-20, License - CC BY-SA 4.0

with open ("processed_NASDAQ.csv", "w") as out_file :
    with open ("NASDAQ100_Historical_Data.csv") as in_file :
        for i in range (100):
            line = in_file[i]
            test_string = line.strip("\n").split(",")
            print(test_string)
            #out_file.write (",".join (test_string [1:]) + "\n")


# train = pd.read_csv("NASDAQ100_Historical_Data.csv")
# train_tensor = torch.tensor(train.values)
# train_tensor = train_tensor[:6571] #1-6571 is all the AAPL; test with this smaller set first

# print(train_tensor)