from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from helpers import getting_loader, get_acc, get_val, train, plotting, DeepQBVariant1, DeepQBVariant2, DeepQBVariant3

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
else:
  DEVICE = torch.device("cpu")

n = 80000
lr = 0.01

#training variant 5

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=0, variant = 5, train_p=0.8, saved=True, distr_analysis=False, get_dataset=True)
net = DeepQBVariant1(input_dim=len(dataset.columns), output_dim=1)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.BCEWithLogitsLoss(), train_loader, val_loader, n, 1, 1, None, f"variant5_lr:{lr}_n:{n}", t=5, pretrained=False)

#training variant 6

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=0, variant = 6, train_p=0.8, saved=True, distr_analysis=False, get_dataset=True)
net = DeepQBVariant1(input_dim=len(dataset.columns), output_dim=3)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, n, 1, 1, None,f"variant6:{lr}_n:{n}", t=6, pretrained=False)
