
from data import PlaysData
from torch.utils.data import DataLoader, Subset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
else:
  DEVICE = torch.device("cpu")

def getting_loader(batch_size, save=False, num_workers=2, variant = 1, train_p=0.7, saved=False):
    if not saved:
        dataset = PlaysData(variant)
        if save:
            dataset.get_csv()

    else:
        dataset = PlaysData(variant, pd.read_csv(f"final_data_variant{variant}.csv"))

    n = len(dataset)

    print(f"**BEFORE CLEANING** Dataset size: {n}")
    dataset.cleaning()
    dataset.converting_numerical()
    if save:
        dataset.get_csv(name = f"./final_data_variant{variant}_cleaned.csv")
    n_clean = len(dataset)
    print(f"**AFTER CLEANING** Dataset size: {n_clean}")

    train_amount = int(n*train_p)
    train_indices = list(range(train_amount + 1))
    val_indices = list(range(train_amount + 1, n))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
    
def train(net, optimizer, loss_f, train_dataloader, val_dataloader, n_minibatch_steps, log_interval, val_interval, scheduler, version, pretrained=False):
  try:
    loss_training = []
    loss_val = []
    acc_training = []
    acc_val = []

    net.to(DEVICE)
    i = 0
    while i < n_minibatch_steps:
      for x,y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_hat = net(x)
        L = loss_f(y_hat, y)
        acc = get_acc(y_hat, y)

        L.backward()
        optimizer.step()

        i += 1
        optimizer.zero_grad()
        if not i % log_interval:
          print(f"LOSS: {L.item()} ACCURACY: {acc}")
          loss_training.append(L.item())
          acc_training.append(acc)

        if not i % val_interval:
          loss_v, accs_v = get_val(net, val_dataloader, loss_f, pretrained)
          print(f"VAL LOSS: {loss_v} VAL ACCURACY: {accs_v}")
          acc_val.append(accs_v)
          loss_val.append(loss_v)

        if n_minibatch_steps < i + 1:
            break

      if scheduler:
          scheduler.step()
      torch.cuda.empty_cache()
    print('Saving model...')
    torch.save(net.state_dict(), f"./models/model_{version}.pkl")

    return loss_training, acc_training, loss_val, acc_val

  except KeyboardInterrupt:
    print('Saving model...')
    torch.save(net.state_dict(), f"./models/model_{version}.pkl")

    return loss_training, acc_training, loss_val, acc_val

getting_loader(32, True, saved=False)

