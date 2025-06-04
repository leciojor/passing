
from data import PlaysData
from torch.utils.data import DataLoader, Subset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
  print("ROCm is working")
else:
  DEVICE = torch.device("cpu")

def getting_loader(batch_size, save=False, num_workers=2, variant = 1, train_p=0.7, saved=False, drop_qb_orientation = False, get_dataset=False, all_frames=False, distr_analysis=True, play_id=None, game_id=None, cleaning=True, split=True, passed_result_extra = False, beta=True, get_receiver_id=False, intended_receiver_input=False, receiver_to_project=0, just_shoulder_orientation=False):
    if not beta:
        file_name = f"./finalFeatures/datasetsAlpha/final_data_variant{variant}_{all_frames}"
    else:
        file_name = f"./finalFeatures/final_data_variant{variant}_{all_frames}"     

    if not saved:
        dataset = PlaysData(variant, all = all_frames, game_id=game_id, play_id=play_id, passed_result_extra=passed_result_extra, beta=beta, get_receiver_id=get_receiver_id, intended_receiver_input=intended_receiver_input, just_shoulder_orientation=just_shoulder_orientation)
        if save:
          if all_frames:
            dataset.get_csv(name = file_name + f"_game{game_id}_play{play_id}.csv")
          else:
            dataset.get_csv(name = file_name + f".csv")

    else:      
        if all_frames:
            dataset = PlaysData(variant, pd.read_csv(file_name + f"_game{game_id}_play{play_id}.csv"), all = all_frames, game_id=game_id, play_id=play_id, passed_result_extra=passed_result_extra, beta=beta, get_receiver_id=get_receiver_id, intended_receiver_input=intended_receiver_input, just_shoulder_orientation=just_shoulder_orientation)
        else:
          dataset = PlaysData(variant, pd.read_csv(file_name + ".csv"), all = all_frames, game_id=game_id, play_id=play_id, passed_result_extra=passed_result_extra, beta=beta, get_receiver_id=get_receiver_id, intended_receiver_input=intended_receiver_input, just_shoulder_orientation=just_shoulder_orientation)

    n = len(dataset)

    if distr_analysis:
      dataset.distributions_analysis()

    if cleaning:
      print(f"**BEFORE CLEANING** Dataset size: {n}")
      dataset.converting_numerical_and_cleaning(receiver_to_project)
      if save:
          if all_frames:
            dataset.get_csv(name = file_name + f"_game{game_id}_play{play_id}_cleaned.csv")
          else:
            dataset.get_csv(name = file_name + f"_cleaned.csv")
      n_clean = len(dataset)
      n = n_clean
      print(f"**AFTER CLEANING** Dataset size: {n_clean}")

    if drop_qb_orientation:
       dataset.data = dataset.data.drop("qb_orientation", axis=1)
       dataset.col_size = dataset.data.shape[1]

    if split:
      train_amount = int(n*train_p)
      train_indices = list(range(train_amount + 1))
      val_indices = list(range(train_amount + 1, n))

      train_dataset = Subset(dataset, train_indices)
      val_dataset = Subset(dataset, val_indices)

      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    else:
       return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), dataset 
    
    if get_dataset:
      return train_loader, val_loader, dataset
    return train_loader, val_loader

def getting_frames_dataset(game_id, play_id, loaded, save, get_angles, receiver_to_project, beta=False):
  loader, dataset = getting_loader(1, save=save, num_workers=0, variant = 5, train_p=0.8, saved=loaded, distr_analysis=False, get_dataset=True, game_id=game_id, play_id=play_id, all_frames=True, beta=beta, split=False, get_receiver_id=get_angles, receiver_to_project=receiver_to_project)
  return dataset

def get_acc(y_hat, y, t):
    with torch.no_grad():
        #regression
        if t == 2:
            tolerance = 10.0
            inferences = (torch.abs(y_hat - y) < tolerance).float()

        #binary classification
        elif t == 5:
            probs = torch.sigmoid(y_hat) 
            preds = (probs >= 0.5)
            inferences = (preds == y).float()

        #multi class classification (3 and 5)
        else:  
            y = torch.argmax(y, dim=1)
            probs = torch.softmax(y_hat, dim=1)
            preds = torch.argmax(probs, dim=1)
            inferences = (preds == y).float()

    return torch.mean(inferences).item()

def get_val(net, val_dataloader, loss_f, t):
  losses = []
  accs = []
  net.to(DEVICE)
  net.eval()
  with torch.no_grad():
    for x,y in val_dataloader:
      x = x.to(DEVICE)
      y = y.to(DEVICE)
      y_hat = net(x)
      L = loss_f(y_hat, y)
      acc = get_acc(y_hat, y, t)
      accs.append(acc)
      losses.append(L.item())
    net.train()
    return sum(losses) / len(losses), sum(accs) / len(accs)

def train(net, optimizer, loss_f, train_dataloader, val_dataloader, n_minibatch_steps, log_interval, val_interval, scheduler, version, t=1, pretrained=False):
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
        acc = get_acc(y_hat, y, t)

        L.backward()
        optimizer.step()

        i += 1
        optimizer.zero_grad()
        if not i % log_interval:
          print(f"LOSS: {L.item()} ACCURACY: {acc}")
          loss_training.append(L.item())
          acc_training.append(acc)

        if not i % val_interval:
          loss_v, accs_v = get_val(net, val_dataloader, loss_f, t)
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


def plotting(version, loss_training, acc_training, loss_val, acc_val):
  def plot_(title, results, ylabel, val=False):
      plt.xlabel("Epochs")
      if val:
        plt.ylabel(f"Validation {ylabel}")
      else:
        plt.ylabel(f"Training {ylabel}")
      plt.plot(results)
      plt.legend()
      plt.title(title)
      plt.savefig("modelsPerformance/" + title + ".png")
      plt.close()

  plot_(f"Loss Training {version}", loss_training, "Loss")
  plot_(f"Accuracy Training {version}", acc_training, "Accuracy")
  plot_(f"Loss Validation {version}", loss_val, "Loss", val=True)
  plot_(f"Accuracy Validation {version}", acc_val, "Accuracy", val=True)

# getting_loader(1, save=False, num_workers=2, variant = 1, train_p=0.7, saved=False, drop_qb_orientation = True, get_dataset=False, all_frames=False, distr_analysis=False, i=4, play_id=None, game_id=None)
# getting_loader(1, save=True, num_workers=2, variant = 1, train_p=0.7, saved=False, drop_qb_orientation = False, get_dataset=False, all_frames=False, distr_analysis=False, i=4, play_id=None, game_id=None)
# getting_loader(1, save=True, num_workers=2, variant = 2, train_p=0.7, saved=False, drop_qb_orientation = False, get_dataset=False, all_frames=False, distr_analysis=False, i=4, play_id=None, game_id=None)
# getting_loader(1, save=True, num_workers=2, variant = 3, train_p=0.7, saved=False, drop_qb_orientation = False, get_dataset=False, all_frames=False, distr_analysis=False, i=4, play_id=None, game_id=None)
# getting_loader(1, save=True, num_workers=2, variant = 5, train_p=0.7, saved=False, drop_qb_orientation = False, get_dataset=False, all_frames=False, distr_analysis=False, i=4, play_id=None, game_id=None)
# getting_loader(1, save=True, num_workers=2, variant = 6, train_p=0.7, saved=False, drop_qb_orientation = False, get_dataset=False, all_frames=False, distr_analysis=False, i=4, play_id=None, game_id=None)
# getting_frames_dataset(2022091200, 109, False, True, beta=False)
