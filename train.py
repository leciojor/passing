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

  