import torch
import torch.nn as nn
from helpers import getting_loader, train, plotting
from archs import DeepQBVariant1

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
else:
  DEVICE = torch.device("cpu")

n = 250000
lr = 0.01

# #training variant 1

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=6, variant = 1, train_p=0.9, saved=True, distr_analysis=False, get_dataset=True, drop_qb_orientation=True, beta=True)
net = DeepQBVariant1(input_dim=dataset.col_size - 5, output_dim=5)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
version = f"variant1_lr{lr}_n{n}"
loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, n, 1000, 1000, None, version, t=1, pretrained=False)
plotting(version, loss_training, acc_training, loss_val, acc_val)

# #training variant 1 with shoulder orientation

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=6, variant = 1, train_p=0.9, saved=True, distr_analysis=False, get_dataset=True, drop_qb_orientation=False, beta=True)
net = DeepQBVariant1(input_dim=dataset.col_size - 5, output_dim=5)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
version = f"variant1_lr{lr}_n{n}_with shoulder orientation"
loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, n, 1000, 1000, None, version, t=1, pretrained=False)
plotting(version, loss_training, acc_training, loss_val, acc_val)

# training variant 1 with shoulder orientation ONLY (ALPHA)

# train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=6, variant = 1, train_p=0.9, saved=False, distr_analysis=False, get_dataset=True, drop_qb_orientation=False, beta=False, just_shoulder_orientation=True)
# net = DeepQBVariant1(input_dim=dataset.col_size - 5, output_dim=5)
# optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
# version = f"variant1_lr{lr}_n{n}_with shoulder orientation"
# loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, n, 1000, 1000, None, version, t=1, pretrained=False)
# plotting(version, loss_training, acc_training, loss_val, acc_val)

#training variant 2

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=6, variant = 2, train_p=0.9, saved=True, distr_analysis=False, get_dataset=True, beta=True)
net = DeepQBVariant1(input_dim=dataset.col_size - 1, output_dim=1)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
version = f"variant2_lr{lr}_n{n}"
loss_training, acc_training, loss_val, acc_val = train(net, optimizer,  nn.MSELoss(), train_loader, val_loader, n, 1000, 1000, None, version, t=2, pretrained=False)
plotting(version, loss_training, acc_training, loss_val, acc_val)

# #training variant 5

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=6, variant = 5, train_p=0.9, saved=True, distr_analysis=False, get_dataset=True, beta=True)
net = DeepQBVariant1(input_dim=dataset.col_size - 1, output_dim=1)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
version = f"variant5_lr{lr}_n{n}"
loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.BCEWithLogitsLoss(), train_loader, val_loader, n, 1000, 1000, None, version, t=5, pretrained=False)
plotting(version, loss_training, acc_training, loss_val, acc_val)

# # #training variant 6

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=6, variant = 6, train_p=0.9, saved=True, distr_analysis=False, get_dataset=True, beta=True)
net = DeepQBVariant1(input_dim=dataset.col_size - 3, output_dim=3)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
version = f"variant6{lr}_n{n}"
loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, n, 1000, 1000, None,f"variant6{lr}_n{n}", t=6, pretrained=False)
plotting(version, loss_training, acc_training, loss_val, acc_val)

n = 250000
lr = 0.001

# # #training variant 5

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=6, variant = 5, train_p=0.9, saved=True, distr_analysis=False, get_dataset=True, beta=True)
net = DeepQBVariant1(input_dim=dataset.col_size - 1, output_dim=1)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
version = f"variant5_lr{lr}_n{n}"
loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.BCEWithLogitsLoss(), train_loader, val_loader, n, 1000, 1000, None, version, t=5, pretrained=False)
plotting(version, loss_training, acc_training, loss_val, acc_val)

# # #training variant 6

train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=6, variant = 6, train_p=0.9, saved=True, distr_analysis=False, get_dataset=True, beta=True)
net = DeepQBVariant1(input_dim=dataset.col_size - 3, output_dim=3)
optimizer = torch.optim.Adam(lr=lr, params=net.parameters())
version = f"variant6{lr}_n{n}"
loss_training, acc_training, loss_val, acc_val = train(net, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, n, 1000, 1000, None,f"variant6{lr}_n{n}", t=6, pretrained=False)
plotting(version, loss_training, acc_training, loss_val, acc_val)