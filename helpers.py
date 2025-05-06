
from data import PlaysData
from torch.utils.data import DataLoader, Subset
import pandas as pd

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
    
def train():
    pass

getting_loader(32, True, saved=True)

