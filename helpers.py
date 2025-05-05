
from data import PlaysData
from torch.utils.data import DataLoader, Subset

def getting_loader(batch_size, display=False, num_workers=2, variant = 1, train_p=0.7):
    dataset = PlaysData(variant)
    if display:
        dataset.get_csv()

    n = len(dataset)
    train_amount = n*train_p
    train_indices = list(range(train_amount + 1))
    val_indices = list(range(train_amount + 1, n))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
    
def train():
    pass

getting_loader(32, True)

