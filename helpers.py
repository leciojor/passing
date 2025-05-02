
from data import PlaysData


def getting_loader():
    dataset = PlaysData(1)
    dataset.get_csv()

getting_loader()