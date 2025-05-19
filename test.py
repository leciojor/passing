import torch
import os
import matplotlib.pyplot as plt
from collections import Counter
from helpers import getting_loader
from archs import DeepQBVariant1
import re

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
else:
  DEVICE = torch.device("cpu")


def getting_results_distribution():

    for filename in os.listdir("models"):
        variant = int(re.search(r'variant(\d+)', filename).group(1))
        print(variant)
        train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=0, variant = variant, train_p=0.8, saved=True, distr_analysis=False, get_dataset=True)

        file_path = os.path.join("models", filename)
        state = torch.load(file_path, map_location=DEVICE)
        if variant == 5:
            output_dim = 1
        else:
            output_dim = 3
        model = DeepQBVariant1(input_dim=dataset.col_size - output_dim, output_dim=output_dim)
        model.load_state_dict(state)
        model.eval()

        predictions = []
        with torch.no_grad():
            for x, y in val_loader:
                y_hat = model(x)
                if variant == 5:
                    probs = torch.sigmoid(y_hat) 
                    preds = (probs >= 0.5)
                elif variant == 6:
                    probs = torch.softmax(y_hat, dim=1)
                    preds = torch.argmax(probs, dim=1)
                predictions.extend(preds.cpu().numpy().flatten().tolist())

        counter = Counter(predictions)
        labels = sorted(counter.keys())
        counts = [counter[label] for label in labels]
        plt.bar(labels, counts, color='skyblue')
        plt.title(f"Prediction Distribution - Variant {variant}")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.xticks(labels)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"distributions/models/distribution_variant{variant}_model {filename}.png")
        plt.show()

def getting_time_series_analysis():
    pass
