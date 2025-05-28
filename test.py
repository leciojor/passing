import torch
import os
import matplotlib.pyplot as plt
from collections import Counter
from helpers import getting_loader
from archs import DeepQBVariant1
import re
import seaborn as sns
import numpy as np

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
else:
  DEVICE = torch.device("cpu")


def getting_results_distribution():

    for filename in os.listdir("models"):
        variant = int(re.search(r'variant(\d+)', filename).group(1))
        if variant == 2:
            continue

        drop = not  "shoulder" in filename and variant == 1
        train_loader, val_loader, dataset = getting_loader(16, save=False, num_workers=0, variant = variant, train_p=0.8, saved=True, distr_analysis=False, get_dataset=True, drop_qb_orientation=drop)

        file_path = os.path.join("models", filename)
        state = torch.load(file_path, map_location=DEVICE)
        if variant == 5 or variant == 2:
            output_dim = 1
        elif variant == 1:
            output_dim = 5
        elif variant == 6:
            output_dim=3
        model = DeepQBVariant1(input_dim=dataset.col_size - output_dim, output_dim=output_dim)
        model.load_state_dict(state)
        model.eval()

        predictions = []
        with torch.no_grad():
            for x, y in val_loader:
                y_hat = model(x)
                if variant == 5 or variant == 3:
                    probs = torch.sigmoid(y_hat) 
                    preds = (probs >= 0.5)
                elif variant == 6 or variant == 1:
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

def shoulder_orientation_feature_correlation_analysis():
    loader, dataset = getting_loader(1, save=False, num_workers=0, variant = 1, train_p=0.8, saved=True, distr_analysis=False, get_dataset=True, drop_qb_orientation=False, cleaning=False, split=False, passed_result_extra=True)
    dataset.data.dropna(subset=['result', 'qb_x', 'qb_y', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4'], inplace=True)
    n = len(dataset.data)
    orientation = []
    projected_orientations = []
    differences = []
    for i in range(n):
        row = dataset.data.iloc[i]
        pass_result = row["passResultExtra"]
        if pass_result == "C":
            receiver = int(row["result"])
            x_qb = row["qb_x"]
            y_qb = row["qb_y"]
            x_receiver = row[f"x_{receiver}"]
            y_receiver = row[f"y_{receiver}"]

            dx = x_receiver - x_qb
            dy = y_receiver - y_qb

            projected = (np.degrees(np.arctan2(dy, dx))) % 360
            projected_orientations.append(projected)
            qb_orientation = row["qb_orientation"]
            orientation.append(qb_orientation)
            differences.append(abs(projected - qb_orientation))


    plt.figure(figsize=(8, 6))
    plt.hexbin(orientation, projected_orientations, gridsize=60, cmap='viridis', mincnt=1)
    plt.colorbar(label='Counts')
    plt.xlabel("QB ACTUAL orientation")
    plt.ylabel("QB projected orientation based on intended Receiver")
    plt.title(f"Analysis of QB orientation and projected orientation based on intented receiver")
    plt.tight_layout()
    plt.savefig("moreAnalysis/actualOrientationVsProjected.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(differences, bins=100, kde=True)
    plt.title("Diffences between QB actual orientation and QB projected orientation")
    plt.xlabel("Angular Difference")
    plt.savefig("moreAnalysis/angularDifference.png")
    plt.show()

def getting_time_series_analysis_binary_classification(model_file, i=4):
    variant = 5
    loader, dataset = getting_loader(1, save=False, num_workers=0, variant = variant, train_p=0.8, saved=False, distr_analysis=False, get_dataset=True, all_frames=True, i=i, split=False)
    state = torch.load(model_file, map_location=DEVICE)
    output_dim = 1
    model = DeepQBVariant1(input_dim=dataset.col_size - output_dim, output_dim=output_dim)
    model.load_state_dict(state)
    model.eval()

    predictions = []
    with torch.no_grad():
        i_ = 0
        for x, y in loader:
            y_hat = model(x)
            probs = torch.sigmoid(y_hat) 
            predictions.extend(probs.cpu().numpy().flatten().tolist())
            i_+=1
    
    plt.title(f"Time Series of pass completion probability {model_file[-20:]} instance {i}")
    plt.plot(list(range(i_)), predictions)
    plt.savefig(f"timeseries/timeseries_analysis_model{model_file[-20:]}instance {i}.png")
    plt.show()

def further_analysis_results_variant_two(file_path):
        loader, dataset = getting_loader(1, save=False, num_workers=0, variant = 2, train_p=0.8, saved=True, distr_analysis=False, get_dataset=True, drop_qb_orientation=False, split=False)

        state = torch.load(file_path, map_location=DEVICE)
        output_dim = 1
        model = DeepQBVariant1(input_dim=dataset.col_size - output_dim, output_dim=output_dim)
        model.load_state_dict(state)
        model.eval()

        results = []
        actual_gained_yards = []
        differences = []

        for x, y in loader:
            y_hat = model(x)
            results.append(y_hat.squeeze().item())
            actual_gained_yards.append(y.squeeze().item())
            diff = abs(y-y_hat)
            differences.append(diff.squeeze().item())

        plt.figure(figsize=(8, 6))
        plt.hexbin(actual_gained_yards, results, gridsize=60, cmap='viridis', mincnt=1)
        plt.colorbar(label='Counts')
        plt.xlabel("Actual Yards Gained")
        plt.ylabel("Model Predicted Yards Gained")
        plt.title(f"Variant 2 results")
        plt.tight_layout()
        plt.savefig("moreAnalysis/variant2Results.png")
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.histplot(differences, bins=100, kde=True)
        plt.title("Diffences between Model Predictions and actual Yards gained")
        plt.xlabel("Difference")
        plt.savefig("moreAnalysis/yardsGainedDifference.png")
        plt.show()



def getting_time_series_analysis_multi_class_classification(model_file):
    pass

def getting_time_series_analysis_for_each_receiver(model_file):
    pass

# shoulder_orientation_feature_correlation_analysis()
further_analysis_results_variant_two("models/model_variant2_lr0.01_n250000.pkl")

# for filename in os.listdir("models"):
#     file_path = os.path.join("models", filename)
#     variant = int(re.search(r'variant(\d+)', filename).group(1))
#     if variant == 5:
#         for i in [4,6]:
#             getting_time_series_analysis_binary_classification(file_path, i=i)




    

