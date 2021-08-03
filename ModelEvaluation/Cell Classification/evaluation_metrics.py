# evaluation_metrics.py

# Import modules
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# Evaluation parameters
MODEL_NAME = "5e-05_50000_64_100_205" ### Update this as required
EPOCH = 160000 ### Update this as required

# Paths for reduced data
MODEL_PATH = "C:\\Users\\spack\\OneDrive - King's College London\\Individual Project\\Single Cell Sequencing with GANs\\Implementation\\models\\"
METRICS_PATH = f"{MODEL_PATH}\\{MODEL_NAME}\\metrics\\"

print("[INFO] Reading data")
# Read in predictions and labels for moana
moana_gan_reduced = pd.read_csv(METRICS_PATH + f"moana_reduced_gan_predictions_{EPOCH:05d}.csv")
moana_baseline = pd.read_csv(METRICS_PATH + f"moana_baseline_predictions_{EPOCH:05d}.csv")

# Read in predictions and labels for scPred
scPred_gan_reduced = pd.read_csv(METRICS_PATH + f"scPred_reduced_gan_predictions_{EPOCH:05d}.csv")
scPred_baseline = pd.read_csv(METRICS_PATH + f"scPred_baseline_predictions_{EPOCH:05d}.csv")


# Create plots for Moana and scPred
axis_size = 8.0
plot_ratios = {'height_ratios': [1,1], 'width_ratios': [1,1]}
fig, axes = plt.subplots(2, 2, figsize=(axis_size*2, axis_size*2), gridspec_kw=plot_ratios, squeeze=True)

# Define a function to compute performance metrics for each dataset
def compute_metrics(data, axes, ax_title):

    # Split out the predictions and the labels
    y_true = data["labels"]
    y_pred = data["predictions"]

    class_labels = [
        "B",
        "MACROPHAGE",
        "MAST",
        "NEUTROPHIL",
        "NK",
        "NKT",
        "T",
        "mDC",
        "pDC"
        ]


    # Compute accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    # Compute Precision, Recall and F1 scores
    precision = metrics.precision_score(y_true, y_pred, average="macro")
    recall = metrics.recall_score(y_true, y_pred, average="macro")
    f1 = metrics.f1_score(y_true, y_pred, average="macro")

    # Create a Series with the metrics
    metrics_arr = pd.Series({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    # Get the confusion matrix plot
    ax = metrics.ConfusionMatrixDisplay(cm, display_labels = class_labels).plot(ax=axes, colorbar=False).ax_
    ax.set_title(ax_title)

    return metrics_arr, ax

print("[INFO] Computing metrics")
# Get the metrics and plots for the datasets
moana_gan_reduced_metrics, moana_gan_reduced_plot = compute_metrics(moana_gan_reduced, axes[0,0], "Moana: GAN Reduced")
moana_baseline_metrics, moana_baseline_plot = compute_metrics(moana_baseline, axes[0,1], "Moana: Baseline")
scPred_gan_reduced_metrics, scPred_gan_reduced_plot = compute_metrics(scPred_gan_reduced, axes[1,0], "scPred: GAN Reduced")
scPred_baseline_metrics, scPred_baseline_plot = compute_metrics(scPred_baseline, axes[1,1], "scPred: Baseline")

# Save plot
plt.savefig(f"{MODEL_PATH}\\{MODEL_NAME}\\images\\confusion_matrices_{EPOCH:05d}.png")

# Create a dataframe with all metrics
col_names = [
    "moana_gan_reduced",
    "moana_baseline",
    "scPred_gan_reduced",
    "scPred_baseline"
]

metrics_arr = pd.concat(
    [moana_gan_reduced_metrics,
    moana_baseline_metrics,
    scPred_gan_reduced_metrics,
    scPred_baseline_metrics],
    axis = 1)

metrics_arr.columns = col_names

# Save metrics as a csv
metrics_arr.to_csv(METRICS_PATH + f"evaluation_metrics_{EPOCH:05d}.csv")

print("[INFO] Metrics saved")