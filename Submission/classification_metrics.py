# evaluation_metrics.py
"""
Create evaluation metrics based on predictions from scPred.

Inputs
------
    
    MODEL_NAME : string
        file name of the trained model
    EPOCH : int
        epoch number that will be evaluated
    MODEL_PATH : string
        folder path to where the models are saved

Outputs
-------
    metrics_arr : DataFrame
        computed classification metrics


Functions
---------
    compute_metrics
        Takes predictions as inputs and calculates classification metrics
"""

# Import modules
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# Set matplotlib settings
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)        
plt.rc('axes', titlesize=MEDIUM_SIZE)   
plt.rc('axes', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=SMALL_SIZE)  
plt.rc('ytick', labelsize=SMALL_SIZE)  
plt.rc('legend', fontsize=MEDIUM_SIZE)  
plt.rc('figure', titlesize=BIGGER_SIZE)

# Evaluation parameters
MODEL_NAME = "5e-05_300000_128_100_1001" ### Update this as required
EPOCH = 150000 ### Update this as required

# Paths for reduced data
MODEL_PATH = "C:\\Users\\spack\\OneDrive - King's College London\\Individual Project\\Single Cell Sequencing with GANs\\Implementation\\models" ### Update this as required
METRICS_PATH = f"{MODEL_PATH}\\{MODEL_NAME}\\metrics\\"

print("[INFO] Reading data")
# Read in predictions and labels for scPred
scPred_gan_reduced = pd.read_csv(METRICS_PATH + f"scPred_reduced_gan_predictions_{EPOCH:05d}.csv")
scPred_baseline = pd.read_csv(METRICS_PATH + f"scPred_baseline_predictions_{EPOCH:05d}.csv")


# Create plots for scPred
axis_size = 8.0
plot_ratios = {'height_ratios': [1], 'width_ratios': [1,1]}
fig, axes = plt.subplots(1, 2, figsize=(axis_size*3, axis_size*2), gridspec_kw=plot_ratios, squeeze=True)
plt.xticks(rotation=90)

# Define a function to compute performance metrics for each dataset
def compute_metrics(data, axes, ax_title):
    """
    Computes classification metrics based on a set of predictions and correct labels.
    Creates a confusion matrix with GAN reduced and baseline performance.

    Parameters
    ----------
        data : array
            predictions and correct labels
        axes : plt.Axes
            name of the plt.Axes object that the plot will be saved to
        ax_title : string
            title of the plot

    Returns
    ----------
        metrics_arr : DataFrame
            set of classification metrics
        ax : plt.Axes
            plot object
    """

    # Split out the predictions and the labels
    y_true = data["labels"]
    y_pred = data["predictions"]

    # Define the class labels
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
    ax = metrics.ConfusionMatrixDisplay(cm, display_labels = class_labels).plot(ax=axes, colorbar=False, cmap = "YlGn").ax_
    ax.set_title(ax_title)
    ax.set_xticklabels(class_labels, rotation=90, ha='right')

    # Return the metrics and axes object
    return metrics_arr, ax

print("[INFO] Computing metrics")
# Get the metrics and plots for the datasets
scPred_gan_reduced_metrics, scPred_gan_reduced_plot = compute_metrics(scPred_gan_reduced, axes[0], "scPred: GAN Reduced")
scPred_baseline_metrics, scPred_baseline_plot = compute_metrics(scPred_baseline, axes[1], "scPred: Baseline")

# Reposition the chart objects
box = scPred_gan_reduced_plot.get_position()
scPred_gan_reduced_plot.set_position([box.x0 - box.width * 0.05, box.y0, box.width * 0.9, box.height])

box = scPred_baseline_plot.get_position()
scPred_baseline_plot.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Save plot
plt.savefig(f"{MODEL_PATH}\\{MODEL_NAME}\\images\\confusion_matrices_{EPOCH:05d}.png")

# Create a dataframe with all metrics
col_names = [
    "scPred_gan_reduced",
    "scPred_baseline"
]

metrics_arr = pd.concat(
    [scPred_gan_reduced_metrics,
    scPred_baseline_metrics],
    axis = 1)

metrics_arr.columns = col_names

# Save metrics as a csv
metrics_arr.to_csv(METRICS_PATH + f"evaluation_metrics_{EPOCH:05d}.csv")

print("[INFO] Metrics saved")