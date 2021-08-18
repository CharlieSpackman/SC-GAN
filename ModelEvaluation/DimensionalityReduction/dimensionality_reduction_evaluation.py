# dimensionality_reduction_evaluation.py
"""
Creates a neural network from the Discriminator where the final hidden layer is the outputs.
Reduces the validation set and calculates cluster metrics compared to the baseline data.
Creates plots of the GAN reduced data and baseline data.
Creates expression profile plots for Discriminator outputs.

Inputs
------
    CHECKPOINT_PATH : string
        folder path for where the models are saved
    EPOCH : int
        epoch that will be evaluated
    SEED : int
        random state seed
    
Outputs
-------
    Dimensionality reduction metrics
    Dimensionality reduction plots
    Reduced GAN data
    Discriminator output examples

Functions 
---------
    get_metrics
        Evaluates cluster performance
"""

# import environment modules
from pathlib import Path
import sys

# Add the WGANGP class to the Python path
sys.path.append(str(Path("../../ModelCreation/").resolve()))

# Import the WGANGP class and helper functions
from WGANGP import WGANGP, get_path, rescale_arr

# import analysis modules
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP

# import cluster metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Set matplotlib settings
plt.style.use('ggplot')
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
red = "#d1495b"
blue = "#00798c"
point_size = 25
axis_size = 8

# Update path to model for evaluation
CHECKPOINT_PATH = get_path("../../models") ### Update this as required
MODEL_NAME = "5e-05_300000_128_100_1001" ### Update this as required
EPOCH = 150000 ### Update this as required
SEED = 1001 ### Update this as required

# Get the latest saved model
model_path = get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/epochs/{EPOCH:05d}")
latest_model = tf.train.latest_checkpoint(get_path(f"{model_path}"))

#-----------------------
# Read Models
#-----------------------

# Create WGANGP object
gan = WGANGP(seed = SEED, ckpt_path = CHECKPOINT_PATH, data_path = get_path("../../DataPreprocessing/GSE114725"))

# Get the data
X_train, X_test, y_train, y_test = gan.init_data("GSE114725_processed_data_10000_3912.csv", "GSE114725_processed_annotations_10000_3912.csv")

# Combine training and validation sets
X = np.concatenate((X_train, X_test), axis = 0)
y = np.concatenate((y_train, y_test), axis = 0)

# Initialise the networks
gan.init_networks()

# Get models
generator, discriminator = gan.get_models()

# Get checkpoint object
checkpoint = gan.get_checkpoint_obj()

# Load the model
if latest_model == None:
    print(f"[ERROR] No model found in {model_path}")
    sys.exit()
    
else:
    print("[INFO] Model dir exists")
    checkpoint.restore(latest_model)


#-----------------------
# Compute Metrics
#-----------------------

# Create a model which outputs the first Discriminator hidden layer
discriminator_hidden = tf.keras.Model(discriminator.layers[1].layers[0].input, discriminator.layers[1].layers[1].output)
print("[INFO] Model created")

# Pass validation data through the Discriminator
X_test_reduced = discriminator_hidden(X_test).numpy()

# Perform dimensionality reduction
print("[INFO] Reducing dimensions...")

# Reduce the datasets with PCA
pca = rescale_arr(PCA(n_components=2).fit_transform(X_test))
pca_gan = rescale_arr(PCA(n_components=2).fit_transform(X_test_reduced))
print("[INFO] PCA complete")

# Get 50 PCs for t-SNE and UMAP
pcs = PCA(n_components=50).fit_transform(X_test)
pcs_reduced = PCA(n_components=50).fit_transform(X_test_reduced)

# Reduce the datasets with t-SNE
tsne = rescale_arr(TSNE().fit_transform(pcs))
tsne_gan = rescale_arr(TSNE().fit_transform(pcs_reduced))
print("[INFO] t-SNE complete")

# Reduce the datasets with UMAP
umap = rescale_arr(UMAP().fit_transform(pcs))
umap_gan = rescale_arr(UMAP().fit_transform(pcs_reduced))
print("[INFO] UMAP complete")

print("[INFO] Data has been reduced")

# Define a function to extract metrics
def get_metrics(data, labels):
    """
    Calculates cluster metrics for given data and labels.

    Parameters
    ----------
        data : array
            data to be evaluated
        labels : array
            cluster labels corresponding to the data

    Returns
    ----------
    metrics_list : DataFrame
        metrics computed from cluster evaluation
    """
    # Create a blank list
    metrics_list = []

    # Compute the Silhouette Score and Calinkski-Harabasz Index
    metrics_list.append(silhouette_score(data, labels))
    metrics_list.append(calinski_harabasz_score(data, labels))

    # Define column labels
    metrics_labels = [
        "Silhouette Score",
        "Calinski-Harabasz Score"
        ]
    
    # Convert lists to arrays and reshape
    metrics_list = np.array(metrics_list).reshape(1, -1)

    # Return a DataFrame
    return pd.DataFrame(data = metrics_list, columns=metrics_labels)

# Compute metrics for PCA
pca_metrics = get_metrics(pca, y_test)
pca_gan_metrics = get_metrics(pca_gan, y_test)

# Compute metrics for t-SNE
tsne_metrics = get_metrics(tsne, y_test)
tsne_gan_metrics = get_metrics(tsne_gan, y_test)

# Compute metrics for umap
umap_metrics = get_metrics(umap, y_test)
umap_gan_metrics = get_metrics(umap_gan, y_test)


# Combine and export metrics
row_names = pd.Series([
    "PCA", "GAN+PCA",
    "TSNE", "GAN+TSNE",
    "UMAP", "GAN+UMAP"], name = "Index")

combined_metrics = pd.concat([
    pca_metrics,
    pca_gan_metrics,
    tsne_metrics,
    tsne_gan_metrics,
    umap_metrics,
    umap_gan_metrics],
    axis = 0).reset_index(drop = True)

combined_metrics["Index"] = row_names
combined_metrics = combined_metrics.set_index("Index").round(2)

# Save dataframe as csv
combined_metrics.to_csv(get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/metrics/dimensionality_reduction_metrics_{EPOCH:05d}.csv"))
print("[INFO] Evaluation metrics created")

#-----------------------
# Plot the Data
#-----------------------

# Create a mapping between classes and colours
color_map = {
    "B"   : "#00798c",
    "MACROPHAGE"    : "#C77CFF",
    "MAST" : "#edae49",
    "NEUTROPHIL"  : "#66a182",
    "NK"   : "#2e4057",
    "NKT": "#8c8c8c",
    "T": "#f37735",
    "mDC": "#d11141",
    "pDC":"#A6611A"}

patchList = []
for key in color_map:
        data_key = plt.scatter([],[], s = point_size*3, marker=".", color = color_map[key], label=key)
        patchList.append(data_key)

plot_colors = list(map(color_map.get, y_test))

# Plot the results for sample correlation
axis_size = 8.0
plot_ratios = {'height_ratios': [1,1,1.1], 'width_ratios': [1,1]}
fig, axs = plt.subplots(3, 2, figsize=(axis_size*3, axis_size*3), gridspec_kw=plot_ratios, squeeze=True)

# PCA plot
axs[0,1].scatter(pca[:,0], pca[:,1], c = plot_colors, s = point_size)
axs[0,1].title.set_text(f"PCA")
axs[0,1].set_xlabel("PC 1")
axs[0,1].set_ylabel("PC 2")

# GAN + PCA plot
axs[0,0].scatter(pca_gan[:,0], pca_gan[:,1], c = plot_colors, s = point_size)
axs[0,0].title.set_text(f"GAN + PCA")
axs[0,0].set_xlabel("PC 1")
axs[0,0].set_ylabel("PC 2")

# t-SNE plot
axs[1,1].scatter(tsne[:,0], tsne[:,1], c = plot_colors, s = point_size)
axs[1,1].title.set_text(f"t-SNE")
axs[1,1].set_xlabel("TSNE 1")
axs[1,1].set_ylabel("TSNE 2")

# GAN + t-SNE plot
axs[1,0].scatter(tsne_gan[:,0], tsne_gan[:,1], c = plot_colors, s = point_size)
axs[1,0].title.set_text(f"GAN + t-SNE")
axs[1,0].set_xlabel("TSNE 1")
axs[1,0].set_ylabel("TSNE 2")

# UMAP plot
axs[2,1].scatter(umap[:,0], umap[:,1], c = plot_colors, s = point_size)
axs[2,1].title.set_text(f"UMAP")
axs[2,1].set_xlabel("UMAP 1")
axs[2,1].set_ylabel("UMAP 2")
box = axs[2,1].get_position()
axs[2,1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# GAN + UMAP plot
axs[2,0].scatter(umap_gan[:,0], umap_gan[:,1], label = "GAN + UMAP", c = plot_colors, s = point_size)
axs[2,0].title.set_text(f"GAN + UMAP")
axs[2,0].set_xlabel("UMAP 1")
axs[2,0].set_ylabel("UMAP 2")
box = axs[2,0].get_position()
axs[2,0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

fig.legend(handles = patchList, loc = "lower center", ncol = 9, frameon = False, markerscale=2, bbox_to_anchor = [0.5, 0.075])

fig.savefig(fname=get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/images/dimensionality_reduction_plot_{EPOCH:05d}.png"))
plt.clf() 
print("[INFO] Evaluation plot saved")

#-----------------------
# Reduce the Data
#-----------------------

# Reduce the dimensionality of the full dataset using the Discriminator
data_gan_reduced = discriminator_hidden(X).numpy()

# Save the data
col_names = [f"Component {i+1}" for i in range(data_gan_reduced.shape[1])]
export_data = pd.DataFrame(data = data_gan_reduced, columns=col_names)

export_data.to_csv(
    get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/data_reduced_gan_{EPOCH:05d}.csv"), 
    sep=",",
    index=False)

print("[INFO] Data has been reduced and saved")

#-----------------------
# Expression Plots
#-----------------------

# Get some examples of the Discriminator output
values = export_data.values[:3]

# Define the plot
plot_ratios = {'height_ratios': [1, 1, 1], 'width_ratios': [1]}
fig, axs = plt.subplots(3, 1, figsize=(axis_size*2, axis_size*3), gridspec_kw=plot_ratios, squeeze=True)

# Example 1
axs[0].plot(values[0], c = red)
axs[0].set_ylabel("Expression")

# Example 2
axs[1].plot(values[1], c = red)
axs[1].set_ylabel("Expression")

# Example 3
axs[2].plot(values[2], c = red)
axs[2].set_ylabel("Expression")
axs[2].set_xlabel("Genes")

fig.savefig(fname=get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/images/dimensionality_reduction_expression_examples_{EPOCH:05d}.png"))
plt.clf() 

print("[INFO] Discriminator example outputs created and plotted")