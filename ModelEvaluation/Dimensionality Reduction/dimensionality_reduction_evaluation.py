# dimensionality_reduction_evaluation.py

# import environment modules
from pathlib import Path
import sys

sys.path.append(str(Path("../../ModelCreation/").resolve()))
sys.path.append(str(Path("pyDRMetrics/").resolve()))

# Import the GAN class
from WGANGP import WGANGP, _get_path, _rescale_arr

# Import pyDRMetrics class
from pyDRMetrics.pyDRMetrics import *

# import standard modules
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
CHECKPOINT_PATH = _get_path("../../models")
MODEL_NAME = "5e-05_50000_64_100_205" ### Update this as required
EPOCH = 160000 ### Update this as required
SEED = 205

model_path = _get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/epochs/{EPOCH:05d}")
latest_model = tf.train.latest_checkpoint(_get_path(f"{model_path}"))


### Read in the model ###
# Create GAN object
gan = WGANGP(seed = SEED, ckpt_path = CHECKPOINT_PATH, data_path = _get_path("../../DataPreprocessing/GSE114725"))

X_train, X_test, y_train, y_test = gan.init_data("GSE114725_processed_data_10000_2000.csv", "GSE114725_processed_annotations_10000_2000.csv")


X_test = np.concatenate((X_train, X_test), axis = 0)
y_test = np.concatenate((y_train, y_test), axis = 0)

gan.init_networks()

# Get models
generator, discriminator = gan.get_models()

# Get checkpoint object
checkpoint = gan.get_checkpoint_obj()

# Load the model
if Path(model_path).exists:
    print("[INFO] Model dir exists")
    checkpoint.restore(latest_model)

else:
    print(f"[ERROR] No model found in {model_path}")
    sys.exit()


### Reduce the data using tha GAN ###
# Create a model which outputs the Discriminator hidden layer
discriminator_hidden = tf.keras.Model(discriminator.layers[1].layers[0].input, discriminator.layers[1].layers[1].output)
print("[INFO] Model created")

# Reduce the dimensionality of the data using the Discriminator
X_test_reduced = discriminator_hidden(X_test).numpy()

### Perform further dimensionality reduction ###
print("[INFO] Reducing dimensions...")

# Reduce the datasets with PCA
pca = PCA(n_components=2).fit_transform(X_test)
pca_gan = PCA(n_components=2).fit_transform(X_test_reduced)
print("[INFO] PCA complete")

pcs = PCA(n_components=50).fit_transform(X_test)
# pcs_reduced = PCA(n_components=50).fit_transform(X_test_reduced)

# Reduce the datasets with TSNE
tsne = _rescale_arr(TSNE().fit_transform(pcs))
tsne_gan = _rescale_arr(TSNE().fit_transform(X_test_reduced))
print("[INFO] t-SNE complete")

# Reduce the datasets with UMAP
umap = _rescale_arr(UMAP().fit_transform(pcs))
umap_gan = _rescale_arr(UMAP().fit_transform(X_test_reduced))
print("[INFO] UMAP complete")

print("[INFO] Data has been reduced")

### Compute metrics ###
# Define a function to extract metrics
def get_metrics(data, labels):

    metrics_list = []

    metrics_list.append(silhouette_score(data, labels))
    metrics_list.append(calinski_harabasz_score(data, labels))

    # Combine results into a Pandas Series
    metrics_labels = [
        "Silhouette Score",
        "Calinski-Harabasz Score"
        ]
    
    metrics_list = np.array(metrics_list).reshape(1, -1)

    # Create Pandas series
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


### Combine and export metrics ###
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
combined_metrics.to_csv(_get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/metrics/dimensionality_reduction_metrics_{EPOCH:05d}.csv"))
print("[INFO] Evaluation metrics created")


### Plot the data ###
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
axs[2,0].scatter(umap_gan[:,0], umap_gan    [:,1], label = "GAN + UMAP", c = plot_colors, s = point_size)
axs[2,0].title.set_text(f"GAN + UMAP")
axs[2,0].set_xlabel("UMAP 1")
axs[2,0].set_ylabel("UMAP 2")
box = axs[2,0].get_position()
axs[2,0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

fig.legend(handles = patchList, loc = "lower center", ncol = 9, frameon = False, markerscale=2, bbox_to_anchor = [0.5, 0.075])

fig.savefig(fname=_get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/images/dimensionality_reduction_plot_{EPOCH:05d}.png"))
plt.clf() 
print("[INFO] Evaluation plot saved")