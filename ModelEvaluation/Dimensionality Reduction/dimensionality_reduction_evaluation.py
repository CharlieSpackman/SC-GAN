# dimensionality_reduction_evaluation.py

# import environment modules
from pathlib import Path
import sys

from tensorflow.python.ops.gen_array_ops import size
sys.path.append(str(Path("../../ModelCreation/").resolve()))
sys.path.append(str(Path("pyDRMetrics/").resolve()))

# Import the GAN class
from TFGAN import SCGAN, get_path, get_data

# Import pyDRMetrics class
from pyDRMetrics.pyDRMetrics import *

# import standard modules
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from pathlib import Path
import sys

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
point_size = 50
axis_size = 8


# Update path to model for evaluation
CHECKPOINT_PATH = get_path("../../models")
MODEL_NAME = "0.001_0.001_100_250_100_30" ### Update this as required
EPOCH = 100 ### Update this as required

model_path = get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/epochs/{EPOCH:05d}")
latest_model = tf.train.latest_checkpoint(get_path(f"{model_path}/ckpt"))

# Define parameters
SEED = 36
VAL_SIZE = 50


### Get data ###
# Read in data
data, cells, anno = get_data("../../DataPreprocessing/GSE114727/")

# Create a validations set
_, validation_data, _, validation_anno = train_test_split(data, anno, test_size=(VAL_SIZE), random_state=SEED)

labels = validation_anno["Celltype (major-lineage)"].tolist()


### Read in the model ###
# Create GAN object
gan = SCGAN(data, CHECKPOINT_PATH)

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
discriminator_hidden = tf.keras.Model(discriminator.input, discriminator.layers[2].output)
print("[INFO] Model created")

# Reduce the dimensionality of the data using the Discriminator
validation_data_gan_reduced = discriminator_hidden(validation_data).numpy()

### Perform further dimensionality reduction ###
# Define a function to return a reduced dataset and the reconstruction error
def reduce_data(data, method):

    if method == "PCA":

        pca = PCA(n_components=2).fit(data)
        reduced_data = pca.transform(data)
        r_error = pca.inverse_transform(reduced_data)

        drm = DRMetrics(data, reduced_data, r_error)

        return (reduced_data, r_error, drm)


    elif method == "TSNE":

        tsne = TSNE(n_components=2).fit(data)
        reduced_data = tsne.embedding_
        drm = DRMetrics(data, reduced_data, None)

        return (reduced_data, drm)

    elif method == "UMAP":

        umap = UMAP(n_components=2).fit(data)
        reduced_data = umap.transform(data)
        r_error = umap.inverse_transform(reduced_data)

        drm = DRMetrics(data, reduced_data, r_error)

        return (reduced_data, r_error, drm)

    return None

    
# Reduce the dataset with PCA
pca = reduce_data(validation_data, "PCA")
validation_data_reduced_PCA = pca[0]
validation_data_reduced_PCA_error = pca[1]
validation_data_reduced_PCA_drm = pca[2]

pca_gan = reduce_data(validation_data_gan_reduced, "PCA")
validation_data_gan_reduced_PCA = pca_gan[0]
validation_data_gan_reduced_PCA_error = pca_gan[1]
validation_data_gan_reduced_PCA_drm = pca_gan[2]
print("[INFO] PCA complete")

# Reduce the dataset with t-SNE
tsne = reduce_data(validation_data, "TSNE")
validation_data_reduced_TSNE = tsne[0]
validation_data_reduced_TSNE_drm = tsne[1]

tsne_gan = reduce_data(validation_data_gan_reduced, "TSNE")
validation_data_gan_reduced_TSNE = tsne_gan[0]
validation_data_gan_reduced_TSNE_drm = tsne_gan[1]
print("[INFO] t-SNE complete")

# Reduce the dataset with umap
umap = reduce_data(validation_data, "UMAP")
validation_data_reduced_UMAP = umap[0]
validation_data_reduced_UMAP_error = umap[1]
validation_data_reduced_UMAP_drm = umap[2]

umap_gan = reduce_data(validation_data_gan_reduced, "UMAP")
validation_data_gan_reduced_UMAP = umap_gan[0]
validation_data_gan_reduced_UMAP_error = umap_gan[1]
validation_data_gan_reduced_UMAP_drm = umap_gan[2]
print("[INFO] UMAP complete")

print("[INFO] Data has been reduced")


### Compute metrics ###
# Define a function to extract metrics from a pyDRMetrics object
def get_metrics(metrics, method = None):

    metrics_list = []

    # Reconstruction error
    if method == "TSNE":
        metrics_list.append(None)
        metrics_list.append(None)
    else:
        metrics_list.append(metrics.mse)
        metrics_list.append(metrics.rmse)

    # Residual variance
    metrics_list.append(metrics.Vr)
    metrics_list.append(metrics.Vrs)

    # co-ranking matrix metrics
    metrics_list.append(metrics.kmax)
    metrics_list.append(metrics.Qlocal)
    metrics_list.append(metrics.Qglobal)

    # Combine results into a Pandas Series
    metrics_labels = [
        "Reconstruction Error",
        "Relative Reconstruction Error",
        "Residual Variance (Pearson)",
        "Residual Variance (Spearman)",
        "LCMC",
        "Q-Local",
        "Q-Global"]
    
    metrics_list = np.array(metrics_list).reshape(1, -1)

    # Create Pandas series
    return pd.DataFrame(data = metrics_list, columns=metrics_labels)

# Compute metrics for PCA
validation_data_reduced_PCA_metrics = get_metrics(validation_data_reduced_PCA_drm)
validation_data_gan_reduced_PCA_metrics = get_metrics(validation_data_gan_reduced_PCA_drm)

# Compute metrics for t-SNE
validation_data_reduced_TSNE_metrics = get_metrics(validation_data_reduced_TSNE_drm, method = "TSNE")
validation_data_gan_reduced_TSNE_metrics = get_metrics(validation_data_gan_reduced_TSNE_drm, method = "TSNE")

# Compute metrics for umap
validation_data_reduced_UMAP_metrics = get_metrics(validation_data_reduced_UMAP_drm)
validation_data_gan_reduced_UMAP_metrics = get_metrics(validation_data_gan_reduced_UMAP_drm)


### Combine and export metrics ###
row_names = pd.Series([
    "PCA", "GAN+PCA",
    "TSNE", "GAN+TSNE",
    "UMAP", "GAN+UMAP"], name = "Index")

combined_metrics = pd.concat([
    validation_data_reduced_PCA_metrics,
    validation_data_gan_reduced_PCA_metrics,
    validation_data_reduced_TSNE_metrics,
    validation_data_gan_reduced_TSNE_metrics,
    validation_data_reduced_UMAP_metrics,
    validation_data_gan_reduced_UMAP_metrics],
    axis = 0).reset_index(drop = True)

combined_metrics["Index"] = row_names
combined_metrics = combined_metrics.set_index("Index").round(2)

# Save dataframe as csv
combined_metrics.to_csv(get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/dimensionality_reduction_metrics_{EPOCH:05d}.csv"))
print("[INFO] Evaluation metrics created")


### Plot the data ###
# Create a mapping between classes and colours
color_map = {
    "CD4Tconv"   : "#00798c",
    "CD8T"    : "#d1495b",
    "CD8Tex" : "#edae49",
    "Tprolif"  : "#66a182",
    "Treg"   : "#2e4057"}

patchList = []
# for key in color_map:
#         data_key = mpatches.Patch(color=color_map[key], label=key, hatch = ".")
#         patchList.append(data_key)

for key in color_map:
        data_key = plt.scatter([],[], s = point_size*3, marker=".", color = color_map[key], label=key)
        patchList.append(data_key)

plot_colors = list(map(color_map.get, labels))

# Plot the results for sample correlation
axis_size = 8.0
plot_ratios = {'height_ratios': [1,1,1.1], 'width_ratios': [1,1]}
fig, axs = plt.subplots(3, 2, figsize=(axis_size*3, axis_size*3), gridspec_kw=plot_ratios, squeeze=True)

# PCA plot
axs[0,1].scatter(validation_data_reduced_PCA[:,0], validation_data_reduced_PCA[:,1], c = plot_colors, s = point_size)
axs[0,1].title.set_text(f"PCA")
axs[0,1].set_xlabel("PC 1")
axs[0,1].set_ylabel("PC 2")

# GAN + PCA plot
axs[0,0].scatter(validation_data_gan_reduced_PCA[:,0], validation_data_gan_reduced_PCA[:,1], c = plot_colors, s = point_size)
axs[0,0].title.set_text(f"GAN + PCA")
axs[0,0].set_xlabel("PC 1")
axs[0,0].set_ylabel("PC 2")

# t-SNE plot
axs[1,1].scatter(validation_data_reduced_TSNE[:,0], validation_data_reduced_TSNE[:,1], c = plot_colors, s = point_size)
axs[1,1].title.set_text(f"t-SNE")
axs[1,1].set_xlabel("TSNE 1")
axs[1,1].set_ylabel("TSNE 2")

# GAN + t-SNE plot
axs[1,0].scatter(validation_data_gan_reduced_TSNE[:,0], validation_data_gan_reduced_TSNE[:,1], c = plot_colors, s = point_size)
axs[1,0].title.set_text(f"GAN + t-SNE")
axs[1,0].set_xlabel("TSNE 1")
axs[1,0].set_ylabel("TSNE 2")

# UMAP plot
axs[2,1].scatter(validation_data_reduced_UMAP[:,0], validation_data_reduced_UMAP[:,1], c = plot_colors, s = point_size)
axs[2,1].title.set_text(f"UMAP")
axs[2,1].set_xlabel("UMAP 1")
axs[2,1].set_ylabel("UMAP 2")
box = axs[2,1].get_position()
axs[2,1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# GAN + UMAP plot
axs[2,0].scatter(validation_data_gan_reduced_UMAP[:,0], validation_data_gan_reduced_UMAP[:,1], label = "GAN + UMAP", c = plot_colors, s = point_size)
axs[2,0].title.set_text(f"GAN + UMAP")
axs[2,0].set_xlabel("UMAP 1")
axs[2,0].set_ylabel("UMAP 2")
box = axs[2,0].get_position()
axs[2,0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

fig.legend(handles = patchList, loc = "lower center", ncol = 5, frameon = False, markerscale=2, bbox_to_anchor = [0.5, 0.075])

fig.savefig(fname=get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/images/dimensionality_reduction_plot_{EPOCH:05d}.png"))
plt.clf() 
print("[INFO] Evaluation plot saved")