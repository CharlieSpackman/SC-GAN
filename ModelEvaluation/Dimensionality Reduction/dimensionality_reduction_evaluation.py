# dimensionality_reduction_evaluation.py

# Import the GAN class
from ...ModelCreation.TFGAN import *

# Import pyDRMetrics class
from .pyDRMetrics.pyDRMetrics import *

# import standard modules
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from pathlib import Path
import sys

# Update path to model for evaluation
CHECKPOINT_PATH = get_path("../../models")
MODEL_NAME = "2021-06-28_0.001_20_256_100_36" ### Update this as required
EPOCH = 20 ### Update this as required

model_path = get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/epochs/{EPOCH:05d}")
latest_model = tf.train.latest_checkpoint(get_path(f"{model_path}/ckpt"))

# Define parameters
SEED = 36
VAL_SIZE = 500


### Get data ###
# Read in data
data, _, _ = get_data("../../DataPreprocessing/GSE114727")

# Create a validations set
_, validation_data = train_test_split(data, test_size=(VAL_SIZE/data.shape[0]), random_state=SEED)


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
    print(f"No model found in {model_path}")
    sys.exit()


### Reduce the data using tha GAN ###
# Create a model which outputs the Discriminator hidden layer
discriminator_hidden = tf.keras.Model(discriminator.input, discriminator.layers[2].output)

# Reduce the dimensionality of the data using the Discriminator
validation_data_gan_reduced = discriminator_hidden(validation_data)

### Perform further dimensionality reduction ###
# Define a function to return a reduced dataset and the reconstruction error
def reduce_data(data, method):

    if method == "PCA":

        pca = PCA(n_components=2).fit(data)
        reduced_data = pca.transform(data)
        r_error = pca.inverse_transform(reduced_data)

        drm = DRMetrics(data, reduced_data, r_error)

        return reduced_data, r_error, drm


    elif method == "TSNE":

        tsne = TSNE(n_components=2).fit(data)
        reduced_data = tsne.transform(data)
        r_error = tsne.inverse_transform(reduced_data)

        drm = DRMetrics(data, reduced_data, r_error)

        return reduced_data, r_error, drm

    elif method == "UMAP":

        umap = UMAP(n_components=2).fit(data)
        reduced_data = umap.transform(data)
        r_error = umap.inverse_transform(reduced_data)

        drm = DRMetrics(data, reduced_data, r_error)

        return reduced_data, r_error, drm

    
# Reduce the dataset with PCA
validation_data_reduced_PCA, validation_data_reduced_PCA_error, validation_data_reduced_PCA_drm = reduce_data(validation_data, "PCA")
validation_data_gan_reduced_PCA, validation_data_gan_reduced_PCA_error, validation_data_gan_reduced_PCA_drm = reduce_data(validation_data_gan_reduced, "PCA")

# Reduce the dataset with t-SNE
validation_data_reduced_TSNE, validation_data_reduced_TSNE_error, validation_data_reduced_TSNE_drm = reduce_data(validation_data, "TSNE")
validation_data_gan_reduced_TSNE, validation_data_gan_reduced_TSNE_error, validation_data_gan_reduced_TSNE_drm = reduce_data(validation_data_gan_reduced, "TSNE")

# Reduce the dataset with umap
validation_data_reduced_UMAP, validation_data_reduced_UMAP_error, validation_data_reduced_UMAP_drm = reduce_data(validation_data, "UMAP")
validation_data_gan_reduced_UMAP, validation_data_gan_reduced_UMAP_error, validation_data_gan_reduced_UMAP_drm = reduce_data(validation_data_gan_reduced, "UMAP")


### Compute metrics ###
# Define a function to extract metrics from a pyDRMetrics object
def get_metrics(metrics):

    metrics_list = []

    # Reconstruction error
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

    # Create Pandas series
    return pd.Series(data = metrics_list, index=metrics_labels)

# Compute metrics for PCA
validation_data_reduced_PCA_metrics = get_metrics(validation_data_reduced_PCA_drm)
validation_data_gan_reduced_PCA_metrics = get_metrics(validation_data_gan_reduced_PCA_drm)

# Compute metrics for t-SNE
validation_data_reduced_TSNE_metrics = get_metrics(validation_data_reduced_TSNE_drm)
validation_data_gan_reduced_TSNE_metrics = get_metrics(validation_data_gan_reduced_TSNE_drm)

# Compute metrics for umap
validation_data_reduced_UMAP_metrics = get_metrics(validation_data_reduced_UMAP_drm)
validation_data_gan_reduced_UMAP_metrics = get_metrics(validation_data_gan_reduced_UMAP_drm)


### Combine and export metrics ###
col_names = [
    "PCA", "GAN+PCA",
    "TSNE", "GAN+TSNE",
    "UMAP", "GAN+UMAP"]

combined_metrics = pd.concat([
    validation_data_reduced_PCA_metrics,
    validation_data_gan_reduced_PCA_metrics,
    validation_data_reduced_TSNE_metrics,
    validation_data_gan_reduced_TSNE_metrics,
    validation_data_reduced_UMAP_metrics,
    validation_data_gan_reduced_UMAP_metrics],
    axis = 1).rename(columns=col_names)

# Save dataframe as csv
combined_metrics.to_csv(path = get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/dimensionality_reduction_metrics.csv"))


### Plot the data ###
# Plot the results for sample correlation
fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69))

# PCA plot
axs[0,0].scatter(validation_data_reduced_PCA[:,0], validation_data_reduced_PCA[:,1], label = "PCA")
axs[0,0].title.set_text(f"PCA")
axs[0,0].set_xlabel("PC1")
axs[0,0].set_ylabel("PC2")

# GAN + PCA plot
axs[0,1].scatter(validation_data_gan_reduced_PCA[:,0], validation_data_gan_reduced_PCA[:,1], label = "GAN + PCA")
axs[0,1].title.set_text(f"GAN + PCA")
axs[0,1].set_xlabel("PC1")
axs[0,1].set_ylabel("PC2")

# t-SNE plot
axs[1,0].scatter(validation_data_reduced_TSNE[:,0], validation_data_reduced_TSNE[:,1], label = "TSNE")
axs[1,0].title.set_text(f"t-SNE")
axs[1,0].set_xlabel("TSNE 1")
axs[1,0].set_ylabel("TSNE 2")

# GAN + t-SNE plot
axs[1,1].scatter(validation_data_gan_reduced_TSNE[:,0], validation_data_gan_reduced_TSNE[:,1], label = "GAN + TSNE")
axs[1,1].title.set_text(f"GAN + t-SNE")
axs[1,1].set_xlabel("TSNE 1")
axs[1,1].set_ylabel("TSNE 2")

# UMAP plot
axs[2,0].scatter(validation_data_reduced_UMAP[:,0], validation_data_reduced_UMAP[:,1], label = "UMAP")
axs[2,0].title.set_text(f"UMAP")
axs[2,0].set_xlabel("UMAP 1")
axs[2,0].set_ylabel("UMAP 2")

# GAN + UMAP plot
axs[2,1].scatter(validation_data_gan_reduced_UMAP[:,0], validation_data_gan_reduced_UMAP[:,1], label = "GAN + UMAP")
axs[2,1].title.set_text(f"GAN + UMAP")
axs[2,1].set_xlabel("UMAP 1")
axs[2,1].set_ylabel("UMAP 2")


fig.savefig(fname=get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/images/dimensionality_reduction_plot.png"))
plt.clf() 