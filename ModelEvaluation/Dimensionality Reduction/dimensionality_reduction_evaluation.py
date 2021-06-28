# dimensionality_reduction_evaluation.py

# import environment modules
from pathlib import Path
import sys
sys.path.append(str(Path("../../ModelCreation/").resolve()))

# Import the GAN class
from TFGAN import SCGAN, get_path

# import modules
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Update path to model for evaluation
CHECKPOINT_PATH = get_path("../../models")
MODEL_NAME = "2021-06-28_0.001_20_256_100_36" ### Update this as required
EPOCH = 20 ### Update this as required

model_path = get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/epochs/{EPOCH:05d}")
latest_model = tf.train.latest_checkpoint(get_path(f"{model_path}/ckpt"))

# Define parameters
SEED = 36
VAL_SIZE = 500

# Read in data
data = pd.read_csv(get_path("../../data.csv"), delimiter=",")
data = data.iloc[:, 1:].to_numpy()
print("[INFO] Data successfully loaded")

# Create a validations set
_, validation_data = train_test_split(data, test_size=(VAL_SIZE/data.shape[0]), random_state=SEED)

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

# Create a model which outputs the Discriminator hidden layer
discriminator_hidden = tf.keras.Model(discriminator.input, discriminator.layers[2].output)

# Reduce the dimensionality of the data using the Discriminator
validation_data_gan_reduced = discriminator_hidden(validation_data)

# Reduce the dataset with PCA
validation_data_reduced_PCA = PCA(n_components=2).fit_transform(validation_data)
validation_data_gan_reduced_PCA = PCA(n_components=2).fit_transform(validation_data_gan_reduced)

# Reduce the dataset with t-SNE
validation_data_reduced_TSNE = TSNE(n_components=2).fit_transform(validation_data)
validation_data_gan_reduced_TSNE = TSNE(n_components=2).fit_transform(validation_data_gan_reduced)

# Compute metrics for dimensionality reduction...


# Plot the results for sample correlation
fig, axs = plt.subplots(2, 2)

# PCA plot
axs[0,0].scatter(validation_data_reduced_PCA[:,0], validation_data_reduced_PCA[:,1], label = "PCA")
axs[0,0].title.set_text(f"PCA")
axs[0,0].set_xlabel("PC1")
axs[0,0].set_ylabel("PC2")

# GAN + PCA plot
axs[0,1].scatter(validation_data_gan_reduced_PCA[:,0], validation_data_gan_reduced_PCA[:,1], label = "GAN + PCA")
axs[0,1].title.set_text(f"PCA")
axs[0,1].set_xlabel("PC1")
axs[0,1].set_ylabel("PC2")

# t-SNE plot
axs[1,0].scatter(validation_data_reduced_TSNE[:,0], validation_data_reduced_TSNE[:,1], label = "TSNE")
axs[1,0].title.set_text(f"t-SNE")
axs[1,0].set_xlabel("TSNE 1")
axs[1,0].set_ylabel("TSNE 2")

# GAN + t-SNE plot
axs[1,1].scatter(validation_data_gan_reduced_TSNE[:,0], validation_data_gan_reduced_TSNE[:,1], label = "GAN + TSNE")
axs[1,1].title.set_text(f"GAN + t-SNE ")
axs[1,1].set_xlabel("TSNE 1")
axs[1,1].set_ylabel("TSNE 2")


fig.savefig(fname=get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/images/dimensionality_reduction_plot.png"))
plt.clf() 