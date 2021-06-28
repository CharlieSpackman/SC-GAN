# model_evaluation.py

# import modules
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import directed_hausdorff
import os
import sys

# Define parameters
SEED = 36
VAL_SIZE = 500
NOISE_DIM = 100

# Update path to model for evaluation
MODEL_PATH = "..\\models\\..."

# Define a function to evaluate the model
def hausdorff_dist(real_samples, gen_samples):
    dist = directed_hausdorff(
        u = real_samples,
        v = gen_samples
    )[0]

    return dist

# Load the model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

else:
    print(f"No model found in {MODEL_PATH}")
    sys.exit()


# Read in data
data = pd.read_csv("..\data.csv", delimiter=",")
data = data.iloc[:, 1:].to_numpy()
print("[INFO] Data successfully loaded")

# Create a validations set
_, validation_data = train_test_split(data, test_size=(VAL_SIZE/data.shape[0]), random_state=SEED)

# Create random noise to generate samples
noise = tf.random.normal([VAL_SIZE, NOISE_DIM])

# Generate the samples using the Generator
generated_samples = model.generator(noise, training = False)

# Integrate generated samples with the real validation set
validation_set =  tf.concat((validation_data, generated_samples), axis = 0)

# Create labels
validation_labels = tf.concat(
    (tf.zeros(VAL_SIZE),
    tf.ones(VAL_SIZE)),
    axis = 0)

# Reduce the dataset with PCA
validation_set_reduced_PCA = PCA(n_components=2).fit_transform(validation_set)
generated_samples_reduced_PCA = validation_set_reduced_PCA[validation_labels==0]
test_set_reduced_PCA = validation_set_reduced_PCA[validation_labels==1]
# Calculate the correlation between samples
hausdorff_dist_PCA = hausdorff_dist(test_set_reduced_PCA, generated_samples_reduced_PCA)

# Reduce the dataset with t-SNE
validation_set_reduced_TSNE = TSNE(n_components=2).fit_transform(validation_set)
generated_samples_reduced_TSNE = validation_set_reduced_TSNE[validation_labels==0]
test_set_reduced_TSNE = validation_set_reduced_TSNE[validation_labels==1]
# Calculate the correlation between samples
hausdorff_dist_TSNE = hausdorff_dist(test_set_reduced_TSNE, generated_samples_reduced_TSNE)

# Plot the results
fig, axs = plt.subplots(1, 2)
fig.suptitle(f"Generator Validation after training")

# PCA plot
axs[0].scatter(generated_samples_reduced_PCA[:,0], generated_samples_reduced_PCA[:,1], label = "Generated", c = "red")
axs[0].scatter(test_set_reduced_PCA[:,0], test_set_reduced_PCA[:,1], label = "Real", c = "blue")
axs[0].title.set_text(f"PCA - Hausdorff dist: {round(hausdorff_dist_PCA,2)}")
axs[0].set_xlabel("PC1")
axs[0].set_ylabel("PC2")

# t-SNE plot
axs[1].scatter(generated_samples_reduced_TSNE[:,0], generated_samples_reduced_TSNE[:,1], label = "Generated", c = "red")
axs[1].scatter(test_set_reduced_TSNE[:,0], test_set_reduced_TSNE[:,1], label = "Real", c = "blue")
axs[1].title.set_text(f"t-SNE - Hausdorff dist: {round(hausdorff_dist_TSNE,2)}")
axs[1].set_xlabel("t-SNE 1")
axs[1].set_ylabel("t-SNE 2")

fig.legend(loc = "lower center", ncol = 2)
fig.savefig(fname="validation_plot.png")
plt.clf() 