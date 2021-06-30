# reduce_all_data.py

# import environment modules
from pathlib import Path
import sys
sys.path.append(str(Path("../../ModelCreation/").resolve()))

# Import the GAN class
from TFGAN import SCGAN, get_path

# Import ML modules
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Update path to model for evaluation
CHECKPOINT_PATH = get_path("../../models")
MODEL_NAME = "2021-06-28_0.001_20_256_100_36" ### Update this as required
EPOCH = 20 ### Update this as required

model_path = get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/epochs/{EPOCH:05d}")
latest_model = tf.train.latest_checkpoint(get_path(f"{model_path}/ckpt"))

# Read in data
data = pd.read_csv(get_path("../../data.csv"), delimiter=",")
data = data.iloc[:, 1:].to_numpy()
print("[INFO] Data successfully loaded")

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
data_gan_reduced = discriminator_hidden(data)

# Save the data
np.savetxt(fname=get_path(
    f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/data_reduced_gan_{EPOCH:05d}.csv"), 
    X = data_gan_reduced, 
    delimiter=",")

# # Reduce the dataset with PCA
# data_gan_reduced_PCA = PCA(n_components=2).fit_transform(data_gan_reduced)

# # Reduce the dataset with t-SNE
# data_gan_reduced_TSNE = TSNE(n_components=2).fit_transform(data_gan_reduced)

# # Save data
# # PCA csv
# np.savetxt(fname=get_path(
#     f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/data_reduced_PCA_{EPOCH:05d}.csv"), 
#     X = data_gan_reduced_PCA, 
#     delimiter=",")
# # t-SNE csv
# np.savetxt(fname=get_path(
#     f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/data_reduced_TSNE_{EPOCH:05d}.csv"), 
#     X = data_gan_reduced_TSNE, 
#     delimiter=",")
