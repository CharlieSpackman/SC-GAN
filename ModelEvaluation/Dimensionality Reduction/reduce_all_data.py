# reduce_all_data.py

# import environment modules
from pathlib import Path
import sys
sys.path.append(str(Path("../../ModelCreation/").resolve()))

# Import the GAN class
from WGANGP import WGANGP, _get_path

# Import ML modules
import tensorflow as tf
import pandas as pd
import numpy as np


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

X_train, X_test, _, _ = gan.init_data("GSE114725_processed_data_10000_2000.csv", "GSE114725_processed_annotations_10000_2000.csv")

X = np.concatenate((X_train, X_test), axis = 0)

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
data_gan_reduced = discriminator_hidden(X).numpy()

# Save the data
col_names = [f"Component {i+1}" for i in range(data_gan_reduced.shape[1])]
export_data = pd.DataFrame(data = data_gan_reduced, columns=col_names)

export_data.to_csv(
    _get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/data_reduced_gan_{EPOCH:05d}.csv"), 
    sep=",",
    index=False)

print("[INFO] Data has been reduced and saved")