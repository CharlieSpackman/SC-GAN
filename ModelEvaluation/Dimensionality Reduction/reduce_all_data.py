# reduce_all_data.py

# Import the GAN class
from ...ModelCreation.TFGAN import *

# Import ML modules
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Update path to model for evaluation
CHECKPOINT_PATH = get_path("../../models")
MODEL_NAME = "2021-06-28_0.001_20_256_100_36" ### Update this as required
EPOCH = 20 ### Update this as required

model_path = get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/epochs/{EPOCH:05d}")
latest_model = tf.train.latest_checkpoint(get_path(f"{model_path}/ckpt"))

# Read in data
data, cells, _ = get_data("../../DataPreprocessing/GSE114727/GSE114727_processed_data.csv")

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
col_names = [f"Component {i+1}" for i in range(data_gan_reduced)]
export_data = pd.DataFrame(data = data_gan_reduced, index=cells, columns=col_names)

export_data.to_csv(fname=get_path(
    f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/data_reduced_gan_{EPOCH:05d}.csv"), 
    delimiter=",")