# reduce_all_data.py

# import environment modules
from pathlib import Path
import sys
sys.path.append(str(Path("../../ModelCreation/").resolve()))

# Import the GAN class
from TFGAN import SCGAN, get_path, get_data

# Import ML modules
import tensorflow as tf
import pandas as pd
from pathlib import Path
import sys

# Update path to model for evaluation
CHECKPOINT_PATH = get_path("../../models")
MODEL_NAME = "0.001_0.001_10000_256_100_36" ### Update this as required
EPOCH = 250 ### Update this as required

model_path = get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/epochs/{EPOCH:05d}")
latest_model = tf.train.latest_checkpoint(get_path(f"{model_path}/ckpt"))

# Read in data
data, cells, _ = get_data("../../DataPreprocessing/GSE114727/")
data = data[:1000,:]
cells = cells.iloc[:1000]

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
    print("[INFO] Model has been loaded")

else:
    print(f"[ERROR] No model found in {model_path}")
    sys.exit()

# Create a model which outputs the Discriminator hidden layer
discriminator_hidden = tf.keras.Model(discriminator.input, discriminator.layers[2].output)

# Reduce the dimensionality of the data using the Discriminator
data_gan_reduced = discriminator_hidden(data).numpy()

# Save the data
col_names = [f"Component {i+1}" for i in range(data_gan_reduced.shape[1])]
export_data = pd.DataFrame(data = data_gan_reduced, index=cells, columns=col_names)

export_data.to_csv(get_path(f"{CHECKPOINT_PATH}/{MODEL_NAME}/data/data_reduced_gan_{EPOCH:05d}.csv"), 
    sep=",")

print("[INFO] Data has been reduced and saved")