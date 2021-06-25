# Main.py

# Import modules
from TFGAN import SCGAN
import numpy as np
import pandas as pd

# Read in the data
data = pd.read_csv("data.csv", delimiter = ",")
data = data.iloc[1000:, 1:].to_numpy()
print("[INFO] Data successfully loaded")

# Create GAN Class
gan = SCGAN(
    data = data,
    CHECKPOINT_PATH = "models", 
    LRATE = 0.001, 
    EPOCHS = 3, 
    BATCH_SIZE = 256, 
    NOISE_DIM = 100, 
    SEED = 36)

# Print model summaries
gan.get_model_summaries()

# print learning paramters
gan.get_learning_parameters()

# Train GAN 
gan.train_network()

# Produce network loss grpah
gan.produce_loss_graph()