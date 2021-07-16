# Main.py

# Import modules
from WGAN_GP import SCGAN
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read in the data
data = np.genfromtxt('four_datasets_combined_lTPM_red_small_clean.csv', delimiter=',', skip_header=1)
data = np.transpose(data)
data = MinMaxScaler().fit_transform(data)

# Create GAN Class
gan = SCGAN(
    data = data,
    CHECKPOINT_PATH = "../models", 
    LRATE = 0.00005,
    EPOCHS = 1000, 
    BATCH_SIZE = 32, 
    NOISE_DIM = 100, 
    SEED = 10,
    checkpoint_freq = 200,
    eval_freq = 200)

# Initialise the data
gan.init_data()

# Print model summaries
gan.get_model_summaries()

# print learning paramters
gan.get_learning_parameters()

# Train GAN 
gan.train_network()

# Produce network loss grpah
gan.produce_loss_graph()