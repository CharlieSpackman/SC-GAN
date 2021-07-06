# Main.py

# Import modules
from TFGAN import SCGAN, get_data

# Read in the data
data, _, _ = get_data("../DataPreprocessing/GSE114727/")#

# Create GAN Class
gan = SCGAN(
    data = data,
    CHECKPOINT_PATH = "../models", 
    GEN_LRATE = 0.001,
    DISC_LRATE = 0.001, 
    EPOCHS = 100, 
    BATCH_SIZE = 250, 
    NOISE_DIM = 100, 
    SEED = 30)

# Print model summaries
gan.get_model_summaries()

# print learning paramters
gan.get_learning_parameters()

# Train GAN 
gan.train_network()

# Produce network loss grpah
gan.produce_loss_graph()