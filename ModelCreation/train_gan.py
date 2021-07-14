# Main.py

# Import modules
from WGAN import SCGAN, get_data

# Read in the data
data, _, _ = get_data("../DataPreprocessing/GSE114727/")

# Create GAN Class
gan = SCGAN(
    data = data,
    CHECKPOINT_PATH = "../models", 
    LRATE = 0.0001,
    EPOCHS = 5000, 
    BATCH_SIZE = 100, 
    NOISE_DIM = 100, 
    SEED = 10,
    checkpoint_freq = 200,
    eval_freq = 500)

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