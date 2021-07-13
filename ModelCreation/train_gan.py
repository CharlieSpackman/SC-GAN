# Main.py

# Import modules
from WGAN import SCGAN, get_data

# Read in the data
data, _, _ = get_data("../DataPreprocessing/GSE114727/")

# Create GAN Class
gan = SCGAN(
    data = data,
    CHECKPOINT_PATH = "../models", 
    GEN_LRATE = 0.00001,
    DISC_LRATE = 0.00001, 
    EPOCHS = 10, 
    BATCH_SIZE = 250, 
    NOISE_DIM = 100, 
    SEED = 30,
    checkpoint_freq = 200,
    eval_freq = 1000)

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