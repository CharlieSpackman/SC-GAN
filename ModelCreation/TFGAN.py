# TF GAN

# Import modules
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(40)
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LeakyReLU, Activation, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime
from scipy.spatial.distance import directed_hausdorff
from pathlib import Path

# Define a helper function for handling paths
def get_path(path):

    return str(Path(path).resolve())

# Define a helper function to read and split the data
def get_data(folder_path):

    # Read data
    data = pd.read_csv(get_path(folder_path + "/GSE114727_processed_data.csv"), delimiter=",")
    anno = pd.read_csv(get_path(folder_path + "/GSE114727_processed_annotations.csv"), delimiter=",")

    # Get cell ids
    cells = data.iloc[:, 0]
    data = data.iloc[:, 1:].to_numpy()
    
    print("[INFO] Data successfully loaded")

    return data, cells, anno

### Define Class container for training procedure ###
class SCGAN():

    def __init__(self, 
        data, 
        CHECKPOINT_PATH = get_path("../models"), 
        LRATE = 0.001, 
        EPOCHS = 10000, 
        BATCH_SIZE = 50, 
        NOISE_DIM = 100, 
        SEED = 36):
        
        self.data = data

        ### Define model parameters ###
        self.LRATE = LRATE
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.NOISE_DIM = NOISE_DIM
        self.SEED = SEED
        self.NUM_FEATURES = self.data.shape[1]
        self.CHECKPOINT_PATH = CHECKPOINT_PATH
        self.VAL_LOSS = []
        self.FILE_NAME = "{}_{}_{}_{}_{}_{}".format(
            datetime.today().strftime('%Y-%m-%d'),
            self.LRATE,
            self.EPOCHS,
            self.BATCH_SIZE,
            self.NOISE_DIM,
            self.SEED)

        ### Define Optimizers ###
        self.gen_optimizer = Adam(self.LRATE)
        self.disc_optimizer = Adam(self.LRATE)

        ### Define the loss functions ###
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        ### Split data into training and test sets ###
        self.train_data, self.test_data = train_test_split(self.data, train_size = 0.8, random_state = self.SEED) 
        self.train_data_n = self.train_data.shape[0]
        self.test_data_n = self.test_data.shape[0]

        ### Batch and Suffle the data ###
        self.train_data = self.train_data.astype('float32')
        self.train_data = tf.data.Dataset.from_tensor_slices(self.train_data).shuffle(self.train_data.shape[0]).batch(self.BATCH_SIZE)

        self.test_data = self.test_data.astype('float32')
        self.test_data = tf.convert_to_tensor(self.test_data.astype('float32'))

        ### Create random validation noise ###
        self.val_noise = tf.random.normal([self.test_data_n, self.NOISE_DIM])

        ### Create Networks ###
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        ### Define checkpoint ###
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)

        # Create checkpoint directory if it doesn't already exist
        p = Path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}")
        if not p.exists():
            p.mkdir()
            p.joinpath("images").mkdir()
            p.joinpath("data").mkdir()


    ### Define Network Architecture ###
    # Generator Network
    def build_generator(self):

        model = Sequential(name="Generator")

        # Layer 1
        model.add(Dense(input_dim=self.NOISE_DIM, units=250))
        model.add(Activation("relu"))

        # Layer 2
        model.add(Dense(units=500))
        model.add(Activation("relu"))

        # Layer 3
        model.add(Dense(units=750))
        model.add(Activation("relu"))

        # Output Layer
        model.add(Dense(units=self.NUM_FEATURES))
        model.add(Activation("tanh"))

        return model

    # Discriminator network
    def build_discriminator(self, alpha=0.2):
        
        model = Sequential(name="Discriminator")

        # Layer 1
        model.add(Dense(input_dim=self.NUM_FEATURES, units = 500))
        model.add(LeakyReLU(alpha=alpha))

        # Layer 2
        model.add(Dense(units = 200))
        model.add(LeakyReLU(alpha=alpha))

        # Layer 3
        model.add(Dense(units = 100))
        model.add(LeakyReLU(alpha=alpha))

        # sigmoid layer outputting a single value
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        
        # return the discriminator model
        return model

    # Define a function to print model summaries
    def get_model_summaries(self):

        print(self.generator.summary(), end = "\n\n")
        print(self.discriminator.summary(), end = "\n\n")

        return None

    # Define a functon to get the models
    def get_models(self):

        return self.generator, self.discriminator

    # Define a function to get the checkpoint option
    def get_checkpoint_obj(self):

        return self.checkpoint

    # Define a function to print learning parameters
    def get_learning_parameters(self):

        parameters = f"""---Learning Parameters---
Learning rate = {self.LRATE}
Epochs = {self.EPOCHS}
Batch size = {self.BATCH_SIZE}
Noise size = {self.NOISE_DIM}
Seed = {self.SEED}
Number of features {self.NUM_FEATURES}
Training set size = {self.train_data_n}
Test set size = {self.test_data_n}
        """

        print(parameters)

        return None

    ### Define loss functions ###
    # Generator loss
    def gen_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    # Discriminator loss
    def disc_loss(self, real_output, fake_output):

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss


    def create_checkpoint(self, epoch):

        path = get_path("{}/{}/epochs/{:05d}/ckpt".format(
            self.CHECKPOINT_PATH,
            self.FILE_NAME,
            epoch+1))

        print("[INFO] creating checkpoint")
        self.checkpoint.save(file_prefix = path)

        return None


    ### Define the main training step based on one input ###
    @tf.function
    def train_step(self, batch):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_samples = self.generator(noise, training=True)

            real_output = self.discriminator(batch, training=True)
            fake_output = self.discriminator(generated_samples, training=True)

            gen_loss = self.gen_loss(fake_output)
            disc_loss = self.disc_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return None


    ### Define the training procedure ###
    def train_network(self):

        start = time.time()
        print("[INFO] starting training...")

        for epoch in range(self.EPOCHS):

            print("[INFO] starting epoch {} of {}...".format(epoch + 1, self.EPOCHS), end = "")
            epoch_start = time.time()

            for batch in self.train_data:
                self.train_step(batch)

            # Get validation losses
            generated_samples = self.generator(self.val_noise, training=False)

            real_output = self.discriminator(self.test_data, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            gen_loss = self.gen_loss(fake_output)
            disc_loss = self.disc_loss(real_output, fake_output)

            self.VAL_LOSS.append([gen_loss, disc_loss])

            print ('completed in {} seconds'.format(round(time.time()-epoch_start, 2)))

            # Save the model every 50 epochs
            if (epoch + 1) % 5 == 0:
                self.create_checkpoint(epoch)

            # Evaluate the model every 2000 epochs
            if (epoch + 1) % 20 == 0:
                self.evaluate_model(epoch)

        
        print("[INFO] training completed in {} seconds!".format(round(time.time()-start, 2)))
        
        return None


    def produce_loss_graph(self):

        # Convert list of lists into a numpy arrays
        gen_losses = np.array([loss[0] for loss in self.VAL_LOSS]).reshape(-1,1)
        disc_losses = np.array([loss[1] for loss in self.VAL_LOSS]).reshape(-1,1)
        model_losses = gen_losses + disc_losses

        combined_losses = np.concatenate((gen_losses, disc_losses, model_losses), axis = 1)

        # Create loss graph
        plt.plot(gen_losses)
        plt.plot(disc_losses)
        plt.plot(model_losses)
        plt.title('Validation Loss vs. Epochs')
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.legend(['Generator', 'Discriminator', 'Total Loss'], loc='upper right')
        plt.savefig(fname=get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/images/losses_plot.png"))
        plt.clf()

        # Save losses as csv
        np.savetxt(fname=get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/data/losses.csv"), X = combined_losses, delimiter=",")

        return None


    # Define a function to evaluate the model
    def evaluate_model(self, epoch):

        print("[INFO] evaluating model")

        def hausdorff_dist(real_samples, gen_samples):
            dist = directed_hausdorff(
                u = real_samples,
                v = gen_samples
            )[0]

            return dist

        # Create generated validation data
        generated_samples = self.generator(self.val_noise, training=False)

        # Integrate generated samples with the real validation set
        validation_set =  tf.concat((self.test_data, generated_samples), axis = 0)
        
        # Create data set labels
        validation_labels = tf.concat(
            (tf.zeros(self.test_data_n),
            tf.ones(self.test_data_n)),
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

        # Reduce the dataset with UMAP
        validation_set_reduced_UMAP = UMAP(n_components=2).fit_transform(validation_set)
        generated_samples_reduced_UMAP = validation_set_reduced_UMAP[validation_labels==0]
        test_set_reduced_UMAP = validation_set_reduced_UMAP[validation_labels==1]
        # Calculate the correlation between samples
        hausdorff_dist_UMAP = hausdorff_dist(test_set_reduced_UMAP, generated_samples_reduced_UMAP)
        
        
        # Visualise the validation set
        fig, axs = plt.subplots(1, 3, figsize=(11.69, 8.27))
        fig.suptitle(f"Generator Validation at epoch {epoch}")

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

        # UMAP plot
        axs[2].scatter(generated_samples_reduced_UMAP[:,0], generated_samples_reduced_UMAP[:,1], label = "Generated", c = "red")
        axs[2].scatter(test_set_reduced_UMAP[:,0], test_set_reduced_UMAP[:,1], label = "Real", c = "blue")
        axs[2].title.set_text(f"UMAP - Hausdorff dist: {round(hausdorff_dist_UMAP,2)}")
        axs[2].set_xlabel("UMAP 1")
        axs[2].set_ylabel("UMAP 2")
        
        fig.legend(loc = "lower center", ncol = 3)
        fig.savefig(fname=get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/images/epoch_{epoch}_validation_plot.png"))
        plt.clf()