# TF GAN

# Import modules
import tensorflow as tf
tf.compat.v1.logging.set_verbosity('ERROR')
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LeakyReLU, Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import directed_hausdorff
from pathlib import Path

# Set matplotlib settings
plt.style.use('ggplot')
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)        
plt.rc('axes', titlesize=MEDIUM_SIZE)   
plt.rc('axes', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=SMALL_SIZE)  
plt.rc('ytick', labelsize=SMALL_SIZE)  
plt.rc('legend', fontsize=MEDIUM_SIZE)  
plt.rc('figure', titlesize=BIGGER_SIZE)
red = "#d1495b"
blue = "#00798c"
point_size = 20
legend_point_size = 40
line_width = 1.5
axis_size = 8


# Define a helper function for handling paths
def get_path(path):

    return str(Path(path).resolve())

# Define a helper function to read and split the data
def get_data(folder_path):

    # Read data
    data = pd.read_csv(get_path(folder_path + "GSE114727_processed_data.csv"), delimiter=",")
    anno = pd.read_csv(get_path(folder_path + "GSE114727_processed_annotations.csv"), delimiter=",")

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
        GEN_LRATE = 0.0001,
        DISC_LRATE = 0.0001, 
        EPOCHS = 10000, 
        BATCH_SIZE = 50, 
        NOISE_DIM = 100, 
        SEED = 36,
        checkpoint_freq = 100,
        eval_freq = 2000):
        
        self.data = data

        ### Define model parameters ###
        self.GEN_LRATE = GEN_LRATE
        self.DISC_LRATE = DISC_LRATE
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.NOISE_DIM = NOISE_DIM
        self.SEED = SEED
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.NUM_FEATURES = self.data.shape[1]
        self.CHECKPOINT_PATH = CHECKPOINT_PATH
        self.VAL_LOSS = []
        self.HAUSDORFF_DIST = []
        self.FILE_NAME = "{}_{}_{}_{}_{}_{}".format(
            self.GEN_LRATE,
            self.DISC_LRATE,
            self.EPOCHS,
            self.BATCH_SIZE,
            self.NOISE_DIM,
            self.SEED)

        ### Define Optimizers ###
        self.gen_optimizer = Adam(self.GEN_LRATE)
        self.disc_optimizer = Adam(self.DISC_LRATE)

        ### Define the loss functions ###
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        ### Create Networks ###
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        ### Define checkpoint ###
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)


    ### Initiatilise the data ###
    def init_data(self):

        ### Split data into training and test sets ###
        self.train_data, self.test_data = train_test_split(self.data, train_size = 0.8, random_state = self.SEED) 
        self.train_data_n = self.train_data.shape[0]
        self.test_data_n = self.test_data.shape[0]

        ### Batch and Suffle the data ###
        self.train_data = self.train_data.astype('float32')
        self.train_data = tf.data.Dataset.from_tensor_slices(self.train_data).shuffle(self.train_data.shape[0]).batch(self.BATCH_SIZE)

        self.test_data = self.test_data.astype('float32')
        val_set = self.test_data.copy()
        self.test_data = tf.convert_to_tensor(self.test_data.astype('float32'))

        ### Create random validation noise ###
        self.val_noise = tf.random.normal([self.test_data_n, self.NOISE_DIM])

        ### Create validation sets ###
        self.val_PCA = PCA(n_components=2).fit_transform(val_set)
        self.val_TSNE = TSNE(n_components=2).fit_transform(val_set)
        self.val_UMAP = UMAP(n_components=2).fit_transform(val_set)

        # Create checkpoint directory if it doesn't already exist
        p = Path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}")
        if not p.exists():
            p.mkdir()
            p.joinpath("images").mkdir()
            p.joinpath("data").mkdir()
            p.joinpath("metrics").mkdir()


    ### Define Network Architecture ###
    # Generator Network
    def build_generator(self, alpha = 0.2):

        model = Sequential(name="Generator")

        # Layer 1
        model.add(Dense(input_dim=self.NOISE_DIM, units=600))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())

        # # Layer 2
        # model.add(Dense(units=500))
        # model.add(LeakyReLU(alpha=alpha))
        # model.add(BatchNormalization())

        # # Layer 3
        # model.add(Dense(units=750))
        # model.add(LeakyReLU(alpha=alpha))
        # model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(units=self.NUM_FEATURES))
        model.add(LeakyReLU(alpha=alpha))

        return model

    # Discriminator network
    def build_discriminator(self, alpha=0.2, rate = 0.3):
        
        model = Sequential(name="Discriminator")

        # Layer 1
        model.add(Dense(input_dim=self.NUM_FEATURES, units = 200))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())

        # # Layer 2
        # model.add(Dropout(rate = rate))
        # model.add(Dense(units = 200))
        # model.add(LeakyReLU(alpha=alpha))
        # model.add(BatchNormalization())

        # Layer 3
        # model.add(Dropout(rate = rate))
        # model.add(Dense(units = 100))
        # model.add(LeakyReLU(alpha=alpha))
        # model.add(BatchNormalization())

        # sigmoid layer outputting a single value
        model.add(Dense(1, activation = "sigmoid"))
        
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
Generator Learning rate = {self.GEN_LRATE}
Discriminator Learning rate = {self.DISC_LRATE}
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
        # return -tf.reduce_mean(fake_output)

    # Discriminator loss
    def disc_loss(self, real_output, fake_output):

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

        # return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


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

            # Save the model at the specified frequency
            if (epoch + 1) % self.checkpoint_freq == 0:
                self.create_checkpoint(epoch)

            # Evaluate the model at the specified frequency
            if (epoch + 1) % self.eval_freq == 0:
                self.evaluate_model(epoch)

            
        print("[INFO] training completed in {} seconds!".format(round(time.time()-start, 2)))
        
        return None


    def produce_loss_graph(self):

        # Model losses
        # Convert list of lists into a numpy arrays
        gen_losses = pd.Series([loss[0].numpy() for loss in self.VAL_LOSS])
        disc_losses = pd.Series([loss[1].numpy() for loss in self.VAL_LOSS])
        model_losses = gen_losses + disc_losses

        # Combine losses in a dataframe
        combined_losses = pd.DataFrame(
            {"gen_loss": gen_losses,
            "disc_loss": disc_losses,
            "total_loss": model_losses}
            )

        # Save losses as csv
        combined_losses.to_csv(get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/data/losses.csv"), index=False)

        # Distances
        # Convert list of lists into a numpy arrays
        epochs = pd.Series([loss[0] for loss in self.HAUSDORFF_DIST])
        PCA_losses = pd.Series([loss[1] for loss in self.HAUSDORFF_DIST])
        TSNE_losses = pd.Series([loss[2] for loss in self.HAUSDORFF_DIST])
        UMAP_losses = pd.Series([loss[3] for loss in self.HAUSDORFF_DIST])

        # Combine losses in a dataframe
        distance_losses = pd.DataFrame(
            {"epoch": epochs,
            "pca_loss": PCA_losses,
            "tsne_loss": TSNE_losses,
            "umap_loss": UMAP_losses}
            )

        # Save losses as csv
        distance_losses.to_csv(get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/data/distance_losses.csv"), index=False)

        # Create loss graph
        fig, ax = plt.subplots(1, 1, figsize=(axis_size*3, axis_size), squeeze=True)
        ax.plot(gen_losses, linewidth = line_width)
        ax.plot(disc_losses, linewidth = line_width)
        ax.plot(model_losses, linewidth = line_width)
        ax.set_ylim(ymin=0)
        ax.set_ylabel('Validation Loss')
        ax.set_xlabel('Epoch')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        fig.legend(['Generator', 'Discriminator', 'Total Loss'], loc='lower center', frameon = False, ncol=3)
        fig.savefig(fname=get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/images/losses_plot.png"))
        plt.clf()

        # Save losses as csv

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
        generated_samples = self.generator(self.val_noise, training=False).numpy()

        # Reduce the dataset with PCA
        noise_PCA = PCA(n_components=2).fit_transform(generated_samples)
        # Calculate the correlation between samples
        hausdorff_dist_PCA = hausdorff_dist(self.val_PCA, noise_PCA)

        # Reduce the dataset with t-SNE
        noise_TSNE = TSNE(n_components=2).fit_transform(generated_samples)
        # Calculate the correlation between samples
        hausdorff_dist_TSNE = hausdorff_dist(self.val_TSNE, noise_TSNE)

        # Reduce the dataset with UMAP
        noise_UMAP = UMAP(n_components=2).fit_transform(generated_samples)
        # Calculate the correlation between samples
        hausdorff_dist_UMAP = hausdorff_dist(self.val_UMAP, noise_UMAP)
        
        # Save distances
        self.HAUSDORFF_DIST.append([epoch+1, hausdorff_dist_PCA, hausdorff_dist_TSNE, hausdorff_dist_UMAP])

        # Visualise the validation set
        plot_ratios = {'height_ratios': [1], 'width_ratios': [1,1,1]}
        fig, axs = plt.subplots(1, 3, figsize=(axis_size*3, axis_size), gridspec_kw=plot_ratios, squeeze=True)

        # PCA plot
        axs[0].scatter(noise_PCA[:,0], noise_PCA[:,1], c = red, s = point_size)
        axs[0].scatter(self.val_PCA[:,0], self.val_PCA[:,1], c = blue, s = point_size)
        axs[0].title.set_text(f"PCA - Hausdorff dist: {round(hausdorff_dist_PCA,2)}")
        axs[0].set_xlabel("PC 1")
        axs[0].set_ylabel("PC 2")
        box = axs[0].get_position()
        axs[0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # t-SNE plot
        axs[1].scatter(noise_TSNE[:,0], noise_TSNE[:,1], label = "Generated", c = red, s = point_size)
        axs[1].scatter(self.val_TSNE[:,0], self.val_TSNE[:,1], label = "Real", c = blue, s = point_size)
        axs[1].title.set_text(f"t-SNE - Hausdorff dist: {round(hausdorff_dist_TSNE,2)}")
        axs[1].set_xlabel("t-SNE 1")
        axs[1].set_ylabel("t-SNE 2")
        box = axs[1].get_position()
        axs[1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # UMAP plot
        axs[2].scatter(noise_UMAP[:,0], noise_UMAP[:,1], c = red, s = point_size)
        axs[2].scatter(self.val_UMAP[:,0], self.val_UMAP[:,1], c = blue, s = point_size)
        axs[2].title.set_text(f"UMAP - Hausdorff dist: {round(hausdorff_dist_UMAP,2)}")
        axs[2].set_xlabel("UMAP 1")
        axs[2].set_ylabel("UMAP 2")
        box = axs[2].get_position()
        axs[2].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        
        fig.legend(loc = "lower center", ncol = 2, frameon = False, markerscale = 2.0)
        fig.savefig(fname=get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/images/training_validation_plot_{epoch+1:05d}.png"))
        plt.clf()

        return None