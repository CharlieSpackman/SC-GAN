# WGAN.py

# Import modules
import tensorflow as tf
tf.compat.v1.logging.set_verbosity('ERROR')
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LeakyReLU, Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import RMSprop
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
point_size = 20.0
legend_point_size = 40.0
line_width = 3.0
axis_size = 8.0


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

# Define a helper function to rescale an array
def rescale_arr(arr):

    arr[:,0] = 10 * (2.*(arr[:,0] - np.min(arr[:,0]))/np.ptp(arr[:,0])-1)
    arr[:,1] = 10 * (2.*(arr[:,1] - np.min(arr[:,1]))/np.ptp(arr[:,1])-1)

    return arr

### Define Class container for training procedure ###
class SCGAN():

    def __init__(self, 
        data, 
        CHECKPOINT_PATH = get_path("../models"), 
        LRATE = 0.00001,
        EPOCHS = 30000, 
        BATCH_SIZE = 250, 
        NOISE_DIM = 100, 
        SEED = 36,
        DISC_UPDATES = 5,
        CLIP_VALUE = 0.01,
        checkpoint_freq = 200,
        eval_freq = 2000):
        
        self.data = data

        ### Define model parameters ###
        self.LRATE = LRATE
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.NOISE_DIM = NOISE_DIM
        self.SEED = SEED
        self.DISC_UPDATES = DISC_UPDATES
        self.CLIP_VALUE = CLIP_VALUE
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
        self.optimizer = RMSprop(self.LRATE)

        ### Define labels ###
        self.valid_labels = -tf.ones((self.BATCH_SIZE, 1), dtype=tf.dtypes.float32)
        self.fake_labels = tf.ones((self.BATCH_SIZE, 1), dtype=tf.dtypes.float32)

        ### Create Networks ###
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
            optimizer=self.optimizer,
            metrics = ['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.NOISE_DIM,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=self.optimizer,
            metrics = ['accuracy'])

        ### Define checkpoint ###
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
            generator=self.generator,
            discriminator=self.discriminator)


    ### Initiatilise the data ###
    def init_data(self):

        ### Split data into training and test sets ###
        self.train_data, self.test_data = train_test_split(self.data, train_size = 0.8, random_state = self.SEED) 
        self.train_data_n = self.train_data.shape[0]
        self.test_data_n = self.test_data.shape[0]

        ### Convert to the correct dtype ###
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')

        ### Create reduced validation sets ###
        self.val_PCA = PCA(n_components=2).fit_transform(self.test_data)
        self.val_TSNE = TSNE(n_components=2).fit_transform(self.test_data)
        self.val_UMAP = UMAP(n_components=2).fit_transform(self.test_data)
        self.val_labels = np.concatenate((np.zeros((self.test_data_n,)), np.ones((self.test_data_n, ))), axis = 0)

        # Create checkpoint directory if it doesn't already exist
        p = Path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}")
        if not p.exists():
            p.mkdir()
            p.joinpath("images").mkdir()
            p.joinpath("data").mkdir()
            p.joinpath("metrics").mkdir()


    ### Define Network Architecture ###
    # Generator Network
    def build_generator(self):

        model = Sequential(name="Generator")

        # Layer 1
        model.add(Dense(input_dim=self.NOISE_DIM, units=400, activation="relu"))
        model.add(BatchNormalization())

        # Layer 2
        model.add(Dense(units=600, activation="relu"))
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(units=self.NUM_FEATURES, activation="tanh"))

        noise = Input(shape=(self.NOISE_DIM,))
        img = model(noise)

        return Model(noise, img)

    # Discriminator network
    def build_discriminator(self, alpha=0.2, rate = 0.2):
        
        model = Sequential(name="Discriminator")

        # Layer 1
        model.add(Dense(input_dim=self.NUM_FEATURES, units = 600))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(rate = rate))

        # Layer 2
        model.add(Dense(units = 400))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(rate = rate))

        # Layer 3
        model.add(Dense(units = 100))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(rate = rate))

        # Output layer
        model.add(Dense(1, activation = "linear"))

        img = Input(shape=self.NUM_FEATURES)
        validity = model(img)

        return Model(img, validity)

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
    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)


    def create_checkpoint(self, epoch):

        path = get_path("{}/{}/epochs/{:05d}/ckpt".format(
            self.CHECKPOINT_PATH,
            self.FILE_NAME,
            epoch+1))

        print("[INFO] creating checkpoint")
        self.checkpoint.save(file_prefix = path)

        return None


    ### Define the training procedure ###
    def train_network(self):

        start = time.time()
        print("[INFO] starting training...")

        # Loop over epochs
        for epoch in range(self.EPOCHS):

            print("[INFO] starting epoch {} of {}...".format(epoch + 1, self.EPOCHS), end = "")
            epoch_start = time.time()

            ### Train Discriminator ###
            # Loop through each batch and update Discriminator
            for _ in range(self.DISC_UPDATES):
                # Get a random set of batches
                idx = np.random.randint(0, self.train_data_n, self.BATCH_SIZE)
                batch = self.train_data[idx]
                # Create random noise and generate samples
                noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])
                generated_samples = self.generator.predict(noise)

                # Train the Discriminator
                real_loss = self.discriminator.train_on_batch(batch, self.valid_labels)
                fake_loss = self.discriminator.train_on_batch(generated_samples, self.fake_labels)
                disc_loss = 0.5 * np.add(fake_loss, real_loss)

                # Clip discriminator weights
                for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight, -self.CLIP_VALUE, self.CLIP_VALUE) for weight in weights]
                    layer.set_weights(weights)
            
            ### Train Generator ###
            gen_loss = self.combined.train_on_batch(noise, self.valid_labels)

            ### Append losses ###
            self.VAL_LOSS.append([1 - gen_loss[0], 1 - disc_loss[0]])

            print('completed in {} seconds [G {}] [D {}]'.format(
                round(time.time()-epoch_start, 2), 
                round(1 - gen_loss[0], 6),
                round(1 - disc_loss[0], 6)))

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
        gen_losses = pd.Series([loss[0] for loss in self.VAL_LOSS])
        disc_losses = pd.Series([loss[1] for loss in self.VAL_LOSS])

        # Combine losses in a dataframe
        combined_losses = pd.DataFrame(
            {"gen_loss": gen_losses,
            "disc_loss": disc_losses}
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

        # Create training loss graph
        fig, ax = plt.subplots(1, 1, figsize=(axis_size*3, axis_size), squeeze=True)
        ax.plot(gen_losses, linewidth = line_width)
        ax.plot(disc_losses, linewidth = line_width)
        ax.set_ylabel('Validation Loss')
        ax.set_xlabel('Epoch')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        fig.legend(['Generator Loss', 'Discriminator Loss'], loc='lower center', frameon = False, ncol=2)

        fig.savefig(fname=get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/images/losses_plot.png"))
        plt.clf()

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
        noise = tf.random.normal([self.test_data_n, self.NOISE_DIM])
        generated_samples = self.generator.predict(noise)

        # Reduce the dataset with PCA
        noise_PCA = PCA(n_components=2).fit_transform(generated_samples)
        # Combine and rescale data
        combined_PCA = rescale_arr(np.concatenate((self.val_PCA, noise_PCA), axis = 0))
        # Calculate the correlation between samples
        hausdorff_dist_PCA = hausdorff_dist(combined_PCA[self.val_labels==0], combined_PCA[self.val_labels==1])

        # Reduce the dataset with TSNE
        noise_TSNE = TSNE(n_components=2).fit_transform(generated_samples)
        # Combine and rescale data
        combined_TSNE = rescale_arr(np.concatenate((self.val_TSNE, noise_TSNE), axis = 0))
        # Calculate the correlation between samples
        hausdorff_dist_TSNE = hausdorff_dist(combined_TSNE[self.val_labels==0], combined_TSNE[self.val_labels==1])

        # Reduce the dataset with UMAP
        noise_UMAP = UMAP(n_components=2).fit_transform(generated_samples)
        # Combine and rescale data
        combined_UMAP = rescale_arr(np.concatenate((self.val_UMAP, noise_UMAP), axis = 0))
        # Calculate the correlation between samples
        hausdorff_dist_UMAP = hausdorff_dist(combined_UMAP[self.val_labels==0], combined_UMAP[self.val_labels==1])
        
        # Save distances
        self.HAUSDORFF_DIST.append([epoch+1, hausdorff_dist_PCA, hausdorff_dist_TSNE, hausdorff_dist_UMAP])

        # Visualise the validation set
        plot_ratios = {'height_ratios': [1], 'width_ratios': [1,1,1]}
        fig, axs = plt.subplots(1, 3, figsize=(axis_size*3, axis_size), gridspec_kw=plot_ratios, squeeze=True)

        # PCA plot
        axs[0].scatter(combined_PCA[self.val_labels==0, 0], combined_PCA[self.val_labels==0, 1], c = red, s = point_size)
        axs[0].scatter(combined_PCA[self.val_labels==1, 0], combined_PCA[self.val_labels==1, 1], c = blue, s = point_size)
        axs[0].title.set_text(f"PCA - Hausdorff dist: {round(hausdorff_dist_PCA,2)}")
        axs[0].set_xlabel("PC 1")
        axs[0].set_ylabel("PC 2")
        box = axs[0].get_position()
        axs[0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # t-SNE plot
        axs[1].scatter(combined_TSNE[self.val_labels==0, 0], combined_TSNE[self.val_labels==0, 1], label = "Real", c = red, s = point_size)
        axs[1].scatter(combined_TSNE[self.val_labels==1, 0], combined_TSNE[self.val_labels==1, 1], label = "Generated", c = blue, s = point_size)
        axs[1].title.set_text(f"t-SNE - Hausdorff dist: {round(hausdorff_dist_TSNE,2)}")
        axs[1].set_xlabel("t-SNE 1")
        axs[1].set_ylabel("t-SNE 2")
        box = axs[1].get_position()
        axs[1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # UMAP plot
        axs[2].scatter(combined_UMAP[self.val_labels==0, 0], combined_UMAP[self.val_labels==0, 1], c = red, s = point_size)
        axs[2].scatter(combined_UMAP[self.val_labels==1, 0], combined_UMAP[self.val_labels==1, 1], c = blue, s = point_size)
        axs[2].title.set_text(f"UMAP - Hausdorff dist: {round(hausdorff_dist_UMAP,2)}")
        axs[2].set_xlabel("UMAP 1")
        axs[2].set_ylabel("UMAP 2")
        box = axs[2].get_position()
        axs[2].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        
        fig.legend(loc = "lower center", ncol = 2, frameon = False, markerscale = 2.0)
        fig.savefig(fname=get_path(f"{self.CHECKPOINT_PATH}/{self.FILE_NAME}/images/training_validation_plot_{epoch+1:05d}.png"))
        plt.clf()

        return None