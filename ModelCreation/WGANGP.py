# WGANGP.py

#-------------------------------
# Module set up
#-------------------------------

# Import python modules
from pathlib import Path
from functools import partial
import time

# Import Tensorflow modules
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LeakyReLU, Dense, BatchNormalization, Input, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.framework.ops import disable_eager_execution

# Set Tensorflow settings
tf.compat.v1.logging.set_verbosity('ERROR')
disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# Import data processing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import make_interp_spline

# Import dimensionality reduction modules
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

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


#-------------------------------
# Helper functions and classes
#-------------------------------

# Define a helper function for handling paths
def _get_path(path):

    return str(Path(path).resolve())

# Define a helper function to rescale an array
def _rescale_arr(arr):

    arr_scaled = MinMaxScaler(feature_range=(-10, 10)).fit_transform(arr)

    return arr_scaled

# Define a helper class to manage to the random weighted average which is called during training
class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, batch_size, n_features):
        super().__init__()
        self.batch_size = batch_size
        self.n_features = n_features

    def _call(self, inputs, **kwargs):
        alpha = tf.random.uniform([self.batch_size, self.n_features], 0.0, 1.0)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

#-------------------------------
# WGANGP Class
#-------------------------------

class WGANGP():

    def __init__(
        self, 
        lrate = 0.00005,
        epochs = 30000, 
        batch_size = 64, 
        noise_dim = 100,
        disc_updates = 5,
        grad_pen = 10,
        seed = 36,
        ckpt_freq = 200,
        ckpt_path = _get_path("../models"),
        eval_freq = 2000):
        
        # Define model parameters
        self.lrate = lrate
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.disc_updates = disc_updates
        self.grad_pen = grad_pen

        # Define algorithm parameters
        self.seed = seed
        self.ckpt_freq = ckpt_freq
        self.ckpt_path = ckpt_path
        self.eval_freq = eval_freq
        self.file_name = "{}_{}_{}_{}_{}".format(
            self.lrate,
            self.epochs,
            self.batch_size,
            self.noise_dim,
            self.seed)


    def init_data(self):

        print("[INFO] Reading data")

        # # Load the dataset
        data = pd.read_csv(_get_path("../DataPreprocessing/GSE114725/GSE114725_processed_data.csv"), delimiter=",", index_col="cellid")        
        data = data.values

        # Scale data
        data = MinMaxScaler(feature_range=(-1,1)).fit_transform(data)

        # Split data into training and test sets
        self.X_train, self.X_test = train_test_split(
            data,
            train_size = 0.8, 
            random_state = self.seed) 

        self.X_train_n = self.X_train.shape[0]
        self.X_test_n = self.X_test.shape[0]
        self.n_features = data.shape[1]

        # Create validation labels and placeholders for metrics
        self.val_labels = np.concatenate(
            (np.zeros(shape=(self.X_test_n,)), 
            np.ones(shape=(self.X_test_n, ))), 
            axis = 0)
        self.val_loss = []
        self.hausdorff_dist = []

        # Create checkpoint directory if it doesn't already exist
        p = Path(f"{self.ckpt_path}/{self.file_name}")
        if not p.exists():
            p.mkdir()
            p.joinpath("epochs").mkdir()
            p.joinpath("images").mkdir()
            p.joinpath("data").mkdir()
            p.joinpath("metrics").mkdir()

        print("[INFO] Data successfully read")

        return None

    def init_networks(self):

        print("[INFO] Initialising newtorks")


        # Define the Optimizer
        self.optimizer = RMSprop(learning_rate=self.lrate)

        # Build the generator and critic
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        #-------------------------------
        # Construct Computational Graph
        # for the Discriminator
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_sample = Input(shape=self.n_features)

        # Noise input
        z_disc = Input(shape=(self.noise_dim,))
        # Generate image based of noise (fake sample)
        fake_sample = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.discriminator(fake_sample)
        valid = self.discriminator(real_sample)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size, self.n_features)._call([real_sample, fake_sample])
        # Determine validity of weighted sample
        validity_interpolated = self.discriminator(interpolated_img)

        # Use Python partial to provide loss function with additional 'averaged_samples' argument
        partial_gp_loss = partial(
            self._gradient_penalty_loss,
            averaged_samples=interpolated_img)
        # Provide a name - required by TF
        partial_gp_loss.__name__ = 'gradient_penalty'

        # Define the combined Discriinator model
        self.discriminator_model = Model(inputs=[real_sample, z_disc],
                            outputs=[valid, fake, validity_interpolated])

        # Compile the model
        self.discriminator_model.compile(
            loss=[
                self._wasserstein_loss,
                self._wasserstein_loss,
                partial_gp_loss],
            optimizer=self.optimizer,
            loss_weights=[1, 1, 10])

        #-------------------------------
        # Construct Computational Graph
        # for the Generator
        #-------------------------------

        # For the generator we freeze the discriminator's layers
        self.discriminator.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.noise_dim,))
        # Generate images based of noise
        sample = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.discriminator(sample)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self._wasserstein_loss, optimizer=self.optimizer)

        #-------------------------------
        # Create checkpoint
        #-------------------------------
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            generator=self.generator,
            discriminator=self.discriminator)

        print("[INFO] Networks successfully initialised")

        return None


    def _gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = tf.gradients(y_pred, averaged_samples)[0]
        # Compute the euclidean norm by squaring 
        gradients_sqr = tf.math.square(gradients)
        # Sum over the rows
        gradients_sqr_sum = tf.math.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        # Take the square root
        gradient_l2_norm = tf.math.sqrt(gradients_sqr_sum)
        # Compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = tf.math.square(1 - gradient_l2_norm)
        # Return the mean as loss over all the batch samples
        return tf.math.reduce_mean(gradient_penalty)


    def _wasserstein_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true * y_pred)


    def _build_generator(self):

        model = Sequential(name="Generator")

        # Layer 1
        model.add(Dense(
            input_dim=self.noise_dim, 
            units=500, 
            kernel_initializer='he_normal',
            name = "Layer_1"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Layer 2
        model.add(Dense(
            units=1000, 
            kernel_initializer='he_normal',
            name = "Layer_2"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        # Layer 3
        model.add(Dense(
            units=1500, 
            kernel_initializer='he_normal',
            name = "Layer_3"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Output Layer
        model.add(Dense(
            units=self.n_features, 
            activation="tanh", 
            kernel_initializer='he_normal',
            name = "Output_layer"))

        noise = Input(shape=(self.noise_dim,))
        img = model(noise)

        return Model(noise, img)


    def _build_discriminator(self):

        model = Sequential(name="Discriminator")

        # Layer 1
        model.add(Dense(
            input_dim=self.n_features, 
            units = 1500, 
            kernel_initializer='he_normal',
            name = "Layer_1"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # Layer 2
        model.add(Dense(
            units = 1000, 
            kernel_initializer='he_normal',
            name = "Layer_2"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # Layer 3
        model.add(Dense(
            units = 500, 
            kernel_initializer='he_normal',
            name = "Layer_3"))
        model.add(LeakyReLU(alpha=0.2))

        # Output layer
        model.add(Dense(units = 1, 
        activation = None, 
        kernel_initializer='he_normal',
        name = "Output_layer"))

        sample = Input(shape=self.n_features)
        validity = model(sample)


        return Model(sample, validity)

    def train(self):

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty

        start = time.time()
        print("[INFO] starting training...")

        for epoch in range(self.epochs):

            print("[INFO] starting epoch {} of {}...".format(epoch + 1, self.epochs), end = "")
            epoch_start = time.time()

            for _ in range(self.disc_updates):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of samples
                idx = np.random.randint(0, self.X_train_n, self.batch_size)
                samples = self.X_train[idx]
                
                # Sample generator input
                noise = np.random.normal(0, 1, size=(self.batch_size, self.noise_dim))
                
                # Train the Discriminator
                disc_loss = self.discriminator_model.train_on_batch(
                    [samples, noise],
                    [valid, fake, dummy])[0]

            # ---------------------
            #  Train Generator
            # ---------------------

            gen_loss = self.generator_model.train_on_batch(noise, valid)

            # ---------------------
            #  Append losses
            # ---------------------

            self.val_loss.append([gen_loss, disc_loss])

            print('completed in {:.2f} seconds [G {:.4f}] [D {:.4f}]'.format(
                time.time()-epoch_start, 
                gen_loss,
                disc_loss))

            # Save the model at the specified frequency
            if (epoch + 1) % self.ckpt_freq == 0:
                self._create_checkpoint(epoch)

            # Evaluate the model at the specified frequency
            if (epoch + 1) % self.eval_freq == 0:
                self._evaluate_model(epoch)

        
        print("[INFO] training completed in {} seconds!".format(round(time.time()-start, 2)))

        return None

    def _evaluate_model(self, epoch):
        
        print("[INFO] evaluating model")

        def hausdorff_dist(real_samples, gen_samples):
            dist = directed_hausdorff(
                u = real_samples,
                v = gen_samples
            )[0]

            return dist

        # Create generated validation data
        noise = noise = np.random.normal(0, 1, size = (self.X_test_n, self.noise_dim))
        generated_samples = self.generator.predict(noise, steps=1)

        # Join generated samples with test set
        combined = np.concatenate((self.X_test, generated_samples), axis = 0)

        # Reduce and scale the dataset with PCA 
        combined_PCA = _rescale_arr(PCA(n_components=2, svd_solver="arpack").fit_transform(combined))
        # Calculate the correlation between samples
        hausdorff_dist_PCA = hausdorff_dist(combined_PCA[self.val_labels==0], combined_PCA[self.val_labels==1])

        # Reduce and scale the dataset with TSNE
        combined_TSNE = TSNE(
            n_components=2, 
            perplexity=10.0,
            learning_rate=100.0,
            init = "pca") 
        combined_TSNE = _rescale_arr(combined_TSNE.fit_transform(combined))
        # Calculate the correlation between samples
        hausdorff_dist_TSNE = hausdorff_dist(combined_TSNE[self.val_labels==0], combined_TSNE[self.val_labels==1])

        # Reduce and scale the dataset with UMAP 
        combined_UMAP = _rescale_arr(UMAP(n_components=2).fit_transform(combined))
        # Calculate the correlation between samples
        hausdorff_dist_UMAP = hausdorff_dist(combined_UMAP[self.val_labels==0], combined_UMAP[self.val_labels==1])

        # Append distances to distance object
        self.hausdorff_dist.append([epoch+1, hausdorff_dist_PCA, hausdorff_dist_TSNE, hausdorff_dist_UMAP])

        # Visualise the validation set
        plot_ratios = {'height_ratios': [1], 'width_ratios': [1,1,1]}
        fig, axs = plt.subplots(1, 3, figsize=(axis_size*3, axis_size), gridspec_kw=plot_ratios, squeeze=True)

        # PCA plot
        axs[0].scatter(combined_PCA[self.val_labels==0, 0], combined_PCA[self.val_labels==0, 1], c = red, s = point_size)
        axs[0].scatter(combined_PCA[self.val_labels==1, 0], combined_PCA[self.val_labels==1, 1], c = blue, s = point_size-4)
        axs[0].title.set_text(f"PCA - Hausdorff dist: {round(hausdorff_dist_PCA,2)}")
        axs[0].set_xlabel("PC 1")
        axs[0].set_ylabel("PC 2")
        box = axs[0].get_position()
        axs[0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # t-SNE plot
        axs[1].scatter(combined_TSNE[self.val_labels==0, 0], combined_TSNE[self.val_labels==0, 1], label = "Real", c = red, s = point_size)
        axs[1].scatter(combined_TSNE[self.val_labels==1, 0], combined_TSNE[self.val_labels==1, 1], label = "Generated", c = blue, s = point_size-4)
        axs[1].title.set_text(f"t-SNE - Hausdorff dist: {round(hausdorff_dist_TSNE,2)}")
        axs[1].set_xlabel("t-SNE 1")
        axs[1].set_ylabel("t-SNE 2")
        box = axs[1].get_position()
        axs[1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # UMAP plot
        axs[2].scatter(combined_UMAP[self.val_labels==0, 0], combined_UMAP[self.val_labels==0, 1], c = red, s = point_size)
        axs[2].scatter(combined_UMAP[self.val_labels==1, 0], combined_UMAP[self.val_labels==1, 1], c = blue, s = point_size-4)
        axs[2].title.set_text(f"UMAP - Hausdorff dist: {round(hausdorff_dist_UMAP,2)}")
        axs[2].set_xlabel("UMAP 1")
        axs[2].set_ylabel("UMAP 2")
        box = axs[2].get_position()
        axs[2].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        
        fig.legend(loc = "lower center", ncol = 2, frameon = False, markerscale = 2.0)
        fig.savefig(fname=_get_path(f"{self.ckpt_path}/{self.file_name}/images/training_validation_plot_{epoch+1:05d}.png"))
        plt.clf()


    def produce_loss_graph(self):

        def interpolate(epochs, data):

            epochs_new = np.linspace(1, epochs.iloc[-1], epochs.shape[0]*100)
            new_line = make_interp_spline(epochs, data)
            data_new = new_line(epochs_new)

            return pd.Series(epochs_new), pd.Series(data_new)

        print("[INFO] producing loss graphs...", end =  "")

        # Model losses
        # Convert list of lists into a numpy arrays
        gen_losses = pd.Series([loss[0] for loss in self.val_loss])
        disc_losses = pd.Series([loss[1] for loss in self.val_loss])

        # Combine losses in a dataframe
        combined_losses = pd.DataFrame(
            {"gen_loss": gen_losses,
            "disc_loss": disc_losses}
            )

        # Save losses as csv
        combined_losses.to_csv(_get_path(f"{self.ckpt_path}/{self.file_name}/data/losses.csv"), index=False)

        # Create training loss graph
        fig, ax = plt.subplots(1, 1, figsize=(axis_size*3, axis_size), squeeze=True)
        ax.plot(gen_losses, linewidth = line_width)
        ax.plot(disc_losses, linewidth = line_width)
        ax.set_ylabel('Wasserstein-1 Distance')
        ax.set_xlabel('Epoch')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        fig.legend(['Generator Loss', 'Discriminator Loss'], loc='lower center', frameon = False, ncol=2)

        fig.savefig(fname=_get_path(f"{self.ckpt_path}/{self.file_name}/images/losses_plot.png"))
        plt.clf()


        # Distances
        # Convert list of lists into a numpy arrays
        epochs = pd.Series([loss[0] for loss in self.hausdorff_dist], dtype=np.float32)
        PCA_losses = pd.Series([loss[1] for loss in self.hausdorff_dist], dtype=np.float32)
        TSNE_losses = pd.Series([loss[2] for loss in self.hausdorff_dist], dtype=np.float32)
        UMAP_losses = pd.Series([loss[3] for loss in self.hausdorff_dist], dtype=np.float32)

        # Interpolate the data to ensure a smooth graph
        _, PCA_losses = interpolate(epochs, PCA_losses)
        _, TSNE_losses = interpolate(epochs, TSNE_losses)
        epochs, UMAP_losses = interpolate(epochs, UMAP_losses)

        # Combine losses in a dataframe
        distance_losses = pd.DataFrame(
            {"epoch": epochs,
            "pca_loss": PCA_losses,
            "tsne_loss": TSNE_losses,
            "umap_loss": UMAP_losses}
            )

        # Save distances as csv
        distance_losses.to_csv(_get_path(f"{self.ckpt_path}/{self.file_name}/data/distance_losses.csv"), index=False)

        # Create distances graph
        fig, ax = plt.subplots(1, 1, figsize=(axis_size*3, axis_size), squeeze=True)
        ax.plot(epochs, PCA_losses, linewidth = line_width)
        ax.plot(epochs, TSNE_losses, linewidth = line_width)
        ax.plot(epochs, UMAP_losses, linewidth = line_width)
        ax.set_ylabel('Hausdorff Distance')
        ax.set_xlabel('Epoch')
        ax.set_ylim(ymin = 0.0)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        fig.legend(['PCA', 't-SNE', 'UMAP'], loc='lower center', frameon = False, ncol=3)

        fig.savefig(fname=_get_path(f"{self.ckpt_path}/{self.file_name}/images/distances_plot.png"))
        plt.clf()

        print("done!")

        return None


    def _create_checkpoint(self, epoch):

        path = _get_path("{}/{}/epochs/{:05d}/ckpt".format(
            self.ckpt_path,
            self.file_name,
            epoch+1))

        print("[INFO] creating checkpoint")
        self.checkpoint.save(file_prefix = path)

        return None



    # Define a function to print model summaries
    def get_model_summaries(self):

        print("[INFO] Printing Model Summaries\n")

        print(self.generator.layers[1].summary(), end = "\n\n")
        print(self.discriminator.layers[1].summary(), end = "\n\n")

        gen, disc = [], []

        self.generator.layers[1].summary(print_fn=lambda x: gen.append(x))
        self.discriminator.layers[1].summary(print_fn=lambda x: gen.append(x))

        with open(_get_path(f"{self.ckpt_path}/{self.file_name}/model_summaries.txt"), "w") as f:

            for item in gen:
                f.write(f"{item}\n")

            f.write("\n")

            for item in disc:
                f.write(f"{item}\n")

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
Learning rate = {self.lrate}
Epochs = {self.epochs}
Batch size = {self.batch_size}
Noise size = {self.noise_dim}
Seed = {self.seed}
Number of features {self.n_features}
Training set size = {self.X_train_n}
Test set size = {self.X_test_n}
        """

        print(parameters)

        with open(_get_path(f"{self.ckpt_path}/{self.file_name}/model_params.txt"), "w") as f:

            f.write(parameters)

        return None


if __name__ == '__main__':
    gan = WGANGP(
        lrate = 0.00005,
        epochs = 30000,
        batch_size = 32,
        noise_dim = 100,
        disc_updates = 5,
        grad_pen = 10,
        seed = 1,
        ckpt_path="../models",
        ckpt_freq=1000,
        eval_freq=2000
    )

    # Initialise data and networks
    gan.init_data()
    gan.init_networks()

    # Get summaries of model and parameters
    gan.get_model_summaries()
    gan.get_learning_parameters()

    # Train the GAN
    gan.train()

    # Evaluate the GAN
    gan.produce_loss_graph()