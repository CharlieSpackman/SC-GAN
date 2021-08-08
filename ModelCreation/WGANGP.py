# WGANGP.py
"""
Create, train and evaluate a WGANGP using sc-RNA seq data.

Change model parameters and specify file paths at the bottom of the module.

Classes:

    RandomWeightedAverage
    WGANGP

Functions:

    get_path(string) -> string
    rescale_arr(array) -> array
"""

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
from tensorflow.keras.layers import LeakyReLU, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution

# Set Tensorflow settings
tf.compat.v1.logging.set_verbosity('ERROR')

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
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP

# import cluster metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score

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
def get_path(path):
    """
    Takes a relative file path and returns an absolute file path.
    """

    return str(Path(path).resolve())

# Define a helper function to rescale an array
def rescale_arr(arr):
    """
    Takes a Numpy array and rescales all features to be between -10 and 10.
    """

    arr_scaled = MinMaxScaler(feature_range=(-10, 10)).fit_transform(arr)

    return arr_scaled

# Define a helper class to manage to the random weighted average which is called during training
class RandomWeightedAverage(tf.keras.layers.Layer):

    """
    A class created a random weighted average of real and generated samples.

    ...

    Attributes
    ----------
    batch_size : int
        size of the batches used during training
    n_features : int
        number of features (genes)

    Methods
    -------
    call(inputs):
        Takes a numpy array of real samples and generates a random weighted sample of real and generated samples
    """


    def __init__(self, batch_size, n_features):
        """
        Constructs all the necessary attributes for object.
    
        ...

        Parameters
        ----------
        batch_size : int
            size of the batches used during training
        n_features : int
            number of features (genes)
        """

        super().__init__()
        self.batch_size = batch_size
        self.n_features = n_features

    def call(self, inputs, **kwargs):
        """
        Takes a numpy array of real and generated samples and generates a random weighted sample of real and generated samples.

        Parameters
        ----------
        inputs : array
            Set of real and generated samples

        Returns
        -------
        output : array
            Randomly weighted sample of real and generated samples
        """

        alpha = tf.random.uniform([self.batch_size, self.n_features], 0.0, 1.0)
        output = (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

        return output

#-------------------------------
# WGANGP Class
#-------------------------------

class WGANGP():
    """
    A class to represent the GAN.

    ...

    Attributes
    ----------
    lrate : float
        learning rate to be used during training
    epochs : int
        number of training steps
    batch_size : int
        number of samples to on train per epoch
    noise_dim : int
        size of the latent vector
    disc_updates : int
        number of time the Discriminator is updated per epoch 
    grad_pen : float
        gradient penality applied to the loss function
    feature_range : tuple(float, float)
        range in which the training data will be scaled before starting training
    seed : int
        random seed state
    data_path : string
        path to the folder with training data
    ckpt_freq : int
        frequency at which a checkpoint is created during training
    ckpt_path : string
        path to the models folder
    eval_freq : int
        frequency at which the model will be evaluated
    file_name : string
        concatentation of model parameters
    X_train : array
        training set 
    X_test : array
        validation set
    y_train
        training labels
    y_test
        validation labels
    X_train_n : int
        number of samples in the training set
    X_test_n : int
        number of samples in the validation set
    n_features : int
        number of features
    data_max_value : float
        maximum expression value in the training set
    val_labels : array
        labels for graphs
    val_loss : list
        training losses, updated per epoch
    hausdorff_dist : list
        evaluation metric, updated once per evaluation frequency
    optimizer : tf.Optimizer
        optimizer object used during training
    generator : tf.Sequential
        neural network model for the Generator
    discriminator : tf.Sequential
        neural network model for the Discriminator
    generator_model : tf.Model
        combined model for the Generator with input (used internally, not by the user)
    discriminator_model : tf.Model
        combined model for the Discriminator with input (userd internally, not by the user)
    

    Methods
    -------   
    init_data(data_fname, anno_fname):
        reads training data.
    init_networks():
        creates network objects.
    _gradient_pentality_loss(y_true, y_pred, averaged_samples):
        calculates the gradient penality loss term.
    info(additional=""):
        prints the person's name and age.
    _wasserstein_loss(y_true, y_pred):
        calculates the Earth-Mover distance
    _hausdorff_distance(real_sample, gen_sample):
        calculates the Hausdorff Distance between real and generated samples
    _build_generator():
        creates the Generator network
    _build_discriminator():
        creates the Discriminator network
    _create_noise(batch_size, dim):
        creates a random noise vector
    train():
        trains the Generator and Discriminator
    _evaluate_model(epoch):
        computes the hausdorff distance between the validation set and generated samples
    produce_loss_graph():
        produces the Hausdorff distance and Wasserstein loss graphs
    produce_similarity_graph():
        produces a similarity graph of a sample of real and generated cells
    _create_checkpoint(epoch):
        saves the model parameters when called
    get_model_summaries():
        writes the initialised model summaries to a txt file
    get_models():
        returns the Generator and Discriminator objects
    get_checkpoint_obj()
        returns the tensorflow checkpoint object
    get_learning_parameters():
        returns a string with all parameters used during training 
    """


    def __init__(
        self, 
        lrate = 0.00001,
        epochs = 100000, 
        batch_size = 32, 
        noise_dim = 100,
        disc_updates = 5,
        grad_pen = 10,
        feature_range = (0,1),
        seed = 2,
        data_path = get_path("../DataPreprocessing/GSE114725"),
        ckpt_freq = 2000,
        ckpt_path = get_path("../models"),
        eval_freq = 5000):

        """
        Constructs all the necessary attributes for the WGANGP object.

        Parameters
        ----------
            lrate : float
                learning rate to be used during training
            epochs : int
                number of training steps
            batch_size : int
                number of samples to on train per epoch
            noise_dim : int
                size of the latent vector
            disc_updates : int
                number of time the Discriminator is updated per epoch 
            grad_pen : float
                gradient penality applied to the loss function
            feature_range : tuple(float, float)
                range in which the training data will be scaled before starting training
            seed : int
                random seed state
            data_path : string
                path to the folder with training data
            ckpt_freq : int
                frequency at which a checkpoint is created during training
            ckpt_path : string
                path to the models folder
            eval_freq : int
                frequency at which the model will be evaluated

        """
        
        # Define model parameters
        self.lrate = lrate
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.disc_updates = disc_updates
        self.grad_pen = grad_pen
        self.feature_range = feature_range

        # Define algorithm parameters
        self.seed = seed
        self.data_path = data_path
        self.ckpt_freq = ckpt_freq
        self.ckpt_path = ckpt_path
        self.eval_freq = eval_freq
        self.file_name = "{}_{}_{}_{}_{}".format(
            self.lrate,
            self.epochs,
            self.batch_size,
            self.noise_dim,
            self.seed)


    def init_data(self, data_fname, anno_fname):

        """
        Reads and initialises the training and validation data.
        Creates placeholder objects for metrics.
        Creates project directory if it does not exist. 

        Parameters
        ----------
            data_fname : string
                name of the file with the sc-RNA seq data
            anno-fname : string
                name of the file with the annotations

        Returns
        ----------
            X_train : array
                training set 
            X_test : array
                validation set
            y_train : array
                training labels
            y_test : array
                validation labels
        """

        print("[INFO] Reading data")

        # Load the dataset
        data = pd.read_csv(get_path(f"{self.data_path}/{data_fname}"), delimiter=",", index_col="cellid")        
        data = data.values

        # Scale data
        data = MinMaxScaler(feature_range=self.feature_range).fit_transform(data)

        # Load the annotations
        anno = pd.read_csv(get_path(f"{self.data_path}/{anno_fname}"), delimiter=",", index_col="cellid")

        # Get the cell labels
        labels = anno['Macro Cell Type'].to_numpy()

        # Split data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data,
            labels,
            train_size = 0.8, 
            random_state = self.seed) 

        self.X_train_n = self.X_train.shape[0]
        self.X_test_n = self.X_test.shape[0]
        self.n_features = data.shape[1]
        self.data_max_value = np.amax(data)

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

        return self.X_train, self.X_test, self.y_train, self.y_test


    def init_networks(self):
        """
        Initialises the networks.
        """

        print("[INFO] Initialising newtorks")


        # Define the Optimizer
        self.optimizer = Adam(learning_rate=self.lrate, beta_1=0.0, beta_2=0.9)

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
        interpolated_img = RandomWeightedAverage(self.batch_size, self.n_features).call([real_sample, fake_sample])
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
        Computes gradient penalty based on prediction and weighted real / fake samples.
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
        """
        Computes the Earth-Mover between two inputs.
        """
        return tf.math.reduce_mean(y_true * y_pred)


    def _hausdorff_dist(self, real_samples, gen_samples):
        """
        Computes the Hausdorff Distance between two arrays.
        """
        dist = directed_hausdorff(
            u = real_samples,
            v = gen_samples
        )[0]

        return dist


    def _build_generator(self):
        """
        Creates the Generator network.
        """

        model = Sequential(name="Generator")

        # Layer 1
        model.add(Dense(
            input_dim=self.noise_dim, 
            units=600, 
            kernel_initializer='he_normal',
            name = "Layer_1"))
        model.add(LeakyReLU(alpha=0.2))

        # Layer 2
        model.add(Dense(
            units=600, 
            kernel_initializer='he_normal',
            name = "Layer_2"))
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
        """
        Creates the Discriminator network.
        """

        model = Sequential(name="Discriminator")

        # Layer 1
        model.add(Dense(
            input_dim=self.n_features, 
            units = 500, 
            kernel_initializer='he_normal',
            name = "Layer_1"))
        model.add(LeakyReLU(alpha=0.2))

        # Layer 2
        model.add(Dense(
            units = 50, 
            kernel_initializer='he_normal',
            name = "Layer_2"))
        model.add(LeakyReLU(alpha=0.2))

        # Output layer
        model.add(Dense(units = 1, 
            activation = None, 
            kernel_initializer='he_normal',
            name = "Output_layer"))

        sample = Input(shape=self.n_features)
        validity = model(sample)


        return Model(sample, validity)

    def _create_noise(self, batch_size, dim):
        """
        Creates a random latent variable of size (batch_size x dim).
        """

        norm = np.random.normal(0.0, self.data_max_value/10, size=(batch_size, dim))
        poisson = np.random.poisson(1, size=(batch_size, dim))

        return np.abs(norm + poisson)

    def train(self):
        """
        Trains the GAN using the WGANGP algorithm.
        Evaluates the network at the eval_freq.
        Saves the model at the ckpt_freq.
        """

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy array for gradient penalty

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
                noise = self._create_noise(self.batch_size, self.noise_dim)
                
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
        """
        Calculates the Hausdorff distance between generated and real samples and saves the graphs.
        """
        
        print("[INFO] evaluating model...", end = "")

        # Create generated validation data
        noise = self._create_noise(self.X_test_n, self.noise_dim)
        generated_samples = self.generator.predict(noise, steps=1)

        # Join generated samples with test set
        combined = np.concatenate((self.X_test, generated_samples), axis = 0)

        # Reduce and scale the dataset with PCA 
        combined_PCA = rescale_arr(PCA(n_components=2).fit_transform(combined))
        pcs = PCA(n_components=50).fit_transform(combined)
        # Calculate the correlation between samples
        hausdorff_dist_PCA = self._hausdorff_dist(combined_PCA[self.val_labels==0], combined_PCA[self.val_labels==1])

        # Reduce and scale the dataset with TSNE
        combined_TSNE = rescale_arr(TSNE(n_components=2).fit_transform(pcs))
        # Calculate the correlation between samples
        hausdorff_dist_TSNE = self._hausdorff_dist(combined_TSNE[self.val_labels==0], combined_TSNE[self.val_labels==1])

        # Reduce and scale the dataset with UMAP 
        combined_UMAP = rescale_arr(UMAP(n_components=2).fit_transform(pcs))
        # Calculate the correlation between samples
        hausdorff_dist_UMAP = self._hausdorff_dist(combined_UMAP[self.val_labels==0], combined_UMAP[self.val_labels==1])

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
        fig.savefig(fname=get_path(f"{self.ckpt_path}/{self.file_name}/images/training_validation_plot_{epoch+1:05d}.png"))
        plt.clf()

        print("done!")

        return None 


    def produce_loss_graph(self):
        """
        Creates the loss graph.
        """

        def interpolate(epochs, data):
            """
            Creates a smoothed array to make the graph continous rather than discrete.
            """

            epochs_new = np.linspace(1, epochs.iloc[-1], epochs.shape[0]*100)
            new_line = make_interp_spline(epochs, data)
            data_new = new_line(epochs_new)

            return pd.Series(epochs_new), pd.Series(data_new)

        print("[INFO] producing loss graph...", end =  "")

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
        combined_losses.to_csv(get_path(f"{self.ckpt_path}/{self.file_name}/data/losses.csv"), index=False)

        # Create training loss graph
        fig, ax = plt.subplots(1, 1, figsize=(axis_size*3, axis_size), squeeze=True)
        ax.plot(gen_losses, linewidth = line_width)
        ax.plot(disc_losses, linewidth = line_width)
        ax.set_ylabel('Wasserstein-1 Distance')
        ax.set_xlabel('Epoch')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        fig.legend(['Generator Loss', 'Discriminator Loss'], loc='lower center', frameon = False, ncol=2)

        fig.savefig(fname=get_path(f"{self.ckpt_path}/{self.file_name}/images/losses_plot.png"))
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
        distance_losses.to_csv(get_path(f"{self.ckpt_path}/{self.file_name}/data/distance_losses.csv"), index=False)

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

        fig.savefig(fname=get_path(f"{self.ckpt_path}/{self.file_name}/images/distances_plot.png"))
        plt.clf()

        print("done!")

        return None

    def produce_similarity_graph(self):
        """
        Creates the similarity graph.
        """

        print("[INFO] Producing similarity plot...", end ="")

        # Define a function to retrieve a sample from the test set
        def get_sample(data, labels, cell_type):
            """
            Returns a single sample from the dataset based on the cell type
            """

            # Subset the cell types
            cells = data[labels == cell_type]
            cells_n = cells.shape[0]

            random_int = np.random.choice(cells_n, size=1, replace=False)

            cell = cells[random_int, :]

            return cell

        # Define a function to get the closest generated cell to a sample
        def get_closest_gen(sample, generated_samples):
            """
            Returns the closest generated cell to a given sample
            """

            distances = []

            for cell in generated_samples:
                distances.append(self._hausdorff_dist(sample, cell.reshape(1,-1)))

            distances = np.array(distances)
            min_id = np.argmin(distances)

            return generated_samples[min_id]


        # Create generated validation data
        noise = self._create_noise(self.X_test_n, self.noise_dim)
        generated_samples = self.generator.predict(noise, steps=1)

        # Get samples for T, NK and B cells
        t_cell = get_sample(self.X_test, self.y_test, "T")
        nk_cell = get_sample(self.X_test, self.y_test, "NK")
        b_cell = get_sample(self.X_test, self.y_test, "B")

        # Get nearest generated sample
        t_cell_gen = get_closest_gen(t_cell, generated_samples)
        nk_cell_gen = get_closest_gen(nk_cell, generated_samples)
        b_cell_gen = get_closest_gen(b_cell, generated_samples)

        # Visualise the validation set
        plot_ratios = {'height_ratios': [1, 1, 1, 1, 1, 1], 'width_ratios': [1]}
        fig, axs = plt.subplots(6, 1, figsize=(axis_size*3, axis_size*4), gridspec_kw=plot_ratios, squeeze=True)

        # T cell plot
        axs[0].plot(t_cell.reshape(-1,), c = red)
        axs[0].title.set_text(f"T cells")
        axs[0].set_ylabel("Expression")

        axs[1].plot(t_cell_gen.reshape(-1,), c = blue)
        axs[1].set_xlabel("Genes")
        axs[1].set_ylabel("Expression")
        box = axs[1].get_position()
        axs[1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # NK cell plot
        axs[2].plot(nk_cell.reshape(-1,), c = red, label = "Real")
        axs[2].title.set_text(f"NK cells")
        axs[2].set_ylabel("Expression")

        axs[3].plot(nk_cell_gen.reshape(-1,), c = blue, label = "Generated")
        axs[3].set_xlabel("Genes")
        axs[3].set_ylabel("Expression")
        box = axs[3].get_position()
        axs[3].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # B cell plot
        axs[4].plot(b_cell.reshape(-1,), c = red)
        axs[4].title.set_text(f"B cells")
        axs[4].set_ylabel("Expression")

        axs[5].plot(b_cell_gen.reshape(-1,), c = blue)
        axs[5].set_xlabel("Genes")
        axs[5].set_ylabel("Expression")
        box = axs[5].get_position()
        axs[5].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        fig.legend(loc = "lower center", ncol = 2, frameon = False, bbox_to_anchor = [0.5, 0.08])

        fig.savefig(fname=get_path(f"{self.ckpt_path}/{self.file_name}/images/similarity_plot.png"))
        plt.clf()

        print("done!")

        return None

    def evaluate_dimensionality_reduction(self):
        """
        Creates a neural network from the Discriminator where the final hidden layer is the outputs.
        Reduces the validation set and calculates cluster metrics compared to the baseline data.
        Creates plots of the GAN reduced data and baseline data.
        """

        # Define a function to extract metrics
        def get_metrics(data, labels):
            """
            Calculates cluster metrics for given data and labels

            Parameters
            ----------
                data : array
                    data to be evaluated
                labels : array
                    cluster labels corresponding to the data

            Returns
            ----------
            metrics_list : DataFrame
                metrics computed from cluster evaluation
            """

            # Create an empty list
            metrics_list = []

            # Append the metrics to a list
            metrics_list.append(silhouette_score(data, labels))
            metrics_list.append(calinski_harabasz_score(data, labels))

            # Combine results into a Pandas Series
            metrics_labels = [
                "Silhouette Score",
                "Calinski-Harabasz Score"
                ]
            
            # Reshape the array
            metrics_list = np.array(metrics_list).reshape(1, -1)

            # Create Pandas series
            metrics_list = pd.DataFrame(data = metrics_list, columns=metrics_labels)

            
            return metrics_list

        # Create a model which outputs the Discriminator hidden layer
        self.discriminator_hidden = tf.keras.Model(self.discriminator.layers[1].layers[0].input, self.discriminator.layers[1].layers[3].output)
        print("[INFO] Model created")

        # Reduce the dimensionality of the data using the Discriminator
        X_test_reduced = self.discriminator_hidden(self.X_test).numpy()

        # Rescale the data
        X_test_reduced = MinMaxScaler().fit_transform(X_test_reduced)

        ### Perform further dimensionality reduction ###
        print("[INFO] Reducing dimensions...")

        # Reduce the datasets with PCA
        pcs = PCA(n_components=50).fit_transform(self.X_test)
        pca = PCA(n_components=2).fit_transform(self.X_test)
        pca_gan = PCA(n_components=2).fit_transform(X_test_reduced)
        print("[INFO] PCA complete")

        # Reduce the datasets with TSNE
        tsne = TSNE().fit_transform(pcs)
        tsne_gan = TSNE().fit_transform(X_test_reduced)
        print("[INFO] t-SNE complete")

        # Reduce the datasets with UMAP
        umap = UMAP().fit_transform(pcs)
        umap_gan = UMAP().fit_transform(X_test_reduced)
        print("[INFO] UMAP complete")

        print("[INFO] Data has been reduced")

        # Compute metrics for PCA
        pca_metrics = get_metrics(pca, self.y_test)
        pca_gan_metrics = get_metrics(pca_gan, self.y_test)

        # Compute metrics for t-SNE
        tsne_metrics = get_metrics(tsne, self.y_test)
        tsne_gan_metrics = get_metrics(tsne_gan, self.y_test)

        # Compute metrics for umap
        umap_metrics = get_metrics(umap, self.y_test)
        umap_gan_metrics = get_metrics(umap_gan, self.y_test)


        ### Combine and export metrics ###
        row_names = pd.Series([
            "PCA", "GAN+PCA",
            "TSNE", "GAN+TSNE",
            "UMAP", "GAN+UMAP"], name = "Index")

        combined_metrics = pd.concat([
            pca_metrics,
            pca_gan_metrics,
            tsne_metrics,
            tsne_gan_metrics,
            umap_metrics,
            umap_gan_metrics],
            axis = 0).reset_index(drop = True)

        combined_metrics["Index"] = row_names
        combined_metrics = combined_metrics.set_index("Index").round(2)

        # Save dataframe as csv
        combined_metrics.to_csv(get_path(f"{self.ckpt_path}/{self.file_name}/metrics/dimensionality_reduction_metrics.csv"))
        print("[INFO] Evaluation metrics created")


        ### Plot the data ###
        # Create a mapping between classes and colours
        color_map = {
            "B"   : "#00798c",
            "MACROPHAGE"    : "#C77CFF",
            "MAST" : "#edae49",
            "NEUTROPHIL"  : "#66a182",
            "NK"   : "#2e4057",
            "NKT": "#8c8c8c",
            "T": "#f37735",
            "mDC": "#d11141",
            "pDC":"#A6611A"}

        patchList = []
        for key in color_map:
                data_key = plt.scatter([],[], s = point_size*3, marker=".", color = color_map[key], label=key)
                patchList.append(data_key)

        plot_colors = list(map(color_map.get, self.y_test))

        # Plot the results for sample correlation
        axis_size = 8.0
        plot_ratios = {'height_ratios': [1,1,1.1], 'width_ratios': [1,1]}
        fig, axs = plt.subplots(3, 2, figsize=(axis_size*3, axis_size*3), gridspec_kw=plot_ratios, squeeze=True)

        # PCA plot
        axs[0,1].scatter(pca[:,0], pca[:,1], c = plot_colors, s = point_size)
        axs[0,1].title.set_text(f"PCA")
        axs[0,1].set_xlabel("PC 1")
        axs[0,1].set_ylabel("PC 2")

        # GAN + PCA plot
        axs[0,0].scatter(pca_gan[:,0], pca_gan[:,1], c = plot_colors, s = point_size)
        axs[0,0].title.set_text(f"GAN + PCA")
        axs[0,0].set_xlabel("PC 1")
        axs[0,0].set_ylabel("PC 2")

        # t-SNE plot
        axs[1,1].scatter(tsne[:,0], tsne[:,1], c = plot_colors, s = point_size)
        axs[1,1].title.set_text(f"t-SNE")
        axs[1,1].set_xlabel("TSNE 1")
        axs[1,1].set_ylabel("TSNE 2")

        # GAN + t-SNE plot
        axs[1,0].scatter(tsne_gan[:,0], tsne_gan[:,1], c = plot_colors, s = point_size)
        axs[1,0].title.set_text(f"GAN + t-SNE")
        axs[1,0].set_xlabel("TSNE 1")
        axs[1,0].set_ylabel("TSNE 2")

        # UMAP plot
        axs[2,1].scatter(umap[:,0], umap[:,1], c = plot_colors, s = point_size)
        axs[2,1].title.set_text(f"UMAP")
        axs[2,1].set_xlabel("UMAP 1")
        axs[2,1].set_ylabel("UMAP 2")
        box = axs[2,1].get_position()
        axs[2,1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # GAN + UMAP plot
        axs[2,0].scatter(umap_gan[:,0], umap_gan[:,1], label = "GAN + UMAP", c = plot_colors, s = point_size)
        axs[2,0].title.set_text(f"GAN + UMAP")
        axs[2,0].set_xlabel("UMAP 1")
        axs[2,0].set_ylabel("UMAP 2")
        box = axs[2,0].get_position()
        axs[2,0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        fig.legend(handles = patchList, loc = "lower center", ncol = 9, frameon = False, markerscale=2, bbox_to_anchor = [0.5, 0.075])

        fig.savefig(fname=get_path(f"{self.ckpt_path}/{self.file_name}/images/dimensionality_reduction_plot.png"))
        plt.clf() 
        print("[INFO] Evaluation plot saved")

        return None


    def reduce_all_data(self):
        """
        Reduces all data to the dimension specified in the final hidden layer of the Discriminator.
        """

        X = np.concatenate(
            (self.X_train, self.X_test),
            axis = 0
        )

        # Reduce the dimensionality of the data using the Discriminator
        data_gan_reduced = self.discriminator_hidden(X).numpy()
        data_gan_reduced = data_gan_reduced + abs(data_gan_reduced.min())
        data_gan_reduced = MinMaxScaler().fit_transform(data_gan_reduced)

        # Save the data
        col_names = [f"Component {i+1}" for i in range(data_gan_reduced.shape[1])]
        export_data = pd.DataFrame(data = data_gan_reduced, columns=col_names)

        export_data.to_csv(
            get_path(f"{self.ckpt_path}/{self.file_name}/data/data_reduced_gan.csv"), 
            sep=",",
            index=False)

        print("[INFO] Data has been reduced and saved")

        return None


    def _create_checkpoint(self, epoch):
        """
        Saves the models and optimizer.
        """

        path = get_path("{}/{}/epochs/{:05d}/ckpt".format(
            self.ckpt_path,
            self.file_name,
            epoch+1))

        print("[INFO] creating checkpoint")
        self.checkpoint.save(file_prefix = path)

        return None


    # Define a function to print model summaries
    def get_model_summaries(self):
        """
        Creates a .txt file with the Generator and Discriminator model summaries
        """

        print("[INFO] Printing Model Summaries\n")

        print(self.generator.layers[1].summary(), end = "\n\n")
        print(self.discriminator.layers[1].summary(), end = "\n\n")

        gen, disc = [], []

        self.generator.layers[1].summary(print_fn=lambda x: gen.append(x))
        self.discriminator.layers[1].summary(print_fn=lambda x: gen.append(x))

        with open(get_path(f"{self.ckpt_path}/{self.file_name}/model_summaries.txt"), "w") as f:

            for item in gen:
                f.write(f"{item}\n")

            f.write("\n")

            for item in disc:
                f.write(f"{item}\n")

        return None

    # Define a functon to get the models
    def get_models(self):
        """
        Returns the Generator and Discriminator objects
        """

        return self.generator, self.discriminator

    # Define a function to get the checkpoint option
    def get_checkpoint_obj(self):
        """
        Returns the tensorflow checkpoint object
        """

        return self.checkpoint

    # Define a function to print learning parameters
    def get_learning_parameters(self):
        """
        Creates a .txt file with a summary of the model parameters
        """

        parameters = f"""---Learning Parameters---
Learning rate = {self.lrate}
Epochs = {self.epochs}
Batch size = {self.batch_size}
Noise size = {self.noise_dim}
Discriminator updates = {self.disc_updates}
Gradient Penality = {self.grad_pen}
Feature Range = {self.feature_range}
Seed = {self.seed}
Number of features {self.n_features}
Training set size = {self.X_train_n}
Test set size = {self.X_test_n}
        """

        print(parameters)

        with open(get_path(f"{self.ckpt_path}/{self.file_name}/model_params.txt"), "w") as f:

            f.write(parameters)

        return None


if __name__ == '__main__':

    # Set up the TensorFlow environment
    disable_eager_execution()
    tf.compat.v1.experimental.output_all_intermediates(True)

    # Create the WGANGP class
    gan = WGANGP(
        lrate = 0.00005,
        epochs = 300000,
        batch_size = 128,
        noise_dim = 100,
        disc_updates = 5,
        grad_pen = 10,
        feature_range=(0,1),
        seed = 1001,
        data_path = "../DataPreprocessing/GSE114725",
        ckpt_path="../models",
        ckpt_freq = 2000,
        eval_freq = 5000
    )

    # Initialise data and networks
    gan.init_data(data_fname = "GSE114725_processed_data_10000_3912.csv", anno_fname = "GSE114725_processed_annotations_10000_3912.csv")
    gan.init_networks()

    # Get summaries of model and parameters
    gan.get_model_summaries()
    gan.get_learning_parameters()

    # Train the GAN
    gan.train()

    # Evaluate the GAN
    gan.produce_similarity_graph()
    gan.produce_loss_graph()