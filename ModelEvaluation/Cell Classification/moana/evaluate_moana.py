# evaluate_moana.py

# Import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import NuSVC

# Evaluation parameters
MODEL_NAME = "0.001_0.001_100_250_100_30" # UPDATE AS NEEDED
EPOCH = 100 # UPDATE AS NEEDED

# Paths for reduced data
MODEL_PATH = "C:\\Users\\spack\\OneDrive - King's College London\\Individual Project\\Single Cell Sequencing with GANs\\Implementation\\models\\"
FILE_PATH = f"{MODEL_PATH}\\{MODEL_NAME}\\data\\data_reduced_gan_{EPOCH:05d}.csv"

# Paths for baseline data
BASELINE_DATA_PATH = "C:\\Users\\spack\\OneDrive - King's College London\\Individual Project\\Single Cell Sequencing with GANs\\Implementation\\DataPreprocessing\\GSE114727\\GSE114727_processed_data.csv"
ANNO_PATH = "C:\\Users\\spack\\OneDrive - King's College London\\Individual Project\\Single Cell Sequencing with GANs\\Implementation\\DataPreprocessing\\GSE114727\\GSE114727_processed_annotations.csv"
SEED = 30

# Read in the annotations
print("[INFO] Reading data")
anno = pd.read_csv(ANNO_PATH, delimiter=',', index_col = 0)
labels = anno["Celltype (major-lineage)"].reset_index(drop=True).rename("labels")


# Define a function to create and evaluate a cell classifier using Moana 
def moana(data, labels):

    # Ignore the index on the dataframe
    data.reset_index(drop=True, inplace=True)

    # Create PCA object
    data = PCA(n_components=2).fit_transform(data)

    # Split into tran and test data
    train_data, test_data, train_y, test_y = train_test_split(data, labels, test_size = 0.25, random_state = SEED)


    # Train SVM model
    svm_model = NuSVC(nu=0.02, 
        kernel='linear',
        decision_function_shape='ovo',
        random_state=SEED)

    # Fit the model
    svm_model.fit(train_data, train_y)

    # Evaluate performance using the test set
    predictions = pd.Series(svm_model.predict(test_data), name = "predictions", index=None)

    output = pd.concat([predictions, test_y.reset_index(drop=True)], axis = 1)

    return output

print("[INFO] Evaluating predictions")
### Get prdictions for GAN reduced data ###
# Read in GAN reduced data
gan_reduced_data = pd.read_csv(FILE_PATH, delimiter=',', index_col=0).astype(np.float64)
# Apply Moana to the GAN reduced data
gan_reduced_predictions =  moana(gan_reduced_data, labels)
# Save the predictions as a csv
gan_reduced_predictions.to_csv(f"{MODEL_PATH}\{MODEL_NAME}\metrics\moana_reduced_gan_predictions_{EPOCH:05d}.csv", index=False)

### Get predictions for baseline data ###
# Read in GAN reduced data
baseline_data = pd.read_csv(BASELINE_DATA_PATH, delimiter=',', index_col=0).astype(np.float64)
# Apply Moana to the GAN reduced data
baseline_predictions =  moana(baseline_data, labels)
# Save the predictions as a csv
baseline_predictions.to_csv(f"{MODEL_PATH}\\{MODEL_NAME}\\metrics\\moana_baseline_predictions_{EPOCH:05d}.csv", index=False)

print("[INFO] Evaluation complete")