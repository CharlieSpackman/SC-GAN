#

# Import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import NuSVC
from sklearn import metrics 

# Evaluation parameters
FILE_PATH = "C:\\Users\\spack\\OneDrive - King's College London\\Individual Project\\Single Cell Sequencing with GANs\\Implementation\\DataPreprocessing\\sc_integrated_data.csv"
SEED = 30

# Read in data
data = pd.read_csv(FILE_PATH, delimiter=',')
data = data.set_index(data.columns.values[0])

# Split out the annotations
anno = data[data.columns.values[-5:]]
data = data.drop(data.columns.values[-5:], axis = 1)
data = data.astype(np.float64)

# Split into tran and test data
train_data, test_data, train_anno, test_anno = train_test_split(data, anno, test_size = 0.9, random_state = SEED)

# Create PCA object
train_data_rm = PCA(n_components=2).fit_transform(train_data)
test_data_rm = PCA(n_components=2).fit_transform(test_data)

# Train SVM model
svm_model = NuSVC(nu=0.02, 
    kernel='linear',
    decision_function_shape='ovo',
    random_state=SEED)

# Fit the model
svm_model.fit(train_data_rm, train_anno.iloc[:,1])

# Evaluate performance using the test set
predictions = svm_model.predict(test_data_rm)
labels = test_anno.iloc[:,1]

# Performance metrics
accuracy = metrics.accuracy_score(predictions, labels)

print(f"Accuracy: {round(accuracy,2)}")



