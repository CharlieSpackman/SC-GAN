#------------------
# A. Archive Contents
#------------------

__init__.py                                 - blank file to enable the WGANGP class on the python path
cell_types.csv                              - annotations (labels) for the training data provided
classification_metrics.py                   - computes prediction performance metrics based on output from scPred
dimensionality_reduction_evaluation.py      - compute dimensionality reduction metrics and reduces the dataset
GSE114725_data_processing.py                - pre-processing for the filtered imputed values
GSE114725_filter_data.py                    - removes outlier samples and samples 10000 items from the raw_imputed.csv 
requirements_python.txt                     - list of Python modules used
requirements_R.txt                          - list of R packages used
scPred.R                                    - trains scPred models on the GAN reduced and baseline data and outputs the cell type predictions
WGANGP.py                                   - main class for training and evaluating the GAN

#------------------
# B. Instructions
#------------------

In order to run the code a directory with the structure specified in C. Directory Structure must be created.
Once the directory has been created the user should then complete the following steps:

    1. Download the raw data (imputed_corrected.csv) from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114725.
    2. Filter the raw imputed data for the tumour cells by running GSE114725_filter_data.py.
    3. Pre-process the data using GSE114725_data_processing.py.
    4. Specify model parameters (or leave the default parameters) in WGANGP.py and run the file to train the model. 
       All relevant evaluation metrics, images and checkpoints will be located within the models/model_name folder. 
       The model_name directory is automatically created when running the file.
    5. Once the GAN training is complete, update the file names in dimensionality_reduction_evaluation.py and run to produce the reduced GAN data and metrics 
    6. Update the file names in scPred.R and run the file to train classification models. 
       Once completed, predictions will be saved in models/model_name/metrics.
    7. Update the file names in classification_metrics.py and run the file to evaluate the model performances. Metrics will be saved in models/model_name/metrics

After completing the above steps the model will have been created and evaluated.
The directory models/model_name will contain the following directories:

    images  - evaluation and training plots
    metrics - evaluation metrics for dimensionality reduction and cell classification
    data    - losses and Discriminator reduced data
    epochs  - checkpoints containing model weights at specific epochs

#------------------
# C. Directory Structure
#------------------

The following directory structure and files should be created and retrieved in order to run the code. 
The files excluding the imputed_corrected.csv file can be found in the source code folder.
The imputed_corrected.csv file can be downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114725.

SCGAN
+---DataPreprocessing
|   |   GSE114725_data_processing.py
|   |   GSE114725_filter_data.py
|   |
|   +---GSE114725
|   |       cell_types.csv
|   |       imputed_corrected.csv
|                  
+---ModelCreation
|   |   WGANGP.py
|   |   __init__.py
|          
+---ModelEvaluation
|   |
|   +---DimensionalityReduction
|   |   |   dimensionality_reduction_evaluation.py
|   |
|   +---CellClassification
|   |   |   classification_metrics.py
|   |   |
|   |   +---scPred
|   |       |   scPred.R
|           
+---models
|   |

#------------------
# D. Requirements
#------------------

Modules and Packages used in the implementation of this project can be found in requirements_python.txt and requirements_R.txt.
It is recommended that users attempting to run the code should have the requirements installed on their system.
Conda was used in order to install the Python requirements.
RStudio was used to install the R requirements.
