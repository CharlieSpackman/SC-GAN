# evaluate_scPred.R

# Load the required libraries
library("scPred")
library("Seurat")
library("magrittr")
library("tidyverse")
library(data.table)
require(caTools)
set.seed(30) 

# Set working directory
setwd("C:/Users/spack/OneDrive - King's College London/Individual Project/Single Cell Sequencing with GANs/Implementation/ModelEvaluation/Cell Classification/scPred")

# Get file path for the GAN reduced data
MODEL_NAME = "5e-05_300000_128_100_1" ### Update this as required
EPOCH = 150000 ### Update this as required

# Define the file path
gan_data_path = paste(
    "C:/Users/spack/OneDrive - King's College London/Individual Project/Single Cell Sequencing with GANs/Implementation/models/",
    MODEL_NAME,
    "/data/data_reduced_gan_",
    EPOCH,
    ".csv",
    sep = "")

# Get file paths for the baseline data
baseline_data_path = "C:/Users/spack/OneDrive - King's College London/Individual Project/Single Cell Sequencing with GANs/Implementation/DataPreprocessing/GSE114725/GSE114725_processed_data_10000_3912.csv" ### Update this as required

# Get the annotations
anno_path = "C:/Users/spack/OneDrive - King's College London/Individual Project/Single Cell Sequencing with GANs/Implementation/DataPreprocessing/GSE114725/GSE114725_processed_annotations_10000_3912.csv" ### Update this as required

#-----------------------
# scPred Function
#-----------------------

scPred <- function(data, anno) {

  # Takes the scRNA-data and cell annotations as inputs
  # Retruns the set of predictions and true labels for the validation data
  
  # transpose the data
  data_t <- transpose(data)

  # get row and colnames in order
  colnames(data_t) <- rownames(data)
  rownames(data_t) <- colnames(data)
  
  # Split the data into training and testing datasets
  smp_size <- floor(0.75 * ncol(data_t))
  train_ind <- sample(seq_len(ncol(data_t)), size = smp_size)
                      
  train_data = data_t[, train_ind]
  test_data  = data_t[, -train_ind]
  
  # Split the labels into training and testing datasets
  train_y = anno[train_ind, ]
  test_y  = anno[-train_ind, ]
  
  # Create Seurat objects
  train_data_S <- CreateSeuratObject(counts = train_data, meta.data = train_y)
  test_data_S <- CreateSeuratObject(counts = test_data, meta.data = test_y)
  
  # Pre-process the data
  train_data_S <- FindVariableFeatures(object = train_data_S)
  train_data_S <- ScaleData(object = train_data_S, assay="RNA")
  train_data_S <- RunPCA(train_data_S, features = VariableFeatures(object = train_data_S))
  train_data_S <- RunUMAP(train_data_S, dims = 1:30)
  
  # Get Feature Space
  train_data_S <- getFeatureSpace(train_data_S, "Macro.Cell.Type")
  
  # Train the classifier
  train_data_S <- trainModel(train_data_S)
  
  # Get predictions
  test_data_S <- scPredict(test_data_S, train_data_S, threshold = 0.0)
  predictions <- test_data_S@meta.data$scpred_prediction
  
  # Get labels
  labels <- test_data_S@meta.data$Macro.Cell.Type
  
  # Create dataframe
  output <- data.frame(predictions, labels)
  
  return(output)
  
}

# Read in labels
anno <- read.csv(anno_path)

#-----------------------
# GAN Predictions
#-----------------------

# Read in GAN reduced data
gan_reduced_data <- read.csv(gan_data_path)

# Get predictions for GAN reduced data
gan_predictions = scPred(gan_reduced_data, anno)

# Define file path to save to
gan_path = paste(
  "C:/Users/spack/OneDrive - King's College London/Individual Project/Single Cell Sequencing with GANs/Implementation/models/",
  MODEL_NAME,
  "/metrics/scPred_reduced_gan_predictions_",
  EPOCH,
  ".csv",
  sep = "")

# Save the predictions as a .csv filex
write.csv(gan_predictions, gan_path, row.names = FALSE)

#-----------------------
# Baseline Predictions
#-----------------------

# Read in GAN reduced data
baseline_data <- read.csv(baseline_data_path)
baseline_data <- baseline_data[, 2:ncol(baseline_data)]

# Get predictions for GAN reduced data
baseline_predictions = scPred(baseline_data, anno)

# Define file path to save to
baseline_path = paste(
  "C:/Users/spack/OneDrive - King's College London/Individual Project/Single Cell Sequencing with GANs/Implementation/models/",
  MODEL_NAME,
  "/metrics/scPred_baseline_predictions_",
  EPOCH,
  ".csv",
  sep = "")

# Save the predictions as a csv file
write.csv(baseline_predictions, baseline_path, row.names = FALSE)