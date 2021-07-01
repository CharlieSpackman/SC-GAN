# evaluate_scPred.R

# Set working directory
setwd("C:/Users/spack/OneDrive - King's College London/Individual Project/Single Cell Sequencing with GANs/Implementation/ModelEvaluation/Cell Classification/scPred")
data_path = "C:/Users/spack/OneDrive - King's College London/Individual Project/Single Cell Sequencing with GANs/Implementation/DataPreprocessing/sc_integrated_data.csv"

# Load the required libraries
library("scPred")
library("Seurat")
library("magrittr")
library("tidyverse")
library("data.table")
require(caTools)
set.seed(30) 

# Define a function to create the Seurat object
create_Seurat <- function(data, anno_names, target_name=NA, train=FALSE) {
  
  # Split out cell labels
  anno <- data[, anno_names]
  data <- data[, ! colnames(data) %in% anno_names]
  
  # transpose the data
  data_t <- transpose(data)
  
  # get row and colnames in order
  colnames(data_t) <- rownames(data)
  rownames(data_t) <- colnames(data)
  
  # Create Seurat object
  data_S <- CreateSeuratObject(counts = data_t, meta.data = anno)
  
  # Preprocess the data
  data_S <- FindVariableFeatures(object = data_S)
  data_S <- ScaleData(object = data_S, assay="RNA")
  
  data_S <- RunPCA(data_S, features = VariableFeatures(object = data_S))
  data_S <- RunUMAP(data_S, dims = 1:30)
  
  if(train==TRUE){
  
    # Get Feature Space
    data_S <- getFeatureSpace(data_S, target_name)
    
    # Train the classifier
    data_S <- trainModel(data_S)

  }
  
  return(data_S)
  
}
  

# Read in data
data <- read.csv(data_path)

anno_names = c("X", 
               "Sample",
               "Celltype..major.lineage.",
               "Celltype..minor.lineage.",
               "Age",
               "Gender")


# Split the data into training and testing datasets
sample = sample.split(data$X, SplitRatio = .80)
train_data = subset(data, sample == TRUE)
test_data  = subset(data, sample == FALSE)

# Create Seurat objects for both
train_data_S = create_Seurat(data = train_data, 
                             anno_names = anno_names, 
                             target_name = "Celltype..major.lineage.",
                             train = TRUE)

test_data_S = create_Seurat(data = test_data, 
                             anno_names = anno_names, 
                             target_name = "Celltype..major.lineage.",
                             train = FALSE)


# Get predictions for samples in the validation set
test_data_S <- scPredict(test_data_S, train_data_S)

# Evaluate the performance of the classifier on the test set

# Get predictions
predictions = test_data_S@meta.data$scpred_prediction

# Get actual labels
labels = test_data_S@meta.data$Celltype..major.lineage.

# Performance metrics
accuracy = mean(predictions == labels)
error = 1 - accuracy
results = table(predictions,labels)

# Print results
print(paste("Accuracy:", accuracy))
print(results)
