# GSE114725_filter_data.py
"""
Takes raw inputed data, filters for Tumor samples and samples 10000 cells 

Inputs
------
    imputed_corrected.csv
        raw data

Outputs
-------
    GSE114725_data_filtered_10000.csv
        filtered and sampled data
"""

# Import pandas
import pandas as pd
import numpy as np

# Read the data - update name as neccessary
data = pd.read_csv("GSE114725\\imputed_corrected.csv", index_col = "cellid")

# Subset Tumor cells
tumor_data = data.loc[data['tissue'] == "TUMOR", :]

# Read in the labels
labels = pd.read_csv("GSE114725\\cell_types.csv", index_col="ClusterID")

# Join the labels onto the data
tumor_data = pd.merge(tumor_data, labels, how = "left", left_on="cluster", right_index=True)

# Reorder the columns
cols = list(tumor_data)
cols.insert(4, cols.pop(cols.index('Macro Cell Type')))
tumor_data = tumor_data.loc[:, cols]

cols = list(tumor_data)
cols.insert(5, cols.pop(cols.index('Micro Cell Type')))
tumor_data = tumor_data.loc[:, cols]

# Remove uninteresting columns
tumor_data = tumor_data.drop(['tissue', 'replicate'], axis = 1)

# Remove Patient 6 and Monocyte
tumor_data = tumor_data.loc[tumor_data["patient"] != "BC6"]
tumor_data = tumor_data.loc[tumor_data["Macro Cell Type"] != "MONOCYTE"]

# Sample the data
sample_n = 10000
sample_data = tumor_data.sample(n = sample_n)

# Split out the annotations and write to csv
tumor_anno = sample_data.iloc[:,:4]
tumor_anno.to_csv("GSE114725\\GSE114725_annotations_filtered_10000.csv")

# Split out the genes and write to csv
tumor_genes = sample_data.columns.values[4:].astype(str)
np.savetxt("GSE114725\\GSE114725_genes_filtered_10000.csv", tumor_genes, delimiter = ",", fmt='%s')


# Write values to csv
np.savetxt("GSE114725\\GSE114725_data_filtered_10000.csv", sample_data.iloc[:, 4:].values, delimiter = ",")


