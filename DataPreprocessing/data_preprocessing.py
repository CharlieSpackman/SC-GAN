# data_preprocessing.py

# Import modules
import numpy as np
import pandas as pd
import scanpy as sc

# Set scanpy settings
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc._settings.ScanpyConfig.cache_compression = "gzip"
results_file = 'results.h5ad'

# Processing parameters
FILE_PATH = "GSE114727/"

### Reading in data ###
print("[INFO] Reading data")

# Read in data into a AnnData objects
adata_09_1 = sc.read_10x_mtx(
    FILE_PATH + 'GSM3148575_BC09_TUMOR1',
    prefix= "GSM3148575_BC09_TUMOR1_",
    cache=True)

adata_09_2 = sc.read_10x_mtx(
    FILE_PATH + 'GSM3148576_BC09_TUMOR2',
    prefix= "GSM3148576_BC09_TUMOR2_",
    cache=True)

adata_10 = sc.read_10x_mtx(
    FILE_PATH + 'GSM3148577_BC10_TUMOR1',
    prefix= "GSM3148577_BC10_TUMOR1_",
    cache=True)

adata_11_1 = sc.read_10x_mtx(
    FILE_PATH + 'GSM3148578_BC11_TUMOR1',
    prefix= "GSM3148578_BC11_TUMOR1_",
    cache=True)

adata_11_2 = sc.read_10x_mtx(
    FILE_PATH + 'GSM3148579_BC11_TUMOR2',
    prefix= "GSM3148579_BC11_TUMOR2_",
    cache=True)

print("[INFO] Data successfully read")


# Print summaries for all datasets
dataset_names = ["GSM3148575_BC09_TUMOR1", "GSM3148576_BC09_TUMOR2", "GSM3148577_BC10_TUMOR1", "GSM3148578_BC11_TUMOR1", "GSM3148579_BC11_TUMOR2"]
dataset_list = [adata_09_1, adata_09_2, adata_10, adata_11_1, adata_11_1]
datasets = zip(dataset_names, dataset_list)
obs_n = 0

print("----Dataset summaries----")
for dataset in datasets:
    print(dataset[0])
    print(f"Obs: {dataset[1].shape[0]}, Vars: {dataset[1].shape[1]}\n")
    obs_n += dataset[1].shape[0]

print(f"Total obs: {obs_n}")
print(f"Total Vars {dataset[1].shape[1]}")

# Integrate datasets
integrated_adata = adata_09_1.concatenate(
    adata_09_2,
    adata_10,
    adata_11_1,
    adata_11_2, 
    batch_categories=["BC09_TUMOR1", "BC09_TUMOR2", "BC10_TUMOR1", "BC11_TUMOR1", "BC11_TUMOR2"],
    join = "outer",
    batch_key = "Sample",
    index_unique = "|")

### Read in metadata ###
print("[INFO] Reading metadata")

# Define a helper function to remap the cell ID in the metadata file
def split_id(id):
    id_split = id.split("@")
    return id_split[1] + "|" + id_split[0]

# Read in metadata
metadata = pd.read_csv(FILE_PATH + "BRCA_GSE114727_10X_CellMetainfo_table.tsv", sep = "\t")

# Remap the cell ID and set it as the dataframe index
metadata["CellID"] = metadata["Cell"].apply(split_id)
metadata = metadata.set_index("CellID")

# Filter out uninteresting variables
metadata = metadata[
    ["Sample",
    "Celltype (major-lineage)",
    "Celltype (minor-lineage)",
    "Age",
    "Gender"]]

# Filter out cells in the dataset that have no metadata
common = integrated_adata.obs.index.isin(metadata.index.to_series())
integrated_adata = integrated_adata[common]

print(f"[INFO] Removed {obs_n - common.sum()} samples due to missing metadata")

# Reorder the metadata rows to be consistent with the data
integrated_adata_index = integrated_adata.obs.index.to_series()
metadata.reindex(integrated_adata_index)

# Add metadata to adata object
integrated_adata.obs = metadata

### Data processing ###
print("[INFO] Commencing data pre-processing")


# Basic filtering
sc.pp.filter_cells(integrated_adata, min_genes=200)
sc.pp.filter_genes(integrated_adata, min_cells=3)

# Compute metrics
sc.pp.calculate_qc_metrics(integrated_adata, percent_top=None, log1p=False, inplace=True)

# Check data quality with violin and scatter plots
sc.pl.violin(integrated_adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)
sc.pl.scatter(integrated_adata, x='total_counts', y='n_genes_by_counts')

# Remove samples with too many gene counts
integrated_adata = integrated_adata[integrated_adata.obs.n_genes_by_counts < 5000, :]

# Normalise and logarithmise the matrix
sc.pp.normalize_total(integrated_adata, target_sum=1e4)
sc.pp.log1p(integrated_adata)

# Identify highly expressed genes
sc.pp.highly_variable_genes(integrated_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(integrated_adata)

# Set raw attribute to be the processed data
integrated_adata.raw = integrated_adata

# Filter out non-highly expressed genes
integrated_adata = integrated_adata[:, integrated_adata.var.highly_variable]

# Scale the data
sc.pp.scale(integrated_adata)

### Checking data quality ###
print("[INFO] Checking data quality")

# Perform PCA
sc.tl.pca(integrated_adata, svd_solver='arpack')
sc.pl.pca(integrated_adata)

# Identify the PCs contributing to the total variance
sc.pl.pca_variance_ratio(integrated_adata, log=True)

# Carry out UMAP
sc.pp.neighbors(integrated_adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(integrated_adata)
sc.pl.umap(integrated_adata)

# Layer the cell labels on the umap
sc.pl.umap(integrated_adata, color='Celltype (major-lineage)')

print("[INFO] Exporting processed data")

# Export data for model creation and evaluation
export_data_values = pd.DataFrame(integrated_adata.X, index = integrated_adata.obs.index, columns = integrated_adata.var.index)
export_data_anno = pd.DataFrame(integrated_adata.obs.values, index = integrated_adata.obs.index, columns = integrated_adata.obs.columns.values)
export_data_anno = export_data_anno.iloc[:,:-3]


### Generate a sample ###
# sample_n = 8000
# export_data_values = export_data_values.sample(sample_n)
# export_data_anno = export_data_anno.loc[export_data_values.index,:]

# Save count matrix
export_data_values.to_csv("GSE114727\\GSE114727_processed_data.csv")

# Save annotations
export_data_anno.to_csv("GSE114727\\GSE114727_processed_annotations.csv")

# Write results
integrated_adata.write(results_file)

print("[INFO] Data successfully exported")