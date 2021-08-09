# GSE114725_data_processing.py
"""
Takes filtered, sampled data and pre-processes the dataset.
Produces overview plots.

Inputs
------
    GSE114725_data_filtered_10000.csv
        filtered, sampled data

Outputs
-------
    GSE114725_processed_data_10000_3912.csv
        processed expression values
    GSE114725_processed_annotations_10000_3912.csv
        processed annotations
    GSE114725_10000_3912_plot
        PCA, t-SNE and UMAP of the processed data with cell labels
    GSE114725_10000_3912_patient_plot
        PCA, t-SNE and UMAP of the processed data with patient labels
    
"""

# Import modules
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP

# Set scanpy settings
sc.settings.verbosity = 3
sc.logging.print_header()
sc._settings.ScanpyConfig.cache_compression = "gzip"
results_file = 'results.h5ad'
FILE_PATH = "GSE114725\\"

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
point_size = 25
axis_size = 8


### Reading in data ###
print("[INFO] Reading data")
adata = sc.read_csv(FILE_PATH + "GSE114725_data_filtered_10000.csv")
# Read in annotations
anno = pd.read_csv(FILE_PATH + "GSE114725_annotations_filtered_10000.csv")
# Read in genes
genes = pd.read_csv(FILE_PATH + "GSE114725_genes_filtered_10000.csv", header=None, names = ["Gene IDs"])

# Add annotations and genes to the AnnData object
adata.obs = anno
adata.var = genes
adata.raw = adata

# Filter genes
sc.pp.filter_genes(adata, min_cells=3)

# Rescale data
adata.X = adata.X + abs(adata.X.min())

# Calculate metrics
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Remove samples with too many gene counts
adata = adata[adata.obs.n_genes_by_counts > 12000, :]

# Identify high variable genes
sc.pp.highly_variable_genes(adata, inplace=True)

# Filter out non-highly expressed genes
adata = adata[:, adata.var.highly_variable]

### Checking data quality ###
print("[INFO] Checking data quality")
# Prepare data for plotting
data = adata.X
print("[INFO] Reducing dimensions...")

# Reduce the datasets with PCA
pcs = PCA(n_components=50).fit_transform(data)
pca = PCA().fit_transform(pcs)
print("[INFO] PCA complete")

# Reduce the datasets with TSNE
tsne = TSNE().fit_transform(pcs)
print("[INFO] t-SNE complete")

# Reduce the datasets with UMAP
umap = UMAP().fit_transform(pcs)
print("[INFO] UMAP complete")

print("[INFO] Data has been reduced")


### Plot the data - cell types ###
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

labels = adata.obs['Macro Cell Type'].values

plot_colors = list(map(color_map.get, labels))

# Plot the samples with cell labels
axis_size = 8.0
plot_ratios = {'height_ratios': [1], 'width_ratios': [1,1,1]}
fig, axs = plt.subplots(1, 3, figsize=(axis_size*3, axis_size), gridspec_kw=plot_ratios, squeeze=True)

# PCA plot
axs[0].scatter(pca[:,0], pca[:,1], c = plot_colors, s = point_size)
axs[0].title.set_text(f"PCA")
axs[0].set_xlabel("PC 1")
axs[0].set_ylabel("PC 2")
box = axs[0].get_position()
axs[0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# t-SNE plot
axs[1].scatter(tsne[:,0], tsne[:,1], c = plot_colors, s = point_size)
axs[1].title.set_text(f"t-SNE")
axs[1].set_xlabel("TSNE 1")
axs[1].set_ylabel("TSNE 2")
box = axs[1].get_position()
axs[1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# UMAP plot
axs[2].scatter(umap[:,0], umap[:,1], c = plot_colors, s = point_size)
axs[2].title.set_text(f"UMAP")
axs[2].set_xlabel("UMAP 1")
axs[2].set_ylabel("UMAP 2")
box = axs[2].get_position()
axs[2].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

fig.legend(handles = patchList, loc = "lower center", ncol = 9, frameon = False, markerscale=2, bbox_to_anchor = [0.5, 0.04])

fig.savefig(fname="GSE114725_10000_3912_plot.png")
plt.clf() 

print("[INFO] Evaluation plot saved")


### Plot the data - patients###
# Create a mapping between classes and colours
color_map = {
    "BC1": "#00798c",
    "BC2": "#C77CFF",
    "BC3": "#edae49",
    "BC4": "#66a182",
    "BC5": "#2e4057",
    "BC7": "#8c8c8c",
    "BC8": "#f37735"}

patchList = []
for key in color_map:
        data_key = plt.scatter([],[], s = point_size*3, marker=".", color = color_map[key], label=key)
        patchList.append(data_key)

labels = adata.obs['patient'].values

plot_colors = list(map(color_map.get, labels))

# Plot the samples with patient labels
axis_size = 8.0
plot_ratios = {'height_ratios': [1], 'width_ratios': [1,1,1]}
fig, axs = plt.subplots(1, 3, figsize=(axis_size*3, axis_size), gridspec_kw=plot_ratios, squeeze=True)

# PCA plot
axs[0].scatter(pca[:,0], pca[:,1], c = plot_colors, s = point_size)
axs[0].title.set_text(f"PCA")
axs[0].set_xlabel("PC 1")
axs[0].set_ylabel("PC 2")
box = axs[0].get_position()
axs[0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# t-SNE plot
axs[1].scatter(tsne[:,0], tsne[:,1], c = plot_colors, s = point_size)
axs[1].title.set_text(f"t-SNE")
axs[1].set_xlabel("TSNE 1")
axs[1].set_ylabel("TSNE 2")
box = axs[1].get_position()
axs[1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# UMAP plot
axs[2].scatter(umap[:,0], umap[:,1], c = plot_colors, s = point_size)
axs[2].title.set_text(f"UMAP")
axs[2].set_xlabel("UMAP 1")
axs[2].set_ylabel("UMAP 2")
box = axs[2].get_position()
axs[2].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

fig.legend(handles = patchList, loc = "lower center", ncol = 7, frameon = False, markerscale=2, bbox_to_anchor = [0.5, 0.04])

fig.savefig(fname="GSE114725_10000_3912_patient_plot.png")
plt.clf() 

print("[INFO] Evaluation plot saved")

# Export data for model creation and evaluation
export_data_values = pd.DataFrame(adata.X, index = adata.obs['cellid'], columns = adata.var['Gene IDs'])
export_data_anno = pd.DataFrame(adata.obs.values[:,1:-1], index = adata.obs.values[:,0], columns = adata.obs.columns.values[1:-1])
export_data_anno.index.set_names("cellid", inplace = True)

# Save count matrix
export_data_values.to_csv("GSE114725\\GSE114725_processed_data_10000_3912.csv")

# Save annotations
export_data_anno.to_csv("GSE114725\\GSE114725_processed_annotations_10000_3912.csv")

# Write results
adata.write(results_file)

print("[INFO] Data successfully exported")


