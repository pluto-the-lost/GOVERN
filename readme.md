# Introduction

Graph-based prediction Of VElocity for infering Regulatory Network (GOVERN) is a machine learning method that learning gene regulatory network (GRN) from single cell RNA-seq data. In scRNA-seq data, timestamp of mRNA is embeded in the splicing stage of the transcript, and can be recovered by RNA velocity methods. Based on this time resolved expression shift, graph convolutional network (GCN) can capture complex regulatory relationship between genes. 

# Install

GOVERN prerequest [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric). Users can follow its tutorial to find proper version and install it to local environment.

Than download the source code of GOVERN.

```bash
git clone https://github.com/pluto-the-lost/GOVERN.git
cd GOVERN
```

# Usage

In python kernel

```python
from src import GOVERN

import scanpy as sc
from itertools import product
adata = sc.read('/path/to/h5ad')
# adata should have a velocity layer, which can be 
# generated from scVelo, dynamo, veloAE, etc.

# set edges, here we use transcription factors to all genes
TFs = pd.read_csv('./Transcription Factors.csv')
tf_set = set(TFs.Gene) & set(bdata.var_names)
edges = list(product(tf_set, bdata.var_names))

# build and train model, if you need cell specific GRN, set single_cell_version=True
gov = GOVERN(bdata, edges = edges, v_key = 'velocity')
loss = gov.train(epoch=50, plot_loss_during_training=True, verbose = False)

# get GRN 
grn = gov.get_GRN()
```