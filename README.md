# Overview
AMDRP: adaptive drug feature fusion and multihead bidirectional cross-attention network for drug-cancer cell response prediction
## Data
- drug_smiles.csv - SMILES (Simplified molecular input line entry system) of drugs
- ECFP.csv -  ECFP (Extended Connectivity Fingerprints) of drug
- 470cell-734dim-miRNA.csv - MicroRNA expression data of cell lines
- 407cell-69641dim-CpG.csv - DNA methylation data of cell lines
- 461cell-23316dim-copynumber.csv - DNA copy number data of cell lines
- drug_cell.csv - response data between drugs and cell lines
- 388-cell-line-list.csv - Names of 388 cell lines with four cell line characteristics
## Source codes
- preprocess.py: load data and convert to pytorch format
- train.py: train the model and make predictions
- functions.py: some custom functions
- simles2graph.py: convert SMILES sequence to graph
- AE.py: learn low_dimensional representations from high-dimensional cell line features
- AMDRP.py: details of AMDRP model
- model_utils.py:AMDRP model components - AFF and MBCA modules
## Requirements
- Python == 3.7.10
- PyTorch == 1.9.0
- sklearn == 0.24.2
- Numpy == 1.19.2
- Pandas == 1.3.4
