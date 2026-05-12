# FusStatGRA-MLoc

## Framework Overview

FusStatGRA-MLoc first constructs multiple miRNA similarity networks from multi-source biological information, including sequence information, miRNA–disease associations, miRNA–drug relationships, and miRNA–mRNA interactions. The similarity networks are then integrated through a non-linear fusion strategy, followed by statistical enhancement and graph representation learning for feature extraction. Finally, the extracted features are fused by the gated residual attention mechanism and used for miRNA subcellular localization prediction.


## Datasets
The 'datasets' folder contains the raw data used in FusStatGRA-MLoc. Their specific sources are detailed in the paper. The following is a brief introduction on each file:

- miRNA_ID_1041.txt: The IDs of the 1041 miRNAs used in PMiSLocMF
- miRNA_disease.csv: The association between 1041 miRNAs and 640 diseases
- miRNA_drug.csv: The association between 1041 miRNAs and 130 drugs
- miRNA_mRNA_matrix.txt: The association between 1041 miRNAs and 2836 mRNAs
- miRNA_seq_sim.csv: The sequence similarity among the 1041 miRNAs
- miRNA_func_sim.csv: The functional similarity among the 1041 miRNAs
- miRNA_localization.csv: The subcellular localization of 1041 miRNAs
- mRNA_localization.txt: The subcellular localization of 2836 mRNAs
- miRNA_have_loc_information_index.txt: The indices of the 1041 miRNAs with localization information

## Code

The 'code' folder contains the relevant codes used in FusStatGRA-MLoc. The 'feature_extraction' folder includes the source code for feature extraction, and 'main.py' is the source code for the model.


## Requirements
- python = 3.7.16
- Tensorflow = 2.11.0
- scikit-learn = 1.0.2
- node2vec = 0.4.3
- networkx = 2.6.3

## Quick start

Run code/main.py to Run FusStatGRA-MLoc
