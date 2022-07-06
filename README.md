# SeqAugGra
Molecular properties prediction is a fundamental problem in drug design and discovery pipelines. Considering the outstanding results achieved by the existing SMILES string or molecular graph molecular representation learning algorithms, little work has been done to integrate the advantages of the two methods to conserve molecular properties for further improvement. Moreover, owing to a lack of labeled data, deep learning-based molecular representation models can only
explore a narrow chemical space and have weak generalizability.In this paper, we propose SeqAugGra, a representation learning model for GNN series variants(e.g., GCN, GIN, DeepGCN,and AttentiveFP) that integrates sequential features of molecules with augmented graph features to predict molecular properties. Specically, SeqAugGra mod-
els two types of chemical inputs by combining sequence models and graph neural networks in a complimentary manner.In contrast to the best baseline approaches, the GNNs variant model series, the best SeqAugGra, may boost ROC-AUC scores by +4.59%, +3.73%, and +3.15%, respectively, for the BACE, Tox21, and ToxCast datasets.While
compared to the original GIN model, the ROC-AUC scores of the augmented GIN model improved by +18.66%, +13.62%, +10.74%,+4.69%, and +5.77% on BACE,Tox21,ToxCast,LogP, and FDA, separately. The aforementioned experimental results demonstrated that our proposed SeqAugGra model outperforms state-of-the-art techniques.
# Installation
1) reference the https://github.com/BaratiLab/AugLiChem
2)reference https://github.com/samoturk/mol2vec
