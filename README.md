# DS-GNN
This repository implements DS-GNN
Dual-stage GNN pipeline for rs-fMRI: learn subject-level representations from functional connectivity (individual graphs), then perform robust multi-site diagnosis on a population graph built from phenotypic similarity × embedding similarity. Includes interpretability (node masking) and hierarchy-resolved concordance with ENIGMA (spin permutations + top-tail enrichment).

# Highlights

Stage 1 (Individual graph encoder, GNNAEP): subject-level embeddings from FC.

Stage 2 (Population graph classifier): multi-site diagnosis with positive–negative orthogonal loss.

Interpretability: node-masking–based importance per subject.

ENIGMA alignment: stratified spin tests and top-tail enrichment analyses.

Multi-center robustness: phenotype-aware population graph construction.

# Environment

Python 3.7

PyTorch + PyTorch Geometric (match wheels to your CUDA version)

NumPy / SciPy / pandas / scikit-learn

nibabel / nilearn

enigmatoolbox

tensorboardX

Tip: Install PyTorch and PyG from their official instructions matching your CUDA. Then pip install the rest.
