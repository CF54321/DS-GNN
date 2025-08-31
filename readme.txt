This repository implements DS-GNN: it first learns subject-level representations of rs-fMRI functional connectivity (individual graphs), then performs robust multi-site diagnosis on a population graph constructed from phenotypic similarity × embedding similarity. It also provides interpretability (node masking) and hierarchy-resolved concordance with ENIGMA (spin permutations + top-tail enrichment).

Directory

data/GNN_dataset.py: converts per-subject FC matrices + labels into PyG Data objects (fMRI_data*.pt).

main/Train_Val_LJX_sex_disease.py: Stage 1 — individual-graph encoder (GNNAEP).

main/Train_Val_node_graph_zhengjiao.py: Stage 2 — population-graph classifier with positive–negative orthogonal loss.

Explainer/Explainer.py: node-masking–based importance analysis.

enigma/transfer.py: map AAL weights → DK+ASEG by voxel overlap.

enigma/analyze.py: correlation/enrichment and stratified spin tests against ENIGMA.

model/: network architectures; tools/: utilities and logging.

Environment

Python 3.7; PyTorch + PyG; NumPy/SciPy/pandas/scikit-learn; nibabel/nilearn; enigmatoolbox; tensorboardX.
Install PyTorch/PyG wheels that match your CUDA version.

Training pipeline

Run data/GNN_dataset.py to generate *.pt under graph/.

Stage 1

python main/Train_Val_LJX_sex_disease.py --root /path/to/graph ...


Stage 2 (requires phenotype CSV and Stage-1 checkpoints)

python main/Train_Val_node_graph_zhengjiao.py --root /path/to/graph ...


Interpretability

python Explainer/Explainer.py   # outputs a 116-dim weight vector per subject


ENIGMA analysis

If needed, map AAL → DK+ASEG with enigma/transfer.py, then

run enigma/analyze.py for spin tests and enrichment.

If a script does not provide command-line arguments, open the file and modify the paths directly.