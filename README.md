# IMHGN-CVD Reproducibility Repository

This repository provides a compact, executable implementation of the **Improved Multimodal Health Graph Network (IMHGN)** for binary cardiovascular disease (CVD/non-CVD) prediction.

The implementation is aligned with the final tabular-GCN methodology described in the revised manuscript:

- modalities: **clinical, demographic, and lifestyle**
- patient-level weighted graph
- cosine-similarity edge weights
- top-`k` nearest-neighbor graph construction (`k = 5` by default)
- no manually chosen similarity threshold
- two GCN layers in the reported final configuration
- adaptive multimodal fusion
- Adaptive Bat Optimization (ABO) for hyperparameter selection
- nested 5×5 cross-validation
- paired Wilcoxon signed-rank testing
- 1,000-sample bootstrap confidence intervals
- Brier score and calibration curves
- ablation and explainability routines
- leakage-safe preprocessing fitted only on training partitions

The repository intentionally contains no SRGAN, Vision Transformer, genetic-data modality, Adaptive Butterfly Optimization, or three-layer final GCN implementation.

## Repository layout

```text
IMHGN-CVD-Reproducibility/
├── README.md
├── requirements.txt
├── config.yaml
├── run_experiment.py
├── DATA.md
├── REPRODUCIBILITY.md
├── CODE_AVAILABILITY.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── graph.py
│   ├── model.py
│   ├── optimization.py
│   ├── train.py
│   ├── evaluation.py
│   └── explainability.py
└── tests/
    ├── test_preprocessing.py
    ├── test_graph.py
    ├── test_model.py
    └── test_end_to_end.py
```

## 1. Environment

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

## 2. Dataset preparation

The code does not redistribute third-party cardiovascular datasets. Place locally obtained CSV files in a directory of choice, then update `config.yaml`.

Example:

```yaml
data:
  df1_path: "./data/df1.csv"
  df2_path: "./data/df2.csv"
```

The expected target is binary:

- `0`: non-CVD / normal
- `1`: CVD / heart disease

Column aliases and modality feature lists can be configured in `config.yaml`.

See `DATA.md` for the expected variables, source notes, harmonization rules, and leakage-control requirements.

## 3. Run the full experiment

```bash
python run_experiment.py --config config.yaml
```

The main pipeline performs:

1. dataset loading and schema harmonization;
2. stratified outer cross-validation;
3. training-only preprocessing;
4. MICE-style iterative imputation;
5. Min-Max normalization;
6. optional modality-preserving RFE and PCA;
7. training-only SMOTE;
8. cosine k-NN graph construction;
9. inner-CV ABO hyperparameter search;
10. IMHGN training;
11. outer-fold evaluation;
12. calibration statistics;
13. bootstrap confidence intervals;
14. baseline comparison and Wilcoxon testing;
15. ablation experiments.

Results are written to the output directory specified in `config.yaml`.

## 4. Graph definition

Each patient is represented by one node. For a transformed patient feature vector \(x_i\), pairwise cosine similarity is

\[
s_{ij} = \frac{x_i^\top x_j}{\|x_i\|_2\|x_j\|_2}.
\]

For every node, the top `k` most similar *other* patients are selected.

Default graph policy:

```text
metric                  cosine similarity
k                       5
manual threshold        none
edge weight             cosine similarity
graph                    undirected
symmetrization          union of directed top-k relations
self loops              handled by GCNConv
```

No manually selected similarity threshold is used. An edge is created because a patient is in the top-`k` neighbor set, not because it exceeds a fixed similarity cut-off.

## 5. IMHGN architecture

The final reported configuration is:

```text
clinical block ---------\
demographic block ------- > learnable modality weights → fused patient representation
lifestyle block --------/

fused representation
    ↓
GCNConv(input_dim, 64)
ReLU
Dropout(0.2)
    ↓
GCNConv(64, 32)
ReLU
Dropout(0.2)
    ↓
Linear(32, 1)
Sigmoid probability at evaluation
```

The fusion weights are learned with a softmax-constrained parameter vector. They therefore sum to one and quantify the relative contribution assigned to each available modality during training.

## 6. Adaptive Bat Optimization

ABO searches a bounded mixed hyperparameter space. The optimizer supports continuous, integer, and categorical variables through explicit search-space definitions.

The default search includes:

- learning rate
- weight decay
- hidden dimension of GCN layer 1
- hidden dimension of GCN layer 2
- dropout
- training epochs

The number of GCN layers remains fixed at two.

The objective is the mean inner-fold validation ROC-AUC. If ROC-AUC cannot be evaluated because only one class is present in a small inner validation partition, balanced accuracy is used for that fold.

## 7. Statistical validation

Outer-fold scores are treated as paired observations.

The implementation provides:

- paired two-sided Wilcoxon signed-rank test;
- 95% bootstrap confidence intervals with 1,000 resamples by default;
- accuracy, precision, recall, F1, ROC-AUC, Cohen's kappa;
- Brier score;
- confusion matrix;
- calibration curve coordinates.

No p-value or reported performance value is hard-coded.

## 8. Ablations

The executable ablation variants are:

```text
full                     complete IMHGN
no_graph                 identity-edge graph (no inter-patient message passing)
no_fusion                fixed equal modality weighting
no_abo                   fixed manuscript configuration without ABO
euclidean_graph          Euclidean k-NN graph
k3                       cosine graph with k=3
k5                       cosine graph with k=5
k7                       cosine graph with k=7
```

Accuracy differences are calculated directly from generated results to avoid narrative/table arithmetic discrepancies.

## 9. Reproducibility controls

The implementation sets random seeds for:

- Python
- NumPy
- PyTorch CPU
- PyTorch CUDA

The default seed is `42`.

Preprocessing transformations are fitted inside the relevant training fold only. SMOTE is applied only to training data. Validation and test data are never used to fit imputers, scalers, feature selectors, PCA transforms, or oversampling procedures.

## 10. Tests

Run:

```bash
pytest -q
```

The test suite checks:

- iterative imputation and scaling;
- training-only SMOTE behavior;
- graph symmetry and top-k construction;
- two-layer model structure;
- fusion-weight normalization;
- valid output shapes and probabilities;
- end-to-end execution on a synthetic software-test fixture.

Synthetic data in the tests are used only to validate software behavior. They are not experimental evidence and are not used to reproduce manuscript performance.

## 11. Reproducibility statement

Exact numerical reproduction requires the same original datasets, feature harmonization, class labels, and experimental conditions described in the manuscript. This repository computes results from data supplied by the researcher and does not embed manuscript accuracy values as constants.

See `REPRODUCIBILITY.md` for the full protocol and `CODE_AVAILABILITY.md` for the manuscript-ready availability statement.
