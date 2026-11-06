# From Clustering to Classification

An MSc Applied AI & ML assignment that studies one dataset in two stages. First the labels are hidden and the data is explored with clustering and dimensionality reduction. Then the labels are introduced, a PyTorch neural network is trained on them, and the two approaches are compared.

All the work is in one notebook: [`Assignment3_template.ipynb`](Assignment3_template.ipynb).

## Results

The neural network classified the held-out test set far better than clustering recovered the true classes (K-Means ARI 0.35, NMI 0.42). The network made one error in 54 test samples.

**Neural network, test set (54 samples)**

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| 0.981 | 0.983 | 0.981 | 0.982 | 1.000 |

Precision, recall, F1-score and ROC-AUC are weighted averages. 5-fold cross-validation gave a mean accuracy of 0.977, with folds ranging from 0.943 to 1.000.

**Clustering (k = 3, unscaled features)**

| Algorithm | Davies-Bouldin (lower is better) | Calinski-Harabasz (higher is better) | ARI | NMI |
|---|---|---|---|---|
| K-Means | 0.550 | 497.0 | 0.352 | 0.423 |
| Agglomerative | 0.536 | 552.9 | not printed | not printed |

Agglomerative clustering scores slightly better on both internal metrics. Its ARI and NMI are calculated in Step 8, but the notebook doesn't print them.

The clustering results are held back by running on unscaled features, so large-valued features such as `proline` dominate the distance calculations. The classifier uses standardised features and has access to the labels, so this comparison is not like for like.

## Dataset

The dataset has 178 samples, 13 numeric chemical attributes and 3 classes, with no missing values. The classes are split 40% / 33% / 27%, so they are mildly imbalanced.

The assignment presents this as a "mystery dataset". Its column names, size and class proportions are consistent with the UCI Wine dataset.

**The data files are not included in this repository.** To run the notebook, place these two files next to it:

| File | Contents |
|---|---|
| `assignment3_data.csv` | The 13 features, used for clustering |
| `assignment3_data_with_labels.csv` | The same features plus a `true_label` column |

## What the notebook does

| Part | Steps | Work |
|---|---|---|
| 1. Unsupervised learning | 1–6 | Exploratory analysis and Z-score outlier checks; K-Means for k = 2 to 10 (elbow and silhouette); K-Means and Agglomerative clustering at k = 3; PCA and t-SNE plots; internal validation metrics |
| 2. Move to supervised learning | 7–10 | Comparison of clusters with true labels (ARI, NMI, confusion matrices); feature scaling; class balance check; 70/30 train-test split |
| 3. Neural network | 11–13 | Network built, trained with early stopping and evaluated on the test set |
| 4. Tuning and cross-validation | 14–15 | Retraining with a fixed configuration; 5-fold cross-validation |
| 5. Comparison | 16–17 | Summary table of clustering and classification metrics; loss curves and confusion matrix |

**Choosing k:** the silhouette score is highest at k = 2 (0.657), and k = 3 and k = 4 are close (0.560 and 0.567). The notebook uses k = 3, which matches the three classes stated in the assignment brief.

**Network:** 13 inputs → 128 → 64 → 32 → 3 outputs, with ReLU activation and 0.5 dropout after each hidden layer. It is trained with Adam (learning rate 0.001), cross-entropy loss and a batch size of 32, for up to 50 epochs with an early-stopping patience of 5.

## Running it

The notebook was last run on Python 3.12.7.

```bash
pip install jupyter numpy pandas scipy scikit-learn matplotlib seaborn torch
jupyter notebook Assignment3_template.ipynb
```

**Before using Run All:** add `import torch.nn.functional as F` to the imports in the Step 11 cell. Step 13 uses `F`, but it is only imported later, in Step 16, so a fresh run stops at Step 13 with a `NameError`. The saved outputs exist because the cells were run out of order.

## Known limitations

These affect how far the results can be trusted. They are listed here rather than corrected, so the code still matches the saved outputs.

- **Validation uses the training data.** In Steps 12 and 14, "validation loss" is calculated on the training set, so early stopping never sees unseen data. The Step 17 discussion concludes there is no overfitting, but these curves can't show that. The test set and cross-validation results are better evidence.
- **The best weights aren't really restored.** `best_model_state = model.state_dict()` stores a reference to the live weights, not a copy, so restoring it keeps the final epoch's weights. This made no difference in Step 12, where the best epoch was the last one. Step 14 doesn't log enough to tell. The fix is to save `copy.deepcopy(model.state_dict())`.
- **Scaling happens before the split.** `StandardScaler` is fitted on all the data, so a little information from the test set and validation folds leaks into training.
- **The train-test split isn't stratified.** Step 10 asks for the label distribution to be maintained where applicable, but `train_test_split` is called without `stratify=y`.
- **Step 13's ROC-AUC is not computed correctly.** It passes predicted labels rather than probabilities to `roc_auc_score`, which is not valid for multi-class one-vs-rest scoring. `step13_roc_auc` is therefore expected to be `None`. Step 16 passes probabilities, and that is the source of the 1.000 reported above.
- **The test set is small.** With only 54 samples, a single prediction moves accuracy by nearly two percentage points.
