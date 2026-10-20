# Machine Learning Capstone Project — 23CSE301

## Overview

This repository contains our Machine Learning Capstone work for **23CSE301 — Machine Learning**. It covers three tracks:

- **Regression:** predicting a continuous target.
- **Classification:** predicting the binary diabetes indicator.
- **Clustering:** exploring structure in health-indicator data without using labels to fit clusters.

The notebooks include data inspection, preprocessing, model training, evaluation, visualisation, and model comparisons. Results below are transcribed from the saved notebook outputs supplied with the project; where the notebook does not provide a final value, this README does not invent one.

## Repository contents

```text
.
├── regression.ipynb
├── classification.ipynb
├── clustering.ipynb
├── regression1-checkpoint.ipynb   # checkpoint; not the primary regression notebook
├── data/                          # place the required datasets here, if not already provided
├── models/                        # optional saved model files used by the regression notebook
├── requirements.txt               # create/maintain for the team's environment
└── README.md
```

Update filenames and folders if the Git repository uses a different structure. The clustering notebook was supplied from the `ram` branch.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

Install the common dependencies:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

If the team has a maintained `requirements.txt`, prefer `pip install -r requirements.txt`.

## Running the notebooks

1. Clone the repository and move into its root directory.
2. Place the required dataset files in the paths expected by the notebooks.
3. Install the dependencies.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open each notebook and run cells from top to bottom.

The regression notebook uses helper functions and may load saved models from `../models/`; ensure the relevant helper code, dataset paths, and model directory are present. The classification notebook uses the CDC BRFSS 2015 diabetes health indicators CSV. The clustering notebook uses the BRFSS 2021 diabetes health indicators CSV.

---

# 1. Regression

## Objective and workflow

The regression track compares algorithms for predicting a continuous target. The notebook explores the data, preprocesses features, uses a train/test split, evaluates models, and visualises actual-versus-predicted values and model diagnostics. It also records cross-validation results for selected tree-based models.

### Algorithms

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors (KNN) Regressor

## Test-set results

Metrics are from the saved outputs in `regression.ipynb`. Higher R² is better; lower RMSE and MAE are better.

| Algorithm | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|
| Random Forest Regressor | **0.6028** | **63.0467** | **29.9309** |
| KNN Regressor | 0.5607 | 66.3012 | 30.7877 |
| Gradient Boosting Regressor | 0.5133 | 69.7882 | 35.7730 |
| Polynomial Regression (degree 2) | 0.3122 | 82.9611 | 48.8972 |
| Decision Tree Regressor | 0.3004 | 83.6738 | 39.5808 |
| SVR | 0.2127 | 88.7608 | 36.5524 |
| ElasticNet Regression | 0.1970 | 89.6447 | 52.0233 |
| Lasso Regression | 0.1969 | 89.6449 | 52.0293 |
| Ridge Regression | 0.1969 | 89.6457 | 52.0317 |
| Linear Regression | 0.1969 | 89.6475 | 52.0410 |

### Best-performing regression model

**Random Forest Regressor** achieved the highest recorded test R² (**0.6028**) and lowest recorded test RMSE (**63.0467**) among the listed models.

Recorded Random Forest settings:

```text
n_estimators = 250
min_samples_split = 2
min_samples_leaf = 1
max_features = "sqrt"
max_depth = 20
```

- Best cross-validation R² recorded in the notebook: **0.5415**
- Test R²: **0.6028**
- Test RMSE: **63.0467**
- Test MAE: **29.9309**

The notebook also reports KNN cross-validation R² of **0.4984** and Decision Tree cross-validation R² of **0.2943**. The KNN training metrics are perfect while its test metrics are lower, which suggests overfitting; use test performance when comparing generalisation.

### Regression visualisations

The notebook includes alpha-versus-error plots for regularised linear models, polynomial-degree comparisons, actual-versus-predicted scatter plots, and tree-model diagnostics/feature importance.

---

# 2. Classification

## Objective and dataset

The classification notebook predicts the binary target `Diabetes_binary` from the **CDC BRFSS 2015 Diabetes Health Indicators** dataset.

- Target: `Diabetes_binary`
- `0`: No diabetes
- `1`: Diabetes
- Original dataset shape: **253,680 rows × 22 columns**
- Duplicate rows reported: **24,206**
- Shape after duplicate removal and feature engineering: **229,474 rows × 23 columns**
- Model features: **22**, including the engineered `BMI_Age_Interaction`
- Train/test split: **80:20**, stratified, with `random_state=42`
- Train shape: **183,579 × 22**
- Test shape: **45,895 × 22**
- Target distribution after cleaning: **84.71% class 0**, **15.29% class 1**
- Missing values reported: **0**

The notebook scales features for models that require scaling, fitting the scaler on training data and transforming the test data.

## Algorithms

The notebook covers ten classification algorithms across Part A and Part B:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Classifier (SVC)
6. Random Forest Classifier
7. AdaBoost Classifier
8. Gradient Boosting Classifier
9. Bagging Classifier with a Decision Tree base estimator
10. MLP Classifier (Neural Network)

## Evaluation

The project evaluates classification using accuracy, precision, recall, weighted F1-score, confusion matrices, and ROC-AUC. Because the target is imbalanced, accuracy alone does not describe positive-class performance; class-1 precision and recall should be reviewed as well.

### Part B: baseline and tuned results

Precision and recall in this table refer to **Class 1**. F1 is the **weighted F1-score**.

| Algorithm | Version | Accuracy | Precision (Class 1) | Recall (Class 1) | Weighted F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Gradient Boosting | Baseline | 0.8549 | 0.5950 | 0.1601 | 0.8176 | 0.8162 |
| Gradient Boosting | Tuned | 0.8548 | 0.5885 | 0.1691 | **0.8191** | 0.8160 |
| Bagging (Decision Tree) | Baseline | 0.8523 | 0.5646 | 0.1500 | 0.8141 | 0.8034 |
| Bagging (Decision Tree) | Tuned | 0.8513 | 0.5475 | 0.1610 | 0.8152 | 0.7993 |
| MLP Neural Network | Baseline | 0.8514 | 0.5609 | 0.1292 | 0.8097 | 0.8097 |
| MLP Neural Network | Tuned | 0.8514 | 0.5466 | 0.1664 | 0.8162 | 0.8100 |

### Tuned parameters

| Algorithm | Selected parameters | CV weighted F1 |
|---|---|---:|
| Gradient Boosting | `n_estimators=150`, `learning_rate=0.1`, `max_depth=3` | 0.8144 |
| Bagging | `n_estimators=30`, `max_samples=0.7`, base tree `max_depth=12`, `min_samples_leaf=5` | 0.8122 |
| MLP | `hidden_layer_sizes=(64,)`, `activation='tanh'` | 0.8099 |

### Other saved classification results

The notebook also records these tuned results:

| Algorithm | Key selected parameters | Accuracy | Weighted F1 | ROC-AUC |
|---|---|---:|---:|---:|
| Decision Tree | `max_depth=10` | 0.8514 | 0.8113 | 0.7979 |
| KNN | `n_neighbors=7`, `metric='euclidean'` | 0.8384 | 0.8093 | 0.7270 |
| SVC | `C=10`, `kernel='rbf'` | 0.8379 | 0.8091 | 0.6985 |

For Decision Tree, KNN, and SVC, the values above are taken from the notebook's final output cells. The README does not claim that these three results represent every Part A model's final tuned metrics; consult the notebook for the remaining algorithms.

### Classification observations

- Gradient Boosting has the highest recorded tuned weighted F1 among the three Part B models: **0.8191**.
- MLP weighted F1 increased from **0.8097 to 0.8162** after tuning.
- Bagging weighted F1 increased from **0.8141 to 0.8152**, while its ROC-AUC decreased.
- Positive-class recall remains low across the reported models. The confusion matrices show many false negatives, so the models miss a substantial number of actual diabetes cases.
- The data imbalance (84.71% class 0 vs. 15.29% class 1) is an important limitation. Future work could evaluate class weighting, resampling, and decision-threshold selection while monitoring precision, recall, weighted F1, and ROC-AUC.

### Classification visualisations

The notebook includes confusion matrices, classification reports, ROC-AUC evaluation, model comparison tables, hyperparameter-tuning outputs, and baseline-versus-tuned plots.

---

# 3. Clustering

## Objective and dataset

The clustering notebook explores groups in the **BRFSS 2021 Diabetes Health Indicators** dataset.

- Dataset: `diabetes_012_health_indicators_BRFSS2021.csv`
- Original shape: **236,378 rows × 22 columns**
- Target column in source data: `Diabetes_012`
- Input features considered: **21**
- Missing values reported: **0**
- Train/test split reported by the notebook: training **178,840 × 21**, testing **44,710 × 21**
- Clustering metric comparison uses a sample of **20,000 observations**
- Number of clusters evaluated: **3**

Clustering is unsupervised: target labels should not be used to fit cluster assignments. If labels are used to interpret clusters afterward, make that distinction clear in the analysis.

## Algorithms and visualisations

The notebook evaluates:

- K-Means clustering, including an Elbow curve
- Agglomerative hierarchical clustering with single, complete, average, and Ward linkage
- PCA projections for cluster visualisation
- t-SNE visualisation for K-Means
- Cluster-size and clustering-metric comparisons
- Hierarchical dendrograms

## Agglomerative clustering results

| Linkage | Clusters | Silhouette Score | Davies–Bouldin Index | Calinski–Harabasz Index |
|---|---:|---:|---:|---:|
| Single | 3 | 0.4862 | 0.3635 | 7.2979 |
| Complete | 3 | 0.2212 | 2.0975 | 586.5302 |
| Average | 3 | **0.4978** | 0.5850 | 10.1434 |
| Ward | 3 | 0.1632 | 2.6409 | **1535.5528** |

Metric interpretation:

- **Silhouette Score:** higher is generally better.
- **Davies–Bouldin Index:** lower is generally better.
- **Calinski–Harabasz Index:** higher is generally better.

### Important clustering limitation

The recorded cluster sizes show that single linkage produced sizes **[19,998, 1, 1]** and average linkage produced **[19,997, 2, 1]**. Their high Silhouette Scores therefore need to be interpreted cautiously: they largely isolate tiny clusters rather than creating balanced, useful groups. Ward linkage produced cluster sizes **[4,523, 13,732, 1,745]**, a more substantial partition, although its Silhouette Score was lower.

The saved notebook output available for this README did not expose a final K-Means metric row. Therefore, K-Means metrics are not listed here rather than being estimated. The notebook's K-Means Elbow curve and visualisations should be consulted for the selected cluster count and interpretation.

---

# Reproducibility notes

- Run notebooks from top to bottom in a clean environment.
- Check dataset paths before running.
- Ensure required helper functions and saved model files are available where referenced.
- Preserve the notebook's random seeds and split strategy when reproducing reported metrics.
- Fit preprocessing transformations on training data only for supervised train/test evaluation.
- Do not compare models using different test splits without noting the difference.
- Keep generated outputs and figures consistent with the final code before submission.

## References

- Course-provided **23CSE301 Machine Learning Capstone Project Guidelines**.
- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [NumPy documentation](https://numpy.org/doc/)
- [Matplotlib documentation](https://matplotlib.org/stable/)
- [Seaborn documentation](https://seaborn.pydata.org/)

## Final note

This README summarises results available in the uploaded notebooks.
