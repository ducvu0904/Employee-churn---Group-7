# Employee Churn: End-to-End Analysis

Explore employee churn and compare traditional machine-learning classifiers. The
target is `left`: **0 = Stayed**, **1 = Left**.

**Start with `Employee_Churn_Analysis.ipynb`.** It combines the dataset overview,
duplicate removal, splitting, EDA, classification, and final evaluation in one
notebook. Its only data input is `data/HR_Dataset.csv`.

## Scope and dataset

The supplied data is `data/HR_Dataset.csv`: 14,999 rows and 10 columns. Its original
source and license have not been documented in this repository; verify and cite
them before publishing or redistributing the data. The current task classifies
recorded churn status. Predicting future departures would additionally require
timestamps, a prediction horizon, and verification that every feature was known
before departure.

The combined notebook removes 3,008 exact duplicate rows, then uses the same
stratified sampling as `split_data.py`, with seed 42, to create these fixed splits:

| Split | Rows | Purpose |
| --- | ---: | --- |
| Training | 8,393 | EDA, five-fold stratified CV, and model fitting |
| Validation | 1,799 | Decision-threshold selection and permutation importance |
| Test | 1,799 | Final comparison of four models and the dummy baseline |

Exact duplicates are treated as repeated records, though employee IDs are not
available to prove identity. Deduplication takes place **before** splitting to
avoid identical records across splits. The test set was inspected in an earlier
version of the project: describe it as a **previously inspected holdout**, not a
fresh independent test or evidence of prospective performance.

## Setup and run order

Tested with Python 3.10.20. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name employee-churn --display-name "Employee churn"
```

1. Open **`Employee_Churn_Analysis.ipynb`** from the project folder and select the
   installed kernel.
2. Choose **Restart Kernel and Run All**. The notebook loads the HR dataset,
   removes duplicates, creates the splits in memory, and runs EDA and modeling.
   You do not need to run `split_data.py` or either separate notebook first.
   Tuning uses two parallel workers and can take several minutes.
3. Read the tables and figures in the notebook. Nine 300-dpi PNG figures are
   saved in `plots/`, including the seven EDA figures, ROC/precision–recall curves
   (`08_classification_roc_pr.png`), and the confusion matrix
   (`09_classification_confusion_matrix.png`). The final comparison table is
   saved to `results/test_metrics.csv`. Rerunning replaces these outputs.

The combined notebook does not read or overwrite `train.csv`, `val.csv`, or
`test.csv`. It keeps the splits in memory and uses a separate EDA copy for the
target-derived `Status` label, so this label never enters the model inputs.

### Original separate workflow (optional)

`EDA.ipynb`, `Classification_Models.ipynb`, and `split_data.py` are preserved as
the original separate workflow. These notebooks use the supplied split CSVs.
If you intentionally need to regenerate those CSVs, run `python split_data.py`
(this overwrites the three split files), then run EDA and classification in order.

## Modeling workflow

- Compare logistic regression, KNN, Random Forest, and Gradient Boosting using
  the **same five stratified training folds**. Select the highest mean CV F1
  for the minority **Left** class; model seeds are fixed, not optimized.
- Each candidate includes its own `ColumnTransformer` and classifier pipeline.
  Salary is ordinal, department is one-hot encoded, and numeric features are
  scaled once for logistic regression/KNN, not for trees. Binary inputs stay 0/1.
  Preprocessing is fitted independently within each CV fold.
- The selected pipeline is refitted on all training rows by `GridSearchCV`.
  Select its probability threshold by validation F1. Start at 0.5 and replace it
  only when F1 improves. If several improved thresholds tie, keep the first one
  tried (in ascending order). Do not refit after threshold selection.
- Explain the fixed model with validation permutation importance at that threshold.
  This is interpretation, not a further feature-selection loop or causal analysis.
- Compare all four CV-tuned models on test using threshold 0.5 and include an
  **always-Stayed** baseline. Report the CV-selected model separately at its
  validation-selected threshold. Show accuracy, precision, recall, F1, ROC-AUC,
  and average precision. Test scores do not choose the model or threshold.
  Show the selected model's confusion matrix at its validation-selected threshold,
  with counts of false positives and false negatives. No subgroup error analysis
  or test-driven model changes are performed.

F1 is the current compromise between precision and recall; no intervention cost
has been supplied. The always-Stayed classifier reaches about 83.4% accuracy but
detects no departures. Average precision summarizes precision–recall ranking; it
is not the same numerical definition as trapezoidal PR-AUC. CV standard deviations
describe fold variation, not confidence intervals. Tuning scores can be optimistic
because they are used to choose hyperparameters.

## Results

The executed workflow selected **Random Forest** (`n_estimators=100`,
`max_depth=None`, `min_samples_split=5`):

| Model | Mean CV F1 | Fold standard deviation |
| --- | ---: | ---: |
| Random Forest | 0.9451 | 0.0112 |
| Gradient Boosting | 0.9436 | 0.0133 |
| KNN | 0.8712 | 0.0103 |
| Logistic Regression | 0.5456 | 0.0066 |

Random Forest and Gradient Boosting are close; the selection rule does not
establish a statistically significant difference. Validation selected threshold
**0.49833333333333335**, increasing validation F1 from **0.9583** to **0.9601**.

| Previously inspected test holdout | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5 | 0.7765 | 0.4130 | 0.8289 | 0.5513 | 0.8453 | 0.3843 |
| KNN | 0.5 | 0.9566 | 0.8374 | 0.9161 | 0.8750 | 0.9532 | 0.8531 |
| Random Forest | 0.5 | 0.9828 | 0.9785 | 0.9161 | 0.9463 | 0.9748 | 0.9611 |
| Gradient Boosting | 0.5 | 0.9844 | 0.9821 | 0.9228 | 0.9516 | 0.9753 | 0.9527 |
| Always Stayed | 0.5 | 0.8344 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1656 |
| Selected Random Forest (validation threshold) | 0.4983 | 0.9839 | 0.9786 | 0.9228 | 0.9499 | 0.9748 | 0.9611 |

Gradient Boosting has a slightly higher test F1, but Random Forest remains the
selected model because selection used training-set CV rather than test scores.

The consistency audit reran the current search grids without changing them.
Gradient Boosting selected 100 estimators, learning rate 0.01, and depth 5.
Earlier saved outputs selected learning rate 0.05, absent from the current grid;
those stale outputs and the exported test CSV have been refreshed. The companion
report and its evidence map are in `../employee_churn_report/` (local sibling
folder, not part of this repository). See `audit/CONSISTENCY_AUDIT.md` there.

Validation permutation importance identifies satisfaction, last evaluation, and
project count as the three largest contributors by mean F1 decrease. These are
predictive associations, not evidence that changing them would prevent churn.

The notebook shows the chosen parameters, CV scores, selected threshold,
validation metrics, and permutation importance. It saves only the final test
summary to `results/test_metrics.csv`: four models at threshold 0.5, the baseline,
and the selected model at its validation-selected threshold. Each row identifies
its decision rule and threshold. It does not write model files, JSON metadata,
checksums, or intermediate CSV tables.

The fitted `best_model` stays in memory for the current kernel session. Rerun the
notebook to fit it again after restarting the kernel.

### Predict within the notebook session

After running the notebook, use its fitted pipeline and selected threshold:

```python
import pandas as pd

employees = pd.read_csv("new_employees.csv")
employees.columns = employees.columns.str.lower().str.strip()
assert employees.columns.is_unique
features = employees[X_train.columns]
assert features.notna().all().all()
probabilities = best_model.predict_proba(features)[:, 1]
predictions = (probabilities >= best_threshold).astype(int)
```

Use the selected threshold: calling `best_model.predict()` alone applies the
default decision rule instead. Inputs must retain the original CSV value types.
Unknown departments are ignored by one-hot encoding; unseen salary values are
encoded as -1, but either case should prompt an input/data-drift check. These are
classification scores, not guaranteed calibrated probabilities. Churn indicators
must not be interpreted as proof of an employee's intentions or used as the sole
basis for consequential employment decisions.
