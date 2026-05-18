# Early Sepsis Prediction in Intensive Care Units

This repository contains the source code and experimental notebooks for an
early-warning machine learning pipeline that predicts sepsis risk in ICU
patients using hourly clinical time-series data.

The project uses the PhysioNet/CinC 2019 sepsis dataset, performs
missingness-aware feature engineering, prepares fixed-length LSTM sequences,
and evaluates single BiLSTM and patient-subset ensemble models.

## Project Objectives

- Predict sepsis early from ICU time-series measurements.
- Preserve clinically meaningful temporal patterns and missingness signals.
- Evaluate performance using metrics suitable for rare-event prediction,
  especially AUROC, AUPRC, sensitivity, specificity, and alert burden.
- Maintain a transparent GitHub history and reproducible repository structure.

## Repository Structure

```text
.
|-- src/
|   |-- __init__.py
|   |-- model_utils.py
|   `-- notebooks/
|       |-- data-processing.ipynb
|       |-- descriptive-analysis.ipynb
|       |-- lstm-preptation.ipynb
|       |-- lstm-model.ipynb
|       `-- lstm-model-ensemble.ipynb
|-- references/
|   `-- .gitkeep
|-- README.md
`-- requirements.txt
```

Main files:

- `src/model_utils.py`: shared model, callback, threshold-search, and
  evaluation utilities.
- `src/notebooks/*.ipynb`: executable notebook source code for preprocessing,
  analysis, training, and evaluation.
- `src/notebooks/data-processing.ipynb`: loads raw `.psv` patient files, adds patient/hour
  identifiers, performs base cleaning, and exports `stage1_base_clean.csv`.
- `src/notebooks/descriptive-analysis.ipynb`: explores missingness, onset-centered vital-sign
  trajectories, and sepsis/non-sepsis reference patterns.
- `src/notebooks/lstm-preptation.ipynb`: builds temporal difference, last-gap, qSOFA, and
  SOFA-inspired features; splits by patient; scales features; and creates LSTM
  sequences.
- `src/notebooks/lstm-model.ipynb`: trains and evaluates a single BiLSTM model.
- `src/notebooks/lstm-model-ensemble.ipynb`: trains the patient-subset ensemble, averages
  member probabilities, selects the validation threshold, and performs final
  test/error analysis.
- `references/`: place papers, dataset documentation, and thesis references
  here.

## Environment Setup

Use Python 3.10 or newer. The notebooks were developed in Kaggle with GPU
support, but the environment can also be created locally.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

## Dataset And Kaggle Artifacts

The original dataset is not committed to this repository because it is large.
Download it from the PhysioNet/CinC Challenge 2019 source or attach the
corresponding Kaggle dataset when running on Kaggle.

Dataset sources:

- Raw data used in this project: https://www.kaggle.com/datasets/thuhiuhong/training-sepsis-dataset/data
- Base-clean hourly table: https://www.kaggle.com/datasets/thuhiuhong/base-clean
- Train/validation/test LSTM sequence arrays: https://www.kaggle.com/datasets/thuhiuhong/traintestval-lstm
- Original PhysioNet/CinC 2019 Challenge resource: https://physionet.org/content/challenge-2019/1.0.0/

Large processed arrays and trained model outputs are not committed to this
repository. When running on Kaggle, attach the dataset above or update the paths
in each notebook.

The shared helper module is included in this repository as `src/model_utils.py`.
When running locally, import it directly from the repository. When running on
Kaggle, either upload the repository code with the notebook or attach a small
Kaggle dataset containing `model_utils.py` if the notebook cannot access the
GitHub source directly.

Expected raw input format:

- One `.psv` file per patient.
- Hourly ICU measurements separated by `|`.
- A `SepsisLabel` column indicating sepsis status at each hour.

The first notebook expects a raw dataset directory similar to:

```text
training-sepsis-dataset/
|-- training_setA/
|   |-- p000001.psv
|   `-- ...
`-- training_setB/
    |-- p100001.psv
    `-- ...
```

In `src/notebooks/data-processing.ipynb`, update `DATA_DIR` to the local or Kaggle dataset
path before running.

### Kaggle Path Configuration

Common Kaggle paths used by the notebooks:

```python
# Raw PhysioNet/CinC 2019 .psv files
DATA_DIR = "/kaggle/input/training-sepsis-dataset"

# Base-clean hourly table exported by data-processing.ipynb
BASE_CLEAN_PATH = "/kaggle/input/datasets/thuhiuhong/base-clean/stage1_base_clean.csv"

# Preprocessed sequence arrays exported by lstm-preptation.ipynb,
# then uploaded/attached as a Kaggle dataset for model notebooks
INPUT_PATH = "/kaggle/input/datasets/thuhiuhong/traintestval-lstm"

# Optional: only needed if model_utils.py is attached as a separate Kaggle dataset
MODULE_PATH = "/kaggle/input/lstm-utils"

```
### Notebook Data Flow

| Step | Notebook | Main input on Kaggle | Main output |
| --- | --- | --- | --- |
| 1 | `src/notebooks/data-processing.ipynb` | `/kaggle/input/training-sepsis-dataset` | `stage1_base_clean.csv` |
| 2 | `src/notebooks/descriptive-analysis.ipynb` | `/kaggle/input/datasets/thuhiuhong/base-clean/stage1_base_clean.csv` | Exploratory plots and missingness summaries |
| 3 | `src/notebooks/lstm-preptation.ipynb` | `/kaggle/input/datasets/thuhiuhong/base-clean/stage1_base_clean.csv` | `X_train.npy`, `y_train.npy`, `id_train.npy`, `X_val.npy`, `y_val.npy`, `id_val.npy`, `X_test.npy`, `y_test.npy`, `id_test.npy`, `meta_train.csv`, `meta_val.csv`, `meta_test.csv` |
| 4 | `src/notebooks/lstm-model.ipynb` | `/kaggle/input/datasets/thuhiuhong/traintestval-lstm` and `src/model_utils.py` or optional `lstm-utils` | Single-model checkpoint and test predictions |
| 5 | `src/notebooks/lstm-model-ensemble.ipynb` | `/kaggle/input/datasets/thuhiuhong/traintestval-lstm` and `src/model_utils.py` or optional `lstm-utils` | Ensemble predictions, PR/ROC/confusion-matrix figures, subgroup tables, missingness error table |

### Reusing Outputs Between Kaggle Notebooks

Kaggle notebooks write generated files to `/kaggle/working`. To use an output
in the next notebook:

1. Run the producing notebook.
2. Save the notebook output as a Kaggle Dataset.
3. Add that Kaggle Dataset to the next notebook.
4. Confirm the new `/kaggle/input/...` path matches the path used in the
   notebook.

For example, `lstm-preptation.ipynb` exports sequence arrays to
`/kaggle/working`. These files are then attached to the model notebooks through
the `traintestval-lstm` Kaggle dataset path.

When running from this GitHub repository locally, imports use:

```python
from src.model_utils import (
    create_bilstm,
    get_callbacks,
    find_best_threshold,
    full_evaluation,
)
```

## How to Run

### Option A: Run on Kaggle

This is the recommended option because the dataset and LSTM training workload
are large.

1. Open Kaggle and create or open the notebook.
2. Add the required Kaggle datasets using the notebook sidebar.
3. Check that the paths in each notebook match the attached dataset names under
   `/kaggle/input/...`.
4. Enable GPU for model training notebooks.
5. Run the notebooks in the order below.

If Kaggle cannot import `src.model_utils` directly from the GitHub repository,
upload the full `src/` folder as a small Kaggle dataset. Another simple option
is to upload a copy of `src/model_utils.py` as `model_utils.py`, then append
that Kaggle dataset path to `sys.path` before importing.

Example:

```python
import sys

MODULE_PATH = "/kaggle/input/your-code-dataset/src"
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from model_utils import (
    create_bilstm,
    get_callbacks,
    find_best_threshold,
    full_evaluation,
)
```

### Option B: Run Locally

Local execution is useful for reviewing code and small tests, but full training
may be slow without a GPU.

```bash
jupyter notebook
```

Run the notebooks in this order:

1. `src/notebooks/data-processing.ipynb`
   - Loads all patient `.psv` files.
   - Adds `id` and `hour`.
   - Exports `stage1_base_clean.csv`.

2. `src/notebooks/descriptive-analysis.ipynb`
   - Reads the base-clean dataset.
   - Produces exploratory missingness and temporal trajectory analysis.

3. `src/notebooks/lstm-preptation.ipynb`
   - Builds engineered features.
   - Splits train/validation/test by patient.
   - Fits the scaler on the training split only.
   - Creates sequence arrays such as `X_train.npy`, `y_train.npy`,
     `X_val.npy`, `y_val.npy`, `X_test.npy`, and `y_test.npy`.

4. `src/notebooks/lstm-model.ipynb`
   - Trains a single BiLSTM baseline.
   - Selects the threshold on validation data.
   - Evaluates the model on the test set.

5. `src/notebooks/lstm-model-ensemble.ipynb`
   - Runs compact hyperparameter search.
   - Trains the patient-subset ensemble.
   - Computes weighted averaged probabilities.
   - Reports final metrics, confusion matrix, PR/ROC curves, subgroup
     performance, and alert burden.

## Main Modeling Design

- Input window length: 10 hourly timesteps.
- Final feature count: 133 features.
- Feature groups include raw clinical variables, temporal differences,
  measurement recency (`last_gap`), qSOFA indicators, and SOFA-inspired
  rolling features.
- Patient-level train/validation/test split prevents patient leakage.
- Positive windows use denser sampling than negative windows to reduce class
  imbalance while preserving sepsis trajectories.
- The ensemble keeps all sepsis patients in each member and samples non-sepsis
  patients per member.
- The decision threshold is selected on validation data with a sensitivity
  target of at least 0.80.

## Reported Final Ensemble Result

On the dense test sequence set:

- AUROC: 0.8482
- AUPRC: 0.1162
- Sensitivity: 0.8314
- Specificity: 0.7093
- Precision: 0.0482
- Decision threshold: 0.43

These values are from the executed ensemble notebook and should be regenerated
when the data split, preprocessing, or model configuration changes.

## Demo and Deployment Notes

This repository is research-oriented and notebook-based. The demonstration is performed by running the notebooks and inspecting the exported metrics, plots, and prediction tables. It does not include a production web service or clinical deployment package.

For a simple demonstration:

1. Run the preprocessing and sequence preparation notebooks.
2. Run `src/notebooks/lstm-model-ensemble.ipynb`.
3. Use the exported prediction tables and plots from the notebook to show
   predicted sepsis probabilities, selected threshold, confusion matrix, PR/ROC
   curves, and alert-burden analysis.

## Source Control Policy

The repository is maintained through regular commits that record milestones such as preprocessing, feature engineering, model training, evaluation, documentation, and report updates.
