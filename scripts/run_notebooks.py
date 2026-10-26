# run_notebooks.py
# Runs every project notebook in order and saves the outputs (tables and charts) inside them.
# Usage (from the project folder):  python scripts/run_notebooks.py

import os
import sys
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "notebooks/01.EDA/01_data_understanding.ipynb",
    "notebooks/01.EDA/02_patterns_and_target.ipynb",
    "notebooks/02.FeatureEngineering/01_cleaning_and_features.ipynb",
    "notebooks/03.ModelTraining/01_baselines_and_xgboost.ipynb",
    "notebooks/04.ModelEvaluation/01_model_comparison.ipynb",
    "notebooks/04.ModelEvaluation/02_feature_importance.ipynb",
]

# Run only the notebooks named on the command line, or all of them
selected = sys.argv[1:] or NOTEBOOKS

for path in selected:
    print("Running", path)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(path)}})
    client.execute()
    nbformat.write(notebook, path)

print("Done. Charts are in reports/figures, tables in data/GeneratedOutput.")
