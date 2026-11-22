# Run every test from the project folder so the data paths in config.py work.

import os
import sys

# The tests use XGBoost and PyTorch in one program. Each brings its own
# multi-threading library, and on some computers they freeze each other.
# One thread avoids that. This line must stay above the other imports.
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
