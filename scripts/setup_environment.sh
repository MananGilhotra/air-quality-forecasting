#!/bin/bash
# Creates the virtual environment and installs all project packages.
# Usage (from the project folder):  ./scripts/setup_environment.sh

set -e
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
echo "Done. Activate it with:  source venv/bin/activate"
