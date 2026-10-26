# Environment Setup

Requires **Python 3.10 or newer**.

## Option 1: setup script (Mac / Linux)

```bash
./scripts/setup_environment.sh
source venv/bin/activate
```

## Option 2: by hand (any system)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `xgboost` fails to import on a Mac with a `libomp` message | `brew install libomp` |
| `ModuleNotFoundError` for a project file | Run commands from the project's top folder |
| `FileNotFoundError` for the dataset | Check `data/PRSA_data_2010.1.1-2014.12.31.csv` exists with its original name |
| Streamlit asks for an email | Run `streamlit run app.py --server.headless true` |
