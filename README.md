# ML Experiment Lab

Desktop application for creating, running, comparing and analysing Machine Learning experiments.

## Current milestone
Milestone 0: project foundation + first PySide6 window.

## Tech stack
- Python
- PySide6
- FastAPI (later)
- scikit-learn (later)
- SQLite/PostgreSQL (later)

## Local setup (Windows)

```powershell
py -3.13 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m ml_experiment_lab
```

Developer tools:

```powershell
pip install -e ".[dev]"
pytest
```
