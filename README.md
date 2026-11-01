# Estimation Robustness of Static Congestion Pricing under Learned Demand

This repository contains the synthetic simulation code, experiment outputs, and generated tables and figures for the research project.

## Reproducing the results

Requirements: Python 3.10 or later, NumPy, SciPy, and Matplotlib. From the repository root, run:

```powershell
python -m pip install -r requirements.txt
python exp_a.py
python exp_b.py
python make_tables.py
```

The experiment scripts use fixed random seeds and write their outputs to `results.json`, `figs/`, and `tables/`. The experiments in this repository are synthetic; no external traffic datasets are bundled.

## Contents

- `exp_a.py`, `exp_b.py`, `common.py`: simulation code
- `make_tables.py`: table generation from `results.json`
- `results.json`: recorded synthetic experiment outputs
- `figs/`, `tables/`: generated experiment figures and tables

## Author

Shereen Anand<br>
Independent Researcher<br>
pes1202201328@pesu.pes.edu
