# Estimation Robustness of Static Congestion Pricing under Learned Demand

This repository contains the paper, its synthetic simulation code, and the generated tables and figures.

## Reproducing the results

Requirements: Python 3.10 or later, NumPy, SciPy, and Matplotlib. From the repository root, run:

```powershell
python -m pip install -r requirements.txt
python exp_a.py
python exp_b.py
python make_tables.py
```

The experiment scripts use fixed random seeds and write their outputs to `results.json`, `figs/`, and `tables/`. To compile the manuscript, install [Tectonic](https://tectonic-typesetting.github.io/) and run:

```powershell
tectonic paper.tex
```

`paper.pdf` is the compiled manuscript included with this release. The experiments in this repository are synthetic; no external traffic datasets are bundled.

## Contents

- `paper.tex`, `appendix.tex`, and `paper.pdf`: manuscript source and compiled paper
- `exp_a.py`, `exp_b.py`, `common.py`: simulation code
- `make_tables.py`: table generation from `results.json`
- `results.json`: recorded synthetic experiment outputs
- `figs/`, `tables/`: generated manuscript figures and tables

## Author

Shereen Anand<br>
Independent Researcher<br>
pes1202201328@pesu.pes.edu
