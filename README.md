# vol-har-conformal

Forecasting tomorrow's SPY volatility, built up from a simple linear regression to a HAR model trained on the QLIKE
loss, and tested against a naive forecast and GARCH(1,1). UTS 32513/31005 Assignment 2.

**Notebook (Colab):**
https://colab.research.google.com/github/tungtran0911/vol-har-conformal/blob/main/notebooks/A2_main.ipynb

| Section | Model | Loss | Optimiser |
|---|---|---|---|
| Simple Linear Regression | today's variance → tomorrow's | MSE | OLS |
| Multiple Linear Regression (HAR) | daily, weekly and monthly variance | MSE | OLS |
| HAR + QLIKE Loss | same inputs in logs, log link | QLIKE | gradient descent (from scratch) |
| Incl. Hyperparameter Tuning | training-window length W | QLIKE | grid search on 2017-2019 |
| Final Findings | test 2020-01 to 2026-08, refit every day, vs naive and GARCH(1,1) | QLIKE, MSE | |

## Folder

```
notebooks/A2_main.ipynb   the Colab notebook (stored with outputs)
notebooks/A2_main.py      the same cells as a script (VS Code "Run Cell"); build_notebook.py makes the .ipynb
data/spy_ohlc.csv         SPY daily prices 2005-01-03 to 2026-08-31 (Yahoo Finance), fixed snapshot
figures/                  the notebook's figures
```

## Run

```bash
pip install -r requirements.txt
python notebooks/A2_main.py                   # the notebook as a script, about 1 minute
python notebooks/build_notebook.py --execute  # rebuild the .ipynb with outputs
```
