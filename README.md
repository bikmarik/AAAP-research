# Autoencoder Asset Pricing Models

A runnable version of the original PyTorch notebook, plus a small CA0 / CA1 / direct-MLP experiment based on Gu, Kelly, and Xiu, *Autoencoder Asset Pricing Models* (September 2019). The [paper](docs/06.%20Autoencoder%20Asset%20Pricing%20Models.pdf) is included in `docs/`; [the authors' SSRN page](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3335536) provides the reference.

**The bundled `clean_data.csv` contains 317 stocks on December 31, 2015 only.** It supports an in-sample reconstruction demonstration. It cannot establish forecasting performance or reproduce the paper's multi-decade empirical study.

## Install and run

From this folder, using Python 3.13 (the verified version):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_notebook.py
.venv/bin/python -m pytest -q
```

The notebook runner starts a fresh Jupyter kernel, executes every cell, checks finite output and loss improvement, and saves the executed `AutoEncoder.ipynb`. Alternatively, open the notebook with a Jupyter-compatible editor using this virtual environment. On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

The original architecture is preserved in the notebook. Repairs include portable CSV loading, inferred dimensions, an explicit random seed, removal of unused TensorFlow, normalization without modifying raw returns, and evaluation without recording gradients. Reconstruction uses the same returns as both input and target; its R² is not a predictive R².

## New experiment

The experiment implements the comparison proposed in the [presentation plan](docs/autoencoder_presentation_plan.md):

- **CA0:** a linear characteristics-to-beta mapping and a linear factor encoder.
- **CA1:** a 32-neuron ReLU hidden layer in the beta mapping and a linear factor encoder.
- **MLP:** the same characteristics directly predict returns through a 32-neuron hidden layer.

Run both known-data demonstrations:

```sh
.venv/bin/python -m asset_pricing compare --synthetic --output artifacts/nonlinear
.venv/bin/python -m asset_pricing compare --synthetic --linear --output artifacts/linear
```

Both commands generate 120 months × 100 stocks with eight characteristics and three true factors. `--linear` changes the data-generating relationship, not the models being compared. These small simulations are inspired by section 4, but are not the paper's calibrated Monte Carlo experiment or real market data. See [RESULTS.md](RESULTS.md) for the verified run.

Each output folder contains:

| File | Contents |
|---|---|
| `report.md`, `comparison.png` | Readable comparison and plot |
| `summary.json` | Settings, dates, feature names, data hash, versions, metrics |
| `predictions.csv` | Per-stock held-out forecasts, actual returns, and separate CA reconstructions |
| `training.csv` | Training error and validation forecast error by epoch |
| `CA0.pt`, `CA1.pt`, `MLP.pt` | Selected model weights, architecture settings, and feature order |
| `synthetic_data.csv` | Generated inputs, when using `--synthetic` |

Generated files live in the ignored `artifacts/` folder; commands recreate them. Checkpoints contain model weights and configuration, not historical factor observations. CA forecasting also requires a strictly earlier panel prepared with the same ordered features, as shown by `evaluate()` in `asset_pricing/experiment.py`.

Available controls include `--seed`, `--epochs`, `--patience`, `--factors`, `--hidden`, `--learning-rate`, and `--l1`. Run `python -m asset_pricing compare --help` for defaults.

## Use an external monthly panel

```sh
.venv/bin/python -m asset_pricing compare --data path/to/monthly_panel.csv --output artifacts/real_panel
```

The CSV must contain `date`, `ticker`, `returns`, and one or more numeric characteristic columns. `returns` should be the month's excess return in decimal units; characteristic values must be publicly available as of the row's date. Use consistent monthly observations and at least 10 distinct months with sufficient overlapping stocks. Ten months is a software minimum, not sufficient evidence for an empirical conclusion.

The loader median-imputes characteristics within each observation month, rank-normalizes them into (−1, 1), then joins them by ticker to returns in the **next calendar month**. It never substitutes a more distant observation when a month is missing. Entirely missing or constant characteristics map to zero. Missing/infinite returns, infinite features, empty tickers, and duplicate ticker-month pairs are rejected. Unmatched ticker-month rows are excluded; input and aligned row counts are saved.

Do not pre-lag the characteristics yourself: the loader performs a one-month lag. Accounting figures must already be delayed to their actual publication dates upstream. The loader cannot establish point-in-time availability or repair revised historical data. It also does not subtract a risk-free rate automatically.

## Method and scope

For each target month, characteristic-managed portfolios follow paper equation (16), using a least-squares solve with an intercept and a rank cutoff of `1e-6` instead of explicitly inverting a potentially singular matrix. Stock-wise beta mappings multiply shared latent factors; the factor encoder has no additive bias, so it remains a linear combination of portfolio returns.

Whole target months are split chronologically into 60% training, 20% validation, and 20% test data. CA models train on reconstruction error; the MLP trains on direct forecast error. All models use Adam and L1 regularization, with the best checkpoint selected by validation **forecast** MSE and patience-based early stopping. No test data enter model fitting or checkpoint selection.

The two reported R² measures use the paper's zero-return benchmark, `1 − sum((actual − estimate)²) / sum(actual²)`:

- **Predictive R²:** CA forecasts use current lagged characteristics and the expanding mean of factors observed strictly before the target month (equation 21). MLP forecasts use characteristics only.
- **Total R²:** CA reconstructions use contemporaneous realized factors (equation 20). This is reported separately and has no direct MLP equivalent.

Weights stay fixed throughout validation and testing. Past validation/test returns may update the factor mean once observed; they do not retrain weights. Tests cover this timing, future-return perturbations, exact month alignment, missing and unbalanced data, stock-wise betas, linear factors, the zero-return metric, deterministic training, and checkpoint serialization.

This is a compact research prototype, not a full replication. It omits annual refitting, ensembles, batch normalization, the original tuning search, IPCA/PCA benchmarks, and portfolio trading analysis. A single generated sample does not establish model superiority. Seeded CPU execution is verified locally; PyTorch does not guarantee identical results across releases or platforms ([reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html)).
