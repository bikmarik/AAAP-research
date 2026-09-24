# Verified results — September 25, 2026

## Original notebook

All cells in `AutoEncoder.ipynb` executed in a fresh Jupyter kernel. The saved notebook contains the verified outputs.

| Check | Result |
|---|---:|
| Stocks / dates | 317 / 1 |
| Epochs / seed | 100 / 42 |
| Initial training MSE | 0.06334798 |
| Final training MSE, before last update | 0.00002763 |
| Raw-return reconstruction MSE, after last update | 0.00002976 |
| In-sample zero-benchmark reconstruction R² | 99.50% |

The two MSE values use different scales: the training loss uses normalized returns; reconstruction MSE uses original return units. The model reconstructs returns it has already seen. These results demonstrate that the code trains and reconstructs; they do not demonstrate future-return prediction.

## New features: synthetic comparisons

Both experiments used 120 observation months, 100 stocks, eight characteristics, three factors, seed 42, and a maximum of 300 epochs. One-month lagging leaves 119 target months: 71 training, 24 validation, and 24 testing. Checkpoints are selected using validation forecast MSE.

| Synthetic data | Model | Held-out predictive R² | Held-out total R² | Selected epoch |
|---|---|---:|---:|---:|
| Nonlinear | CA0 | −1.255% | 25.999% | 227 |
| Nonlinear | CA1 | 1.223% | 19.460% | 91 |
| Nonlinear | Direct MLP | 8.790% | N/A | 300 |
| Linear | CA0 | 11.543% | 35.625% | 77 |
| Linear | CA1 | 13.103% | 35.733% | 42 |
| Linear | Direct MLP | 11.922% | N/A | 298 |

The direct MLP led in the nonlinear sample; CA1 led in the linear sample. This is one small synthetic draw per scenario, without a hyperparameter search or repeated-seed uncertainty estimates. It does not establish a general ranking or replicate the paper's empirical results. The nonlinear MLP selected the maximum epoch, so its best checkpoint within this run does not establish convergence.

The generated reports, plots, predictions, and checkpoints are in `artifacts/nonlinear/` and `artifacts/linear/`. Reproduce them using the commands in [README.md](README.md).

## Verification

- Fresh-kernel notebook execution succeeded, including output-shape, finite-value, and loss-improvement assertions.
- All 11 automated tests passed, including future-return leakage, test-independent weight selection, calendar lagging, changing stock coverage, missing features, factor linearity, deterministic reruns, and checkpoint serialization.
- All six saved model checkpoints were reloaded and reproduced their recorded held-out forecasts within numerical tolerance.
- The external-CSV workflow completed a three-epoch smoke test using an exported synthetic panel.
- The forecasting command correctly rejects the bundled single-date CSV with an actionable explanation.
- Dependency consistency check passed.

Verified locally with Python 3.13.12, PyTorch 2.14.0, NumPy 2.5.3, and pandas 3.0.6 on CPU. Pinned direct dependencies are in `requirements.txt`.
