"""Run with python -m asset_pricing compare --synthetic."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import synthetic_panel
from .experiment import Config, run_comparison


def write_results(output, summary, predictions, curves, models):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    predictions.to_csv(output / "predictions.csv", index=False)
    curves.to_csv(output / "training.csv", index=False)
    for name, model in models.items():
        torch.save({"model": name, "config": summary["config"],
                    "features": summary["features"], "state_dict": model.state_dict()},
                   output / f"{name}.pt")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"CA0": "#3872a8", "CA1": "#128477", "MLP": "#ba6937"}
    for name, group in curves.groupby("model", sort=False):
        axes[0].plot(group.epoch, group.validation_forecast_mse, label=name, color=colors[name])
    axes[0].set(title="Validation forecast error", xlabel="Training epoch", ylabel="Mean squared error")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    scores = summary["models"]
    values = [s["test_predictive_r2"] for s in scores]
    if all(v is not None for v in values):
        bars = axes[1].bar([s["model"] for s in scores], np.array(values) * 100,
                           color=list(colors.values()))
        axes[1].bar_label(bars, fmt="%.2f%%", padding=4)
    else:
        axes[1].text(0.5, 0.5, "R² undefined: all target returns are zero", ha="center")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(title="Held-out predictive R²", ylabel="R² against zero return (%)")
    axes[1].margins(y=0.2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Synthetic demonstration" if summary["synthetic"] else "Monthly panel experiment")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)

    def percentage(value):
        return f"{100 * value:.3f}%" if value is not None else "N/A"

    lines = ["# Experiment results", "", summary["description"], "",
             "| Model | Predictive R² | Total R² (reconstruction) | Selected epoch |",
             "|---|---:|---:|---:|"]
    for score in scores:
        lines.append(f"| {score['model']} | {percentage(score['test_predictive_r2'])} | "
                     f"{percentage(score['test_total_r2'])} | {score['best_epoch']} |")
    lines.extend(["", "![Model comparison](comparison.png)", "", "## Evaluation", ""])
    for name, part in summary["splits"].items():
        lines.append(f"- {name.title()}: {part['start']}–{part['end']}, "
                     f"{part['months']} months, {part['rows']} observations.")
    lines.extend(["", "## Interpretation", "",
                  "Forecasts use characteristics from the preceding calendar month. "
                  "CA forecasts use expanding means of factors observed strictly before the target month. "
                  "Total R² uses contemporaneous returns and measures reconstruction, not forecasting.",
                  "", "All models use the same dates, characteristics, training budget, and "
                  "validation-forecast early stopping criterion. Model weights stay fixed during testing.",
                  "", "This compact implementation does not reproduce the full paper: it omits "
                  "annual refitting, ensembles, batch normalization, and the original hyperparameter search. "
                  "The external CSV must contain point-in-time available features and monthly excess returns. "
                  "A one-month lag cannot correct unreleased or revised source data.", "",
                  "No trading-performance conclusion is supported by the bundled single-date CSV "
                  "or by synthetic results.", ""])
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="Compare CA0, CA1, and direct MLP on whole-date splits")
    source = compare.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Monthly CSV with date, ticker, returns, characteristics")
    source.add_argument("--synthetic", action="store_true", help="Use explicitly synthetic monthly data")
    compare.add_argument("--linear", action="store_true", help="Use a linear synthetic factor DGP")
    compare.add_argument("--output", type=Path, default=Path("artifacts/comparison"))
    compare.add_argument("--epochs", type=int, default=300)
    compare.add_argument("--patience", type=int, default=40)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--factors", type=int, default=3)
    compare.add_argument("--hidden", type=int, default=32)
    compare.add_argument("--learning-rate", type=float, default=0.003)
    compare.add_argument("--l1", type=float, default=1e-5)
    args = parser.parse_args()
    if args.linear and not args.synthetic:
        parser.error("--linear only applies to --synthetic")
    try:
        frame = (synthetic_panel(seed=args.seed, nonlinear=not args.linear) if args.synthetic
                 else pd.read_csv(args.data))
        config = Config(seed=args.seed, epochs=args.epochs, patience=args.patience,
                        learning_rate=args.learning_rate, factors=args.factors,
                        hidden=args.hidden, l1=args.l1)
        print("Running CA0 / CA1 / MLP on chronological train, validation, and test splits...", flush=True)
        summary, predictions, curves, models = run_comparison(frame, config)
        summary.update({"synthetic": args.synthetic,
                        "description": (f"Synthetic {'linear' if args.linear else 'nonlinear'} factor data; "
                                        "pipeline demonstration, not empirical evidence." if args.synthetic
                                        else f"External monthly panel: {args.data.name}."),
                        "data_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                        "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                                     "pandas": pd.__version__}})
        write_results(args.output, summary, predictions, curves, models)
        if args.synthetic:
            frame.to_csv(args.output / "synthetic_data.csv", index=False)
        print(pd.DataFrame(summary["models"]).fillna("N/A").to_string(index=False))
        print(f"Results: {(args.output / 'report.md').resolve()}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
