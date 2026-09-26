"""Training, causal forecasts, and separate reconstruction diagnostics."""

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from .data import Month, prepare_panel, split_panel
from .models import ConditionalAutoencoder, DirectMLP


@dataclass
class Config:
    seed: int = 42
    epochs: int = 300
    patience: int = 40
    learning_rate: float = 0.003
    factors: int = 3
    hidden: int = 32
    l1: float = 1e-5

    def validate(self):
        if self.epochs < 1 or self.patience < 1 or self.hidden < 1 or self.factors < 1:
            raise ValueError("epochs, patience, hidden, and factors must be positive.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if not np.isfinite(self.l1) or self.l1 < 0:
            raise ValueError("l1 must be nonnegative and finite.")


def tensor(array):
    return torch.tensor(np.asarray(array), dtype=torch.float32)


def pack(months):
    return (tensor(np.concatenate([m.z for m in months])),
            tensor(np.stack([m.managed for m in months])),
            torch.repeat_interleave(torch.arange(len(months)),
                                    torch.tensor([len(m.returns) for m in months])),
            tensor(np.concatenate([m.returns for m in months])))


def zero_r2(actual, predicted):
    """Paper eqs. (20)-(21): pooled SSE relative to a zero-return forecast."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not np.isfinite([actual, predicted]).all():
        raise ValueError("Metrics need matching finite arrays.")
    denominator = float(np.sum(actual ** 2))
    return float(1 - np.sum((actual - predicted) ** 2) / denominator) if denominator else None


def evaluate(model, history: list[Month], target: list[Month]) -> pd.DataFrame:
    """Weights stay fixed; the factor mean expands only after each month's forecast."""
    if not history or not target or history[-1].date >= target[0].date:
        raise ValueError("Forecast history must end strictly before target dates.")
    model.eval()
    past = [m.managed for m in history]
    rows = []
    with torch.no_grad():
        for month in target:
            z = tensor(month.z)
            if isinstance(model, ConditionalAutoencoder):
                forecast = model.forecast(z, tensor(np.stack(past)))
                reconstruction = model(z, tensor(month.managed[None]),
                                       torch.zeros(len(z), dtype=torch.long)).numpy()
            else:
                forecast = model(z)
                reconstruction = np.full(len(z), np.nan)
            rows.append(pd.DataFrame({"date": month.date, "ticker": month.tickers,
                                      "actual": month.returns, "forecast": forecast.numpy(),
                                      "reconstruction": reconstruction}))
            past.append(month.managed)
    return pd.concat(rows, ignore_index=True)


def train_model(model, train, validation, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    z, managed, index, actual = pack(train)
    records, best_state, best_loss, best_epoch = [], None, float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = (model(z, managed, index) if isinstance(model, ConditionalAutoencoder)
                      else model(z))
        mse = torch.mean((prediction - actual) ** 2)
        regularizer = sum(p.abs().sum() for p in model.parameters())
        loss = mse + config.l1 * regularizer
        if not torch.isfinite(loss):
            raise ValueError("Training diverged; reduce the learning rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        result = evaluate(model, train, validation)
        validation_loss = float(np.mean((result.actual - result.forecast) ** 2))
        if not np.isfinite(validation_loss):
            raise ValueError("Validation produced non-finite predictions.")
        records.append({"epoch": epoch, "train_mse": mse.item(),
                        "validation_forecast_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = deepcopy(model.state_dict())
        if epoch - best_epoch >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return records, best_epoch


def run_comparison(frame: pd.DataFrame, config: Config):
    config.validate()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    months, features = prepare_panel(frame)
    train, validation, test = split_panel(months)
    models, predictions, curves, scores = {}, [], [], []
    for name in ("CA0", "CA1", "MLP"):
        torch.manual_seed(config.seed)
        model = (DirectMLP(len(features), config.hidden) if name == "MLP" else
                 ConditionalAutoencoder(len(features), config.factors,
                                        config.hidden if name == "CA1" else 0))
        records, best_epoch = train_model(model, train, validation, config)
        result = evaluate(model, train + validation, test)
        result.insert(0, "model", name)
        scores.append({"model": name, "best_epoch": best_epoch,
                       "test_forecast_mse": float(np.mean((result.actual - result.forecast) ** 2)),
                       "test_predictive_r2": zero_r2(result.actual, result.forecast),
                       "test_total_r2": (zero_r2(result.actual, result.reconstruction)
                                         if name != "MLP" else None)})
        predictions.append(result)
        curves.extend({"model": name, **r} for r in records)
        models[name] = model
    summary = {"config": asdict(config), "features": features,
               "input_rows": len(frame), "aligned_rows": sum(len(m.returns) for m in months),
               "splits": {name: {"start": part[0].date, "end": part[-1].date,
                                  "months": len(part), "rows": sum(len(m.returns) for m in part)}
                          for name, part in zip(("train", "validation", "test"),
                                                (train, validation, test))},
               "models": scores}
    return summary, pd.concat(predictions, ignore_index=True), pd.DataFrame(curves), models
