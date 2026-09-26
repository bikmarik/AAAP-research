from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from asset_pricing.data import prepare_panel, split_panel, synthetic_panel
from asset_pricing.experiment import Config, evaluate, run_comparison, zero_r2
from asset_pricing.models import ConditionalAutoencoder


@pytest.fixture
def frame():
    return synthetic_panel(seed=7, months=16, stocks=16, features=3)


def test_rejects_bundled_single_date():
    with pytest.raises(ValueError, match="found 1"):
        prepare_panel(pd.read_csv("clean_data.csv"))


def test_temporal_splits_and_exact_calendar_lag(frame):
    panel, features = prepare_panel(frame)
    train, validation, test = split_panel(panel)
    assert train[-1].date < validation[0].date < test[0].date
    first = frame[frame.date == "2000-01-31"].sort_values("ticker")[features]
    np.testing.assert_allclose(panel[0].z, 2 * first.rank() / (len(first) + 1) - 1)
    # A gap must not make January characteristics look available as February data.
    gap = frame[frame.date != "2000-02-29"]
    gapped, _ = prepare_panel(gap)
    assert "2000-03" not in [m.date for m in gapped]


def test_missing_constant_features_and_unbalanced_universe(frame):
    frame.loc[::4, "characteristic_1"] = np.nan
    frame["constant"] = 5.0
    frame = frame.drop(index=[3, 20, 70])
    panel, _ = prepare_panel(frame)
    for month in panel:
        assert np.isfinite(month.z).all()
        assert np.isfinite(month.managed).all()
        assert (month.z[:, -1] == 0).all()
    assert len({len(m.returns) for m in panel}) > 1


@pytest.mark.parametrize("damage", ["duplicate", "infinite", "missing_return", "empty_ticker"])
def test_invalid_data_rejected(frame, damage):
    if damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif damage == "infinite":
        frame.loc[0, "characteristic_1"] = np.inf
    elif damage == "missing_return":
        frame.loc[0, "returns"] = np.nan
    else:
        frame.loc[0, "ticker"] = " "
    with pytest.raises(ValueError):
        prepare_panel(frame)


def test_forecasts_do_not_use_current_or_future_returns(frame):
    panel, _ = prepare_panel(frame)
    train, validation, test = split_panel(panel)
    model = ConditionalAutoencoder(3, 2, 8)
    original = evaluate(model, train + validation, test)
    altered = deepcopy(test)
    for month in altered:
        month.returns += 100
        month.managed += 100
    changed = evaluate(model, train + validation, altered)
    first = original.date == test[0].date
    np.testing.assert_array_equal(original.loc[first, "forecast"], changed.loc[first, "forecast"])
    assert not np.allclose(original.loc[first, "reconstruction"], changed.loc[first, "reconstruction"])
    # Changing the last test month's returns cannot alter any forecast.
    altered = deepcopy(test)
    altered[-1].returns *= -100
    altered[-1].managed *= -100
    changed = evaluate(model, train + validation, altered)
    np.testing.assert_array_equal(original.forecast, changed.forecast)


def test_beta_is_stockwise_and_portfolios_are_linear(frame):
    panel, _ = prepare_panel(frame)
    model = ConditionalAutoencoder(3, 2, 8)
    z = torch.tensor(panel[0].z, dtype=torch.float32)
    permutation = torch.randperm(len(z))
    torch.testing.assert_close(model.beta(z)[permutation], model.beta(z[permutation]))
    x = torch.randn(4)
    torch.testing.assert_close(model.factor(2 * x), 2 * model.factor(x))


def test_zero_benchmark_metric():
    assert zero_r2([1, 2], [0, 0]) == 0
    assert zero_r2([1, 2], [1, 2]) == 1
    assert zero_r2([0, 0], [0, 0]) is None
    assert zero_r2([1, 2], [-1, -2]) == -3


def test_training_is_reproducible_and_test_data_cannot_select_weights(frame, tmp_path):
    config = Config(seed=4, epochs=10, patience=5, hidden=8, factors=2)
    summary, predictions, curves, models = run_comparison(frame, config)
    again, same_predictions, _, same_models = run_comparison(frame, config)
    assert summary == again
    pd.testing.assert_frame_equal(predictions, same_predictions)
    assert set(predictions.model) == {"CA0", "CA1", "MLP"}
    assert np.isfinite(predictions[["actual", "forecast"]]).all().all()
    assert set(curves.model) == {"CA0", "CA1", "MLP"}
    test_start = summary["splits"]["test"]["start"]
    changed = frame.copy()
    mask = changed.date.str[:7] >= test_start
    changed.loc[mask, "returns"] += 0.5
    _, _, _, changed_models = run_comparison(changed, config)
    for name, model in models.items():
        for key, weight in model.state_dict().items():
            torch.testing.assert_close(weight, same_models[name].state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(weight, changed_models[name].state_dict()[key], rtol=0, atol=0)
        path = tmp_path / f"{name}.pt"
        torch.save(model.state_dict(), path)
        same_models[name].load_state_dict(torch.load(path, weights_only=True))
