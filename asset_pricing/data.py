"""Monthly panels with characteristics aligned strictly before target returns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Month:
    date: str
    tickers: list[str]
    z: np.ndarray
    returns: np.ndarray
    managed: np.ndarray


def prepare_panel(frame: pd.DataFrame) -> tuple[list[Month], list[str]]:
    """Rank at the observation month, then join to the next calendar month.

    Input characteristics must already reflect public availability on `date`.
    A one-month lag cannot repair unreleased accounting information upstream.
    """
    required = {"date", "ticker", "returns"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV must contain date, ticker, returns, and numeric characteristics.")
    features = [c for c in frame.columns if c not in required]
    if not features or frame.empty:
        raise ValueError("Provide nonempty data and at least one characteristic.")
    df = frame.copy()
    if df[list(required)].isna().any().any():
        raise ValueError("date, ticker, and returns cannot be missing.")
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="raise").dt.to_period("M")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if (df["ticker"] == "").any() or df.duplicated(["date", "ticker"]).any():
        raise ValueError("Each ticker must be nonempty and unique within a calendar month.")
    numeric = df[["returns", *features]].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(numeric["returns"]).all() or np.isinf(numeric.to_numpy()).any():
        raise ValueError("Returns must be finite; characteristics may be missing but not infinite.")
    df[["returns", *features]] = numeric
    if df["date"].nunique() < 10:
        raise ValueError(
            f"Need at least 10 observation months for lagging and temporal evaluation; "
            f"found {df['date'].nunique()}. The bundled clean_data.csv is a one-date "
            "reconstruction demo. Use --synthetic to exercise the forecasting pipeline."
        )
    ranked = []
    for _, group in df.groupby("date", sort=True):
        values = group[features]
        values = values.fillna(values.median()).fillna(0.0)
        scaled = 2 * values.rank(method="average") / (len(values) + 1) - 1
        rows = group[["date", "ticker"]].copy()
        rows["date"] += 1
        rows[features] = scaled
        ranked.append(rows)
    aligned = df[["date", "ticker", "returns"]].merge(
        pd.concat(ranked), on=["date", "ticker"], validate="one_to_one"
    ).sort_values(["date", "ticker"])
    months = []
    for date, group in aligned.groupby("date", sort=True):
        if len(group) < 2:
            raise ValueError(f"{date}: need at least two stocks with prior-month characteristics.")
        z = group[features].to_numpy(dtype=np.float64, copy=True)
        r = group["returns"].to_numpy(dtype=np.float64, copy=True)
        # Paper eq. (16), using least squares rather than an unstable explicit inverse.
        design = np.column_stack([np.ones(len(z)), z])
        managed = np.linalg.lstsq(design, r, rcond=1e-6)[0]
        months.append(Month(str(date), group["ticker"].tolist(), z, r, managed))
    if len(months) < 9:
        raise ValueError("Need at least 9 usable target months after exact one-month alignment.")
    return months, features


def split_panel(months: list[Month]) -> tuple[list[Month], list[Month], list[Month]]:
    """60/20/20 split of whole dates; never split stocks from the same date."""
    if len(months) < 9:
        raise ValueError("Need at least 9 months for a temporal split.")
    dates = [m.date for m in months]
    if dates != sorted(set(dates)):
        raise ValueError("Months must be unique and chronologically sorted.")
    train_end, validation_end = int(len(months) * 0.6), int(len(months) * 0.8)
    return months[:train_end], months[train_end:validation_end], months[validation_end:]


def synthetic_panel(seed: int = 42, months: int = 120, stocks: int = 100,
                    features: int = 8, nonlinear: bool = True) -> pd.DataFrame:
    """Known factor DGP inspired by section 4, not the paper's calibrated simulation."""
    if months < 10 or stocks < 2 or features < 3:
        raise ValueError("Synthetic data needs >=10 months, >=2 stocks, and >=3 features.")
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(stocks, features))
    previous = np.zeros_like(state)
    rows = []
    for date in pd.date_range("2000-01-31", periods=months, freq="ME"):
        state = 0.9 * state + rng.normal(size=state.shape)
        current = 2 * pd.DataFrame(state).rank().to_numpy() / (stocks + 1) - 1
        if nonlinear:
            beta = np.column_stack([previous[:, 0] ** 2,
                                    2 * previous[:, 0] * previous[:, 1],
                                    0.6 * np.sign(previous[:, 2])])
        else:
            beta = previous[:, :3] * np.array([1.2, 1.0, 0.8])
        factors = rng.normal(0.03, 0.06, 3)
        returns = beta @ factors + rng.normal(0, 0.04, stocks)
        part = pd.DataFrame(current, columns=[f"characteristic_{i+1}" for i in range(features)])
        part.insert(0, "returns", returns)
        part.insert(0, "ticker", [f"S{i:04d}" for i in range(stocks)])
        part.insert(0, "date", date.strftime("%Y-%m-%d"))
        rows.append(part)
        previous = current
    return pd.concat(rows, ignore_index=True)
