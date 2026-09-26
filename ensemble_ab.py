"""ensemble_ab.py — сравнение двух наборов признаков по метрикам статьи.

Считаем не только predictive R² (eq. 20-21), но и главный экономический результат
статьи (раздел 3.4): каждый месяц сортируем акции по прогнозу в децили, покупаем
10-й дециль, продаём 1-й, ребалансируем ежемесячно, берём годовой Sharpe — в
equal-weight и value-weight версиях. Decile-сортировка зависит только от порядка
прогнозов, поэтому видит улучшение ранжирования там, где pooled R² его не видит:
R² съедает идиосинкратическая дисперсия, которую модель и не пытается объяснить.

Прогнозы усредняются по сидам (приём GKX): разброс между сидами ±0.12 п.п. R²
больше вклада одной фичи, и без усреднения сравнение меряет удачу инициализации.

Разницы бутстрапируются по МЕСЯЦАМ: внутри месяца зависимость сильная, между
месяцами почти нет.

Запуск:
  python ensemble_ab.py "<glob_A>" "<glob_B>" --labels с_ear без_ear
"""
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)


def load(paths):
    parts = []
    for p in paths:
        f = p / "predictions.csv"
        if f.exists():
            parts.append(pd.read_csv(f).assign(seed=p.name.rsplit("_", 1)[-1]))
    if not parts:
        raise SystemExit("не найдено ни одного predictions.csv")
    return pd.concat(parts, ignore_index=True)


def ensemble(d):
    """Средний прогноз по сидам для каждой (модель, месяц, акция)."""
    return (d.groupby(["model", "date", "ticker"])
             .agg(actual=("actual", "first"), forecast=("forecast", "mean"),
                  seeds=("seed", "nunique"))
             .reset_index())


def market_caps(panel_path="data/panel.parquet"):
    p = pd.read_parquet(panel_path, columns=["date", "ticker", "raw_mvel1"])
    p["date"] = pd.to_datetime(p["date"]).dt.to_period("M").astype(str)
    return p.assign(mcap=np.exp(p["raw_mvel1"])).drop(columns="raw_mvel1")


def long_short(g, weight):
    """Доходность зеро-инвестиционного портфеля 10-й дециль минус 1-й."""
    if weight == "vw":
        g = g[g["mcap"].notna() & (g["mcap"] > 0)]
    if len(g) < 30:
        return np.nan
    q = pd.qcut(g["forecast"].rank(method="first"), 10, labels=False)
    lo, hi = g[q == 0], g[q == 9]
    if weight == "ew":
        return hi["actual"].mean() - lo["actual"].mean()
    return (np.average(hi["actual"], weights=hi["mcap"])
            - np.average(lo["actual"], weights=lo["mcap"]))


def monthly_ls(e, mcap, weight):
    d = e.merge(mcap, on=["date", "ticker"], how="left")
    s = d.groupby("date").apply(long_short, weight=weight, include_groups=False)
    return s.dropna().sort_index()


def sharpe(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(12) if len(x) > 2 and x.std() else np.nan


def r2(g):
    return 1 - np.sum((g.actual - g.forecast) ** 2) / np.sum(g.actual ** 2)


def boot_r2(a, b, draws):
    m = a.merge(b, on=["date", "ticker"], suffixes=("_a", "_b"))
    by = {d: g for d, g in m.groupby("date")}
    months = np.array(list(by))
    out = []
    for _ in range(draws):
        s = pd.concat([by[d] for d in RNG.choice(months, len(months), replace=True)],
                      ignore_index=True)
        den = np.sum(s.actual_a ** 2)
        out.append((np.sum((s.actual_b - s.forecast_b) ** 2)
                    - np.sum((s.actual_a - s.forecast_a) ** 2)) / den)
    return np.array(out)


def boot_sharpe(sa, sb, draws):
    common = sa.index.intersection(sb.index)
    sa, sb = sa.loc[common].to_numpy(), sb.loc[common].to_numpy()
    out = []
    for _ in range(draws):
        i = RNG.integers(0, len(common), len(common))
        out.append(sharpe(pd.Series(sa[i])) - sharpe(pd.Series(sb[i])))
    return np.array(out)


def report(name, va, vb, d, unit=""):
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{name:<18} {va:>9.3f}{unit} {vb:>9.3f}{unit} {d.mean():>9.3f}{unit} "
          f"[{lo:>7.3f}, {hi:>7.3f}] {(d > 0).mean():>6.2f}")


def main(args):
    ea = ensemble(load(sorted(Path(p) for p in glob(args.a))))
    eb = ensemble(load(sorted(Path(p) for p in glob(args.b))))
    mcap = market_caps(args.panel)
    la, lb = args.labels
    print(f"{la}: сидов {ea.seeds.max()} | {lb}: сидов {eb.seeds.max()} | "
          f"месяцев теста {ea.date.nunique()}\n")
    print(f"{'метрика':<18} {la:>10} {lb:>9} {'разница':>10} {'95% ДИ':>18} {'P(>0)':>6}")
    for model in ["CA0", "CA1", "MLP"]:
        ga, gb = ea[ea.model == model], eb[eb.model == model]
        if ga.empty or gb.empty:
            continue
        print(f"--- {model} ---")
        report("predictive R², %", 100 * r2(ga), 100 * r2(gb),
               100 * boot_r2(ga, gb, args.draws))
        for w, tag in [("ew", "Sharpe EW"), ("vw", "Sharpe VW")]:
            sa, sb = monthly_ls(ga, mcap, w), monthly_ls(gb, mcap, w)
            report(tag, sharpe(sa), sharpe(sb), boot_sharpe(sa, sb, args.draws))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--panel", default="data/panel.parquet")
    p.add_argument("--draws", type=int, default=2000)
    main(p.parse_args())
