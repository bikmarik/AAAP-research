"""screen_features.py — быстрый отбор фич по маржинальной информации.

Гонять CA0/CA1/MLP на каждого кандидата бессмысленно: шум оптимизации (±0.15 п.п.
predictive R²) больше эффекта одной фичи. Поэтому меряем то же, что раздел 3.6
статьи, но без нейросети:

  1. в каждом месяце регрессируем ret_next на УЖЕ имеющиеся признаки (МНК);
  2. берём остатки — это то, что панель ещё не объясняет;
  3. считаем кросс-секционную корреляцию Спирмена кандидата с остатками;
  4. усредняем по месяцам и берём t-статистику (Newey-West не нужен: IC по
     месяцам почти не автокоррелированы, но t считаем по разбросу IC).

Фича полезна, если |mean IC| ощутимо больше нуля и |t| > 2. Метод стабилен:
никакой зависимости от сида, инициализации и ранней остановки.

Запуск:
  python screen_features.py --email you@hse.ru
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import SEC_FACTS_URL, SEC_TICKERS_URL, rank_norm, sec_get


# ---------------------------------------------------------------- кандидаты
def dividend_yield(px, months):
    """dy: дивиденды за 12 месяцев / цена. Канал выплат — в панели его нет."""
    d = px[["date", "ticker", "dividends", "raw_close"]].copy()
    d["m"] = d["date"] + pd.offsets.MonthEnd(0)
    g = d.groupby(["m", "ticker"])
    div = g["dividends"].sum().unstack("ticker").fillna(0.0)
    price = g["raw_close"].last().unstack("ticker")
    return (div.rolling(12, min_periods=6).sum() / price).reindex(months)


def dolvol_vol(px, months):
    """std_dolvol: разброс дневного оборота внутри месяца (как у GKX)."""
    d = px[["date", "ticker", "dvol"]].copy()
    d["m"] = d["date"] + pd.offsets.MonthEnd(0)
    d["ldv"] = np.log(d["dvol"].where(d["dvol"] > 0))
    return d.groupby(["m", "ticker"])["ldv"].std().unstack("ticker").reindex(months)


def filing_dates(tickers, email, cache):
    """Даты подачи 10-K/10-Q по каждому тикеру из локального кэша EDGAR."""
    sec_cache = cache / "sec"
    tmap = sec_get(SEC_TICKERS_URL, email, cache / "company_tickers.json")
    cik = {v["ticker"].upper(): int(v["cik_str"]) for v in tmap.values()}
    rows = []
    for t in tickers:
        f = sec_cache / f"{t}.json"
        if not f.exists():
            continue
        facts = json.loads(f.read_text()).get("facts", {}).get("us-gaap", {})
        seen = set()
        for concept in ("Assets", "NetIncomeLoss"):
            for rec in facts.get(concept, {}).get("units", {}).get("USD", []):
                if rec.get("form") in ("10-K", "10-Q") and rec.get("filed") not in seen:
                    seen.add(rec["filed"])
                    rows.append((t, pd.Timestamp(rec["filed"])))
    return pd.DataFrame(rows, columns=["ticker", "filed"]).drop_duplicates()


def staleness_and_ear(px, fil, months):
    """Две фичи вокруг даты отчёта — канал раскрытия информации:

    staleness — сколько дней прошло с последней подачи 10-K/10-Q. В статье такой
      переменной нет вовсе: её 94 характеристики молча считаются одинаково свежими.
    ear — доходность за 3 дня с даты подачи (реакция рынка на отчёт, PEAD).
    """
    px = px.sort_values(["ticker", "date"])
    stale, ear = {}, {}
    ret_by_t = {t: g.set_index("date")["r"] for t, g in px.groupby("ticker")}
    fil = fil.sort_values("filed")
    for t, g in fil.groupby("ticker"):
        f = g["filed"].to_numpy()
        if t not in ret_by_t:
            continue
        r = ret_by_t[t]
        pos = np.searchsorted(f, months.to_numpy(), side="right") - 1
        last = np.where(pos >= 0, f[np.clip(pos, 0, None)], np.datetime64("NaT"))
        stale[t] = (months.to_numpy() - last) / np.timedelta64(1, "D")
        # 3-дневная реакция на последний отчёт, известный к концу месяца.
        # Окно обязано ЗАКРЫТЬСЯ до конца месяца: отчёт, поданный в последние дни,
        # иначе затянул бы в признак доходности следующего месяца — это утечка.
        idx, rv = r.index.to_numpy(), r.to_numpy()
        win = []
        for d, m_end in zip(last, months.to_numpy()):
            a = np.searchsorted(idx, d)
            if np.isnat(d) or a + 3 > len(idx) or idx[a + 2] > m_end:
                win.append(np.nan)
                continue
            win.append(float(np.prod(1 + rv[a:a + 3]) - 1))
        ear[t] = win
    return (pd.DataFrame(stale, index=months), pd.DataFrame(ear, index=months))


# ---------------------------------------------------------------- скрининг
def marginal_ic(panel, candidate, base):
    """Средний IC кандидата с остатками от регрессии на имеющиеся признаки."""
    ics = []
    for date, g in panel.groupby("date"):
        x = g[base].to_numpy()
        y = g["ret_next"].to_numpy()
        c = g[candidate].to_numpy()
        ok = np.isfinite(c)
        if ok.sum() < 30:
            continue
        design = np.column_stack([np.ones(len(x)), x])
        resid = y - design @ np.linalg.lstsq(design, y, rcond=1e-6)[0]
        # Спирмен = Пирсон по рангам; так не нужен scipy
        s = pd.Series(resid[ok]).rank().corr(pd.Series(c[ok]).rank())
        if np.isfinite(s):
            ics.append(s)
    ic = np.array(ics)
    t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 2 else np.nan
    return ic.mean(), t, len(ic)


def main(args):
    cache = Path("data") / "cache"
    panel = pd.read_parquet("data/panel.parquet")
    months = pd.DatetimeIndex(sorted(panel.date.unique()))
    px = pd.read_parquet(cache / "prices_daily.parquet")
    px = px[px.ticker != "^GSPC"].sort_values(["ticker", "date"])
    px["r"] = px.groupby("ticker")["adj_close"].pct_change()
    px["raw_close"] = px["close"]
    px["dvol"] = px["close"] * px["volume"]

    tickers = sorted(panel.ticker.unique())
    fil = filing_dates(tickers, args.email, cache)
    print(f"дат подачи: {len(fil)} по {fil.ticker.nunique()} тикерам", flush=True)
    stale, ear = staleness_and_ear(px, fil, months)

    cands = {"dy": dividend_yield(px, months), "std_dolvol": dolvol_vol(px, months),
             "staleness": stale, "ear": ear}
    for name, wide in cands.items():
        long = wide.stack(future_stack=True).rename(name)
        long.index.names = ["date", "ticker"]
        panel = panel.merge(rank_norm(wide.replace([np.inf, -np.inf], np.nan))
                            .stack(future_stack=True).rename(name)
                            .rename_axis(["date", "ticker"]).reset_index(),
                            on=["date", "ticker"], how="left")

    base = [c for c in panel.columns
            if not c.startswith("raw_") and c not in ("date", "ticker", "ret_next")
            and c not in cands]
    print(f"\nбазовые признаки: {len(base)}")
    print(f"{'кандидат':<12} {'mean IC':>9} {'t-stat':>8} {'месяцев':>8}   вывод")
    # baspread/idiovol как контроль: про них уже известно, что прироста нет
    for name in ["baspread", "idiovol", *cands]:
        b = [c for c in base if c != name]
        m, t, n = marginal_ic(panel, name, b)
        verdict = "есть сигнал" if abs(t) > 2 else "шум"
        print(f"{name:<12} {m:>9.4f} {t:>8.2f} {n:>8}   {verdict}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    main(p.parse_args())
