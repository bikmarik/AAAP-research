"""
build_panel.py — месячная панель акций из открытых источников для CA0/CA1/MLP.

Источники (всё бесплатно, без WRDS):
  * состав S&P 500 на каждую дату (point-in-time) — github.com/fja05680/sp500
  * дневные цены, объёмы, сплиты — Yahoo Finance через yfinance
  * отчётность (10-K / 10-Q) с датой подачи — SEC EDGAR XBRL companyfacts API

Что на выходе (data/panel.parquet), одна строка = (месяц t, тикер):
  ret_next           — доходность месяца t+1  (таргет)
  <char>             — характеристики, известные на конец месяца t,
                       ранг-нормированные в [-1, 1] внутри месяца (как у Gu-Kelly-Xiu)
  raw_<char>         — те же характеристики до нормировки

Плюс data/panel_model.csv под `python -m asset_pricing compare --data ...`.
Там другой контракт: пайплайн лагирует признаки сам, поэтому returns — это
сверхдоходность ТОГО ЖЕ месяца, а характеристики отдаются сырыми.

Запуск:
  pip install yfinance pandas pyarrow requests
  python build_panel.py --start 2005-01-01 --end 2025-12-31 --email you@hse.ru

Всё скачанное кэшируется в data/cache, повторный запуск быстрый.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SP500_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
             "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# концепты us-gaap; внутри списка — по приоритету (компании меняют теги со временем)
INSTANT = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liab": ["Liabilities"],
    "liab_eq": ["LiabilitiesAndStockholdersEquity"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
}
ANNUAL_FLOW = {
    "ni": ["NetIncomeLoss", "ProfitLoss"],
    "sales": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
              "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
}


# ---------------------------------------------------------------- universe
def load_universe(start, end, cache):
    f = cache / "sp500_hist.csv"
    if not f.exists():
        f.write_bytes(requests.get(SP500_URL, timeout=60).content)
    h = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    months = pd.date_range(start, end, freq="ME")
    # на каждый конец месяца — последний известный состав индекса
    snap = pd.merge_asof(pd.DataFrame({"date": months}),
                         h.rename(columns={"date": "d"}),
                         left_on="date", right_on="d")
    mem = (snap.assign(ticker=snap.tickers.str.split(","))
               .explode("ticker")[["date", "ticker"]])
    mem["ticker"] = mem.ticker.str.strip().str.replace(".", "-", regex=False)
    mem["in_index"] = True
    return mem


# ---------------------------------------------------------------- prices
def _fetch_batch(batch, start, end):
    import yfinance as yf
    d = yf.download(batch, start=start, end=end, auto_adjust=False,
                    actions=True, progress=False, threads=True,
                    group_by="column")
    if d.empty:
        return None
    if not isinstance(d.columns, pd.MultiIndex):       # одиночный тикер
        d = pd.concat({batch[0]: d}, axis=1).swaplevel(axis=1)
    d = d.stack(level=1, future_stack=True).reset_index()
    d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
    return d.rename(columns={"level_1": "ticker"})


def download_prices(tickers, start, end, cache, chunk=50, retries=3):
    """Качает дневные цены чанками. Каждый чанк кэшируется отдельно, поэтому
    обрыв не обнуляет прогресс. Упавшие тикеры добираются мелкими партиями:
    таймаут Yahoo выглядит как делистинг, и без ретраев живые компании молча
    выпадают из выборки."""
    f = cache / "prices_daily.parquet"
    if f.exists():
        return pd.read_parquet(f)
    chunk_dir = cache / "px_chunks"
    chunk_dir.mkdir(exist_ok=True)
    tickers = sorted(set(tickers) | {"^GSPC"})

    parts = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        cf = chunk_dir / f"chunk_{i:05d}.parquet"
        if cf.exists():
            parts.append(pd.read_parquet(cf))
            continue
        print(f"yfinance {i}/{len(tickers)}", flush=True)
        d = _fetch_batch(batch, start, end)
        if d is None:
            continue
        d.to_parquet(cf)
        parts.append(d)

    px = pd.concat(parts, ignore_index=True).dropna(subset=["adj_close"])
    for attempt in range(1, retries + 1):
        missing = sorted(set(tickers) - set(px.ticker.unique()))
        if not missing:
            break
        print(f"retry {attempt}: {len(missing)} tickers without data", flush=True)
        got = []
        for i in range(0, len(missing), 10):
            cf = chunk_dir / f"retry{attempt}_{i:05d}.parquet"
            if cf.exists():
                got.append(pd.read_parquet(cf))
                continue
            d = _fetch_batch(missing[i:i + 10], start, end)
            if d is None:
                continue
            d.to_parquet(cf)
            got.append(d)
        if not got:
            break
        px = pd.concat([px] + got, ignore_index=True).dropna(subset=["adj_close"])

    still = sorted(set(tickers) - set(px.ticker.unique()))
    print(f"prices: {px.ticker.nunique()} tickers, {len(still)} unavailable "
          f"(делистинг/переименование): {still[:15]}{'...' if len(still) > 15 else ''}",
          flush=True)
    px.to_parquet(f)
    return px


def risk_free(start, end, cache):
    """Месячная безрисковая ставка из ^IRX (13-недельные векселя, % годовых).
    Статья моделирует сверхдоходности, поэтому ставку нужно вычесть."""
    f = cache / "irx.parquet"
    if f.exists():
        d = pd.read_parquet(f)
    else:
        import yfinance as yf
        d = yf.download("^IRX", start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Close"]].dropna()
        d.to_parquet(f)
    m = d["Close"].resample("ME").last() / 100 / 12
    m.index = pd.DatetimeIndex(m.index).as_unit("ns")
    return m.ffill()


def export_for_model(panel, rf, out):
    """CSV под asset_pricing.prepare_panel: он сам лагирует признаки на месяц и
    ранжирует их, поэтому returns здесь — сверхдоходность ТОГО ЖЕ месяца
    (raw_mom1m минус безрисковая), а характеристики отдаются сырыми."""
    raw = [c for c in panel.columns if c.startswith("raw_")]
    csv = pd.concat([panel[["date", "ticker"]].reset_index(drop=True),
                     panel[raw].reset_index(drop=True).rename(columns=lambda c: c[4:])],
                    axis=1)
    csv.insert(2, "returns", panel["raw_mom1m"].to_numpy()
               - panel["date"].map(rf).to_numpy())
    csv = csv[np.isfinite(csv["returns"])]
    csv.to_csv(out, index=False)
    print(f"{out}: {csv.shape}, средняя excess-доходность {csv.returns.mean():.4f}",
          flush=True)


def drop_corrupted(px):
    """У делистнутых тикеров Yahoo иногда не записывает сплит, и цена скачет ровно
    в N раз туда-обратно. В таргете это выглядит как ±90% за месяц и отравляет
    обучение, поэтому такой тикер убираем целиком."""
    px = px.sort_values(["ticker", "date"])
    g = (px.groupby("ticker")["adj_close"].pct_change() + 1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(g > 1, g, np.where(g > 0, 1 / g, np.nan))
    mask = ((np.round(f) >= 2) & (np.abs(f - np.round(f)) < 0.005)
            & (px["stock_splits"].fillna(0).to_numpy() == 0)
            & (np.abs(g - 1) > 0.5))
    bad = sorted(set(px["ticker"].to_numpy()[mask]))
    if bad:
        print(f"убраны тикеры с неучтёнными сплитами: {bad}", flush=True)
    return px[~px.ticker.isin(bad)]


def monthly_price_features(px):
    """Все характеристики из цен. На выходе wide-таблицы (месяцы x тикеры)."""
    px = px.sort_values(["ticker", "date"]).copy()
    # Yahoo 'Close' уже сплит-скорректирован; восстанавливаем «сырую» цену,
    # чтобы капитализация = цена * shares из EDGAR не ломалась на старых датах
    s = px["stock_splits"].replace(0, 1).fillna(1)
    after = (s.groupby(px.ticker).transform(lambda x: x[::-1].cumprod()[::-1])
             / s)  # произведение сплитов строго после даты
    px["raw_close"] = px["close"] * after
    px["r"] = px.groupby("ticker")["adj_close"].pct_change()
    px["dvol"] = px["raw_close"] * px["volume"]
    # baspread как у Gu-Kelly-Xiu: среднее за месяц от дневного (ask-bid)/midpoint.
    # В CRSP поля ASKHI/BIDLO — это "Ask ИЛИ High" и "Bid ИЛИ Low", поэтому у большой
    # части наблюдений оригинал тоже считается из дневного диапазона high-low.
    px["hl_spread"] = (px["high"] - px["low"]) / ((px["high"] + px["low"]) / 2)
    px["amihud"] = px["r"].abs() / px["dvol"].replace(0, np.nan)
    px["m"] = px["date"] + pd.offsets.MonthEnd(0)

    g = px.groupby(["m", "ticker"])
    wide = lambda ser: ser.unstack("ticker")
    adj_last = wide(g["adj_close"].last())
    out = {
        "raw_close": wide(g["raw_close"].last()),
        "maxret": wide(g["r"].max()),
        "retvol": wide(g["r"].std()),
        "dolvol": np.log(wide(g["dvol"].sum()).replace(0, np.nan)),
        "ill": np.log(wide(g["amihud"].mean()) * 1e6),
        "baspread": wide(g["hl_spread"].mean()),
        "volume_m": wide(g["volume"].sum()),
    }
    # idiovol: std остатков регрессии дневной доходности на рыночную внутри месяца.
    # Считаем через групповые суммы — это те же МНК-формулы, но без 100k отдельных
    # регрессий. В статье idiovol — 3-я по важности характеристика из 94.
    px["rm"] = px["date"].map(px[px.ticker == "^GSPC"].set_index("date")["r"])
    d = px[["m", "ticker", "r", "rm"]].dropna()
    d = d.assign(rr=d.r ** 2, mm=d.rm ** 2, rxm=d.r * d.rm)
    a = d.groupby(["m", "ticker"]).agg(n=("r", "size"), sr=("r", "sum"), sm=("rm", "sum"),
                                       srr=("rr", "sum"), smm=("mm", "sum"),
                                       srm=("rxm", "sum"))
    slope = (a.n * a.srm - a.sr * a.sm) / (a.n * a.smm - a.sm ** 2).replace(0, np.nan)
    sse = (a.srr - a.sr ** 2 / a.n) - slope * (a.srm - a.sr * a.sm / a.n)
    out["idiovol"] = wide(np.sqrt((sse / (a.n - 2)).clip(lower=0)).where(a.n >= 15))

    ret = adj_last.pct_change(fill_method=None)
    mkt = ret.pop("^GSPC")
    for k in list(out):
        out[k] = out[k].drop(columns="^GSPC", errors="ignore")

    logp = np.log(adj_last.drop(columns="^GSPC"))
    out["ret"] = ret                                   # r_t (для таргета)
    out["mom1m"] = ret                                 # краткосрочный разворот
    out["mom6m"] = np.exp(logp.shift(1) - logp.shift(6)) - 1
    out["mom12m"] = np.exp(logp.shift(1) - logp.shift(12)) - 1   # 12-1 моментум
    out["mom36m"] = np.exp(logp.shift(13) - logp.shift(36)) - 1  # долгосрочный разворот
    out["chmom"] = (np.exp(logp - logp.shift(6)) - 1) - (np.exp(logp.shift(6) - logp.shift(12)) - 1)
    cov = ret.rolling(36, min_periods=24).cov(mkt)
    var = mkt.rolling(36, min_periods=24).var()
    out["beta"] = cov.div(var, axis=0)
    out["betasq"] = out["beta"] ** 2
    return out


# ---------------------------------------------------------------- EDGAR
def sec_get(url, email, cache_file):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    time.sleep(0.12)  # лимит SEC — 10 запросов/сек
    r = requests.get(url, headers={"User-Agent": f"HSE student research {email}"}, timeout=60)
    if r.status_code != 200:
        return None
    cache_file.write_text(r.text)
    return r.json()


def _pit_series(facts, concepts, unit, annual):
    """Point-in-time ряд: (filed, value) — последнее значение, известное на дату подачи."""
    rows = []
    for pr, c in enumerate(concepts):
        for rec in facts.get(c, {}).get("units", {}).get(unit, []):
            if rec.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                continue
            if annual:
                if "start" not in rec:
                    continue
                dur = (pd.Timestamp(rec["end"]) - pd.Timestamp(rec["start"])).days
                if not 330 <= dur <= 400:
                    continue
            rows.append((pd.Timestamp(rec["end"]), pd.Timestamp(rec["filed"]), pr, rec["val"]))
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["end", "filed", "pr", "val"])
    # первое раскрытие для каждого отчётного периода (без будущих пересчётов)
    d = d.sort_values(["end", "filed", "pr"]).drop_duplicates("end", keep="first")
    d = d.sort_values("filed")
    d = d[d["end"] >= d["end"].cummax()]  # только когда пришёл более свежий период
    return d[["filed", "val"]]


def fundamentals(tickers, email, cache):
    sec_cache = cache / "sec"
    sec_cache.mkdir(exist_ok=True)
    tmap = sec_get(SEC_TICKERS_URL, email, cache / "company_tickers.json")
    cik = {v["ticker"].upper(): int(v["cik_str"]) for v in tmap.values()}
    out = []
    for i, t in enumerate(sorted(tickers)):
        if t not in cik:
            continue  # делистнутые тикеры SEC-маппинг не знает — см. ограничения
        if i % 50 == 0:
            print(f"EDGAR {i}/{len(tickers)}", flush=True)
        j = sec_get(SEC_FACTS_URL.format(cik=cik[t]), email, sec_cache / f"{t}.json")
        if not j:
            continue
        gaap = j.get("facts", {}).get("us-gaap", {})
        dei = j.get("facts", {}).get("dei", {})
        series = {k: _pit_series(gaap, c, "USD", False) for k, c in INSTANT.items()}
        series |= {k: _pit_series(gaap, c, "USD", True) for k, c in ANNUAL_FLOW.items()}
        series["shares"] = _pit_series(dei, ["EntityCommonStockSharesOutstanding"], "shares", False)
        for k, s in series.items():
            if s is not None:
                out.append(s.assign(ticker=t, var=k))
    return pd.concat(out, ignore_index=True)


def filing_dates(tickers, email, cache):
    """Даты подачи 10-K/10-Q по тикеру — из локального кэша EDGAR."""
    rows = []
    for t in tickers:
        f = cache / "sec" / f"{t}.json"
        if not f.exists():
            continue
        facts = json.loads(f.read_text()).get("facts", {}).get("us-gaap", {})
        seen = set()
        for concept in ("Assets", "NetIncomeLoss"):
            for rec in facts.get(concept, {}).get("units", {}).get("USD", []):
                if rec.get("form") in ("10-K", "10-Q") and rec["filed"] not in seen:
                    seen.add(rec["filed"])
                    rows.append((t, pd.Timestamp(rec["filed"])))
    return pd.DataFrame(rows, columns=["ticker", "filed"]).drop_duplicates()


def earnings_reaction(px, fil, months):
    """Две характеристики вокруг даты подачи последнего отчёта:

    ear    — доходность за 3 дня с даты подачи (реакция на новость, канал PEAD);
    aeavol — объём за те же 3 дня к среднему объёму месяца (канал внимания).

    Окно обязано закрыться ДО конца месяца: отчёт, поданный в последние дни,
    иначе затянул бы в признак данные следующего месяца — это утечка.

    В статье ear стоит примерно на 75-м месте из 94 по важности, но там рядом есть
    десяток характеристик вокруг отчётности (aeavol, rsup, chtx, roaq, nincr),
    которые его дублируют. В нашей панели канала «реакция на отчёт» нет вообще.
    """
    ear, aeavol = {}, {}
    mm = months.to_numpy()
    for t, g in px.groupby("ticker"):
        f = fil.loc[fil.ticker == t, "filed"].sort_values().to_numpy()
        if not len(f):
            continue
        g = g.set_index("date")
        idx = g.index.to_numpy()
        rv, vv = g["r"].to_numpy(), g["volume"].to_numpy()
        vm = g["volume"].groupby(g.index + pd.offsets.MonthEnd(0)).mean()
        pos = np.searchsorted(f, mm, side="right") - 1
        e_col, a_col = [], []
        for p, m_end in zip(pos, mm):
            a = np.searchsorted(idx, f[p]) if p >= 0 else 0
            if p < 0 or a + 3 > len(idx) or idx[a + 2] > m_end:
                e_col.append(np.nan)
                a_col.append(np.nan)
                continue
            e_col.append(float(np.prod(1 + rv[a:a + 3]) - 1))
            base = vm.get(pd.Timestamp(m_end), np.nan)
            a_col.append(float(vv[a:a + 3].mean() / base - 1) if base else np.nan)
        ear[t], aeavol[t] = e_col, a_col
    return {"ear": pd.DataFrame(ear, index=months),
            "aeavol": pd.DataFrame(aeavol, index=months)}


def asof_to_months(fund, months):
    """Для каждого (месяц, тикер) — последнее значение, поданное до конца месяца."""
    res = {}
    # pandas 3 требует одинакового разрешения у ключей merge_asof, а даты из
    # parquet и из pd.Timestamp приходят с разным (M8[us] против M8[s])
    grid = pd.DataFrame({"date": pd.DatetimeIndex(months).as_unit("ns")})
    fund = fund.assign(filed=fund["filed"].astype("datetime64[ns]"))
    for (t, v), s in fund.groupby(["ticker", "var"]):
        m = pd.merge_asof(grid, s.sort_values("filed"), left_on="date", right_on="filed")
        # отчётность старше 15 месяцев считаем протухшей
        stale = (m["date"] - m["filed"]).dt.days > 460
        res.setdefault(v, {})[t] = m["val"].where(~stale).values
    return {v: pd.DataFrame(d, index=months) for v, d in res.items()}


# ---------------------------------------------------------------- assemble
def rank_norm(wide):
    """Ранг внутри месяца -> [-1, 1]; пропуск -> 0 (медиана), как в GKX."""
    r = wide.rank(axis=1, pct=True)
    n = wide.notna().sum(axis=1)
    r = (r.mul(n, axis=0) - 0.5).div(n, axis=0)  # центрируем ранги
    return (2 * r - 1).fillna(0.0)


def build(args):
    cache = Path(args.out) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    mem = load_universe(args.start, args.end, cache)
    tickers = sorted(mem.ticker.unique())
    print(f"universe: {len(tickers)} tickers ever in S&P 500 since {args.start}")

    # цены качаем с запасом в 3 года под моментум/бету
    px_start = (pd.Timestamp(args.start) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    px = drop_corrupted(download_prices(tickers, px_start, args.end, cache))
    pf = monthly_price_features(px)
    months = pf["ret"].index
    px["r"] = px.groupby("ticker")["adj_close"].pct_change()
    for k, w in earnings_reaction(px[px.ticker != "^GSPC"],
                                  filing_dates(tickers, args.email, cache),
                                  months).items():
        pf[k] = w.reindex(columns=pf["ret"].columns)
    fw = asof_to_months(fundamentals(tickers, args.email, cache), months)
    get = lambda k: fw.get(k, pd.DataFrame(index=months)).reindex(columns=pf["ret"].columns)

    shares = get("shares")
    mcap = pf["raw_close"] * shares
    liab = get("liab").fillna(get("liab_eq") - get("equity"))
    assets = get("assets")
    ch = {k: pf[k] for k in ["mom1m", "mom6m", "mom12m", "mom36m", "chmom",
                             "maxret", "retvol", "beta", "betasq", "dolvol", "ill",
                             "baspread", "idiovol", "ear", "aeavol"]}
    ch |= {
        "mvel1": np.log(mcap),                          # размер
        "turn": pf["volume_m"] / shares,                # оборачиваемость
        "bm": get("equity") / mcap,                     # book-to-market
        "ep": get("ni") / mcap,
        "sp": get("sales") / mcap,
        "cfp": get("cfo") / mcap,
        "lev": liab / mcap,
        "cash": get("cash") / assets,
        "roe": get("ni") / get("equity"),
        "agr": assets / assets.shift(12) - 1,           # рост активов
        "sgr": get("sales") / get("sales").shift(12) - 1,
        # канал качества прибыли и эмиссии — в панели его до сих пор не было
        "acc": (get("ni") - get("cfo")) / assets,       # начисления (Sloan)
        "roaq": get("ni") / assets,
        "chcsho": shares / shares.shift(12) - 1,        # чистая эмиссия акций
        "egr": get("equity") / get("equity").shift(12) - 1,
        "cashdebt": get("cfo") / liab,
    }
    ret_next = pf["ret"].shift(-1)

    def long(wide, name):
        return wide.stack(future_stack=True).rename(name)

    cols = [long(ret_next, "ret_next")]
    cols += [long(rank_norm(w.replace([np.inf, -np.inf], np.nan)), k) for k, w in ch.items()]
    cols += [long(w.replace([np.inf, -np.inf], np.nan), f"raw_{k}") for k, w in ch.items()]
    panel = pd.concat(cols, axis=1).reset_index()
    panel.columns = ["date", "ticker"] + list(panel.columns[2:])

    panel["date"] = panel["date"].astype("datetime64[ns]")
    mem = mem.assign(date=mem["date"].astype("datetime64[ns]"))
    panel = panel.merge(mem, on=["date", "ticker"], how="inner")  # только члены индекса в t
    if panel.empty:
        raise RuntimeError("после merge с составом индекса не осталось строк — "
                           "проверьте совпадение дат и тикеров")
    panel = panel.dropna(subset=["ret_next", "raw_mom1m"]).drop(columns="in_index")
    panel = panel[(panel.date >= args.start)]
    # rank_norm считался по всем тикерам с ценами; пересчитаем внутри реальной вселенной
    for k in ch:
        panel[k] = panel.groupby("date")[f"raw_{k}"].transform(
            lambda x: rank_norm(x.to_frame().T).iloc[0].values)
    panel.to_parquet(Path(args.out) / "panel.parquet", index=False)
    print(panel.shape, panel.date.min().date(), "->", panel.date.max().date())
    print("coverage (share non-missing):")
    print(panel.filter(like="raw_").notna().mean().round(2).to_string())
    export_for_model(panel, risk_free(px_start, args.end, cache),
                     Path(args.out) / "panel_model.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--email", required=True, help="SEC требует контакт в User-Agent")
    p.add_argument("--out", default="data")
    build(p.parse_args())
