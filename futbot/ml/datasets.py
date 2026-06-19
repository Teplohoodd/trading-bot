"""Kaggle dataset loaders.

Downloads via kagglehub (cached under ~/.cache/kagglehub) and converts each
dataset into a uniform daily OHLCV DataFrame with columns:
    time, open, high, low, close, volume, ticker, concept

`concept` is our internal asset-class label used for per-class ML training:
    "oil"   → CL=F (closest public proxy to Brent / our BR contract)
    "gas"   → NG=F (NatGas — proxy for NG contract)
    "gold"  → GC=F (Gold — proxy for GD contract)
    "sber"  → SBER (MOEX share — proxy for SR contract)
    "lkoh"  → LKOH (MOEX share — proxy for LK contract)
    "moex"  → daily-volume-weighted MOEX index proxy (proxy for MX contract)

These are DAILY series.  Our intraday bot uses them as a longer-horizon
prior for the ML gate (see ml/trainer.py and pipeline/ml_gate.py).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("futbot.ml.datasets")


# Mapping concept → (kaggle slug, internal loader function name)
CONCEPTS = {
    "oil": ("guillemservera/fuels-futures-data", "CL=F"),
    "gas": ("guillemservera/fuels-futures-data", "NG=F"),
    "gold": ("guillemservera/precious-metals-data", "GC=F"),
    "sber": ("alexanderkobzar/moex-shares-prices-since-2004", "SBER"),
    "lkoh": ("alexanderkobzar/moex-shares-prices-since-2004", "LKOH"),
    "moex": ("alexanderkobzar/moex-shares-prices-since-2004", "__moex_index__"),
    "gazp": ("alexanderkobzar/moex-shares-prices-since-2004", "GAZP"),
}

# Which futbot contract base maps to which concept (used by ml_gate at runtime).
CONTRACT_TO_CONCEPT = {
    "BR": "oil",
    "NG": "gas",
    "GD": "gold",
    "SR": "sber",
    "LK": "lkoh",
    "MX": "moex",
    "GZ": "gazp",
    "RT": "moex",  # RTS index = USD-denominated MOEX proxy
    # Si, EURRUBF: no good Kaggle daily proxy → ML gate stays pass-through.
}


def _ensure_kagglehub():
    try:
        import kagglehub  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "kagglehub is required for ML training but is not installed. "
            "Run: pip install kagglehub"
        ) from e


def _download(slug: str) -> Path:
    """Cached download of a Kaggle dataset.  Returns the unpacked dir."""
    _ensure_kagglehub()
    import kagglehub

    path = kagglehub.dataset_download(slug)
    return Path(path)


def _load_fuel_or_metal(slug: str, ticker: str) -> pd.DataFrame:
    """Both fuels-futures-data and precious-metals-data share the same
    'all_*_data.csv' shape: ticker, commodity, date, open, high, low,
    close, volume."""
    root = _download(slug)
    # The CSV name varies by dataset — pick the first 'all_*.csv'.
    csvs = [
        p
        for p in root.iterdir()
        if p.is_file() and p.name.startswith("all_") and p.name.endswith(".csv")
    ]
    if not csvs:
        raise FileNotFoundError(f"no all_*.csv inside {root}")
    df = pd.read_csv(csvs[0])
    df = df[df["ticker"] == ticker].copy()
    if df.empty:
        raise ValueError(f"ticker {ticker} not present in {csvs[0].name}")
    df["time"] = pd.to_datetime(df["date"], utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0].reset_index(drop=True)
    return df


def _load_moex_share(ticker: str) -> pd.DataFrame:
    """One row per (secid, tradedate).  We pick a secid and pivot to daily OHLCV."""
    root = _download("alexanderkobzar/moex-shares-prices-since-2004")
    csv = root / "MOEX_shares_prices.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    # 142 MB — read in chunks to avoid blowing memory on small machines.
    chunks = []
    for chunk in pd.read_csv(csv, chunksize=200_000):
        chunk = chunk[chunk["secid"] == ticker]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"secid {ticker} not present")
    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["tradedate"], utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["close"] > 0].reset_index(drop=True)
    return df


def _load_moex_index_proxy() -> pd.DataFrame:
    """Build a daily volume-weighted-price MOEX proxy from all Russian shares.
    Not the official IMOEX (no licence), but close enough for regime
    classification and direction priors."""
    root = _download("alexanderkobzar/moex-shares-prices-since-2004")
    csv = root / "MOEX_shares_prices.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    parts = []
    for chunk in pd.read_csv(csv, chunksize=200_000):
        chunk = chunk[chunk["is_russian"] == True]  # noqa: E712
        chunk = chunk.dropna(subset=["close"])
        chunk = chunk[chunk["close"] > 0]
        if chunk.empty:
            continue
        # daily VWAP across all Russian shares
        chunk = chunk.assign(notional=chunk["close"] * chunk["volume"])
        agg = chunk.groupby("tradedate").agg(
            close=("close", "mean"),  # equal-weight close, simple
            open=("open", "mean"),
            high=("high", "max"),
            low=("low", "min"),
            volume=("volume", "sum"),
        )
        parts.append(agg)
    if not parts:
        raise ValueError("no russian rows in MOEX dataset?")
    df = (
        pd.concat(parts)
        .groupby(level=0)
        .agg(
            close=("close", "mean"),
            open=("open", "mean"),
            high=("high", "max"),
            low=("low", "min"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    df["time"] = pd.to_datetime(df["tradedate"], utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def load_concept(concept: str) -> pd.DataFrame:
    """Public entry: get daily OHLCV for a `concept`.  Caches to parquet
    under data/futbot_external/ for subsequent runs."""
    if concept not in CONCEPTS:
        raise ValueError(f"unknown concept '{concept}' — known: {list(CONCEPTS)}")

    cache_dir = Path("data/futbot_external")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{concept}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

    slug, ticker = CONCEPTS[concept]
    if ticker == "__moex_index__":
        df = _load_moex_index_proxy()
    elif "moex" in slug:
        df = _load_moex_share(ticker)
    else:
        df = _load_fuel_or_metal(slug, ticker)

    df.to_parquet(cache_file, index=False)
    logger.info(f"loaded concept={concept}: {len(df)} rows, cached → {cache_file}")
    return df


def load_all_concepts() -> dict[str, pd.DataFrame]:
    out = {}
    for c in CONCEPTS:
        try:
            out[c] = load_concept(c)
        except Exception as e:
            logger.warning(f"  {c}: load failed ({e})")
    return out
