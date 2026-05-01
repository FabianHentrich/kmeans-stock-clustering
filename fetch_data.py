"""Regenerate stockdata.csv and sectors.csv from Slickcharts + Yahoo Finance.

Run once before opening the notebook:
    python fetch_data.py

Outputs land in ./data/:
  - sectors.csv   : Ticker, Name, Sector (GICS Sector via yfinance.info)
  - stockdata.csv : daily adjusted close, indexed by Date, columns = tickers
"""
from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
START = "2010-01-01"
END = "2019-12-31"

SLICKCHARTS_URL = "https://www.slickcharts.com/sp500"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
}

CHUNK_SIZE = 50          # tickers per yf.download call — bulk calls > ~100 silently drop data
MAX_RETRIES = 3
RETRY_SLEEP = 5          # seconds between retries


def _normalize_ticker(t: str) -> str:
    # yfinance uses "-" instead of "." (e.g. BRK.B -> BRK-B)
    return t.replace(".", "-").strip()


def fetch_sp500_metadata() -> pd.DataFrame:
    """Slickcharts gives Ticker + Name. Sector is filled in via yfinance.Ticker.info."""
    resp = requests.get(SLICKCHARTS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]
    meta = (
        table[["Company", "Symbol"]]
        .rename(columns={"Symbol": "Ticker", "Company": "Name"})
    )
    meta["Ticker"] = meta["Ticker"].map(_normalize_ticker)
    print(f"  slickcharts: {len(meta)} tickers; fetching GICS sectors via yfinance.info (slow)...")
    sectors: list[str] = []
    for i, t in enumerate(meta["Ticker"], 1):
        try:
            sectors.append(yf.Ticker(t).info.get("sector", "") or "")
        except Exception:
            sectors.append("")
        if i % 50 == 0:
            print(f"    {i}/{len(meta)}")
    meta["Sector"] = sectors
    return meta[["Ticker", "Name", "Sector"]]


def _extract_close(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """yf.download returns MultiIndex columns for multi-ticker, flat for single-ticker."""
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"].copy()
    return data[["Close"]].rename(columns={"Close": tickers[0]})


def _download_chunk(tickers: list[str]) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers,
                start=START,
                end=END,
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if data.empty:
                raise RuntimeError("empty frame returned")
            return _extract_close(data, tickers)
        except Exception as exc:
            last_err = exc
            print(f"    retry {attempt}/{MAX_RETRIES} for chunk {tickers[0]}..{tickers[-1]}: {exc}")
            time.sleep(RETRY_SLEEP)
    raise RuntimeError(f"yf.download failed after {MAX_RETRIES} retries: {last_err}")


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    n_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(tickers), CHUNK_SIZE):
        batch = tickers[i : i + CHUNK_SIZE]
        idx = i // CHUNK_SIZE + 1
        print(f"  chunk {idx}/{n_chunks} ({len(batch)} tickers): {batch[0]}..{batch[-1]}")
        chunks.append(_download_chunk(batch))
    prices = pd.concat(chunks, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices


def _validate(prices: pd.DataFrame) -> None:
    if prices.empty:
        raise RuntimeError("yfinance returned an empty frame")
    actual_start = prices.index.min()
    actual_end = prices.index.max()
    expected_start = pd.Timestamp(START)
    expected_end = pd.Timestamp(END) - pd.Timedelta(days=1)
    if actual_start > expected_start + pd.Timedelta(days=14):
        raise RuntimeError(
            f"truncated start: data begins {actual_start.date()}, expected near {expected_start.date()}"
        )
    if actual_end < expected_end - pd.Timedelta(days=14):
        raise RuntimeError(
            f"truncated end: data ends {actual_end.date()}, expected near {expected_end.date()}"
        )
    nonempty_cols = prices.notna().any(axis=0).sum()
    if nonempty_cols < 0.5 * prices.shape[1]:
        raise RuntimeError(
            f"only {nonempty_cols}/{prices.shape[1]} tickers returned any data — likely a bulk failure"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching S&P 500 metadata from Slickcharts...")
    meta = fetch_sp500_metadata()
    print(f"  {len(meta)} tickers")
    meta.to_csv(OUTPUT_DIR / "sectors.csv", index=False)
    print(f"  -> {OUTPUT_DIR / 'sectors.csv'}")

    print(f"Fetching prices {START}..{END} from Yahoo Finance (chunks of {CHUNK_SIZE})...")
    prices = fetch_prices(meta["Ticker"].tolist())
    prices.index.name = "Date"
    print(
        f"  shape: {prices.shape[0]} days x {prices.shape[1]} tickers "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )
    _validate(prices)
    prices.to_csv(OUTPUT_DIR / "stockdata.csv")
    print(f"  -> {OUTPUT_DIR / 'stockdata.csv'}")


if __name__ == "__main__":
    main()
