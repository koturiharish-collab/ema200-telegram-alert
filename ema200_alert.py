#!/usr/bin/env python3
"""LIVE 200 EMA cross alert for NSE stocks -> Telegram.

Runs every 15 minutes during Indian market hours (Mon-Fri, 09:15-15:30 IST).
For every NSE stock it compares the live price with the daily 200 EMA:
  🟢 price moves ABOVE the 200 EMA  -> alert
  🔴 price moves BELOW the 200 EMA  -> alert
and alerts again on every later flip during the day.

How it stays fast:
  * First run of each trading day: downloads ~2 years of history for all stocks and
    stores each stock's 200 EMA (as of yesterday's close) in ema200_state.json.
  * Every later run: downloads only the last few days (live price) and updates the
    EMA with today's live price.

Files:  watchlist.txt  ("ALL" = every NSE equity, or your own symbols one per line)
        ema200_state.json (EMA values and which side of the EMA each stock is on)
Env:    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, FORCE_RUN (true = ignore market hours)
Completely independent of any other bot or repo.
"""
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
FORCE = os.environ.get("FORCE_RUN", "").strip().lower() == "true"
WATCHLIST_FILE = "watchlist.txt"
STATE_FILE = "ema200_state.json"

EMA_PERIOD = 200
ALPHA = 2.0 / (EMA_PERIOD + 1)
MIN_BARS = EMA_PERIOD + 5          # need enough history for a stable EMA
HISTORY_PERIOD = "2y"              # ~500 trading days
LIVE_PERIOD = "5d"
BATCH_SIZE = 100
CHUNK_SIZE = 40                    # stocks per Telegram message
BAND = 0.001                       # 0.1%: price must be this far past the EMA to count as a cross
MIN_PRICE = 20.0                   # skip penny stocks (0 = off)
MIN_AVG_VOLUME = 50000             # skip thinly traded stocks, 20-day avg volume (0 = off)
REF_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]  # used to read the trading calendar
NSE_EQUITY_LIST = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = (9, 15)
MARKET_END = (15, 45)              # a little after the 15:30 close, to allow for delayed runs


# ----------------------------------------------------------------------------- symbols
def fetch_all_nse_symbols():
    """Download the official list of all NSE-listed equities (series EQ)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    last_err = None
    for attempt in range(1, 5):
        try:
            resp = requests.get(NSE_EQUITY_LIST, headers=headers, timeout=40)
            resp.raise_for_status()
            rows = csv.DictReader(io.StringIO(resp.text))
            symbols = []
            for row in rows:
                row = {(k or "").strip().upper(): (v or "").strip() for k, v in row.items()}
                if row.get("SERIES") == "EQ" and row.get("SYMBOL"):
                    symbols.append(row["SYMBOL"].upper())
            if len(symbols) < 500:
                raise ValueError(f"only {len(symbols)} symbols in NSE list")
            print(f"Downloaded NSE equity list: {len(symbols)} symbols")
            return list(dict.fromkeys(symbols))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"NSE list attempt {attempt} failed: {exc}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download the NSE equity list: {last_err}")


def load_watchlist():
    """watchlist.txt containing only the word ALL = every NSE equity (auto-downloaded)."""
    symbols = []
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            s = line.split("#", 1)[0].strip().upper()
            if not s:
                continue
            if s.endswith(".NS"):
                s = s[:-3]
            symbols.append(s)
    if symbols == ["ALL"]:
        return fetch_all_nse_symbols()
    return list(dict.fromkeys(symbols))


# ----------------------------------------------------------------------------- state
def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}  # old list-format file -> start fresh
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read {STATE_FILE}: {exc}")
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, sort_keys=True, separators=(",", ":"))


# ----------------------------------------------------------------------------- prices
def download(symbols, period):
    """Download daily Open/High/Low/Close/Volume in batches. Returns {symbol: DataFrame(Close, Volume)}."""
    out = {}
    nbatches = max(1, (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE)
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        tickers = [f"{s}.NS" for s in batch]
        n = i // BATCH_SIZE + 1
        data = None
        for attempt in range(1, 4):
            try:
                data = yf.download(
                    tickers,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                )
                if data is not None and not data.empty:
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"Batch {n}/{nbatches} ({period}) attempt {attempt} failed: {exc}")
            time.sleep(3 * attempt)
        if data is None or data.empty:
            print(f"Batch {n}/{nbatches} ({period}): no data")
            continue
        for sym, tk in zip(batch, tickers):
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    df = data[tk][["Close", "Volume"]]
                else:  # single ticker in the batch
                    df = data[["Close", "Volume"]]
                df = df.dropna(subset=["Close"])
                if not df.empty:
                    out[sym] = df
            except KeyError:
                continue
        time.sleep(1)
    return out


def is_liquid(done):
    """done = completed daily bars. Skip penny / thinly traded stocks (they flip constantly)."""
    if MIN_PRICE and float(done["Close"].iloc[-1]) < MIN_PRICE:
        return False
    if MIN_AVG_VOLUME:
        avg_vol = float(done["Volume"].tail(20).mean())
        if not avg_vol >= MIN_AVG_VOLUME:
            return False
    return True


def rebuild_state(symbols, today_ref, last_done, old_state):
    """Once per trading day: compute each stock's EMA200 as of the last completed close."""
    print(f"Rebuilding EMA200 for {len(symbols)} symbols (as of {last_done})")
    hist = download(symbols, HISTORY_PERIOD)
    old_side = old_state.get("side", {}) if isinstance(old_state, dict) else {}
    ema_map, side_map = {}, {}
    short = 0
    for sym, df in hist.items():
        mask = [d < today_ref for d in df.index.date]
        done = df.loc[mask]
        if len(done) < MIN_BARS:
            short += 1
            continue
        if not is_liquid(done):
            continue
        close = done["Close"]
        ema = close.ewm(span=EMA_PERIOD, adjust=False).mean()
        ema_map[sym] = round(float(ema.iloc[-1]), 4)
        computed = "ABOVE" if float(close.iloc[-1]) > float(ema.iloc[-1]) else "BELOW"
        side_map[sym] = old_side.get(sym, computed)
    print(
        f"History for {len(hist)}/{len(symbols)} symbols; tracking {len(ema_map)} liquid stocks "
        f"(skipped {short} with too little history)"
    )
    return {"asof": last_done.isoformat(), "fails": 0, "ema": ema_map, "side": side_map}


# ----------------------------------------------------------------------------- telegram
def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        print("Telegram secrets missing, would have sent:\n" + text)
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": "true"},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"Telegram error {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Telegram request failed: {exc}")
        return False


def build_message(direction, when, items):
    if direction == "ABOVE":
        head = f"🟢 200 EMA CROSSED UP · {when} IST · {len(items)} stock(s)"
    else:
        head = f"🔴 200 EMA CROSSED DOWN · {when} IST · {len(items)} stock(s)"
    lines = [head, "(daily 200 EMA vs live price)", ""]
    for sym, price, ema in items:
        gap = (price / ema - 1) * 100
        lines.append(f"📊 {sym}: ₹{price:,.2f} (EMA200 ₹{ema:,.2f}, {gap:+.2f}%)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main():
    now = datetime.now(IST)
    if not FORCE:
        if now.weekday() >= 5:
            print("Weekend: market closed.")
            return 0
        hm = (now.hour, now.minute)
        if hm < MARKET_OPEN or hm > MARKET_END:
            print(f"{now:%H:%M} IST is outside market hours (09:15-15:30): nothing to do.")
            return 0

    symbols = load_watchlist()
    if not symbols:
        print("watchlist.txt is empty.")
        return 1
    print(f"Watchlist: {len(symbols)} symbols")

    state = load_state()

    # Trading calendar from a few big stocks: is the market open today, and what was the last completed day?
    cal = download(REF_SYMBOLS, LIVE_PERIOD)
    dates = sorted({d for df in cal.values() for d in df.index.date})
    if not dates:
        print("Could not read any price data (Yahoo Finance down?).")
        return _record_failure(state)
    if FORCE:
        today_ref = dates[-1]
        print(f"FORCE_RUN: treating {today_ref} as today.")
    else:
        today_ref = now.date()
        if dates[-1] != today_ref:
            print(f"No price bar for {today_ref}: market holiday or not open yet. Nothing to do.")
            return 0
    done_dates = [d for d in dates if d < today_ref]
    if not done_dates:
        print("Could not find the previous trading day.")
        return _record_failure(state)
    last_done = done_dates[-1]

    # Phase 1: once per trading day, rebuild the EMA table.
    if state.get("asof") != last_done.isoformat() or not state.get("ema"):
        state = rebuild_state(symbols, today_ref, last_done, state)
        if not state["ema"]:
            print("EMA rebuild produced no stocks.")
            return 1
        save_state(state)

    # Phase 2: live price check.
    tracked = list(state["ema"].keys())
    live = download(tracked, LIVE_PERIOD)
    got = 0
    changes = {"ABOVE": [], "BELOW": []}
    old_sides = {}
    for sym, df in live.items():
        if df.index[-1].date() != today_ref:
            continue  # no trade yet today (or halted)
        got += 1
        price = float(df["Close"].iloc[-1])
        ema_live = ALPHA * price + (1 - ALPHA) * state["ema"][sym]
        prev_side = state["side"].get(sym)
        if price > ema_live * (1 + BAND):
            new_side = "ABOVE"
        elif price < ema_live * (1 - BAND):
            new_side = "BELOW"
        else:
            new_side = prev_side or ("ABOVE" if price > ema_live else "BELOW")
        if prev_side is None:
            state["side"][sym] = new_side
            continue
        if new_side != prev_side:
            old_sides[sym] = prev_side
            state["side"][sym] = new_side
            changes[new_side].append((sym, price, ema_live))
    print(f"Live prices for {got}/{len(tracked)} tracked stocks")

    if got < max(1, len(tracked) // 2):
        print("Too much live data missing.")
        return _record_failure(state)
    state["fails"] = 0

    when = now.strftime("%H:%M")
    sent = failed = 0
    for direction in ("ABOVE", "BELOW"):
        items = sorted(changes[direction])
        for j in range(0, len(items), CHUNK_SIZE):
            chunk = items[j : j + CHUNK_SIZE]
            if send_telegram(build_message(direction, when, chunk)):
                sent += 1
            else:
                failed += 1
                for sym, _, _ in chunk:  # retry on the next run
                    state["side"][sym] = old_sides[sym]
            time.sleep(1.2)

    save_state(state)
    crossed = len(changes["ABOVE"]) + len(changes["BELOW"])
    print(f"Done. Crosses: {crossed} (up {len(changes['ABOVE'])}, down {len(changes['BELOW'])}); "
          f"messages sent: {sent}, failed: {failed}")
    return 1 if failed else 0


def _record_failure(state):
    """Only raise the alarm (exit 1) after 2 failed runs in a row, so one Yahoo hiccup stays quiet."""
    fails = int(state.get("fails", 0)) + 1
    state["fails"] = fails
    if state.get("ema"):
        save_state(state)
    print(f"Failure count: {fails}")
    return 1 if fails >= 2 else 0


if __name__ == "__main__":
    sys.exit(main())
