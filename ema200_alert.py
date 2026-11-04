#!/usr/bin/env python3
"""200 EMA cross alert (daily) for NSE stocks -> Telegram.

Alerts when a stock's daily CLOSE crosses the 200-day EMA:
  🟢 closes ABOVE the 200 EMA (previous close was at/below it)
  🔴 closes BELOW the 200 EMA (previous close was at/above it)

Files:  watchlist.txt  (one NSE symbol per line, e.g. RELIANCE; '#' = comment)
        ema200_state.json (alerts already sent, so nothing is sent twice)
Env:    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Completely independent of any other bot or repo.
"""
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
WATCHLIST_FILE = "watchlist.txt"
STATE_FILE = "ema200_state.json"

EMA_PERIOD = 200
MIN_BARS = EMA_PERIOD + 5      # need enough history for a stable EMA
HISTORY_PERIOD = "3y"          # ~750 trading days
BATCH_SIZE = 40
IST = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE_IST = (15, 45)    # before this, today's bar is incomplete and is ignored


def load_watchlist():
    symbols = []
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            s = line.split("#", 1)[0].strip().upper()
            if not s:
                continue
            if s.endswith(".NS"):
                s = s[:-3]
            symbols.append(s)
    return list(dict.fromkeys(symbols))  # de-duplicate, keep order


def load_state():
    """Return dict of sent keys, or None if the state file doesn't exist yet."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(map(str, data)) if isinstance(data, list) else set()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read {STATE_FILE}: {exc}")
        return set()


def save_state(seen):
    # keep the file small: only the newest 5000 keys (keys start with the date)
    keep = sorted(seen)[-5000:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=1)


def drop_incomplete_bar(close):
    """If today's candle is still forming (market open), ignore it."""
    now = datetime.now(IST)
    if close.empty:
        return close
    last_date = close.index[-1].date()
    before_close = (now.hour, now.minute) < MARKET_CLOSE_IST
    if last_date == now.date() and before_close:
        return close.iloc[:-1]
    return close


def detect_cross(close):
    """Return (direction, date, last_close, last_ema) or None.

    direction is 'ABOVE' or 'BELOW' when the latest completed daily close
    crossed the EMA relative to the previous close.
    """
    close = close.dropna()
    if len(close) < MIN_BARS:
        return None
    ema = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    prev_c, last_c = float(close.iloc[-2]), float(close.iloc[-1])
    prev_e, last_e = float(ema.iloc[-2]), float(ema.iloc[-1])
    if prev_c <= prev_e and last_c > last_e:
        direction = "ABOVE"
    elif prev_c >= prev_e and last_c < last_e:
        direction = "BELOW"
    else:
        return None
    return direction, close.index[-1].date(), last_c, last_e


def fetch_closes(symbols):
    """Download daily closes for all symbols in batches. Returns {symbol: Series}."""
    out = {}
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        tickers = [f"{s}.NS" for s in batch]
        data = None
        for attempt in range(1, 4):
            try:
                data = yf.download(
                    tickers,
                    period=HISTORY_PERIOD,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if data is not None and not data.empty:
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"Batch {i // BATCH_SIZE + 1} attempt {attempt} failed: {exc}")
            time.sleep(5 * attempt)
        if data is None or data.empty:
            print(f"Batch {i // BATCH_SIZE + 1}: no data")
            continue
        for sym, tk in zip(batch, tickers):
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    series = data[tk]["Close"]
                else:  # single ticker in the batch
                    series = data["Close"]
                series = series.dropna()
                if not series.empty:
                    out[sym] = series
            except KeyError:
                continue
    return out


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


def build_message(direction, cross_date, items):
    if direction == "ABOVE":
        head = f"🟢 200 EMA CROSS UP (Daily) — {len(items)} stock(s)"
        note = "Closed ABOVE the 200 EMA"
    else:
        head = f"🔴 200 EMA CROSS DOWN (Daily) — {len(items)} stock(s)"
        note = "Closed BELOW the 200 EMA"
    lines = [head, f"📅 {cross_date.strftime('%d-%b-%Y')} · {note}", ""]
    for sym, close, ema in items:
        gap = (close / ema - 1) * 100
        lines.append(f"📊 {sym}: ₹{close:,.2f} (EMA200 ₹{ema:,.2f}, {gap:+.2f}%)")
    return "\n".join(lines)


def main():
    symbols = load_watchlist()
    if not symbols:
        print("watchlist.txt is empty.")
        return 1
    print(f"Watchlist: {len(symbols)} symbols")

    if datetime.now(IST).weekday() >= 5:
        print("Weekend: markets closed, nothing to do.")
        return 0

    closes = fetch_closes(symbols)
    print(f"Got price data for {len(closes)}/{len(symbols)} symbols")
    if len(closes) < max(1, len(symbols) // 2):
        print("Too much data missing: treating this run as failed.")
        return 1

    seen = load_state()
    first_run = seen is None
    if first_run:
        seen = set()
        print("No state file yet: recording current crosses without sending.")

    groups = {"ABOVE": {}, "BELOW": {}}  # direction -> {date: [(sym, close, ema)]}
    skipped = []
    for sym, series in closes.items():
        series = drop_incomplete_bar(series)
        res = detect_cross(series)
        if len(series.dropna()) < MIN_BARS:
            skipped.append(sym)
        if not res:
            continue
        direction, cdate, last_c, last_e = res
        key = f"{cdate.isoformat()}|{sym}|{direction}"
        if key in seen:
            continue
        seen.add(key)
        if not first_run:
            groups[direction].setdefault(cdate, []).append((sym, last_c, last_e))

    if skipped:
        print(f"Not enough history for EMA200 (skipped): {', '.join(sorted(skipped))}")

    failed = 0
    sent = 0
    for direction in ("ABOVE", "BELOW"):
        for cdate, items in sorted(groups[direction].items()):
            items.sort()
            # Telegram limit is ~4096 chars: send in chunks of 30 stocks
            for j in range(0, len(items), 30):
                chunk = items[j : j + 30]
                if send_telegram(build_message(direction, cdate, chunk)):
                    sent += 1
                else:
                    failed += 1
                    # un-mark so it is retried on the next run
                    for sym, _, _ in chunk:
                        seen.discard(f"{cdate.isoformat()}|{sym}|{direction}")
                time.sleep(1.2)

    save_state(seen)
    print(f"Done. Messages sent: {sent}, failed: {failed}, state entries: {len(seen)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
