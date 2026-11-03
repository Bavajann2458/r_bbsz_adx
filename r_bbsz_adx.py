"""
Coiled-Spring Reversal Scanner (Top 30 Coins)
Binance Public Data -> Telegram Alerts

Setup (All 3 conditions true on a CLOSED candle):
  1. Strong trend: ADX (14) >= 30
  2. Extreme RSI (14): RSI >= 90 (SHORT) or RSI <= 10 (LONG)
  3. Tight Squeeze: High-Low range of last 5 candles <= 2.0 x ATR(14)
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import requests

# ==================== CONFIG (TUNE THESE EASILY) ====================
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "NEARUSDT", "SUIUSDT", "APTUSDT", "MATICUSDT", "LTCUSDT",
    "BCHUSDT", "UNIUSDT", "ICPUSDT", "FETUSDT", "SHIBUSDT",
    "PEPEUSDT", "FLOKIUSDT", "STXUSDT", "RNDRUSDT", "INJUSDT",
    "TIAUSDT", "FILUSDT", "ATOMUSDT", "SEIUSDT", "ARBUSDT"
]

TIMEFRAMES = [ "5m", "15m", "30m", "45m", "1h", "2h", "4h", "1d"]

RSI_PERIOD = 14
RSI_HIGH = 90
RSI_LOW = 10

ADX_PERIOD = 14
ADX_MIN = 30

SQUEEZE_N = 5            # Number of recent candles that must be tight
SQUEEZE_ATR_MULT = 2.0   # Range <= SQUEEZE_ATR_MULT * ATR

VOLUME_FILTER = False    # Require volume below 20-candle MA
BACKTEST_CANDLES = 2000
HORIZONS = [3, 5, 10]
# ====================================================================

# Binance Public Vision Data API (Bypasses API key requirements)
KLINE_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch(symbol, interval, total=500):
    """Fetch closed candle data from Binance public endpoint."""
    rows, end = [], None
    while len(rows) < total:
        params = {"symbol": symbol, "interval": interval, "limit": min(1000, total - len(rows))}
        if end is not None:
            params["endTime"] = end
        try:
            r = requests.get(KLINE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException:
            # Silence per-request network failure to prevent terminal spam on local ISP block
            break

        if not data:
            break
        rows = data + rows
        end = data[0][0] - 1
        if len(data) < params["limit"]:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms].reset_index(drop=True)  # Drop forming candle
    return df


def wilder(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def add_indicators(df):
    if df.empty or len(df) < max(RSI_PERIOD, ADX_PERIOD, SQUEEZE_N) + 5:
        df["signal"] = 0
        return df

    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]

    # RSI Calculation
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder(gain, RSI_PERIOD)
    avg_loss = wilder(loss, RSI_PERIOD)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).where(avg_loss != 0, 100.0)

    # ADX & ATR Calculation
    prev_close = close.shift()
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = wilder(tr, ADX_PERIOD)
    plus_di = 100 * wilder(plus_dm, ADX_PERIOD) / atr.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, ADX_PERIOD) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = wilder(dx.fillna(0), ADX_PERIOD)
    df["atr"] = atr

    # Squeeze Filter
    rng = high.rolling(SQUEEZE_N).max() - low.rolling(SQUEEZE_N).min()
    df["squeeze"] = rng <= SQUEEZE_ATR_MULT * atr

    # Signal Rules
    base = (df["adx"] >= ADX_MIN) & df["squeeze"]
    if VOLUME_FILTER:
        base &= df["volume"] < df["volume"].rolling(20).mean()

    df["signal"] = 0
    df.loc[base & (df["rsi"] >= RSI_HIGH), "signal"] = -1  # SHORT
    df.loc[base & (df["rsi"] <= RSI_LOW), "signal"] = 1   # LONG
    return df


def send(msg):
    print(msg, flush=True)
    token = os.getenv("TELEGRAM_TOKEN", "8307615821:AAHJHhSsRhBCzORo-5MoQi1gA05Dznoaksk")
    chat = os.getenv("TELEGRAM_CHAT_ID", "8826743783")
    
    if not token or not chat:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat, "text": msg}, timeout=15)
    except requests.RequestException as e:
        print("Telegram error:", e)


def scan(fresh_seconds, sent):
    now = time.time()
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                raw_df = fetch(sym, tf, 500)
                if raw_df.empty:
                    continue
                df = add_indicators(raw_df)
            except Exception:
                continue

            if len(df) < 60:
                continue

            last = df.iloc[-1]
            if last["signal"] == 0:
                continue

            age = now - last["close_time"] / 1000
            key = (sym, tf, int(last["close_time"]))
            if age > fresh_seconds or key in sent:
                continue

            sent.add(key)
            side = "LONG" if last["signal"] == 1 else "SHORT"
            send(f"🚨 {side} SETUP | {sym} [{tf}]\n"
                 f"Price: {last['close']:.4f}\n"
                 f"RSI: {last['rsi']:.1f} | ADX: {last['adx']:.1f}\n"
                 f"Tight squeeze confirmed ({SQUEEZE_N} candles)")


def backtest():
    print(f"Starting backtest across {len(SYMBOLS)} coins...")
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                raw_df = fetch(sym, tf, BACKTEST_CANDLES)
                if raw_df.empty:
                    continue
                df = add_indicators(raw_df)
            except Exception as e:
                print(f"{sym} {tf} backtest error: {e}")
                continue

            closes, sig = df["close"].values, df["signal"].values
            idxs, last_i = [], -10**9
            for i in np.nonzero(sig)[0]:
                if i - last_i >= max(HORIZONS):
                    idxs.append(i)
                    last_i = i

            if len(idxs) > 0:
                line = f"{sym} {tf}: {len(idxs)} signals"
                for h in HORIZONS:
                    valid = [i for i in idxs if i + h < len(df)]
                    if valid:
                        hits = sum(1 for i in valid if (closes[i + h] - closes[i]) * sig[i] > 0)
                        line += f" | +{h}c right {hits / len(valid) * 100:.0f}% ({len(valid)})"
                print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()

    if args.backtest:
        backtest()
    elif args.once:
        scan(fresh_seconds=300, sent=set())
    else:
        sent = set()
        while True:
            scan(fresh_seconds=120, sent=sent)
            time.sleep(30)
