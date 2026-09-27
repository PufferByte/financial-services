"""
Earnings Reviewer — historical backtest add-on (free data, yfinance only)

Question: does the tool's post-earnings signal (EPS beat vs. miss) actually
precede positive/negative stock performance over the following 1 week,
1 month, and 3 months?

Method:
- Pull full earnings history (EPS Estimate / Reported EPS / Surprise%) per ticker.
- Pull daily close-price history for the same span.
- For each *past* reported quarter, anchor the "reaction price" at the close
  of the first trading day on/after the announcement date (all recent prints
  for these three names are after-market, ~4pm ET, so the market's reaction
  shows up the next session).
- Measure forward return from that reaction close at +1 week (5 trading days),
  +1 month (21 trading days), +3 months (63 trading days).
- Signal = BEAT if Surprise% > 0 else MISS. "Hit" = BEAT followed by a
  positive forward return (or MISS followed by a negative one) at that horizon.
- Skip horizons that haven't happened yet (not enough trading days between
  the reaction date and today).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

TICKERS = ["AAPL", "MSFT", "NVDA"]
HORIZONS = [("1wk", 5), ("1mo", 21), ("3mo", 63)]  # trading days

def get_reaction_and_forward(ticker):
    t = yf.Ticker(ticker)
    ed = t.earnings_dates.dropna(subset=["Reported EPS", "Surprise(%)"]).copy()
    ed.index = ed.index.tz_localize(None)
    ed = ed.sort_index()

    earliest = ed.index.min() - pd.Timedelta(days=5)
    hist = t.history(start=earliest.date().isoformat(), end=None, auto_adjust=True)
    hist.index = hist.index.tz_localize(None)
    closes = hist["Close"]
    trading_days = closes.index

    rows = []
    for edate, r in ed.iterrows():
        # first trading day on/after the announcement date = reaction day
        pos = trading_days.searchsorted(edate)
        if pos >= len(trading_days):
            continue
        reaction_date = trading_days[pos]
        base_price = closes.iloc[pos]

        row = {
            "ticker": ticker,
            "earnings_date": edate.date(),
            "reaction_date": reaction_date.date(),
            "eps_est": r["EPS Estimate"],
            "eps_rep": r["Reported EPS"],
            "surprise_pct": r["Surprise(%)"],
            "signal": "BEAT" if r["Surprise(%)"] > 0 else "MISS",
            "base_price": base_price,
        }
        for label, ndays in HORIZONS:
            fpos = pos + ndays
            if fpos < len(trading_days):
                fwd_price = closes.iloc[fpos]
                row[f"ret_{label}"] = (fwd_price - base_price) / base_price * 100
            else:
                row[f"ret_{label}"] = None  # not enough future data yet
        rows.append(row)
    return pd.DataFrame(rows)

all_rows = []
for tk in TICKERS:
    df = get_reaction_and_forward(tk)
    all_rows.append(df)
big = pd.concat(all_rows, ignore_index=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

print("=" * 100)
print("  PER-QUARTER DETAIL")
print("=" * 100)
show_cols = ["ticker","earnings_date","signal","surprise_pct","base_price","ret_1wk","ret_1mo","ret_3mo"]
detail = big[show_cols].copy()
for c in ["surprise_pct","ret_1wk","ret_1mo","ret_3mo"]:
    detail[c] = detail[c].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "pending")
detail["base_price"] = detail["base_price"].map(lambda v: f"${v:.2f}")
print(detail.to_string(index=False))

print()
print("=" * 100)
print("  SIGNAL PERFORMANCE — does BEAT/MISS predict the direction of the move?")
print("=" * 100)
for label, _ in HORIZONS:
    col = f"ret_{label}"
    sub = big.dropna(subset=[col])
    if sub.empty:
        continue
    print(f"\n--- Horizon: {label} (n={len(sub)} quarters with data) ---")
    for sig in ["BEAT", "MISS"]:
        g = sub[sub["signal"] == sig]
        if g.empty:
            continue
        avg_ret = g[col].mean()
        if sig == "BEAT":
            hit = (g[col] > 0).mean() * 100
        else:
            hit = (g[col] < 0).mean() * 100
        print(f"  {sig:<5} n={len(g):<3}  avg return {avg_ret:+6.2f}%   "
              f"'hit rate' (moved in predicted direction) {hit:5.1f}%")
    overall = sub[col].mean()
    print(f"  ALL   n={len(sub):<3}  avg return {overall:+6.2f}%  (unconditional, for reference)")

print()
print("=" * 100)
print("  PER-TICKER SUMMARY")
print("=" * 100)
for tk in TICKERS:
    sub_t = big[big["ticker"] == tk]
    print(f"\n{tk}:")
    for label, _ in HORIZONS:
        col = f"ret_{label}"
        g = sub_t.dropna(subset=[col])
        beat = g[g.signal=="BEAT"]
        miss = g[g.signal=="MISS"]
        beat_hit = (beat[col] > 0).mean()*100 if len(beat) else float("nan")
        print(f"  {label:<5} BEAT n={len(beat):<2} avg={beat[col].mean() if len(beat) else float('nan'):+6.2f}%  hit={beat_hit:5.1f}%   "
              f"MISS n={len(miss):<2} avg={miss[col].mean() if len(miss) else float('nan'):+6.2f}%")
