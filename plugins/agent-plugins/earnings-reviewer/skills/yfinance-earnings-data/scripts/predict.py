"""
Earnings Reviewer — historical backtest add-on (free data, yfinance only)

Question: does the tool's post-earnings signal (EPS beat / inline / miss)
line up with the stock's *earnings reaction* and the moves that follow?

Return definition (the prediction target):
- pre_close  = the last regular-session close BEFORE the announcement became
  public. After-market (>= 16:00 ET) print -> that day's close; pre-market /
  intraday print -> the previous trading day's close.
- post_close = the first regular-session close AFTER the announcement, i.e.
  the next trading day after pre_close.
- ret_1d  = post_close / pre_close - 1            (the earnings reaction itself)
- ret_1wk / ret_1mo / ret_3mo = close 5 / 21 / 63 trading days after
  pre_close, over pre_close. These INCLUDE the day-1 reaction.
- Every return is also reported in excess of the benchmark (default SPY)
  over the same window.

Timing: Yahoo's earnings timestamps carry a time of day (ET). A timestamp at
exactly 00:00 has no usable time, so we can't tell pre- from after-market;
those quarters are labelled UNKNOWN and excluded from the stats (they are
still shown in the detail table). Pass --assume-unknown bmo|amc to force them.

Signal = BEAT if Surprise% > +band, MISS if < -band, else INLINE (default
band 0, i.e. only an exact match is INLINE). "Hit" = BEAT followed by a
positive return, or MISS by a negative one. INLINE makes no directional call.

The hit rate is compared with the base rate: the share of ALL quarters with
a positive (or negative) return at that horizon. A BEAT hit rate of 75% means
nothing if the stock rose after 75% of all prints anyway.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings, argparse, sys
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--tickers", default="AAPL,MSFT,NVDA",
                     help="Comma-separated tickers to backtest, e.g. --tickers TSLA,GOOGL")
parser.add_argument("--benchmark", default="SPY",
                    help="Benchmark for excess returns (default SPY); pass '' to disable")
parser.add_argument("--limit", type=int, default=100,
                    help="Max earnings dates to request per ticker (yfinance default is only 12)")
parser.add_argument("--inline-band", type=float, default=0.0,
                    help="|Surprise%%| <= this is INLINE (default 0 = exact match only)")
parser.add_argument("--assume-unknown", choices=["skip", "bmo", "amc"], default="skip",
                    help="How to treat earnings timestamps with no time of day (00:00)")
args, _ = parser.parse_known_args()

TICKERS = [tk.strip().upper() for tk in args.tickers.split(",") if tk.strip()]
BENCH = args.benchmark.strip().upper()
HORIZONS = [("1d", 1), ("1wk", 5), ("1mo", 21), ("3mo", 63)]  # trading days after pre_close
MARKET_CLOSE = pd.Timedelta(hours=16)


def to_ny_naive(idx):
    """Express a DatetimeIndex as naive New York wall-clock time."""
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return idx


def classify(surprise):
    if surprise > args.inline_band:
        return "BEAT"
    if surprise < -args.inline_band:
        return "MISS"
    return "INLINE"


def load_closes(ticker, start):
    hist = yf.Ticker(ticker).history(start=start, end=None, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist:
        return None
    closes = hist["Close"].dropna()
    closes.index = to_ny_naive(closes.index).normalize()
    return closes


def get_reaction_and_forward(ticker, bench_closes):
    t = yf.Ticker(ticker)
    ed = t.get_earnings_dates(limit=args.limit)
    if ed is None or ed.empty:
        raise ValueError("no earnings dates returned")
    ed = ed.dropna(subset=["Reported EPS", "Surprise(%)"]).copy()
    if ed.empty:
        raise ValueError("no reported quarters with a surprise figure")
    ed.index = to_ny_naive(ed.index)
    ed = ed[~ed.index.duplicated()].sort_index()

    closes = load_closes(ticker, (ed.index.min() - pd.Timedelta(days=10)).date().isoformat())
    if closes is None:
        raise ValueError("no price history returned")
    trading_days = closes.index
    close_times = trading_days + MARKET_CLOSE

    rows = []
    for edate, r in ed.iterrows():
        has_time = edate != edate.normalize()
        if has_time:
            timing = "AMC" if edate >= edate.normalize() + MARKET_CLOSE else "BMO/DMH"
            ann = edate
        else:
            timing = "UNKNOWN"
            if args.assume_unknown == "amc":
                ann = edate + MARKET_CLOSE
            elif args.assume_unknown == "bmo":
                ann = edate
            else:
                ann = None

        row = {
            "ticker": ticker,
            "earnings_ts": edate,
            "timing": timing,
            "eps_est": r["EPS Estimate"],
            "eps_rep": r["Reported EPS"],
            "surprise_pct": r["Surprise(%)"],
            "signal": classify(r["Surprise(%)"]),
            "pre_date": None, "post_date": None, "pre_close": np.nan,
        }
        for label, _ in HORIZONS:
            row[f"ret_{label}"] = np.nan
            row[f"xret_{label}"] = np.nan

        if ann is not None:
            # last session whose 16:00 close is at or before the announcement
            pre_pos = close_times.searchsorted(ann, side="right") - 1
            if 0 <= pre_pos < len(trading_days) - 1:
                pre_close = closes.iloc[pre_pos]
                row["pre_date"] = trading_days[pre_pos].date()
                row["post_date"] = trading_days[pre_pos + 1].date()
                row["pre_close"] = pre_close
                for label, ndays in HORIZONS:
                    fpos = pre_pos + ndays
                    if fpos >= len(trading_days):
                        continue  # horizon hasn't happened yet
                    ret = (closes.iloc[fpos] / pre_close - 1) * 100
                    row[f"ret_{label}"] = ret
                    if bench_closes is not None:
                        d0, d1 = trading_days[pre_pos], trading_days[fpos]
                        if d0 in bench_closes.index and d1 in bench_closes.index:
                            bret = (bench_closes[d1] / bench_closes[d0] - 1) * 100
                            row[f"xret_{label}"] = ret - bret
        rows.append(row)
    return pd.DataFrame(rows)


bench_closes = None
if BENCH:
    bench_closes = load_closes(BENCH, "1995-01-01")
    if bench_closes is None:
        print(f"[WARN] could not load benchmark {BENCH}; excess returns disabled")

all_rows = []
for tk in TICKERS:
    try:
        all_rows.append(get_reaction_and_forward(tk, bench_closes))
    except Exception as e:
        print(f"[SKIP] {tk}: {e}")
if not all_rows:
    print("[UNSOURCED] no usable earnings history for any requested ticker.")
    sys.exit(1)
big = pd.concat(all_rows, ignore_index=True)
usable = big[big["timing"] != "UNKNOWN"] if args.assume_unknown == "skip" else big

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 500)

print("=" * 110)
print("  PER-QUARTER DETAIL   (returns measured from the last close BEFORE the print)")
print("=" * 110)
show_cols = ["ticker", "earnings_ts", "timing", "signal", "surprise_pct", "pre_date",
             "pre_close", "ret_1d", "xret_1d", "ret_1wk", "ret_1mo", "ret_3mo"]
detail = big[show_cols].copy()
for c in ["surprise_pct", "ret_1d", "xret_1d", "ret_1wk", "ret_1mo", "ret_3mo"]:
    detail[c] = detail[c].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "-")
detail["pre_close"] = detail["pre_close"].map(lambda v: f"${v:.2f}" if pd.notna(v) else "-")
detail["earnings_ts"] = detail["earnings_ts"].map(lambda v: v.strftime("%Y-%m-%d %H:%M"))
print(detail.to_string(index=False))
n_unknown = (big["timing"] == "UNKNOWN").sum()
if n_unknown and args.assume_unknown == "skip":
    print(f"\n[NOTE] {n_unknown} quarter(s) have no time of day -> excluded from stats "
          f"(use --assume-unknown bmo|amc to include).")


def signal_stats(sub, col):
    """Print per-signal avg / hit rate vs. base rate for one return column."""
    base_up = (sub[col] > 0).mean() * 100
    for sig in ["BEAT", "INLINE", "MISS"]:
        g = sub[sub["signal"] == sig]
        if g.empty:
            continue
        avg = g[col].mean()
        if sig == "BEAT":
            hit, base = (g[col] > 0).mean() * 100, base_up
        elif sig == "MISS":
            hit, base = (g[col] < 0).mean() * 100, 100 - base_up
        else:
            print(f"  {sig:<6} n={len(g):<3} avg {avg:+6.2f}%   (no directional call)")
            continue
        small = "  [n<10: not meaningful]" if len(g) < 10 else ""
        print(f"  {sig:<6} n={len(g):<3} avg {avg:+6.2f}%   hit {hit:5.1f}%  "
              f"vs base rate {base:5.1f}%  -> edge {hit - base:+5.1f}pp{small}")
    print(f"  ALL    n={len(sub):<3} avg {sub[col].mean():+6.2f}%   "
          f"share positive {base_up:5.1f}%  (unconditional, for reference)")


print()
print("=" * 110)
print("  SIGNAL PERFORMANCE — does BEAT/MISS predict the direction of the move?")
print("  hit rate is only informative relative to the base rate (edge = hit - base).")
print("=" * 110)
for label, _ in HORIZONS:
    for prefix, name in [("ret", "raw"), ("xret", f"excess vs {BENCH}")]:
        col = f"{prefix}_{label}"
        sub = usable.dropna(subset=[col])
        if sub.empty:
            continue
        print(f"\n--- Horizon: {label} [{name}] (n={len(sub)} quarters with data) ---")
        signal_stats(sub, col)

print()
print("=" * 110)
print("  PER-TICKER SUMMARY  (raw returns)")
print("=" * 110)
for tk in TICKERS:
    sub_t = usable[usable["ticker"] == tk]
    if sub_t.empty:
        continue
    print(f"\n{tk}:")
    for label, _ in HORIZONS:
        col = f"ret_{label}"
        g = sub_t.dropna(subset=[col])
        if g.empty:
            continue
        beat, miss = g[g.signal == "BEAT"], g[g.signal == "MISS"]
        beat_hit = (beat[col] > 0).mean() * 100 if len(beat) else float("nan")
        base_up = (g[col] > 0).mean() * 100
        print(f"  {label:<4} BEAT n={len(beat):<2} avg={beat[col].mean() if len(beat) else float('nan'):+6.2f}% "
              f"hit={beat_hit:5.1f}% (base {base_up:5.1f}%)   "
              f"MISS n={len(miss):<2} avg={miss[col].mean() if len(miss) else float('nan'):+6.2f}%")
