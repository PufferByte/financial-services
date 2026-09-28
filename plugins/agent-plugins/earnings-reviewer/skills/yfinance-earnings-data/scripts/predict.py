"""
Earnings Reviewer — historical backtest add-on (free data, yfinance only)

Question: does the tool's post-earnings signal (EPS beat / inline / miss)
line up with the stock's move after the print?

Two anchors (the price t0 every return is measured from), picked with --anchor:

- pre  (default) -- Task A, the full EARNINGS REACTION. t0 = the last
  regular-session close BEFORE the announcement became public. After-market
  (>= 16:00 ET) print -> that day's close; pre-market / intraday print -> the
  previous trading day's close. ret_1d = next close / t0 - 1.

- post-call -- Task B, POST-CALL RETURN. t0 = the first regular-session price
  (09:30 open or 16:00 close) at or after the transcript is available, taken
  as announcement time + --call-lag hours (default 3: release, then a ~1h
  call, then some slack). After-market print -> next day's open (the
  after-hours / gap move is excluded); pre-market print -> that day's open,
  or its close if the call runs into the session. Use this anchor whenever
  the model's inputs include the call transcript: the pre anchor would credit
  the model with price moves that happened before the transcript existed.

Horizons 1d / 1wk / 1mo / 3mo = 1 / 5 / 21 / 63 trading days: the return runs
from t0 to that many sessions' closes later (an open anchor's 1d is that same
day's close). Every return is also reported in excess of the benchmark
(default SPY) over exactly the same window.

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
import warnings, argparse, sys, json, math, os
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--tickers", default="AAPL,MSFT,NVDA",
                     help="Comma-separated tickers to backtest, e.g. --tickers TSLA,GOOGL")
parser.add_argument("--anchor", choices=["pre", "post-call"], default="pre",
                    help="pre = full earnings reaction from the pre-print close (Task A); "
                         "post-call = return from the first price after the call (Task B)")
parser.add_argument("--call-lag", type=float, default=3.0,
                    help="Hours from announcement until the transcript is available (post-call only)")
parser.add_argument("--benchmark", default="SPY",
                    help="Benchmark for excess returns (default SPY); pass '' to disable")
parser.add_argument("--limit", type=int, default=100,
                    help="Max earnings dates to request per ticker (yfinance default is only 12)")
parser.add_argument("--inline-band", type=float, default=0.0,
                    help="|Surprise%%| <= this is INLINE (default 0 = exact match only)")
parser.add_argument("--assume-unknown", choices=["skip", "bmo", "amc"], default="skip",
                    help="How to treat earnings timestamps with no time of day (00:00)")
parser.add_argument("--out", default="",
                    help="Also write one JSON record per quarter (features + labels) to this .jsonl path")
parser.add_argument("--raw-dir", default="",
                    help="Save the raw Yahoo responses (earnings dates, financials, prices) as CSVs here")
args, _ = parser.parse_known_args()

RETRIEVED_AT = pd.Timestamp.now(tz="UTC").floor("s")
RUN_STAMP = RETRIEVED_AT.strftime("%Y%m%dT%H%M%SZ")
TICKERS = [tk.strip().upper() for tk in args.tickers.split(",") if tk.strip()]
BENCH = args.benchmark.strip().upper()
HORIZONS = [("1d", 1), ("1wk", 5), ("1mo", 21), ("3mo", 63)]  # trading days after t0
MARKET_OPEN = pd.Timedelta(hours=9, minutes=30)
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


def load_timeline(ticker, start):
    """Regular-session prices as one time-ordered Series: open, close, open, close, ...

    Event i belongs to trading day i // 2 and is that day's open (i even) or close (i odd).
    """
    hist = yf.Ticker(ticker).history(start=start, end=None, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist:
        return None
    save_raw(ticker, "prices", hist)
    hist = hist[["Open", "Close"]].dropna()
    days = to_ny_naive(hist.index).normalize()
    times = np.empty(2 * len(days), dtype="datetime64[ns]")
    times[0::2] = (days + MARKET_OPEN).values
    times[1::2] = (days + MARKET_CLOSE).values
    prices = np.empty(2 * len(days))
    prices[0::2] = hist["Open"].values
    prices[1::2] = hist["Close"].values
    return pd.Series(prices, index=pd.DatetimeIndex(times))


def anchor_index(tl, ann):
    """Timeline index of t0 for an announcement at `ann`, or None if not in the data."""
    if args.anchor == "pre":
        closes = tl.index[1::2]
        pos = closes.searchsorted(ann, side="right") - 1   # last close at/before the print
        return 2 * pos + 1 if pos >= 0 else None
    i = tl.index.searchsorted(ann + pd.Timedelta(hours=args.call_lag), side="left")
    return i if i < len(tl) else None


def horizon_index(i0, ndays):
    """Close `ndays` sessions after t0 (an open's first session is its own day)."""
    day0 = i0 // 2
    end_day = day0 + ndays - (1 if i0 % 2 == 0 else 0)
    return 2 * end_day + 1


def _fin(qf, row, col):
    try:
        v = qf.loc[row, col]
        return float(v) if pd.notna(v) else None
    except (KeyError, TypeError, ValueError):
        return None


def financial_features(qf, edate):
    """Revenue YoY and gross margin for the quarter this print reports.

    The reported quarter is the latest quarter-end within 60 days before the print.
    Yahoo's quarterly_financials only covers ~5 recent quarters, so older prints get
    None. Values are as currently reported (may include later restatements).
    """
    out = {"fiscal_quarter_end": None, "revenue_yoy": None, "gross_margin": None}
    if qf is None or qf.empty:
        return out
    cands = [c for c in qf.columns if pd.Timedelta(0) <= edate - c <= pd.Timedelta(days=60)]
    if not cands:
        return out
    q = max(cands)
    out["fiscal_quarter_end"] = q.date().isoformat()
    rev = _fin(qf, "Total Revenue", q)
    gp = _fin(qf, "Gross Profit", q)
    yr_ago = [c for c in qf.columns if abs((q - c).days - 365) <= 20]
    rev_py = _fin(qf, "Total Revenue", yr_ago[0]) if yr_ago else None
    if rev and gp is not None:
        out["gross_margin"] = gp / rev
    if rev is not None and rev_py:
        out["revenue_yoy"] = rev / rev_py - 1
    return out


def save_raw(ticker, name, df):
    """Keep the exact Yahoo response on disk: Yahoo revises history silently, so a
    result is only reproducible from the snapshot it was computed on."""
    if args.raw_dir and df is not None and not df.empty:
        os.makedirs(args.raw_dir, exist_ok=True)
        df.to_csv(os.path.join(args.raw_dir, f"{ticker}_{name}_{RUN_STAMP}.csv"))


def get_returns(ticker, bench_tl):
    t = yf.Ticker(ticker)
    try:
        qf = t.quarterly_financials
        qf.columns = pd.to_datetime(qf.columns)
    except Exception:
        qf = None
    ed = t.get_earnings_dates(limit=args.limit)
    save_raw(ticker, "quarterly_financials", qf)
    save_raw(ticker, "earnings_dates", ed)
    if ed is None or ed.empty:
        raise ValueError("no earnings dates returned")
    ed = ed.dropna(subset=["Reported EPS", "Surprise(%)"]).copy()
    if ed.empty:
        raise ValueError("no reported quarters with a surprise figure")
    ed.index = to_ny_naive(ed.index)
    ed = ed[~ed.index.duplicated()].sort_index()

    tl = load_timeline(ticker, (ed.index.min() - pd.Timedelta(days=10)).date().isoformat())
    if tl is None:
        raise ValueError("no price history returned")

    rows = []
    for edate, r in ed.iterrows():
        if edate != edate.normalize():
            timing = "AMC" if edate >= edate.normalize() + MARKET_CLOSE else "BMO/DMH"
            ann = edate
        else:
            timing = "UNKNOWN"
            ann = {"amc": edate + MARKET_CLOSE, "bmo": edate}.get(args.assume_unknown)

        row = {
            "ticker": ticker,
            "earnings_ts": edate,
            "timing": timing,
            "eps_est": r["EPS Estimate"],
            "eps_rep": r["Reported EPS"],
            "surprise_pct": r["Surprise(%)"],
            "signal": classify(r["Surprise(%)"]),
            "t0": None, "p0": np.nan,
            # when the model's inputs are public: the print itself, or the end of the call
            "available_at": None if ann is None else
                (ann + pd.Timedelta(hours=args.call_lag) if args.anchor == "post-call" else ann),
            **financial_features(qf, edate),
        }
        for label, _ in HORIZONS:
            row[f"ret_{label}"] = np.nan
            row[f"xret_{label}"] = np.nan

        i0 = anchor_index(tl, ann) if ann is not None else None
        if i0 is not None:
            p0 = tl.iloc[i0]
            row["t0"] = tl.index[i0].strftime("%Y-%m-%d ") + ("open" if i0 % 2 == 0 else "close")
            row["p0"] = p0
            for label, ndays in HORIZONS:
                i1 = horizon_index(i0, ndays)
                if i1 >= len(tl):
                    continue  # horizon hasn't happened yet
                ret = (tl.iloc[i1] / p0 - 1) * 100
                row[f"ret_{label}"] = ret
                if bench_tl is not None:
                    t_0, t_1 = tl.index[i0], tl.index[i1]
                    if t_0 in bench_tl.index and t_1 in bench_tl.index:
                        bret = (bench_tl[t_1] / bench_tl[t_0] - 1) * 100
                        row[f"xret_{label}"] = ret - bret
        rows.append(row)
    return pd.DataFrame(rows)


bench_tl = None
if BENCH:
    bench_tl = load_timeline(BENCH, "1995-01-01")
    if bench_tl is None:
        print(f"[WARN] could not load benchmark {BENCH}; excess returns disabled")

all_rows = []
for tk in TICKERS:
    try:
        all_rows.append(get_returns(tk, bench_tl))
    except Exception as e:
        print(f"[SKIP] {tk}: {e}")
if not all_rows:
    print("[UNSOURCED] no usable earnings history for any requested ticker.")
    sys.exit(1)
big = pd.concat(all_rows, ignore_index=True)
usable = big[big["timing"] != "UNKNOWN"] if args.assume_unknown == "skip" else big


def _num(v, scale=1.0):
    """NaN/None -> None (JSON null); otherwise a plain float."""
    return None if v is None or pd.isna(v) else float(v) * scale


def _str(v):
    """NaN/None -> None; pandas turns a None in a mostly-string column into NaN."""
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def write_records(df, path):
    """One JSON line per quarter: features (model inputs) + labels (realized returns).

    Returns are fractions (0.023 = +2.3%). A label is null when its horizon hasn't
    happened yet or the print can't be placed in time. Transcript features are
    null placeholders for the extraction step to fill in.
    """
    with open(path, "w") as fh:
        for r in df.itertuples(index=False):
            rec = {
                "ticker": r.ticker,
                "retrieved_at": RETRIEVED_AT.isoformat(),
                "data_source": "yfinance " + yf.__version__,
                "earnings_ts": r.earnings_ts.isoformat(),
                "timing": r.timing,
                "fiscal_quarter_end": _str(r.fiscal_quarter_end),
                "anchor": args.anchor,
                "available_at": r.available_at.isoformat() if _str(r.available_at) is not None and pd.notna(r.available_at) else None,
                "t0": _str(r.t0),
                "p0": _num(r.p0),
                "features": {
                    "eps_estimate": _num(r.eps_est),
                    "eps_reported": _num(r.eps_rep),
                    "eps_surprise": _num(r.surprise_pct, 0.01),
                    "eps_signal": r.signal,
                    "revenue_yoy": _num(r.revenue_yoy),
                    "gross_margin": _num(r.gross_margin),
                    "management_sentiment": None,
                    "guidance_sentiment": None,
                    "risk_sentiment": None,
                },
                "labels": {
                    **{f"return_{lbl}": _num(getattr(r, f"ret_{lbl}"), 0.01) for lbl, _ in HORIZONS},
                    **{f"excess_return_{lbl}": _num(getattr(r, f"xret_{lbl}"), 0.01) for lbl, _ in HORIZONS},
                },
            }
            fh.write(json.dumps(rec, allow_nan=False) + "\n")
    print(f"[OUT] wrote {len(df)} record(s) -> {path}")


if args.out:
    write_records(big, args.out)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 500)

anchor_desc = ("last close BEFORE the print (full earnings reaction)" if args.anchor == "pre"
               else f"first price >= {args.call_lag:g}h after the print (post-call)")
print("=" * 110)
print(f"  PER-QUARTER DETAIL   (t0 = {anchor_desc})")
print("=" * 110)
show_cols = ["ticker", "earnings_ts", "timing", "signal", "surprise_pct", "t0",
             "p0", "ret_1d", "xret_1d", "ret_1wk", "ret_1mo", "ret_3mo"]
detail = big[show_cols].copy()
for c in ["surprise_pct", "ret_1d", "xret_1d", "ret_1wk", "ret_1mo", "ret_3mo"]:
    detail[c] = detail[c].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "-")
detail["p0"] = detail["p0"].map(lambda v: f"${v:.2f}" if pd.notna(v) else "-")
detail["t0"] = detail["t0"].fillna("-")
detail["earnings_ts"] = detail["earnings_ts"].map(lambda v: v.strftime("%Y-%m-%d %H:%M"))
print(detail.to_string(index=False))
n_unknown = (big["timing"] == "UNKNOWN").sum()
if n_unknown and args.assume_unknown == "skip":
    print(f"\n[NOTE] {n_unknown} quarter(s) have no time of day -> excluded from stats "
          f"(use --assume-unknown bmo|amc to include).")


def binom_p_two_sided(k, n, p):
    """Exact two-sided binomial test: P(outcome at least as unlikely as k | n, p)."""
    if n == 0 or p <= 0 or p >= 1:
        return float("nan")
    pmf = [math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)]
    return min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-9)))


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z**2 / n
    mid = (ph + z**2 / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return (mid - half) * 100, (mid + half) * 100


def signal_stats(sub, col):
    """Print per-signal avg / hit rate vs. base rate for one return column.

    Significance: exact binomial test of the signal's hit count against the base
    rate (H0: the signal's hit rate equals the unconditional rate), plus a 95%
    Wilson interval on the hit rate. The base rate is estimated from the same
    sample, so treat p-values as approximate; with many horizons x tickers,
    expect ~5% of p-values < 0.05 by chance alone.
    """
    base_up = (sub[col] > 0).mean() * 100
    base_down = (sub[col] < 0).mean() * 100   # zero returns count as neither
    for sig in ["BEAT", "INLINE", "MISS"]:
        g = sub[sub["signal"] == sig]
        if g.empty:
            continue
        avg = g[col].mean()
        if sig == "BEAT":
            k, base = int((g[col] > 0).sum()), base_up
        elif sig == "MISS":
            k, base = int((g[col] < 0).sum()), base_down
        else:
            print(f"  {sig:<6} n={len(g):<3} avg {avg:+6.2f}%   (no directional call)")
            continue
        hit = k / len(g) * 100
        lo, hi = wilson_ci(k, len(g))
        pval = binom_p_two_sided(k, len(g), base / 100)
        small = "  [n<10: not meaningful]" if len(g) < 10 else ""
        print(f"  {sig:<6} n={len(g):<3} avg {avg:+6.2f}%   hit {hit:5.1f}% [95% CI {lo:4.1f}-{hi:5.1f}]  "
              f"vs base {base:5.1f}%  -> edge {hit - base:+5.1f}pp  p={pval:.3f}{small}")
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
