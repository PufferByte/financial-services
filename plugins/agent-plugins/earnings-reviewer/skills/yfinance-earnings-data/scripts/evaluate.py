"""
Out-of-sample evaluation of earnings-event records (the .jsonl written by
`predict.py --out`).

Task: predict the DIRECTION of a label (e.g. excess_return_1d > 0) from the
record's `features`, training only on events before --split-date and scoring
only on events on/after it. No random shuffling -- a random split would let the
model learn from quarters that come after the ones it is tested on.

Models compared on the same test set:
- majority  : always predict the training set's more common direction
- eps_rule  : BEAT -> up, MISS -> down, INLINE -> majority   (the B1 baseline)
- logit     : L2-regularised logistic regression on --features (numpy only).
              Missing values are filled with the TRAINING median and flagged
              with a 0/1 indicator column, so the model can use "missing".

Metrics: accuracy, balanced accuracy (mean of per-class recall -- not fooled by
an always-up stock), MCC (Matthews correlation; 0 = no skill, 1 = perfect),
and the average return of trading the call (long if up, short if down).

Transcript features (management_sentiment etc.) are simply more --features:
once the extraction step fills them in, rerun with them added and compare.
"""

import argparse, json, sys
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("records", nargs="+", help="One or more .jsonl files from predict.py --out")
parser.add_argument("--split-date", required=True,
                    help="Train on earnings before this date, test on/after it (YYYY-MM-DD)")
parser.add_argument("--label", default="excess_return_1d",
                    help="Label to predict the sign of (default excess_return_1d)")
parser.add_argument("--features", default="eps_surprise",
                    help="Comma-separated feature names for the logit model")
parser.add_argument("--l2", type=float, default=1.0, help="L2 penalty for the logit model")
args = parser.parse_args()

FEATURES = [f.strip() for f in args.features.split(",") if f.strip()]
SPLIT = pd.Timestamp(args.split_date)

# ── load ────────────────────────────────────────────────────────────────────
recs = []
for path in args.records:
    with open(path) as fh:
        recs += [json.loads(line) for line in fh if line.strip()]
if not recs:
    sys.exit("no records")
df = pd.DataFrame([{
    "ticker": r["ticker"],
    "earnings_ts": pd.Timestamp(r["earnings_ts"]),
    "timing": r.get("timing"),
    "signal": r["features"].get("eps_signal"),
    **{f: r["features"].get(f) for f in FEATURES},
    "y_ret": r["labels"].get(args.label),
} for r in recs])
df = df.drop_duplicates(subset=["ticker", "earnings_ts"])
n_all = len(df)
df = df[df["timing"] != "UNKNOWN"].dropna(subset=["y_ret"])
df = df[df["y_ret"] != 0].sort_values("earnings_ts")   # a zero return has no direction
df["y"] = (df["y_ret"] > 0).astype(int)
for f in FEATURES:
    df[f] = pd.to_numeric(df[f], errors="coerce")

train, test = df[df["earnings_ts"] < SPLIT], df[df["earnings_ts"] >= SPLIT]
print(f"records: {n_all} loaded, {len(df)} usable (label '{args.label}' known, timing known)")
print(f"train: {len(train)} events  {train.earnings_ts.min():%Y-%m-%d} .. {train.earnings_ts.max():%Y-%m-%d}"
      if len(train) else "train: 0 events")
print(f"test : {len(test)} events  {test.earnings_ts.min():%Y-%m-%d} .. {test.earnings_ts.max():%Y-%m-%d}"
      if len(test) else "test : 0 events")
print(f"tickers: {', '.join(sorted(df.ticker.unique()))}")
if len(train) < 10 or len(test) < 5:
    sys.exit("[STOP] too few events on one side of the split for a meaningful evaluation; "
             "add tickers or move --split-date.")


# ── models ──────────────────────────────────────────────────────────────────
def design(frame, stats):
    """Feature matrix with train-median imputation + missing indicators + intercept."""
    cols = [np.ones(len(frame))]
    for f, (med, mu, sd, flag) in stats.items():
        x = frame[f].to_numpy(dtype=float)
        miss = np.isnan(x)
        cols.append((np.where(miss, med, x) - mu) / sd)
        if flag:
            cols.append(miss.astype(float))
    return np.column_stack(cols)


def fit_logit(X, y, l2):
    """Newton-Raphson for L2-penalised logistic regression (intercept unpenalised)."""
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) + pen * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


majority = int(train["y"].mean() >= 0.5)
preds = {"majority": np.full(len(test), majority)}
preds["eps_rule"] = test["signal"].map({"BEAT": 1, "MISS": 0}).fillna(majority).astype(int).to_numpy()

stats = {}
for f in FEATURES:
    x = train[f]
    if x.notna().sum() < 5:
        print(f"[WARN] feature '{f}' has <5 non-null training values -- dropped from logit")
        continue
    med = x.median()
    filled = x.fillna(med)
    sd = filled.std() or 1.0
    stats[f] = (med, filled.mean(), sd, bool(x.isna().any() or test[f].isna().any()))
if stats:
    w = fit_logit(design(train, stats), train["y"].to_numpy(dtype=float), args.l2)
    preds["logit"] = (design(test, stats) @ w > 0).astype(int)
    print(f"logit features used: {', '.join(stats)}")


# ── metrics ─────────────────────────────────────────────────────────────────
def metrics(y, p, ret):
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    rec_up = tp / (tp + fn) if tp + fn else np.nan
    rec_dn = tn / (tn + fp) if tn + fp else np.nan
    bal = np.nanmean([rec_up, rec_dn])
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    signed = np.where(p == 1, ret, -ret).mean()
    return acc, bal, mcc, signed


y, ret = test["y"].to_numpy(), test["y_ret"].to_numpy()
print()
print(f"OUT-OF-SAMPLE  (predicting sign of {args.label}; test share up = {y.mean()*100:.1f}%)")
print(f"{'model':<10} {'accuracy':>9} {'bal.acc':>8} {'MCC':>7} {'avg ret of call':>16}")
print("-" * 54)
for name, p in preds.items():
    acc, bal, mcc, signed = metrics(y, p, ret)
    print(f"{name:<10} {acc*100:8.1f}% {bal*100:7.1f}% {mcc:+7.3f} {signed*100:+15.2f}%")
print()
print("Read: a model only adds information if it beats 'majority' on balanced accuracy / MCC.")
print("'avg ret of call' ignores costs; with a small test set, differences of a few points are noise.")
