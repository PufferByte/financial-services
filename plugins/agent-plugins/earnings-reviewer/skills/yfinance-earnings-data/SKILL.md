---
name: yfinance-earnings-data
description: Free-data fallback for pulling reported earnings actuals, consensus estimates, analyst price targets, and a historical beat/miss backtest via Yahoo Finance (yfinance) -- for use INSTEAD OF the paid FactSet/Daloopa MCP connectors. Use when FactSet/Daloopa aren't configured, or whenever the firm wants to avoid the subscription cost for a given coverage name. Triggers on "pull earnings data", "no FactSet", "use free data", "yfinance", "check the historical signal", "has this stock's earnings reaction been reliable".
---

# yfinance Earnings Data (free-data fallback)

Replaces `mcp__factset__*` / `mcp__daloopa__*` calls with two bundled, self-contained Python scripts that call the free `yfinance` library directly. No API key, no subscription, no MCP server to configure -- just outbound internet access.

**Trade-off to be upfront about, every time this skill is used:** Yahoo Finance data is not institutional-grade. It is delayed/best-effort, occasionally revises historical estimates, and its "Reported EPS" figure is sometimes street-adjusted (non-GAAP) while `quarterly_financials`' "Diluted EPS" is GAAP -- the two can disagree by several cents in the same quarter. Flag any figure you can't independently corroborate as `[UNSOURCED]`, per this agent's standing guardrail.

## Step 1 -- Pull the print

```bash
python scripts/earnings_yfinance.py --ticker <TICKER>
```

Prints a full earnings-update report to stdout (beat/miss vs. consensus, quarterly progression, YoY changes, margin analysis, valuation, EPS beat/miss history, and a rule-based bullish/bearish signal count) and saves a 6-panel PNG dashboard to `./out/<ticker>_earnings_dashboard.png`.

Exits with a clear `[UNSOURCED]` message (not a crash) if the ticker has no usable quarterly earnings history on Yahoo Finance -- expected for ETFs, indices, delisted symbols, or a very recent IPO with fewer than 5 reported quarters.

Use this in place of workflow step 1 ("Pull the print. FactSet/Daloopa MCP for reported actuals, consensus...") in the main agent prompt.

## Step 2 -- Check whether the signal is actually reliable for this name

```bash
python scripts/predict.py --tickers <TICKER>
```

Historical backtest of the **earnings reaction**. For every past reported quarter (up to `--limit`, default 100 earnings dates), returns are measured from the **last close before the print became public** (after-market print: that day's close; pre-market print: the prior day's close):

- `1d` = next close vs. that pre-print close -- the earnings reaction itself
- `1wk` / `1mo` / `3mo` = 5 / 21 / 63 trading days after the pre-print close (these include the day-1 reaction)
- each also reported **in excess of SPY** (`--benchmark`) over the same window

`--out records.jsonl` also writes one JSON line per quarter -- `features` (EPS estimate/reported/surprise, revenue YoY, gross margin, plus null transcript-sentiment placeholders) and `labels` (raw and excess-vs-SPY returns as fractions; null until the horizon has elapsed), with `available_at` / `t0` recording when the inputs were public and where returns start. Revenue YoY and gross margin are only filled for the ~5 most recent quarters Yahoo's `quarterly_financials` covers.

`--raw-dir DIR` saves the raw Yahoo responses (earnings dates, quarterly financials, prices) as timestamped CSVs, and every JSONL record carries `retrieved_at` -- Yahoo revises history silently, so a result is only reproducible from its snapshot. Hit rates are printed with a 95% Wilson interval and an exact binomial p-value against the base rate.

Out-of-sample check on those records (time-based split, never random):

```bash
python scripts/evaluate.py records.jsonl --split-date 2019-01-01 --label excess_return_1d --features eps_surprise
```

Compares an always-majority baseline, the BEAT/MISS rule, and a logistic regression on `--features` by accuracy, balanced accuracy, MCC, and the average return of trading the call. Transcript features are just more `--features` once they are filled in.

`--anchor post-call` switches to the **post-call return**: t0 is the first regular-session open/close at least `--call-lag` hours (default 3) after the print, i.e. once the call transcript exists (after-market print: next day's open). Use this whenever the question is "does what was said on the call predict the move from here" -- the default pre-print anchor includes the price move that happened before the transcript was available.

Signal is BEAT / INLINE / MISS from Yahoo's Surprise% (`--inline-band`, default 0). For each horizon it reports the hit rate **next to the base rate** (share of all quarters that moved that way) and the difference ("edge"). Quarters whose Yahoo timestamp has no time of day can't be placed before/after the open, so they are excluded unless `--assume-unknown bmo|amc` is given.

Fold this into the note as context, e.g. "historically, a beat at this name has been followed by a positive 1-day reaction in N of M quarters (X%, vs. a Y% base rate)" -- always quote the base rate alongside the hit rate; a high hit rate on a stock that rises after most prints anyway carries no information.

**Caveats to carry into the note, not just this skill:**
- Small-sample risk: a name that rarely misses will have very few MISS data points; don't quote a MISS-conditional average return as if it were statistically meaningful with n < ~10.
- This is EPS-surprise-direction only, not the full multi-factor signal (revenue growth, margin trend, FCF, PT upside) from Step 1 -- Yahoo's free `quarterly_financials` only gives ~5 quarters of history, not enough to reconstruct the full rule further back.
- No transaction costs, slippage, or statistical significance testing. Report it as descriptive history, not a trading recommendation.
- The default ticker list (AAPL, MSFT, NVDA) is three long-run winners -- raw returns are biased upward; prefer the excess-vs-SPY rows.
- Step 1's revenue beat/miss is `[UNSOURCED]`: Yahoo has no historical revenue consensus for an already-reported quarter. EPS beat/miss is computed on the street basis (Yahoo "Reported EPS" vs. "EPS Estimate"), not GAAP diluted EPS.

## Both scripts accept comma-separated / repeated use across a coverage list

`earnings_yfinance.py` takes one `--ticker` per run. `predict.py` takes `--tickers AAPL,MSFT,NVDA` (comma-separated) to backtest several names in one pass -- useful for a coverage-list fan-out.

## When NOT to use

If `mcp__factset__*` / `mcp__daloopa__*` tools are configured and available, prefer those -- they're the higher-quality, properly-licensed data source this agent was originally built around. This skill exists for firms that don't have that subscription, or for the backtest capability that FactSet/Daloopa don't provide at all.
