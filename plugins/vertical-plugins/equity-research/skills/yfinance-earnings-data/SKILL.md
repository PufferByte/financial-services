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

Historical backtest: for every past reported quarter (as far back as Yahoo has earnings-date history, often 5+ years), it anchors the post-earnings "reaction price" the first trading day after the print, then measures the real forward return at +1 week / +1 month / +3 months, and reports whether a BEAT was actually followed by a positive move (and a MISS by a negative one) -- i.e. an empirical hit rate per horizon, not just today's single data point.

Fold this into the note as context, e.g. "historically, a beat at this name has preceded a positive 1-month move in N of M quarters (X%)" -- it tells the reader how much weight to put on today's beat/miss, which the single-quarter pull from Step 1 cannot.

**Caveats to carry into the note, not just this skill:**
- Small-sample risk: a name that rarely misses will have very few MISS data points; don't quote a MISS-conditional average return as if it were statistically meaningful with n < ~10.
- This is EPS-surprise-direction only, not the full multi-factor signal (revenue growth, margin trend, FCF, PT upside) from Step 1 -- Yahoo's free `quarterly_financials` only gives ~5 quarters of history, not enough to reconstruct the full rule further back.
- No transaction costs, slippage, or statistical significance testing. Report it as descriptive history, not a trading recommendation.

## Both scripts accept comma-separated / repeated use across a coverage list

`earnings_yfinance.py` takes one `--ticker` per run. `predict.py` takes `--tickers AAPL,MSFT,NVDA` (comma-separated) to backtest several names in one pass -- useful for a coverage-list fan-out.

## When NOT to use

If `mcp__factset__*` / `mcp__daloopa__*` tools are configured and available, prefer those -- they're the higher-quality, properly-licensed data source this agent was originally built around. This skill exists for firms that don't have that subscription, or for the backtest capability that FactSet/Daloopa don't provide at all.
