"""
Earnings Reviewer Pipeline — Yahoo Finance edition
Replaces FactSet/Daloopa MCPs with yfinance free data.
Follows the earnings-analysis skill structure from financial-services repo.
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings, sys, os, argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--ticker", default="AAPL")
args, _ = parser.parse_known_args()

TICKER = args.ticker.upper()
OUT_DIR = "./out"
os.makedirs(OUT_DIR, exist_ok=True)

t = yf.Ticker(TICKER)
COMPANY_NAME = (t.info.get('shortName') or t.info.get('longName') or TICKER)

# ── 1. PULL DATA ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  EARNINGS REVIEWER PIPELINE  (Yahoo Finance edition)")
print(f"  Company: {COMPANY_NAME} ({TICKER})")
print(f"{'='*60}\n")

qf   = t.quarterly_financials
bs   = t.quarterly_balance_sheet
cf   = t.quarterly_cashflow
info = t.info
edates = t.earnings_dates

if qf is None or qf.empty:
    print(f"[UNSOURCED] '{TICKER}' has no quarterly financials on Yahoo Finance. "
          f"This is expected for ETFs, indices, delisted, or invalid tickers. "
          f"Not a network error -- the data genuinely isn't there.")
    sys.exit(1)
# Yahoo often returns only 4-5 quarters. Anything that needs a prior-year or prior
# quarter shows N/A when it's missing instead of rejecting the whole ticker.
qf = qf.sort_index(axis=1, ascending=False)
if cf is None:
    cf = pd.DataFrame()

# ── 2. IDENTIFY LATEST QUARTER ─────────────────────────────────────────────
latest_col  = qf.columns[0]   # most recent quarter
prev_q_col  = qf.columns[1] if len(qf.columns) > 1 else None   # previous quarter
# same quarter last year: match by date, not position -- Yahoo sometimes drops a quarter,
# in which case columns[4] would silently be 5 quarters back
_yr_ago = [c for c in qf.columns[1:] if abs((latest_col - c).days - 365) <= 20]
prev_yr_col = _yr_ago[0] if _yr_ago else None

def fmt_quarter(ts):
    return ts.strftime('%b %Y')

def b(x): return x / 1e9 if x is not None else None   # to billions
def m(x): return x / 1e6 if x is not None else None   # to millions

def ratio(num, den, scale=1):
    """num / den * scale, or None if either is missing or den is 0."""
    return num / den * scale if (num is not None and den) else None

def nan(v):
    """None -> NaN, so matplotlib leaves a gap instead of raising."""
    return np.nan if v is None else v

qname = fmt_quarter(latest_col)

# ── 3. EXTRACT KEY METRICS ──────────────────────────────────────────────────
def get(df, row, col):
    try:
        v = df.loc[row, col]
        return float(v) if not pd.isna(v) else None
    except:
        return None

# Latest quarter actuals
rev     = get(qf, 'Total Revenue',  latest_col)
rev_py  = get(qf, 'Total Revenue',  prev_yr_col)
rev_pq  = get(qf, 'Total Revenue',  prev_q_col)
gp      = get(qf, 'Gross Profit',   latest_col)
gp_py   = get(qf, 'Gross Profit',   prev_yr_col)
oi      = get(qf, 'Operating Income', latest_col)
oi_py   = get(qf, 'Operating Income', prev_yr_col)
ni      = get(qf, 'Net Income',     latest_col)
eps     = get(qf, 'Diluted EPS',    latest_col)
eps_py  = get(qf, 'Diluted EPS',    prev_yr_col)

fcf     = get(cf, 'Free Cash Flow', latest_col)
capex   = get(cf, 'Capital Expenditure', latest_col)
ocf     = get(cf, 'Operating Cash Flow', latest_col)

gm      = ratio(gp, rev)
oi_m    = ratio(oi, rev)
gm_py   = ratio(gp_py, rev_py)
oi_m_py = ratio(oi_py, rev_py)

# Consensus EPS from earnings_dates.
# Matched by REPORT DATE proximity to the quarter-end, not by comparing EPS values:
# quarterly_financials' "Diluted EPS" is GAAP, while earnings_dates' "Reported EPS" is
# often a street-adjusted (non-GAAP) figure. The two can differ by more than a few cents
# even for an in-line quarter (e.g. MSFT FY24 Q4: 4.81 GAAP vs. 4.74 street, a $0.07 gap),
# so matching on "close enough EPS value" silently fails for some tickers/quarters while
# working for others by coincidence. A report is reliably the first earnings_dates entry
# shortly after the quarter's period end, so match on that instead.
consensus_eps = None
street_eps    = None   # Yahoo "Reported EPS" -- same (street/adjusted) basis as the consensus
surprise_pct  = None
latest_edate_idx = None
if edates is None:
    edates = pd.DataFrame(columns=['EPS Estimate', 'Reported EPS', 'Surprise(%)'],
                          index=pd.DatetimeIndex([]))
reported_dates = edates.dropna(subset=['Reported EPS']).copy()
if not reported_dates.empty:
    reported_dates.index = reported_dates.index.tz_localize(None)
    window = reported_dates[
        (reported_dates.index >= latest_col - pd.Timedelta(days=10)) &
        (reported_dates.index <= latest_col + pd.Timedelta(days=45))
    ]
    if not window.empty:
        latest_edate_idx = window.index.min()
        row_ed = window.loc[latest_edate_idx]
        consensus_eps = row_ed['EPS Estimate'] if pd.notna(row_ed['EPS Estimate']) else None
        street_eps    = row_ed['Reported EPS'] if pd.notna(row_ed['Reported EPS']) else None
        surprise_pct  = row_ed['Surprise(%)'] if pd.notna(row_ed['Surprise(%)']) else None

# Analyst consensus price target -- a LIVE snapshot as of today, not as of the print date.
# Fine for a current-quarter note; never feed these fields into a historical backtest
# (that would be look-ahead: today's targets already reflect the post-earnings move).
price     = info.get('currentPrice')
pt_mean   = info.get('targetMeanPrice')
pt_high   = info.get('targetHighPrice')
pt_low    = info.get('targetLowPrice')
fwd_pe    = info.get('forwardPE')
trail_pe  = info.get('trailingPE')
mktcap    = info.get('marketCap')
fwd_eps   = info.get('forwardEps')
rec_mean  = info.get('recommendationMean')   # 1=Strong Buy, 5=Sell
n_analysts= info.get('numberOfAnalystOpinions')

rec_label = {1:'Strong Buy', 2:'Buy', 3:'Hold', 4:'Underperform', 5:'Sell'}
def rec_str(v):
    if v is None: return 'N/A'
    return rec_label.get(round(v), f'{v:.1f}')

# Revenue consensus: yfinance has no historical revenue consensus for an already-reported
# quarter. t.revenue_estimate['0q'] is the estimate for the CURRENT (not yet reported)
# quarter, so comparing it to the latest reported revenue is apples-to-oranges. Leave it
# unsourced rather than print a meaningless beat/miss.
rev_consensus = None
rev_beat = None

# EPS beat/miss: compare street-basis reported EPS to the street-basis consensus. Using the
# GAAP "Diluted EPS" from quarterly_financials here would mix GAAP and adjusted figures.
eps_beat = (float(street_eps) - float(consensus_eps)) \
    if (street_eps is not None and consensus_eps is not None) else None

# ── 4. PRINT REPORT ─────────────────────────────────────────────────────────

SEP = '─' * 58

def pct(a, b):
    if a and b and b != 0:
        return (a - b) / abs(b) * 100
    return None

def pp(v, unit='B', dec=2):
    if v is None: return 'N/A'
    if unit == 'B': return f"${v/1e9:.{dec}f}B"
    if unit == '%': return f"{v*100:.{dec}f}%"
    if unit == 'bps': return f"{v*100:.0f}bps"
    if unit == '$': return f"${v:.{dec}f}"
    return str(round(v, dec))

def chg(v):
    if v is None: return ''
    sign = '+' if v >= 0 else ''
    return f"({sign}{v:.1f}%)"

def fnum(v, spec, prefix='', suffix=''):
    """Format a possibly-missing number; Yahoo omits fields for many tickers."""
    if v is None or (isinstance(v, float) and np.isnan(v)): return 'N/A'
    return f"{prefix}{v:{spec}}{suffix}"

# ── PAGE 1 ──────────────────────────────────────────────────────────────────
rev_yoy   = pct(rev, rev_py)
rev_qoq   = pct(rev, rev_pq)
gm_delta  = (gm - gm_py) * 10000 if (gm is not None and gm_py is not None) else None  # bps
oim_delta = (oi_m - oi_m_py) * 10000 if (oi_m is not None and oi_m_py is not None) else None
pt_upside = pct(pt_mean, price)

if eps_beat is None:
    beat_miss = "N/A (no matching consensus)"
elif eps_beat > 0:
    beat_miss = "BEAT"
elif eps_beat < 0:
    beat_miss = "MISS"
else:
    beat_miss = "INLINE"

print(f"{COMPANY_NAME.upper()} ({TICKER})  ·  {qname} EARNINGS UPDATE")
print(f"Analysis Date: {pd.Timestamp.now().date()}  ·  Source: Yahoo Finance (free)")
print()
print(f"Analyst Consensus:  {rec_str(rec_mean)}  |  "
      f"Price Target: {fnum(pt_mean, '.0f', '$')}  "
      f"(range {fnum(pt_low, '.0f', '$')}–{fnum(pt_high, '.0f', '$')})")
print(f"Current Price: {fnum(price, '.2f', '$')}  |  Upside to PT: {fnum(pt_upside, '+.1f', suffix='%')}")
print()
print("EARNINGS SUMMARY")
print(SEP)
print(f"{qname} RESULTS:  {beat_miss}")
print()
print(f"{'Metric':<22} {'Reported':>10} {'Consensus':>10} {'Beat/(Miss)':>12}")
print(f"{'─'*22} {'─'*10} {'─'*10} {'─'*12}")
print(f"{'Revenue':<22} {pp(rev,'B'):>10} {'N/A':>10} {'N/A':>12}  [UNSOURCED: no historical revenue consensus on Yahoo]")
eps_c_str  = f"${float(consensus_eps):.2f}" if consensus_eps is not None else 'N/A'
eps_b_str  = f"${eps_beat:+.2f}" if eps_beat is not None else 'N/A'
eps_s_str  = f"${float(street_eps):.2f}" if street_eps is not None else 'N/A'
print(f"{'EPS (street basis)':<22} {eps_s_str:>10} {eps_c_str:>10} {eps_b_str:>12}")
print(f"{'EPS (GAAP diluted)':<22} {pp(eps,'$'):>10} {'':>10} {'':>12}  (not comparable to consensus)")
print(f"{'Gross Margin':<22} {pp(gm,'%'):>10}")
print(f"{'Operating Margin':<22} {pp(oi_m,'%'):>10}")
if surprise_pct is not None: print(f"\nEPS Surprise: {surprise_pct:+.2f}%")
print()

# 3 key takeaways
print("Key Takeaways:")
if rev_yoy is not None:
    print(f"■ Revenue {pp(rev,'B')} {'grew' if rev_yoy >= 0 else 'declined'} {rev_yoy:+.1f}% YoY")
else:
    print(f"■ Revenue {pp(rev,'B')} (no prior-year quarter for YoY)")
if gm_delta is not None:
    gm_dir = "expanded" if gm_delta > 0 else "compressed"
    print(f"■ Gross margin {pp(gm,'%')} — {gm_dir} {abs(gm_delta):.0f}bps vs. prior year")
else:
    print(f"■ Gross margin {pp(gm,'%')} (no prior-year comparison)")
print(f"■ Free cash flow {pp(fcf,'B')} in quarter; "
      f"net cash flow from operations {pp(ocf,'B')}")

# ── QUARTERLY PROGRESSION TABLE ─────────────────────────────────────────────
print()
print("QUARTERLY PROGRESSION  (last 4 quarters)")
print(SEP)
cols = qf.columns[:4]
labels = [fmt_quarter(c) for c in cols]

rows_data = {
    'Revenue ($B)'    : [b(get(qf,'Total Revenue', c)) for c in cols],
    'Gross Margin'    : [ratio(get(qf,'Gross Profit',c), get(qf,'Total Revenue',c), 100) for c in cols],
    'Operating Margin': [ratio(get(qf,'Operating Income',c), get(qf,'Total Revenue',c), 100) for c in cols],
    'Net Income ($B)' : [b(get(qf,'Net Income', c)) for c in cols],
    'Diluted EPS ($)' : [get(qf,'Diluted EPS', c) for c in cols],
    'Free CF ($B)'    : [b(get(cf,'Free Cash Flow', c)) for c in cols],
}

header = f"{'Metric':<22}" + "".join(f"{l:>9}" for l in labels)
print(header)
print('─' * (22 + 9*4))
for metric, vals in rows_data.items():
    row_str = f"{metric:<22}"
    for v in vals:
        if v is None:
            row_str += f"{'N/A':>9}"
        elif 'Margin' in metric:
            row_str += f"{v:>8.1f}%"
        else:
            row_str += f"{v:>9.2f}"
    print(row_str)

# ── YoY CHANGES ──────────────────────────────────────────────────────────────
print()
print(f"YoY CHANGES  (vs. {fmt_quarter(prev_yr_col) if prev_yr_col is not None else 'N/A -- no prior-year quarter'})")
print(SEP)
metrics_yoy = [
    ("Revenue",        rev,    rev_py,  'B'),
    ("Gross Profit",   gp,     gp_py,   'B'),
    ("Operating Inc.", oi,     oi_py,   'B'),
    ("EPS (Diluted)",  eps,    eps_py,  '$'),
]
for name, cur, prior, unit in metrics_yoy:
    if cur and prior:
        delta_pct = pct(cur, prior)
        print(f"  {name:<20} {pp(cur, unit):>8}  vs  {pp(prior, unit):>8}  "
              f"→  {delta_pct:+.1f}% YoY")

# ── MARGIN ANALYSIS ──────────────────────────────────────────────────────────
print()
print("MARGIN ANALYSIS")
print(SEP)
if gm and gm_py:
    print(f"  Gross Margin   {pp(gm,'%'):>7}  vs  {pp(gm_py,'%'):>7} prior year  "
          f"({gm_delta:+.0f} bps YoY)")
if oi_m and oi_m_py:
    print(f"  Op. Margin     {pp(oi_m,'%'):>7}  vs  {pp(oi_m_py,'%'):>7} prior year  "
          f"({oim_delta:+.0f} bps YoY)")

# ── VALUATION ────────────────────────────────────────────────────────────────
print()
print("VALUATION & ESTIMATES")
print(SEP)
print(f"  Current Price   {fnum(price, '.2f', '$')}")
print(f"  Market Cap      {pp(mktcap,'B')}")
print(f"  Trailing P/E    {fnum(trail_pe, '.1f', suffix='x')}")
print(f"  Forward P/E     {fnum(fwd_pe, '.1f', suffix='x')}  (based on fwd EPS {fnum(fwd_eps, '.2f', '$')})")
print(f"  52-Week Range   {fnum(info.get('fiftyTwoWeekLow'), '.2f', '$')} – {fnum(info.get('fiftyTwoWeekHigh'), '.2f', '$')}")
print(f"  Analyst PT      {fnum(pt_mean, '.0f', '$')}  (consensus of {n_analysts or 'N/A'} analysts)")
print(f"  Implied Upside  {fnum(pt_upside, '+.1f', suffix='%')}")
print(f"  Rating          {rec_str(rec_mean)}")

# ── EPS BEAT HISTORY ─────────────────────────────────────────────────────────
print()
print("EPS BEAT/MISS HISTORY  (last 6 quarters)")
print(SEP)
recent = edates.dropna(subset=['Reported EPS']).head(6)
print(f"{'Quarter':<28} {'Estimate':>9} {'Reported':>9} {'Surprise':>10}")
print('─' * 58)
for idx, row_ed in recent.iterrows():
    qtr_label = idx.strftime('%b %Y')
    est   = row_ed['EPS Estimate']
    rep   = row_ed['Reported EPS']
    surp  = row_ed['Surprise(%)']
    if pd.isna(surp):  flag = ""
    elif surp > 0:     flag = "✓ BEAT"
    elif surp < 0:     flag = "✗ MISS"
    else:              flag = "= INLINE"
    est_s = f"${est:.2f}" if pd.notna(est) else "N/A"
    rep_s = f"${rep:.2f}" if pd.notna(rep) else "N/A"
    surp_s= f"{surp:+.1f}%" if pd.notna(surp) else "N/A"
    print(f"  {qtr_label:<26} {est_s:>9} {rep_s:>9}  {surp_s:>7}  {flag}")

# ── INVESTMENT ASSESSMENT ────────────────────────────────────────────────────
print()
print("INVESTMENT ASSESSMENT")
print(SEP)

bullish_signals = 0
bearish_signals = 0

if eps_beat is not None and eps_beat > 0:
    bullish_signals += 1
    print(f"  ✓ EPS beat consensus by ${eps_beat:+.2f} ({fnum(surprise_pct, '+.1f', suffix='%')})")
elif eps_beat is not None and eps_beat < 0:
    bearish_signals += 1
    print(f"  ✗ EPS missed consensus by ${eps_beat:.2f}")
elif eps_beat is not None:
    print(f"  ~ EPS in-line with consensus (${eps_beat:+.2f})")
else:
    print(f"  ~ EPS vs. consensus: no matching estimate found near this quarter (data gap, not counted as a signal)")

if rev_yoy is not None and rev_yoy > 5:
    bullish_signals += 1
    print(f"  ✓ Revenue growth {rev_yoy:+.1f}% YoY — healthy top-line momentum")
elif rev_yoy is not None and rev_yoy > 0:
    print(f"  ~ Revenue growth {rev_yoy:+.1f}% YoY — modest but positive")
elif rev_yoy is not None:
    bearish_signals += 1
    print(f"  ✗ Revenue declined {rev_yoy:.1f}% YoY")
else:
    print(f"  ~ Revenue YoY: no prior-year quarter available for comparison (data gap, not counted as a signal)")

if gm_delta is None:
    print(f"  ~ Gross margin YoY: no prior-year comparison (data gap, not counted as a signal)")
elif gm_delta > 0:
    bullish_signals += 1
    print(f"  ✓ Gross margin expanded {gm_delta:.0f}bps YoY → pricing power intact")
elif gm_delta < -50:
    bearish_signals += 1
    print(f"  ✗ Gross margin compressed {abs(gm_delta):.0f}bps YoY → cost pressure")
else:
    print(f"  ~ Gross margin {gm_delta:+.0f}bps YoY — largely stable")

# FCF judged as a margin so the test scales with company size (was a fixed $25B, which
# only mega-caps could ever pass)
fcf_margin = fcf / rev if (fcf is not None and rev) else None
if fcf_margin is not None and fcf_margin >= 0.20:
    bullish_signals += 1
    print(f"  ✓ Strong free cash flow {pp(fcf,'B')} ({fcf_margin*100:.0f}% of revenue) — supports buybacks/dividends")
elif fcf_margin is not None and fcf_margin < 0:
    bearish_signals += 1
    print(f"  ✗ Negative free cash flow {pp(fcf,'B')} in quarter")

if pt_upside is not None and pt_upside > 10:
    bullish_signals += 1
    print(f"  ✓ Analyst consensus PT ${pt_mean:.0f} implies {pt_upside:+.1f}% upside")
elif pt_upside is not None and pt_upside < 0:
    bearish_signals += 1
    print(f"  ✗ Analyst consensus PT ${pt_mean:.0f} is below the current price ({pt_upside:+.1f}%)")

print()
# net count, so bearish signals actually pull the rating down
net_signals = bullish_signals - bearish_signals
overall = "BUY / OUTPERFORM" if net_signals >= 2 else ("UNDERPERFORM" if net_signals <= -1 else "HOLD")
print(f"  Overall signal: {bullish_signals} bullish / {bearish_signals} bearish → {overall}")

print()
print("=" * 60)
print("  Data source: Yahoo Finance (yfinance) — no paid API needed")
print("=" * 60)

# ── 5. CHARTS ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f"{COMPANY_NAME} ({TICKER}) — Earnings Dashboard\n{qname} | Yahoo Finance Data",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

quarters  = [fmt_quarter(c) for c in qf.columns[:8]][::-1]
# missing values become NaN so the bar/line is left out instead of crashing matplotlib
revenues  = [nan(b(get(qf, 'Total Revenue', c))) for c in qf.columns[:8]][::-1]
eps_vals  = [nan(get(qf, 'Diluted EPS', c)) for c in qf.columns[:8]][::-1]
gm_vals   = [nan(ratio(get(qf,'Gross Profit',c), get(qf,'Total Revenue',c), 100)) for c in qf.columns[:8]][::-1]
om_vals   = [nan(ratio(get(qf,'Operating Income',c), get(qf,'Total Revenue',c), 100)) for c in qf.columns[:8]][::-1]
fcf_vals  = [nan(b(get(cf, 'Free Cash Flow', c))) for c in qf.columns[:8]][::-1]

colors = ['#1f77b4'] * len(quarters)
colors[-1] = '#d62728'  # highlight latest quarter

# Chart 1: Quarterly Revenue
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(len(quarters)), revenues, color=colors)
ax1.set_xticks(range(len(quarters)))
ax1.set_xticklabels(quarters, rotation=45, ha='right', fontsize=7)
ax1.set_title('Quarterly Revenue ($B)', fontweight='bold', fontsize=9)
ax1.set_ylabel('$B')
for bar, val in zip(bars[-2:], revenues[-2:]):
    if pd.notna(val): ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'${val:.0f}B', ha='center', va='bottom', fontsize=7)

# Chart 2: Quarterly EPS
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(range(len(quarters)), eps_vals, color=colors)
ax2.set_xticks(range(len(quarters)))
ax2.set_xticklabels(quarters, rotation=45, ha='right', fontsize=7)
ax2.set_title('EPS: GAAP bars vs. street consensus', fontweight='bold', fontsize=9)
ax2.set_ylabel('EPS ($)')

# EPS consensus overlay. Estimates are keyed by REPORT date, bars by QUARTER-END date, so
# map each quarter to its report with the same window used for the headline match above
# (keying both by month label never lined up: a Mar quarter is reported in Apr/May).
est_line, street_line = [], []
for c in list(qf.columns[:8])[::-1]:
    w = reported_dates[(reported_dates.index >= c - pd.Timedelta(days=10)) &
                       (reported_dates.index <= c + pd.Timedelta(days=45))] \
        if not reported_dates.empty else reported_dates
    r0 = w.loc[w.index.min()] if not w.empty else None
    est_line.append(float(r0['EPS Estimate']) if r0 is not None and pd.notna(r0['EPS Estimate']) else None)
    street_line.append(float(r0['Reported EPS']) if r0 is not None and pd.notna(r0['Reported EPS']) else None)
for series, style, lbl in [(est_line, 'k--', 'Consensus (street)'),
                           (street_line, 'go', 'Reported (street)')]:
    xs = [i for i, v in enumerate(series) if v is not None]
    if xs:
        ax2.plot(xs, [series[i] for i in xs], style, linewidth=1, markersize=4, label=lbl, zorder=5)
ax2.legend(fontsize=7)

# Chart 3: Margin trend
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(range(len(quarters)), gm_vals, 'b-o', label='Gross Margin', linewidth=2, markersize=4)
ax3.plot(range(len(quarters)), om_vals, 'r-s', label='Op. Margin', linewidth=2, markersize=4)
ax3.set_xticks(range(len(quarters)))
ax3.set_xticklabels(quarters, rotation=45, ha='right', fontsize=7)
ax3.set_title('Margin Trends (%)', fontweight='bold', fontsize=9)
ax3.set_ylabel('%')
ax3.legend(fontsize=7)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Chart 4: Free Cash Flow
ax4 = fig.add_subplot(gs[1, 0])
fcf_colors = ['#2ca02c' if (v and v > 0) else '#d62728' for v in fcf_vals]
ax4.bar(range(len(quarters)), fcf_vals, color=fcf_colors)
ax4.set_xticks(range(len(quarters)))
ax4.set_xticklabels(quarters, rotation=45, ha='right', fontsize=7)
ax4.set_title('Free Cash Flow ($B)', fontweight='bold', fontsize=9)
ax4.set_ylabel('$B')

# Chart 5: EPS Beat/Miss history
ax5 = fig.add_subplot(gs[1, 1])
ed_plot = edates.dropna(subset=['Reported EPS', 'EPS Estimate', 'Surprise(%)']).head(6)
surps = ed_plot['Surprise(%)'].values.astype(float)[::-1]
qlabels5 = [i.strftime('%b %y') for i in ed_plot.index[::-1]]
bar_colors = ['#2ca02c' if s > 0 else ('#d62728' if s < 0 else '#7f7f7f') for s in surps]
ax5.bar(range(len(surps)), surps, color=bar_colors)
ax5.axhline(0, color='black', linewidth=0.8)
ax5.set_xticks(range(len(surps)))
ax5.set_xticklabels(qlabels5, rotation=45, ha='right', fontsize=7)
ax5.set_title('EPS Surprise % (Beat/Miss)', fontweight='bold', fontsize=9)
ax5.set_ylabel('Surprise %')
for i, s in enumerate(surps):
    ax5.text(i, s + (0.2 if s>=0 else -0.5), f'{s:+.1f}%', ha='center', fontsize=7)

# Chart 6: Analyst price target gauge
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis('off')
ax6.set_title('Analyst Consensus', fontweight='bold', fontsize=9)

# Simple valuation summary as text
lines = [
    f"Current Price:  {fnum(price, '.2f', '$')}",
    f"",
    f"PT Low:         {fnum(pt_low, '.0f', '$')}",
    f"PT Mean:        {fnum(pt_mean, '.0f', '$')}",
    f"PT High:        {fnum(pt_high, '.0f', '$')}",
    f"",
    f"Upside to PT:   {fnum(pt_upside, '+.1f', suffix='%')}",
    f"",
    f"Trailing P/E:   {trail_pe:.1f}x" if trail_pe else "",
    f"Forward P/E:    {fwd_pe:.1f}x" if fwd_pe else "",
    f"",
    f"Rating: {rec_str(rec_mean)}",
    f"({n_analysts} analysts)",
]
y = 0.95
for line in lines:
    weight = 'bold' if 'Rating' in line or 'Upside' in line else 'normal'
    color  = '#d62728' if 'Upside' in line else 'black'
    ax6.text(0.1, y, line, transform=ax6.transAxes,
             fontsize=8.5, verticalalignment='top',
             fontweight=weight, color=color)
    y -= 0.08

out_path = f"{OUT_DIR}/{TICKER.lower()}_earnings_dashboard.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved → {out_path}")
plt.close()
