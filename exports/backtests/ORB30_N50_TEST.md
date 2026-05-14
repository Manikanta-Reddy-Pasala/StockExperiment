# ORB-30 — Nifty 50 vs Nifty 100 (Conservative Universe Test)

_6-month window Nov 2025 → May 2026. ₹10L capital. 0.13% round-trip cost. 1 concurrent position. Exit by 15:00._

## Hypothesis
User asked: "use Nifty 50 / 100 only — won't drop below -20%". Conservative universe = safer intraday returns.

## Result: REJECTED. N50 WORSE than N100.

| Variant | 6-mo Total | Avg/mo | Win% | MaxDD | Sharpe | Trades |
|---------|----------:|------:|----:|-----:|------:|------:|
| N100_top_by_ADV (30 stocks) | **+2.82%** | **+0.40%** | 50.0 | -10.6 | **+0.27** | 158 |
| N50_long_only (53 stocks) | -2.33% | -0.33% | 47.7 | -11.3 | -0.32 | 155 |
| N50_both_sides (53 stocks) | -1.01% | -0.14% | 50.3 | -13.3 | -0.13 | 157 |

## Per-Month ROI

| Variant | Nov-25 | Dec-25 | Jan-26 | Feb-26 | Mar-26 | Apr-26 | May-26 |
|---------|------:|------:|------:|------:|------:|------:|------:|
| N100 top by ADV | -0.32 | -1.02 | +4.04 | +1.42 | +0.08 | -3.77 | +2.40 |
| N50 long-only | -3.70 | -0.94 | +4.46 | -3.61 | -2.80 | +0.23 | +4.03 |
| N50 both-sides | -0.14 | +0.36 | -0.62 | -8.38 | +7.93 | -1.65 | +1.49 |

## Why N50 Fails

The "safety" thesis is irrelevant for ORB-30. Strategy exits by 15:00 daily → no overnight tail-risk exposure. DD comes from accumulated intraday losses, not crashes.

**N50 trades in tight ORB ranges that get chopped:**
- Tight range = small target = costs eat profit
- Choppy day = stopped repeatedly
- HDFCBANK / RELIANCE / ICICIBANK barely move 0.5-1% intraday → ORB target rarely hit

**N100 by ADV includes more volatile names** (ETERNAL, MCX, BAJFINANCE) whose ranges expand on breakout → real edge.

**Restricting volatility KILLS the edge.**

## Top 5 N50 Stocks (Long-Only, tiny sample)

| Stock | Trades | P&L | Win% |
|-------|------:|----:|----:|
| APOLLOHOSP | 5 | +₹39,123 | 80% |
| TMPV | 5 | +₹23,466 | 80% |
| ULTRACEMCO | 4 | +₹22,219 | 75% |
| BAJAJ-AUTO | 4 | +₹17,099 | 75% |
| ADANIENT | 1 | +₹15,750 | 100% |

Sample size 1-5 trades per stock = not statistically reliable. Different leader-board on both-sides variant (AXISBANK / TRENT / SBIN) confirms signal noise.

## Verdict

| Claim | Result |
|-------|-------|
| "N50 = safer" | False for intraday — DD GOT WORSE (-13.3 vs -10.6) |
| "5-10%/mo achievable on N50" | Got -0.33%/mo (worst variant) |
| Conservative universe wins | Loses on EVERY axis: ROI, DD, Sharpe |

## Recommendation

**Conservative N50 intraday is NOT viable for ORB-30.**

Best option remains:
1. **Stay with momentum rotation Model 3** (+87%/yr validated, 6% DD) — long-only positional, no intraday needed
2. If still want intraday: stick to N100 by ADV (+0.4%/mo marginal but at least non-negative)
3. Do NOT restrict to N50 — kills the volatility-driven edge

## Files

```
exports/backtests/ORB30_N50_TEST.md
tools/backtests/intraday_orb30_n50.py
```
