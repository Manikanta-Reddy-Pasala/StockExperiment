# Sun Pharmaceutical Industries Ltd. (SUNPHARMA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1847.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 1
- **Avg / median % per leg:** 0.24% / -2.28%
- **Sum % (uncompounded):** 0.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.24% | 0.7% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.24% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.24% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 05:30:00 | 1782.60 | 1716.33 | 1751.56 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=27.13 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 1741.91 | 1716.90 | 1750.28 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2026-03-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 05:30:00 | 1799.40 | 1711.52 | 1737.08 | Stage2 pullback-breakout RSI=66 vol=1.7x ATR=33.09 |
| Stop hit — per-position SL triggered | 2026-03-19 05:30:00 | 1749.76 | 1718.84 | 1769.49 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2026-04-27 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 05:30:00 | 1733.50 | 1716.38 | 1698.10 | Stage2 pullback-breakout RSI=54 vol=5.3x ATR=49.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 05:30:00 | 1833.22 | 1719.26 | 1729.29 | T1 booked 50% @ 1833.22 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-01-07 05:30:00 | 1782.60 | 2026-01-09 05:30:00 | 1741.91 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2026-03-06 05:30:00 | 1799.40 | 2026-03-19 05:30:00 | 1749.76 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest1 | 2026-04-27 05:30:00 | 1733.50 | 2026-05-04 05:30:00 | 1833.22 | PARTIAL | 0.50 | 5.75% |
