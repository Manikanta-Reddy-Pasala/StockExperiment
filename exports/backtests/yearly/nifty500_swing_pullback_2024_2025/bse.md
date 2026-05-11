# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 3907.40
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 1
- **Avg / median % per leg:** -2.82% / -5.94%
- **Sum % (uncompounded):** -16.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -2.82% | -16.9% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | -2.82% | -16.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 2 | 33.3% | 1 | 4 | 1 | -2.82% | -16.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 05:30:00 | 852.15 | 769.08 | 806.60 | Stage2 pullback-breakout RSI=59 vol=3.1x ATR=33.75 |
| Stop hit — per-position SL triggered | 2024-08-05 05:30:00 | 801.52 | 771.28 | 815.52 | SL hit (bars_held=3) |

### Cycle 2 — BUY (started 2024-12-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 05:30:00 | 1731.63 | 1083.59 | 1537.99 | Stage2 pullback-breakout RSI=68 vol=3.5x ATR=79.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 05:30:00 | 1891.59 | 1120.15 | 1655.44 | T1 booked 50% @ 1891.59 |
| Target hit | 2024-12-27 05:30:00 | 1759.50 | 1190.17 | 1775.58 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-01-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 05:30:00 | 1928.67 | 1262.25 | 1783.54 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=78.66 |
| Stop hit — per-position SL triggered | 2025-01-27 05:30:00 | 1810.69 | 1314.76 | 1867.35 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2025-02-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 05:30:00 | 1877.00 | 1394.52 | 1807.01 | Stage2 pullback-breakout RSI=55 vol=2.0x ATR=91.82 |
| Stop hit — per-position SL triggered | 2025-02-27 05:30:00 | 1739.27 | 1417.45 | 1825.38 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2025-03-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 05:30:00 | 1826.60 | 1425.94 | 1549.24 | Stage2 pullback-breakout RSI=64 vol=3.0x ATR=101.58 |
| Stop hit — per-position SL triggered | 2025-04-07 05:30:00 | 1674.23 | 1445.59 | 1657.57 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-31 05:30:00 | 852.15 | 2024-08-05 05:30:00 | 801.52 | STOP_HIT | 1.00 | -5.94% |
| BUY | retest1 | 2024-12-05 05:30:00 | 1731.63 | 2024-12-12 05:30:00 | 1891.59 | PARTIAL | 0.50 | 9.24% |
| BUY | retest1 | 2024-12-05 05:30:00 | 1731.63 | 2024-12-27 05:30:00 | 1759.50 | TARGET_HIT | 0.50 | 1.61% |
| BUY | retest1 | 2025-01-15 05:30:00 | 1928.67 | 2025-01-27 05:30:00 | 1810.69 | STOP_HIT | 1.00 | -6.12% |
| BUY | retest1 | 2025-02-19 05:30:00 | 1877.00 | 2025-02-27 05:30:00 | 1739.27 | STOP_HIT | 1.00 | -7.34% |
| BUY | retest1 | 2025-03-28 05:30:00 | 1826.60 | 2025-04-07 05:30:00 | 1674.23 | STOP_HIT | 1.00 | -8.34% |
