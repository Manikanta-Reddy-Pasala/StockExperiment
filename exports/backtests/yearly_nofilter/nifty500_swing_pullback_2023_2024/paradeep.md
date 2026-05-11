# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 121.59
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 0.05% / 1.28%
- **Sum % (uncompounded):** 0.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.05% | 0.3% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.05% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.05% | 0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 00:00:00 | 66.35 | 59.10 | 63.91 | Stage2 pullback-breakout RSI=60 vol=1.8x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 00:00:00 | 70.71 | 59.31 | 64.88 | T1 booked 50% @ 70.71 |
| Target hit | 2023-09-12 00:00:00 | 67.20 | 60.73 | 69.48 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-12-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 00:00:00 | 69.30 | 63.02 | 65.18 | Stage2 pullback-breakout RSI=68 vol=3.3x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 00:00:00 | 73.28 | 63.33 | 66.90 | T1 booked 50% @ 73.28 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 69.30 | 63.67 | 68.19 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2024-01-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 00:00:00 | 81.60 | 65.98 | 75.92 | Stage2 pullback-breakout RSI=68 vol=1.6x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-01-24 00:00:00 | 76.72 | 66.23 | 76.45 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-02-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 00:00:00 | 82.15 | 67.38 | 77.53 | Stage2 pullback-breakout RSI=59 vol=6.3x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-02-13 00:00:00 | 76.13 | 67.73 | 77.88 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-22 00:00:00 | 66.35 | 2023-08-24 00:00:00 | 70.71 | PARTIAL | 0.50 | 6.57% |
| BUY | retest1 | 2023-08-22 00:00:00 | 66.35 | 2023-09-12 00:00:00 | 67.20 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2023-12-06 00:00:00 | 69.30 | 2023-12-13 00:00:00 | 73.28 | PARTIAL | 0.50 | 5.74% |
| BUY | retest1 | 2023-12-06 00:00:00 | 69.30 | 2023-12-20 00:00:00 | 69.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-20 00:00:00 | 81.60 | 2024-01-24 00:00:00 | 76.72 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest1 | 2024-02-08 00:00:00 | 82.15 | 2024-02-13 00:00:00 | 76.13 | STOP_HIT | 1.00 | -7.33% |
