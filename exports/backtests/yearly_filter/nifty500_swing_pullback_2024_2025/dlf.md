# DLF Ltd. (DLF)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 608.25
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
- **Avg / median % per leg:** 0.47% / 0.90%
- **Sum % (uncompounded):** 2.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.47% | 2.8% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.47% | 2.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.47% | 2.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 00:00:00 | 870.00 | 779.67 | 833.39 | Stage2 pullback-breakout RSI=60 vol=2.2x ATR=27.21 |
| Stop hit — per-position SL triggered | 2024-08-05 00:00:00 | 829.19 | 783.55 | 842.19 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2024-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 00:00:00 | 866.90 | 787.39 | 837.98 | Stage2 pullback-breakout RSI=57 vol=1.9x ATR=27.95 |
| Stop hit — per-position SL triggered | 2024-08-29 00:00:00 | 824.98 | 792.84 | 844.09 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 00:00:00 | 863.60 | 797.62 | 840.35 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=20.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 00:00:00 | 904.44 | 801.85 | 854.86 | T1 booked 50% @ 904.44 |
| Stop hit — per-position SL triggered | 2024-10-03 00:00:00 | 863.60 | 808.94 | 879.64 | SL hit (bars_held=13) |

### Cycle 4 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 823.30 | 810.51 | 797.03 | Stage2 pullback-breakout RSI=55 vol=2.2x ATR=27.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 00:00:00 | 878.76 | 814.53 | 835.07 | T1 booked 50% @ 878.76 |
| Target hit | 2024-12-20 00:00:00 | 830.70 | 818.08 | 850.11 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-29 00:00:00 | 870.00 | 2024-08-05 00:00:00 | 829.19 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest1 | 2024-08-16 00:00:00 | 866.90 | 2024-08-29 00:00:00 | 824.98 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest1 | 2024-09-13 00:00:00 | 863.60 | 2024-09-23 00:00:00 | 904.44 | PARTIAL | 0.50 | 4.73% |
| BUY | retest1 | 2024-09-13 00:00:00 | 863.60 | 2024-10-03 00:00:00 | 863.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 00:00:00 | 823.30 | 2024-12-11 00:00:00 | 878.76 | PARTIAL | 0.50 | 6.74% |
| BUY | retest1 | 2024-11-25 00:00:00 | 823.30 | 2024-12-20 00:00:00 | 830.70 | TARGET_HIT | 0.50 | 0.90% |
