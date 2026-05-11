# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 923.80
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 3
- **Target hits / Stop hits / Partials:** 3 / 4 / 4
- **Avg / median % per leg:** 8.59% / 6.93%
- **Sum % (uncompounded):** 94.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 8 | 72.7% | 3 | 4 | 4 | 8.59% | 94.5% |
| BUY @ 2nd Alert (retest1) | 11 | 8 | 72.7% | 3 | 4 | 4 | 8.59% | 94.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 8 | 72.7% | 3 | 4 | 4 | 8.59% | 94.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 00:00:00 | 497.90 | 383.58 | 475.82 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=16.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 00:00:00 | 529.96 | 390.56 | 489.43 | T1 booked 50% @ 529.96 |
| Target hit | 2023-09-04 00:00:00 | 720.20 | 500.99 | 751.81 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-10-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 00:00:00 | 739.10 | 555.49 | 706.68 | Stage2 pullback-breakout RSI=61 vol=2.7x ATR=27.57 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 697.74 | 558.52 | 706.96 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-10-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 00:00:00 | 768.55 | 564.03 | 717.28 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=33.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 00:00:00 | 835.86 | 566.71 | 728.31 | T1 booked 50% @ 835.86 |
| Target hit | 2023-11-23 00:00:00 | 821.80 | 610.60 | 824.84 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 853.90 | 639.79 | 821.81 | Stage2 pullback-breakout RSI=60 vol=2.7x ATR=30.98 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 807.43 | 645.42 | 823.76 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-01-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 00:00:00 | 869.70 | 660.85 | 836.17 | Stage2 pullback-breakout RSI=60 vol=2.4x ATR=34.90 |
| Stop hit — per-position SL triggered | 2024-01-16 00:00:00 | 879.95 | 683.82 | 876.66 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-01-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 00:00:00 | 933.30 | 702.96 | 895.18 | Stage2 pullback-breakout RSI=61 vol=2.2x ATR=35.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 00:00:00 | 1003.84 | 713.52 | 920.73 | T1 booked 50% @ 1003.84 |
| Target hit | 2024-03-13 00:00:00 | 1078.50 | 833.38 | 1241.39 | Trail-exit close<EMA20 |

### Cycle 7 — BUY (started 2024-03-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 00:00:00 | 1408.35 | 860.15 | 1267.59 | Stage2 pullback-breakout RSI=61 vol=2.4x ATR=101.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-01 00:00:00 | 1612.33 | 888.75 | 1341.40 | T1 booked 50% @ 1612.33 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 1408.35 | 944.71 | 1452.60 | SL hit (bars_held=14) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-27 00:00:00 | 497.90 | 2023-07-06 00:00:00 | 529.96 | PARTIAL | 0.50 | 6.44% |
| BUY | retest1 | 2023-06-27 00:00:00 | 497.90 | 2023-09-04 00:00:00 | 720.20 | TARGET_HIT | 0.50 | 44.65% |
| BUY | retest1 | 2023-10-20 00:00:00 | 739.10 | 2023-10-25 00:00:00 | 697.74 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest1 | 2023-10-30 00:00:00 | 768.55 | 2023-10-31 00:00:00 | 835.86 | PARTIAL | 0.50 | 8.76% |
| BUY | retest1 | 2023-10-30 00:00:00 | 768.55 | 2023-11-23 00:00:00 | 821.80 | TARGET_HIT | 0.50 | 6.93% |
| BUY | retest1 | 2023-12-15 00:00:00 | 853.90 | 2023-12-20 00:00:00 | 807.43 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest1 | 2024-01-02 00:00:00 | 869.70 | 2024-01-16 00:00:00 | 879.95 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest1 | 2024-01-30 00:00:00 | 933.30 | 2024-02-05 00:00:00 | 1003.84 | PARTIAL | 0.50 | 7.56% |
| BUY | retest1 | 2024-01-30 00:00:00 | 933.30 | 2024-03-13 00:00:00 | 1078.50 | TARGET_HIT | 0.50 | 15.56% |
| BUY | retest1 | 2024-03-21 00:00:00 | 1408.35 | 2024-04-01 00:00:00 | 1612.33 | PARTIAL | 0.50 | 14.48% |
| BUY | retest1 | 2024-03-21 00:00:00 | 1408.35 | 2024-04-15 00:00:00 | 1408.35 | STOP_HIT | 0.50 | 0.00% |
