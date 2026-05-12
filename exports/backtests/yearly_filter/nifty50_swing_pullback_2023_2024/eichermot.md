# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 7200.50
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 2.34% / 3.98%
- **Sum % (uncompounded):** 14.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.34% | 14.0% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.34% | 14.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 3 | 2 | 2.34% | 14.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 3630.85 | 3381.41 | 3574.40 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=69.47 |
| Stop hit — per-position SL triggered | 2023-07-04 00:00:00 | 3526.65 | 3381.61 | 3557.97 | SL hit (bars_held=1) |

### Cycle 2 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 3427.20 | 3368.95 | 3376.13 | Stage2 pullback-breakout RSI=56 vol=2.1x ATR=60.70 |
| Stop hit — per-position SL triggered | 2023-10-03 00:00:00 | 3336.15 | 3374.69 | 3408.41 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2023-11-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 00:00:00 | 3428.00 | 3384.35 | 3399.49 | Stage2 pullback-breakout RSI=52 vol=1.7x ATR=68.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 00:00:00 | 3564.32 | 3388.46 | 3431.98 | T1 booked 50% @ 3564.32 |
| Target hit | 2023-12-20 00:00:00 | 3912.40 | 3521.33 | 3959.12 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-02-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-01 00:00:00 | 3933.20 | 3602.52 | 3789.61 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=95.14 |
| Stop hit — per-position SL triggered | 2024-02-09 00:00:00 | 3790.49 | 3618.25 | 3826.20 | SL hit (bars_held=6) |

### Cycle 5 — BUY (started 2024-03-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 00:00:00 | 3873.60 | 3671.14 | 3801.88 | Stage2 pullback-breakout RSI=55 vol=4.3x ATR=98.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 00:00:00 | 4070.61 | 3684.77 | 3861.37 | T1 booked 50% @ 4070.61 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 3630.85 | 2023-07-04 00:00:00 | 3526.65 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest1 | 2023-09-15 00:00:00 | 3427.20 | 2023-10-03 00:00:00 | 3336.15 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest1 | 2023-11-03 00:00:00 | 3428.00 | 2023-11-08 00:00:00 | 3564.32 | PARTIAL | 0.50 | 3.98% |
| BUY | retest1 | 2023-11-03 00:00:00 | 3428.00 | 2023-12-20 00:00:00 | 3912.40 | TARGET_HIT | 0.50 | 14.13% |
| BUY | retest1 | 2024-02-01 00:00:00 | 3933.20 | 2024-02-09 00:00:00 | 3790.49 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest1 | 2024-03-20 00:00:00 | 3873.60 | 2024-03-28 00:00:00 | 4070.61 | PARTIAL | 0.50 | 5.09% |
