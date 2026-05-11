# Oracle Financial Services Software Ltd. (OFSS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 9230.50
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / Stop hits / Partials:** 1 / 6 / 3
- **Avg / median % per leg:** 8.08% / 0.00%
- **Sum % (uncompounded):** 80.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 8.08% | 80.8% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 8.08% | 80.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 4 | 40.0% | 1 | 6 | 3 | 8.08% | 80.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 4021.50 | 3380.08 | 3823.93 | Stage2 pullback-breakout RSI=68 vol=3.2x ATR=91.41 |
| Stop hit — per-position SL triggered | 2023-07-21 00:00:00 | 3884.39 | 3406.49 | 3861.07 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-08-16 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 00:00:00 | 4099.05 | 3484.26 | 3922.17 | Stage2 pullback-breakout RSI=69 vol=5.3x ATR=86.21 |
| Stop hit — per-position SL triggered | 2023-08-18 00:00:00 | 3969.73 | 3494.98 | 3940.72 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-08-31 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 00:00:00 | 4109.30 | 3538.94 | 3984.31 | Stage2 pullback-breakout RSI=65 vol=1.6x ATR=79.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 00:00:00 | 4267.85 | 3570.81 | 4068.58 | T1 booked 50% @ 4267.85 |
| Stop hit — per-position SL triggered | 2023-09-13 00:00:00 | 4109.30 | 3599.79 | 4146.20 | SL hit (bars_held=9) |

### Cycle 4 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 4180.00 | 3764.17 | 4037.24 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=88.06 |
| Stop hit — per-position SL triggered | 2023-11-22 00:00:00 | 4047.90 | 3775.38 | 4063.92 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2023-12-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 00:00:00 | 4360.75 | 3823.88 | 4117.40 | Stage2 pullback-breakout RSI=68 vol=2.6x ATR=108.57 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 4197.90 | 3838.29 | 4166.23 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-01-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 00:00:00 | 4338.85 | 3863.86 | 4194.90 | Stage2 pullback-breakout RSI=62 vol=2.4x ATR=100.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 00:00:00 | 4539.31 | 3884.88 | 4262.50 | T1 booked 50% @ 4539.31 |
| Stop hit — per-position SL triggered | 2024-01-08 00:00:00 | 4338.85 | 3889.52 | 4270.99 | SL hit (bars_held=5) |

### Cycle 7 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 4656.75 | 3910.24 | 4322.98 | Stage2 pullback-breakout RSI=69 vol=3.2x ATR=131.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 00:00:00 | 4919.56 | 3921.78 | 4394.18 | T1 booked 50% @ 4919.56 |
| Target hit | 2024-04-09 00:00:00 | 8390.40 | 5628.90 | 8485.57 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-14 00:00:00 | 4021.50 | 2023-07-21 00:00:00 | 3884.39 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest1 | 2023-08-16 00:00:00 | 4099.05 | 2023-08-18 00:00:00 | 3969.73 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest1 | 2023-08-31 00:00:00 | 4109.30 | 2023-09-07 00:00:00 | 4267.85 | PARTIAL | 0.50 | 3.86% |
| BUY | retest1 | 2023-08-31 00:00:00 | 4109.30 | 2023-09-13 00:00:00 | 4109.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-17 00:00:00 | 4180.00 | 2023-11-22 00:00:00 | 4047.90 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest1 | 2023-12-15 00:00:00 | 4360.75 | 2023-12-20 00:00:00 | 4197.90 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest1 | 2024-01-01 00:00:00 | 4338.85 | 2024-01-05 00:00:00 | 4539.31 | PARTIAL | 0.50 | 4.62% |
| BUY | retest1 | 2024-01-01 00:00:00 | 4338.85 | 2024-01-08 00:00:00 | 4338.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-12 00:00:00 | 4656.75 | 2024-01-15 00:00:00 | 4919.56 | PARTIAL | 0.50 | 5.64% |
| BUY | retest1 | 2024-01-12 00:00:00 | 4656.75 | 2024-04-09 00:00:00 | 8390.40 | TARGET_HIT | 0.50 | 80.18% |
