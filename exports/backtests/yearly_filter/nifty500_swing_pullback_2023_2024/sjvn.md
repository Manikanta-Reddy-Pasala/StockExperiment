# SJVN Ltd. (SJVN)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 77.26
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 5.28% / 7.93%
- **Sum % (uncompounded):** 36.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 5.28% | 36.9% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 5.28% | 36.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 1 | 3 | 3 | 5.28% | 36.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 00:00:00 | 60.80 | 40.94 | 56.42 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 00:00:00 | 66.01 | 42.02 | 59.05 | T1 booked 50% @ 66.01 |
| Target hit | 2023-09-26 00:00:00 | 68.55 | 46.11 | 69.19 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 00:00:00 | 81.50 | 54.71 | 75.03 | Stage2 pullback-breakout RSI=67 vol=3.7x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 88.02 | 56.94 | 80.05 | T1 booked 50% @ 88.02 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 81.50 | 60.73 | 87.54 | SL hit (bars_held=20) |

### Cycle 3 — BUY (started 2024-01-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 00:00:00 | 102.50 | 65.81 | 92.88 | Stage2 pullback-breakout RSI=70 vol=3.8x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 00:00:00 | 110.63 | 67.66 | 97.31 | T1 booked 50% @ 110.63 |
| Stop hit — per-position SL triggered | 2024-01-23 00:00:00 | 102.50 | 68.02 | 97.94 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-03-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 00:00:00 | 124.60 | 85.31 | 119.27 | Stage2 pullback-breakout RSI=54 vol=3.3x ATR=9.36 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 124.20 | 92.16 | 127.03 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-28 00:00:00 | 60.80 | 2023-09-04 00:00:00 | 66.01 | PARTIAL | 0.50 | 8.57% |
| BUY | retest1 | 2023-08-28 00:00:00 | 60.80 | 2023-09-26 00:00:00 | 68.55 | TARGET_HIT | 0.50 | 12.75% |
| BUY | retest1 | 2023-11-21 00:00:00 | 81.50 | 2023-12-04 00:00:00 | 88.02 | PARTIAL | 0.50 | 8.00% |
| BUY | retest1 | 2023-11-21 00:00:00 | 81.50 | 2023-12-20 00:00:00 | 81.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-15 00:00:00 | 102.50 | 2024-01-20 00:00:00 | 110.63 | PARTIAL | 0.50 | 7.93% |
| BUY | retest1 | 2024-01-15 00:00:00 | 102.50 | 2024-01-23 00:00:00 | 102.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-15 00:00:00 | 124.60 | 2024-04-15 00:00:00 | 124.20 | STOP_HIT | 1.00 | -0.32% |
