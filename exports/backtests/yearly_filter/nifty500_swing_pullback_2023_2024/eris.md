# Eris Lifesciences Ltd. (ERIS)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1373.30
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
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 1.36% / 0.75%
- **Sum % (uncompounded):** 9.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.36% | 9.5% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.36% | 9.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 1.36% | 9.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 00:00:00 | 850.15 | 716.77 | 820.54 | Stage2 pullback-breakout RSI=63 vol=3.6x ATR=22.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 00:00:00 | 895.99 | 718.36 | 825.85 | T1 booked 50% @ 895.99 |
| Stop hit — per-position SL triggered | 2023-10-16 00:00:00 | 868.45 | 734.77 | 865.79 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 897.80 | 753.12 | 863.31 | Stage2 pullback-breakout RSI=62 vol=2.3x ATR=27.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 00:00:00 | 952.32 | 754.71 | 867.97 | T1 booked 50% @ 952.32 |
| Stop hit — per-position SL triggered | 2023-11-13 00:00:00 | 897.80 | 759.28 | 878.68 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-12-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 00:00:00 | 907.70 | 794.68 | 891.65 | Stage2 pullback-breakout RSI=55 vol=1.7x ATR=28.15 |
| Stop hit — per-position SL triggered | 2024-01-09 00:00:00 | 914.55 | 806.36 | 908.55 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-02-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 00:00:00 | 943.25 | 824.82 | 905.10 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=29.98 |
| Stop hit — per-position SL triggered | 2024-02-14 00:00:00 | 898.29 | 827.89 | 903.95 | SL hit (bars_held=4) |

### Cycle 5 — BUY (started 2024-04-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 00:00:00 | 870.30 | 839.05 | 854.57 | Stage2 pullback-breakout RSI=57 vol=1.5x ATR=22.32 |
| Stop hit — per-position SL triggered | 2024-05-03 00:00:00 | 869.70 | 843.29 | 873.29 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-09-28 00:00:00 | 850.15 | 2023-09-29 00:00:00 | 895.99 | PARTIAL | 0.50 | 5.39% |
| BUY | retest1 | 2023-09-28 00:00:00 | 850.15 | 2023-10-16 00:00:00 | 868.45 | STOP_HIT | 0.50 | 2.15% |
| BUY | retest1 | 2023-11-08 00:00:00 | 897.80 | 2023-11-09 00:00:00 | 952.32 | PARTIAL | 0.50 | 6.07% |
| BUY | retest1 | 2023-11-08 00:00:00 | 897.80 | 2023-11-13 00:00:00 | 897.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 00:00:00 | 907.70 | 2024-01-09 00:00:00 | 914.55 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest1 | 2024-02-08 00:00:00 | 943.25 | 2024-02-14 00:00:00 | 898.29 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest1 | 2024-04-18 00:00:00 | 870.30 | 2024-05-03 00:00:00 | 869.70 | STOP_HIT | 1.00 | -0.07% |
