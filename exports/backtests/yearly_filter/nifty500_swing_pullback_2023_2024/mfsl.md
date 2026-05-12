# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1680.20
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
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 3 / 2 / 3
- **Avg / median % per leg:** 3.06% / 5.52%
- **Sum % (uncompounded):** 24.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 3 | 2 | 3 | 3.06% | 24.5% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 3 | 2 | 3 | 3.06% | 24.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 3 | 2 | 3 | 3.06% | 24.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 00:00:00 | 855.10 | 735.64 | 801.88 | Stage2 pullback-breakout RSI=66 vol=10.5x ATR=28.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 00:00:00 | 911.34 | 747.06 | 844.02 | T1 booked 50% @ 911.34 |
| Target hit | 2023-09-22 00:00:00 | 918.30 | 780.51 | 920.17 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 951.85 | 822.80 | 921.83 | Stage2 pullback-breakout RSI=59 vol=2.6x ATR=26.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 00:00:00 | 1004.43 | 831.67 | 946.79 | T1 booked 50% @ 1004.43 |
| Target hit | 2023-12-15 00:00:00 | 973.90 | 855.22 | 1003.09 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-02-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 00:00:00 | 960.25 | 875.07 | 910.14 | Stage2 pullback-breakout RSI=59 vol=5.5x ATR=32.68 |
| Stop hit — per-position SL triggered | 2024-02-21 00:00:00 | 953.10 | 884.20 | 947.31 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-03-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 00:00:00 | 987.75 | 900.38 | 961.74 | Stage2 pullback-breakout RSI=57 vol=3.7x ATR=32.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 00:00:00 | 1053.57 | 913.11 | 1004.32 | T1 booked 50% @ 1053.57 |
| Target hit | 2024-04-16 00:00:00 | 1001.80 | 915.07 | 1005.69 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-04-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 00:00:00 | 1060.30 | 918.44 | 1012.22 | Stage2 pullback-breakout RSI=64 vol=1.9x ATR=33.02 |
| Stop hit — per-position SL triggered | 2024-04-25 00:00:00 | 1010.78 | 921.91 | 1017.98 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-10 00:00:00 | 855.10 | 2023-08-24 00:00:00 | 911.34 | PARTIAL | 0.50 | 6.58% |
| BUY | retest1 | 2023-08-10 00:00:00 | 855.10 | 2023-09-22 00:00:00 | 918.30 | TARGET_HIT | 0.50 | 7.39% |
| BUY | retest1 | 2023-11-17 00:00:00 | 951.85 | 2023-11-28 00:00:00 | 1004.43 | PARTIAL | 0.50 | 5.52% |
| BUY | retest1 | 2023-11-17 00:00:00 | 951.85 | 2023-12-15 00:00:00 | 973.90 | TARGET_HIT | 0.50 | 2.32% |
| BUY | retest1 | 2024-02-07 00:00:00 | 960.25 | 2024-02-21 00:00:00 | 953.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest1 | 2024-03-26 00:00:00 | 987.75 | 2024-04-12 00:00:00 | 1053.57 | PARTIAL | 0.50 | 6.66% |
| BUY | retest1 | 2024-03-26 00:00:00 | 987.75 | 2024-04-16 00:00:00 | 1001.80 | TARGET_HIT | 0.50 | 1.42% |
| BUY | retest1 | 2024-04-22 00:00:00 | 1060.30 | 2024-04-25 00:00:00 | 1010.78 | STOP_HIT | 1.00 | -4.67% |
