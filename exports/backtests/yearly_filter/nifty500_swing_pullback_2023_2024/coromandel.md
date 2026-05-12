# Coromandel International Ltd. (COROMANDEL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1901.00
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
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 1
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 2.92% / 4.13%
- **Sum % (uncompounded):** 17.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 2 | 2 | 2 | 2.92% | 17.5% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 2 | 2 | 2.92% | 17.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 2 | 2 | 2 | 2.92% | 17.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 00:00:00 | 992.80 | 945.92 | 961.80 | Stage2 pullback-breakout RSI=63 vol=1.7x ATR=22.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 00:00:00 | 1038.28 | 948.60 | 979.79 | T1 booked 50% @ 1038.28 |
| Target hit | 2023-09-20 00:00:00 | 1091.10 | 989.76 | 1105.58 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 00:00:00 | 1146.30 | 998.29 | 1112.30 | Stage2 pullback-breakout RSI=64 vol=1.7x ATR=28.81 |
| Stop hit — per-position SL triggered | 2023-10-16 00:00:00 | 1154.60 | 1013.35 | 1140.11 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 00:00:00 | 1163.15 | 1039.32 | 1121.96 | Stage2 pullback-breakout RSI=67 vol=2.2x ATR=24.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 00:00:00 | 1211.15 | 1045.50 | 1147.44 | T1 booked 50% @ 1211.15 |
| Target hit | 2023-12-20 00:00:00 | 1187.75 | 1063.54 | 1201.37 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 00:00:00 | 1114.55 | 1091.37 | 1091.23 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=28.96 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 1071.11 | 1091.76 | 1094.33 | SL hit (bars_held=3) |

### Cycle 5 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 1184.50 | 1096.75 | 1121.44 | Stage2 pullback-breakout RSI=65 vol=4.3x ATR=36.33 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-26 00:00:00 | 992.80 | 2023-08-01 00:00:00 | 1038.28 | PARTIAL | 0.50 | 4.58% |
| BUY | retest1 | 2023-07-26 00:00:00 | 992.80 | 2023-09-20 00:00:00 | 1091.10 | TARGET_HIT | 0.50 | 9.90% |
| BUY | retest1 | 2023-09-29 00:00:00 | 1146.30 | 2023-10-16 00:00:00 | 1154.60 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest1 | 2023-11-30 00:00:00 | 1163.15 | 2023-12-06 00:00:00 | 1211.15 | PARTIAL | 0.50 | 4.13% |
| BUY | retest1 | 2023-11-30 00:00:00 | 1163.15 | 2023-12-20 00:00:00 | 1187.75 | TARGET_HIT | 0.50 | 2.11% |
| BUY | retest1 | 2024-03-07 00:00:00 | 1114.55 | 2024-03-13 00:00:00 | 1071.11 | STOP_HIT | 1.00 | -3.90% |
