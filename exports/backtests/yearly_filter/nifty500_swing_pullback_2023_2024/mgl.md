# Mahanagar Gas Ltd. (MGL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1120.10
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
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 4
- **Target hits / Stop hits / Partials:** 2 / 5 / 3
- **Avg / median % per leg:** 3.35% / 4.37%
- **Sum % (uncompounded):** 33.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 2 | 5 | 3 | 3.35% | 33.5% |
| BUY @ 2nd Alert (retest1) | 10 | 6 | 60.0% | 2 | 5 | 3 | 3.35% | 33.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 6 | 60.0% | 2 | 5 | 3 | 3.35% | 33.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 1080.95 | 953.13 | 1040.43 | Stage2 pullback-breakout RSI=64 vol=2.4x ATR=23.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 00:00:00 | 1128.18 | 963.01 | 1071.23 | T1 booked 50% @ 1128.18 |
| Stop hit — per-position SL triggered | 2023-07-14 00:00:00 | 1080.95 | 965.40 | 1073.41 | SL hit (bars_held=9) |

### Cycle 2 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 1112.05 | 975.93 | 1077.31 | Stage2 pullback-breakout RSI=63 vol=1.8x ATR=25.29 |
| Stop hit — per-position SL triggered | 2023-08-04 00:00:00 | 1074.11 | 982.11 | 1086.33 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2023-09-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 00:00:00 | 1073.80 | 989.44 | 1030.60 | Stage2 pullback-breakout RSI=64 vol=3.2x ATR=23.25 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 1038.92 | 991.58 | 1034.55 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-10-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 00:00:00 | 1109.10 | 997.04 | 1037.02 | Stage2 pullback-breakout RSI=66 vol=6.9x ATR=27.41 |
| Stop hit — per-position SL triggered | 2023-10-17 00:00:00 | 1140.60 | 1008.85 | 1091.55 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 1063.10 | 1016.59 | 1040.43 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=23.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 1109.96 | 1017.41 | 1045.99 | T1 booked 50% @ 1109.96 |
| Target hit | 2024-01-10 00:00:00 | 1179.60 | 1056.18 | 1186.75 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 1264.20 | 1060.06 | 1198.72 | Stage2 pullback-breakout RSI=69 vol=1.7x ATR=33.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 00:00:00 | 1331.17 | 1070.87 | 1232.62 | T1 booked 50% @ 1331.17 |
| Target hit | 2024-02-28 00:00:00 | 1466.00 | 1165.37 | 1475.80 | Trail-exit close<EMA20 |

### Cycle 7 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 1565.40 | 1182.60 | 1494.11 | Stage2 pullback-breakout RSI=69 vol=2.1x ATR=38.60 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 1507.51 | 1184.06 | 1478.44 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 1080.95 | 2023-07-12 00:00:00 | 1128.18 | PARTIAL | 0.50 | 4.37% |
| BUY | retest1 | 2023-07-03 00:00:00 | 1080.95 | 2023-07-14 00:00:00 | 1080.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-28 00:00:00 | 1112.05 | 2023-08-04 00:00:00 | 1074.11 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest1 | 2023-09-06 00:00:00 | 1073.80 | 2023-09-12 00:00:00 | 1038.92 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest1 | 2023-10-03 00:00:00 | 1109.10 | 2023-10-17 00:00:00 | 1140.60 | STOP_HIT | 1.00 | 2.84% |
| BUY | retest1 | 2023-12-01 00:00:00 | 1063.10 | 2023-12-04 00:00:00 | 1109.96 | PARTIAL | 0.50 | 4.41% |
| BUY | retest1 | 2023-12-01 00:00:00 | 1063.10 | 2024-01-10 00:00:00 | 1179.60 | TARGET_HIT | 0.50 | 10.96% |
| BUY | retest1 | 2024-01-12 00:00:00 | 1264.20 | 2024-01-19 00:00:00 | 1331.17 | PARTIAL | 0.50 | 5.30% |
| BUY | retest1 | 2024-01-12 00:00:00 | 1264.20 | 2024-02-28 00:00:00 | 1466.00 | TARGET_HIT | 0.50 | 15.96% |
| BUY | retest1 | 2024-03-05 00:00:00 | 1565.40 | 2024-03-06 00:00:00 | 1507.51 | STOP_HIT | 1.00 | -3.70% |
