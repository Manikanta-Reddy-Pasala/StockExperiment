# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1847.90
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
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 1
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 8.96% / 4.09%
- **Sum % (uncompounded):** 53.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 2 | 2 | 2 | 8.96% | 53.8% |
| BUY @ 2nd Alert (retest1) | 6 | 5 | 83.3% | 2 | 2 | 2 | 8.96% | 53.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 5 | 83.3% | 2 | 2 | 2 | 8.96% | 53.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 05:30:00 | 1021.80 | 971.71 | 991.88 | Stage2 pullback-breakout RSI=68 vol=2.0x ATR=15.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-30 05:30:00 | 1053.64 | 972.50 | 997.57 | T1 booked 50% @ 1053.64 |
| Target hit | 2023-08-23 05:30:00 | 1123.70 | 1015.19 | 1128.60 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 05:30:00 | 1142.35 | 1024.91 | 1122.61 | Stage2 pullback-breakout RSI=62 vol=1.6x ATR=19.33 |
| Stop hit — per-position SL triggered | 2023-09-21 05:30:00 | 1146.15 | 1036.21 | 1136.81 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-02 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 05:30:00 | 1132.85 | 1058.64 | 1124.20 | Stage2 pullback-breakout RSI=52 vol=1.8x ATR=23.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 05:30:00 | 1179.20 | 1062.61 | 1136.43 | T1 booked 50% @ 1179.20 |
| Target hit | 2024-03-15 05:30:00 | 1548.20 | 1260.44 | 1554.51 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-03-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 05:30:00 | 1608.90 | 1275.48 | 1560.85 | Stage2 pullback-breakout RSI=64 vol=2.0x ATR=34.12 |
| Stop hit — per-position SL triggered | 2024-04-09 05:30:00 | 1602.55 | 1307.55 | 1593.05 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-28 05:30:00 | 1021.80 | 2023-06-30 05:30:00 | 1053.64 | PARTIAL | 0.50 | 3.12% |
| BUY | retest1 | 2023-06-28 05:30:00 | 1021.80 | 2023-08-23 05:30:00 | 1123.70 | TARGET_HIT | 0.50 | 9.97% |
| BUY | retest1 | 2023-09-06 05:30:00 | 1142.35 | 2023-09-21 05:30:00 | 1146.15 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest1 | 2023-11-02 05:30:00 | 1132.85 | 2023-11-08 05:30:00 | 1179.20 | PARTIAL | 0.50 | 4.09% |
| BUY | retest1 | 2023-11-02 05:30:00 | 1132.85 | 2024-03-15 05:30:00 | 1548.20 | TARGET_HIT | 0.50 | 36.66% |
| BUY | retest1 | 2024-03-22 05:30:00 | 1608.90 | 2024-04-09 05:30:00 | 1602.55 | STOP_HIT | 1.00 | -0.39% |
