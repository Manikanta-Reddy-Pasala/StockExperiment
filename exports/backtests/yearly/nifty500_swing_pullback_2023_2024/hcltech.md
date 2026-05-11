# HCL Technologies Ltd. (HCLTECH)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1198.40
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
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 2.53% / 3.30%
- **Sum % (uncompounded):** 17.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 2 | 3 | 2 | 2.53% | 17.7% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 2 | 3 | 2 | 2.53% | 17.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 2 | 3 | 2 | 2.53% | 17.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 05:30:00 | 1171.45 | 1090.01 | 1139.49 | Stage2 pullback-breakout RSI=60 vol=2.8x ATR=23.90 |
| Stop hit — per-position SL triggered | 2023-08-28 05:30:00 | 1146.45 | 1097.71 | 1158.14 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-11-16 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 05:30:00 | 1311.05 | 1165.03 | 1267.02 | Stage2 pullback-breakout RSI=67 vol=2.2x ATR=21.62 |
| Stop hit — per-position SL triggered | 2023-12-01 05:30:00 | 1336.70 | 1180.33 | 1305.22 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-12-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 05:30:00 | 1364.10 | 1187.90 | 1317.65 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=22.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 05:30:00 | 1409.17 | 1195.58 | 1339.89 | T1 booked 50% @ 1409.17 |
| Target hit | 2024-01-04 05:30:00 | 1419.95 | 1230.52 | 1429.68 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2024-01-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 05:30:00 | 1492.10 | 1239.46 | 1439.91 | Stage2 pullback-breakout RSI=64 vol=1.6x ATR=30.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-12 05:30:00 | 1552.42 | 1244.88 | 1453.39 | T1 booked 50% @ 1552.42 |
| Target hit | 2024-03-04 05:30:00 | 1637.95 | 1358.42 | 1639.90 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-03-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 05:30:00 | 1679.25 | 1377.53 | 1641.53 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=37.09 |
| Stop hit — per-position SL triggered | 2024-03-19 05:30:00 | 1623.62 | 1384.98 | 1638.11 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-11 05:30:00 | 1171.45 | 2023-08-28 05:30:00 | 1146.45 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2023-11-16 05:30:00 | 1311.05 | 2023-12-01 05:30:00 | 1336.70 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest1 | 2023-12-08 05:30:00 | 1364.10 | 2023-12-14 05:30:00 | 1409.17 | PARTIAL | 0.50 | 3.30% |
| BUY | retest1 | 2023-12-08 05:30:00 | 1364.10 | 2024-01-04 05:30:00 | 1419.95 | TARGET_HIT | 0.50 | 4.09% |
| BUY | retest1 | 2024-01-10 05:30:00 | 1492.10 | 2024-01-12 05:30:00 | 1552.42 | PARTIAL | 0.50 | 4.04% |
| BUY | retest1 | 2024-01-10 05:30:00 | 1492.10 | 2024-03-04 05:30:00 | 1637.95 | TARGET_HIT | 0.50 | 9.77% |
| BUY | retest1 | 2024-03-14 05:30:00 | 1679.25 | 2024-03-19 05:30:00 | 1623.62 | STOP_HIT | 1.00 | -3.31% |
