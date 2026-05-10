# Welspun Corp Ltd. (WELCORP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1282.30
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
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 4
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 2.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.08% | -0.7% |
| BUY @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.08% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.46% | 2.8% |
| SELL @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.46% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 5 | 33.3% | 1 | 10 | 4 | 0.14% | 2.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 844.90 | 839.06 | 0.00 | ORB-long ORB[832.40,842.00] vol=2.4x ATR=2.86 |
| Stop hit — per-position SL triggered | 2026-02-10 09:40:00 | 842.04 | 839.45 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:30:00 | 800.00 | 805.21 | 0.00 | ORB-short ORB[805.75,811.05] vol=2.4x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-02-18 10:50:00 | 801.96 | 804.54 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:30:00 | 782.15 | 789.34 | 0.00 | ORB-short ORB[787.45,794.95] vol=3.0x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:35:00 | 778.74 | 787.26 | 0.00 | T1 1.5R @ 778.74 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 782.15 | 784.30 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:40:00 | 778.95 | 782.97 | 0.00 | ORB-short ORB[782.75,787.80] vol=1.6x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 781.53 | 782.79 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 784.50 | 781.82 | 0.00 | ORB-long ORB[774.45,783.10] vol=2.4x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-02-25 10:10:00 | 782.39 | 781.90 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:40:00 | 828.15 | 824.51 | 0.00 | ORB-long ORB[815.55,826.80] vol=1.8x ATR=3.19 |
| Stop hit — per-position SL triggered | 2026-03-06 09:45:00 | 824.96 | 824.61 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 818.70 | 828.39 | 0.00 | ORB-short ORB[831.00,842.40] vol=2.6x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:00:00 | 813.55 | 824.73 | 0.00 | T1 1.5R @ 813.55 |
| Target hit | 2026-03-13 15:20:00 | 800.10 | 810.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:50:00 | 961.90 | 955.54 | 0.00 | ORB-long ORB[948.20,955.00] vol=3.8x ATR=2.82 |
| Stop hit — per-position SL triggered | 2026-04-10 11:25:00 | 959.08 | 956.81 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 1085.00 | 1083.80 | 0.00 | ORB-long ORB[1069.90,1084.00] vol=4.0x ATR=5.38 |
| Stop hit — per-position SL triggered | 2026-04-17 11:10:00 | 1079.62 | 1083.20 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 1089.95 | 1086.48 | 0.00 | ORB-long ORB[1078.20,1087.00] vol=3.2x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:00:00 | 1094.95 | 1089.31 | 0.00 | T1 1.5R @ 1094.95 |
| Stop hit — per-position SL triggered | 2026-04-21 10:20:00 | 1089.95 | 1090.02 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:20:00 | 1307.00 | 1294.61 | 0.00 | ORB-long ORB[1288.00,1299.00] vol=2.1x ATR=5.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:40:00 | 1314.78 | 1298.43 | 0.00 | T1 1.5R @ 1314.78 |
| Stop hit — per-position SL triggered | 2026-05-08 10:55:00 | 1307.00 | 1298.88 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:35:00 | 844.90 | 2026-02-10 09:40:00 | 842.04 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-18 10:30:00 | 800.00 | 2026-02-18 10:50:00 | 801.96 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-23 10:30:00 | 782.15 | 2026-02-23 10:35:00 | 778.74 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-23 10:30:00 | 782.15 | 2026-02-23 11:00:00 | 782.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:40:00 | 778.95 | 2026-02-24 09:45:00 | 781.53 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-25 09:55:00 | 784.50 | 2026-02-25 10:10:00 | 782.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-06 09:40:00 | 828.15 | 2026-03-06 09:45:00 | 824.96 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-13 10:45:00 | 818.70 | 2026-03-13 11:00:00 | 813.55 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2026-03-13 10:45:00 | 818.70 | 2026-03-13 15:20:00 | 800.10 | TARGET_HIT | 0.50 | 2.27% |
| BUY | retest1 | 2026-04-10 10:50:00 | 961.90 | 2026-04-10 11:25:00 | 959.08 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-17 09:50:00 | 1085.00 | 2026-04-17 11:10:00 | 1079.62 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1089.95 | 2026-04-21 10:00:00 | 1094.95 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-21 09:40:00 | 1089.95 | 2026-04-21 10:20:00 | 1089.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 10:20:00 | 1307.00 | 2026-05-08 10:40:00 | 1314.78 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-05-08 10:20:00 | 1307.00 | 2026-05-08 10:55:00 | 1307.00 | STOP_HIT | 0.50 | 0.00% |
