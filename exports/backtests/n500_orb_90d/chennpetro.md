# Chennai Petroleum Corporation Ltd. (CHENNPETRO)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1079.90
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 1.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.13% | -0.9% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.13% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.28% | 1.9% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 3 | 3 | 0.28% | 1.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 14 | 6 | 42.9% | 2 | 8 | 4 | 0.07% | 1.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:45:00 | 931.90 | 925.23 | 0.00 | ORB-long ORB[916.00,926.75] vol=5.9x ATR=3.45 |
| Stop hit — per-position SL triggered | 2026-02-11 09:55:00 | 928.45 | 927.95 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 915.50 | 921.62 | 0.00 | ORB-short ORB[918.40,927.00] vol=2.3x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:20:00 | 912.36 | 920.67 | 0.00 | T1 1.5R @ 912.36 |
| Target hit | 2026-02-12 12:30:00 | 908.50 | 908.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:30:00 | 875.50 | 883.17 | 0.00 | ORB-short ORB[882.15,892.20] vol=1.6x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:55:00 | 871.41 | 881.45 | 0.00 | T1 1.5R @ 871.41 |
| Stop hit — per-position SL triggered | 2026-02-17 11:20:00 | 875.50 | 879.75 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 902.95 | 898.06 | 0.00 | ORB-long ORB[892.00,902.85] vol=2.0x ATR=3.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:40:00 | 908.50 | 906.02 | 0.00 | T1 1.5R @ 908.50 |
| Target hit | 2026-02-18 11:50:00 | 907.05 | 910.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 920.45 | 914.01 | 0.00 | ORB-long ORB[908.15,915.50] vol=2.4x ATR=3.22 |
| Stop hit — per-position SL triggered | 2026-02-19 09:50:00 | 917.23 | 917.90 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:35:00 | 903.55 | 897.93 | 0.00 | ORB-long ORB[889.65,900.85] vol=1.6x ATR=4.05 |
| Stop hit — per-position SL triggered | 2026-02-20 09:55:00 | 899.50 | 899.09 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 899.55 | 887.37 | 0.00 | ORB-long ORB[882.40,891.05] vol=4.7x ATR=3.56 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 895.99 | 888.78 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 950.20 | 955.29 | 0.00 | ORB-short ORB[951.25,964.00] vol=1.6x ATR=3.48 |
| Stop hit — per-position SL triggered | 2026-04-15 09:35:00 | 953.68 | 955.13 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:50:00 | 983.35 | 975.27 | 0.00 | ORB-long ORB[970.85,978.30] vol=4.0x ATR=3.86 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 979.49 | 975.64 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 1078.60 | 1088.56 | 0.00 | ORB-short ORB[1082.60,1097.70] vol=2.1x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:50:00 | 1070.76 | 1085.46 | 0.00 | T1 1.5R @ 1070.76 |
| Stop hit — per-position SL triggered | 2026-05-06 10:55:00 | 1078.60 | 1082.01 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-11 09:45:00 | 931.90 | 2026-02-11 09:55:00 | 928.45 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-12 11:15:00 | 915.50 | 2026-02-12 11:20:00 | 912.36 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-12 11:15:00 | 915.50 | 2026-02-12 12:30:00 | 908.50 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2026-02-17 10:30:00 | 875.50 | 2026-02-17 10:55:00 | 871.41 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-17 10:30:00 | 875.50 | 2026-02-17 11:20:00 | 875.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:30:00 | 902.95 | 2026-02-18 09:40:00 | 908.50 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-02-18 09:30:00 | 902.95 | 2026-02-18 11:50:00 | 907.05 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-19 09:35:00 | 920.45 | 2026-02-19 09:50:00 | 917.23 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-20 09:35:00 | 903.55 | 2026-02-20 09:55:00 | 899.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-02-23 10:55:00 | 899.55 | 2026-02-23 11:05:00 | 895.99 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-15 09:30:00 | 950.20 | 2026-04-15 09:35:00 | 953.68 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-16 09:50:00 | 983.35 | 2026-04-16 09:55:00 | 979.49 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-05-06 09:40:00 | 1078.60 | 2026-05-06 09:50:00 | 1070.76 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2026-05-06 09:40:00 | 1078.60 | 2026-05-06 10:55:00 | 1078.60 | STOP_HIT | 0.50 | 0.00% |
