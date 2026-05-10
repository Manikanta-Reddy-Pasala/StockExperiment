# Fortis Healthcare Ltd. (FORTIS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 951.30
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
| ENTRY1 | 16 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 7
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 4.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.27% | 4.9% |
| BUY @ 2nd Alert (retest1) | 18 | 8 | 44.4% | 2 | 10 | 6 | 0.27% | 4.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -0.13% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 23 | 9 | 39.1% | 2 | 14 | 7 | 0.18% | 4.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 869.70 | 865.10 | 0.00 | ORB-long ORB[853.20,865.55] vol=5.0x ATR=4.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:00:00 | 876.22 | 866.97 | 0.00 | T1 1.5R @ 876.22 |
| Target hit | 2026-02-09 15:20:00 | 892.00 | 880.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:00:00 | 935.25 | 928.30 | 0.00 | ORB-long ORB[921.30,929.90] vol=1.8x ATR=3.03 |
| Stop hit — per-position SL triggered | 2026-02-12 10:05:00 | 932.22 | 929.17 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 917.15 | 921.80 | 0.00 | ORB-short ORB[918.50,930.90] vol=1.7x ATR=3.63 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 920.78 | 921.16 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 922.00 | 916.66 | 0.00 | ORB-long ORB[910.00,920.75] vol=1.6x ATR=3.20 |
| Stop hit — per-position SL triggered | 2026-02-16 09:35:00 | 918.80 | 916.99 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 921.85 | 917.59 | 0.00 | ORB-long ORB[911.10,920.70] vol=3.1x ATR=2.27 |
| Stop hit — per-position SL triggered | 2026-02-19 11:45:00 | 919.58 | 918.49 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 930.00 | 925.10 | 0.00 | ORB-long ORB[919.40,926.50] vol=2.3x ATR=3.33 |
| Stop hit — per-position SL triggered | 2026-02-23 09:50:00 | 926.67 | 925.83 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:10:00 | 929.75 | 923.74 | 0.00 | ORB-long ORB[918.00,924.30] vol=2.7x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:25:00 | 932.39 | 925.62 | 0.00 | T1 1.5R @ 932.39 |
| Stop hit — per-position SL triggered | 2026-02-25 13:00:00 | 929.75 | 928.61 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 951.00 | 944.08 | 0.00 | ORB-long ORB[935.15,948.00] vol=1.8x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:45:00 | 955.66 | 947.98 | 0.00 | T1 1.5R @ 955.66 |
| Stop hit — per-position SL triggered | 2026-02-26 11:40:00 | 951.00 | 953.16 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 912.85 | 915.44 | 0.00 | ORB-short ORB[913.10,920.45] vol=2.6x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:45:00 | 909.59 | 914.95 | 0.00 | T1 1.5R @ 909.59 |
| Stop hit — per-position SL triggered | 2026-03-05 12:55:00 | 912.85 | 913.33 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 909.40 | 919.64 | 0.00 | ORB-short ORB[913.95,927.50] vol=2.2x ATR=2.76 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 912.16 | 918.76 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:35:00 | 846.35 | 838.14 | 0.00 | ORB-long ORB[829.20,840.30] vol=1.9x ATR=3.68 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 842.67 | 840.29 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:15:00 | 828.40 | 826.13 | 0.00 | ORB-long ORB[820.05,828.25] vol=2.4x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:20:00 | 831.98 | 826.34 | 0.00 | T1 1.5R @ 831.98 |
| Stop hit — per-position SL triggered | 2026-03-20 12:05:00 | 828.40 | 826.88 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 904.85 | 900.53 | 0.00 | ORB-long ORB[890.95,901.30] vol=2.3x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:55:00 | 908.27 | 902.22 | 0.00 | T1 1.5R @ 908.27 |
| Target hit | 2026-04-21 14:55:00 | 918.40 | 918.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2026-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:45:00 | 921.55 | 925.03 | 0.00 | ORB-short ORB[921.70,934.40] vol=3.0x ATR=2.74 |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 924.29 | 924.45 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:50:00 | 946.35 | 941.17 | 0.00 | ORB-long ORB[935.00,945.35] vol=1.9x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:50:00 | 951.09 | 944.21 | 0.00 | T1 1.5R @ 951.09 |
| Stop hit — per-position SL triggered | 2026-04-27 11:25:00 | 946.35 | 944.83 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:55:00 | 964.00 | 957.95 | 0.00 | ORB-long ORB[947.00,960.50] vol=2.2x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 961.05 | 960.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 869.70 | 2026-02-09 11:00:00 | 876.22 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2026-02-09 10:25:00 | 869.70 | 2026-02-09 15:20:00 | 892.00 | TARGET_HIT | 0.50 | 2.56% |
| BUY | retest1 | 2026-02-12 10:00:00 | 935.25 | 2026-02-12 10:05:00 | 932.22 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-13 09:30:00 | 917.15 | 2026-02-13 09:40:00 | 920.78 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-16 09:30:00 | 922.00 | 2026-02-16 09:35:00 | 918.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-19 10:55:00 | 921.85 | 2026-02-19 11:45:00 | 919.58 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-23 09:40:00 | 930.00 | 2026-02-23 09:50:00 | 926.67 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-25 11:10:00 | 929.75 | 2026-02-25 11:25:00 | 932.39 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-25 11:10:00 | 929.75 | 2026-02-25 13:00:00 | 929.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:30:00 | 951.00 | 2026-02-26 09:45:00 | 955.66 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-26 09:30:00 | 951.00 | 2026-02-26 11:40:00 | 951.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 11:15:00 | 912.85 | 2026-03-05 11:45:00 | 909.59 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-03-05 11:15:00 | 912.85 | 2026-03-05 12:55:00 | 912.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 909.40 | 2026-03-06 11:00:00 | 912.16 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-03-16 09:35:00 | 846.35 | 2026-03-16 09:55:00 | 842.67 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-03-20 11:15:00 | 828.40 | 2026-03-20 11:20:00 | 831.98 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-20 11:15:00 | 828.40 | 2026-03-20 12:05:00 | 828.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:50:00 | 904.85 | 2026-04-21 09:55:00 | 908.27 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-21 09:50:00 | 904.85 | 2026-04-21 14:55:00 | 918.40 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2026-04-24 10:45:00 | 921.55 | 2026-04-24 11:15:00 | 924.29 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-27 09:50:00 | 946.35 | 2026-04-27 10:50:00 | 951.09 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-04-27 09:50:00 | 946.35 | 2026-04-27 11:25:00 | 946.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 09:55:00 | 964.00 | 2026-05-08 10:10:00 | 961.05 | STOP_HIT | 1.00 | -0.31% |
