# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 949.90
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 15
- **Target hits / Stop hits / Partials:** 4 / 15 / 9
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 4.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.22% | 2.9% |
| BUY @ 2nd Alert (retest1) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.22% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.14% | 2.0% |
| SELL @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.14% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 13 | 46.4% | 4 | 15 | 9 | 0.17% | 4.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 913.65 | 917.42 | 0.00 | ORB-short ORB[915.00,922.05] vol=1.9x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:40:00 | 910.16 | 916.08 | 0.00 | T1 1.5R @ 910.16 |
| Stop hit — per-position SL triggered | 2026-02-11 09:45:00 | 913.65 | 915.88 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:10:00 | 915.00 | 908.97 | 0.00 | ORB-long ORB[903.10,914.00] vol=3.3x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 912.05 | 909.46 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 906.00 | 893.68 | 0.00 | ORB-long ORB[882.55,893.80] vol=2.2x ATR=4.89 |
| Stop hit — per-position SL triggered | 2026-02-16 09:55:00 | 901.11 | 899.25 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 910.05 | 914.30 | 0.00 | ORB-short ORB[913.50,919.30] vol=1.9x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:45:00 | 906.15 | 912.04 | 0.00 | T1 1.5R @ 906.15 |
| Target hit | 2026-02-18 15:20:00 | 897.20 | 901.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 888.85 | 881.90 | 0.00 | ORB-long ORB[871.20,883.15] vol=2.2x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:00:00 | 892.84 | 885.73 | 0.00 | T1 1.5R @ 892.84 |
| Target hit | 2026-02-26 11:00:00 | 903.00 | 905.09 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 834.85 | 840.89 | 0.00 | ORB-short ORB[838.80,846.30] vol=1.8x ATR=3.06 |
| Stop hit — per-position SL triggered | 2026-03-12 09:40:00 | 837.91 | 839.61 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:45:00 | 830.85 | 826.32 | 0.00 | ORB-long ORB[817.30,828.95] vol=2.4x ATR=2.91 |
| Stop hit — per-position SL triggered | 2026-03-17 11:10:00 | 827.94 | 826.67 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 841.45 | 836.36 | 0.00 | ORB-long ORB[826.10,838.40] vol=1.5x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 09:45:00 | 847.44 | 839.08 | 0.00 | T1 1.5R @ 847.44 |
| Stop hit — per-position SL triggered | 2026-03-18 10:00:00 | 841.45 | 841.57 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 881.20 | 874.23 | 0.00 | ORB-long ORB[865.85,875.95] vol=4.0x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:45:00 | 886.46 | 881.65 | 0.00 | T1 1.5R @ 886.46 |
| Target hit | 2026-04-10 10:00:00 | 885.40 | 886.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 910.50 | 916.11 | 0.00 | ORB-short ORB[915.65,924.00] vol=1.8x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:00:00 | 906.06 | 914.75 | 0.00 | T1 1.5R @ 906.06 |
| Stop hit — per-position SL triggered | 2026-04-16 10:40:00 | 910.50 | 913.78 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 914.00 | 916.75 | 0.00 | ORB-short ORB[915.15,920.00] vol=1.6x ATR=2.36 |
| Stop hit — per-position SL triggered | 2026-04-17 09:40:00 | 916.36 | 916.45 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 919.85 | 914.16 | 0.00 | ORB-long ORB[904.60,913.40] vol=2.6x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:50:00 | 924.23 | 916.68 | 0.00 | T1 1.5R @ 924.23 |
| Target hit | 2026-04-21 10:45:00 | 921.75 | 922.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2026-04-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:30:00 | 929.00 | 932.88 | 0.00 | ORB-short ORB[929.20,939.90] vol=1.7x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:40:00 | 924.47 | 932.19 | 0.00 | T1 1.5R @ 924.47 |
| Stop hit — per-position SL triggered | 2026-04-23 11:10:00 | 929.00 | 931.65 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 913.45 | 917.87 | 0.00 | ORB-short ORB[915.10,923.00] vol=1.9x ATR=2.60 |
| Stop hit — per-position SL triggered | 2026-04-28 09:45:00 | 916.05 | 917.56 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 905.90 | 909.75 | 0.00 | ORB-short ORB[906.05,916.10] vol=2.2x ATR=2.83 |
| Stop hit — per-position SL triggered | 2026-04-29 10:00:00 | 908.73 | 909.64 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:40:00 | 900.95 | 896.60 | 0.00 | ORB-long ORB[891.00,900.60] vol=1.8x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-04-30 09:50:00 | 897.17 | 896.74 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 09:30:00 | 887.50 | 891.21 | 0.00 | ORB-short ORB[888.50,899.85] vol=1.9x ATR=3.07 |
| Stop hit — per-position SL triggered | 2026-05-04 10:00:00 | 890.57 | 890.34 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 882.50 | 890.15 | 0.00 | ORB-short ORB[885.00,895.55] vol=1.8x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:55:00 | 878.79 | 887.42 | 0.00 | T1 1.5R @ 878.79 |
| Stop hit — per-position SL triggered | 2026-05-05 12:10:00 | 882.50 | 886.84 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 906.60 | 902.29 | 0.00 | ORB-long ORB[893.00,904.80] vol=7.4x ATR=4.00 |
| Stop hit — per-position SL triggered | 2026-05-07 15:20:00 | 906.35 | 904.94 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 09:30:00 | 913.65 | 2026-02-11 09:40:00 | 910.16 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-11 09:30:00 | 913.65 | 2026-02-11 09:45:00 | 913.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 10:10:00 | 915.00 | 2026-02-12 10:15:00 | 912.05 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-16 09:30:00 | 906.00 | 2026-02-16 09:55:00 | 901.11 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-02-18 09:35:00 | 910.05 | 2026-02-18 09:45:00 | 906.15 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-18 09:35:00 | 910.05 | 2026-02-18 15:20:00 | 897.20 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2026-02-26 09:40:00 | 888.85 | 2026-02-26 10:00:00 | 892.84 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-26 09:40:00 | 888.85 | 2026-02-26 11:00:00 | 903.00 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2026-03-12 09:30:00 | 834.85 | 2026-03-12 09:40:00 | 837.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-17 10:45:00 | 830.85 | 2026-03-17 11:10:00 | 827.94 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-18 09:30:00 | 841.45 | 2026-03-18 09:45:00 | 847.44 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-18 09:30:00 | 841.45 | 2026-03-18 10:00:00 | 841.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:40:00 | 881.20 | 2026-04-10 09:45:00 | 886.46 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-10 09:40:00 | 881.20 | 2026-04-10 10:00:00 | 885.40 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-16 09:45:00 | 910.50 | 2026-04-16 10:00:00 | 906.06 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-16 09:45:00 | 910.50 | 2026-04-16 10:40:00 | 910.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-17 09:35:00 | 914.00 | 2026-04-17 09:40:00 | 916.36 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-21 09:45:00 | 919.85 | 2026-04-21 09:50:00 | 924.23 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-21 09:45:00 | 919.85 | 2026-04-21 10:45:00 | 921.75 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-04-23 10:30:00 | 929.00 | 2026-04-23 10:40:00 | 924.47 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-23 10:30:00 | 929.00 | 2026-04-23 11:10:00 | 929.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:40:00 | 913.45 | 2026-04-28 09:45:00 | 916.05 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-04-29 09:55:00 | 905.90 | 2026-04-29 10:00:00 | 908.73 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-30 09:40:00 | 900.95 | 2026-04-30 09:50:00 | 897.17 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-05-04 09:30:00 | 887.50 | 2026-05-04 10:00:00 | 890.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-05-05 11:15:00 | 882.50 | 2026-05-05 11:55:00 | 878.79 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-05 11:15:00 | 882.50 | 2026-05-05 12:10:00 | 882.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-07 09:35:00 | 906.60 | 2026-05-07 15:20:00 | 906.35 | STOP_HIT | 1.00 | -0.03% |
