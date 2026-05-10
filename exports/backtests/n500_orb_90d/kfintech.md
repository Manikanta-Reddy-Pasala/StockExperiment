# Kfin Technologies Ltd. (KFINTECH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 917.00
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 6
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 3.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.27% | 2.2% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.27% | 2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.12% | 1.4% |
| SELL @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.12% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 8 | 42.1% | 2 | 11 | 6 | 0.19% | 3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 1022.60 | 1016.97 | 0.00 | ORB-long ORB[1011.20,1019.90] vol=2.9x ATR=3.87 |
| Stop hit — per-position SL triggered | 2026-02-20 10:00:00 | 1018.73 | 1018.52 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:15:00 | 997.00 | 1001.13 | 0.00 | ORB-short ORB[1000.00,1009.60] vol=2.4x ATR=1.93 |
| Stop hit — per-position SL triggered | 2026-02-25 11:30:00 | 998.93 | 1000.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 958.80 | 967.98 | 0.00 | ORB-short ORB[974.00,980.60] vol=1.6x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 961.55 | 967.25 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 913.45 | 923.16 | 0.00 | ORB-short ORB[919.25,931.10] vol=1.8x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 12:30:00 | 909.39 | 919.55 | 0.00 | T1 1.5R @ 909.39 |
| Stop hit — per-position SL triggered | 2026-03-06 14:40:00 | 913.45 | 914.58 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:15:00 | 941.50 | 945.62 | 0.00 | ORB-short ORB[945.10,954.20] vol=1.7x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 12:05:00 | 937.85 | 944.45 | 0.00 | T1 1.5R @ 937.85 |
| Target hit | 2026-03-11 15:20:00 | 925.30 | 937.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:35:00 | 905.75 | 911.26 | 0.00 | ORB-short ORB[909.00,919.10] vol=1.9x ATR=3.80 |
| Stop hit — per-position SL triggered | 2026-03-12 09:40:00 | 909.55 | 910.96 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 906.90 | 900.54 | 0.00 | ORB-long ORB[892.60,903.30] vol=2.4x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:40:00 | 912.51 | 904.14 | 0.00 | T1 1.5R @ 912.51 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 906.90 | 905.54 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:10:00 | 909.80 | 917.63 | 0.00 | ORB-short ORB[917.10,924.90] vol=2.7x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 12:05:00 | 905.42 | 914.56 | 0.00 | T1 1.5R @ 905.42 |
| Stop hit — per-position SL triggered | 2026-03-20 12:25:00 | 909.80 | 913.06 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:55:00 | 912.10 | 906.23 | 0.00 | ORB-long ORB[900.55,908.60] vol=1.6x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 12:00:00 | 916.82 | 908.20 | 0.00 | T1 1.5R @ 916.82 |
| Stop hit — per-position SL triggered | 2026-04-10 14:15:00 | 912.10 | 911.43 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 910.35 | 917.30 | 0.00 | ORB-short ORB[913.00,924.40] vol=2.6x ATR=3.79 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 914.14 | 916.01 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:20:00 | 990.95 | 984.83 | 0.00 | ORB-long ORB[975.95,983.60] vol=2.3x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 987.69 | 985.66 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 10:50:00 | 883.00 | 869.02 | 0.00 | ORB-long ORB[858.05,871.00] vol=3.1x ATR=3.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:05:00 | 888.45 | 874.26 | 0.00 | T1 1.5R @ 888.45 |
| Target hit | 2026-05-05 15:20:00 | 893.05 | 888.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 913.40 | 916.47 | 0.00 | ORB-short ORB[913.65,926.50] vol=2.0x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-05-08 09:40:00 | 916.54 | 916.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-20 09:30:00 | 1022.60 | 2026-02-20 10:00:00 | 1018.73 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-25 11:15:00 | 997.00 | 2026-02-25 11:30:00 | 998.93 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-27 10:55:00 | 958.80 | 2026-02-27 11:00:00 | 961.55 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 10:45:00 | 913.45 | 2026-03-06 12:30:00 | 909.39 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-06 10:45:00 | 913.45 | 2026-03-06 14:40:00 | 913.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 11:15:00 | 941.50 | 2026-03-11 12:05:00 | 937.85 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-11 11:15:00 | 941.50 | 2026-03-11 15:20:00 | 925.30 | TARGET_HIT | 0.50 | 1.72% |
| SELL | retest1 | 2026-03-12 09:35:00 | 905.75 | 2026-03-12 09:40:00 | 909.55 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-16 09:30:00 | 906.90 | 2026-03-16 09:40:00 | 912.51 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-03-16 09:30:00 | 906.90 | 2026-03-16 09:55:00 | 906.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 11:10:00 | 909.80 | 2026-03-20 12:05:00 | 905.42 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-20 11:10:00 | 909.80 | 2026-03-20 12:25:00 | 909.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 10:55:00 | 912.10 | 2026-04-10 12:00:00 | 916.82 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-04-10 10:55:00 | 912.10 | 2026-04-10 14:15:00 | 912.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-15 09:30:00 | 910.35 | 2026-04-15 09:40:00 | 914.14 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-21 10:20:00 | 990.95 | 2026-04-21 10:40:00 | 987.69 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-05-05 10:50:00 | 883.00 | 2026-05-05 11:05:00 | 888.45 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-05-05 10:50:00 | 883.00 | 2026-05-05 15:20:00 | 893.05 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2026-05-08 09:30:00 | 913.40 | 2026-05-08 09:40:00 | 916.54 | STOP_HIT | 1.00 | -0.34% |
