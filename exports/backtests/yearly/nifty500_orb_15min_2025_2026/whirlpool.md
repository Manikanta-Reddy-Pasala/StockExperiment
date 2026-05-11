# Whirlpool of India Ltd. (WHIRLPOOL)

## Backtest Summary

- **Window:** 2026-01-06 09:15:00 → 2026-05-08 15:25:00 (6150 bars)
- **Last close:** 954.90
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 18
- **Target hits / Stop hits / Partials:** 4 / 18 / 8
- **Avg / median % per leg:** 0.02% / 0.00%
- **Sum % (uncompounded):** 0.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.09% | 1.3% |
| BUY @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 1 | 9 | 5 | 0.09% | 1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 6 | 40.0% | 3 | 9 | 3 | -0.04% | -0.6% |
| SELL @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 3 | 9 | 3 | -0.04% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 30 | 12 | 40.0% | 4 | 18 | 8 | 0.02% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 870.15 | 875.56 | 0.00 | ORB-short ORB[873.05,881.00] vol=1.8x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 872.41 | 873.14 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-01-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:30:00 | 825.10 | 828.29 | 0.00 | ORB-short ORB[825.55,833.50] vol=1.6x ATR=2.73 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 827.83 | 827.44 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-01-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:45:00 | 789.80 | 797.11 | 0.00 | ORB-short ORB[790.00,799.45] vol=1.6x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-01-22 14:30:00 | 793.04 | 793.09 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-01-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 11:05:00 | 777.70 | 778.35 | 0.00 | ORB-short ORB[778.00,785.60] vol=2.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-01-28 12:00:00 | 779.66 | 778.61 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-01-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:55:00 | 777.25 | 778.43 | 0.00 | ORB-short ORB[777.60,783.40] vol=1.9x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:40:00 | 773.87 | 777.01 | 0.00 | T1 1.5R @ 773.87 |
| Target hit | 2026-01-29 15:05:00 | 776.45 | 775.83 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 773.70 | 766.41 | 0.00 | ORB-long ORB[761.60,771.05] vol=2.0x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-01-30 09:35:00 | 770.86 | 766.99 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 09:50:00 | 799.30 | 793.60 | 0.00 | ORB-long ORB[784.30,792.95] vol=1.8x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 09:55:00 | 803.33 | 796.70 | 0.00 | T1 1.5R @ 803.33 |
| Stop hit — per-position SL triggered | 2026-02-04 10:00:00 | 799.30 | 797.44 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:45:00 | 901.85 | 905.69 | 0.00 | ORB-short ORB[904.95,915.90] vol=1.6x ATR=3.44 |
| Stop hit — per-position SL triggered | 2026-02-12 10:00:00 | 905.29 | 905.12 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:05:00 | 930.00 | 922.53 | 0.00 | ORB-long ORB[909.55,922.00] vol=1.5x ATR=3.25 |
| Target hit | 2026-02-20 15:20:00 | 930.05 | 928.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 920.70 | 916.85 | 0.00 | ORB-long ORB[907.00,919.75] vol=1.7x ATR=3.85 |
| Stop hit — per-position SL triggered | 2026-02-25 12:45:00 | 916.85 | 919.55 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:10:00 | 914.30 | 908.21 | 0.00 | ORB-long ORB[901.75,910.05] vol=2.0x ATR=3.03 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 911.27 | 908.48 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:40:00 | 916.50 | 910.19 | 0.00 | ORB-long ORB[907.20,913.95] vol=1.6x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:45:00 | 919.52 | 912.20 | 0.00 | T1 1.5R @ 919.52 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 916.50 | 912.18 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:10:00 | 872.10 | 873.77 | 0.00 | ORB-short ORB[873.00,884.80] vol=2.5x ATR=2.23 |
| Stop hit — per-position SL triggered | 2026-03-10 11:15:00 | 874.33 | 873.80 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:40:00 | 817.00 | 821.42 | 0.00 | ORB-short ORB[819.55,824.75] vol=1.6x ATR=2.52 |
| Stop hit — per-position SL triggered | 2026-03-17 10:45:00 | 819.52 | 821.38 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 827.00 | 820.93 | 0.00 | ORB-long ORB[813.05,823.45] vol=1.9x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:20:00 | 832.91 | 825.60 | 0.00 | T1 1.5R @ 832.91 |
| Stop hit — per-position SL triggered | 2026-03-20 10:30:00 | 827.00 | 826.44 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-03-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:10:00 | 779.75 | 786.76 | 0.00 | ORB-short ORB[782.80,794.45] vol=1.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2026-03-24 11:20:00 | 782.79 | 786.67 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:55:00 | 852.40 | 843.27 | 0.00 | ORB-long ORB[835.00,845.80] vol=2.0x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:00:00 | 857.89 | 845.10 | 0.00 | T1 1.5R @ 857.89 |
| Stop hit — per-position SL triggered | 2026-04-15 10:05:00 | 852.40 | 845.32 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 849.00 | 853.14 | 0.00 | ORB-short ORB[850.55,858.35] vol=1.7x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 10:00:00 | 844.75 | 851.80 | 0.00 | T1 1.5R @ 844.75 |
| Target hit | 2026-04-16 14:40:00 | 846.00 | 845.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 922.85 | 916.59 | 0.00 | ORB-long ORB[910.05,920.55] vol=1.7x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:50:00 | 927.39 | 920.76 | 0.00 | T1 1.5R @ 927.39 |
| Stop hit — per-position SL triggered | 2026-04-22 10:00:00 | 922.85 | 921.35 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 984.60 | 992.62 | 0.00 | ORB-short ORB[993.45,1005.60] vol=2.4x ATR=4.44 |
| Stop hit — per-position SL triggered | 2026-04-29 09:40:00 | 989.04 | 992.12 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:15:00 | 980.00 | 975.70 | 0.00 | ORB-long ORB[965.15,974.85] vol=1.8x ATR=2.89 |
| Stop hit — per-position SL triggered | 2026-05-07 11:40:00 | 977.11 | 976.33 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:15:00 | 959.35 | 965.61 | 0.00 | ORB-short ORB[966.00,978.65] vol=1.8x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 12:45:00 | 954.26 | 961.66 | 0.00 | T1 1.5R @ 954.26 |
| Target hit | 2026-05-08 15:20:00 | 954.10 | 959.70 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-01-08 11:00:00 | 870.15 | 2026-01-08 11:35:00 | 872.41 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-13 09:30:00 | 825.10 | 2026-01-13 09:45:00 | 827.83 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-22 10:45:00 | 789.80 | 2026-01-22 14:30:00 | 793.04 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-01-28 11:05:00 | 777.70 | 2026-01-28 12:00:00 | 779.66 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-29 09:55:00 | 777.25 | 2026-01-29 10:40:00 | 773.87 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-29 09:55:00 | 777.25 | 2026-01-29 15:05:00 | 776.45 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-01-30 09:30:00 | 773.70 | 2026-01-30 09:35:00 | 770.86 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-04 09:50:00 | 799.30 | 2026-02-04 09:55:00 | 803.33 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-02-04 09:50:00 | 799.30 | 2026-02-04 10:00:00 | 799.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 09:45:00 | 901.85 | 2026-02-12 10:00:00 | 905.29 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-02-20 11:05:00 | 930.00 | 2026-02-20 15:20:00 | 930.05 | TARGET_HIT | 1.00 | 0.01% |
| BUY | retest1 | 2026-02-25 09:40:00 | 920.70 | 2026-02-25 12:45:00 | 916.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-27 10:10:00 | 914.30 | 2026-02-27 10:20:00 | 911.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-06 10:40:00 | 916.50 | 2026-03-06 10:45:00 | 919.52 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-03-06 10:40:00 | 916.50 | 2026-03-06 10:50:00 | 916.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 11:10:00 | 872.10 | 2026-03-10 11:15:00 | 874.33 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-17 10:40:00 | 817.00 | 2026-03-17 10:45:00 | 819.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-03-20 09:35:00 | 827.00 | 2026-03-20 10:20:00 | 832.91 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-20 09:35:00 | 827.00 | 2026-03-20 10:30:00 | 827.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 11:10:00 | 779.75 | 2026-03-24 11:20:00 | 782.79 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-15 09:55:00 | 852.40 | 2026-04-15 10:00:00 | 857.89 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-04-15 09:55:00 | 852.40 | 2026-04-15 10:05:00 | 852.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 09:40:00 | 849.00 | 2026-04-16 10:00:00 | 844.75 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-16 09:40:00 | 849.00 | 2026-04-16 14:40:00 | 846.00 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-22 09:35:00 | 922.85 | 2026-04-22 09:50:00 | 927.39 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-22 09:35:00 | 922.85 | 2026-04-22 10:00:00 | 922.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-29 09:35:00 | 984.60 | 2026-04-29 09:40:00 | 989.04 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-07 11:15:00 | 980.00 | 2026-05-07 11:40:00 | 977.11 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-08 10:15:00 | 959.35 | 2026-05-08 12:45:00 | 954.26 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-08 10:15:00 | 959.35 | 2026-05-08 15:20:00 | 954.10 | TARGET_HIT | 0.50 | 0.55% |
