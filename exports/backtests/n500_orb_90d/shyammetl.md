# Shyam Metalics and Energy Ltd. (SHYAMMETL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 905.65
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 12
- **Target hits / Stop hits / Partials:** 6 / 12 / 10
- **Avg / median % per leg:** 0.30% / 0.35%
- **Sum % (uncompounded):** 8.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.29% | 4.0% |
| BUY @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.29% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 9 | 64.3% | 3 | 5 | 6 | 0.32% | 4.4% |
| SELL @ 2nd Alert (retest1) | 14 | 9 | 64.3% | 3 | 5 | 6 | 0.32% | 4.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 16 | 57.1% | 6 | 12 | 10 | 0.30% | 8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 880.65 | 870.78 | 0.00 | ORB-long ORB[856.20,864.80] vol=5.9x ATR=5.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 12:15:00 | 888.80 | 878.03 | 0.00 | T1 1.5R @ 888.80 |
| Target hit | 2026-02-09 15:20:00 | 891.50 | 885.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 910.20 | 905.96 | 0.00 | ORB-long ORB[897.55,909.20] vol=2.0x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-02-11 10:50:00 | 907.42 | 906.01 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:00:00 | 890.00 | 886.07 | 0.00 | ORB-long ORB[876.80,888.60] vol=2.9x ATR=2.83 |
| Stop hit — per-position SL triggered | 2026-02-17 10:50:00 | 887.17 | 887.89 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 905.40 | 898.43 | 0.00 | ORB-long ORB[889.40,901.00] vol=3.7x ATR=3.29 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 902.11 | 899.31 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 835.15 | 836.30 | 0.00 | ORB-short ORB[836.00,840.90] vol=3.1x ATR=2.31 |
| Stop hit — per-position SL triggered | 2026-02-24 09:55:00 | 837.46 | 835.92 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 839.00 | 835.46 | 0.00 | ORB-long ORB[829.00,837.30] vol=1.8x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 09:55:00 | 842.30 | 837.83 | 0.00 | T1 1.5R @ 842.30 |
| Target hit | 2026-02-25 12:55:00 | 852.00 | 852.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2026-02-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:40:00 | 851.55 | 856.65 | 0.00 | ORB-short ORB[855.30,866.05] vol=1.6x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:55:00 | 848.07 | 854.25 | 0.00 | T1 1.5R @ 848.07 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 851.55 | 851.12 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 844.80 | 850.50 | 0.00 | ORB-short ORB[851.30,856.35] vol=2.1x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:10:00 | 841.85 | 849.18 | 0.00 | T1 1.5R @ 841.85 |
| Stop hit — per-position SL triggered | 2026-02-27 11:30:00 | 844.80 | 848.48 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 790.00 | 793.69 | 0.00 | ORB-short ORB[790.65,801.50] vol=1.8x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:40:00 | 786.62 | 792.32 | 0.00 | T1 1.5R @ 786.62 |
| Target hit | 2026-03-11 12:55:00 | 789.55 | 788.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 773.95 | 776.15 | 0.00 | ORB-short ORB[774.20,779.80] vol=1.7x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 770.49 | 774.15 | 0.00 | T1 1.5R @ 770.49 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 773.95 | 772.06 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:10:00 | 789.05 | 780.89 | 0.00 | ORB-long ORB[770.10,781.50] vol=4.8x ATR=2.27 |
| Stop hit — per-position SL triggered | 2026-03-19 11:20:00 | 786.78 | 781.43 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:05:00 | 794.25 | 791.47 | 0.00 | ORB-long ORB[785.90,794.15] vol=4.5x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:15:00 | 797.52 | 792.85 | 0.00 | T1 1.5R @ 797.52 |
| Stop hit — per-position SL triggered | 2026-03-20 11:55:00 | 794.25 | 793.18 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:30:00 | 855.00 | 862.90 | 0.00 | ORB-short ORB[862.70,869.95] vol=1.8x ATR=2.66 |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 857.66 | 859.63 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 843.05 | 847.67 | 0.00 | ORB-short ORB[846.50,858.90] vol=1.7x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 839.44 | 846.28 | 0.00 | T1 1.5R @ 839.44 |
| Target hit | 2026-04-21 15:20:00 | 824.80 | 834.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 818.00 | 813.77 | 0.00 | ORB-long ORB[804.00,814.50] vol=11.4x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:20:00 | 822.45 | 814.01 | 0.00 | T1 1.5R @ 822.45 |
| Target hit | 2026-04-23 13:35:00 | 826.20 | 828.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 896.80 | 888.49 | 0.00 | ORB-long ORB[879.50,890.55] vol=3.5x ATR=3.92 |
| Stop hit — per-position SL triggered | 2026-05-04 12:20:00 | 892.88 | 895.68 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 887.00 | 892.62 | 0.00 | ORB-short ORB[892.00,897.00] vol=2.5x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 12:10:00 | 882.52 | 886.93 | 0.00 | T1 1.5R @ 882.52 |
| Target hit | 2026-05-07 15:00:00 | 885.00 | 884.89 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — BUY (started 2026-05-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:25:00 | 891.80 | 885.56 | 0.00 | ORB-long ORB[879.90,887.95] vol=6.5x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-05-08 10:30:00 | 889.03 | 887.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 880.65 | 2026-02-09 12:15:00 | 888.80 | PARTIAL | 0.50 | 0.93% |
| BUY | retest1 | 2026-02-09 10:30:00 | 880.65 | 2026-02-09 15:20:00 | 891.50 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2026-02-11 10:40:00 | 910.20 | 2026-02-11 10:50:00 | 907.42 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-17 10:00:00 | 890.00 | 2026-02-17 10:50:00 | 887.17 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-18 09:35:00 | 905.40 | 2026-02-18 09:40:00 | 902.11 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-24 09:30:00 | 835.15 | 2026-02-24 09:55:00 | 837.46 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-25 09:30:00 | 839.00 | 2026-02-25 09:55:00 | 842.30 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-25 09:30:00 | 839.00 | 2026-02-25 12:55:00 | 852.00 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2026-02-26 10:40:00 | 851.55 | 2026-02-26 10:55:00 | 848.07 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-02-26 10:40:00 | 851.55 | 2026-02-26 11:30:00 | 851.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:50:00 | 844.80 | 2026-02-27 11:10:00 | 841.85 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-27 10:50:00 | 844.80 | 2026-02-27 11:30:00 | 844.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:10:00 | 790.00 | 2026-03-11 10:40:00 | 786.62 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-03-11 10:10:00 | 790.00 | 2026-03-11 12:55:00 | 789.55 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2026-03-13 09:50:00 | 773.95 | 2026-03-13 10:10:00 | 770.49 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-13 09:50:00 | 773.95 | 2026-03-13 10:50:00 | 773.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-19 11:10:00 | 789.05 | 2026-03-19 11:20:00 | 786.78 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-20 11:05:00 | 794.25 | 2026-03-20 11:15:00 | 797.52 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-03-20 11:05:00 | 794.25 | 2026-03-20 11:55:00 | 794.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-17 10:30:00 | 855.00 | 2026-04-17 11:15:00 | 857.66 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-21 10:00:00 | 843.05 | 2026-04-21 10:05:00 | 839.44 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-04-21 10:00:00 | 843.05 | 2026-04-21 15:20:00 | 824.80 | TARGET_HIT | 0.50 | 2.16% |
| BUY | retest1 | 2026-04-23 11:15:00 | 818.00 | 2026-04-23 11:20:00 | 822.45 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-23 11:15:00 | 818.00 | 2026-04-23 13:35:00 | 826.20 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2026-05-04 10:30:00 | 896.80 | 2026-05-04 12:20:00 | 892.88 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-05-07 10:45:00 | 887.00 | 2026-05-07 12:10:00 | 882.52 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-05-07 10:45:00 | 887.00 | 2026-05-07 15:00:00 | 885.00 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2026-05-08 10:25:00 | 891.80 | 2026-05-08 10:30:00 | 889.03 | STOP_HIT | 1.00 | -0.31% |
