# Lodha Developers Ltd. (LODHA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 960.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 12
- **Target hits / Stop hits / Partials:** 2 / 12 / 7
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 3.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.15% | 1.2% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 0.15% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.16% | 2.1% |
| SELL @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.16% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 9 | 42.9% | 2 | 12 | 7 | 0.15% | 3.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1093.15 | 1095.78 | 0.00 | ORB-short ORB[1094.00,1101.15] vol=2.5x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:10:00 | 1088.86 | 1093.58 | 0.00 | T1 1.5R @ 1088.86 |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 1093.15 | 1093.52 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 1087.35 | 1095.13 | 0.00 | ORB-short ORB[1099.00,1108.15] vol=1.9x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:20:00 | 1083.55 | 1094.36 | 0.00 | T1 1.5R @ 1083.55 |
| Target hit | 2026-02-19 15:20:00 | 1064.00 | 1073.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:15:00 | 1059.55 | 1064.16 | 0.00 | ORB-short ORB[1065.00,1078.45] vol=1.8x ATR=3.62 |
| Stop hit — per-position SL triggered | 2026-02-24 10:40:00 | 1063.17 | 1063.66 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 1043.45 | 1048.59 | 0.00 | ORB-short ORB[1044.45,1059.15] vol=2.0x ATR=4.97 |
| Stop hit — per-position SL triggered | 2026-02-25 09:55:00 | 1048.42 | 1048.38 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 1009.85 | 1020.34 | 0.00 | ORB-short ORB[1017.05,1030.90] vol=1.6x ATR=3.92 |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 1013.77 | 1017.66 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 995.15 | 1002.02 | 0.00 | ORB-short ORB[1002.30,1013.00] vol=1.9x ATR=3.44 |
| Stop hit — per-position SL triggered | 2026-02-27 11:05:00 | 998.59 | 1001.64 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 891.00 | 908.70 | 0.00 | ORB-short ORB[903.90,913.40] vol=2.2x ATR=4.10 |
| Stop hit — per-position SL triggered | 2026-03-11 10:45:00 | 895.10 | 907.42 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 832.30 | 830.47 | 0.00 | ORB-long ORB[822.40,832.00] vol=3.4x ATR=3.51 |
| Stop hit — per-position SL triggered | 2026-03-20 09:55:00 | 828.79 | 830.50 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 865.00 | 870.06 | 0.00 | ORB-short ORB[868.55,878.70] vol=2.4x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:15:00 | 859.74 | 868.73 | 0.00 | T1 1.5R @ 859.74 |
| Stop hit — per-position SL triggered | 2026-04-17 13:50:00 | 865.00 | 866.95 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:20:00 | 886.95 | 880.29 | 0.00 | ORB-long ORB[872.55,881.25] vol=2.4x ATR=2.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:55:00 | 891.32 | 883.64 | 0.00 | T1 1.5R @ 891.32 |
| Stop hit — per-position SL triggered | 2026-04-22 12:55:00 | 886.95 | 885.82 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:55:00 | 845.95 | 856.56 | 0.00 | ORB-short ORB[856.50,868.20] vol=1.6x ATR=3.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 12:00:00 | 841.16 | 853.09 | 0.00 | T1 1.5R @ 841.16 |
| Stop hit — per-position SL triggered | 2026-04-24 14:35:00 | 845.95 | 848.86 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 885.00 | 879.45 | 0.00 | ORB-long ORB[872.05,882.60] vol=3.8x ATR=3.38 |
| Stop hit — per-position SL triggered | 2026-04-28 11:25:00 | 881.62 | 881.18 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:25:00 | 900.65 | 891.49 | 0.00 | ORB-long ORB[882.00,894.55] vol=1.8x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:35:00 | 906.29 | 893.03 | 0.00 | T1 1.5R @ 906.29 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 900.65 | 895.15 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:35:00 | 918.65 | 914.52 | 0.00 | ORB-long ORB[909.25,918.45] vol=1.5x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:40:00 | 923.70 | 917.51 | 0.00 | T1 1.5R @ 923.70 |
| Target hit | 2026-05-06 12:35:00 | 921.50 | 922.13 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-18 09:45:00 | 1093.15 | 2026-02-18 10:10:00 | 1088.86 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-18 09:45:00 | 1093.15 | 2026-02-18 10:15:00 | 1093.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:05:00 | 1087.35 | 2026-02-19 11:20:00 | 1083.55 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-19 11:05:00 | 1087.35 | 2026-02-19 15:20:00 | 1064.00 | TARGET_HIT | 0.50 | 2.15% |
| SELL | retest1 | 2026-02-24 10:15:00 | 1059.55 | 2026-02-24 10:40:00 | 1063.17 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-25 09:45:00 | 1043.45 | 2026-02-25 09:55:00 | 1048.42 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-02-26 10:45:00 | 1009.85 | 2026-02-26 11:15:00 | 1013.77 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-02-27 10:50:00 | 995.15 | 2026-02-27 11:05:00 | 998.59 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-11 10:40:00 | 891.00 | 2026-03-11 10:45:00 | 895.10 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2026-03-20 09:30:00 | 832.30 | 2026-03-20 09:55:00 | 828.79 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-17 11:05:00 | 865.00 | 2026-04-17 12:15:00 | 859.74 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-04-17 11:05:00 | 865.00 | 2026-04-17 13:50:00 | 865.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:20:00 | 886.95 | 2026-04-22 11:55:00 | 891.32 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-22 10:20:00 | 886.95 | 2026-04-22 12:55:00 | 886.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 10:55:00 | 845.95 | 2026-04-24 12:00:00 | 841.16 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-24 10:55:00 | 845.95 | 2026-04-24 14:35:00 | 845.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 11:05:00 | 885.00 | 2026-04-28 11:25:00 | 881.62 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-29 10:25:00 | 900.65 | 2026-04-29 10:35:00 | 906.29 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-29 10:25:00 | 900.65 | 2026-04-29 10:50:00 | 900.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:35:00 | 918.65 | 2026-05-06 10:40:00 | 923.70 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-05-06 09:35:00 | 918.65 | 2026-05-06 12:35:00 | 921.50 | TARGET_HIT | 0.50 | 0.31% |
