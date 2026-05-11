# State Bank of India (SBIN)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1018.50
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
| ENTRY1 | 101 |
| ENTRY2 | 0 |
| PARTIAL | 42 |
| TARGET_HIT | 18 |
| STOP_HIT | 83 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 143 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 60 / 83
- **Target hits / Stop hits / Partials:** 18 / 83 / 42
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 12.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 27 | 36.5% | 7 | 47 | 20 | 0.04% | 2.8% |
| BUY @ 2nd Alert (retest1) | 74 | 27 | 36.5% | 7 | 47 | 20 | 0.04% | 2.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 69 | 33 | 47.8% | 11 | 36 | 22 | 0.14% | 9.8% |
| SELL @ 2nd Alert (retest1) | 69 | 33 | 47.8% | 11 | 36 | 22 | 0.14% | 9.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 143 | 60 | 42.0% | 18 | 83 | 42 | 0.09% | 12.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 11:15:00 | 802.90 | 798.31 | 0.00 | ORB-long ORB[796.55,802.25] vol=3.0x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-05-12 11:35:00 | 800.25 | 798.73 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:35:00 | 807.35 | 801.79 | 0.00 | ORB-long ORB[798.00,804.25] vol=2.7x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-05-13 09:40:00 | 805.36 | 802.39 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:35:00 | 796.60 | 800.14 | 0.00 | ORB-short ORB[799.00,805.95] vol=1.5x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-05-15 10:30:00 | 798.28 | 798.33 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:20:00 | 799.55 | 796.17 | 0.00 | ORB-long ORB[792.10,797.60] vol=2.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-05-19 10:25:00 | 798.02 | 796.50 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:15:00 | 797.60 | 793.76 | 0.00 | ORB-long ORB[791.10,795.95] vol=4.1x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 11:25:00 | 799.65 | 794.96 | 0.00 | T1 1.5R @ 799.65 |
| Stop hit — per-position SL triggered | 2025-05-27 11:50:00 | 797.60 | 795.98 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 10:50:00 | 809.60 | 810.20 | 0.00 | ORB-short ORB[809.70,817.00] vol=2.3x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:15:00 | 807.62 | 810.11 | 0.00 | T1 1.5R @ 807.62 |
| Target hit | 2025-06-04 15:20:00 | 806.05 | 808.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 810.20 | 807.85 | 0.00 | ORB-long ORB[803.10,808.65] vol=4.8x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 808.31 | 807.73 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:30:00 | 821.90 | 820.17 | 0.00 | ORB-long ORB[815.70,821.50] vol=1.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-06-09 10:35:00 | 820.51 | 820.19 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 11:00:00 | 816.00 | 819.42 | 0.00 | ORB-short ORB[818.65,822.05] vol=1.7x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-06-10 11:10:00 | 817.21 | 819.27 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:10:00 | 819.00 | 816.56 | 0.00 | ORB-long ORB[814.05,817.75] vol=1.8x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-06-11 13:20:00 | 817.80 | 818.60 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:40:00 | 811.50 | 812.99 | 0.00 | ORB-short ORB[812.00,816.00] vol=2.5x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-06-12 10:50:00 | 812.74 | 812.92 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:35:00 | 798.55 | 795.37 | 0.00 | ORB-long ORB[792.50,796.45] vol=1.9x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 797.03 | 795.58 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-06-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:55:00 | 790.90 | 793.14 | 0.00 | ORB-short ORB[791.00,794.65] vol=2.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-06-18 12:05:00 | 792.08 | 792.57 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:00:00 | 785.50 | 790.10 | 0.00 | ORB-short ORB[788.30,793.25] vol=1.6x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:05:00 | 783.34 | 789.16 | 0.00 | T1 1.5R @ 783.34 |
| Stop hit — per-position SL triggered | 2025-06-19 11:20:00 | 785.50 | 787.04 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:40:00 | 794.25 | 790.13 | 0.00 | ORB-long ORB[786.10,791.35] vol=1.6x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:20:00 | 796.76 | 791.30 | 0.00 | T1 1.5R @ 796.76 |
| Stop hit — per-position SL triggered | 2025-06-20 12:25:00 | 794.25 | 792.88 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-06-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:45:00 | 793.95 | 794.76 | 0.00 | ORB-short ORB[798.00,803.35] vol=14.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:55:00 | 792.21 | 794.72 | 0.00 | T1 1.5R @ 792.21 |
| Stop hit — per-position SL triggered | 2025-06-26 12:00:00 | 793.95 | 794.45 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-06-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:00:00 | 815.40 | 812.70 | 0.00 | ORB-long ORB[807.05,813.50] vol=2.2x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 10:10:00 | 817.59 | 813.29 | 0.00 | T1 1.5R @ 817.59 |
| Target hit | 2025-06-30 15:20:00 | 821.00 | 816.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:15:00 | 816.45 | 818.83 | 0.00 | ORB-short ORB[819.60,824.05] vol=2.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-07-02 11:25:00 | 817.59 | 818.75 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 10:40:00 | 814.35 | 812.27 | 0.00 | ORB-long ORB[809.20,813.60] vol=2.0x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-07-10 10:45:00 | 813.23 | 812.69 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:35:00 | 812.15 | 810.12 | 0.00 | ORB-long ORB[808.00,811.50] vol=1.7x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:40:00 | 814.05 | 811.07 | 0.00 | T1 1.5R @ 814.05 |
| Stop hit — per-position SL triggered | 2025-07-14 10:30:00 | 812.15 | 812.09 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:30:00 | 820.15 | 818.26 | 0.00 | ORB-long ORB[815.30,819.80] vol=1.8x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 09:35:00 | 822.08 | 819.79 | 0.00 | T1 1.5R @ 822.08 |
| Stop hit — per-position SL triggered | 2025-07-16 09:50:00 | 820.15 | 820.58 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:00:00 | 841.60 | 835.49 | 0.00 | ORB-long ORB[829.90,840.50] vol=2.4x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-07-17 10:05:00 | 839.30 | 835.83 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:00:00 | 820.85 | 823.30 | 0.00 | ORB-short ORB[823.30,828.05] vol=2.5x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 11:15:00 | 818.92 | 822.80 | 0.00 | T1 1.5R @ 818.92 |
| Target hit | 2025-07-22 15:20:00 | 815.10 | 819.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 11:15:00 | 811.95 | 814.73 | 0.00 | ORB-short ORB[813.70,818.90] vol=2.4x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:45:00 | 810.30 | 814.14 | 0.00 | T1 1.5R @ 810.30 |
| Target hit | 2025-07-25 15:20:00 | 806.25 | 810.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2025-07-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 10:45:00 | 803.40 | 806.50 | 0.00 | ORB-short ORB[804.20,807.90] vol=1.5x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-07-28 11:00:00 | 804.64 | 806.29 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-07-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 09:50:00 | 800.15 | 798.67 | 0.00 | ORB-long ORB[796.20,798.95] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-07-30 10:25:00 | 798.97 | 799.44 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 798.60 | 795.32 | 0.00 | ORB-long ORB[793.00,797.65] vol=2.0x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:30:00 | 800.44 | 796.07 | 0.00 | T1 1.5R @ 800.44 |
| Stop hit — per-position SL triggered | 2025-07-31 14:25:00 | 798.60 | 798.75 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 11:15:00 | 798.15 | 796.90 | 0.00 | ORB-long ORB[793.10,796.55] vol=1.5x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:25:00 | 800.08 | 797.08 | 0.00 | T1 1.5R @ 800.08 |
| Stop hit — per-position SL triggered | 2025-08-01 12:00:00 | 798.15 | 797.68 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 09:45:00 | 802.90 | 805.04 | 0.00 | ORB-short ORB[804.65,807.80] vol=1.8x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 10:50:00 | 800.62 | 803.41 | 0.00 | T1 1.5R @ 800.62 |
| Stop hit — per-position SL triggered | 2025-08-08 13:15:00 | 802.90 | 800.89 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 09:40:00 | 822.40 | 817.28 | 0.00 | ORB-long ORB[808.00,819.30] vol=2.2x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-08-11 09:45:00 | 819.54 | 817.55 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:15:00 | 827.25 | 823.83 | 0.00 | ORB-long ORB[819.10,822.90] vol=3.0x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 826.02 | 825.26 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:15:00 | 829.55 | 828.11 | 0.00 | ORB-long ORB[825.80,829.10] vol=1.5x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-08-19 10:20:00 | 828.41 | 828.14 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:15:00 | 820.60 | 822.38 | 0.00 | ORB-short ORB[823.30,826.00] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-08-22 11:40:00 | 821.50 | 822.22 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-08-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 11:05:00 | 816.25 | 816.83 | 0.00 | ORB-short ORB[817.00,820.05] vol=1.5x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 817.27 | 816.81 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 11:10:00 | 808.50 | 808.08 | 0.00 | ORB-long ORB[804.60,807.60] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-09-02 11:50:00 | 807.59 | 808.04 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:45:00 | 807.85 | 808.97 | 0.00 | ORB-short ORB[809.75,812.50] vol=1.8x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-09-05 11:10:00 | 809.09 | 808.93 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:45:00 | 806.50 | 807.59 | 0.00 | ORB-short ORB[806.80,812.35] vol=2.0x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-09-09 10:50:00 | 807.43 | 807.52 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:15:00 | 817.90 | 814.17 | 0.00 | ORB-long ORB[810.40,814.50] vol=1.6x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 10:20:00 | 819.97 | 814.80 | 0.00 | T1 1.5R @ 819.97 |
| Stop hit — per-position SL triggered | 2025-09-10 14:30:00 | 817.90 | 819.14 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:25:00 | 840.00 | 834.75 | 0.00 | ORB-long ORB[831.10,834.55] vol=2.9x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:30:00 | 842.32 | 835.87 | 0.00 | T1 1.5R @ 842.32 |
| Stop hit — per-position SL triggered | 2025-09-17 11:25:00 | 840.00 | 839.58 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:20:00 | 860.30 | 855.08 | 0.00 | ORB-long ORB[849.30,855.60] vol=1.7x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:25:00 | 862.87 | 855.95 | 0.00 | T1 1.5R @ 862.87 |
| Stop hit — per-position SL triggered | 2025-09-19 10:35:00 | 860.30 | 856.72 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 10:10:00 | 859.45 | 855.01 | 0.00 | ORB-long ORB[851.30,855.85] vol=1.8x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-09-23 10:20:00 | 857.59 | 855.79 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 11:10:00 | 870.00 | 866.27 | 0.00 | ORB-long ORB[862.95,867.85] vol=2.2x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-09-25 11:15:00 | 868.45 | 866.31 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-09-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:00:00 | 856.30 | 859.68 | 0.00 | ORB-short ORB[857.90,864.35] vol=2.6x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-09-26 11:05:00 | 858.12 | 859.62 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:35:00 | 868.25 | 870.51 | 0.00 | ORB-short ORB[870.30,876.45] vol=1.6x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 11:20:00 | 864.93 | 869.38 | 0.00 | T1 1.5R @ 864.93 |
| Stop hit — per-position SL triggered | 2025-10-01 13:45:00 | 868.25 | 867.72 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:25:00 | 868.05 | 870.59 | 0.00 | ORB-short ORB[870.00,877.00] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-10-07 10:30:00 | 869.68 | 870.49 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 861.00 | 864.30 | 0.00 | ORB-short ORB[861.10,867.60] vol=1.6x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:15:00 | 858.47 | 863.36 | 0.00 | T1 1.5R @ 858.47 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 861.00 | 863.04 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:00:00 | 874.80 | 870.41 | 0.00 | ORB-long ORB[861.30,870.85] vol=1.6x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:05:00 | 877.71 | 871.84 | 0.00 | T1 1.5R @ 877.71 |
| Target hit | 2025-10-10 15:20:00 | 880.90 | 879.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:45:00 | 883.00 | 879.78 | 0.00 | ORB-long ORB[876.80,880.60] vol=1.6x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 09:55:00 | 885.50 | 881.00 | 0.00 | T1 1.5R @ 885.50 |
| Target hit | 2025-10-13 13:00:00 | 884.30 | 884.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — SELL (started 2025-10-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:40:00 | 878.95 | 879.78 | 0.00 | ORB-short ORB[879.50,884.15] vol=1.7x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:05:00 | 876.86 | 879.60 | 0.00 | T1 1.5R @ 876.86 |
| Target hit | 2025-10-14 14:40:00 | 877.30 | 876.96 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — BUY (started 2025-10-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:00:00 | 883.65 | 881.15 | 0.00 | ORB-long ORB[877.95,881.25] vol=2.2x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-10-15 11:45:00 | 882.33 | 881.81 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:10:00 | 893.95 | 889.13 | 0.00 | ORB-long ORB[885.10,890.35] vol=4.5x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-10-17 11:20:00 | 892.43 | 889.49 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:45:00 | 898.65 | 896.14 | 0.00 | ORB-long ORB[891.05,897.95] vol=2.6x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-10-20 10:00:00 | 896.34 | 896.21 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:55:00 | 913.50 | 909.20 | 0.00 | ORB-long ORB[904.80,912.10] vol=1.5x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 11:15:00 | 916.14 | 912.23 | 0.00 | T1 1.5R @ 916.14 |
| Stop hit — per-position SL triggered | 2025-10-23 12:45:00 | 913.50 | 913.59 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-10-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:05:00 | 905.00 | 908.25 | 0.00 | ORB-short ORB[909.10,912.00] vol=1.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-10-24 11:20:00 | 906.45 | 907.94 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-10-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:05:00 | 916.50 | 912.48 | 0.00 | ORB-long ORB[906.00,913.95] vol=1.9x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:20:00 | 919.00 | 914.82 | 0.00 | T1 1.5R @ 919.00 |
| Stop hit — per-position SL triggered | 2025-10-27 12:35:00 | 916.50 | 915.77 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 935.65 | 931.41 | 0.00 | ORB-long ORB[926.90,933.80] vol=1.8x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-10-28 09:35:00 | 933.29 | 931.65 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 11:10:00 | 943.20 | 939.85 | 0.00 | ORB-long ORB[936.35,942.45] vol=1.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-10-30 11:30:00 | 941.56 | 940.55 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:15:00 | 944.30 | 945.62 | 0.00 | ORB-short ORB[944.50,952.55] vol=2.1x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-11-04 10:25:00 | 946.10 | 945.60 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-11-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:25:00 | 976.45 | 975.67 | 0.00 | ORB-long ORB[971.30,976.30] vol=2.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-11-24 10:35:00 | 975.26 | 975.68 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-11-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:35:00 | 998.15 | 992.55 | 0.00 | ORB-long ORB[986.60,990.75] vol=1.6x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-11-26 10:40:00 | 996.46 | 992.82 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:15:00 | 979.30 | 981.18 | 0.00 | ORB-short ORB[983.10,988.55] vol=1.7x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:40:00 | 977.03 | 980.75 | 0.00 | T1 1.5R @ 977.03 |
| Stop hit — per-position SL triggered | 2025-11-27 12:45:00 | 979.30 | 979.38 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:00:00 | 974.00 | 976.59 | 0.00 | ORB-short ORB[975.55,980.00] vol=1.7x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 975.56 | 976.41 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:05:00 | 952.00 | 959.82 | 0.00 | ORB-short ORB[961.45,972.55] vol=3.5x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:25:00 | 949.32 | 957.86 | 0.00 | T1 1.5R @ 949.32 |
| Stop hit — per-position SL triggered | 2025-12-03 11:35:00 | 952.00 | 955.81 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:05:00 | 952.95 | 951.02 | 0.00 | ORB-long ORB[946.70,951.25] vol=3.3x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:10:00 | 955.37 | 952.30 | 0.00 | T1 1.5R @ 955.37 |
| Stop hit — per-position SL triggered | 2025-12-05 10:25:00 | 952.95 | 952.62 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:00:00 | 961.60 | 965.73 | 0.00 | ORB-short ORB[968.00,972.50] vol=1.8x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:40:00 | 959.12 | 963.64 | 0.00 | T1 1.5R @ 959.12 |
| Target hit | 2025-12-08 15:00:00 | 958.00 | 957.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:15:00 | 962.40 | 957.05 | 0.00 | ORB-long ORB[951.70,956.00] vol=1.9x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-12-09 11:20:00 | 960.48 | 957.18 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 11:00:00 | 977.70 | 973.83 | 0.00 | ORB-long ORB[964.30,976.00] vol=2.3x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-12-17 11:25:00 | 976.01 | 974.16 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 10:45:00 | 976.85 | 979.45 | 0.00 | ORB-short ORB[980.00,983.00] vol=1.5x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:00:00 | 975.35 | 978.67 | 0.00 | T1 1.5R @ 975.35 |
| Target hit | 2025-12-22 15:20:00 | 974.20 | 975.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2025-12-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:10:00 | 962.80 | 964.15 | 0.00 | ORB-short ORB[964.35,968.90] vol=1.6x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 963.87 | 963.83 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 967.80 | 965.55 | 0.00 | ORB-long ORB[960.20,964.75] vol=2.7x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:20:00 | 969.81 | 966.42 | 0.00 | T1 1.5R @ 969.81 |
| Target hit | 2025-12-30 15:20:00 | 973.60 | 971.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2025-12-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:55:00 | 983.10 | 978.96 | 0.00 | ORB-long ORB[973.50,978.90] vol=5.3x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 11:35:00 | 985.40 | 980.78 | 0.00 | T1 1.5R @ 985.40 |
| Stop hit — per-position SL triggered | 2025-12-31 12:00:00 | 983.10 | 981.10 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:00:00 | 985.70 | 983.40 | 0.00 | ORB-long ORB[980.35,984.60] vol=2.3x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-01-01 10:10:00 | 984.33 | 983.62 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 11:00:00 | 992.50 | 989.79 | 0.00 | ORB-long ORB[983.95,990.40] vol=1.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-01-02 11:10:00 | 990.97 | 990.07 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 10:50:00 | 1014.85 | 1012.02 | 0.00 | ORB-long ORB[1004.95,1013.20] vol=1.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:05:00 | 1017.58 | 1012.93 | 0.00 | T1 1.5R @ 1017.58 |
| Target hit | 2026-01-06 15:20:00 | 1018.70 | 1018.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2026-01-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:10:00 | 1008.75 | 1012.41 | 0.00 | ORB-short ORB[1013.10,1022.70] vol=2.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2026-01-07 10:30:00 | 1010.68 | 1011.95 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1002.95 | 1007.22 | 0.00 | ORB-short ORB[1003.05,1008.20] vol=1.8x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 12:00:00 | 1000.26 | 1006.00 | 0.00 | T1 1.5R @ 1000.26 |
| Target hit | 2026-01-08 15:20:00 | 998.15 | 1001.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2026-01-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:55:00 | 1004.85 | 1003.24 | 0.00 | ORB-long ORB[994.00,1003.70] vol=2.2x ATR=1.99 |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 1002.86 | 1003.74 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-01-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:00:00 | 1037.20 | 1033.65 | 0.00 | ORB-long ORB[1029.15,1034.70] vol=3.4x ATR=2.41 |
| Stop hit — per-position SL triggered | 2026-01-16 10:05:00 | 1034.79 | 1033.79 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-01-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:50:00 | 1025.90 | 1034.74 | 0.00 | ORB-short ORB[1031.40,1038.50] vol=1.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-01-21 10:55:00 | 1028.15 | 1034.08 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-01-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:50:00 | 1062.30 | 1052.24 | 0.00 | ORB-long ORB[1044.15,1059.40] vol=1.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2026-01-28 11:10:00 | 1059.42 | 1053.91 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 11:10:00 | 1074.45 | 1068.42 | 0.00 | ORB-long ORB[1060.30,1068.70] vol=2.0x ATR=2.18 |
| Stop hit — per-position SL triggered | 2026-01-29 12:05:00 | 1072.27 | 1069.83 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-02-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 10:20:00 | 1080.00 | 1073.20 | 0.00 | ORB-long ORB[1067.50,1074.90] vol=1.9x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-02-05 10:25:00 | 1077.89 | 1073.51 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-02-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:30:00 | 1058.90 | 1067.13 | 0.00 | ORB-short ORB[1067.20,1073.60] vol=1.8x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 10:40:00 | 1054.79 | 1065.37 | 0.00 | T1 1.5R @ 1054.79 |
| Stop hit — per-position SL triggered | 2026-02-06 11:35:00 | 1058.90 | 1062.40 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:10:00 | 1143.20 | 1144.43 | 0.00 | ORB-short ORB[1144.50,1154.00] vol=1.7x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-02-10 12:50:00 | 1145.46 | 1143.51 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-02-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:30:00 | 1161.40 | 1150.93 | 0.00 | ORB-long ORB[1142.80,1151.10] vol=2.9x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:35:00 | 1165.55 | 1153.96 | 0.00 | T1 1.5R @ 1165.55 |
| Target hit | 2026-02-11 15:20:00 | 1180.80 | 1173.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2026-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:00:00 | 1197.70 | 1189.64 | 0.00 | ORB-long ORB[1174.80,1188.00] vol=1.6x ATR=4.30 |
| Stop hit — per-position SL triggered | 2026-02-12 11:35:00 | 1193.40 | 1194.91 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 1215.60 | 1210.37 | 0.00 | ORB-long ORB[1204.10,1212.00] vol=1.5x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:20:00 | 1220.28 | 1212.56 | 0.00 | T1 1.5R @ 1220.28 |
| Target hit | 2026-02-17 14:00:00 | 1217.90 | 1217.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 88 — SELL (started 2026-02-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:35:00 | 1189.90 | 1198.77 | 0.00 | ORB-short ORB[1199.00,1205.60] vol=1.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-02-26 10:40:00 | 1192.74 | 1197.74 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-03-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 11:00:00 | 1165.30 | 1167.14 | 0.00 | ORB-short ORB[1166.20,1181.00] vol=2.0x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:30:00 | 1160.11 | 1166.59 | 0.00 | T1 1.5R @ 1160.11 |
| Stop hit — per-position SL triggered | 2026-03-04 14:00:00 | 1165.30 | 1164.31 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:40:00 | 1168.10 | 1174.83 | 0.00 | ORB-short ORB[1174.50,1183.20] vol=2.3x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:30:00 | 1163.82 | 1169.96 | 0.00 | T1 1.5R @ 1163.82 |
| Target hit | 2026-03-05 14:35:00 | 1164.50 | 1163.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 91 — SELL (started 2026-03-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1150.00 | 1160.19 | 0.00 | ORB-short ORB[1162.60,1169.30] vol=3.4x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 1153.08 | 1158.44 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2026-03-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:45:00 | 1067.90 | 1071.55 | 0.00 | ORB-short ORB[1072.10,1081.30] vol=3.9x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:35:00 | 1061.77 | 1068.76 | 0.00 | T1 1.5R @ 1061.77 |
| Target hit | 2026-03-13 15:20:00 | 1044.80 | 1057.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 93 — SELL (started 2026-03-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:55:00 | 1064.10 | 1066.54 | 0.00 | ORB-short ORB[1064.20,1072.70] vol=2.3x ATR=2.71 |
| Stop hit — per-position SL triggered | 2026-03-18 11:25:00 | 1066.81 | 1065.78 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-04-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 09:35:00 | 1026.35 | 1022.32 | 0.00 | ORB-long ORB[1011.00,1023.00] vol=1.7x ATR=4.79 |
| Stop hit — per-position SL triggered | 2026-04-06 09:40:00 | 1021.56 | 1022.37 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2026-04-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:35:00 | 1055.05 | 1058.37 | 0.00 | ORB-short ORB[1056.15,1064.35] vol=1.6x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:20:00 | 1049.94 | 1057.07 | 0.00 | T1 1.5R @ 1049.94 |
| Target hit | 2026-04-09 15:20:00 | 1040.00 | 1048.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 96 — SELL (started 2026-04-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:50:00 | 1074.45 | 1080.20 | 0.00 | ORB-short ORB[1075.00,1088.00] vol=1.5x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:05:00 | 1070.17 | 1079.25 | 0.00 | T1 1.5R @ 1070.17 |
| Stop hit — per-position SL triggered | 2026-04-15 14:20:00 | 1074.45 | 1074.74 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2026-04-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:00:00 | 1075.00 | 1077.90 | 0.00 | ORB-short ORB[1078.00,1084.15] vol=2.1x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:35:00 | 1071.05 | 1076.21 | 0.00 | T1 1.5R @ 1071.05 |
| Target hit | 2026-04-16 14:35:00 | 1071.40 | 1071.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 98 — SELL (started 2026-04-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:30:00 | 1108.00 | 1109.13 | 0.00 | ORB-short ORB[1108.20,1114.80] vol=1.6x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-04-27 13:20:00 | 1110.97 | 1108.12 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 1056.00 | 1062.52 | 0.00 | ORB-short ORB[1060.00,1068.00] vol=1.9x ATR=2.40 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 1058.40 | 1061.24 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 1065.70 | 1071.09 | 0.00 | ORB-short ORB[1072.00,1079.50] vol=1.5x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-05-06 11:45:00 | 1068.20 | 1070.30 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 1092.00 | 1096.40 | 0.00 | ORB-short ORB[1092.50,1108.00] vol=1.6x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:25:00 | 1087.90 | 1095.08 | 0.00 | T1 1.5R @ 1087.90 |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 1092.00 | 1093.38 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-12 11:15:00 | 802.90 | 2025-05-12 11:35:00 | 800.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-13 09:35:00 | 807.35 | 2025-05-13 09:40:00 | 805.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-15 09:35:00 | 796.60 | 2025-05-15 10:30:00 | 798.28 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-05-19 10:20:00 | 799.55 | 2025-05-19 10:25:00 | 798.02 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-05-27 11:15:00 | 797.60 | 2025-05-27 11:25:00 | 799.65 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-05-27 11:15:00 | 797.60 | 2025-05-27 11:50:00 | 797.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 10:50:00 | 809.60 | 2025-06-04 11:15:00 | 807.62 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-06-04 10:50:00 | 809.60 | 2025-06-04 15:20:00 | 806.05 | TARGET_HIT | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-06 10:05:00 | 810.20 | 2025-06-06 10:10:00 | 808.31 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-09 10:30:00 | 821.90 | 2025-06-09 10:35:00 | 820.51 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-06-10 11:00:00 | 816.00 | 2025-06-10 11:10:00 | 817.21 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-06-11 10:10:00 | 819.00 | 2025-06-11 13:20:00 | 817.80 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-06-12 10:40:00 | 811.50 | 2025-06-12 10:50:00 | 812.74 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-06-17 09:35:00 | 798.55 | 2025-06-17 09:40:00 | 797.03 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-06-18 10:55:00 | 790.90 | 2025-06-18 12:05:00 | 792.08 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-06-19 10:00:00 | 785.50 | 2025-06-19 10:05:00 | 783.34 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-06-19 10:00:00 | 785.50 | 2025-06-19 11:20:00 | 785.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-20 10:40:00 | 794.25 | 2025-06-20 11:20:00 | 796.76 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-06-20 10:40:00 | 794.25 | 2025-06-20 12:25:00 | 794.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-26 10:45:00 | 793.95 | 2025-06-26 10:55:00 | 792.21 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-06-26 10:45:00 | 793.95 | 2025-06-26 12:00:00 | 793.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-30 10:00:00 | 815.40 | 2025-06-30 10:10:00 | 817.59 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-06-30 10:00:00 | 815.40 | 2025-06-30 15:20:00 | 821.00 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-07-02 11:15:00 | 816.45 | 2025-07-02 11:25:00 | 817.59 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-07-10 10:40:00 | 814.35 | 2025-07-10 10:45:00 | 813.23 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-07-14 09:35:00 | 812.15 | 2025-07-14 09:40:00 | 814.05 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-07-14 09:35:00 | 812.15 | 2025-07-14 10:30:00 | 812.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-16 09:30:00 | 820.15 | 2025-07-16 09:35:00 | 822.08 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-07-16 09:30:00 | 820.15 | 2025-07-16 09:50:00 | 820.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-17 10:00:00 | 841.60 | 2025-07-17 10:05:00 | 839.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-22 11:00:00 | 820.85 | 2025-07-22 11:15:00 | 818.92 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-07-22 11:00:00 | 820.85 | 2025-07-22 15:20:00 | 815.10 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2025-07-25 11:15:00 | 811.95 | 2025-07-25 11:45:00 | 810.30 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-07-25 11:15:00 | 811.95 | 2025-07-25 15:20:00 | 806.25 | TARGET_HIT | 0.50 | 0.70% |
| SELL | retest1 | 2025-07-28 10:45:00 | 803.40 | 2025-07-28 11:00:00 | 804.64 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-07-30 09:50:00 | 800.15 | 2025-07-30 10:25:00 | 798.97 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-07-31 11:00:00 | 798.60 | 2025-07-31 11:30:00 | 800.44 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-07-31 11:00:00 | 798.60 | 2025-07-31 14:25:00 | 798.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-01 11:15:00 | 798.15 | 2025-08-01 11:25:00 | 800.08 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-08-01 11:15:00 | 798.15 | 2025-08-01 12:00:00 | 798.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-08 09:45:00 | 802.90 | 2025-08-08 10:50:00 | 800.62 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-08-08 09:45:00 | 802.90 | 2025-08-08 13:15:00 | 802.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 09:40:00 | 822.40 | 2025-08-11 09:45:00 | 819.54 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-14 10:15:00 | 827.25 | 2025-08-14 11:15:00 | 826.02 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-08-19 10:15:00 | 829.55 | 2025-08-19 10:20:00 | 828.41 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-08-22 11:15:00 | 820.60 | 2025-08-22 11:40:00 | 821.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2025-08-25 11:05:00 | 816.25 | 2025-08-25 11:15:00 | 817.27 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-09-02 11:10:00 | 808.50 | 2025-09-02 11:50:00 | 807.59 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2025-09-05 10:45:00 | 807.85 | 2025-09-05 11:10:00 | 809.09 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-09 10:45:00 | 806.50 | 2025-09-09 10:50:00 | 807.43 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-09-10 10:15:00 | 817.90 | 2025-09-10 10:20:00 | 819.97 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-09-10 10:15:00 | 817.90 | 2025-09-10 14:30:00 | 817.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 10:25:00 | 840.00 | 2025-09-17 10:30:00 | 842.32 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-09-17 10:25:00 | 840.00 | 2025-09-17 11:25:00 | 840.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-19 10:20:00 | 860.30 | 2025-09-19 10:25:00 | 862.87 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-19 10:20:00 | 860.30 | 2025-09-19 10:35:00 | 860.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-23 10:10:00 | 859.45 | 2025-09-23 10:20:00 | 857.59 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-25 11:10:00 | 870.00 | 2025-09-25 11:15:00 | 868.45 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-26 11:00:00 | 856.30 | 2025-09-26 11:05:00 | 858.12 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-01 10:35:00 | 868.25 | 2025-10-01 11:20:00 | 864.93 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-01 10:35:00 | 868.25 | 2025-10-01 13:45:00 | 868.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-07 10:25:00 | 868.05 | 2025-10-07 10:30:00 | 869.68 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-08 11:00:00 | 861.00 | 2025-10-08 11:15:00 | 858.47 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-08 11:00:00 | 861.00 | 2025-10-08 11:25:00 | 861.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 10:00:00 | 874.80 | 2025-10-10 10:05:00 | 877.71 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-10 10:00:00 | 874.80 | 2025-10-10 15:20:00 | 880.90 | TARGET_HIT | 0.50 | 0.70% |
| BUY | retest1 | 2025-10-13 09:45:00 | 883.00 | 2025-10-13 09:55:00 | 885.50 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-10-13 09:45:00 | 883.00 | 2025-10-13 13:00:00 | 884.30 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-10-14 10:40:00 | 878.95 | 2025-10-14 11:05:00 | 876.86 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-10-14 10:40:00 | 878.95 | 2025-10-14 14:40:00 | 877.30 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-10-15 11:00:00 | 883.65 | 2025-10-15 11:45:00 | 882.33 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-10-17 11:10:00 | 893.95 | 2025-10-17 11:20:00 | 892.43 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-20 09:45:00 | 898.65 | 2025-10-20 10:00:00 | 896.34 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-23 09:55:00 | 913.50 | 2025-10-23 11:15:00 | 916.14 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-10-23 09:55:00 | 913.50 | 2025-10-23 12:45:00 | 913.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-24 11:05:00 | 905.00 | 2025-10-24 11:20:00 | 906.45 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-27 10:05:00 | 916.50 | 2025-10-27 11:20:00 | 919.00 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-10-27 10:05:00 | 916.50 | 2025-10-27 12:35:00 | 916.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 09:30:00 | 935.65 | 2025-10-28 09:35:00 | 933.29 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-30 11:10:00 | 943.20 | 2025-10-30 11:30:00 | 941.56 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-04 10:15:00 | 944.30 | 2025-11-04 10:25:00 | 946.10 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-24 10:25:00 | 976.45 | 2025-11-24 10:35:00 | 975.26 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-11-26 10:35:00 | 998.15 | 2025-11-26 10:40:00 | 996.46 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-27 11:15:00 | 979.30 | 2025-11-27 11:40:00 | 977.03 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-11-27 11:15:00 | 979.30 | 2025-11-27 12:45:00 | 979.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-02 10:00:00 | 974.00 | 2025-12-02 10:15:00 | 975.56 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-03 11:05:00 | 952.00 | 2025-12-03 11:25:00 | 949.32 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-12-03 11:05:00 | 952.00 | 2025-12-03 11:35:00 | 952.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-05 10:05:00 | 952.95 | 2025-12-05 10:10:00 | 955.37 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-12-05 10:05:00 | 952.95 | 2025-12-05 10:25:00 | 952.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 11:00:00 | 961.60 | 2025-12-08 11:40:00 | 959.12 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-08 11:00:00 | 961.60 | 2025-12-08 15:00:00 | 958.00 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-09 11:15:00 | 962.40 | 2025-12-09 11:20:00 | 960.48 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-17 11:00:00 | 977.70 | 2025-12-17 11:25:00 | 976.01 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-22 10:45:00 | 976.85 | 2025-12-22 11:00:00 | 975.35 | PARTIAL | 0.50 | 0.15% |
| SELL | retest1 | 2025-12-22 10:45:00 | 976.85 | 2025-12-22 15:20:00 | 974.20 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-29 11:10:00 | 962.80 | 2025-12-29 12:15:00 | 963.87 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2025-12-30 10:50:00 | 967.80 | 2025-12-30 11:20:00 | 969.81 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-12-30 10:50:00 | 967.80 | 2025-12-30 15:20:00 | 973.60 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-12-31 10:55:00 | 983.10 | 2025-12-31 11:35:00 | 985.40 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-12-31 10:55:00 | 983.10 | 2025-12-31 12:00:00 | 983.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-01 10:00:00 | 985.70 | 2026-01-01 10:10:00 | 984.33 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2026-01-02 11:00:00 | 992.50 | 2026-01-02 11:10:00 | 990.97 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2026-01-06 10:50:00 | 1014.85 | 2026-01-06 11:05:00 | 1017.58 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-01-06 10:50:00 | 1014.85 | 2026-01-06 15:20:00 | 1018.70 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-07 10:10:00 | 1008.75 | 2026-01-07 10:30:00 | 1010.68 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-01-08 11:15:00 | 1002.95 | 2026-01-08 12:00:00 | 1000.26 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-01-08 11:15:00 | 1002.95 | 2026-01-08 15:20:00 | 998.15 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2026-01-09 10:55:00 | 1004.85 | 2026-01-09 11:15:00 | 1002.86 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-16 10:00:00 | 1037.20 | 2026-01-16 10:05:00 | 1034.79 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-21 10:50:00 | 1025.90 | 2026-01-21 10:55:00 | 1028.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-28 10:50:00 | 1062.30 | 2026-01-28 11:10:00 | 1059.42 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-29 11:10:00 | 1074.45 | 2026-01-29 12:05:00 | 1072.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-05 10:20:00 | 1080.00 | 2026-02-05 10:25:00 | 1077.89 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-06 10:30:00 | 1058.90 | 2026-02-06 10:40:00 | 1054.79 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-06 10:30:00 | 1058.90 | 2026-02-06 11:35:00 | 1058.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-10 11:10:00 | 1143.20 | 2026-02-10 12:50:00 | 1145.46 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-11 10:30:00 | 1161.40 | 2026-02-11 10:35:00 | 1165.55 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-11 10:30:00 | 1161.40 | 2026-02-11 15:20:00 | 1180.80 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2026-02-12 10:00:00 | 1197.70 | 2026-02-12 11:35:00 | 1193.40 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-17 10:05:00 | 1215.60 | 2026-02-17 10:20:00 | 1220.28 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-17 10:05:00 | 1215.60 | 2026-02-17 14:00:00 | 1217.90 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-02-26 10:35:00 | 1189.90 | 2026-02-26 10:40:00 | 1192.74 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-04 11:00:00 | 1165.30 | 2026-03-04 11:30:00 | 1160.11 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-03-04 11:00:00 | 1165.30 | 2026-03-04 14:00:00 | 1165.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1168.10 | 2026-03-05 11:30:00 | 1163.82 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-05 10:40:00 | 1168.10 | 2026-03-05 14:35:00 | 1164.50 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-03-06 10:30:00 | 1150.00 | 2026-03-06 11:00:00 | 1153.08 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-03-13 09:45:00 | 1067.90 | 2026-03-13 10:35:00 | 1061.77 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-03-13 09:45:00 | 1067.90 | 2026-03-13 15:20:00 | 1044.80 | TARGET_HIT | 0.50 | 2.16% |
| SELL | retest1 | 2026-03-18 10:55:00 | 1064.10 | 2026-03-18 11:25:00 | 1066.81 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-06 09:35:00 | 1026.35 | 2026-04-06 09:40:00 | 1021.56 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-09 10:35:00 | 1055.05 | 2026-04-09 11:20:00 | 1049.94 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-04-09 10:35:00 | 1055.05 | 2026-04-09 15:20:00 | 1040.00 | TARGET_HIT | 0.50 | 1.43% |
| SELL | retest1 | 2026-04-15 10:50:00 | 1074.45 | 2026-04-15 11:05:00 | 1070.17 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-04-15 10:50:00 | 1074.45 | 2026-04-15 14:20:00 | 1074.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 10:00:00 | 1075.00 | 2026-04-16 11:35:00 | 1071.05 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-04-16 10:00:00 | 1075.00 | 2026-04-16 14:35:00 | 1071.40 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-27 10:30:00 | 1108.00 | 2026-04-27 13:20:00 | 1110.97 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-05-05 11:00:00 | 1056.00 | 2026-05-05 11:15:00 | 1058.40 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-06 11:15:00 | 1065.70 | 2026-05-06 11:45:00 | 1068.20 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-07 11:10:00 | 1092.00 | 2026-05-07 11:25:00 | 1087.90 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-05-07 11:10:00 | 1092.00 | 2026-05-07 12:15:00 | 1092.00 | STOP_HIT | 0.50 | 0.00% |
