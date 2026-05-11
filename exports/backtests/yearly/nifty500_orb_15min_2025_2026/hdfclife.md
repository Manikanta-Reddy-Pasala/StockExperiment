# HDFC Life Insurance Company Ltd. (HDFCLIFE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 619.60
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
| ENTRY1 | 93 |
| ENTRY2 | 0 |
| PARTIAL | 42 |
| TARGET_HIT | 19 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 74
- **Target hits / Stop hits / Partials:** 19 / 74 / 42
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 16.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 28 | 40.6% | 8 | 41 | 20 | 0.07% | 5.1% |
| BUY @ 2nd Alert (retest1) | 69 | 28 | 40.6% | 8 | 41 | 20 | 0.07% | 5.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 33 | 50.0% | 11 | 33 | 22 | 0.17% | 11.0% |
| SELL @ 2nd Alert (retest1) | 66 | 33 | 50.0% | 11 | 33 | 22 | 0.17% | 11.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 135 | 61 | 45.2% | 19 | 74 | 42 | 0.12% | 16.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 11:10:00 | 744.95 | 740.29 | 0.00 | ORB-long ORB[736.40,742.80] vol=2.6x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 11:25:00 | 747.80 | 741.80 | 0.00 | T1 1.5R @ 747.80 |
| Stop hit — per-position SL triggered | 2025-05-14 12:50:00 | 744.95 | 744.32 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:25:00 | 755.95 | 751.70 | 0.00 | ORB-long ORB[746.00,751.50] vol=1.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 11:05:00 | 758.55 | 753.63 | 0.00 | T1 1.5R @ 758.55 |
| Stop hit — per-position SL triggered | 2025-05-21 12:10:00 | 755.95 | 755.38 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:45:00 | 784.95 | 780.17 | 0.00 | ORB-long ORB[776.05,780.75] vol=2.1x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-05-28 09:50:00 | 783.08 | 780.69 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:20:00 | 761.10 | 759.28 | 0.00 | ORB-long ORB[755.50,759.85] vol=1.8x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-06-05 10:45:00 | 759.46 | 759.44 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 09:45:00 | 747.00 | 750.78 | 0.00 | ORB-short ORB[749.00,759.70] vol=1.6x ATR=2.29 |
| Stop hit — per-position SL triggered | 2025-06-09 10:10:00 | 749.29 | 749.88 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:30:00 | 771.00 | 765.87 | 0.00 | ORB-long ORB[760.25,768.85] vol=1.7x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-06-11 10:35:00 | 768.92 | 766.45 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 11:00:00 | 766.00 | 759.50 | 0.00 | ORB-long ORB[750.10,759.70] vol=2.1x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-06-16 11:10:00 | 764.17 | 759.72 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:55:00 | 761.90 | 764.53 | 0.00 | ORB-short ORB[762.30,767.95] vol=2.2x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 759.57 | 763.70 | 0.00 | T1 1.5R @ 759.57 |
| Target hit | 2025-06-19 13:15:00 | 758.00 | 757.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — BUY (started 2025-06-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 11:10:00 | 778.45 | 771.43 | 0.00 | ORB-long ORB[758.00,767.95] vol=1.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-06-20 14:45:00 | 776.49 | 776.40 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:50:00 | 804.10 | 799.78 | 0.00 | ORB-long ORB[797.20,804.00] vol=3.2x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:55:00 | 807.41 | 800.73 | 0.00 | T1 1.5R @ 807.41 |
| Stop hit — per-position SL triggered | 2025-06-27 11:00:00 | 804.10 | 800.85 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:15:00 | 810.00 | 814.65 | 0.00 | ORB-short ORB[812.00,816.15] vol=1.7x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:40:00 | 806.62 | 813.67 | 0.00 | T1 1.5R @ 806.62 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 810.00 | 812.53 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:45:00 | 799.60 | 804.94 | 0.00 | ORB-short ORB[804.70,812.60] vol=4.0x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:30:00 | 797.16 | 802.95 | 0.00 | T1 1.5R @ 797.16 |
| Stop hit — per-position SL triggered | 2025-07-02 10:55:00 | 799.60 | 802.48 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 10:45:00 | 789.70 | 787.67 | 0.00 | ORB-long ORB[783.10,789.20] vol=1.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-07-08 11:50:00 | 787.91 | 788.52 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:10:00 | 779.35 | 783.69 | 0.00 | ORB-short ORB[782.00,788.00] vol=1.8x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 10:50:00 | 777.00 | 781.96 | 0.00 | T1 1.5R @ 777.00 |
| Stop hit — per-position SL triggered | 2025-07-10 11:10:00 | 779.35 | 781.42 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:45:00 | 742.70 | 748.36 | 0.00 | ORB-short ORB[748.75,757.00] vol=1.8x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:55:00 | 739.70 | 745.49 | 0.00 | T1 1.5R @ 739.70 |
| Target hit | 2025-07-18 15:20:00 | 737.45 | 741.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-07-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 10:20:00 | 759.10 | 756.19 | 0.00 | ORB-long ORB[751.95,758.60] vol=6.0x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:25:00 | 761.73 | 757.21 | 0.00 | T1 1.5R @ 761.73 |
| Stop hit — per-position SL triggered | 2025-07-22 10:45:00 | 759.10 | 758.00 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 11:00:00 | 756.15 | 754.00 | 0.00 | ORB-long ORB[752.00,756.05] vol=3.2x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-07-25 11:10:00 | 754.66 | 754.07 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:45:00 | 769.10 | 766.83 | 0.00 | ORB-long ORB[760.90,766.50] vol=1.9x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-07-28 09:55:00 | 767.36 | 766.90 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:15:00 | 764.05 | 762.10 | 0.00 | ORB-long ORB[758.30,762.90] vol=1.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-07-29 10:25:00 | 762.38 | 762.15 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 09:55:00 | 749.20 | 750.36 | 0.00 | ORB-short ORB[751.05,756.35] vol=1.8x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-08-01 10:10:00 | 751.06 | 750.36 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 11:10:00 | 758.00 | 758.15 | 0.00 | ORB-short ORB[759.50,764.00] vol=2.4x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:30:00 | 755.70 | 757.95 | 0.00 | T1 1.5R @ 755.70 |
| Target hit | 2025-08-11 12:25:00 | 757.65 | 757.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — BUY (started 2025-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:30:00 | 770.55 | 767.86 | 0.00 | ORB-long ORB[762.65,767.70] vol=2.8x ATR=1.81 |
| Stop hit — per-position SL triggered | 2025-08-12 10:05:00 | 768.74 | 769.45 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-08-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:40:00 | 776.00 | 772.14 | 0.00 | ORB-long ORB[763.60,772.65] vol=2.1x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:00:00 | 778.41 | 773.44 | 0.00 | T1 1.5R @ 778.41 |
| Target hit | 2025-08-13 14:15:00 | 777.55 | 777.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2025-08-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 09:35:00 | 789.45 | 786.21 | 0.00 | ORB-long ORB[777.95,788.55] vol=3.0x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-08-14 09:55:00 | 787.49 | 787.02 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:50:00 | 809.30 | 806.11 | 0.00 | ORB-long ORB[798.05,809.05] vol=1.5x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:05:00 | 813.87 | 807.15 | 0.00 | T1 1.5R @ 813.87 |
| Stop hit — per-position SL triggered | 2025-08-18 10:35:00 | 809.30 | 808.09 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 10:55:00 | 782.00 | 777.04 | 0.00 | ORB-long ORB[772.30,778.95] vol=1.9x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:10:00 | 785.31 | 778.36 | 0.00 | T1 1.5R @ 785.31 |
| Stop hit — per-position SL triggered | 2025-08-28 12:20:00 | 782.00 | 779.93 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:50:00 | 768.70 | 770.65 | 0.00 | ORB-short ORB[772.30,778.45] vol=2.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 770.78 | 770.29 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:15:00 | 782.45 | 778.39 | 0.00 | ORB-long ORB[771.50,782.00] vol=7.1x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-09-01 12:35:00 | 780.54 | 780.24 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:40:00 | 753.95 | 754.58 | 0.00 | ORB-short ORB[754.00,757.60] vol=5.8x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-09-09 11:05:00 | 755.31 | 754.57 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:55:00 | 773.95 | 775.63 | 0.00 | ORB-short ORB[774.45,779.30] vol=2.3x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:35:00 | 772.03 | 775.40 | 0.00 | T1 1.5R @ 772.03 |
| Stop hit — per-position SL triggered | 2025-09-11 12:20:00 | 773.95 | 774.83 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 11:15:00 | 777.70 | 773.65 | 0.00 | ORB-long ORB[773.20,777.40] vol=1.6x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:45:00 | 779.57 | 775.47 | 0.00 | T1 1.5R @ 779.57 |
| Stop hit — per-position SL triggered | 2025-09-12 12:05:00 | 777.70 | 776.25 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:30:00 | 776.90 | 774.61 | 0.00 | ORB-long ORB[766.20,775.25] vol=1.8x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 10:40:00 | 779.04 | 775.12 | 0.00 | T1 1.5R @ 779.04 |
| Target hit | 2025-09-18 14:20:00 | 782.40 | 783.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:15:00 | 769.05 | 770.09 | 0.00 | ORB-short ORB[770.45,774.70] vol=3.0x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-09-24 10:40:00 | 770.38 | 769.95 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:05:00 | 764.70 | 765.91 | 0.00 | ORB-short ORB[765.20,769.95] vol=3.9x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-09-25 10:45:00 | 766.48 | 765.63 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 10:55:00 | 759.10 | 764.50 | 0.00 | ORB-short ORB[762.00,767.50] vol=1.8x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-09-29 11:05:00 | 760.70 | 764.28 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 11:10:00 | 760.45 | 757.94 | 0.00 | ORB-long ORB[752.65,759.00] vol=2.9x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 759.02 | 757.96 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 764.10 | 766.28 | 0.00 | ORB-short ORB[764.50,770.00] vol=1.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:10:00 | 761.01 | 764.01 | 0.00 | T1 1.5R @ 761.01 |
| Target hit | 2025-10-07 15:20:00 | 756.50 | 758.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2025-10-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:45:00 | 752.55 | 753.80 | 0.00 | ORB-short ORB[752.85,757.05] vol=2.1x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:10:00 | 750.30 | 753.22 | 0.00 | T1 1.5R @ 750.30 |
| Target hit | 2025-10-08 15:20:00 | 747.75 | 748.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2025-10-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:45:00 | 748.90 | 746.89 | 0.00 | ORB-long ORB[742.15,748.85] vol=2.2x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-10-13 11:20:00 | 747.61 | 747.17 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:40:00 | 741.70 | 743.43 | 0.00 | ORB-short ORB[742.60,748.15] vol=2.9x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-10-14 10:55:00 | 743.15 | 743.12 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:45:00 | 745.05 | 742.57 | 0.00 | ORB-long ORB[737.50,742.85] vol=5.6x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:50:00 | 747.70 | 742.96 | 0.00 | T1 1.5R @ 747.70 |
| Stop hit — per-position SL triggered | 2025-10-17 11:10:00 | 745.05 | 743.68 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:35:00 | 744.20 | 748.04 | 0.00 | ORB-short ORB[746.20,753.90] vol=2.6x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:40:00 | 741.73 | 747.71 | 0.00 | T1 1.5R @ 741.73 |
| Stop hit — per-position SL triggered | 2025-10-20 11:05:00 | 744.20 | 746.71 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:50:00 | 740.00 | 742.62 | 0.00 | ORB-short ORB[742.15,745.50] vol=2.9x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:05:00 | 737.93 | 741.05 | 0.00 | T1 1.5R @ 737.93 |
| Target hit | 2025-10-24 15:20:00 | 735.95 | 737.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-10-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 11:10:00 | 745.10 | 740.15 | 0.00 | ORB-long ORB[735.55,741.95] vol=1.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-10-28 12:45:00 | 743.62 | 742.16 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:00:00 | 759.15 | 755.81 | 0.00 | ORB-long ORB[749.00,757.90] vol=1.8x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-10-29 10:10:00 | 757.05 | 756.24 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 09:50:00 | 739.75 | 737.78 | 0.00 | ORB-long ORB[731.05,737.70] vol=1.8x ATR=1.62 |
| Stop hit — per-position SL triggered | 2025-11-07 10:15:00 | 738.13 | 738.50 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:35:00 | 758.30 | 753.42 | 0.00 | ORB-long ORB[749.70,753.75] vol=2.1x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-11-10 09:50:00 | 756.63 | 755.65 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:25:00 | 755.45 | 752.57 | 0.00 | ORB-long ORB[749.05,754.40] vol=3.0x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-11-11 10:35:00 | 753.68 | 753.03 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:45:00 | 771.80 | 769.97 | 0.00 | ORB-long ORB[765.00,771.20] vol=1.6x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 11:00:00 | 773.98 | 770.32 | 0.00 | T1 1.5R @ 773.98 |
| Target hit | 2025-11-12 15:20:00 | 779.65 | 778.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2025-11-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 11:00:00 | 769.95 | 773.16 | 0.00 | ORB-short ORB[770.00,778.00] vol=1.7x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 11:20:00 | 768.13 | 771.40 | 0.00 | T1 1.5R @ 768.13 |
| Stop hit — per-position SL triggered | 2025-11-17 14:05:00 | 769.95 | 768.66 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 11:10:00 | 756.70 | 757.65 | 0.00 | ORB-short ORB[756.80,760.60] vol=2.2x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-11-19 11:40:00 | 757.98 | 757.59 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:35:00 | 751.10 | 755.08 | 0.00 | ORB-short ORB[753.20,760.70] vol=2.2x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 752.86 | 753.47 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 10:10:00 | 763.70 | 762.53 | 0.00 | ORB-long ORB[758.60,763.20] vol=1.9x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-11-21 10:25:00 | 762.22 | 762.54 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 774.75 | 771.53 | 0.00 | ORB-long ORB[762.75,773.65] vol=1.9x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 09:40:00 | 777.79 | 773.26 | 0.00 | T1 1.5R @ 777.79 |
| Target hit | 2025-11-26 10:35:00 | 777.15 | 777.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — SELL (started 2025-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 09:30:00 | 771.25 | 773.36 | 0.00 | ORB-short ORB[771.75,780.40] vol=3.2x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-11-28 09:55:00 | 772.94 | 772.96 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:10:00 | 762.15 | 765.24 | 0.00 | ORB-short ORB[764.15,767.95] vol=2.0x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:40:00 | 760.39 | 764.72 | 0.00 | T1 1.5R @ 760.39 |
| Stop hit — per-position SL triggered | 2025-12-01 14:25:00 | 762.15 | 762.99 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:40:00 | 756.70 | 760.59 | 0.00 | ORB-short ORB[759.25,766.40] vol=1.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-12-02 10:30:00 | 758.41 | 759.06 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 11:00:00 | 767.75 | 764.40 | 0.00 | ORB-long ORB[759.00,761.65] vol=1.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:10:00 | 769.98 | 765.12 | 0.00 | T1 1.5R @ 769.98 |
| Target hit | 2025-12-10 14:10:00 | 769.85 | 771.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — SELL (started 2025-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 11:05:00 | 771.35 | 775.52 | 0.00 | ORB-short ORB[772.50,779.15] vol=1.8x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-12-12 11:45:00 | 772.95 | 774.81 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:50:00 | 779.65 | 775.07 | 0.00 | ORB-long ORB[768.90,776.30] vol=1.9x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-12-16 11:40:00 | 777.70 | 777.45 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:45:00 | 761.70 | 761.00 | 0.00 | ORB-long ORB[758.60,761.55] vol=1.8x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-12-23 11:00:00 | 760.35 | 760.99 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:00:00 | 743.95 | 745.96 | 0.00 | ORB-short ORB[745.20,748.00] vol=1.9x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 745.36 | 745.77 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:05:00 | 743.05 | 745.64 | 0.00 | ORB-short ORB[744.55,748.65] vol=1.6x ATR=1.62 |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 744.67 | 745.19 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:15:00 | 750.60 | 747.64 | 0.00 | ORB-long ORB[744.80,748.60] vol=2.4x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-12-31 11:30:00 | 748.97 | 747.87 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:45:00 | 754.90 | 753.77 | 0.00 | ORB-long ORB[749.05,752.55] vol=6.3x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:55:00 | 756.49 | 753.94 | 0.00 | T1 1.5R @ 756.49 |
| Stop hit — per-position SL triggered | 2026-01-02 12:50:00 | 754.90 | 755.46 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:40:00 | 760.25 | 758.39 | 0.00 | ORB-long ORB[751.50,759.50] vol=2.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2026-01-05 09:55:00 | 758.61 | 758.69 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:15:00 | 721.50 | 725.64 | 0.00 | ORB-short ORB[723.95,728.15] vol=3.0x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:40:00 | 719.22 | 725.02 | 0.00 | T1 1.5R @ 719.22 |
| Stop hit — per-position SL triggered | 2026-01-22 12:05:00 | 721.50 | 724.32 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 09:55:00 | 716.00 | 717.53 | 0.00 | ORB-short ORB[716.30,723.00] vol=6.7x ATR=1.85 |
| Stop hit — per-position SL triggered | 2026-01-23 10:10:00 | 717.85 | 717.28 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 11:00:00 | 718.05 | 713.61 | 0.00 | ORB-long ORB[708.10,716.50] vol=2.0x ATR=2.29 |
| Stop hit — per-position SL triggered | 2026-01-27 12:25:00 | 715.76 | 715.35 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:05:00 | 732.60 | 730.06 | 0.00 | ORB-long ORB[726.30,732.40] vol=1.5x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-02-01 11:25:00 | 730.88 | 730.17 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 11:00:00 | 710.65 | 716.53 | 0.00 | ORB-short ORB[714.15,719.30] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2026-02-02 11:40:00 | 712.79 | 715.20 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:50:00 | 712.30 | 716.57 | 0.00 | ORB-short ORB[717.70,723.30] vol=2.4x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-02-05 11:10:00 | 713.58 | 716.13 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:20:00 | 714.85 | 717.77 | 0.00 | ORB-short ORB[718.00,721.90] vol=1.7x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 10:40:00 | 712.55 | 716.50 | 0.00 | T1 1.5R @ 712.55 |
| Target hit | 2026-02-06 15:00:00 | 703.45 | 703.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 707.45 | 708.76 | 0.00 | ORB-short ORB[707.60,713.95] vol=2.1x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:25:00 | 705.20 | 708.16 | 0.00 | T1 1.5R @ 705.20 |
| Target hit | 2026-02-10 15:20:00 | 703.90 | 706.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2026-02-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:20:00 | 699.60 | 701.11 | 0.00 | ORB-short ORB[700.30,705.60] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 700.88 | 700.71 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:05:00 | 700.15 | 698.33 | 0.00 | ORB-long ORB[693.00,697.70] vol=5.2x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:20:00 | 702.21 | 698.78 | 0.00 | T1 1.5R @ 702.21 |
| Target hit | 2026-02-16 15:20:00 | 706.90 | 702.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — SELL (started 2026-02-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:30:00 | 738.20 | 738.87 | 0.00 | ORB-short ORB[738.40,742.80] vol=3.2x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 739.59 | 738.81 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:00:00 | 722.75 | 727.33 | 0.00 | ORB-short ORB[729.75,736.80] vol=2.2x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:10:00 | 720.47 | 726.74 | 0.00 | T1 1.5R @ 720.47 |
| Target hit | 2026-02-27 15:20:00 | 715.05 | 717.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-03-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:40:00 | 705.25 | 698.85 | 0.00 | ORB-long ORB[690.00,698.70] vol=2.2x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-03-04 10:00:00 | 702.11 | 701.35 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 671.55 | 673.51 | 0.00 | ORB-short ORB[672.15,679.20] vol=1.8x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 15:05:00 | 668.19 | 671.53 | 0.00 | T1 1.5R @ 668.19 |
| Target hit | 2026-03-06 15:20:00 | 668.30 | 671.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — SELL (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 650.95 | 652.32 | 0.00 | ORB-short ORB[651.70,657.50] vol=4.1x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-03-11 10:40:00 | 652.58 | 652.28 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 635.50 | 637.29 | 0.00 | ORB-short ORB[635.70,640.60] vol=2.2x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:40:00 | 633.03 | 636.57 | 0.00 | T1 1.5R @ 633.03 |
| Target hit | 2026-03-13 15:20:00 | 626.30 | 631.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2026-03-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:40:00 | 619.70 | 623.40 | 0.00 | ORB-short ORB[622.80,628.00] vol=2.7x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:55:00 | 617.24 | 622.90 | 0.00 | T1 1.5R @ 617.24 |
| Stop hit — per-position SL triggered | 2026-03-16 11:00:00 | 619.70 | 622.74 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:40:00 | 632.75 | 628.58 | 0.00 | ORB-long ORB[623.80,628.00] vol=2.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 630.92 | 630.47 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:10:00 | 632.65 | 628.62 | 0.00 | ORB-long ORB[624.00,631.50] vol=2.3x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:30:00 | 635.34 | 629.01 | 0.00 | T1 1.5R @ 635.34 |
| Stop hit — per-position SL triggered | 2026-03-19 14:10:00 | 632.65 | 632.27 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-04-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:50:00 | 613.55 | 603.86 | 0.00 | ORB-long ORB[592.00,600.55] vol=1.8x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:10:00 | 617.45 | 608.53 | 0.00 | T1 1.5R @ 617.45 |
| Target hit | 2026-04-13 15:20:00 | 619.15 | 617.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — SELL (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 605.35 | 606.61 | 0.00 | ORB-short ORB[605.50,610.00] vol=2.5x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 09:55:00 | 603.35 | 605.71 | 0.00 | T1 1.5R @ 603.35 |
| Stop hit — per-position SL triggered | 2026-04-21 10:05:00 | 605.35 | 605.47 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:35:00 | 606.05 | 610.74 | 0.00 | ORB-short ORB[611.30,617.25] vol=2.1x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-04-22 11:40:00 | 607.53 | 608.41 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-04-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:20:00 | 600.15 | 596.32 | 0.00 | ORB-long ORB[590.10,597.15] vol=2.5x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:50:00 | 603.35 | 597.92 | 0.00 | T1 1.5R @ 603.35 |
| Stop hit — per-position SL triggered | 2026-04-27 11:05:00 | 600.15 | 598.26 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:00:00 | 598.20 | 595.44 | 0.00 | ORB-long ORB[589.60,593.15] vol=1.5x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 11:10:00 | 600.31 | 596.10 | 0.00 | T1 1.5R @ 600.31 |
| Stop hit — per-position SL triggered | 2026-04-29 14:45:00 | 598.20 | 599.84 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-05-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 10:45:00 | 590.95 | 592.86 | 0.00 | ORB-short ORB[591.05,595.80] vol=1.6x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:00:00 | 588.82 | 592.43 | 0.00 | T1 1.5R @ 588.82 |
| Stop hit — per-position SL triggered | 2026-05-04 11:15:00 | 590.95 | 592.20 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 594.35 | 591.79 | 0.00 | ORB-long ORB[586.00,592.00] vol=3.3x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:55:00 | 597.12 | 592.72 | 0.00 | T1 1.5R @ 597.12 |
| Target hit | 2026-05-05 10:30:00 | 595.90 | 596.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 93 — BUY (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 615.15 | 612.08 | 0.00 | ORB-long ORB[608.85,614.55] vol=2.5x ATR=2.40 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 612.75 | 612.39 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 11:10:00 | 744.95 | 2025-05-14 11:25:00 | 747.80 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-05-14 11:10:00 | 744.95 | 2025-05-14 12:50:00 | 744.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-21 10:25:00 | 755.95 | 2025-05-21 11:05:00 | 758.55 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-21 10:25:00 | 755.95 | 2025-05-21 12:10:00 | 755.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 09:45:00 | 784.95 | 2025-05-28 09:50:00 | 783.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-05 10:20:00 | 761.10 | 2025-06-05 10:45:00 | 759.46 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-09 09:45:00 | 747.00 | 2025-06-09 10:10:00 | 749.29 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-11 10:30:00 | 771.00 | 2025-06-11 10:35:00 | 768.92 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-16 11:00:00 | 766.00 | 2025-06-16 11:10:00 | 764.17 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-19 10:55:00 | 761.90 | 2025-06-19 11:15:00 | 759.57 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-06-19 10:55:00 | 761.90 | 2025-06-19 13:15:00 | 758.00 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2025-06-20 11:10:00 | 778.45 | 2025-06-20 14:45:00 | 776.49 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-27 10:50:00 | 804.10 | 2025-06-27 10:55:00 | 807.41 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-06-27 10:50:00 | 804.10 | 2025-06-27 11:00:00 | 804.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:15:00 | 810.00 | 2025-07-01 11:40:00 | 806.62 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-01 10:15:00 | 810.00 | 2025-07-01 13:15:00 | 810.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 09:45:00 | 799.60 | 2025-07-02 10:30:00 | 797.16 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-02 09:45:00 | 799.60 | 2025-07-02 10:55:00 | 799.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-08 10:45:00 | 789.70 | 2025-07-08 11:50:00 | 787.91 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-10 10:10:00 | 779.35 | 2025-07-10 10:50:00 | 777.00 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-10 10:10:00 | 779.35 | 2025-07-10 11:10:00 | 779.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:45:00 | 742.70 | 2025-07-18 10:55:00 | 739.70 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-18 10:45:00 | 742.70 | 2025-07-18 15:20:00 | 737.45 | TARGET_HIT | 0.50 | 0.71% |
| BUY | retest1 | 2025-07-22 10:20:00 | 759.10 | 2025-07-22 10:25:00 | 761.73 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-07-22 10:20:00 | 759.10 | 2025-07-22 10:45:00 | 759.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-25 11:00:00 | 756.15 | 2025-07-25 11:10:00 | 754.66 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-28 09:45:00 | 769.10 | 2025-07-28 09:55:00 | 767.36 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-29 10:15:00 | 764.05 | 2025-07-29 10:25:00 | 762.38 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-01 09:55:00 | 749.20 | 2025-08-01 10:10:00 | 751.06 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-11 11:10:00 | 758.00 | 2025-08-11 11:30:00 | 755.70 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-08-11 11:10:00 | 758.00 | 2025-08-11 12:25:00 | 757.65 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2025-08-12 09:30:00 | 770.55 | 2025-08-12 10:05:00 | 768.74 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-13 10:40:00 | 776.00 | 2025-08-13 11:00:00 | 778.41 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-08-13 10:40:00 | 776.00 | 2025-08-13 14:15:00 | 777.55 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-08-14 09:35:00 | 789.45 | 2025-08-14 09:55:00 | 787.49 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-18 09:50:00 | 809.30 | 2025-08-18 10:05:00 | 813.87 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-08-18 09:50:00 | 809.30 | 2025-08-18 10:35:00 | 809.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-28 10:55:00 | 782.00 | 2025-08-28 11:10:00 | 785.31 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-08-28 10:55:00 | 782.00 | 2025-08-28 12:20:00 | 782.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-29 09:50:00 | 768.70 | 2025-08-29 10:15:00 | 770.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-01 11:15:00 | 782.45 | 2025-09-01 12:35:00 | 780.54 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-09 10:40:00 | 753.95 | 2025-09-09 11:05:00 | 755.31 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-11 10:55:00 | 773.95 | 2025-09-11 11:35:00 | 772.03 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-09-11 10:55:00 | 773.95 | 2025-09-11 12:20:00 | 773.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-12 11:15:00 | 777.70 | 2025-09-12 11:45:00 | 779.57 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-09-12 11:15:00 | 777.70 | 2025-09-12 12:05:00 | 777.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 10:30:00 | 776.90 | 2025-09-18 10:40:00 | 779.04 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-09-18 10:30:00 | 776.90 | 2025-09-18 14:20:00 | 782.40 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2025-09-24 10:15:00 | 769.05 | 2025-09-24 10:40:00 | 770.38 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-25 10:05:00 | 764.70 | 2025-09-25 10:45:00 | 766.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-29 10:55:00 | 759.10 | 2025-09-29 11:05:00 | 760.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-06 11:10:00 | 760.45 | 2025-10-06 11:15:00 | 759.02 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-07 09:30:00 | 764.10 | 2025-10-07 10:10:00 | 761.01 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-10-07 09:30:00 | 764.10 | 2025-10-07 15:20:00 | 756.50 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2025-10-08 09:45:00 | 752.55 | 2025-10-08 10:10:00 | 750.30 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-08 09:45:00 | 752.55 | 2025-10-08 15:20:00 | 747.75 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2025-10-13 10:45:00 | 748.90 | 2025-10-13 11:20:00 | 747.61 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-14 10:40:00 | 741.70 | 2025-10-14 10:55:00 | 743.15 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-17 10:45:00 | 745.05 | 2025-10-17 10:50:00 | 747.70 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-17 10:45:00 | 745.05 | 2025-10-17 11:10:00 | 745.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-20 10:35:00 | 744.20 | 2025-10-20 10:40:00 | 741.73 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-20 10:35:00 | 744.20 | 2025-10-20 11:05:00 | 744.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-24 10:50:00 | 740.00 | 2025-10-24 11:05:00 | 737.93 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-10-24 10:50:00 | 740.00 | 2025-10-24 15:20:00 | 735.95 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2025-10-28 11:10:00 | 745.10 | 2025-10-28 12:45:00 | 743.62 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-29 10:00:00 | 759.15 | 2025-10-29 10:10:00 | 757.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-11-07 09:50:00 | 739.75 | 2025-11-07 10:15:00 | 738.13 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-10 09:35:00 | 758.30 | 2025-11-10 09:50:00 | 756.63 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-11 10:25:00 | 755.45 | 2025-11-11 10:35:00 | 753.68 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-12 10:45:00 | 771.80 | 2025-11-12 11:00:00 | 773.98 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-11-12 10:45:00 | 771.80 | 2025-11-12 15:20:00 | 779.65 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2025-11-17 11:00:00 | 769.95 | 2025-11-17 11:20:00 | 768.13 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-11-17 11:00:00 | 769.95 | 2025-11-17 14:05:00 | 769.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-19 11:10:00 | 756.70 | 2025-11-19 11:40:00 | 757.98 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-20 09:35:00 | 751.10 | 2025-11-20 10:15:00 | 752.86 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-21 10:10:00 | 763.70 | 2025-11-21 10:25:00 | 762.22 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-26 09:30:00 | 774.75 | 2025-11-26 09:40:00 | 777.79 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-11-26 09:30:00 | 774.75 | 2025-11-26 10:35:00 | 777.15 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-28 09:30:00 | 771.25 | 2025-11-28 09:55:00 | 772.94 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-01 11:10:00 | 762.15 | 2025-12-01 11:40:00 | 760.39 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-12-01 11:10:00 | 762.15 | 2025-12-01 14:25:00 | 762.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-02 09:40:00 | 756.70 | 2025-12-02 10:30:00 | 758.41 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-10 11:00:00 | 767.75 | 2025-12-10 11:10:00 | 769.98 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-12-10 11:00:00 | 767.75 | 2025-12-10 14:10:00 | 769.85 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-12 11:05:00 | 771.35 | 2025-12-12 11:45:00 | 772.95 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-16 09:50:00 | 779.65 | 2025-12-16 11:40:00 | 777.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-23 10:45:00 | 761.70 | 2025-12-23 11:00:00 | 760.35 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-29 10:00:00 | 743.95 | 2025-12-29 10:15:00 | 745.36 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-30 10:05:00 | 743.05 | 2025-12-30 10:15:00 | 744.67 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-31 11:15:00 | 750.60 | 2025-12-31 11:30:00 | 748.97 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-01-02 10:45:00 | 754.90 | 2026-01-02 10:55:00 | 756.49 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2026-01-02 10:45:00 | 754.90 | 2026-01-02 12:50:00 | 754.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-05 09:40:00 | 760.25 | 2026-01-05 09:55:00 | 758.61 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-22 11:15:00 | 721.50 | 2026-01-22 11:40:00 | 719.22 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-01-22 11:15:00 | 721.50 | 2026-01-22 12:05:00 | 721.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-23 09:55:00 | 716.00 | 2026-01-23 10:10:00 | 717.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-27 11:00:00 | 718.05 | 2026-01-27 12:25:00 | 715.76 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-01 11:05:00 | 732.60 | 2026-02-01 11:25:00 | 730.88 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-02 11:00:00 | 710.65 | 2026-02-02 11:40:00 | 712.79 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-05 10:50:00 | 712.30 | 2026-02-05 11:10:00 | 713.58 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-06 10:20:00 | 714.85 | 2026-02-06 10:40:00 | 712.55 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-06 10:20:00 | 714.85 | 2026-02-06 15:00:00 | 703.45 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2026-02-10 10:35:00 | 707.45 | 2026-02-10 11:25:00 | 705.20 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-10 10:35:00 | 707.45 | 2026-02-10 15:20:00 | 703.90 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-11 10:20:00 | 699.60 | 2026-02-11 11:30:00 | 700.88 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-16 11:05:00 | 700.15 | 2026-02-16 11:20:00 | 702.21 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-16 11:05:00 | 700.15 | 2026-02-16 15:20:00 | 706.90 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2026-02-24 10:30:00 | 738.20 | 2026-02-24 11:15:00 | 739.59 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-27 11:00:00 | 722.75 | 2026-02-27 11:10:00 | 720.47 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-27 11:00:00 | 722.75 | 2026-02-27 15:20:00 | 715.05 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2026-03-04 09:40:00 | 705.25 | 2026-03-04 10:00:00 | 702.11 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-03-06 10:45:00 | 671.55 | 2026-03-06 15:05:00 | 668.19 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-06 10:45:00 | 671.55 | 2026-03-06 15:20:00 | 668.30 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-11 10:25:00 | 650.95 | 2026-03-11 10:40:00 | 652.58 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-13 10:50:00 | 635.50 | 2026-03-13 11:40:00 | 633.03 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-03-13 10:50:00 | 635.50 | 2026-03-13 15:20:00 | 626.30 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2026-03-16 10:40:00 | 619.70 | 2026-03-16 10:55:00 | 617.24 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-03-16 10:40:00 | 619.70 | 2026-03-16 11:00:00 | 619.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:40:00 | 632.75 | 2026-03-17 11:05:00 | 630.92 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-19 11:10:00 | 632.65 | 2026-03-19 11:30:00 | 635.34 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-19 11:10:00 | 632.65 | 2026-03-19 14:10:00 | 632.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 09:50:00 | 613.55 | 2026-04-13 10:10:00 | 617.45 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-04-13 09:50:00 | 613.55 | 2026-04-13 15:20:00 | 619.15 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2026-04-21 09:35:00 | 605.35 | 2026-04-21 09:55:00 | 603.35 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-04-21 09:35:00 | 605.35 | 2026-04-21 10:05:00 | 605.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-22 10:35:00 | 606.05 | 2026-04-22 11:40:00 | 607.53 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-27 10:20:00 | 600.15 | 2026-04-27 10:50:00 | 603.35 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-27 10:20:00 | 600.15 | 2026-04-27 11:05:00 | 600.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 11:00:00 | 598.20 | 2026-04-29 11:10:00 | 600.31 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-29 11:00:00 | 598.20 | 2026-04-29 14:45:00 | 598.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-04 10:45:00 | 590.95 | 2026-05-04 11:00:00 | 588.82 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-05-04 10:45:00 | 590.95 | 2026-05-04 11:15:00 | 590.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:40:00 | 594.35 | 2026-05-05 09:55:00 | 597.12 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-05 09:40:00 | 594.35 | 2026-05-05 10:30:00 | 595.90 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-05-07 09:30:00 | 615.15 | 2026-05-07 09:50:00 | 612.75 | STOP_HIT | 1.00 | -0.39% |
