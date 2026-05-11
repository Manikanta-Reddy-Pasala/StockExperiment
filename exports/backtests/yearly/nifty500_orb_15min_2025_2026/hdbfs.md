# HDB Financial Services Ltd. (HDBFS)

## Backtest Summary

- **Window:** 2025-07-02 09:40:00 → 2026-05-08 15:25:00 (12908 bars)
- **Last close:** 700.00
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
| ENTRY1 | 83 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 23 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 60
- **Target hits / Stop hits / Partials:** 23 / 60 / 38
- **Avg / median % per leg:** 0.13% / 0.09%
- **Sum % (uncompounded):** 15.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 28 | 46.7% | 9 | 32 | 19 | 0.13% | 7.5% |
| BUY @ 2nd Alert (retest1) | 60 | 28 | 46.7% | 9 | 32 | 19 | 0.13% | 7.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 61 | 33 | 54.1% | 14 | 28 | 19 | 0.13% | 7.7% |
| SELL @ 2nd Alert (retest1) | 61 | 33 | 54.1% | 14 | 28 | 19 | 0.13% | 7.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 121 | 61 | 50.4% | 23 | 60 | 38 | 0.13% | 15.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:25:00 | 841.00 | 843.53 | 0.00 | ORB-short ORB[842.05,845.90] vol=3.6x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-07-11 10:40:00 | 842.70 | 843.32 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-07-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 10:45:00 | 841.00 | 842.47 | 0.00 | ORB-short ORB[841.30,847.00] vol=2.5x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:20:00 | 838.81 | 841.89 | 0.00 | T1 1.5R @ 838.81 |
| Target hit | 2025-07-14 13:45:00 | 839.80 | 839.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2025-07-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:05:00 | 813.85 | 816.16 | 0.00 | ORB-short ORB[815.20,822.50] vol=2.0x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 12:10:00 | 812.04 | 815.49 | 0.00 | T1 1.5R @ 812.04 |
| Stop hit — per-position SL triggered | 2025-07-17 13:25:00 | 813.85 | 814.72 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-07-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 09:30:00 | 793.05 | 794.39 | 0.00 | ORB-short ORB[793.50,797.80] vol=2.1x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-07-21 09:35:00 | 794.55 | 794.46 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-07-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:40:00 | 808.00 | 805.84 | 0.00 | ORB-long ORB[796.85,807.85] vol=6.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-07-22 09:50:00 | 806.10 | 806.09 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-07-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:05:00 | 799.25 | 800.44 | 0.00 | ORB-short ORB[801.40,806.40] vol=9.1x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-07-23 15:10:00 | 800.95 | 799.86 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:50:00 | 794.35 | 797.27 | 0.00 | ORB-short ORB[797.50,802.50] vol=1.7x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 10:55:00 | 792.96 | 796.82 | 0.00 | T1 1.5R @ 792.96 |
| Target hit | 2025-07-24 15:20:00 | 784.50 | 785.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2025-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:30:00 | 768.30 | 774.97 | 0.00 | ORB-short ORB[772.65,784.00] vol=2.5x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:50:00 | 764.66 | 771.06 | 0.00 | T1 1.5R @ 764.66 |
| Target hit | 2025-07-25 13:35:00 | 760.00 | 759.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 9 — SELL (started 2025-07-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 10:35:00 | 749.65 | 753.63 | 0.00 | ORB-short ORB[753.00,762.30] vol=2.4x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 12:25:00 | 746.85 | 751.69 | 0.00 | T1 1.5R @ 746.85 |
| Stop hit — per-position SL triggered | 2025-07-28 13:00:00 | 749.65 | 751.19 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:25:00 | 742.55 | 744.87 | 0.00 | ORB-short ORB[743.05,748.10] vol=2.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-07-29 11:10:00 | 743.89 | 744.09 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:30:00 | 771.90 | 755.09 | 0.00 | ORB-long ORB[743.10,754.50] vol=2.4x ATR=3.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 10:35:00 | 777.41 | 757.01 | 0.00 | T1 1.5R @ 777.41 |
| Target hit | 2025-07-30 12:30:00 | 774.10 | 774.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2025-08-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:35:00 | 742.90 | 746.17 | 0.00 | ORB-short ORB[747.05,753.60] vol=3.2x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 10:50:00 | 740.45 | 745.10 | 0.00 | T1 1.5R @ 740.45 |
| Target hit | 2025-08-04 13:05:00 | 741.75 | 741.72 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — SELL (started 2025-08-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:30:00 | 739.95 | 743.26 | 0.00 | ORB-short ORB[741.95,749.60] vol=1.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-08-05 10:50:00 | 741.46 | 743.04 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-08-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:50:00 | 735.70 | 737.62 | 0.00 | ORB-short ORB[738.00,742.85] vol=4.7x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-08-06 10:25:00 | 737.37 | 737.24 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-08-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 09:35:00 | 744.45 | 739.13 | 0.00 | ORB-long ORB[732.70,739.00] vol=4.3x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-08-07 09:50:00 | 742.59 | 741.15 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-08 09:40:00 | 759.00 | 754.58 | 0.00 | ORB-long ORB[747.50,757.00] vol=1.8x ATR=2.71 |
| Stop hit — per-position SL triggered | 2025-08-08 09:45:00 | 756.29 | 754.89 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:55:00 | 742.00 | 746.64 | 0.00 | ORB-short ORB[744.70,752.30] vol=1.6x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-08-11 10:10:00 | 744.35 | 746.19 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:00:00 | 739.50 | 742.69 | 0.00 | ORB-short ORB[742.00,746.60] vol=2.1x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-08-12 11:10:00 | 740.62 | 742.57 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:50:00 | 769.80 | 766.10 | 0.00 | ORB-long ORB[758.45,768.00] vol=1.8x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:05:00 | 773.74 | 767.22 | 0.00 | T1 1.5R @ 773.74 |
| Target hit | 2025-08-18 15:20:00 | 780.00 | 776.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-08-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:40:00 | 796.85 | 793.59 | 0.00 | ORB-long ORB[788.10,794.50] vol=4.8x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 09:45:00 | 800.48 | 795.19 | 0.00 | T1 1.5R @ 800.48 |
| Stop hit — per-position SL triggered | 2025-08-21 09:55:00 | 796.85 | 795.75 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:35:00 | 806.00 | 803.51 | 0.00 | ORB-long ORB[797.00,804.40] vol=2.1x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-08-22 11:35:00 | 803.35 | 804.91 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:00:00 | 787.50 | 785.06 | 0.00 | ORB-long ORB[773.95,781.90] vol=13.0x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-09-01 11:45:00 | 785.14 | 785.17 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 11:05:00 | 765.45 | 768.93 | 0.00 | ORB-short ORB[766.60,774.95] vol=1.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-09-08 11:10:00 | 766.85 | 766.61 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 11:00:00 | 779.10 | 773.66 | 0.00 | ORB-long ORB[766.95,774.00] vol=2.7x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 11:20:00 | 781.66 | 777.21 | 0.00 | T1 1.5R @ 781.66 |
| Target hit | 2025-09-09 15:20:00 | 781.00 | 778.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-09-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:45:00 | 791.00 | 786.94 | 0.00 | ORB-long ORB[780.50,788.00] vol=3.0x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-09-11 10:00:00 | 788.65 | 788.16 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:30:00 | 784.40 | 780.78 | 0.00 | ORB-long ORB[773.30,782.75] vol=4.3x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 09:35:00 | 787.88 | 782.15 | 0.00 | T1 1.5R @ 787.88 |
| Target hit | 2025-09-15 10:10:00 | 785.20 | 786.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2025-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:35:00 | 791.35 | 793.81 | 0.00 | ORB-short ORB[793.00,798.00] vol=1.7x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-09-17 09:40:00 | 792.82 | 793.74 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:45:00 | 792.55 | 790.42 | 0.00 | ORB-long ORB[788.30,792.00] vol=2.0x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-09-18 09:50:00 | 790.66 | 790.52 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:40:00 | 785.00 | 789.54 | 0.00 | ORB-short ORB[787.75,792.95] vol=1.8x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:35:00 | 782.35 | 788.24 | 0.00 | T1 1.5R @ 782.35 |
| Stop hit — per-position SL triggered | 2025-09-19 14:45:00 | 785.00 | 785.23 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:30:00 | 786.45 | 784.37 | 0.00 | ORB-long ORB[780.60,785.00] vol=2.8x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-09-22 09:35:00 | 784.53 | 784.46 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 09:35:00 | 770.95 | 772.34 | 0.00 | ORB-short ORB[771.00,774.45] vol=1.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-09-25 09:55:00 | 772.59 | 771.68 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 09:35:00 | 745.55 | 748.80 | 0.00 | ORB-short ORB[746.65,757.05] vol=2.2x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:05:00 | 741.74 | 746.02 | 0.00 | T1 1.5R @ 741.74 |
| Target hit | 2025-09-29 11:05:00 | 743.00 | 742.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — BUY (started 2025-10-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:20:00 | 760.85 | 758.76 | 0.00 | ORB-long ORB[752.00,760.35] vol=3.8x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:25:00 | 763.73 | 761.21 | 0.00 | T1 1.5R @ 763.73 |
| Target hit | 2025-10-01 11:10:00 | 768.65 | 770.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2025-10-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:50:00 | 759.25 | 765.54 | 0.00 | ORB-short ORB[766.45,771.35] vol=1.7x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 13:00:00 | 756.42 | 762.01 | 0.00 | T1 1.5R @ 756.42 |
| Target hit | 2025-10-03 15:20:00 | 757.00 | 760.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2025-10-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:10:00 | 746.00 | 750.59 | 0.00 | ORB-short ORB[750.20,758.30] vol=2.0x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 747.58 | 750.48 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:05:00 | 734.70 | 737.66 | 0.00 | ORB-short ORB[738.00,740.60] vol=2.2x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:15:00 | 733.05 | 736.51 | 0.00 | T1 1.5R @ 733.05 |
| Target hit | 2025-10-09 14:40:00 | 733.05 | 732.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — BUY (started 2025-10-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:45:00 | 736.20 | 735.32 | 0.00 | ORB-long ORB[731.05,736.00] vol=1.5x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 11:20:00 | 737.94 | 735.65 | 0.00 | T1 1.5R @ 737.94 |
| Stop hit — per-position SL triggered | 2025-10-10 14:40:00 | 736.20 | 736.70 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 11:00:00 | 729.85 | 732.25 | 0.00 | ORB-short ORB[733.30,742.35] vol=1.5x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 731.07 | 732.00 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:15:00 | 742.90 | 740.56 | 0.00 | ORB-long ORB[737.40,742.85] vol=2.4x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 11:00:00 | 744.83 | 741.90 | 0.00 | T1 1.5R @ 744.83 |
| Stop hit — per-position SL triggered | 2025-10-23 11:20:00 | 742.90 | 742.34 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:55:00 | 734.20 | 736.65 | 0.00 | ORB-short ORB[737.50,742.50] vol=1.7x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 13:35:00 | 732.87 | 735.20 | 0.00 | T1 1.5R @ 732.87 |
| Stop hit — per-position SL triggered | 2025-10-24 14:40:00 | 734.20 | 734.88 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:55:00 | 730.50 | 732.25 | 0.00 | ORB-short ORB[733.25,736.95] vol=2.1x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-10-28 11:25:00 | 731.46 | 732.03 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:00:00 | 731.80 | 733.66 | 0.00 | ORB-short ORB[733.00,737.10] vol=3.2x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-10-31 11:20:00 | 732.59 | 733.38 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 11:10:00 | 727.40 | 729.38 | 0.00 | ORB-short ORB[728.60,731.35] vol=2.2x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:15:00 | 726.37 | 729.06 | 0.00 | T1 1.5R @ 726.37 |
| Target hit | 2025-11-03 15:20:00 | 723.15 | 725.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-11-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:30:00 | 728.90 | 727.04 | 0.00 | ORB-long ORB[723.35,727.95] vol=1.8x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:35:00 | 730.95 | 728.03 | 0.00 | T1 1.5R @ 730.95 |
| Target hit | 2025-11-04 10:10:00 | 729.70 | 730.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 11:15:00 | 714.05 | 717.10 | 0.00 | ORB-short ORB[714.25,723.55] vol=2.9x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:30:00 | 711.70 | 716.31 | 0.00 | T1 1.5R @ 711.70 |
| Target hit | 2025-11-07 15:05:00 | 712.30 | 711.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — BUY (started 2025-11-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:45:00 | 731.60 | 730.61 | 0.00 | ORB-long ORB[727.00,731.05] vol=2.8x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:00:00 | 733.88 | 731.11 | 0.00 | T1 1.5R @ 733.88 |
| Stop hit — per-position SL triggered | 2025-11-11 10:45:00 | 731.60 | 731.52 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:55:00 | 736.80 | 732.04 | 0.00 | ORB-long ORB[726.20,733.25] vol=6.2x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-11-14 11:20:00 | 735.56 | 733.71 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:40:00 | 735.45 | 732.77 | 0.00 | ORB-long ORB[728.20,732.40] vol=9.7x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 09:45:00 | 738.18 | 735.79 | 0.00 | T1 1.5R @ 738.18 |
| Target hit | 2025-11-17 10:30:00 | 737.65 | 738.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2025-11-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 09:55:00 | 745.50 | 743.37 | 0.00 | ORB-long ORB[740.60,745.45] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-11-19 10:05:00 | 743.71 | 743.46 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:50:00 | 739.60 | 740.03 | 0.00 | ORB-short ORB[740.00,743.45] vol=1.8x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 09:55:00 | 737.52 | 739.84 | 0.00 | T1 1.5R @ 737.52 |
| Target hit | 2025-11-21 12:00:00 | 738.90 | 738.53 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — BUY (started 2025-11-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:05:00 | 741.75 | 740.05 | 0.00 | ORB-long ORB[735.55,740.90] vol=1.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-11-24 15:20:00 | 741.50 | 741.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2025-11-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:30:00 | 764.65 | 760.26 | 0.00 | ORB-long ORB[753.90,761.30] vol=2.4x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-11-26 10:50:00 | 763.02 | 761.60 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:00:00 | 759.70 | 757.59 | 0.00 | ORB-long ORB[754.00,759.00] vol=2.0x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:15:00 | 762.29 | 759.64 | 0.00 | T1 1.5R @ 762.29 |
| Stop hit — per-position SL triggered | 2025-11-27 11:50:00 | 759.70 | 760.45 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:55:00 | 751.00 | 753.58 | 0.00 | ORB-short ORB[754.20,760.35] vol=1.8x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:05:00 | 748.82 | 752.99 | 0.00 | T1 1.5R @ 748.82 |
| Target hit | 2025-12-02 15:20:00 | 744.25 | 746.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 740.15 | 742.59 | 0.00 | ORB-short ORB[741.05,746.80] vol=1.7x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:35:00 | 738.28 | 741.53 | 0.00 | T1 1.5R @ 738.28 |
| Stop hit — per-position SL triggered | 2025-12-03 09:40:00 | 740.15 | 740.72 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:25:00 | 744.40 | 742.19 | 0.00 | ORB-long ORB[734.45,740.65] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-12-04 10:50:00 | 743.25 | 742.48 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:15:00 | 750.35 | 745.91 | 0.00 | ORB-long ORB[743.00,747.55] vol=2.7x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-12-05 10:30:00 | 748.57 | 746.94 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:45:00 | 760.00 | 761.16 | 0.00 | ORB-short ORB[761.00,769.25] vol=3.8x ATR=1.31 |
| Stop hit — per-position SL triggered | 2026-01-07 11:10:00 | 761.31 | 761.10 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 759.60 | 762.55 | 0.00 | ORB-short ORB[760.55,765.50] vol=2.3x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 757.32 | 762.33 | 0.00 | T1 1.5R @ 757.32 |
| Target hit | 2026-01-08 13:50:00 | 756.95 | 756.60 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — BUY (started 2026-01-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 10:10:00 | 753.45 | 744.39 | 0.00 | ORB-long ORB[737.25,747.90] vol=1.5x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 757.02 | 745.95 | 0.00 | T1 1.5R @ 757.02 |
| Stop hit — per-position SL triggered | 2026-01-12 10:45:00 | 753.45 | 748.79 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:15:00 | 759.80 | 763.05 | 0.00 | ORB-short ORB[761.45,768.95] vol=1.6x ATR=2.06 |
| Stop hit — per-position SL triggered | 2026-01-14 10:25:00 | 761.86 | 762.88 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-01-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:55:00 | 712.10 | 709.61 | 0.00 | ORB-long ORB[705.60,711.45] vol=2.1x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-01-30 10:05:00 | 710.38 | 709.91 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:15:00 | 713.00 | 710.24 | 0.00 | ORB-long ORB[705.10,708.95] vol=3.6x ATR=1.38 |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 711.62 | 711.31 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-02-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 11:05:00 | 709.25 | 712.90 | 0.00 | ORB-short ORB[714.10,717.95] vol=2.1x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-02-04 11:35:00 | 710.64 | 711.97 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:40:00 | 706.20 | 703.32 | 0.00 | ORB-long ORB[700.55,704.10] vol=5.8x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 09:45:00 | 708.71 | 703.35 | 0.00 | T1 1.5R @ 708.71 |
| Stop hit — per-position SL triggered | 2026-02-06 09:50:00 | 706.20 | 703.36 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-02-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:50:00 | 720.65 | 721.04 | 0.00 | ORB-short ORB[721.40,724.00] vol=9.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2026-02-11 10:35:00 | 722.14 | 721.02 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 706.55 | 711.45 | 0.00 | ORB-short ORB[708.90,717.50] vol=1.8x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 708.15 | 710.52 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-02-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:10:00 | 714.75 | 710.99 | 0.00 | ORB-long ORB[707.30,711.35] vol=1.5x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:15:00 | 717.00 | 711.14 | 0.00 | T1 1.5R @ 717.00 |
| Stop hit — per-position SL triggered | 2026-02-17 11:40:00 | 714.75 | 712.36 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:15:00 | 724.90 | 721.82 | 0.00 | ORB-long ORB[718.95,724.00] vol=1.7x ATR=1.95 |
| Stop hit — per-position SL triggered | 2026-02-18 10:25:00 | 722.95 | 721.91 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 728.05 | 725.31 | 0.00 | ORB-long ORB[720.95,725.80] vol=1.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:30:00 | 730.65 | 726.76 | 0.00 | T1 1.5R @ 730.65 |
| Stop hit — per-position SL triggered | 2026-02-20 13:10:00 | 728.05 | 727.95 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:05:00 | 722.85 | 720.80 | 0.00 | ORB-long ORB[715.00,722.05] vol=2.1x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-02-24 10:25:00 | 721.08 | 720.93 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-02-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:10:00 | 724.30 | 721.42 | 0.00 | ORB-long ORB[716.80,720.50] vol=2.4x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-02-25 10:35:00 | 722.86 | 721.83 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:45:00 | 719.15 | 721.99 | 0.00 | ORB-short ORB[721.90,725.80] vol=1.5x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:55:00 | 717.25 | 721.11 | 0.00 | T1 1.5R @ 717.25 |
| Target hit | 2026-02-26 15:20:00 | 713.65 | 717.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2026-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:10:00 | 714.35 | 710.29 | 0.00 | ORB-long ORB[708.00,713.60] vol=3.2x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 716.91 | 710.88 | 0.00 | T1 1.5R @ 716.91 |
| Stop hit — per-position SL triggered | 2026-02-27 10:45:00 | 714.35 | 712.60 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-03-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:50:00 | 678.50 | 684.61 | 0.00 | ORB-short ORB[684.40,691.00] vol=2.3x ATR=2.54 |
| Stop hit — per-position SL triggered | 2026-03-04 10:05:00 | 681.04 | 682.76 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-03-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:50:00 | 671.95 | 674.37 | 0.00 | ORB-short ORB[675.35,683.85] vol=10.2x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 673.56 | 674.35 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 11:15:00 | 634.50 | 632.73 | 0.00 | ORB-long ORB[627.15,634.45] vol=3.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 11:50:00 | 636.43 | 633.02 | 0.00 | T1 1.5R @ 636.43 |
| Target hit | 2026-04-15 15:20:00 | 643.40 | 637.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2026-04-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:10:00 | 676.40 | 673.02 | 0.00 | ORB-long ORB[668.85,675.55] vol=3.1x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:35:00 | 679.08 | 674.55 | 0.00 | T1 1.5R @ 679.08 |
| Target hit | 2026-04-21 15:20:00 | 683.20 | 681.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 655.65 | 658.65 | 0.00 | ORB-short ORB[656.05,664.90] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-04-24 10:00:00 | 657.76 | 657.71 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-04-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:20:00 | 679.85 | 676.48 | 0.00 | ORB-long ORB[669.05,677.50] vol=4.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-04-28 10:35:00 | 678.11 | 676.75 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:15:00 | 661.35 | 663.92 | 0.00 | ORB-short ORB[662.30,666.00] vol=1.6x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:50:00 | 658.79 | 662.34 | 0.00 | T1 1.5R @ 658.79 |
| Target hit | 2026-04-30 13:45:00 | 658.60 | 658.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 82 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 695.35 | 690.90 | 0.00 | ORB-long ORB[683.25,690.70] vol=2.2x ATR=3.10 |
| Stop hit — per-position SL triggered | 2026-05-07 09:50:00 | 692.25 | 692.08 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 703.85 | 701.42 | 0.00 | ORB-long ORB[696.00,703.65] vol=2.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-05-08 13:35:00 | 701.57 | 702.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-07-11 10:25:00 | 841.00 | 2025-07-11 10:40:00 | 842.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-14 10:45:00 | 841.00 | 2025-07-14 11:20:00 | 838.81 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-07-14 10:45:00 | 841.00 | 2025-07-14 13:45:00 | 839.80 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-07-17 11:05:00 | 813.85 | 2025-07-17 12:10:00 | 812.04 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-17 11:05:00 | 813.85 | 2025-07-17 13:25:00 | 813.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-21 09:30:00 | 793.05 | 2025-07-21 09:35:00 | 794.55 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-22 09:40:00 | 808.00 | 2025-07-22 09:50:00 | 806.10 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-23 10:05:00 | 799.25 | 2025-07-23 15:10:00 | 800.95 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-24 10:50:00 | 794.35 | 2025-07-24 10:55:00 | 792.96 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-07-24 10:50:00 | 794.35 | 2025-07-24 15:20:00 | 784.50 | TARGET_HIT | 0.50 | 1.24% |
| SELL | retest1 | 2025-07-25 09:30:00 | 768.30 | 2025-07-25 09:50:00 | 764.66 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-25 09:30:00 | 768.30 | 2025-07-25 13:35:00 | 760.00 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2025-07-28 10:35:00 | 749.65 | 2025-07-28 12:25:00 | 746.85 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-28 10:35:00 | 749.65 | 2025-07-28 13:00:00 | 749.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-29 10:25:00 | 742.55 | 2025-07-29 11:10:00 | 743.89 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-30 10:30:00 | 771.90 | 2025-07-30 10:35:00 | 777.41 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-07-30 10:30:00 | 771.90 | 2025-07-30 12:30:00 | 774.10 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-08-04 10:35:00 | 742.90 | 2025-08-04 10:50:00 | 740.45 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-08-04 10:35:00 | 742.90 | 2025-08-04 13:05:00 | 741.75 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-08-05 10:30:00 | 739.95 | 2025-08-05 10:50:00 | 741.46 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-06 09:50:00 | 735.70 | 2025-08-06 10:25:00 | 737.37 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-08-07 09:35:00 | 744.45 | 2025-08-07 09:50:00 | 742.59 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-08 09:40:00 | 759.00 | 2025-08-08 09:45:00 | 756.29 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-08-11 09:55:00 | 742.00 | 2025-08-11 10:10:00 | 744.35 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-12 11:00:00 | 739.50 | 2025-08-12 11:10:00 | 740.62 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-08-18 09:50:00 | 769.80 | 2025-08-18 10:05:00 | 773.74 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-08-18 09:50:00 | 769.80 | 2025-08-18 15:20:00 | 780.00 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2025-08-21 09:40:00 | 796.85 | 2025-08-21 09:45:00 | 800.48 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-21 09:40:00 | 796.85 | 2025-08-21 09:55:00 | 796.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 09:35:00 | 806.00 | 2025-08-22 11:35:00 | 803.35 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-09-01 11:00:00 | 787.50 | 2025-09-01 11:45:00 | 785.14 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-08 11:05:00 | 765.45 | 2025-09-08 11:10:00 | 766.85 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-09 11:00:00 | 779.10 | 2025-09-09 11:20:00 | 781.66 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-09 11:00:00 | 779.10 | 2025-09-09 15:20:00 | 781.00 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2025-09-11 09:45:00 | 791.00 | 2025-09-11 10:00:00 | 788.65 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-15 09:30:00 | 784.40 | 2025-09-15 09:35:00 | 787.88 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-09-15 09:30:00 | 784.40 | 2025-09-15 10:10:00 | 785.20 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-09-17 09:35:00 | 791.35 | 2025-09-17 09:40:00 | 792.82 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-18 09:45:00 | 792.55 | 2025-09-18 09:50:00 | 790.66 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-19 10:40:00 | 785.00 | 2025-09-19 11:35:00 | 782.35 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-19 10:40:00 | 785.00 | 2025-09-19 14:45:00 | 785.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 09:30:00 | 786.45 | 2025-09-22 09:35:00 | 784.53 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-25 09:35:00 | 770.95 | 2025-09-25 09:55:00 | 772.59 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-29 09:35:00 | 745.55 | 2025-09-29 10:05:00 | 741.74 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-09-29 09:35:00 | 745.55 | 2025-09-29 11:05:00 | 743.00 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-10-01 10:20:00 | 760.85 | 2025-10-01 10:25:00 | 763.73 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-01 10:20:00 | 760.85 | 2025-10-01 11:10:00 | 768.65 | TARGET_HIT | 0.50 | 1.03% |
| SELL | retest1 | 2025-10-03 10:50:00 | 759.25 | 2025-10-03 13:00:00 | 756.42 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-03 10:50:00 | 759.25 | 2025-10-03 15:20:00 | 757.00 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-06 10:10:00 | 746.00 | 2025-10-06 10:15:00 | 747.58 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-09 10:05:00 | 734.70 | 2025-10-09 10:15:00 | 733.05 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-10-09 10:05:00 | 734.70 | 2025-10-09 14:40:00 | 733.05 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-10-10 10:45:00 | 736.20 | 2025-10-10 11:20:00 | 737.94 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-10-10 10:45:00 | 736.20 | 2025-10-10 14:40:00 | 736.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 11:00:00 | 729.85 | 2025-10-17 11:15:00 | 731.07 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-23 10:15:00 | 742.90 | 2025-10-23 11:00:00 | 744.83 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-10-23 10:15:00 | 742.90 | 2025-10-23 11:20:00 | 742.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-24 10:55:00 | 734.20 | 2025-10-24 13:35:00 | 732.87 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-10-24 10:55:00 | 734.20 | 2025-10-24 14:40:00 | 734.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 10:55:00 | 730.50 | 2025-10-28 11:25:00 | 731.46 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-10-31 11:00:00 | 731.80 | 2025-10-31 11:20:00 | 732.59 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2025-11-03 11:10:00 | 727.40 | 2025-11-03 11:15:00 | 726.37 | PARTIAL | 0.50 | 0.14% |
| SELL | retest1 | 2025-11-03 11:10:00 | 727.40 | 2025-11-03 15:20:00 | 723.15 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-11-04 09:30:00 | 728.90 | 2025-11-04 09:35:00 | 730.95 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-11-04 09:30:00 | 728.90 | 2025-11-04 10:10:00 | 729.70 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-11-07 11:15:00 | 714.05 | 2025-11-07 11:30:00 | 711.70 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-07 11:15:00 | 714.05 | 2025-11-07 15:05:00 | 712.30 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-11-11 09:45:00 | 731.60 | 2025-11-11 10:00:00 | 733.88 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-11-11 09:45:00 | 731.60 | 2025-11-11 10:45:00 | 731.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 10:55:00 | 736.80 | 2025-11-14 11:20:00 | 735.56 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-17 09:40:00 | 735.45 | 2025-11-17 09:45:00 | 738.18 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-11-17 09:40:00 | 735.45 | 2025-11-17 10:30:00 | 737.65 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-11-19 09:55:00 | 745.50 | 2025-11-19 10:05:00 | 743.71 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-21 09:50:00 | 739.60 | 2025-11-21 09:55:00 | 737.52 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-21 09:50:00 | 739.60 | 2025-11-21 12:00:00 | 738.90 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-11-24 10:05:00 | 741.75 | 2025-11-24 15:20:00 | 741.50 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest1 | 2025-11-26 10:30:00 | 764.65 | 2025-11-26 10:50:00 | 763.02 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-27 10:00:00 | 759.70 | 2025-11-27 11:15:00 | 762.29 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-11-27 10:00:00 | 759.70 | 2025-11-27 11:50:00 | 759.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-02 10:55:00 | 751.00 | 2025-12-02 11:05:00 | 748.82 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-02 10:55:00 | 751.00 | 2025-12-02 15:20:00 | 744.25 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2025-12-03 09:30:00 | 740.15 | 2025-12-03 09:35:00 | 738.28 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-03 09:30:00 | 740.15 | 2025-12-03 09:40:00 | 740.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-04 10:25:00 | 744.40 | 2025-12-04 10:50:00 | 743.25 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-12-05 10:15:00 | 750.35 | 2025-12-05 10:30:00 | 748.57 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-07 10:45:00 | 760.00 | 2026-01-07 11:10:00 | 761.31 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-01-08 11:10:00 | 759.60 | 2026-01-08 11:15:00 | 757.32 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-01-08 11:10:00 | 759.60 | 2026-01-08 13:50:00 | 756.95 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2026-01-12 10:10:00 | 753.45 | 2026-01-12 10:15:00 | 757.02 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-01-12 10:10:00 | 753.45 | 2026-01-12 10:45:00 | 753.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-14 10:15:00 | 759.80 | 2026-01-14 10:25:00 | 761.86 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-30 09:55:00 | 712.10 | 2026-01-30 10:05:00 | 710.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-01 11:15:00 | 713.00 | 2026-02-01 12:15:00 | 711.62 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-04 11:05:00 | 709.25 | 2026-02-04 11:35:00 | 710.64 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-02-06 09:40:00 | 706.20 | 2026-02-06 09:45:00 | 708.71 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-06 09:40:00 | 706.20 | 2026-02-06 09:50:00 | 706.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 09:50:00 | 720.65 | 2026-02-11 10:35:00 | 722.14 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-13 09:30:00 | 706.55 | 2026-02-13 09:40:00 | 708.15 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-17 11:10:00 | 714.75 | 2026-02-17 11:15:00 | 717.00 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2026-02-17 11:10:00 | 714.75 | 2026-02-17 11:40:00 | 714.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 10:15:00 | 724.90 | 2026-02-18 10:25:00 | 722.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-20 10:40:00 | 728.05 | 2026-02-20 11:30:00 | 730.65 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-20 10:40:00 | 728.05 | 2026-02-20 13:10:00 | 728.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 10:05:00 | 722.85 | 2026-02-24 10:25:00 | 721.08 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 10:10:00 | 724.30 | 2026-02-25 10:35:00 | 722.86 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-26 10:45:00 | 719.15 | 2026-02-26 10:55:00 | 717.25 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-26 10:45:00 | 719.15 | 2026-02-26 15:20:00 | 713.65 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2026-02-27 10:10:00 | 714.35 | 2026-02-27 10:15:00 | 716.91 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-27 10:10:00 | 714.35 | 2026-02-27 10:45:00 | 714.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-04 09:50:00 | 678.50 | 2026-03-04 10:05:00 | 681.04 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-06 10:50:00 | 671.95 | 2026-03-06 11:00:00 | 673.56 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-15 11:15:00 | 634.50 | 2026-04-15 11:50:00 | 636.43 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-04-15 11:15:00 | 634.50 | 2026-04-15 15:20:00 | 643.40 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2026-04-21 11:10:00 | 676.40 | 2026-04-21 11:35:00 | 679.08 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-21 11:10:00 | 676.40 | 2026-04-21 15:20:00 | 683.20 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-04-24 09:35:00 | 655.65 | 2026-04-24 10:00:00 | 657.76 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-28 10:20:00 | 679.85 | 2026-04-28 10:35:00 | 678.11 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-30 10:15:00 | 661.35 | 2026-04-30 10:50:00 | 658.79 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-04-30 10:15:00 | 661.35 | 2026-04-30 13:45:00 | 658.60 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2026-05-07 09:35:00 | 695.35 | 2026-05-07 09:50:00 | 692.25 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-08 09:50:00 | 703.85 | 2026-05-08 13:35:00 | 701.57 | STOP_HIT | 1.00 | -0.32% |
