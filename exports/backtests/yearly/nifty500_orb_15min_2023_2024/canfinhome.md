# Can Fin Homes Ltd. (CANFINHOME)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (53844 bars)
- **Last close:** 878.10
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
| ENTRY1 | 109 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 16 |
| STOP_HIT | 93 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 144 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 93
- **Target hits / Stop hits / Partials:** 16 / 93 / 35
- **Avg / median % per leg:** 0.04% / -0.22%
- **Sum % (uncompounded):** 6.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 21 | 29.2% | 5 | 51 | 16 | -0.00% | -0.1% |
| BUY @ 2nd Alert (retest1) | 72 | 21 | 29.2% | 5 | 51 | 16 | -0.00% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 72 | 30 | 41.7% | 11 | 42 | 19 | 0.09% | 6.4% |
| SELL @ 2nd Alert (retest1) | 72 | 30 | 41.7% | 11 | 42 | 19 | 0.09% | 6.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 144 | 51 | 35.4% | 16 | 93 | 35 | 0.04% | 6.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:45:00 | 654.00 | 648.46 | 0.00 | ORB-long ORB[640.45,648.00] vol=2.9x ATR=2.26 |
| Stop hit — per-position SL triggered | 2023-05-15 09:55:00 | 651.74 | 649.53 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:50:00 | 680.30 | 676.78 | 0.00 | ORB-long ORB[671.05,679.00] vol=2.8x ATR=2.43 |
| Stop hit — per-position SL triggered | 2023-05-23 09:55:00 | 677.87 | 677.11 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 09:30:00 | 674.95 | 677.85 | 0.00 | ORB-short ORB[675.05,683.20] vol=1.7x ATR=1.93 |
| Stop hit — per-position SL triggered | 2023-05-25 09:35:00 | 676.88 | 677.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-26 10:45:00 | 685.00 | 681.01 | 0.00 | ORB-long ORB[673.00,682.05] vol=1.5x ATR=1.95 |
| Stop hit — per-position SL triggered | 2023-05-26 10:55:00 | 683.05 | 681.17 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:40:00 | 687.10 | 682.20 | 0.00 | ORB-long ORB[678.10,683.55] vol=3.1x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 09:50:00 | 690.40 | 686.13 | 0.00 | T1 1.5R @ 690.40 |
| Stop hit — per-position SL triggered | 2023-05-29 09:55:00 | 687.10 | 686.27 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 09:45:00 | 694.05 | 690.81 | 0.00 | ORB-long ORB[686.35,692.75] vol=2.1x ATR=2.07 |
| Stop hit — per-position SL triggered | 2023-05-30 09:55:00 | 691.98 | 691.07 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 11:05:00 | 692.25 | 688.63 | 0.00 | ORB-long ORB[685.90,690.55] vol=2.4x ATR=1.52 |
| Stop hit — per-position SL triggered | 2023-05-31 11:55:00 | 690.73 | 689.95 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-01 10:50:00 | 720.05 | 713.01 | 0.00 | ORB-long ORB[705.00,711.85] vol=2.5x ATR=2.45 |
| Stop hit — per-position SL triggered | 2023-06-01 12:20:00 | 717.60 | 714.89 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 10:55:00 | 719.20 | 721.75 | 0.00 | ORB-short ORB[722.75,729.05] vol=3.4x ATR=2.38 |
| Stop hit — per-position SL triggered | 2023-06-02 11:30:00 | 721.58 | 721.35 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 10:00:00 | 718.75 | 725.42 | 0.00 | ORB-short ORB[725.60,731.50] vol=3.4x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-06-05 10:05:00 | 721.30 | 724.82 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:30:00 | 717.30 | 713.31 | 0.00 | ORB-long ORB[707.05,715.00] vol=3.9x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-06-06 09:35:00 | 714.96 | 713.96 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 11:05:00 | 745.60 | 748.43 | 0.00 | ORB-short ORB[747.25,752.25] vol=1.6x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-06-08 11:10:00 | 747.41 | 748.39 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 09:40:00 | 743.60 | 739.40 | 0.00 | ORB-long ORB[735.70,742.35] vol=1.6x ATR=2.47 |
| Stop hit — per-position SL triggered | 2023-06-09 10:20:00 | 741.13 | 741.31 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:25:00 | 745.35 | 742.69 | 0.00 | ORB-long ORB[735.05,744.70] vol=9.6x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 12:00:00 | 749.16 | 743.70 | 0.00 | T1 1.5R @ 749.16 |
| Stop hit — per-position SL triggered | 2023-06-12 14:20:00 | 745.35 | 747.54 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:15:00 | 742.15 | 745.24 | 0.00 | ORB-short ORB[747.05,754.45] vol=2.5x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-06-14 12:00:00 | 743.74 | 744.73 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-06-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 10:25:00 | 755.10 | 751.61 | 0.00 | ORB-long ORB[745.35,752.80] vol=2.6x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 10:45:00 | 757.89 | 753.13 | 0.00 | T1 1.5R @ 757.89 |
| Stop hit — per-position SL triggered | 2023-06-15 11:05:00 | 755.10 | 753.73 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-06-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 10:20:00 | 764.65 | 759.36 | 0.00 | ORB-long ORB[752.45,762.00] vol=2.6x ATR=2.63 |
| Stop hit — per-position SL triggered | 2023-06-19 10:25:00 | 762.02 | 759.56 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:15:00 | 748.70 | 753.31 | 0.00 | ORB-short ORB[752.00,757.20] vol=1.6x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 14:15:00 | 745.29 | 749.81 | 0.00 | T1 1.5R @ 745.29 |
| Target hit | 2023-06-21 15:20:00 | 745.30 | 748.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2023-06-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:30:00 | 755.50 | 751.44 | 0.00 | ORB-long ORB[745.65,754.40] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2023-06-22 09:35:00 | 753.39 | 751.72 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:40:00 | 755.80 | 751.16 | 0.00 | ORB-long ORB[743.00,752.85] vol=1.5x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 09:50:00 | 759.59 | 753.35 | 0.00 | T1 1.5R @ 759.59 |
| Target hit | 2023-06-27 15:20:00 | 783.60 | 771.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2023-06-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 09:40:00 | 783.25 | 791.63 | 0.00 | ORB-short ORB[790.00,799.50] vol=2.1x ATR=3.09 |
| Stop hit — per-position SL triggered | 2023-06-28 10:35:00 | 786.34 | 788.87 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 10:45:00 | 781.80 | 793.12 | 0.00 | ORB-short ORB[794.00,804.00] vol=1.6x ATR=2.80 |
| Stop hit — per-position SL triggered | 2023-07-05 11:15:00 | 784.60 | 790.64 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 10:00:00 | 784.25 | 780.67 | 0.00 | ORB-long ORB[773.85,781.15] vol=2.0x ATR=3.22 |
| Stop hit — per-position SL triggered | 2023-07-10 10:15:00 | 781.03 | 781.64 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:40:00 | 806.50 | 803.05 | 0.00 | ORB-long ORB[791.05,798.00] vol=8.5x ATR=3.76 |
| Stop hit — per-position SL triggered | 2023-07-14 10:00:00 | 802.74 | 803.90 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 11:10:00 | 801.00 | 807.69 | 0.00 | ORB-short ORB[807.55,814.10] vol=5.2x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:35:00 | 797.72 | 805.70 | 0.00 | T1 1.5R @ 797.72 |
| Target hit | 2023-07-18 15:20:00 | 798.20 | 800.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2023-08-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:10:00 | 743.00 | 747.75 | 0.00 | ORB-short ORB[746.00,753.80] vol=2.0x ATR=2.46 |
| Stop hit — per-position SL triggered | 2023-08-01 10:15:00 | 745.46 | 747.55 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 11:00:00 | 730.80 | 738.02 | 0.00 | ORB-short ORB[735.05,743.40] vol=2.0x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-08-02 11:15:00 | 733.14 | 736.59 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:45:00 | 739.35 | 743.70 | 0.00 | ORB-short ORB[740.00,749.50] vol=2.5x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 11:15:00 | 736.36 | 741.94 | 0.00 | T1 1.5R @ 736.36 |
| Stop hit — per-position SL triggered | 2023-08-08 12:00:00 | 739.35 | 741.15 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:40:00 | 733.85 | 736.09 | 0.00 | ORB-short ORB[735.45,742.00] vol=2.2x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 11:20:00 | 731.53 | 735.28 | 0.00 | T1 1.5R @ 731.53 |
| Stop hit — per-position SL triggered | 2023-08-09 11:35:00 | 733.85 | 735.15 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:25:00 | 723.80 | 728.49 | 0.00 | ORB-short ORB[727.45,736.95] vol=4.3x ATR=2.43 |
| Stop hit — per-position SL triggered | 2023-08-10 10:30:00 | 726.23 | 725.44 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-08-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 10:00:00 | 730.65 | 733.78 | 0.00 | ORB-short ORB[731.30,737.95] vol=1.5x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 10:05:00 | 726.86 | 733.25 | 0.00 | T1 1.5R @ 726.86 |
| Stop hit — per-position SL triggered | 2023-08-18 10:15:00 | 730.65 | 732.29 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-21 09:30:00 | 735.05 | 732.26 | 0.00 | ORB-long ORB[727.00,734.00] vol=1.7x ATR=2.32 |
| Stop hit — per-position SL triggered | 2023-08-21 09:45:00 | 732.73 | 732.67 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 10:25:00 | 749.30 | 745.62 | 0.00 | ORB-long ORB[740.00,747.00] vol=3.9x ATR=2.00 |
| Stop hit — per-position SL triggered | 2023-08-22 10:35:00 | 747.30 | 745.83 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 10:40:00 | 754.00 | 751.34 | 0.00 | ORB-long ORB[748.30,753.70] vol=2.0x ATR=1.92 |
| Stop hit — per-position SL triggered | 2023-08-30 11:00:00 | 752.08 | 751.61 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 09:55:00 | 754.00 | 751.44 | 0.00 | ORB-long ORB[746.30,753.20] vol=3.0x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 12:00:00 | 756.83 | 753.56 | 0.00 | T1 1.5R @ 756.83 |
| Stop hit — per-position SL triggered | 2023-08-31 12:55:00 | 754.00 | 753.86 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 09:30:00 | 760.80 | 763.53 | 0.00 | ORB-short ORB[761.00,769.50] vol=2.7x ATR=2.39 |
| Stop hit — per-position SL triggered | 2023-09-04 09:45:00 | 763.19 | 762.97 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 09:40:00 | 782.10 | 778.48 | 0.00 | ORB-long ORB[770.40,779.45] vol=1.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2023-09-08 09:55:00 | 779.97 | 779.70 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 09:35:00 | 782.95 | 778.18 | 0.00 | ORB-long ORB[774.70,780.70] vol=2.4x ATR=2.43 |
| Stop hit — per-position SL triggered | 2023-09-11 09:40:00 | 780.52 | 778.90 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 10:55:00 | 790.75 | 785.78 | 0.00 | ORB-long ORB[780.55,788.90] vol=4.9x ATR=2.50 |
| Stop hit — per-position SL triggered | 2023-09-15 11:40:00 | 788.25 | 787.54 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-09-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 10:45:00 | 768.25 | 775.96 | 0.00 | ORB-short ORB[771.30,781.20] vol=2.4x ATR=2.47 |
| Stop hit — per-position SL triggered | 2023-09-21 10:50:00 | 770.72 | 775.75 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:50:00 | 758.35 | 762.46 | 0.00 | ORB-short ORB[759.00,767.45] vol=1.5x ATR=3.37 |
| Stop hit — per-position SL triggered | 2023-09-22 10:10:00 | 761.72 | 761.55 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:45:00 | 769.00 | 765.78 | 0.00 | ORB-long ORB[758.00,766.15] vol=1.6x ATR=2.23 |
| Stop hit — per-position SL triggered | 2023-09-27 10:00:00 | 766.77 | 766.21 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 09:45:00 | 766.25 | 766.59 | 0.00 | ORB-short ORB[768.15,773.95] vol=1.9x ATR=1.96 |
| Stop hit — per-position SL triggered | 2023-09-28 10:00:00 | 768.21 | 766.59 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-09-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:10:00 | 765.65 | 760.39 | 0.00 | ORB-long ORB[753.00,758.40] vol=1.5x ATR=2.60 |
| Stop hit — per-position SL triggered | 2023-09-29 10:50:00 | 763.05 | 761.76 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-10-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 09:40:00 | 767.75 | 763.77 | 0.00 | ORB-long ORB[756.20,766.00] vol=1.9x ATR=2.57 |
| Stop hit — per-position SL triggered | 2023-10-05 09:45:00 | 765.18 | 763.99 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 10:50:00 | 771.45 | 765.87 | 0.00 | ORB-long ORB[759.10,766.65] vol=6.3x ATR=2.29 |
| Stop hit — per-position SL triggered | 2023-10-06 11:00:00 | 769.16 | 766.45 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 09:45:00 | 751.05 | 753.77 | 0.00 | ORB-short ORB[752.00,756.95] vol=2.2x ATR=1.62 |
| Stop hit — per-position SL triggered | 2023-10-13 09:50:00 | 752.67 | 753.69 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 09:45:00 | 739.50 | 741.68 | 0.00 | ORB-short ORB[739.70,746.85] vol=1.8x ATR=1.96 |
| Stop hit — per-position SL triggered | 2023-10-16 10:50:00 | 741.46 | 740.95 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-10-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 09:40:00 | 777.55 | 773.01 | 0.00 | ORB-long ORB[764.00,774.30] vol=4.3x ATR=3.08 |
| Stop hit — per-position SL triggered | 2023-10-20 09:45:00 | 774.47 | 773.87 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:10:00 | 759.00 | 763.93 | 0.00 | ORB-short ORB[759.35,767.00] vol=1.5x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:20:00 | 754.85 | 762.12 | 0.00 | T1 1.5R @ 754.85 |
| Target hit | 2023-10-23 11:55:00 | 756.80 | 756.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — BUY (started 2023-10-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 11:10:00 | 738.45 | 729.63 | 0.00 | ORB-long ORB[721.35,727.10] vol=1.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2023-10-27 13:35:00 | 736.20 | 734.83 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 09:30:00 | 768.00 | 766.02 | 0.00 | ORB-long ORB[760.80,766.15] vol=2.2x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 09:35:00 | 771.39 | 767.83 | 0.00 | T1 1.5R @ 771.39 |
| Target hit | 2023-11-01 10:50:00 | 770.95 | 771.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — BUY (started 2023-11-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:00:00 | 775.95 | 773.65 | 0.00 | ORB-long ORB[770.05,775.85] vol=3.1x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 10:20:00 | 779.23 | 776.55 | 0.00 | T1 1.5R @ 779.23 |
| Stop hit — per-position SL triggered | 2023-11-03 10:30:00 | 775.95 | 776.73 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:25:00 | 779.50 | 775.34 | 0.00 | ORB-long ORB[772.05,776.50] vol=4.3x ATR=1.99 |
| Stop hit — per-position SL triggered | 2023-11-06 10:35:00 | 777.51 | 775.71 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 09:45:00 | 771.20 | 773.47 | 0.00 | ORB-short ORB[772.55,776.50] vol=1.8x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 10:35:00 | 767.93 | 771.77 | 0.00 | T1 1.5R @ 767.93 |
| Target hit | 2023-11-07 15:20:00 | 755.45 | 762.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2023-12-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 09:40:00 | 796.20 | 799.70 | 0.00 | ORB-short ORB[798.45,803.65] vol=3.4x ATR=4.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 13:00:00 | 789.19 | 795.17 | 0.00 | T1 1.5R @ 789.19 |
| Target hit | 2023-12-08 15:20:00 | 790.30 | 791.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2023-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:45:00 | 802.00 | 799.32 | 0.00 | ORB-long ORB[792.10,799.50] vol=1.5x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-12-11 09:50:00 | 799.45 | 799.36 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:20:00 | 742.00 | 754.96 | 0.00 | ORB-short ORB[757.70,764.90] vol=2.1x ATR=5.38 |
| Stop hit — per-position SL triggered | 2023-12-12 12:35:00 | 747.38 | 746.93 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-12-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 09:40:00 | 787.05 | 784.45 | 0.00 | ORB-long ORB[779.80,786.55] vol=1.6x ATR=3.00 |
| Stop hit — per-position SL triggered | 2023-12-14 10:25:00 | 784.05 | 785.14 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-12-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 10:45:00 | 782.95 | 788.78 | 0.00 | ORB-short ORB[789.00,793.40] vol=2.9x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 10:50:00 | 779.64 | 787.72 | 0.00 | T1 1.5R @ 779.64 |
| Stop hit — per-position SL triggered | 2023-12-15 11:05:00 | 782.95 | 786.75 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-12-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:45:00 | 785.00 | 789.56 | 0.00 | ORB-short ORB[787.05,794.85] vol=1.9x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-12-19 10:55:00 | 787.51 | 787.91 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-12-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 10:45:00 | 757.50 | 761.38 | 0.00 | ORB-short ORB[758.65,768.10] vol=1.5x ATR=2.50 |
| Stop hit — per-position SL triggered | 2023-12-22 10:55:00 | 760.00 | 761.17 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:30:00 | 784.00 | 782.61 | 0.00 | ORB-long ORB[777.05,783.00] vol=4.4x ATR=2.60 |
| Stop hit — per-position SL triggered | 2023-12-27 09:40:00 | 781.40 | 782.62 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:35:00 | 785.10 | 782.89 | 0.00 | ORB-long ORB[779.45,783.95] vol=3.9x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 09:40:00 | 788.69 | 785.00 | 0.00 | T1 1.5R @ 788.69 |
| Stop hit — per-position SL triggered | 2023-12-28 09:50:00 | 785.10 | 785.34 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:40:00 | 779.75 | 778.65 | 0.00 | ORB-long ORB[775.00,779.00] vol=2.5x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-12-29 10:50:00 | 777.91 | 778.69 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-01-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:30:00 | 766.00 | 774.79 | 0.00 | ORB-short ORB[776.60,780.75] vol=1.7x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-01-02 12:20:00 | 768.38 | 771.55 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-01-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 11:00:00 | 770.20 | 765.42 | 0.00 | ORB-long ORB[763.10,769.60] vol=1.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-01-03 11:55:00 | 767.74 | 767.09 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-01-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:30:00 | 773.50 | 772.01 | 0.00 | ORB-long ORB[768.00,773.05] vol=1.8x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 09:40:00 | 776.81 | 773.65 | 0.00 | T1 1.5R @ 776.81 |
| Stop hit — per-position SL triggered | 2024-01-04 09:50:00 | 773.50 | 773.76 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-01-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:50:00 | 783.15 | 790.41 | 0.00 | ORB-short ORB[787.75,795.25] vol=1.6x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-01-05 11:05:00 | 785.43 | 789.36 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 10:55:00 | 769.60 | 775.16 | 0.00 | ORB-short ORB[775.10,782.55] vol=2.7x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-01-09 11:05:00 | 771.79 | 775.02 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-01-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:50:00 | 760.95 | 761.34 | 0.00 | ORB-short ORB[765.05,773.00] vol=11.8x ATR=2.73 |
| Stop hit — per-position SL triggered | 2024-01-15 10:05:00 | 763.68 | 761.50 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:35:00 | 787.60 | 783.33 | 0.00 | ORB-long ORB[779.40,784.80] vol=2.4x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-01-19 09:40:00 | 784.64 | 783.53 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 10:25:00 | 769.75 | 774.04 | 0.00 | ORB-short ORB[770.50,778.30] vol=1.8x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 11:20:00 | 765.80 | 772.38 | 0.00 | T1 1.5R @ 765.80 |
| Target hit | 2024-01-29 15:20:00 | 767.00 | 769.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2024-01-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:55:00 | 771.85 | 770.31 | 0.00 | ORB-long ORB[765.00,770.45] vol=1.7x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 11:30:00 | 775.02 | 771.18 | 0.00 | T1 1.5R @ 775.02 |
| Stop hit — per-position SL triggered | 2024-01-30 14:05:00 | 771.85 | 773.15 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-01-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 10:00:00 | 772.90 | 769.12 | 0.00 | ORB-long ORB[764.45,769.95] vol=2.1x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 10:45:00 | 776.87 | 771.98 | 0.00 | T1 1.5R @ 776.87 |
| Target hit | 2024-01-31 12:35:00 | 775.05 | 775.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 76 — BUY (started 2024-02-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:35:00 | 820.30 | 810.29 | 0.00 | ORB-long ORB[799.90,809.00] vol=4.7x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 10:15:00 | 827.05 | 819.98 | 0.00 | T1 1.5R @ 827.05 |
| Target hit | 2024-02-02 12:35:00 | 823.05 | 823.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 77 — SELL (started 2024-02-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 11:10:00 | 816.00 | 823.56 | 0.00 | ORB-short ORB[825.90,836.85] vol=2.0x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-02-05 11:20:00 | 819.13 | 823.30 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 11:15:00 | 793.85 | 786.27 | 0.00 | ORB-long ORB[775.65,787.40] vol=1.9x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-02-13 12:05:00 | 790.50 | 787.00 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-02-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 09:35:00 | 824.10 | 821.61 | 0.00 | ORB-long ORB[819.00,823.85] vol=2.6x ATR=2.59 |
| Stop hit — per-position SL triggered | 2024-02-16 09:40:00 | 821.51 | 821.69 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-02-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:20:00 | 795.55 | 800.70 | 0.00 | ORB-short ORB[797.10,806.20] vol=1.5x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-20 10:35:00 | 792.04 | 799.93 | 0.00 | T1 1.5R @ 792.04 |
| Stop hit — per-position SL triggered | 2024-02-20 11:30:00 | 795.55 | 798.22 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-02-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 11:00:00 | 809.60 | 802.84 | 0.00 | ORB-long ORB[797.95,804.15] vol=5.1x ATR=2.59 |
| Stop hit — per-position SL triggered | 2024-02-21 11:05:00 | 807.01 | 803.11 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 09:30:00 | 808.15 | 807.75 | 0.00 | ORB-long ORB[800.00,807.20] vol=11.4x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-02-23 09:40:00 | 805.81 | 807.65 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 09:45:00 | 805.95 | 801.33 | 0.00 | ORB-long ORB[795.55,805.10] vol=3.2x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-02-26 09:50:00 | 803.57 | 801.85 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 10:50:00 | 796.10 | 799.47 | 0.00 | ORB-short ORB[799.30,803.40] vol=1.8x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 11:35:00 | 793.80 | 798.12 | 0.00 | T1 1.5R @ 793.80 |
| Target hit | 2024-02-27 12:55:00 | 794.95 | 794.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 85 — SELL (started 2024-02-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:50:00 | 783.00 | 789.06 | 0.00 | ORB-short ORB[789.40,795.35] vol=4.6x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-02-28 10:55:00 | 785.09 | 788.49 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-03-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 10:40:00 | 789.00 | 783.99 | 0.00 | ORB-long ORB[774.45,785.35] vol=1.9x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-03-01 10:50:00 | 786.40 | 784.23 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 10:15:00 | 795.30 | 791.75 | 0.00 | ORB-long ORB[788.30,794.70] vol=1.7x ATR=2.59 |
| Stop hit — per-position SL triggered | 2024-03-04 10:55:00 | 792.71 | 792.76 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 10:40:00 | 788.50 | 790.20 | 0.00 | ORB-short ORB[789.00,795.00] vol=2.1x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-03-05 10:55:00 | 790.30 | 790.15 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-03-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:45:00 | 779.00 | 785.52 | 0.00 | ORB-short ORB[788.90,795.60] vol=3.9x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:55:00 | 774.55 | 780.75 | 0.00 | T1 1.5R @ 774.55 |
| Target hit | 2024-03-06 13:05:00 | 773.95 | 772.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 90 — SELL (started 2024-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 11:10:00 | 776.80 | 782.54 | 0.00 | ORB-short ORB[782.50,787.70] vol=2.9x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 11:25:00 | 774.01 | 781.08 | 0.00 | T1 1.5R @ 774.01 |
| Target hit | 2024-03-11 12:35:00 | 774.80 | 774.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 91 — SELL (started 2024-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 10:45:00 | 758.00 | 764.39 | 0.00 | ORB-short ORB[761.35,771.60] vol=3.4x ATR=2.61 |
| Stop hit — per-position SL triggered | 2024-03-12 10:55:00 | 760.61 | 764.06 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:55:00 | 748.25 | 753.16 | 0.00 | ORB-short ORB[753.10,761.20] vol=1.6x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-03-13 10:15:00 | 750.99 | 752.04 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-03-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 10:30:00 | 716.70 | 720.88 | 0.00 | ORB-short ORB[717.85,727.55] vol=1.5x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-03-18 10:35:00 | 719.44 | 720.55 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-03-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:55:00 | 705.40 | 712.31 | 0.00 | ORB-short ORB[712.25,722.00] vol=2.1x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 11:45:00 | 701.97 | 707.91 | 0.00 | T1 1.5R @ 701.97 |
| Stop hit — per-position SL triggered | 2024-03-19 11:50:00 | 705.40 | 707.84 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 09:30:00 | 732.65 | 730.22 | 0.00 | ORB-long ORB[723.35,732.35] vol=2.3x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 09:50:00 | 736.32 | 732.28 | 0.00 | T1 1.5R @ 736.32 |
| Target hit | 2024-03-21 11:40:00 | 734.30 | 735.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 96 — BUY (started 2024-03-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 09:35:00 | 749.95 | 746.94 | 0.00 | ORB-long ORB[736.25,742.60] vol=5.0x ATR=2.62 |
| Stop hit — per-position SL triggered | 2024-03-22 09:50:00 | 747.33 | 747.40 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-03-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 11:05:00 | 751.15 | 747.01 | 0.00 | ORB-long ORB[743.20,748.20] vol=2.2x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 12:00:00 | 754.11 | 748.61 | 0.00 | T1 1.5R @ 754.11 |
| Stop hit — per-position SL triggered | 2024-03-26 12:15:00 | 751.15 | 748.74 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-03-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 10:40:00 | 750.50 | 753.76 | 0.00 | ORB-short ORB[753.80,761.30] vol=1.9x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-03-28 10:55:00 | 752.48 | 753.51 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2024-04-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:50:00 | 797.00 | 793.32 | 0.00 | ORB-long ORB[789.00,795.65] vol=1.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-04-03 09:55:00 | 794.92 | 793.50 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-04-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:30:00 | 802.35 | 805.85 | 0.00 | ORB-short ORB[803.00,813.70] vol=2.7x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 09:40:00 | 798.51 | 803.53 | 0.00 | T1 1.5R @ 798.51 |
| Target hit | 2024-04-12 15:20:00 | 784.30 | 795.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 101 — SELL (started 2024-04-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 10:35:00 | 769.45 | 771.28 | 0.00 | ORB-short ORB[769.85,775.45] vol=3.7x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-04-18 11:15:00 | 771.47 | 770.82 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2024-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 09:40:00 | 750.70 | 754.24 | 0.00 | ORB-short ORB[752.45,759.45] vol=1.5x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-04-23 09:55:00 | 753.13 | 753.28 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2024-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 09:35:00 | 740.10 | 742.68 | 0.00 | ORB-short ORB[740.95,749.00] vol=1.8x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-04-25 09:55:00 | 743.81 | 742.12 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-04-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 09:50:00 | 745.50 | 747.91 | 0.00 | ORB-short ORB[746.35,752.00] vol=1.7x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-26 13:10:00 | 741.94 | 745.56 | 0.00 | T1 1.5R @ 741.94 |
| Stop hit — per-position SL triggered | 2024-04-26 14:25:00 | 745.50 | 745.27 | 0.00 | SL hit |

### Cycle 105 — SELL (started 2024-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 09:35:00 | 746.20 | 749.53 | 0.00 | ORB-short ORB[747.65,753.75] vol=2.2x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 09:50:00 | 742.53 | 747.78 | 0.00 | T1 1.5R @ 742.53 |
| Target hit | 2024-04-29 13:05:00 | 736.85 | 736.37 | 0.00 | Trail-exit close>VWAP |

### Cycle 106 — SELL (started 2024-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:35:00 | 749.40 | 753.29 | 0.00 | ORB-short ORB[752.00,759.90] vol=2.4x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-05-07 09:45:00 | 751.87 | 752.90 | 0.00 | SL hit |

### Cycle 107 — BUY (started 2024-05-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 09:55:00 | 743.00 | 739.11 | 0.00 | ORB-long ORB[732.40,741.60] vol=2.1x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 10:05:00 | 746.65 | 741.10 | 0.00 | T1 1.5R @ 746.65 |
| Stop hit — per-position SL triggered | 2024-05-08 10:20:00 | 743.00 | 741.31 | 0.00 | SL hit |

### Cycle 108 — SELL (started 2024-05-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:25:00 | 734.90 | 738.99 | 0.00 | ORB-short ORB[738.30,745.80] vol=1.6x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:45:00 | 732.05 | 737.69 | 0.00 | T1 1.5R @ 732.05 |
| Stop hit — per-position SL triggered | 2024-05-09 11:45:00 | 734.90 | 735.81 | 0.00 | SL hit |

### Cycle 109 — BUY (started 2024-05-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 09:40:00 | 729.30 | 723.31 | 0.00 | ORB-long ORB[718.20,726.45] vol=3.3x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-10 11:20:00 | 734.57 | 725.68 | 0.00 | T1 1.5R @ 734.57 |
| Stop hit — per-position SL triggered | 2024-05-10 12:10:00 | 729.30 | 726.42 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:45:00 | 654.00 | 2023-05-15 09:55:00 | 651.74 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-05-23 09:50:00 | 680.30 | 2023-05-23 09:55:00 | 677.87 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-05-25 09:30:00 | 674.95 | 2023-05-25 09:35:00 | 676.88 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-05-26 10:45:00 | 685.00 | 2023-05-26 10:55:00 | 683.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-05-29 09:40:00 | 687.10 | 2023-05-29 09:50:00 | 690.40 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-05-29 09:40:00 | 687.10 | 2023-05-29 09:55:00 | 687.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-30 09:45:00 | 694.05 | 2023-05-30 09:55:00 | 691.98 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-05-31 11:05:00 | 692.25 | 2023-05-31 11:55:00 | 690.73 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-01 10:50:00 | 720.05 | 2023-06-01 12:20:00 | 717.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-06-02 10:55:00 | 719.20 | 2023-06-02 11:30:00 | 721.58 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-06-05 10:00:00 | 718.75 | 2023-06-05 10:05:00 | 721.30 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-06-06 09:30:00 | 717.30 | 2023-06-06 09:35:00 | 714.96 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-06-08 11:05:00 | 745.60 | 2023-06-08 11:10:00 | 747.41 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-06-09 09:40:00 | 743.60 | 2023-06-09 10:20:00 | 741.13 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-06-12 10:25:00 | 745.35 | 2023-06-12 12:00:00 | 749.16 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-06-12 10:25:00 | 745.35 | 2023-06-12 14:20:00 | 745.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-14 11:15:00 | 742.15 | 2023-06-14 12:00:00 | 743.74 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-15 10:25:00 | 755.10 | 2023-06-15 10:45:00 | 757.89 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-06-15 10:25:00 | 755.10 | 2023-06-15 11:05:00 | 755.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-19 10:20:00 | 764.65 | 2023-06-19 10:25:00 | 762.02 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-06-21 10:15:00 | 748.70 | 2023-06-21 14:15:00 | 745.29 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-06-21 10:15:00 | 748.70 | 2023-06-21 15:20:00 | 745.30 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2023-06-22 09:30:00 | 755.50 | 2023-06-22 09:35:00 | 753.39 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-27 09:40:00 | 755.80 | 2023-06-27 09:50:00 | 759.59 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-06-27 09:40:00 | 755.80 | 2023-06-27 15:20:00 | 783.60 | TARGET_HIT | 0.50 | 3.68% |
| SELL | retest1 | 2023-06-28 09:40:00 | 783.25 | 2023-06-28 10:35:00 | 786.34 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-07-05 10:45:00 | 781.80 | 2023-07-05 11:15:00 | 784.60 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-07-10 10:00:00 | 784.25 | 2023-07-10 10:15:00 | 781.03 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-07-14 09:40:00 | 806.50 | 2023-07-14 10:00:00 | 802.74 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2023-07-18 11:10:00 | 801.00 | 2023-07-18 11:35:00 | 797.72 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-07-18 11:10:00 | 801.00 | 2023-07-18 15:20:00 | 798.20 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2023-08-01 10:10:00 | 743.00 | 2023-08-01 10:15:00 | 745.46 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-08-02 11:00:00 | 730.80 | 2023-08-02 11:15:00 | 733.14 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-08-08 10:45:00 | 739.35 | 2023-08-08 11:15:00 | 736.36 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-08-08 10:45:00 | 739.35 | 2023-08-08 12:00:00 | 739.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-09 10:40:00 | 733.85 | 2023-08-09 11:20:00 | 731.53 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-08-09 10:40:00 | 733.85 | 2023-08-09 11:35:00 | 733.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-10 10:25:00 | 723.80 | 2023-08-10 10:30:00 | 726.23 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-08-18 10:00:00 | 730.65 | 2023-08-18 10:05:00 | 726.86 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-08-18 10:00:00 | 730.65 | 2023-08-18 10:15:00 | 730.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-21 09:30:00 | 735.05 | 2023-08-21 09:45:00 | 732.73 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-22 10:25:00 | 749.30 | 2023-08-22 10:35:00 | 747.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-08-30 10:40:00 | 754.00 | 2023-08-30 11:00:00 | 752.08 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-08-31 09:55:00 | 754.00 | 2023-08-31 12:00:00 | 756.83 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-08-31 09:55:00 | 754.00 | 2023-08-31 12:55:00 | 754.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-04 09:30:00 | 760.80 | 2023-09-04 09:45:00 | 763.19 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-09-08 09:40:00 | 782.10 | 2023-09-08 09:55:00 | 779.97 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-09-11 09:35:00 | 782.95 | 2023-09-11 09:40:00 | 780.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-09-15 10:55:00 | 790.75 | 2023-09-15 11:40:00 | 788.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-09-21 10:45:00 | 768.25 | 2023-09-21 10:50:00 | 770.72 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-09-22 09:50:00 | 758.35 | 2023-09-22 10:10:00 | 761.72 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-09-27 09:45:00 | 769.00 | 2023-09-27 10:00:00 | 766.77 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-09-28 09:45:00 | 766.25 | 2023-09-28 10:00:00 | 768.21 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-09-29 10:10:00 | 765.65 | 2023-09-29 10:50:00 | 763.05 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-10-05 09:40:00 | 767.75 | 2023-10-05 09:45:00 | 765.18 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-10-06 10:50:00 | 771.45 | 2023-10-06 11:00:00 | 769.16 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-10-13 09:45:00 | 751.05 | 2023-10-13 09:50:00 | 752.67 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-10-16 09:45:00 | 739.50 | 2023-10-16 10:50:00 | 741.46 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-10-20 09:40:00 | 777.55 | 2023-10-20 09:45:00 | 774.47 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-10-23 10:10:00 | 759.00 | 2023-10-23 10:20:00 | 754.85 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-10-23 10:10:00 | 759.00 | 2023-10-23 11:55:00 | 756.80 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2023-10-27 11:10:00 | 738.45 | 2023-10-27 13:35:00 | 736.20 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-11-01 09:30:00 | 768.00 | 2023-11-01 09:35:00 | 771.39 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-11-01 09:30:00 | 768.00 | 2023-11-01 10:50:00 | 770.95 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2023-11-03 10:00:00 | 775.95 | 2023-11-03 10:20:00 | 779.23 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-11-03 10:00:00 | 775.95 | 2023-11-03 10:30:00 | 775.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-06 10:25:00 | 779.50 | 2023-11-06 10:35:00 | 777.51 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-11-07 09:45:00 | 771.20 | 2023-11-07 10:35:00 | 767.93 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-11-07 09:45:00 | 771.20 | 2023-11-07 15:20:00 | 755.45 | TARGET_HIT | 0.50 | 2.04% |
| SELL | retest1 | 2023-12-08 09:40:00 | 796.20 | 2023-12-08 13:00:00 | 789.19 | PARTIAL | 0.50 | 0.88% |
| SELL | retest1 | 2023-12-08 09:40:00 | 796.20 | 2023-12-08 15:20:00 | 790.30 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2023-12-11 09:45:00 | 802.00 | 2023-12-11 09:50:00 | 799.45 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-12-12 10:20:00 | 742.00 | 2023-12-12 12:35:00 | 747.38 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2023-12-14 09:40:00 | 787.05 | 2023-12-14 10:25:00 | 784.05 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-12-15 10:45:00 | 782.95 | 2023-12-15 10:50:00 | 779.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-12-15 10:45:00 | 782.95 | 2023-12-15 11:05:00 | 782.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-19 09:45:00 | 785.00 | 2023-12-19 10:55:00 | 787.51 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-12-22 10:45:00 | 757.50 | 2023-12-22 10:55:00 | 760.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-27 09:30:00 | 784.00 | 2023-12-27 09:40:00 | 781.40 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-28 09:35:00 | 785.10 | 2023-12-28 09:40:00 | 788.69 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-12-28 09:35:00 | 785.10 | 2023-12-28 09:50:00 | 785.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-29 10:40:00 | 779.75 | 2023-12-29 10:50:00 | 777.91 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-02 10:30:00 | 766.00 | 2024-01-02 12:20:00 | 768.38 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-01-03 11:00:00 | 770.20 | 2024-01-03 11:55:00 | 767.74 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-01-04 09:30:00 | 773.50 | 2024-01-04 09:40:00 | 776.81 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-01-04 09:30:00 | 773.50 | 2024-01-04 09:50:00 | 773.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-05 10:50:00 | 783.15 | 2024-01-05 11:05:00 | 785.43 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-09 10:55:00 | 769.60 | 2024-01-09 11:05:00 | 771.79 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-15 09:50:00 | 760.95 | 2024-01-15 10:05:00 | 763.68 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-01-19 09:35:00 | 787.60 | 2024-01-19 09:40:00 | 784.64 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-01-29 10:25:00 | 769.75 | 2024-01-29 11:20:00 | 765.80 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-01-29 10:25:00 | 769.75 | 2024-01-29 15:20:00 | 767.00 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2024-01-30 10:55:00 | 771.85 | 2024-01-30 11:30:00 | 775.02 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-01-30 10:55:00 | 771.85 | 2024-01-30 14:05:00 | 771.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-31 10:00:00 | 772.90 | 2024-01-31 10:45:00 | 776.87 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-01-31 10:00:00 | 772.90 | 2024-01-31 12:35:00 | 775.05 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-02-02 09:35:00 | 820.30 | 2024-02-02 10:15:00 | 827.05 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2024-02-02 09:35:00 | 820.30 | 2024-02-02 12:35:00 | 823.05 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2024-02-05 11:10:00 | 816.00 | 2024-02-05 11:20:00 | 819.13 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-02-13 11:15:00 | 793.85 | 2024-02-13 12:05:00 | 790.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-02-16 09:35:00 | 824.10 | 2024-02-16 09:40:00 | 821.51 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-02-20 10:20:00 | 795.55 | 2024-02-20 10:35:00 | 792.04 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-02-20 10:20:00 | 795.55 | 2024-02-20 11:30:00 | 795.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-21 11:00:00 | 809.60 | 2024-02-21 11:05:00 | 807.01 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-02-23 09:30:00 | 808.15 | 2024-02-23 09:40:00 | 805.81 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-02-26 09:45:00 | 805.95 | 2024-02-26 09:50:00 | 803.57 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-02-27 10:50:00 | 796.10 | 2024-02-27 11:35:00 | 793.80 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-02-27 10:50:00 | 796.10 | 2024-02-27 12:55:00 | 794.95 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-02-28 10:50:00 | 783.00 | 2024-02-28 10:55:00 | 785.09 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-03-01 10:40:00 | 789.00 | 2024-03-01 10:50:00 | 786.40 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-04 10:15:00 | 795.30 | 2024-03-04 10:55:00 | 792.71 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-03-05 10:40:00 | 788.50 | 2024-03-05 10:55:00 | 790.30 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-03-06 09:45:00 | 779.00 | 2024-03-06 09:55:00 | 774.55 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-03-06 09:45:00 | 779.00 | 2024-03-06 13:05:00 | 773.95 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2024-03-11 11:10:00 | 776.80 | 2024-03-11 11:25:00 | 774.01 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-03-11 11:10:00 | 776.80 | 2024-03-11 12:35:00 | 774.80 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2024-03-12 10:45:00 | 758.00 | 2024-03-12 10:55:00 | 760.61 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-03-13 09:55:00 | 748.25 | 2024-03-13 10:15:00 | 750.99 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-03-18 10:30:00 | 716.70 | 2024-03-18 10:35:00 | 719.44 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-03-19 10:55:00 | 705.40 | 2024-03-19 11:45:00 | 701.97 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-03-19 10:55:00 | 705.40 | 2024-03-19 11:50:00 | 705.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 09:30:00 | 732.65 | 2024-03-21 09:50:00 | 736.32 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-03-21 09:30:00 | 732.65 | 2024-03-21 11:40:00 | 734.30 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-03-22 09:35:00 | 749.95 | 2024-03-22 09:50:00 | 747.33 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-03-26 11:05:00 | 751.15 | 2024-03-26 12:00:00 | 754.11 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-03-26 11:05:00 | 751.15 | 2024-03-26 12:15:00 | 751.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-28 10:40:00 | 750.50 | 2024-03-28 10:55:00 | 752.48 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-04-03 09:50:00 | 797.00 | 2024-04-03 09:55:00 | 794.92 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-12 09:30:00 | 802.35 | 2024-04-12 09:40:00 | 798.51 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-04-12 09:30:00 | 802.35 | 2024-04-12 15:20:00 | 784.30 | TARGET_HIT | 0.50 | 2.25% |
| SELL | retest1 | 2024-04-18 10:35:00 | 769.45 | 2024-04-18 11:15:00 | 771.47 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-23 09:40:00 | 750.70 | 2024-04-23 09:55:00 | 753.13 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-04-25 09:35:00 | 740.10 | 2024-04-25 09:55:00 | 743.81 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-04-26 09:50:00 | 745.50 | 2024-04-26 13:10:00 | 741.94 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-04-26 09:50:00 | 745.50 | 2024-04-26 14:25:00 | 745.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-29 09:35:00 | 746.20 | 2024-04-29 09:50:00 | 742.53 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-04-29 09:35:00 | 746.20 | 2024-04-29 13:05:00 | 736.85 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2024-05-07 09:35:00 | 749.40 | 2024-05-07 09:45:00 | 751.87 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-08 09:55:00 | 743.00 | 2024-05-08 10:05:00 | 746.65 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-05-08 09:55:00 | 743.00 | 2024-05-08 10:20:00 | 743.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-09 10:25:00 | 734.90 | 2024-05-09 10:45:00 | 732.05 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-09 10:25:00 | 734.90 | 2024-05-09 11:45:00 | 734.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-10 09:40:00 | 729.30 | 2024-05-10 11:20:00 | 734.57 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-05-10 09:40:00 | 729.30 | 2024-05-10 12:10:00 | 729.30 | STOP_HIT | 0.50 | 0.00% |
