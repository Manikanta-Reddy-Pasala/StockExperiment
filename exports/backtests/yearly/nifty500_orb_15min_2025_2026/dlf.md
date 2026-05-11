# DLF Ltd. (DLF)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-05 15:25:00 (15225 bars)
- **Last close:** 597.00
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
| ENTRY1 | 86 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 11 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 75
- **Target hits / Stop hits / Partials:** 11 / 75 / 36
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 11.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 21 | 35.0% | 5 | 39 | 16 | 0.02% | 1.4% |
| BUY @ 2nd Alert (retest1) | 60 | 21 | 35.0% | 5 | 39 | 16 | 0.02% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 62 | 26 | 41.9% | 6 | 36 | 20 | 0.16% | 10.0% |
| SELL @ 2nd Alert (retest1) | 62 | 26 | 41.9% | 6 | 36 | 20 | 0.16% | 10.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 122 | 47 | 38.5% | 11 | 75 | 36 | 0.09% | 11.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:30:00 | 688.45 | 682.79 | 0.00 | ORB-long ORB[677.30,684.50] vol=1.9x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 09:45:00 | 691.98 | 687.26 | 0.00 | T1 1.5R @ 691.98 |
| Target hit | 2025-05-14 12:20:00 | 691.70 | 692.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2025-05-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:55:00 | 698.30 | 692.88 | 0.00 | ORB-long ORB[686.55,694.20] vol=2.9x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-05-15 11:05:00 | 696.09 | 693.09 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:00:00 | 714.30 | 710.78 | 0.00 | ORB-long ORB[705.05,711.50] vol=1.8x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 10:15:00 | 718.34 | 712.60 | 0.00 | T1 1.5R @ 718.34 |
| Stop hit — per-position SL triggered | 2025-05-16 10:20:00 | 714.30 | 712.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 11:00:00 | 767.55 | 757.64 | 0.00 | ORB-long ORB[753.85,764.80] vol=2.7x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 11:05:00 | 772.14 | 759.05 | 0.00 | T1 1.5R @ 772.14 |
| Stop hit — per-position SL triggered | 2025-05-21 12:15:00 | 767.55 | 764.12 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:30:00 | 788.45 | 781.16 | 0.00 | ORB-long ORB[775.00,780.00] vol=3.0x ATR=2.61 |
| Stop hit — per-position SL triggered | 2025-05-26 09:45:00 | 785.84 | 784.37 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 11:00:00 | 780.30 | 778.54 | 0.00 | ORB-long ORB[770.25,780.00] vol=9.8x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-05-27 12:20:00 | 778.32 | 778.91 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 784.00 | 781.37 | 0.00 | ORB-long ORB[777.20,783.00] vol=2.4x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-05-29 09:55:00 | 781.58 | 782.04 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:55:00 | 806.80 | 809.17 | 0.00 | ORB-short ORB[808.05,816.75] vol=3.0x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:20:00 | 803.23 | 808.07 | 0.00 | T1 1.5R @ 803.23 |
| Stop hit — per-position SL triggered | 2025-06-03 12:05:00 | 806.80 | 807.51 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 797.60 | 802.58 | 0.00 | ORB-short ORB[800.85,809.80] vol=1.9x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-06-04 10:05:00 | 800.16 | 801.04 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:40:00 | 811.95 | 808.42 | 0.00 | ORB-long ORB[805.20,809.65] vol=5.4x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-06-05 10:50:00 | 810.27 | 809.02 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 850.00 | 835.55 | 0.00 | ORB-long ORB[822.15,832.60] vol=5.5x ATR=3.78 |
| Stop hit — per-position SL triggered | 2025-06-06 10:10:00 | 846.22 | 838.44 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 10:55:00 | 868.00 | 875.53 | 0.00 | ORB-short ORB[876.00,882.70] vol=1.8x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-06-10 11:10:00 | 869.93 | 874.79 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 876.15 | 871.76 | 0.00 | ORB-long ORB[860.25,871.55] vol=6.7x ATR=3.45 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 872.70 | 872.74 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 11:15:00 | 856.35 | 847.12 | 0.00 | ORB-long ORB[837.30,847.30] vol=1.6x ATR=2.48 |
| Stop hit — per-position SL triggered | 2025-06-20 11:40:00 | 853.87 | 848.30 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:50:00 | 841.55 | 842.71 | 0.00 | ORB-short ORB[846.10,855.00] vol=1.5x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-06-26 11:05:00 | 843.86 | 842.71 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 11:15:00 | 841.00 | 846.05 | 0.00 | ORB-short ORB[844.05,851.40] vol=2.5x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:30:00 | 837.97 | 845.59 | 0.00 | T1 1.5R @ 837.97 |
| Stop hit — per-position SL triggered | 2025-07-01 12:20:00 | 841.00 | 844.71 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:55:00 | 832.80 | 843.47 | 0.00 | ORB-short ORB[842.35,850.65] vol=2.0x ATR=2.49 |
| Stop hit — per-position SL triggered | 2025-07-02 10:05:00 | 835.29 | 841.85 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 833.95 | 831.72 | 0.00 | ORB-long ORB[826.65,833.30] vol=2.1x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 09:45:00 | 836.53 | 832.77 | 0.00 | T1 1.5R @ 836.53 |
| Stop hit — per-position SL triggered | 2025-07-04 10:05:00 | 833.95 | 833.18 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:35:00 | 838.50 | 835.22 | 0.00 | ORB-long ORB[829.45,835.75] vol=2.0x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-07-07 10:45:00 | 836.79 | 835.38 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:35:00 | 826.05 | 828.59 | 0.00 | ORB-short ORB[826.50,832.20] vol=2.1x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-07-08 09:45:00 | 828.07 | 828.27 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:45:00 | 821.90 | 827.31 | 0.00 | ORB-short ORB[826.50,832.25] vol=1.9x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:00:00 | 819.16 | 826.42 | 0.00 | T1 1.5R @ 819.16 |
| Stop hit — per-position SL triggered | 2025-07-11 11:30:00 | 821.90 | 824.02 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:40:00 | 833.10 | 828.93 | 0.00 | ORB-long ORB[825.80,831.45] vol=1.8x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 11:05:00 | 836.71 | 830.15 | 0.00 | T1 1.5R @ 836.71 |
| Stop hit — per-position SL triggered | 2025-07-15 13:25:00 | 833.10 | 834.05 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 843.55 | 849.59 | 0.00 | ORB-short ORB[846.05,856.00] vol=1.7x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 09:40:00 | 839.35 | 848.44 | 0.00 | T1 1.5R @ 839.35 |
| Stop hit — per-position SL triggered | 2025-07-17 09:50:00 | 843.55 | 847.96 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 11:10:00 | 841.85 | 846.01 | 0.00 | ORB-short ORB[846.00,852.55] vol=1.5x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-07-18 11:25:00 | 843.65 | 845.80 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:15:00 | 849.45 | 846.81 | 0.00 | ORB-long ORB[840.75,848.00] vol=2.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-07-21 11:35:00 | 847.96 | 846.92 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 843.85 | 851.03 | 0.00 | ORB-short ORB[850.10,856.45] vol=2.3x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 09:40:00 | 840.69 | 848.48 | 0.00 | T1 1.5R @ 840.69 |
| Stop hit — per-position SL triggered | 2025-07-22 09:45:00 | 843.85 | 848.13 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-07-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:05:00 | 830.65 | 837.03 | 0.00 | ORB-short ORB[835.50,842.95] vol=2.9x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:20:00 | 827.77 | 835.99 | 0.00 | T1 1.5R @ 827.77 |
| Stop hit — per-position SL triggered | 2025-07-24 14:10:00 | 830.65 | 833.46 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 10:00:00 | 789.15 | 783.68 | 0.00 | ORB-long ORB[779.30,787.45] vol=2.2x ATR=3.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 10:20:00 | 793.88 | 785.69 | 0.00 | T1 1.5R @ 793.88 |
| Stop hit — per-position SL triggered | 2025-08-01 11:25:00 | 789.15 | 787.16 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:40:00 | 759.10 | 768.65 | 0.00 | ORB-short ORB[771.10,781.75] vol=1.7x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:10:00 | 755.17 | 765.73 | 0.00 | T1 1.5R @ 755.17 |
| Stop hit — per-position SL triggered | 2025-08-06 11:15:00 | 759.10 | 765.57 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:25:00 | 757.20 | 759.66 | 0.00 | ORB-short ORB[757.50,764.60] vol=1.5x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 11:05:00 | 753.85 | 758.94 | 0.00 | T1 1.5R @ 753.85 |
| Stop hit — per-position SL triggered | 2025-08-12 12:30:00 | 757.20 | 757.88 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:40:00 | 768.25 | 764.62 | 0.00 | ORB-long ORB[759.05,766.95] vol=3.2x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:00:00 | 772.22 | 766.69 | 0.00 | T1 1.5R @ 772.22 |
| Stop hit — per-position SL triggered | 2025-08-18 10:45:00 | 768.25 | 768.68 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:30:00 | 781.50 | 778.23 | 0.00 | ORB-long ORB[770.50,777.95] vol=4.8x ATR=1.99 |
| Stop hit — per-position SL triggered | 2025-08-21 09:40:00 | 779.51 | 779.04 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 767.95 | 770.23 | 0.00 | ORB-short ORB[768.15,773.20] vol=1.7x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 09:40:00 | 765.34 | 769.36 | 0.00 | T1 1.5R @ 765.34 |
| Stop hit — per-position SL triggered | 2025-08-22 09:45:00 | 767.95 | 769.21 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-08-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:45:00 | 771.35 | 768.23 | 0.00 | ORB-long ORB[764.00,770.40] vol=2.4x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:00:00 | 773.83 | 769.59 | 0.00 | T1 1.5R @ 773.83 |
| Stop hit — per-position SL triggered | 2025-08-25 11:35:00 | 771.35 | 770.20 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 756.40 | 762.96 | 0.00 | ORB-short ORB[763.10,769.60] vol=2.1x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:45:00 | 753.36 | 759.99 | 0.00 | T1 1.5R @ 753.36 |
| Stop hit — per-position SL triggered | 2025-08-26 09:55:00 | 756.40 | 759.56 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:40:00 | 755.95 | 750.89 | 0.00 | ORB-long ORB[745.55,754.25] vol=1.9x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:55:00 | 759.21 | 752.80 | 0.00 | T1 1.5R @ 759.21 |
| Target hit | 2025-09-02 13:15:00 | 757.50 | 758.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2025-09-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:25:00 | 761.00 | 764.99 | 0.00 | ORB-short ORB[762.15,770.90] vol=1.5x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:30:00 | 757.59 | 764.77 | 0.00 | T1 1.5R @ 757.59 |
| Target hit | 2025-09-05 15:20:00 | 756.90 | 758.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:05:00 | 759.85 | 755.36 | 0.00 | ORB-long ORB[752.50,758.85] vol=2.3x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-09-10 11:10:00 | 758.07 | 755.52 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 09:55:00 | 756.25 | 759.49 | 0.00 | ORB-short ORB[757.60,764.40] vol=1.6x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-09-12 10:00:00 | 758.01 | 759.34 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:45:00 | 786.80 | 788.03 | 0.00 | ORB-short ORB[786.85,790.50] vol=2.2x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-09-18 11:00:00 | 788.20 | 788.01 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:55:00 | 779.35 | 784.97 | 0.00 | ORB-short ORB[782.70,790.25] vol=3.2x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-09-19 11:25:00 | 780.89 | 784.14 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 10:50:00 | 786.35 | 781.55 | 0.00 | ORB-long ORB[774.20,778.00] vol=2.3x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-09-22 11:05:00 | 784.55 | 781.99 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-09-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:45:00 | 761.55 | 767.24 | 0.00 | ORB-short ORB[765.00,775.70] vol=1.7x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-09-23 10:05:00 | 764.01 | 765.64 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:40:00 | 747.90 | 754.03 | 0.00 | ORB-short ORB[754.15,761.20] vol=1.9x ATR=2.18 |
| Stop hit — per-position SL triggered | 2025-09-24 10:05:00 | 750.08 | 750.81 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:35:00 | 724.15 | 720.87 | 0.00 | ORB-long ORB[713.50,722.75] vol=1.5x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-10-01 09:40:00 | 721.98 | 721.11 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:35:00 | 729.05 | 725.95 | 0.00 | ORB-long ORB[719.20,728.20] vol=1.7x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-10-03 09:45:00 | 726.93 | 726.42 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:00:00 | 733.15 | 735.69 | 0.00 | ORB-short ORB[733.20,736.85] vol=2.1x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:20:00 | 730.68 | 735.15 | 0.00 | T1 1.5R @ 730.68 |
| Stop hit — per-position SL triggered | 2025-10-07 12:05:00 | 733.15 | 734.69 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:35:00 | 767.15 | 762.85 | 0.00 | ORB-long ORB[757.15,765.45] vol=1.7x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-11-10 09:45:00 | 764.35 | 763.14 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:45:00 | 772.65 | 768.89 | 0.00 | ORB-long ORB[763.00,768.40] vol=1.7x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-11-13 10:00:00 | 770.69 | 769.99 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:50:00 | 757.60 | 761.18 | 0.00 | ORB-short ORB[758.70,769.85] vol=1.6x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-11-18 10:05:00 | 759.51 | 760.74 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 11:05:00 | 740.25 | 745.35 | 0.00 | ORB-short ORB[741.50,750.85] vol=1.9x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-11-19 12:00:00 | 742.03 | 743.62 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:45:00 | 728.70 | 725.14 | 0.00 | ORB-long ORB[720.25,725.60] vol=1.7x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 10:00:00 | 731.21 | 727.67 | 0.00 | T1 1.5R @ 731.21 |
| Target hit | 2025-11-26 12:30:00 | 729.90 | 730.02 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — SELL (started 2025-12-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:50:00 | 721.20 | 724.27 | 0.00 | ORB-short ORB[722.90,728.90] vol=1.9x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:40:00 | 718.99 | 723.11 | 0.00 | T1 1.5R @ 718.99 |
| Target hit | 2025-12-01 15:20:00 | 713.05 | 716.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:35:00 | 710.70 | 707.93 | 0.00 | ORB-long ORB[705.15,708.85] vol=1.9x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-12-04 10:50:00 | 709.15 | 708.36 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:00:00 | 719.00 | 714.14 | 0.00 | ORB-long ORB[707.15,715.75] vol=2.9x ATR=2.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 722.20 | 717.11 | 0.00 | T1 1.5R @ 722.20 |
| Stop hit — per-position SL triggered | 2025-12-05 10:20:00 | 719.00 | 717.34 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:30:00 | 699.60 | 707.71 | 0.00 | ORB-short ORB[712.10,720.05] vol=1.9x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:55:00 | 696.31 | 705.73 | 0.00 | T1 1.5R @ 696.31 |
| Target hit | 2025-12-08 15:20:00 | 687.10 | 694.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2025-12-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:40:00 | 689.20 | 684.58 | 0.00 | ORB-long ORB[681.50,687.90] vol=1.5x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-12-11 10:55:00 | 687.34 | 684.88 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:40:00 | 704.25 | 701.07 | 0.00 | ORB-long ORB[694.60,703.00] vol=3.3x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-12-12 09:45:00 | 702.22 | 701.32 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:10:00 | 697.50 | 694.37 | 0.00 | ORB-long ORB[688.35,693.90] vol=2.1x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-12-23 10:20:00 | 695.80 | 694.80 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:45:00 | 692.20 | 694.83 | 0.00 | ORB-short ORB[694.10,696.95] vol=1.8x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:15:00 | 690.41 | 694.19 | 0.00 | T1 1.5R @ 690.41 |
| Target hit | 2025-12-29 15:20:00 | 688.95 | 690.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:15:00 | 681.10 | 684.50 | 0.00 | ORB-short ORB[683.20,686.25] vol=1.7x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 12:15:00 | 679.48 | 683.64 | 0.00 | T1 1.5R @ 679.48 |
| Stop hit — per-position SL triggered | 2025-12-30 15:10:00 | 681.10 | 680.82 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:05:00 | 686.70 | 684.87 | 0.00 | ORB-long ORB[680.65,684.90] vol=4.6x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 12:05:00 | 688.78 | 685.38 | 0.00 | T1 1.5R @ 688.78 |
| Stop hit — per-position SL triggered | 2025-12-31 14:50:00 | 686.70 | 687.09 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:45:00 | 699.90 | 697.71 | 0.00 | ORB-long ORB[692.65,698.50] vol=2.6x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:55:00 | 702.71 | 698.94 | 0.00 | T1 1.5R @ 702.71 |
| Target hit | 2026-01-02 11:05:00 | 701.70 | 701.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — BUY (started 2026-01-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:25:00 | 705.05 | 701.62 | 0.00 | ORB-long ORB[696.45,702.00] vol=1.9x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 10:35:00 | 707.88 | 702.17 | 0.00 | T1 1.5R @ 707.88 |
| Stop hit — per-position SL triggered | 2026-01-05 10:40:00 | 705.05 | 702.31 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-02-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:50:00 | 650.10 | 652.96 | 0.00 | ORB-short ORB[650.25,660.00] vol=2.1x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-02-05 11:10:00 | 652.85 | 652.45 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 675.50 | 672.31 | 0.00 | ORB-long ORB[668.65,674.30] vol=6.0x ATR=1.92 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 673.58 | 672.52 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 666.00 | 669.39 | 0.00 | ORB-short ORB[668.50,674.40] vol=1.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:00:00 | 663.70 | 667.40 | 0.00 | T1 1.5R @ 663.70 |
| Stop hit — per-position SL triggered | 2026-02-11 10:20:00 | 666.00 | 666.71 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 662.60 | 664.19 | 0.00 | ORB-short ORB[662.80,670.25] vol=2.8x ATR=1.72 |
| Stop hit — per-position SL triggered | 2026-02-12 09:55:00 | 664.32 | 663.27 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:35:00 | 637.05 | 640.00 | 0.00 | ORB-short ORB[638.10,644.45] vol=1.6x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:10:00 | 634.65 | 638.90 | 0.00 | T1 1.5R @ 634.65 |
| Target hit | 2026-02-19 15:20:00 | 618.70 | 629.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2026-02-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:35:00 | 625.35 | 627.76 | 0.00 | ORB-short ORB[627.00,634.00] vol=2.6x ATR=1.76 |
| Stop hit — per-position SL triggered | 2026-02-23 10:50:00 | 627.11 | 627.57 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:55:00 | 618.30 | 621.38 | 0.00 | ORB-short ORB[619.65,623.90] vol=2.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-02-24 11:05:00 | 619.67 | 621.06 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:50:00 | 615.30 | 618.19 | 0.00 | ORB-short ORB[615.70,623.85] vol=1.5x ATR=2.25 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 617.55 | 618.02 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:50:00 | 614.20 | 612.26 | 0.00 | ORB-long ORB[608.75,614.00] vol=1.8x ATR=1.80 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 612.40 | 612.26 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 577.90 | 581.35 | 0.00 | ORB-short ORB[578.95,586.15] vol=1.6x ATR=2.83 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 580.73 | 579.73 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 593.65 | 589.97 | 0.00 | ORB-long ORB[584.60,590.90] vol=2.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-03-11 09:35:00 | 591.62 | 590.69 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 536.20 | 534.55 | 0.00 | ORB-long ORB[529.70,533.30] vol=2.2x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 534.24 | 535.12 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:45:00 | 555.35 | 551.64 | 0.00 | ORB-long ORB[546.15,551.35] vol=1.6x ATR=2.72 |
| Stop hit — per-position SL triggered | 2026-03-20 10:25:00 | 552.63 | 553.50 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-03-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:35:00 | 511.55 | 515.66 | 0.00 | ORB-short ORB[512.80,520.40] vol=3.1x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 11:00:00 | 508.01 | 514.93 | 0.00 | T1 1.5R @ 508.01 |
| Target hit | 2026-03-30 15:20:00 | 503.35 | 508.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 79 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 570.00 | 567.93 | 0.00 | ORB-long ORB[564.20,569.95] vol=1.6x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-04-10 09:50:00 | 567.74 | 568.19 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-04-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:35:00 | 575.80 | 578.67 | 0.00 | ORB-short ORB[576.30,583.75] vol=1.7x ATR=2.54 |
| Stop hit — per-position SL triggered | 2026-04-15 09:40:00 | 578.34 | 578.67 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:00:00 | 594.10 | 590.45 | 0.00 | ORB-long ORB[587.20,592.70] vol=2.2x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:50:00 | 597.22 | 593.01 | 0.00 | T1 1.5R @ 597.22 |
| Target hit | 2026-04-17 15:20:00 | 602.50 | 597.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — BUY (started 2026-04-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:50:00 | 611.60 | 608.89 | 0.00 | ORB-long ORB[605.45,611.40] vol=2.1x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-04-22 11:00:00 | 610.04 | 609.08 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 595.60 | 601.50 | 0.00 | ORB-short ORB[601.00,605.00] vol=2.3x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-04-23 11:40:00 | 597.10 | 600.57 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-04-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:50:00 | 586.15 | 588.98 | 0.00 | ORB-short ORB[588.15,595.50] vol=1.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:00:00 | 583.93 | 588.68 | 0.00 | T1 1.5R @ 583.93 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 586.15 | 588.37 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-04-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:40:00 | 589.50 | 591.79 | 0.00 | ORB-short ORB[591.05,595.70] vol=1.9x ATR=1.68 |
| Stop hit — per-position SL triggered | 2026-04-28 10:00:00 | 591.18 | 591.39 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-04-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:55:00 | 598.95 | 594.21 | 0.00 | ORB-long ORB[589.35,595.20] vol=2.8x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:00:00 | 602.01 | 596.05 | 0.00 | T1 1.5R @ 602.01 |
| Stop hit — per-position SL triggered | 2026-04-29 10:55:00 | 598.95 | 597.42 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:30:00 | 688.45 | 2025-05-14 09:45:00 | 691.98 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-05-14 09:30:00 | 688.45 | 2025-05-14 12:20:00 | 691.70 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2025-05-15 10:55:00 | 698.30 | 2025-05-15 11:05:00 | 696.09 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-16 10:00:00 | 714.30 | 2025-05-16 10:15:00 | 718.34 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-05-16 10:00:00 | 714.30 | 2025-05-16 10:20:00 | 714.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-21 11:00:00 | 767.55 | 2025-05-21 11:05:00 | 772.14 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-05-21 11:00:00 | 767.55 | 2025-05-21 12:15:00 | 767.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-26 09:30:00 | 788.45 | 2025-05-26 09:45:00 | 785.84 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-27 11:00:00 | 780.30 | 2025-05-27 12:20:00 | 778.32 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-05-29 09:30:00 | 784.00 | 2025-05-29 09:55:00 | 781.58 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-06-03 09:55:00 | 806.80 | 2025-06-03 11:20:00 | 803.23 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-06-03 09:55:00 | 806.80 | 2025-06-03 12:05:00 | 806.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:35:00 | 797.60 | 2025-06-04 10:05:00 | 800.16 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-05 10:40:00 | 811.95 | 2025-06-05 10:50:00 | 810.27 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-06-06 10:05:00 | 850.00 | 2025-06-06 10:10:00 | 846.22 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-06-10 10:55:00 | 868.00 | 2025-06-10 11:10:00 | 869.93 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-17 09:30:00 | 876.15 | 2025-06-17 09:40:00 | 872.70 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-20 11:15:00 | 856.35 | 2025-06-20 11:40:00 | 853.87 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-26 10:50:00 | 841.55 | 2025-06-26 11:05:00 | 843.86 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-01 11:15:00 | 841.00 | 2025-07-01 11:30:00 | 837.97 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-07-01 11:15:00 | 841.00 | 2025-07-01 12:20:00 | 841.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 09:55:00 | 832.80 | 2025-07-02 10:05:00 | 835.29 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-04 09:30:00 | 833.95 | 2025-07-04 09:45:00 | 836.53 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-07-04 09:30:00 | 833.95 | 2025-07-04 10:05:00 | 833.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-07 10:35:00 | 838.50 | 2025-07-07 10:45:00 | 836.79 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-08 09:35:00 | 826.05 | 2025-07-08 09:45:00 | 828.07 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-11 10:45:00 | 821.90 | 2025-07-11 11:00:00 | 819.16 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-11 10:45:00 | 821.90 | 2025-07-11 11:30:00 | 821.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:40:00 | 833.10 | 2025-07-15 11:05:00 | 836.71 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-07-15 10:40:00 | 833.10 | 2025-07-15 13:25:00 | 833.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-17 09:30:00 | 843.55 | 2025-07-17 09:40:00 | 839.35 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-17 09:30:00 | 843.55 | 2025-07-17 09:50:00 | 843.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 11:10:00 | 841.85 | 2025-07-18 11:25:00 | 843.65 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-21 11:15:00 | 849.45 | 2025-07-21 11:35:00 | 847.96 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-22 09:30:00 | 843.85 | 2025-07-22 09:40:00 | 840.69 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-22 09:30:00 | 843.85 | 2025-07-22 09:45:00 | 843.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 11:05:00 | 830.65 | 2025-07-24 11:20:00 | 827.77 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-24 11:05:00 | 830.65 | 2025-07-24 14:10:00 | 830.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-01 10:00:00 | 789.15 | 2025-08-01 10:20:00 | 793.88 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-08-01 10:00:00 | 789.15 | 2025-08-01 11:25:00 | 789.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 10:40:00 | 759.10 | 2025-08-06 11:10:00 | 755.17 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-08-06 10:40:00 | 759.10 | 2025-08-06 11:15:00 | 759.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-12 10:25:00 | 757.20 | 2025-08-12 11:05:00 | 753.85 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-08-12 10:25:00 | 757.20 | 2025-08-12 12:30:00 | 757.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-18 09:40:00 | 768.25 | 2025-08-18 10:00:00 | 772.22 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-08-18 09:40:00 | 768.25 | 2025-08-18 10:45:00 | 768.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-21 09:30:00 | 781.50 | 2025-08-21 09:40:00 | 779.51 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-22 09:30:00 | 767.95 | 2025-08-22 09:40:00 | 765.34 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-22 09:30:00 | 767.95 | 2025-08-22 09:45:00 | 767.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-25 10:45:00 | 771.35 | 2025-08-25 11:00:00 | 773.83 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-08-25 10:45:00 | 771.35 | 2025-08-25 11:35:00 | 771.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 09:35:00 | 756.40 | 2025-08-26 09:45:00 | 753.36 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-26 09:35:00 | 756.40 | 2025-08-26 09:55:00 | 756.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 09:40:00 | 755.95 | 2025-09-02 09:55:00 | 759.21 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-02 09:40:00 | 755.95 | 2025-09-02 13:15:00 | 757.50 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-09-05 10:25:00 | 761.00 | 2025-09-05 10:30:00 | 757.59 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-09-05 10:25:00 | 761.00 | 2025-09-05 15:20:00 | 756.90 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2025-09-10 11:05:00 | 759.85 | 2025-09-10 11:10:00 | 758.07 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-12 09:55:00 | 756.25 | 2025-09-12 10:00:00 | 758.01 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-18 10:45:00 | 786.80 | 2025-09-18 11:00:00 | 788.20 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-19 10:55:00 | 779.35 | 2025-09-19 11:25:00 | 780.89 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-22 10:50:00 | 786.35 | 2025-09-22 11:05:00 | 784.55 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-23 09:45:00 | 761.55 | 2025-09-23 10:05:00 | 764.01 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-24 09:40:00 | 747.90 | 2025-09-24 10:05:00 | 750.08 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-01 09:35:00 | 724.15 | 2025-10-01 09:40:00 | 721.98 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-03 09:35:00 | 729.05 | 2025-10-03 09:45:00 | 726.93 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-07 11:00:00 | 733.15 | 2025-10-07 11:20:00 | 730.68 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-07 11:00:00 | 733.15 | 2025-10-07 12:05:00 | 733.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 09:35:00 | 767.15 | 2025-11-10 09:45:00 | 764.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-11-13 09:45:00 | 772.65 | 2025-11-13 10:00:00 | 770.69 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-18 09:50:00 | 757.60 | 2025-11-18 10:05:00 | 759.51 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-19 11:05:00 | 740.25 | 2025-11-19 12:00:00 | 742.03 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-26 09:45:00 | 728.70 | 2025-11-26 10:00:00 | 731.21 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-11-26 09:45:00 | 728.70 | 2025-11-26 12:30:00 | 729.90 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-12-01 10:50:00 | 721.20 | 2025-12-01 11:40:00 | 718.99 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-01 10:50:00 | 721.20 | 2025-12-01 15:20:00 | 713.05 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2025-12-04 10:35:00 | 710.70 | 2025-12-04 10:50:00 | 709.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-12-05 10:00:00 | 719.00 | 2025-12-05 10:15:00 | 722.20 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-05 10:00:00 | 719.00 | 2025-12-05 10:20:00 | 719.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 10:30:00 | 699.60 | 2025-12-08 10:55:00 | 696.31 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-08 10:30:00 | 699.60 | 2025-12-08 15:20:00 | 687.10 | TARGET_HIT | 0.50 | 1.79% |
| BUY | retest1 | 2025-12-11 10:40:00 | 689.20 | 2025-12-11 10:55:00 | 687.34 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-12 09:40:00 | 704.25 | 2025-12-12 09:45:00 | 702.22 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-23 10:10:00 | 697.50 | 2025-12-23 10:20:00 | 695.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-29 10:45:00 | 692.20 | 2025-12-29 11:15:00 | 690.41 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-12-29 10:45:00 | 692.20 | 2025-12-29 15:20:00 | 688.95 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-30 11:15:00 | 681.10 | 2025-12-30 12:15:00 | 679.48 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-30 11:15:00 | 681.10 | 2025-12-30 15:10:00 | 681.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 11:05:00 | 686.70 | 2025-12-31 12:05:00 | 688.78 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-31 11:05:00 | 686.70 | 2025-12-31 14:50:00 | 686.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:45:00 | 699.90 | 2026-01-02 09:55:00 | 702.71 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-01-02 09:45:00 | 699.90 | 2026-01-02 11:05:00 | 701.70 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2026-01-05 10:25:00 | 705.05 | 2026-01-05 10:35:00 | 707.88 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-01-05 10:25:00 | 705.05 | 2026-01-05 10:40:00 | 705.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 10:50:00 | 650.10 | 2026-02-05 11:10:00 | 652.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-10 09:40:00 | 675.50 | 2026-02-10 09:45:00 | 673.58 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-11 09:35:00 | 666.00 | 2026-02-11 10:00:00 | 663.70 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-11 09:35:00 | 666.00 | 2026-02-11 10:20:00 | 666.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-12 09:30:00 | 662.60 | 2026-02-12 09:55:00 | 664.32 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-19 09:35:00 | 637.05 | 2026-02-19 10:10:00 | 634.65 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-19 09:35:00 | 637.05 | 2026-02-19 15:20:00 | 618.70 | TARGET_HIT | 0.50 | 2.88% |
| SELL | retest1 | 2026-02-23 10:35:00 | 625.35 | 2026-02-23 10:50:00 | 627.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-24 10:55:00 | 618.30 | 2026-02-24 11:05:00 | 619.67 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-25 09:50:00 | 615.30 | 2026-02-25 10:00:00 | 617.55 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-26 09:50:00 | 614.20 | 2026-02-26 09:55:00 | 612.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-10 09:30:00 | 577.90 | 2026-03-10 10:15:00 | 580.73 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-03-11 09:30:00 | 593.65 | 2026-03-11 09:35:00 | 591.62 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-17 11:05:00 | 536.20 | 2026-03-17 11:25:00 | 534.24 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-20 09:45:00 | 555.35 | 2026-03-20 10:25:00 | 552.63 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-03-30 10:35:00 | 511.55 | 2026-03-30 11:00:00 | 508.01 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-30 10:35:00 | 511.55 | 2026-03-30 15:20:00 | 503.35 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2026-04-10 09:40:00 | 570.00 | 2026-04-10 09:50:00 | 567.74 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-15 09:35:00 | 575.80 | 2026-04-15 09:40:00 | 578.34 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-17 10:00:00 | 594.10 | 2026-04-17 11:50:00 | 597.22 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-17 10:00:00 | 594.10 | 2026-04-17 15:20:00 | 602.50 | TARGET_HIT | 0.50 | 1.41% |
| BUY | retest1 | 2026-04-22 10:50:00 | 611.60 | 2026-04-22 11:00:00 | 610.04 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-23 11:10:00 | 595.60 | 2026-04-23 11:40:00 | 597.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-04-24 10:50:00 | 586.15 | 2026-04-24 11:00:00 | 583.93 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-24 10:50:00 | 586.15 | 2026-04-24 11:30:00 | 586.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-28 09:40:00 | 589.50 | 2026-04-28 10:00:00 | 591.18 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-29 09:55:00 | 598.95 | 2026-04-29 10:00:00 | 602.01 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-04-29 09:55:00 | 598.95 | 2026-04-29 10:55:00 | 598.95 | STOP_HIT | 0.50 | 0.00% |
