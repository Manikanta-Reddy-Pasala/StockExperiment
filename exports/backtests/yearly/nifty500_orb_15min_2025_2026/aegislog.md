# Aegis Logistics Ltd. (AEGISLOG)

## Backtest Summary

- **Window:** 2025-09-08 09:15:00 → 2026-05-08 15:25:00 (9463 bars)
- **Last close:** 725.00
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
| ENTRY1 | 41 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 34
- **Target hits / Stop hits / Partials:** 7 / 34 / 19
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 14.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 8 | 38.1% | 3 | 13 | 5 | 0.38% | 8.0% |
| BUY @ 2nd Alert (retest1) | 21 | 8 | 38.1% | 3 | 13 | 5 | 0.38% | 8.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 18 | 46.2% | 4 | 21 | 14 | 0.16% | 6.2% |
| SELL @ 2nd Alert (retest1) | 39 | 18 | 46.2% | 4 | 21 | 14 | 0.16% | 6.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 60 | 26 | 43.3% | 7 | 34 | 19 | 0.24% | 14.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 11:00:00 | 708.25 | 704.40 | 0.00 | ORB-long ORB[700.70,705.60] vol=3.3x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-09-11 11:05:00 | 706.30 | 704.49 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-09-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:10:00 | 702.30 | 707.04 | 0.00 | ORB-short ORB[706.10,716.55] vol=3.9x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 704.52 | 706.94 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:30:00 | 789.60 | 791.78 | 0.00 | ORB-short ORB[790.00,800.95] vol=1.9x ATR=3.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:50:00 | 784.44 | 789.73 | 0.00 | T1 1.5R @ 784.44 |
| Stop hit — per-position SL triggered | 2025-09-18 12:05:00 | 789.60 | 789.58 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-09-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:35:00 | 801.15 | 795.10 | 0.00 | ORB-long ORB[787.00,796.95] vol=1.8x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-09-22 09:50:00 | 797.63 | 797.74 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:35:00 | 770.10 | 766.03 | 0.00 | ORB-long ORB[760.55,767.45] vol=2.2x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-09-25 09:45:00 | 767.30 | 767.27 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:35:00 | 764.95 | 768.65 | 0.00 | ORB-short ORB[767.05,776.95] vol=2.0x ATR=3.00 |
| Stop hit — per-position SL triggered | 2025-09-26 09:45:00 | 767.95 | 768.50 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-10-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:35:00 | 813.85 | 798.94 | 0.00 | ORB-long ORB[782.00,792.45] vol=6.3x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:45:00 | 820.24 | 823.23 | 0.00 | T1 1.5R @ 820.24 |
| Target hit | 2025-10-03 11:15:00 | 879.40 | 890.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2025-10-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:50:00 | 822.80 | 829.85 | 0.00 | ORB-short ORB[824.00,835.00] vol=2.1x ATR=3.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 13:10:00 | 817.70 | 827.13 | 0.00 | T1 1.5R @ 817.70 |
| Stop hit — per-position SL triggered | 2025-10-07 13:40:00 | 822.80 | 825.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 816.00 | 813.15 | 0.00 | ORB-long ORB[805.90,815.20] vol=2.6x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-10-10 09:35:00 | 812.84 | 813.28 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 11:15:00 | 807.05 | 812.55 | 0.00 | ORB-short ORB[810.00,815.00] vol=4.1x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 13:30:00 | 803.37 | 810.54 | 0.00 | T1 1.5R @ 803.37 |
| Target hit | 2025-10-16 15:20:00 | 802.70 | 807.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 811.30 | 807.66 | 0.00 | ORB-long ORB[799.85,809.60] vol=1.5x ATR=2.62 |
| Stop hit — per-position SL triggered | 2025-10-17 09:50:00 | 808.68 | 808.71 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-10-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:35:00 | 810.45 | 806.33 | 0.00 | ORB-long ORB[799.95,807.80] vol=2.3x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-10-20 09:40:00 | 808.10 | 806.48 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:35:00 | 800.60 | 798.09 | 0.00 | ORB-long ORB[795.00,799.95] vol=1.6x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-10-24 09:45:00 | 798.52 | 798.29 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-10-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:55:00 | 761.40 | 767.85 | 0.00 | ORB-short ORB[766.10,773.45] vol=1.5x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:10:00 | 758.42 | 765.84 | 0.00 | T1 1.5R @ 758.42 |
| Stop hit — per-position SL triggered | 2025-10-31 11:35:00 | 761.40 | 765.14 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:15:00 | 769.90 | 771.24 | 0.00 | ORB-short ORB[770.40,776.50] vol=2.1x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-11-04 11:25:00 | 771.94 | 771.22 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:15:00 | 753.00 | 759.58 | 0.00 | ORB-short ORB[760.00,769.05] vol=3.9x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-11-06 10:55:00 | 755.69 | 758.56 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:30:00 | 771.90 | 769.88 | 0.00 | ORB-long ORB[766.00,770.90] vol=2.0x ATR=3.16 |
| Stop hit — per-position SL triggered | 2025-11-10 09:45:00 | 768.74 | 770.35 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:15:00 | 794.65 | 797.85 | 0.00 | ORB-short ORB[797.25,806.60] vol=2.0x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-11-14 11:25:00 | 797.17 | 796.87 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:15:00 | 798.40 | 793.33 | 0.00 | ORB-long ORB[789.95,797.00] vol=1.5x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-11-17 10:35:00 | 795.75 | 794.25 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-11-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:35:00 | 793.20 | 794.65 | 0.00 | ORB-short ORB[793.60,798.95] vol=1.7x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:10:00 | 790.23 | 794.02 | 0.00 | T1 1.5R @ 790.23 |
| Target hit | 2025-11-18 12:05:00 | 791.90 | 791.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — SELL (started 2025-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:35:00 | 775.20 | 779.91 | 0.00 | ORB-short ORB[778.20,789.05] vol=2.2x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-11-20 09:40:00 | 777.24 | 779.44 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:15:00 | 768.30 | 772.38 | 0.00 | ORB-short ORB[772.55,777.45] vol=4.6x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 12:40:00 | 765.61 | 770.22 | 0.00 | T1 1.5R @ 765.61 |
| Target hit | 2025-11-21 15:20:00 | 765.00 | 768.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:15:00 | 757.30 | 758.49 | 0.00 | ORB-short ORB[759.00,766.75] vol=2.5x ATR=2.29 |
| Stop hit — per-position SL triggered | 2025-11-27 12:10:00 | 759.59 | 758.26 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-11-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:55:00 | 751.75 | 753.82 | 0.00 | ORB-short ORB[752.65,760.00] vol=2.0x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 11:45:00 | 749.74 | 752.46 | 0.00 | T1 1.5R @ 749.74 |
| Stop hit — per-position SL triggered | 2025-11-28 11:50:00 | 751.75 | 752.46 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-12-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:45:00 | 766.20 | 771.63 | 0.00 | ORB-short ORB[770.15,779.00] vol=3.9x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-12-03 10:50:00 | 767.91 | 771.39 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-12-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:35:00 | 753.00 | 756.39 | 0.00 | ORB-short ORB[754.50,763.50] vol=1.6x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-12-05 10:05:00 | 755.20 | 754.99 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2026-01-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 10:00:00 | 753.20 | 750.94 | 0.00 | ORB-long ORB[743.95,751.85] vol=3.0x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:40:00 | 756.84 | 752.00 | 0.00 | T1 1.5R @ 756.84 |
| Stop hit — per-position SL triggered | 2026-01-08 10:50:00 | 753.20 | 752.18 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2026-01-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:05:00 | 698.15 | 698.44 | 0.00 | ORB-short ORB[700.10,706.00] vol=4.7x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 11:35:00 | 695.24 | 697.84 | 0.00 | T1 1.5R @ 695.24 |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 698.15 | 697.42 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2026-02-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 09:55:00 | 696.85 | 698.75 | 0.00 | ORB-short ORB[697.95,703.60] vol=1.8x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 10:20:00 | 693.10 | 697.58 | 0.00 | T1 1.5R @ 693.10 |
| Stop hit — per-position SL triggered | 2026-02-04 10:25:00 | 696.85 | 697.25 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2026-02-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:45:00 | 696.00 | 689.19 | 0.00 | ORB-long ORB[685.25,689.35] vol=2.9x ATR=2.94 |
| Stop hit — per-position SL triggered | 2026-02-06 09:55:00 | 693.06 | 690.21 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2026-02-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:40:00 | 707.65 | 713.10 | 0.00 | ORB-short ORB[713.10,722.40] vol=2.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-02-11 11:35:00 | 709.78 | 712.27 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2026-02-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:50:00 | 703.50 | 701.60 | 0.00 | ORB-long ORB[695.65,700.80] vol=2.5x ATR=2.95 |
| Stop hit — per-position SL triggered | 2026-02-16 10:05:00 | 700.55 | 701.70 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 694.00 | 697.10 | 0.00 | ORB-short ORB[695.35,705.00] vol=1.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:40:00 | 691.03 | 695.82 | 0.00 | T1 1.5R @ 691.03 |
| Stop hit — per-position SL triggered | 2026-02-17 13:40:00 | 694.00 | 692.82 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 691.75 | 693.82 | 0.00 | ORB-short ORB[692.55,700.00] vol=8.4x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 689.50 | 692.51 | 0.00 | T1 1.5R @ 689.50 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 691.75 | 692.29 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-02-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:45:00 | 689.05 | 693.58 | 0.00 | ORB-short ORB[691.00,697.95] vol=4.4x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-02-19 10:50:00 | 690.68 | 692.72 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 686.00 | 682.42 | 0.00 | ORB-long ORB[679.50,684.75] vol=2.1x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:00:00 | 689.97 | 684.97 | 0.00 | T1 1.5R @ 689.97 |
| Target hit | 2026-02-20 13:05:00 | 690.70 | 691.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — SELL (started 2026-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:55:00 | 659.80 | 663.00 | 0.00 | ORB-short ORB[661.70,670.50] vol=2.1x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:20:00 | 654.94 | 660.31 | 0.00 | T1 1.5R @ 654.94 |
| Target hit | 2026-03-04 15:20:00 | 647.95 | 654.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — SELL (started 2026-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:00:00 | 714.65 | 720.76 | 0.00 | ORB-short ORB[719.10,727.90] vol=1.5x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:20:00 | 710.96 | 718.49 | 0.00 | T1 1.5R @ 710.96 |
| Stop hit — per-position SL triggered | 2026-04-22 10:45:00 | 714.65 | 717.34 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 722.10 | 715.88 | 0.00 | ORB-long ORB[709.20,715.00] vol=3.2x ATR=2.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:35:00 | 726.17 | 719.96 | 0.00 | T1 1.5R @ 726.17 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 722.10 | 720.31 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-05-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:55:00 | 710.40 | 713.55 | 0.00 | ORB-short ORB[711.00,721.00] vol=2.0x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:45:00 | 706.81 | 711.11 | 0.00 | T1 1.5R @ 706.81 |
| Stop hit — per-position SL triggered | 2026-05-05 12:55:00 | 710.40 | 710.87 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 718.00 | 716.91 | 0.00 | ORB-long ORB[713.30,717.80] vol=3.0x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:05:00 | 721.26 | 717.85 | 0.00 | T1 1.5R @ 721.26 |
| Target hit | 2026-05-06 11:00:00 | 720.00 | 720.20 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-11 11:00:00 | 708.25 | 2025-09-11 11:05:00 | 706.30 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-12 10:10:00 | 702.30 | 2025-09-12 10:15:00 | 704.52 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-09-18 09:30:00 | 789.60 | 2025-09-18 11:50:00 | 784.44 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-09-18 09:30:00 | 789.60 | 2025-09-18 12:05:00 | 789.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 09:35:00 | 801.15 | 2025-09-22 09:50:00 | 797.63 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-09-25 09:35:00 | 770.10 | 2025-09-25 09:45:00 | 767.30 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-26 09:35:00 | 764.95 | 2025-09-26 09:45:00 | 767.95 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-10-03 10:35:00 | 813.85 | 2025-10-03 10:45:00 | 820.24 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-10-03 10:35:00 | 813.85 | 2025-10-03 11:15:00 | 879.40 | TARGET_HIT | 0.50 | 8.05% |
| SELL | retest1 | 2025-10-07 10:50:00 | 822.80 | 2025-10-07 13:10:00 | 817.70 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-10-07 10:50:00 | 822.80 | 2025-10-07 13:40:00 | 822.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 09:30:00 | 816.00 | 2025-10-10 09:35:00 | 812.84 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-10-16 11:15:00 | 807.05 | 2025-10-16 13:30:00 | 803.37 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-16 11:15:00 | 807.05 | 2025-10-16 15:20:00 | 802.70 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2025-10-17 09:35:00 | 811.30 | 2025-10-17 09:50:00 | 808.68 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-20 09:35:00 | 810.45 | 2025-10-20 09:40:00 | 808.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-24 09:35:00 | 800.60 | 2025-10-24 09:45:00 | 798.52 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-31 10:55:00 | 761.40 | 2025-10-31 11:10:00 | 758.42 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-31 10:55:00 | 761.40 | 2025-10-31 11:35:00 | 761.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 11:15:00 | 769.90 | 2025-11-04 11:25:00 | 771.94 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-06 10:15:00 | 753.00 | 2025-11-06 10:55:00 | 755.69 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-11-10 09:30:00 | 771.90 | 2025-11-10 09:45:00 | 768.74 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-11-14 10:15:00 | 794.65 | 2025-11-14 11:25:00 | 797.17 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-17 10:15:00 | 798.40 | 2025-11-17 10:35:00 | 795.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-11-18 09:35:00 | 793.20 | 2025-11-18 10:10:00 | 790.23 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-11-18 09:35:00 | 793.20 | 2025-11-18 12:05:00 | 791.90 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-11-20 09:35:00 | 775.20 | 2025-11-20 09:40:00 | 777.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-21 10:15:00 | 768.30 | 2025-11-21 12:40:00 | 765.61 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-21 10:15:00 | 768.30 | 2025-11-21 15:20:00 | 765.00 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-27 11:15:00 | 757.30 | 2025-11-27 12:10:00 | 759.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-28 10:55:00 | 751.75 | 2025-11-28 11:45:00 | 749.74 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-11-28 10:55:00 | 751.75 | 2025-11-28 11:50:00 | 751.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 10:45:00 | 766.20 | 2025-12-03 10:50:00 | 767.91 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-05 09:35:00 | 753.00 | 2025-12-05 10:05:00 | 755.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-01-08 10:00:00 | 753.20 | 2026-01-08 10:40:00 | 756.84 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-01-08 10:00:00 | 753.20 | 2026-01-08 10:50:00 | 753.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 11:05:00 | 698.15 | 2026-01-29 11:35:00 | 695.24 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-01-29 11:05:00 | 698.15 | 2026-01-29 12:15:00 | 698.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-04 09:55:00 | 696.85 | 2026-02-04 10:20:00 | 693.10 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-04 09:55:00 | 696.85 | 2026-02-04 10:25:00 | 696.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-06 09:45:00 | 696.00 | 2026-02-06 09:55:00 | 693.06 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-11 10:40:00 | 707.65 | 2026-02-11 11:35:00 | 709.78 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-16 09:50:00 | 703.50 | 2026-02-16 10:05:00 | 700.55 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-02-17 09:55:00 | 694.00 | 2026-02-17 10:40:00 | 691.03 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-17 09:55:00 | 694.00 | 2026-02-17 13:40:00 | 694.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 11:05:00 | 691.75 | 2026-02-18 11:10:00 | 689.50 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-18 11:05:00 | 691.75 | 2026-02-18 12:15:00 | 691.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:45:00 | 689.05 | 2026-02-19 10:50:00 | 690.68 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-20 09:40:00 | 686.00 | 2026-02-20 10:00:00 | 689.97 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-02-20 09:40:00 | 686.00 | 2026-02-20 13:05:00 | 690.70 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2026-03-04 09:55:00 | 659.80 | 2026-03-04 13:20:00 | 654.94 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2026-03-04 09:55:00 | 659.80 | 2026-03-04 15:20:00 | 647.95 | TARGET_HIT | 0.50 | 1.80% |
| SELL | retest1 | 2026-04-22 10:00:00 | 714.65 | 2026-04-22 10:20:00 | 710.96 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-04-22 10:00:00 | 714.65 | 2026-04-22 10:45:00 | 714.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-23 09:30:00 | 722.10 | 2026-04-23 09:35:00 | 726.17 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-23 09:30:00 | 722.10 | 2026-04-23 09:40:00 | 722.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 09:55:00 | 710.40 | 2026-05-05 11:45:00 | 706.81 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-05-05 09:55:00 | 710.40 | 2026-05-05 12:55:00 | 710.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 09:50:00 | 718.00 | 2026-05-06 10:05:00 | 721.26 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-05-06 09:50:00 | 718.00 | 2026-05-06 11:00:00 | 720.00 | TARGET_HIT | 0.50 | 0.28% |
