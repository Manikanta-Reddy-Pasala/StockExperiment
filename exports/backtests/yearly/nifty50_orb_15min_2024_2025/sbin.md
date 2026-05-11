# SBIN (SBIN)

## Backtest Summary

- **Window:** 2024-09-09 09:15:00 → 2026-05-08 15:25:00 (30775 bars)
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
| ENTRY1 | 57 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 9 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 48
- **Target hits / Stop hits / Partials:** 9 / 48 / 18
- **Avg / median % per leg:** 0.02% / -0.16%
- **Sum % (uncompounded):** 1.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 11 | 29.7% | 3 | 26 | 8 | -0.05% | -1.7% |
| BUY @ 2nd Alert (retest1) | 37 | 11 | 29.7% | 3 | 26 | 8 | -0.05% | -1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 16 | 42.1% | 6 | 22 | 10 | 0.09% | 3.3% |
| SELL @ 2nd Alert (retest1) | 38 | 16 | 42.1% | 6 | 22 | 10 | 0.09% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 75 | 27 | 36.0% | 9 | 48 | 18 | 0.02% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:30:00 | 787.70 | 786.96 | 0.00 | ORB-long ORB[782.00,787.50] vol=2.8x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:40:00 | 789.53 | 787.79 | 0.00 | T1 1.5R @ 789.53 |
| Stop hit — per-position SL triggered | 2024-09-18 10:45:00 | 787.70 | 788.14 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 799.75 | 793.61 | 0.00 | ORB-long ORB[784.50,791.55] vol=2.1x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:10:00 | 802.58 | 795.56 | 0.00 | T1 1.5R @ 802.58 |
| Target hit | 2024-09-23 15:20:00 | 800.95 | 799.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-09-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 11:05:00 | 801.50 | 800.26 | 0.00 | ORB-long ORB[797.50,801.30] vol=2.4x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 12:25:00 | 803.68 | 801.01 | 0.00 | T1 1.5R @ 803.68 |
| Stop hit — per-position SL triggered | 2024-09-24 14:20:00 | 801.50 | 801.70 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:00:00 | 793.55 | 793.76 | 0.00 | ORB-short ORB[795.30,798.00] vol=2.0x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 12:20:00 | 791.36 | 793.36 | 0.00 | T1 1.5R @ 791.36 |
| Target hit | 2024-09-25 14:10:00 | 793.00 | 792.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2024-09-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:50:00 | 807.30 | 804.71 | 0.00 | ORB-long ORB[801.10,805.50] vol=1.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-09-27 10:55:00 | 806.04 | 804.81 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-09-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 09:50:00 | 792.25 | 796.64 | 0.00 | ORB-short ORB[797.05,802.60] vol=1.8x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-09-30 09:55:00 | 793.99 | 796.55 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-10-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:50:00 | 801.95 | 795.63 | 0.00 | ORB-long ORB[785.35,796.00] vol=2.1x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 799.57 | 797.14 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-10-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:00:00 | 783.55 | 795.28 | 0.00 | ORB-short ORB[798.10,804.00] vol=2.1x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-10-07 11:20:00 | 786.33 | 792.74 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-10-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:05:00 | 800.90 | 792.59 | 0.00 | ORB-long ORB[786.55,795.60] vol=1.8x ATR=3.94 |
| Stop hit — per-position SL triggered | 2024-10-09 10:25:00 | 796.96 | 794.42 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-10-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:20:00 | 801.10 | 799.76 | 0.00 | ORB-long ORB[795.00,799.85] vol=2.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-10-11 10:35:00 | 799.62 | 799.79 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 806.90 | 804.02 | 0.00 | ORB-long ORB[800.80,804.00] vol=1.7x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-10-14 10:10:00 | 805.31 | 804.80 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 794.40 | 796.67 | 0.00 | ORB-short ORB[794.65,800.75] vol=1.7x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:40:00 | 791.55 | 795.08 | 0.00 | T1 1.5R @ 791.55 |
| Target hit | 2024-10-25 12:00:00 | 783.75 | 783.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:15:00 | 810.85 | 815.77 | 0.00 | ORB-short ORB[813.00,823.90] vol=1.6x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 11:00:00 | 807.59 | 813.48 | 0.00 | T1 1.5R @ 807.59 |
| Stop hit — per-position SL triggered | 2024-11-04 11:15:00 | 810.85 | 813.03 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-11-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 10:05:00 | 836.70 | 833.38 | 0.00 | ORB-long ORB[827.50,835.60] vol=2.9x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-11-05 10:35:00 | 833.82 | 834.33 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-11-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 11:00:00 | 859.45 | 853.55 | 0.00 | ORB-long ORB[846.20,855.20] vol=1.6x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-11-06 13:35:00 | 856.93 | 855.70 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-11-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 10:35:00 | 851.20 | 858.20 | 0.00 | ORB-short ORB[856.00,862.95] vol=1.7x ATR=2.66 |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 853.86 | 856.68 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-11-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:35:00 | 850.85 | 846.17 | 0.00 | ORB-long ORB[842.00,849.95] vol=1.6x ATR=3.28 |
| Stop hit — per-position SL triggered | 2024-11-11 12:20:00 | 847.57 | 848.52 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-11-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:40:00 | 842.50 | 847.41 | 0.00 | ORB-short ORB[848.50,853.40] vol=1.9x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 12:10:00 | 839.09 | 845.41 | 0.00 | T1 1.5R @ 839.09 |
| Target hit | 2024-11-12 15:20:00 | 827.90 | 835.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 820.35 | 825.77 | 0.00 | ORB-short ORB[822.65,830.40] vol=1.6x ATR=3.13 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 823.48 | 825.60 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-11-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 11:00:00 | 805.00 | 811.57 | 0.00 | ORB-short ORB[808.65,815.80] vol=2.2x ATR=2.79 |
| Stop hit — per-position SL triggered | 2024-11-14 11:25:00 | 807.79 | 810.41 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-11-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:45:00 | 840.35 | 836.52 | 0.00 | ORB-long ORB[830.50,837.10] vol=1.5x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 09:50:00 | 843.27 | 837.49 | 0.00 | T1 1.5R @ 843.27 |
| Stop hit — per-position SL triggered | 2024-11-28 10:30:00 | 840.35 | 840.20 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 11:15:00 | 835.75 | 839.75 | 0.00 | ORB-short ORB[840.35,844.05] vol=1.7x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-11-29 11:25:00 | 837.51 | 839.66 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-12-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:35:00 | 850.50 | 845.68 | 0.00 | ORB-long ORB[836.90,846.90] vol=2.7x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 09:40:00 | 853.26 | 846.92 | 0.00 | T1 1.5R @ 853.26 |
| Stop hit — per-position SL triggered | 2024-12-03 09:45:00 | 850.50 | 847.42 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:40:00 | 858.70 | 855.28 | 0.00 | ORB-long ORB[851.25,856.75] vol=2.0x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-12-04 09:45:00 | 856.98 | 855.58 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 10:20:00 | 865.15 | 859.61 | 0.00 | ORB-long ORB[857.10,861.80] vol=2.7x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-12-05 10:25:00 | 863.05 | 859.88 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 859.35 | 862.95 | 0.00 | ORB-short ORB[860.50,865.25] vol=1.8x ATR=2.87 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 862.22 | 863.33 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 11:05:00 | 863.80 | 860.62 | 0.00 | ORB-long ORB[858.05,862.00] vol=2.6x ATR=1.39 |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 862.41 | 860.92 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:05:00 | 854.75 | 857.17 | 0.00 | ORB-short ORB[860.50,864.30] vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-12-12 11:50:00 | 856.15 | 856.49 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 11:15:00 | 841.45 | 845.88 | 0.00 | ORB-short ORB[844.65,852.00] vol=1.5x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-12-18 11:35:00 | 843.17 | 845.49 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:35:00 | 811.10 | 814.07 | 0.00 | ORB-short ORB[813.55,818.30] vol=1.5x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-12-27 09:40:00 | 812.82 | 813.88 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 783.20 | 787.13 | 0.00 | ORB-short ORB[788.50,797.60] vol=1.8x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 785.12 | 786.59 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-01-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:20:00 | 776.50 | 779.52 | 0.00 | ORB-short ORB[778.85,783.60] vol=2.1x ATR=2.45 |
| Stop hit — per-position SL triggered | 2025-01-07 11:30:00 | 778.95 | 778.24 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-01-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 10:25:00 | 771.70 | 776.18 | 0.00 | ORB-short ORB[776.25,783.95] vol=1.5x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:40:00 | 769.05 | 775.01 | 0.00 | T1 1.5R @ 769.05 |
| Target hit | 2025-01-08 13:45:00 | 767.35 | 766.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — BUY (started 2025-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:40:00 | 774.70 | 771.52 | 0.00 | ORB-long ORB[767.85,773.30] vol=1.5x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:05:00 | 777.84 | 774.11 | 0.00 | T1 1.5R @ 777.84 |
| Target hit | 2025-01-20 15:20:00 | 778.00 | 778.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 11:15:00 | 748.95 | 754.38 | 0.00 | ORB-short ORB[753.55,763.00] vol=1.7x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:20:00 | 746.11 | 752.70 | 0.00 | T1 1.5R @ 746.11 |
| Stop hit — per-position SL triggered | 2025-01-22 14:55:00 | 748.95 | 748.54 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-27 09:40:00 | 747.85 | 743.77 | 0.00 | ORB-long ORB[735.90,745.50] vol=2.4x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-01-27 10:15:00 | 745.12 | 745.93 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 11:00:00 | 750.45 | 750.86 | 0.00 | ORB-short ORB[751.20,757.50] vol=1.7x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-01-28 11:50:00 | 752.54 | 750.83 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:10:00 | 758.10 | 756.16 | 0.00 | ORB-long ORB[749.80,755.50] vol=3.1x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-01-29 11:15:00 | 756.58 | 756.25 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:35:00 | 774.50 | 770.65 | 0.00 | ORB-long ORB[765.65,772.00] vol=2.3x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-02-04 10:10:00 | 772.25 | 772.40 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-02-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:05:00 | 763.60 | 765.49 | 0.00 | ORB-short ORB[763.70,770.85] vol=1.8x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:50:00 | 760.82 | 764.52 | 0.00 | T1 1.5R @ 760.82 |
| Target hit | 2025-02-06 14:15:00 | 762.00 | 761.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2025-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:40:00 | 724.30 | 728.72 | 0.00 | ORB-short ORB[729.55,736.70] vol=1.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:45:00 | 721.05 | 727.11 | 0.00 | T1 1.5R @ 721.05 |
| Stop hit — per-position SL triggered | 2025-02-12 10:35:00 | 724.30 | 724.38 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-02-14 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:35:00 | 727.00 | 729.88 | 0.00 | ORB-short ORB[728.10,732.90] vol=2.0x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-02-14 10:55:00 | 728.85 | 729.08 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:15:00 | 724.80 | 728.35 | 0.00 | ORB-short ORB[726.70,731.50] vol=1.5x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-02-21 11:30:00 | 726.37 | 726.98 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 10:45:00 | 714.20 | 716.17 | 0.00 | ORB-short ORB[714.50,718.95] vol=1.5x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-02-25 10:55:00 | 715.34 | 716.11 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:40:00 | 737.70 | 734.83 | 0.00 | ORB-long ORB[730.50,736.00] vol=1.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-03-10 09:55:00 | 735.99 | 735.48 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 11:00:00 | 733.50 | 728.15 | 0.00 | ORB-long ORB[722.70,727.70] vol=1.8x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-03-11 11:10:00 | 731.79 | 728.36 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:15:00 | 728.00 | 731.14 | 0.00 | ORB-short ORB[730.60,737.00] vol=2.2x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:25:00 | 725.94 | 730.54 | 0.00 | T1 1.5R @ 725.94 |
| Target hit | 2025-03-12 15:20:00 | 722.70 | 726.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 09:35:00 | 725.35 | 728.20 | 0.00 | ORB-short ORB[727.85,731.25] vol=1.7x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 09:50:00 | 723.50 | 726.85 | 0.00 | T1 1.5R @ 723.50 |
| Stop hit — per-position SL triggered | 2025-03-17 11:00:00 | 725.35 | 725.73 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:00:00 | 728.75 | 725.69 | 0.00 | ORB-long ORB[722.60,727.50] vol=1.7x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:25:00 | 730.65 | 727.03 | 0.00 | T1 1.5R @ 730.65 |
| Stop hit — per-position SL triggered | 2025-03-18 10:55:00 | 728.75 | 727.78 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-03-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 11:00:00 | 770.00 | 764.10 | 0.00 | ORB-long ORB[755.10,760.95] vol=1.7x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-03-24 11:05:00 | 768.36 | 764.25 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:10:00 | 773.30 | 768.01 | 0.00 | ORB-long ORB[760.50,767.90] vol=3.5x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 771.65 | 768.16 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-04-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 11:05:00 | 779.00 | 775.59 | 0.00 | ORB-long ORB[769.95,774.75] vol=1.6x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 777.53 | 775.82 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-04-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 11:00:00 | 753.25 | 760.47 | 0.00 | ORB-short ORB[762.00,767.95] vol=1.8x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-04-09 12:25:00 | 755.60 | 757.29 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-04-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 11:05:00 | 823.40 | 815.79 | 0.00 | ORB-long ORB[801.40,813.40] vol=1.6x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-04-21 11:35:00 | 821.17 | 816.42 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:00:00 | 821.90 | 817.67 | 0.00 | ORB-long ORB[814.00,821.15] vol=3.1x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:15:00 | 825.02 | 819.30 | 0.00 | T1 1.5R @ 825.02 |
| Target hit | 2025-04-22 13:30:00 | 827.00 | 827.94 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2025-04-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:00:00 | 797.60 | 806.77 | 0.00 | ORB-short ORB[808.35,819.00] vol=2.0x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-04-25 10:05:00 | 800.13 | 805.89 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:35:00 | 810.75 | 806.82 | 0.00 | ORB-long ORB[797.40,809.30] vol=2.1x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-04-28 09:40:00 | 807.96 | 807.01 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-18 10:30:00 | 787.70 | 2024-09-18 10:40:00 | 789.53 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2024-09-18 10:30:00 | 787.70 | 2024-09-18 10:45:00 | 787.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-23 11:00:00 | 799.75 | 2024-09-23 11:10:00 | 802.58 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-23 11:00:00 | 799.75 | 2024-09-23 15:20:00 | 800.95 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-09-24 11:05:00 | 801.50 | 2024-09-24 12:25:00 | 803.68 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-09-24 11:05:00 | 801.50 | 2024-09-24 14:20:00 | 801.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 11:00:00 | 793.55 | 2024-09-25 12:20:00 | 791.36 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-09-25 11:00:00 | 793.55 | 2024-09-25 14:10:00 | 793.00 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-09-27 10:50:00 | 807.30 | 2024-09-27 10:55:00 | 806.04 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-09-30 09:50:00 | 792.25 | 2024-09-30 09:55:00 | 793.99 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-10-03 09:50:00 | 801.95 | 2024-10-03 10:15:00 | 799.57 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-07 11:00:00 | 783.55 | 2024-10-07 11:20:00 | 786.33 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-09 10:05:00 | 800.90 | 2024-10-09 10:25:00 | 796.96 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-10-11 10:20:00 | 801.10 | 2024-10-11 10:35:00 | 799.62 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-10-14 09:45:00 | 806.90 | 2024-10-14 10:10:00 | 805.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-10-25 09:30:00 | 794.40 | 2024-10-25 09:40:00 | 791.55 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-10-25 09:30:00 | 794.40 | 2024-10-25 12:00:00 | 783.75 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2024-11-04 10:15:00 | 810.85 | 2024-11-04 11:00:00 | 807.59 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-11-04 10:15:00 | 810.85 | 2024-11-04 11:15:00 | 810.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-05 10:05:00 | 836.70 | 2024-11-05 10:35:00 | 833.82 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-06 11:00:00 | 859.45 | 2024-11-06 13:35:00 | 856.93 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-08 10:35:00 | 851.20 | 2024-11-08 11:15:00 | 853.86 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-11 10:35:00 | 850.85 | 2024-11-11 12:20:00 | 847.57 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-12 10:40:00 | 842.50 | 2024-11-12 12:10:00 | 839.09 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-11-12 10:40:00 | 842.50 | 2024-11-12 15:20:00 | 827.90 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2024-11-13 09:45:00 | 820.35 | 2024-11-13 09:50:00 | 823.48 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-14 11:00:00 | 805.00 | 2024-11-14 11:25:00 | 807.79 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-11-28 09:45:00 | 840.35 | 2024-11-28 09:50:00 | 843.27 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-11-28 09:45:00 | 840.35 | 2024-11-28 10:30:00 | 840.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-29 11:15:00 | 835.75 | 2024-11-29 11:25:00 | 837.51 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-03 09:35:00 | 850.50 | 2024-12-03 09:40:00 | 853.26 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-12-03 09:35:00 | 850.50 | 2024-12-03 09:45:00 | 850.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 09:40:00 | 858.70 | 2024-12-04 09:45:00 | 856.98 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-12-05 10:20:00 | 865.15 | 2024-12-05 10:25:00 | 863.05 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-06 10:05:00 | 859.35 | 2024-12-06 10:20:00 | 862.22 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-10 11:05:00 | 863.80 | 2024-12-10 11:15:00 | 862.41 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-12-12 11:05:00 | 854.75 | 2024-12-12 11:50:00 | 856.15 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-12-18 11:15:00 | 841.45 | 2024-12-18 11:35:00 | 843.17 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-27 09:35:00 | 811.10 | 2024-12-27 09:40:00 | 812.82 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-01-06 11:10:00 | 783.20 | 2025-01-06 11:30:00 | 785.12 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-07 10:20:00 | 776.50 | 2025-01-07 11:30:00 | 778.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-08 10:25:00 | 771.70 | 2025-01-08 10:40:00 | 769.05 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-01-08 10:25:00 | 771.70 | 2025-01-08 13:45:00 | 767.35 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2025-01-20 09:40:00 | 774.70 | 2025-01-20 11:05:00 | 777.84 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-20 09:40:00 | 774.70 | 2025-01-20 15:20:00 | 778.00 | TARGET_HIT | 0.50 | 0.43% |
| SELL | retest1 | 2025-01-22 11:15:00 | 748.95 | 2025-01-22 12:20:00 | 746.11 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-22 11:15:00 | 748.95 | 2025-01-22 14:55:00 | 748.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-27 09:40:00 | 747.85 | 2025-01-27 10:15:00 | 745.12 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-28 11:00:00 | 750.45 | 2025-01-28 11:50:00 | 752.54 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-29 11:10:00 | 758.10 | 2025-01-29 11:15:00 | 756.58 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-02-04 09:35:00 | 774.50 | 2025-02-04 10:10:00 | 772.25 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-02-06 11:05:00 | 763.60 | 2025-02-06 11:50:00 | 760.82 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-02-06 11:05:00 | 763.60 | 2025-02-06 14:15:00 | 762.00 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-02-12 09:40:00 | 724.30 | 2025-02-12 09:45:00 | 721.05 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-02-12 09:40:00 | 724.30 | 2025-02-12 10:35:00 | 724.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-14 10:35:00 | 727.00 | 2025-02-14 10:55:00 | 728.85 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-21 10:15:00 | 724.80 | 2025-02-21 11:30:00 | 726.37 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-02-25 10:45:00 | 714.20 | 2025-02-25 10:55:00 | 715.34 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-03-10 09:40:00 | 737.70 | 2025-03-10 09:55:00 | 735.99 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-03-11 11:00:00 | 733.50 | 2025-03-11 11:10:00 | 731.79 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-03-12 11:15:00 | 728.00 | 2025-03-12 11:25:00 | 725.94 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-03-12 11:15:00 | 728.00 | 2025-03-12 15:20:00 | 722.70 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2025-03-17 09:35:00 | 725.35 | 2025-03-17 09:50:00 | 723.50 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-03-17 09:35:00 | 725.35 | 2025-03-17 11:00:00 | 725.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:00:00 | 728.75 | 2025-03-18 10:25:00 | 730.65 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-03-18 10:00:00 | 728.75 | 2025-03-18 10:55:00 | 728.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-24 11:00:00 | 770.00 | 2025-03-24 11:05:00 | 768.36 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-03-27 11:10:00 | 773.30 | 2025-03-27 11:15:00 | 771.65 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-04-03 11:05:00 | 779.00 | 2025-04-03 11:15:00 | 777.53 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-04-09 11:00:00 | 753.25 | 2025-04-09 12:25:00 | 755.60 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-21 11:05:00 | 823.40 | 2025-04-21 11:35:00 | 821.17 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-04-22 10:00:00 | 821.90 | 2025-04-22 10:15:00 | 825.02 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-04-22 10:00:00 | 821.90 | 2025-04-22 13:30:00 | 827.00 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2025-04-25 10:00:00 | 797.60 | 2025-04-25 10:05:00 | 800.13 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-28 09:35:00 | 810.75 | 2025-04-28 09:40:00 | 807.96 | STOP_HIT | 1.00 | -0.34% |
