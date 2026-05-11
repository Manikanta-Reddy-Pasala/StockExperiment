# Indian Railway Catering And Tourism Corporation Ltd. (IRCTC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 565.50
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
| ENTRY1 | 89 |
| ENTRY2 | 0 |
| PARTIAL | 39 |
| TARGET_HIT | 18 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 71
- **Target hits / Stop hits / Partials:** 18 / 71 / 39
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 12.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 25 | 42.4% | 8 | 34 | 17 | 0.06% | 3.8% |
| BUY @ 2nd Alert (retest1) | 59 | 25 | 42.4% | 8 | 34 | 17 | 0.06% | 3.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 69 | 32 | 46.4% | 10 | 37 | 22 | 0.12% | 8.6% |
| SELL @ 2nd Alert (retest1) | 69 | 32 | 46.4% | 10 | 37 | 22 | 0.12% | 8.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 57 | 44.5% | 18 | 71 | 39 | 0.10% | 12.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:55:00 | 773.45 | 769.87 | 0.00 | ORB-long ORB[762.40,770.95] vol=3.6x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-14 11:15:00 | 776.51 | 772.18 | 0.00 | T1 1.5R @ 776.51 |
| Target hit | 2025-05-14 12:45:00 | 774.15 | 774.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2025-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:30:00 | 796.75 | 793.06 | 0.00 | ORB-long ORB[784.00,795.50] vol=2.2x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 09:35:00 | 799.72 | 795.11 | 0.00 | T1 1.5R @ 799.72 |
| Target hit | 2025-05-16 11:00:00 | 804.50 | 804.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2025-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 11:00:00 | 783.35 | 779.73 | 0.00 | ORB-long ORB[776.60,782.35] vol=1.8x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 12:00:00 | 786.06 | 781.07 | 0.00 | T1 1.5R @ 786.06 |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 783.35 | 782.51 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 787.55 | 790.53 | 0.00 | ORB-short ORB[788.10,794.95] vol=1.6x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-05-27 10:05:00 | 789.29 | 789.44 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 09:30:00 | 789.50 | 793.16 | 0.00 | ORB-short ORB[791.50,796.75] vol=2.5x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-05-28 09:35:00 | 791.17 | 792.86 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 09:35:00 | 769.80 | 776.65 | 0.00 | ORB-short ORB[774.90,781.60] vol=1.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 09:45:00 | 766.27 | 774.46 | 0.00 | T1 1.5R @ 766.27 |
| Target hit | 2025-05-30 15:20:00 | 755.05 | 762.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2025-06-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:40:00 | 785.00 | 782.75 | 0.00 | ORB-long ORB[781.00,784.50] vol=2.1x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 09:55:00 | 787.47 | 784.99 | 0.00 | T1 1.5R @ 787.47 |
| Target hit | 2025-06-09 10:10:00 | 785.75 | 785.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2025-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 09:30:00 | 793.30 | 791.94 | 0.00 | ORB-long ORB[790.25,792.70] vol=1.6x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 09:35:00 | 795.34 | 792.85 | 0.00 | T1 1.5R @ 795.34 |
| Stop hit — per-position SL triggered | 2025-06-11 09:40:00 | 793.30 | 792.87 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:10:00 | 780.85 | 783.84 | 0.00 | ORB-short ORB[783.00,788.45] vol=1.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:35:00 | 778.64 | 783.53 | 0.00 | T1 1.5R @ 778.64 |
| Target hit | 2025-06-12 15:20:00 | 769.90 | 776.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 759.80 | 764.99 | 0.00 | ORB-short ORB[763.00,771.80] vol=1.5x ATR=1.97 |
| Stop hit — per-position SL triggered | 2025-06-16 10:05:00 | 761.77 | 763.04 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 09:50:00 | 757.80 | 759.55 | 0.00 | ORB-short ORB[758.00,761.95] vol=1.8x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:05:00 | 755.48 | 759.14 | 0.00 | T1 1.5R @ 755.48 |
| Stop hit — per-position SL triggered | 2025-06-19 10:10:00 | 757.80 | 759.05 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 11:15:00 | 755.00 | 751.05 | 0.00 | ORB-long ORB[743.85,751.95] vol=2.4x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-06-20 12:20:00 | 753.50 | 751.89 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:05:00 | 756.00 | 752.32 | 0.00 | ORB-long ORB[746.50,753.75] vol=2.4x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 13:30:00 | 758.89 | 754.13 | 0.00 | T1 1.5R @ 758.89 |
| Stop hit — per-position SL triggered | 2025-06-23 13:55:00 | 756.00 | 754.35 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:25:00 | 769.85 | 765.32 | 0.00 | ORB-long ORB[761.70,765.00] vol=1.7x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-06-24 10:30:00 | 768.31 | 765.72 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:55:00 | 764.85 | 767.66 | 0.00 | ORB-short ORB[767.90,771.40] vol=1.9x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 11:20:00 | 763.02 | 767.19 | 0.00 | T1 1.5R @ 763.02 |
| Stop hit — per-position SL triggered | 2025-06-26 11:30:00 | 764.85 | 767.08 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 11:00:00 | 785.45 | 781.20 | 0.00 | ORB-long ORB[773.20,779.30] vol=2.1x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:35:00 | 787.64 | 782.24 | 0.00 | T1 1.5R @ 787.64 |
| Stop hit — per-position SL triggered | 2025-06-27 12:10:00 | 785.45 | 783.30 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 11:00:00 | 771.65 | 776.71 | 0.00 | ORB-short ORB[778.00,785.25] vol=2.2x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 773.09 | 776.47 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 784.90 | 782.92 | 0.00 | ORB-long ORB[778.50,784.00] vol=2.9x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-07-04 09:35:00 | 783.33 | 782.94 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:05:00 | 784.10 | 782.21 | 0.00 | ORB-long ORB[775.05,783.00] vol=1.6x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:25:00 | 786.27 | 782.91 | 0.00 | T1 1.5R @ 786.27 |
| Stop hit — per-position SL triggered | 2025-07-07 10:50:00 | 784.10 | 783.13 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:30:00 | 789.80 | 788.79 | 0.00 | ORB-long ORB[786.30,789.00] vol=2.1x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-07-10 09:40:00 | 788.27 | 788.67 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:00:00 | 780.00 | 781.80 | 0.00 | ORB-short ORB[780.25,783.70] vol=1.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:10:00 | 778.25 | 781.05 | 0.00 | T1 1.5R @ 778.25 |
| Target hit | 2025-07-11 15:20:00 | 773.40 | 776.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-07-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 11:00:00 | 781.10 | 778.81 | 0.00 | ORB-long ORB[775.20,780.60] vol=1.8x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 779.86 | 778.94 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:05:00 | 769.50 | 772.07 | 0.00 | ORB-short ORB[771.75,775.70] vol=1.7x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 767.56 | 770.80 | 0.00 | T1 1.5R @ 767.56 |
| Target hit | 2025-07-18 15:20:00 | 765.15 | 767.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2025-07-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:05:00 | 768.60 | 763.85 | 0.00 | ORB-long ORB[759.00,768.45] vol=1.8x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-07-21 11:40:00 | 767.46 | 764.40 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:40:00 | 751.75 | 753.56 | 0.00 | ORB-short ORB[752.25,757.00] vol=3.7x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 749.56 | 752.22 | 0.00 | T1 1.5R @ 749.56 |
| Target hit | 2025-07-25 15:05:00 | 747.55 | 747.39 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — BUY (started 2025-07-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:25:00 | 740.25 | 737.68 | 0.00 | ORB-long ORB[735.15,738.20] vol=1.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-07-30 10:35:00 | 738.93 | 738.25 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:50:00 | 723.55 | 726.96 | 0.00 | ORB-short ORB[726.45,731.75] vol=1.9x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-08-05 11:10:00 | 725.27 | 725.58 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:35:00 | 734.90 | 738.16 | 0.00 | ORB-short ORB[736.00,743.00] vol=2.4x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:10:00 | 730.90 | 736.76 | 0.00 | T1 1.5R @ 730.90 |
| Stop hit — per-position SL triggered | 2025-08-06 10:20:00 | 734.90 | 736.66 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:35:00 | 720.50 | 724.24 | 0.00 | ORB-short ORB[724.00,728.60] vol=1.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-08-07 10:50:00 | 722.03 | 723.75 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:00:00 | 723.55 | 731.61 | 0.00 | ORB-short ORB[727.70,737.75] vol=1.8x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-08-14 10:10:00 | 725.71 | 730.72 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:55:00 | 725.50 | 727.37 | 0.00 | ORB-short ORB[725.60,731.80] vol=2.5x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-08-18 10:00:00 | 727.33 | 727.38 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:15:00 | 732.10 | 729.63 | 0.00 | ORB-long ORB[726.40,731.65] vol=1.7x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-08-20 11:10:00 | 730.85 | 730.42 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 721.30 | 723.38 | 0.00 | ORB-short ORB[722.60,727.00] vol=2.3x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-08-22 10:20:00 | 722.58 | 722.49 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:00:00 | 721.00 | 718.19 | 0.00 | ORB-long ORB[714.10,720.15] vol=4.0x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 719.69 | 718.40 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:30:00 | 723.30 | 722.17 | 0.00 | ORB-long ORB[717.65,722.10] vol=2.1x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-09-11 10:45:00 | 722.09 | 722.21 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:45:00 | 730.30 | 727.60 | 0.00 | ORB-long ORB[722.55,729.75] vol=2.0x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-09-15 09:50:00 | 728.92 | 727.71 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:50:00 | 735.35 | 734.19 | 0.00 | ORB-long ORB[731.30,734.90] vol=1.5x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-09-18 10:10:00 | 734.05 | 734.30 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:10:00 | 721.75 | 724.85 | 0.00 | ORB-short ORB[726.15,730.00] vol=2.2x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-09-23 12:00:00 | 723.13 | 724.42 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:55:00 | 708.85 | 709.85 | 0.00 | ORB-short ORB[708.95,713.45] vol=2.0x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:30:00 | 706.66 | 709.54 | 0.00 | T1 1.5R @ 706.66 |
| Stop hit — per-position SL triggered | 2025-09-26 12:50:00 | 708.85 | 709.04 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 10:55:00 | 700.05 | 703.23 | 0.00 | ORB-short ORB[701.10,707.00] vol=1.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-10-01 11:55:00 | 701.44 | 701.89 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:55:00 | 706.60 | 710.36 | 0.00 | ORB-short ORB[709.30,716.90] vol=1.6x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:05:00 | 704.43 | 709.93 | 0.00 | T1 1.5R @ 704.43 |
| Stop hit — per-position SL triggered | 2025-10-08 12:00:00 | 706.60 | 708.59 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:30:00 | 716.65 | 713.89 | 0.00 | ORB-long ORB[708.65,713.20] vol=3.6x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-10-10 10:35:00 | 715.42 | 713.93 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 705.65 | 708.72 | 0.00 | ORB-short ORB[709.00,712.65] vol=1.7x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:45:00 | 704.13 | 707.48 | 0.00 | T1 1.5R @ 704.13 |
| Target hit | 2025-10-14 15:20:00 | 704.60 | 704.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-10-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:55:00 | 712.80 | 710.13 | 0.00 | ORB-long ORB[704.15,708.95] vol=2.2x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:05:00 | 714.56 | 710.59 | 0.00 | T1 1.5R @ 714.56 |
| Stop hit — per-position SL triggered | 2025-10-15 11:35:00 | 712.80 | 711.20 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:15:00 | 720.00 | 717.60 | 0.00 | ORB-long ORB[715.40,719.90] vol=2.4x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-10-17 10:40:00 | 718.62 | 717.93 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:55:00 | 726.00 | 722.52 | 0.00 | ORB-long ORB[719.80,723.00] vol=3.1x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-10-20 11:30:00 | 724.34 | 724.93 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:45:00 | 724.95 | 722.00 | 0.00 | ORB-long ORB[716.10,721.00] vol=1.9x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-10-24 11:50:00 | 723.52 | 723.66 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 730.30 | 727.19 | 0.00 | ORB-long ORB[720.15,726.80] vol=2.0x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:00:00 | 732.79 | 728.57 | 0.00 | T1 1.5R @ 732.79 |
| Target hit | 2025-10-29 12:25:00 | 730.60 | 730.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — SELL (started 2025-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 11:10:00 | 726.90 | 729.55 | 0.00 | ORB-short ORB[727.60,733.00] vol=4.9x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-10-30 11:30:00 | 728.07 | 728.90 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 11:15:00 | 721.40 | 719.99 | 0.00 | ORB-long ORB[716.10,720.45] vol=2.2x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-11-03 11:50:00 | 720.29 | 720.09 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:30:00 | 721.15 | 722.50 | 0.00 | ORB-short ORB[721.55,725.00] vol=2.9x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:30:00 | 719.43 | 721.88 | 0.00 | T1 1.5R @ 719.43 |
| Target hit | 2025-11-04 15:20:00 | 719.20 | 719.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:15:00 | 708.95 | 712.84 | 0.00 | ORB-short ORB[714.35,717.65] vol=2.4x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:40:00 | 707.08 | 712.28 | 0.00 | T1 1.5R @ 707.08 |
| Stop hit — per-position SL triggered | 2025-11-06 11:45:00 | 708.95 | 712.23 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:30:00 | 705.75 | 704.40 | 0.00 | ORB-long ORB[701.00,705.20] vol=1.5x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-11-10 10:45:00 | 704.56 | 704.44 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:55:00 | 715.30 | 714.88 | 0.00 | ORB-long ORB[711.10,714.95] vol=1.5x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 11:45:00 | 717.01 | 715.04 | 0.00 | T1 1.5R @ 717.01 |
| Target hit | 2025-11-12 14:25:00 | 715.40 | 715.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — BUY (started 2025-11-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:35:00 | 715.15 | 712.53 | 0.00 | ORB-long ORB[706.00,715.05] vol=3.0x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-11-17 13:40:00 | 713.51 | 713.64 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:45:00 | 703.65 | 707.20 | 0.00 | ORB-short ORB[706.40,713.05] vol=2.3x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-11-18 11:50:00 | 705.11 | 705.14 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:15:00 | 694.70 | 696.94 | 0.00 | ORB-short ORB[695.30,700.90] vol=1.8x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:55:00 | 692.93 | 696.08 | 0.00 | T1 1.5R @ 692.93 |
| Stop hit — per-position SL triggered | 2025-11-21 11:50:00 | 694.70 | 695.77 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-11-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 10:55:00 | 682.75 | 683.12 | 0.00 | ORB-short ORB[682.80,686.05] vol=3.0x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-11-25 11:40:00 | 683.80 | 683.12 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-11-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:55:00 | 685.00 | 682.82 | 0.00 | ORB-long ORB[677.75,683.75] vol=2.3x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 11:15:00 | 686.36 | 683.12 | 0.00 | T1 1.5R @ 686.36 |
| Target hit | 2025-11-26 15:20:00 | 687.65 | 686.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:15:00 | 693.35 | 691.37 | 0.00 | ORB-long ORB[689.95,693.15] vol=2.3x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-11-27 10:30:00 | 692.17 | 691.76 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:35:00 | 682.50 | 683.31 | 0.00 | ORB-short ORB[682.65,685.50] vol=2.1x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:50:00 | 680.89 | 682.98 | 0.00 | T1 1.5R @ 680.89 |
| Stop hit — per-position SL triggered | 2025-12-02 10:35:00 | 682.50 | 682.41 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:35:00 | 677.20 | 679.24 | 0.00 | ORB-short ORB[678.10,682.60] vol=1.9x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:00:00 | 675.64 | 678.01 | 0.00 | T1 1.5R @ 675.64 |
| Target hit | 2025-12-03 15:10:00 | 675.10 | 674.11 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — SELL (started 2025-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:40:00 | 671.40 | 672.88 | 0.00 | ORB-short ORB[672.30,677.00] vol=1.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 672.67 | 672.52 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 667.95 | 670.79 | 0.00 | ORB-short ORB[669.00,673.75] vol=1.9x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-12-10 11:15:00 | 669.05 | 670.54 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 674.20 | 672.18 | 0.00 | ORB-long ORB[669.05,673.45] vol=1.8x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-12-12 10:00:00 | 672.91 | 672.82 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:35:00 | 671.65 | 670.68 | 0.00 | ORB-long ORB[669.35,671.60] vol=1.6x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-12-17 09:40:00 | 670.45 | 670.68 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:50:00 | 679.00 | 681.38 | 0.00 | ORB-short ORB[679.20,684.80] vol=4.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-12-24 10:25:00 | 680.30 | 680.97 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:30:00 | 689.05 | 685.96 | 0.00 | ORB-long ORB[678.80,686.00] vol=6.7x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:35:00 | 691.43 | 688.12 | 0.00 | T1 1.5R @ 691.43 |
| Stop hit — per-position SL triggered | 2025-12-26 09:40:00 | 689.05 | 688.95 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-12-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:45:00 | 686.15 | 689.72 | 0.00 | ORB-short ORB[687.65,696.80] vol=1.7x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:00:00 | 683.32 | 688.27 | 0.00 | T1 1.5R @ 683.32 |
| Stop hit — per-position SL triggered | 2025-12-30 10:10:00 | 686.15 | 687.78 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:50:00 | 668.50 | 669.94 | 0.00 | ORB-short ORB[668.80,673.50] vol=1.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 669.84 | 669.83 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 09:30:00 | 667.80 | 670.20 | 0.00 | ORB-short ORB[669.00,673.70] vol=1.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-01-08 09:40:00 | 669.19 | 669.37 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:35:00 | 648.60 | 652.88 | 0.00 | ORB-short ORB[652.00,658.15] vol=1.9x ATR=2.14 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 650.74 | 651.67 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-01-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:35:00 | 633.50 | 634.81 | 0.00 | ORB-short ORB[635.40,640.70] vol=2.0x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 14:00:00 | 629.96 | 633.61 | 0.00 | T1 1.5R @ 629.96 |
| Target hit | 2026-01-13 15:20:00 | 629.80 | 632.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2026-01-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 09:35:00 | 610.20 | 614.10 | 0.00 | ORB-short ORB[611.40,618.25] vol=1.9x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:45:00 | 606.69 | 612.30 | 0.00 | T1 1.5R @ 606.69 |
| Stop hit — per-position SL triggered | 2026-01-21 09:50:00 | 610.20 | 612.14 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:05:00 | 615.70 | 619.09 | 0.00 | ORB-short ORB[621.40,626.00] vol=2.2x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 12:50:00 | 612.95 | 617.53 | 0.00 | T1 1.5R @ 612.95 |
| Stop hit — per-position SL triggered | 2026-01-29 13:15:00 | 615.70 | 617.33 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-01-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:40:00 | 618.60 | 615.20 | 0.00 | ORB-long ORB[608.80,615.30] vol=5.3x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 10:55:00 | 621.53 | 615.65 | 0.00 | T1 1.5R @ 621.53 |
| Stop hit — per-position SL triggered | 2026-01-30 11:20:00 | 618.60 | 616.08 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 631.00 | 629.88 | 0.00 | ORB-long ORB[623.05,630.35] vol=1.7x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 629.06 | 629.87 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 10:45:00 | 601.75 | 605.77 | 0.00 | ORB-short ORB[602.85,610.00] vol=1.9x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:55:00 | 598.58 | 604.83 | 0.00 | T1 1.5R @ 598.58 |
| Stop hit — per-position SL triggered | 2026-02-02 11:40:00 | 601.75 | 603.05 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:40:00 | 614.35 | 616.38 | 0.00 | ORB-short ORB[615.50,620.95] vol=2.6x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-02-06 09:45:00 | 615.76 | 616.31 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 627.25 | 625.31 | 0.00 | ORB-long ORB[621.80,627.00] vol=3.0x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-02-13 09:35:00 | 625.51 | 625.32 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:40:00 | 607.40 | 610.58 | 0.00 | ORB-short ORB[609.15,617.35] vol=1.8x ATR=2.27 |
| Stop hit — per-position SL triggered | 2026-02-16 10:00:00 | 609.67 | 609.92 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 627.95 | 625.84 | 0.00 | ORB-long ORB[619.30,627.85] vol=1.8x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:50:00 | 630.32 | 627.14 | 0.00 | T1 1.5R @ 630.32 |
| Target hit | 2026-02-19 15:00:00 | 631.70 | 632.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 83 — SELL (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 542.20 | 544.33 | 0.00 | ORB-short ORB[543.00,550.00] vol=1.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 544.18 | 543.73 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-03-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:45:00 | 533.80 | 528.91 | 0.00 | ORB-long ORB[523.50,531.20] vol=2.0x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:55:00 | 536.29 | 529.80 | 0.00 | T1 1.5R @ 536.29 |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 533.80 | 530.27 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 524.80 | 527.45 | 0.00 | ORB-short ORB[526.40,532.00] vol=2.3x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-03-20 10:10:00 | 526.55 | 525.99 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-04-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:50:00 | 550.00 | 548.34 | 0.00 | ORB-long ORB[545.15,549.80] vol=1.6x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:05:00 | 553.18 | 550.68 | 0.00 | T1 1.5R @ 553.18 |
| Target hit | 2026-04-15 15:20:00 | 553.40 | 552.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — SELL (started 2026-04-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:50:00 | 553.00 | 555.17 | 0.00 | ORB-short ORB[554.50,558.10] vol=1.7x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 14:25:00 | 551.24 | 553.98 | 0.00 | T1 1.5R @ 551.24 |
| Target hit | 2026-04-23 15:20:00 | 551.00 | 553.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2026-05-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:55:00 | 551.50 | 546.61 | 0.00 | ORB-long ORB[542.05,546.85] vol=2.1x ATR=1.62 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 549.88 | 547.95 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-05-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:20:00 | 569.45 | 573.08 | 0.00 | ORB-short ORB[571.25,578.05] vol=2.2x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-05-07 11:50:00 | 571.20 | 570.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:55:00 | 773.45 | 2025-05-14 11:15:00 | 776.51 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-05-14 09:55:00 | 773.45 | 2025-05-14 12:45:00 | 774.15 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-05-16 09:30:00 | 796.75 | 2025-05-16 09:35:00 | 799.72 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-05-16 09:30:00 | 796.75 | 2025-05-16 11:00:00 | 804.50 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2025-05-23 11:00:00 | 783.35 | 2025-05-23 12:00:00 | 786.06 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-05-23 11:00:00 | 783.35 | 2025-05-23 14:15:00 | 783.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-27 09:30:00 | 787.55 | 2025-05-27 10:05:00 | 789.29 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-28 09:30:00 | 789.50 | 2025-05-28 09:35:00 | 791.17 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-30 09:35:00 | 769.80 | 2025-05-30 09:45:00 | 766.27 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-05-30 09:35:00 | 769.80 | 2025-05-30 15:20:00 | 755.05 | TARGET_HIT | 0.50 | 1.92% |
| BUY | retest1 | 2025-06-09 09:40:00 | 785.00 | 2025-06-09 09:55:00 | 787.47 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-06-09 09:40:00 | 785.00 | 2025-06-09 10:10:00 | 785.75 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-06-11 09:30:00 | 793.30 | 2025-06-11 09:35:00 | 795.34 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-06-11 09:30:00 | 793.30 | 2025-06-11 09:40:00 | 793.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-12 11:10:00 | 780.85 | 2025-06-12 11:35:00 | 778.64 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-06-12 11:10:00 | 780.85 | 2025-06-12 15:20:00 | 769.90 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2025-06-16 09:30:00 | 759.80 | 2025-06-16 10:05:00 | 761.77 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-19 09:50:00 | 757.80 | 2025-06-19 10:05:00 | 755.48 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-06-19 09:50:00 | 757.80 | 2025-06-19 10:10:00 | 757.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-20 11:15:00 | 755.00 | 2025-06-20 12:20:00 | 753.50 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-23 11:05:00 | 756.00 | 2025-06-23 13:30:00 | 758.89 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-06-23 11:05:00 | 756.00 | 2025-06-23 13:55:00 | 756.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-24 10:25:00 | 769.85 | 2025-06-24 10:30:00 | 768.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-06-26 10:55:00 | 764.85 | 2025-06-26 11:20:00 | 763.02 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-06-26 10:55:00 | 764.85 | 2025-06-26 11:30:00 | 764.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 11:00:00 | 785.45 | 2025-06-27 11:35:00 | 787.64 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-06-27 11:00:00 | 785.45 | 2025-06-27 12:10:00 | 785.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 11:00:00 | 771.65 | 2025-07-01 11:15:00 | 773.09 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-07-04 09:30:00 | 784.90 | 2025-07-04 09:35:00 | 783.33 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-07 10:05:00 | 784.10 | 2025-07-07 10:25:00 | 786.27 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-07-07 10:05:00 | 784.10 | 2025-07-07 10:50:00 | 784.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-10 09:30:00 | 789.80 | 2025-07-10 09:40:00 | 788.27 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-11 10:00:00 | 780.00 | 2025-07-11 10:10:00 | 778.25 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-11 10:00:00 | 780.00 | 2025-07-11 15:20:00 | 773.40 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2025-07-17 11:00:00 | 781.10 | 2025-07-17 11:15:00 | 779.86 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-18 10:05:00 | 769.50 | 2025-07-18 10:15:00 | 767.56 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-18 10:05:00 | 769.50 | 2025-07-18 15:20:00 | 765.15 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2025-07-21 11:05:00 | 768.60 | 2025-07-21 11:40:00 | 767.46 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-07-25 09:40:00 | 751.75 | 2025-07-25 10:15:00 | 749.56 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-07-25 09:40:00 | 751.75 | 2025-07-25 15:05:00 | 747.55 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2025-07-30 10:25:00 | 740.25 | 2025-07-30 10:35:00 | 738.93 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-05 09:50:00 | 723.55 | 2025-08-05 11:10:00 | 725.27 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-06 09:35:00 | 734.90 | 2025-08-06 10:10:00 | 730.90 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-08-06 09:35:00 | 734.90 | 2025-08-06 10:20:00 | 734.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 10:35:00 | 720.50 | 2025-08-07 10:50:00 | 722.03 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-14 10:00:00 | 723.55 | 2025-08-14 10:10:00 | 725.71 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-08-18 09:55:00 | 725.50 | 2025-08-18 10:00:00 | 727.33 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-20 10:15:00 | 732.10 | 2025-08-20 11:10:00 | 730.85 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-08-22 09:30:00 | 721.30 | 2025-08-22 10:20:00 | 722.58 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-10 11:00:00 | 721.00 | 2025-09-10 11:15:00 | 719.69 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-11 10:30:00 | 723.30 | 2025-09-11 10:45:00 | 722.09 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-15 09:45:00 | 730.30 | 2025-09-15 09:50:00 | 728.92 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-18 09:50:00 | 735.35 | 2025-09-18 10:10:00 | 734.05 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-23 11:10:00 | 721.75 | 2025-09-23 12:00:00 | 723.13 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-26 10:55:00 | 708.85 | 2025-09-26 11:30:00 | 706.66 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-09-26 10:55:00 | 708.85 | 2025-09-26 12:50:00 | 708.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-01 10:55:00 | 700.05 | 2025-10-01 11:55:00 | 701.44 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-08 10:55:00 | 706.60 | 2025-10-08 11:05:00 | 704.43 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-08 10:55:00 | 706.60 | 2025-10-08 12:00:00 | 706.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 10:30:00 | 716.65 | 2025-10-10 10:35:00 | 715.42 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-14 11:15:00 | 705.65 | 2025-10-14 11:45:00 | 704.13 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-10-14 11:15:00 | 705.65 | 2025-10-14 15:20:00 | 704.60 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-10-15 10:55:00 | 712.80 | 2025-10-15 11:05:00 | 714.56 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-10-15 10:55:00 | 712.80 | 2025-10-15 11:35:00 | 712.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 10:15:00 | 720.00 | 2025-10-17 10:40:00 | 718.62 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-20 09:55:00 | 726.00 | 2025-10-20 11:30:00 | 724.34 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-24 10:45:00 | 724.95 | 2025-10-24 11:50:00 | 723.52 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-10-29 10:25:00 | 730.30 | 2025-10-29 11:00:00 | 732.79 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-10-29 10:25:00 | 730.30 | 2025-10-29 12:25:00 | 730.60 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2025-10-30 11:10:00 | 726.90 | 2025-10-30 11:30:00 | 728.07 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-11-03 11:15:00 | 721.40 | 2025-11-03 11:50:00 | 720.29 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-11-04 10:30:00 | 721.15 | 2025-11-04 11:30:00 | 719.43 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-11-04 10:30:00 | 721.15 | 2025-11-04 15:20:00 | 719.20 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-11-06 11:15:00 | 708.95 | 2025-11-06 11:40:00 | 707.08 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-11-06 11:15:00 | 708.95 | 2025-11-06 11:45:00 | 708.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 10:30:00 | 705.75 | 2025-11-10 10:45:00 | 704.56 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-12 10:55:00 | 715.30 | 2025-11-12 11:45:00 | 717.01 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-11-12 10:55:00 | 715.30 | 2025-11-12 14:25:00 | 715.40 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2025-11-17 09:35:00 | 715.15 | 2025-11-17 13:40:00 | 713.51 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-18 09:45:00 | 703.65 | 2025-11-18 11:50:00 | 705.11 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-21 10:15:00 | 694.70 | 2025-11-21 10:55:00 | 692.93 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-11-21 10:15:00 | 694.70 | 2025-11-21 11:50:00 | 694.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-25 10:55:00 | 682.75 | 2025-11-25 11:40:00 | 683.80 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-11-26 10:55:00 | 685.00 | 2025-11-26 11:15:00 | 686.36 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2025-11-26 10:55:00 | 685.00 | 2025-11-26 15:20:00 | 687.65 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2025-11-27 10:15:00 | 693.35 | 2025-11-27 10:30:00 | 692.17 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-12-02 09:35:00 | 682.50 | 2025-12-02 09:50:00 | 680.89 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-02 09:35:00 | 682.50 | 2025-12-02 10:35:00 | 682.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 09:35:00 | 677.20 | 2025-12-03 10:00:00 | 675.64 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-12-03 09:35:00 | 677.20 | 2025-12-03 15:10:00 | 675.10 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-05 09:40:00 | 671.40 | 2025-12-05 10:00:00 | 672.67 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-10 10:55:00 | 667.95 | 2025-12-10 11:15:00 | 669.05 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-12 09:30:00 | 674.20 | 2025-12-12 10:00:00 | 672.91 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-17 09:35:00 | 671.65 | 2025-12-17 09:40:00 | 670.45 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-24 09:50:00 | 679.00 | 2025-12-24 10:25:00 | 680.30 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-26 09:30:00 | 689.05 | 2025-12-26 09:35:00 | 691.43 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-26 09:30:00 | 689.05 | 2025-12-26 09:40:00 | 689.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 09:45:00 | 686.15 | 2025-12-30 10:00:00 | 683.32 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-12-30 09:45:00 | 686.15 | 2025-12-30 10:10:00 | 686.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 09:50:00 | 668.50 | 2026-01-07 10:15:00 | 669.84 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-08 09:30:00 | 667.80 | 2026-01-08 09:40:00 | 669.19 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-09 09:35:00 | 648.60 | 2026-01-09 09:45:00 | 650.74 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-13 10:35:00 | 633.50 | 2026-01-13 14:00:00 | 629.96 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-01-13 10:35:00 | 633.50 | 2026-01-13 15:20:00 | 629.80 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2026-01-21 09:35:00 | 610.20 | 2026-01-21 09:45:00 | 606.69 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-01-21 09:35:00 | 610.20 | 2026-01-21 09:50:00 | 610.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 11:05:00 | 615.70 | 2026-01-29 12:50:00 | 612.95 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-29 11:05:00 | 615.70 | 2026-01-29 13:15:00 | 615.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-30 10:40:00 | 618.60 | 2026-01-30 10:55:00 | 621.53 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-01-30 10:40:00 | 618.60 | 2026-01-30 11:20:00 | 618.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:10:00 | 631.00 | 2026-02-01 11:15:00 | 629.06 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-02 10:45:00 | 601.75 | 2026-02-02 10:55:00 | 598.58 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-02 10:45:00 | 601.75 | 2026-02-02 11:40:00 | 601.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-06 09:40:00 | 614.35 | 2026-02-06 09:45:00 | 615.76 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-13 09:30:00 | 627.25 | 2026-02-13 09:35:00 | 625.51 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-16 09:40:00 | 607.40 | 2026-02-16 10:00:00 | 609.67 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-19 09:30:00 | 627.95 | 2026-02-19 09:50:00 | 630.32 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-19 09:30:00 | 627.95 | 2026-02-19 15:00:00 | 631.70 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2026-03-10 09:35:00 | 542.20 | 2026-03-10 10:15:00 | 544.18 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-18 10:45:00 | 533.80 | 2026-03-18 10:55:00 | 536.29 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-18 10:45:00 | 533.80 | 2026-03-18 11:15:00 | 533.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 09:35:00 | 524.80 | 2026-03-20 10:10:00 | 526.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-15 09:50:00 | 550.00 | 2026-04-15 10:05:00 | 553.18 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-15 09:50:00 | 550.00 | 2026-04-15 15:20:00 | 553.40 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2026-04-23 10:50:00 | 553.00 | 2026-04-23 14:25:00 | 551.24 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-04-23 10:50:00 | 553.00 | 2026-04-23 15:20:00 | 551.00 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-05-04 09:55:00 | 551.50 | 2026-05-04 10:20:00 | 549.88 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-05-07 10:20:00 | 569.45 | 2026-05-07 11:50:00 | 571.20 | STOP_HIT | 1.00 | -0.31% |
