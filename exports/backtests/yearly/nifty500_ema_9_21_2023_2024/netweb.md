# Netweb Technologies India Ltd. (NETWEB)

## Backtest Summary

- **Window:** 2023-07-27 09:15:00 → 2026-05-08 15:15:00 (4792 bars)
- **Last close:** 4424.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 193 |
| ALERT1 | 147 |
| ALERT2 | 146 |
| ALERT2_SKIP | 89 |
| ALERT3 | 303 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 114 |
| PARTIAL | 17 |
| TARGET_HIT | 17 |
| STOP_HIT | 106 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 53 / 87
- **Target hits / Stop hits / Partials:** 17 / 106 / 17
- **Avg / median % per leg:** 0.53% / -0.94%
- **Sum % (uncompounded):** 73.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 19 | 27.9% | 14 | 53 | 1 | 0.44% | 29.7% |
| BUY @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 9 | 1 | -1.98% | -19.8% |
| BUY @ 3rd Alert (retest2) | 58 | 17 | 29.3% | 14 | 44 | 0 | 0.85% | 49.5% |
| SELL (all) | 72 | 34 | 47.2% | 3 | 53 | 16 | 0.61% | 44.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 34 | 47.2% | 3 | 53 | 16 | 0.61% | 44.3% |
| retest1 (combined) | 10 | 2 | 20.0% | 0 | 9 | 1 | -1.98% | -19.8% |
| retest2 (combined) | 130 | 51 | 39.2% | 17 | 97 | 16 | 0.72% | 93.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 886.10 | 878.21 | 877.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 11:15:00 | 893.10 | 881.19 | 879.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 14:15:00 | 891.15 | 893.75 | 889.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 15:15:00 | 883.00 | 891.60 | 888.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 15:15:00 | 883.00 | 891.60 | 888.51 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 868.55 | 884.97 | 885.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 11:15:00 | 862.90 | 880.56 | 883.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-10 12:15:00 | 864.20 | 855.48 | 862.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 12:15:00 | 864.20 | 855.48 | 862.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 12:15:00 | 864.20 | 855.48 | 862.41 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 10:15:00 | 876.00 | 865.15 | 864.72 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 10:15:00 | 854.00 | 866.08 | 866.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 15:15:00 | 848.00 | 856.40 | 860.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 10:15:00 | 824.95 | 824.20 | 833.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 794.70 | 813.89 | 823.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 794.70 | 813.89 | 823.80 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 840.00 | 811.13 | 808.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 13:15:00 | 887.50 | 837.82 | 822.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 14:15:00 | 869.80 | 873.00 | 854.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 15:15:00 | 861.00 | 870.60 | 854.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 861.00 | 870.60 | 854.79 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 13:15:00 | 849.95 | 857.99 | 858.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 15:15:00 | 844.95 | 853.79 | 856.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 853.15 | 846.36 | 850.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 853.15 | 846.36 | 850.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 853.15 | 846.36 | 850.26 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 11:15:00 | 851.65 | 844.85 | 844.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 10:15:00 | 858.00 | 848.83 | 846.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 13:15:00 | 848.30 | 849.78 | 847.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 13:15:00 | 848.30 | 849.78 | 847.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 848.30 | 849.78 | 847.43 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-09-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 12:15:00 | 844.25 | 846.21 | 846.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 10:15:00 | 840.55 | 844.22 | 845.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 13:15:00 | 843.80 | 843.56 | 844.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 13:15:00 | 843.80 | 843.56 | 844.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 843.80 | 843.56 | 844.64 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 13:15:00 | 845.70 | 844.33 | 844.22 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 831.00 | 842.15 | 843.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 819.40 | 833.46 | 837.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-15 10:15:00 | 819.75 | 819.29 | 823.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 11:15:00 | 810.50 | 810.83 | 814.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 810.50 | 810.83 | 814.58 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 11:15:00 | 820.00 | 809.06 | 807.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 12:15:00 | 822.10 | 811.67 | 808.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 15:15:00 | 812.00 | 813.06 | 810.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 12:15:00 | 809.10 | 813.00 | 811.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 809.10 | 813.00 | 811.29 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-10-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 15:15:00 | 824.20 | 827.51 | 827.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 10:15:00 | 822.05 | 825.93 | 826.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 815.60 | 813.24 | 816.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 815.60 | 813.24 | 816.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 815.60 | 813.24 | 816.64 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 851.60 | 824.30 | 820.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 867.75 | 838.26 | 827.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 860.00 | 873.36 | 861.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 860.00 | 873.36 | 861.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 860.00 | 873.36 | 861.84 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 875.15 | 878.49 | 878.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 13:15:00 | 873.90 | 877.57 | 878.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 13:15:00 | 887.30 | 873.31 | 874.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 13:15:00 | 887.30 | 873.31 | 874.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 887.30 | 873.31 | 874.78 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 14:15:00 | 897.80 | 878.21 | 876.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-20 09:15:00 | 905.35 | 885.20 | 880.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 12:15:00 | 884.00 | 887.13 | 882.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 13:15:00 | 892.70 | 888.25 | 883.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 13:15:00 | 892.70 | 888.25 | 883.62 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 866.00 | 879.14 | 880.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 856.50 | 874.61 | 878.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-30 09:15:00 | 760.55 | 755.39 | 771.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 762.00 | 754.01 | 758.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 762.00 | 754.01 | 758.54 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 831.00 | 771.33 | 764.70 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 14:15:00 | 784.80 | 789.53 | 789.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 15:15:00 | 779.00 | 787.42 | 788.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 785.00 | 779.92 | 783.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 785.00 | 779.92 | 783.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 785.00 | 779.92 | 783.07 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 13:15:00 | 787.60 | 777.55 | 776.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 792.15 | 783.83 | 780.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 14:15:00 | 815.00 | 817.89 | 809.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 12:15:00 | 818.20 | 822.50 | 819.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 12:15:00 | 818.20 | 822.50 | 819.25 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 810.20 | 817.21 | 817.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 12:15:00 | 809.00 | 815.57 | 817.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 14:15:00 | 820.00 | 815.37 | 816.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 14:15:00 | 820.00 | 815.37 | 816.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 820.00 | 815.37 | 816.70 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-11-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 10:15:00 | 820.00 | 817.42 | 817.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 12:15:00 | 821.60 | 818.87 | 818.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 14:15:00 | 819.00 | 819.79 | 818.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 15:15:00 | 822.00 | 820.23 | 819.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 15:15:00 | 822.00 | 820.23 | 819.02 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 11:15:00 | 816.75 | 818.24 | 818.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 813.25 | 817.24 | 817.83 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 879.00 | 827.46 | 822.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 893.25 | 871.98 | 851.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 884.20 | 891.69 | 874.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 09:15:00 | 898.15 | 896.16 | 890.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 898.15 | 896.16 | 890.62 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 09:15:00 | 1195.00 | 1230.59 | 1231.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 1164.60 | 1196.63 | 1211.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 1188.85 | 1162.27 | 1177.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 14:15:00 | 1188.85 | 1162.27 | 1177.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 1188.85 | 1162.27 | 1177.67 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 1237.95 | 1189.76 | 1187.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 1245.20 | 1218.87 | 1205.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 12:15:00 | 1219.40 | 1222.18 | 1210.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 12:15:00 | 1219.40 | 1222.18 | 1210.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 12:15:00 | 1219.40 | 1222.18 | 1210.39 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 12:15:00 | 1208.00 | 1214.65 | 1215.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 09:15:00 | 1195.60 | 1206.67 | 1210.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 09:15:00 | 1192.50 | 1188.39 | 1194.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 09:15:00 | 1192.50 | 1188.39 | 1194.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 1192.50 | 1188.39 | 1194.02 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 14:15:00 | 1199.00 | 1190.91 | 1190.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 1227.20 | 1197.70 | 1193.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 11:15:00 | 1197.60 | 1200.77 | 1195.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 11:15:00 | 1197.60 | 1200.77 | 1195.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 1197.60 | 1200.77 | 1195.66 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 13:15:00 | 1186.80 | 1194.05 | 1194.43 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 09:15:00 | 1200.00 | 1195.35 | 1194.89 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 1188.00 | 1193.88 | 1194.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 1184.10 | 1191.93 | 1193.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-08 12:15:00 | 1192.75 | 1192.09 | 1193.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 12:15:00 | 1192.75 | 1192.09 | 1193.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 1192.75 | 1192.09 | 1193.29 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 13:15:00 | 1228.00 | 1199.27 | 1196.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 09:15:00 | 1263.15 | 1215.23 | 1204.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 09:15:00 | 1356.00 | 1371.49 | 1335.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 1356.00 | 1371.49 | 1335.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 1356.00 | 1371.49 | 1335.32 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 1388.95 | 1401.04 | 1401.61 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-01-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 10:15:00 | 1414.00 | 1403.63 | 1402.74 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-01-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 12:15:00 | 1382.00 | 1400.53 | 1401.56 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 10:15:00 | 1412.00 | 1400.90 | 1400.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 1417.10 | 1408.06 | 1404.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 1417.95 | 1420.33 | 1414.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 15:15:00 | 1418.25 | 1419.92 | 1414.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 1418.25 | 1419.92 | 1414.39 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 1400.00 | 1410.77 | 1411.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 1366.25 | 1399.64 | 1406.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 1428.90 | 1399.14 | 1404.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 1428.90 | 1399.14 | 1404.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 1428.90 | 1399.14 | 1404.40 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 11:15:00 | 1428.90 | 1409.85 | 1408.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 1493.00 | 1435.48 | 1422.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 10:15:00 | 1482.80 | 1485.88 | 1462.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 14:15:00 | 1455.00 | 1476.95 | 1465.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 14:15:00 | 1455.00 | 1476.95 | 1465.37 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 11:15:00 | 1442.75 | 1459.38 | 1459.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 1414.00 | 1443.66 | 1451.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 11:15:00 | 1388.00 | 1372.95 | 1391.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 12:15:00 | 1390.00 | 1376.36 | 1391.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 12:15:00 | 1390.00 | 1376.36 | 1391.47 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 14:15:00 | 1425.00 | 1396.65 | 1393.38 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 09:15:00 | 1390.00 | 1399.24 | 1399.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 11:15:00 | 1372.00 | 1391.42 | 1395.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 15:15:00 | 1385.00 | 1382.03 | 1389.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 1330.00 | 1371.62 | 1383.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 1330.00 | 1371.62 | 1383.85 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 11:15:00 | 1393.10 | 1364.21 | 1361.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 1462.75 | 1397.56 | 1379.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 10:15:00 | 1501.00 | 1529.03 | 1506.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 10:15:00 | 1501.00 | 1529.03 | 1506.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 1501.00 | 1529.03 | 1506.35 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 15:15:00 | 1739.00 | 1757.32 | 1759.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1656.00 | 1737.05 | 1749.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 1700.75 | 1685.60 | 1710.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 1700.75 | 1685.60 | 1710.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 1700.75 | 1685.60 | 1710.19 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 09:15:00 | 1561.75 | 1508.79 | 1504.30 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 12:15:00 | 1498.00 | 1513.51 | 1514.01 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 10:15:00 | 1546.50 | 1514.66 | 1513.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 11:15:00 | 1559.75 | 1532.82 | 1523.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 10:15:00 | 1603.60 | 1607.59 | 1583.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 13:15:00 | 1597.80 | 1602.61 | 1587.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 1597.80 | 1602.61 | 1587.15 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 1649.00 | 1657.18 | 1657.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 13:15:00 | 1638.05 | 1651.57 | 1655.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 10:15:00 | 1652.85 | 1648.73 | 1652.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 10:15:00 | 1652.85 | 1648.73 | 1652.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 1652.85 | 1648.73 | 1652.35 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 1668.30 | 1656.52 | 1655.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 1673.35 | 1661.82 | 1657.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 09:15:00 | 1655.55 | 1662.73 | 1660.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 1655.55 | 1662.73 | 1660.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 1655.55 | 1662.73 | 1660.16 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 1631.90 | 1653.70 | 1656.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 09:15:00 | 1595.95 | 1636.77 | 1647.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 13:15:00 | 1639.20 | 1624.77 | 1637.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 13:15:00 | 1639.20 | 1624.77 | 1637.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 1639.20 | 1624.77 | 1637.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 1644.65 | 1625.08 | 1635.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1620.45 | 1624.15 | 1633.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 1560.00 | 1621.20 | 1627.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 13:15:00 | 1626.00 | 1613.27 | 1612.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 13:15:00 | 1626.00 | 1613.27 | 1612.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 15:15:00 | 1634.55 | 1618.91 | 1615.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 1649.90 | 1672.70 | 1653.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 1649.90 | 1672.70 | 1653.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1649.90 | 1672.70 | 1653.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:45:00 | 1641.70 | 1672.70 | 1653.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 10:15:00 | 1634.40 | 1665.04 | 1651.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 11:00:00 | 1634.40 | 1665.04 | 1651.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 1670.00 | 1664.27 | 1653.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 13:45:00 | 1684.80 | 1669.92 | 1657.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 10:45:00 | 1679.95 | 1670.47 | 1661.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 09:30:00 | 1684.90 | 1669.41 | 1664.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 10:00:00 | 1682.00 | 1669.41 | 1664.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 1670.90 | 1671.80 | 1667.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:30:00 | 1668.15 | 1671.80 | 1667.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 1658.80 | 1669.20 | 1666.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 1658.80 | 1669.20 | 1666.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 1665.00 | 1668.36 | 1666.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 1672.00 | 1668.36 | 1666.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 11:15:00 | 1657.20 | 1664.57 | 1665.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 11:15:00 | 1657.20 | 1664.57 | 1665.19 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 13:15:00 | 1679.25 | 1668.19 | 1666.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 14:15:00 | 1721.50 | 1678.85 | 1671.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 14:15:00 | 1715.00 | 1720.15 | 1701.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 14:45:00 | 1716.25 | 1720.15 | 1701.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 1721.35 | 1721.13 | 1708.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 12:00:00 | 1721.35 | 1721.13 | 1708.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 12:15:00 | 1709.90 | 1718.89 | 1708.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 12:30:00 | 1707.80 | 1718.89 | 1708.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 1705.10 | 1716.13 | 1708.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:30:00 | 1706.65 | 1716.13 | 1708.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 1690.00 | 1710.90 | 1706.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 1690.00 | 1710.90 | 1706.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 1713.75 | 1711.47 | 1707.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:15:00 | 1672.80 | 1711.47 | 1707.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 1638.95 | 1696.97 | 1700.87 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 09:15:00 | 1764.45 | 1699.35 | 1692.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 09:15:00 | 1852.65 | 1768.44 | 1735.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 14:15:00 | 1934.90 | 1935.52 | 1897.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 15:00:00 | 1934.90 | 1935.52 | 1897.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 1940.00 | 1932.59 | 1905.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 1967.90 | 1935.81 | 1917.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 11:00:00 | 1969.85 | 1958.22 | 1956.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-15 09:15:00 | 2164.69 | 2134.30 | 2074.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 15:15:00 | 2191.35 | 2213.79 | 2214.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-18 09:15:00 | 2160.00 | 2203.03 | 2209.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 2158.25 | 2109.04 | 2143.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 2158.25 | 2109.04 | 2143.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 2158.25 | 2109.04 | 2143.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 2158.25 | 2109.04 | 2143.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 2139.05 | 2115.04 | 2143.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:30:00 | 2141.00 | 2115.04 | 2143.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 2137.90 | 2119.61 | 2142.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 2149.00 | 2119.61 | 2142.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 2144.15 | 2124.52 | 2142.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:45:00 | 2147.55 | 2124.52 | 2142.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 2154.40 | 2130.49 | 2143.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 2154.40 | 2130.49 | 2143.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 2148.95 | 2134.19 | 2144.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:30:00 | 2156.50 | 2134.19 | 2144.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 2150.00 | 2137.35 | 2144.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 2245.00 | 2137.35 | 2144.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 2211.00 | 2152.08 | 2150.86 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 2099.30 | 2145.64 | 2148.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 14:15:00 | 2080.00 | 2126.09 | 2138.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 2134.00 | 2125.26 | 2136.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:00:00 | 2134.00 | 2125.26 | 2136.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 2147.00 | 2129.61 | 2137.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:00:00 | 2147.00 | 2129.61 | 2137.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 2153.45 | 2134.38 | 2138.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:30:00 | 2134.70 | 2131.67 | 2137.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 2170.65 | 2137.57 | 2137.60 | SL hit (close>static) qty=1.00 sl=2168.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 10:15:00 | 2190.30 | 2148.11 | 2142.39 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 2125.00 | 2144.80 | 2145.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 2105.00 | 2136.84 | 2142.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 15:15:00 | 2139.00 | 2133.69 | 2139.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 15:15:00 | 2139.00 | 2133.69 | 2139.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 2139.00 | 2133.69 | 2139.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 2173.75 | 2133.69 | 2139.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 2177.20 | 2142.39 | 2142.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:30:00 | 2184.30 | 2142.39 | 2142.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 2173.00 | 2148.51 | 2145.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 12:15:00 | 2234.70 | 2169.89 | 2155.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 14:15:00 | 2230.00 | 2247.86 | 2215.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-30 15:00:00 | 2230.00 | 2247.86 | 2215.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 2190.85 | 2235.04 | 2215.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:45:00 | 2202.75 | 2235.04 | 2215.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 2194.65 | 2226.96 | 2213.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 11:30:00 | 2226.00 | 2233.57 | 2217.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-03 09:15:00 | 2448.60 | 2310.35 | 2264.44 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 2284.10 | 2306.59 | 2307.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 2169.90 | 2279.25 | 2294.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 2264.00 | 2211.65 | 2242.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 2264.00 | 2211.65 | 2242.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 2264.00 | 2211.65 | 2242.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 2279.15 | 2211.65 | 2242.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 2272.00 | 2223.72 | 2245.25 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 2279.15 | 2255.37 | 2254.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 2481.05 | 2300.50 | 2274.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 2471.50 | 2483.61 | 2424.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 2471.50 | 2483.61 | 2424.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 2433.35 | 2469.78 | 2428.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:45:00 | 2425.90 | 2469.78 | 2428.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 2422.40 | 2460.30 | 2427.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:00:00 | 2422.40 | 2460.30 | 2427.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 2412.95 | 2450.83 | 2426.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:00:00 | 2412.95 | 2450.83 | 2426.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 2404.00 | 2441.47 | 2424.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:45:00 | 2397.95 | 2441.47 | 2424.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 2413.00 | 2435.77 | 2423.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 14:30:00 | 2448.25 | 2430.65 | 2422.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 15:15:00 | 2424.75 | 2430.65 | 2422.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:45:00 | 2447.90 | 2440.66 | 2433.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 09:15:00 | 2667.23 | 2559.51 | 2530.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 2550.90 | 2574.13 | 2575.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 2528.15 | 2558.65 | 2567.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 2578.35 | 2562.59 | 2568.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 10:15:00 | 2578.35 | 2562.59 | 2568.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 2578.35 | 2562.59 | 2568.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:30:00 | 2573.80 | 2562.59 | 2568.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 2578.40 | 2565.75 | 2569.29 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 2578.05 | 2571.61 | 2571.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 2629.00 | 2588.40 | 2579.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 13:15:00 | 2576.85 | 2587.67 | 2582.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 13:15:00 | 2576.85 | 2587.67 | 2582.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 2576.85 | 2587.67 | 2582.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:45:00 | 2579.00 | 2587.67 | 2582.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 2558.00 | 2581.73 | 2580.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 2558.00 | 2581.73 | 2580.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 2555.00 | 2576.39 | 2577.84 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 2604.60 | 2582.33 | 2579.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 2637.65 | 2605.86 | 2592.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 2586.25 | 2604.05 | 2594.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 2586.25 | 2604.05 | 2594.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 2586.25 | 2604.05 | 2594.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 2586.25 | 2604.05 | 2594.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 2593.85 | 2602.01 | 2594.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 2624.95 | 2599.21 | 2593.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 11:15:00 | 2573.35 | 2594.93 | 2593.21 | SL hit (close<static) qty=1.00 sl=2575.05 alert=retest2 |

### Cycle 66 — SELL (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 12:15:00 | 2555.00 | 2586.95 | 2589.73 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 2742.00 | 2616.27 | 2601.88 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 13:15:00 | 2616.00 | 2641.98 | 2642.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 14:15:00 | 2606.35 | 2634.85 | 2639.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 10:15:00 | 2633.10 | 2631.46 | 2636.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 11:00:00 | 2633.10 | 2631.46 | 2636.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 2625.05 | 2630.18 | 2635.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 12:15:00 | 2617.00 | 2630.18 | 2635.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:30:00 | 2617.90 | 2627.19 | 2631.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 14:15:00 | 2647.10 | 2635.56 | 2634.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 14:15:00 | 2647.10 | 2635.56 | 2634.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 2700.00 | 2649.48 | 2641.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 2675.00 | 2677.90 | 2663.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:45:00 | 2676.20 | 2677.90 | 2663.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 2695.30 | 2681.38 | 2666.47 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 2591.00 | 2650.52 | 2656.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 09:15:00 | 2541.50 | 2584.91 | 2614.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 2487.45 | 2440.72 | 2476.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 2487.45 | 2440.72 | 2476.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 2487.45 | 2440.72 | 2476.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 2500.45 | 2440.72 | 2476.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 2481.05 | 2448.79 | 2477.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:30:00 | 2488.80 | 2448.79 | 2477.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 2486.50 | 2456.33 | 2478.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:30:00 | 2474.10 | 2463.47 | 2477.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 2465.00 | 2469.57 | 2478.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:45:00 | 2468.35 | 2464.49 | 2475.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 2350.39 | 2406.36 | 2424.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 2440.40 | 2406.36 | 2424.55 | SL hit (close>static) qty=0.50 sl=2406.36 alert=retest2 |

### Cycle 71 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 2466.20 | 2422.20 | 2420.31 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 2403.60 | 2419.37 | 2420.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 2370.40 | 2397.69 | 2408.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 14:15:00 | 2386.90 | 2386.00 | 2395.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 14:45:00 | 2390.35 | 2386.00 | 2395.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 2393.00 | 2387.40 | 2395.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:15:00 | 2337.10 | 2387.40 | 2395.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 2336.95 | 2377.31 | 2390.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 10:00:00 | 2324.00 | 2339.27 | 2353.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 10:30:00 | 2318.35 | 2334.86 | 2350.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 10:30:00 | 2325.05 | 2322.30 | 2334.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 11:30:00 | 2322.90 | 2319.74 | 2332.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 2389.35 | 2309.30 | 2319.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 2389.35 | 2309.30 | 2319.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 2365.90 | 2320.62 | 2323.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 2394.20 | 2320.62 | 2323.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-02 11:15:00 | 2386.00 | 2333.70 | 2329.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 11:15:00 | 2386.00 | 2333.70 | 2329.20 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 2253.25 | 2325.54 | 2330.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 09:15:00 | 2178.90 | 2260.40 | 2291.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 12:15:00 | 2253.00 | 2239.65 | 2272.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 13:00:00 | 2253.00 | 2239.65 | 2272.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 2222.85 | 2213.21 | 2244.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 2227.25 | 2213.21 | 2244.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 2235.00 | 2216.99 | 2238.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 2235.00 | 2216.99 | 2238.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 2275.80 | 2228.75 | 2242.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 2275.80 | 2228.75 | 2242.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 2268.70 | 2236.74 | 2244.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 2250.10 | 2237.77 | 2244.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 13:15:00 | 2261.15 | 2247.94 | 2247.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 13:15:00 | 2261.15 | 2247.94 | 2247.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 2278.00 | 2265.59 | 2257.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 2359.05 | 2374.27 | 2339.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:00:00 | 2359.05 | 2374.27 | 2339.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 2296.35 | 2358.68 | 2336.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 2296.35 | 2358.68 | 2336.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 2303.05 | 2347.56 | 2333.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 2278.50 | 2347.56 | 2333.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 2249.05 | 2313.98 | 2319.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 13:15:00 | 2230.85 | 2278.27 | 2300.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 2252.10 | 2250.88 | 2280.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 10:00:00 | 2252.10 | 2250.88 | 2280.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 2262.90 | 2253.28 | 2278.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 2264.95 | 2253.28 | 2278.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 2281.15 | 2258.85 | 2278.82 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 2379.60 | 2304.76 | 2295.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 2450.00 | 2375.90 | 2339.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 2517.10 | 2527.27 | 2460.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:00:00 | 2517.10 | 2527.27 | 2460.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 2493.50 | 2512.31 | 2490.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 2493.50 | 2512.31 | 2490.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 2492.50 | 2508.35 | 2490.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:15:00 | 2517.00 | 2498.95 | 2490.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 2507.10 | 2511.61 | 2502.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 2499.85 | 2508.09 | 2502.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 2545.00 | 2586.07 | 2587.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 2545.00 | 2586.07 | 2587.99 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 2706.90 | 2605.55 | 2593.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 2739.95 | 2660.29 | 2625.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 09:15:00 | 2763.75 | 2769.37 | 2718.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 10:00:00 | 2763.75 | 2769.37 | 2718.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 2695.90 | 2748.00 | 2732.55 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 2687.50 | 2720.74 | 2722.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 15:15:00 | 2677.50 | 2702.40 | 2712.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 2645.90 | 2629.01 | 2650.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 10:00:00 | 2645.90 | 2629.01 | 2650.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 2675.30 | 2638.27 | 2652.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:45:00 | 2682.00 | 2638.27 | 2652.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 2721.60 | 2654.94 | 2658.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 12:00:00 | 2721.60 | 2654.94 | 2658.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 12:15:00 | 2746.25 | 2673.20 | 2666.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 2776.75 | 2722.31 | 2702.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 2712.20 | 2724.37 | 2708.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 12:15:00 | 2712.20 | 2724.37 | 2708.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 2712.20 | 2724.37 | 2708.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:00:00 | 2712.20 | 2724.37 | 2708.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 2700.00 | 2719.50 | 2707.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:30:00 | 2709.10 | 2719.50 | 2707.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 2696.60 | 2714.92 | 2706.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 2704.25 | 2714.92 | 2706.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 2708.95 | 2713.73 | 2706.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 2720.00 | 2713.73 | 2706.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:30:00 | 2714.65 | 2717.33 | 2709.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:00:00 | 2724.60 | 2714.36 | 2710.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 15:15:00 | 2714.00 | 2714.17 | 2713.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 2714.00 | 2714.14 | 2713.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:45:00 | 2795.00 | 2731.11 | 2720.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 11:15:00 | 2689.00 | 2762.50 | 2759.23 | SL hit (close<static) qty=1.00 sl=2690.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 2680.35 | 2746.07 | 2752.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 2624.10 | 2697.05 | 2724.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 2655.00 | 2641.30 | 2682.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 2655.00 | 2641.30 | 2682.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 2643.80 | 2644.63 | 2676.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 15:15:00 | 2600.00 | 2667.67 | 2675.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 09:30:00 | 2595.50 | 2646.51 | 2663.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:00:00 | 2612.65 | 2619.32 | 2637.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 12:15:00 | 2482.02 | 2602.97 | 2625.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 2524.45 | 2522.91 | 2539.75 | SL hit (close>ema200) qty=0.50 sl=2522.91 alert=retest2 |

### Cycle 83 — BUY (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 12:15:00 | 2575.00 | 2497.09 | 2493.72 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 13:15:00 | 2445.40 | 2499.84 | 2501.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 2422.95 | 2447.78 | 2461.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 14:15:00 | 2452.75 | 2448.77 | 2460.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 15:00:00 | 2452.75 | 2448.77 | 2460.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 2437.05 | 2446.43 | 2458.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 2482.00 | 2453.40 | 2460.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 10:15:00 | 2595.85 | 2481.89 | 2472.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 11:15:00 | 2610.00 | 2507.51 | 2485.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 2625.00 | 2649.30 | 2617.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 2625.00 | 2649.30 | 2617.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 2625.00 | 2649.30 | 2617.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 2625.00 | 2649.30 | 2617.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 2604.50 | 2640.34 | 2616.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 2609.20 | 2640.34 | 2616.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 2602.10 | 2632.69 | 2614.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 2599.70 | 2632.69 | 2614.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 2568.00 | 2603.62 | 2605.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 2543.95 | 2584.46 | 2596.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 2586.00 | 2580.54 | 2591.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 14:15:00 | 2586.00 | 2580.54 | 2591.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 2586.00 | 2580.54 | 2591.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:30:00 | 2576.50 | 2580.54 | 2591.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 2595.00 | 2583.44 | 2591.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 2503.65 | 2583.44 | 2591.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 13:15:00 | 2651.30 | 2584.29 | 2586.27 | SL hit (close>static) qty=1.00 sl=2599.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 2677.00 | 2602.83 | 2594.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 2797.95 | 2655.00 | 2620.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 2660.00 | 2717.99 | 2680.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 2660.00 | 2717.99 | 2680.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 2660.00 | 2717.99 | 2680.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 2660.00 | 2717.99 | 2680.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 2652.00 | 2704.80 | 2677.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 2649.25 | 2704.80 | 2677.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 2655.35 | 2694.91 | 2675.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:30:00 | 2652.00 | 2694.91 | 2675.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 2688.30 | 2691.19 | 2677.37 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 2630.00 | 2664.24 | 2666.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 2588.00 | 2617.44 | 2635.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 2560.00 | 2540.89 | 2574.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:00:00 | 2560.00 | 2540.89 | 2574.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 2551.40 | 2542.99 | 2571.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:45:00 | 2565.25 | 2542.99 | 2571.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 2549.60 | 2545.63 | 2566.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:45:00 | 2562.00 | 2545.63 | 2566.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 2545.00 | 2545.51 | 2564.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:30:00 | 2564.25 | 2545.51 | 2564.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 2522.70 | 2543.55 | 2560.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 2515.20 | 2539.27 | 2556.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 2635.00 | 2556.26 | 2556.43 | SL hit (close>static) qty=1.00 sl=2568.55 alert=retest2 |

### Cycle 89 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 2607.15 | 2566.44 | 2561.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 2677.80 | 2601.42 | 2578.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 2679.75 | 2687.64 | 2654.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 09:45:00 | 2682.00 | 2687.64 | 2654.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 2635.30 | 2677.17 | 2652.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 2635.30 | 2677.17 | 2652.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 2665.25 | 2674.78 | 2653.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 2799.90 | 2655.83 | 2653.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 2729.05 | 2819.18 | 2829.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 2729.05 | 2819.18 | 2829.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 2667.80 | 2749.55 | 2787.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 2730.10 | 2727.59 | 2763.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 11:45:00 | 2730.00 | 2727.59 | 2763.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 2710.15 | 2692.34 | 2717.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:45:00 | 2712.95 | 2692.34 | 2717.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 2835.00 | 2722.59 | 2725.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 2835.00 | 2722.59 | 2725.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 2845.50 | 2747.17 | 2736.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 2880.55 | 2791.63 | 2759.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 2796.25 | 2815.03 | 2783.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 10:00:00 | 2796.25 | 2815.03 | 2783.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 2806.70 | 2813.36 | 2785.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:00:00 | 2818.95 | 2813.67 | 2790.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 2876.90 | 2795.52 | 2792.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 12:30:00 | 2823.40 | 2805.62 | 2799.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 13:15:00 | 2814.50 | 2805.62 | 2799.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 3043.75 | 2877.68 | 2845.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 2804.00 | 2851.11 | 2854.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 2804.00 | 2851.11 | 2854.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 2702.00 | 2813.11 | 2836.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 2770.90 | 2768.37 | 2803.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 14:00:00 | 2770.90 | 2768.37 | 2803.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2800.50 | 2769.46 | 2795.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 2800.50 | 2769.46 | 2795.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 2798.80 | 2775.33 | 2795.39 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 2893.95 | 2811.31 | 2803.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 14:15:00 | 2900.00 | 2860.83 | 2832.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 2850.00 | 2865.90 | 2842.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:30:00 | 2857.95 | 2865.90 | 2842.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2833.95 | 2859.51 | 2841.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 2829.65 | 2859.51 | 2841.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 2848.50 | 2857.31 | 2842.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 2835.00 | 2857.31 | 2842.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 2865.05 | 2858.34 | 2845.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:30:00 | 2845.95 | 2858.34 | 2845.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2887.60 | 2865.74 | 2851.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 2946.00 | 2873.37 | 2861.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 14:00:00 | 2915.05 | 2903.58 | 2883.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 15:15:00 | 2914.95 | 2902.50 | 2884.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:15:00 | 2921.10 | 2899.42 | 2898.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2890.00 | 2897.54 | 2897.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 2894.80 | 2897.54 | 2897.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 2893.90 | 2896.81 | 2897.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 10:15:00 | 2893.90 | 2896.81 | 2897.10 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 2899.95 | 2897.45 | 2897.32 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 2885.00 | 2894.96 | 2896.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 15:15:00 | 2879.90 | 2891.95 | 2894.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 2793.50 | 2789.59 | 2822.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:45:00 | 2797.35 | 2789.59 | 2822.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 2822.75 | 2794.68 | 2819.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 2822.75 | 2794.68 | 2819.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 2819.20 | 2799.59 | 2819.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:15:00 | 2820.30 | 2799.59 | 2819.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 2828.05 | 2805.28 | 2820.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 2828.05 | 2805.28 | 2820.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 2837.00 | 2811.62 | 2821.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 2837.00 | 2811.62 | 2821.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 2834.05 | 2816.11 | 2822.83 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 2855.00 | 2830.11 | 2828.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 10:15:00 | 2876.00 | 2838.95 | 2832.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 14:15:00 | 2935.00 | 2948.03 | 2915.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 14:15:00 | 2935.00 | 2948.03 | 2915.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 2935.00 | 2948.03 | 2915.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:45:00 | 2923.50 | 2948.03 | 2915.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 2859.35 | 2927.89 | 2912.24 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 12:15:00 | 2869.00 | 2898.40 | 2901.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 2833.20 | 2873.66 | 2887.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 2722.00 | 2717.04 | 2765.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 2722.00 | 2717.04 | 2765.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 2714.90 | 2700.26 | 2721.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 2713.25 | 2700.26 | 2721.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 2719.90 | 2705.20 | 2718.48 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 2750.00 | 2728.40 | 2726.41 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 2684.70 | 2723.93 | 2725.30 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 2740.00 | 2709.62 | 2709.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 2819.95 | 2731.68 | 2719.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 2863.80 | 2866.31 | 2834.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 10:30:00 | 2864.25 | 2866.31 | 2834.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2844.85 | 2859.67 | 2841.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 2844.85 | 2859.67 | 2841.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 2832.75 | 2854.29 | 2840.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 2797.05 | 2854.29 | 2840.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2829.75 | 2849.38 | 2839.37 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 2765.45 | 2832.60 | 2832.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 2749.00 | 2799.79 | 2816.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2784.50 | 2781.43 | 2802.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 2818.55 | 2788.85 | 2803.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 2818.55 | 2788.85 | 2803.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 2818.55 | 2788.85 | 2803.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 2816.20 | 2794.32 | 2804.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 12:15:00 | 2813.75 | 2794.32 | 2804.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 2832.70 | 2811.13 | 2810.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 2832.70 | 2811.13 | 2810.81 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 2791.60 | 2810.08 | 2810.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 2760.90 | 2796.11 | 2803.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 2797.25 | 2785.43 | 2794.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 2797.25 | 2785.43 | 2794.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 2797.25 | 2785.43 | 2794.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 2797.25 | 2785.43 | 2794.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 2786.50 | 2785.65 | 2793.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:30:00 | 2777.95 | 2784.32 | 2792.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:00:00 | 2779.00 | 2784.32 | 2792.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 2639.05 | 2748.40 | 2771.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 2640.05 | 2748.40 | 2771.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 10:15:00 | 2500.15 | 2605.66 | 2676.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 105 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 1715.45 | 1654.97 | 1646.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 1738.00 | 1689.58 | 1667.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1772.45 | 1778.30 | 1743.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 12:30:00 | 1785.00 | 1778.30 | 1743.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1737.65 | 1772.68 | 1752.67 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 1699.85 | 1738.44 | 1740.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 1677.05 | 1726.16 | 1734.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 15:15:00 | 1705.00 | 1694.98 | 1707.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 15:15:00 | 1705.00 | 1694.98 | 1707.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 1705.00 | 1694.98 | 1707.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 1738.55 | 1694.98 | 1707.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1736.70 | 1703.33 | 1710.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 1750.00 | 1703.33 | 1710.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 1775.00 | 1717.66 | 1716.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 09:15:00 | 1838.45 | 1763.01 | 1741.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 1795.90 | 1796.05 | 1766.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 1795.90 | 1796.05 | 1766.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1813.75 | 1800.03 | 1776.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 1817.20 | 1803.72 | 1779.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 1825.50 | 1806.04 | 1783.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 15:15:00 | 1755.00 | 1797.65 | 1787.06 | SL hit (close<static) qty=1.00 sl=1771.05 alert=retest2 |

### Cycle 108 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1697.55 | 1777.63 | 1778.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1646.40 | 1736.16 | 1758.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1566.90 | 1553.11 | 1605.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 1566.90 | 1553.11 | 1605.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1467.95 | 1368.69 | 1389.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1467.95 | 1368.69 | 1389.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1474.65 | 1389.88 | 1397.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 1474.65 | 1389.88 | 1397.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 1474.65 | 1406.84 | 1404.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 1516.40 | 1460.78 | 1434.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 1584.00 | 1590.90 | 1550.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 1544.15 | 1590.90 | 1550.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1534.50 | 1579.62 | 1549.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:15:00 | 1525.05 | 1579.62 | 1549.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1528.20 | 1569.33 | 1547.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 1554.95 | 1569.33 | 1547.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:30:00 | 1543.85 | 1555.11 | 1547.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 10:15:00 | 1512.70 | 1547.40 | 1551.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 1512.70 | 1547.40 | 1551.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 1501.95 | 1538.31 | 1546.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 1454.50 | 1453.16 | 1485.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 15:00:00 | 1454.50 | 1453.16 | 1485.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 1380.50 | 1436.68 | 1472.65 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1498.00 | 1454.08 | 1449.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 1515.50 | 1474.80 | 1459.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1581.80 | 1588.70 | 1563.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 10:00:00 | 1581.80 | 1588.70 | 1563.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1575.35 | 1585.46 | 1566.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 1575.35 | 1585.46 | 1566.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1560.35 | 1580.44 | 1566.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 1560.35 | 1580.44 | 1566.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 1553.35 | 1575.02 | 1564.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 1553.35 | 1575.02 | 1564.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1525.50 | 1565.12 | 1561.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 1525.50 | 1565.12 | 1561.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1525.00 | 1557.09 | 1558.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1507.40 | 1547.16 | 1553.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 1544.90 | 1528.66 | 1538.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 15:15:00 | 1544.90 | 1528.66 | 1538.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 1544.90 | 1528.66 | 1538.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 1514.15 | 1528.66 | 1538.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:45:00 | 1511.00 | 1522.91 | 1535.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 12:15:00 | 1438.44 | 1457.82 | 1478.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 12:15:00 | 1435.45 | 1457.82 | 1478.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1491.00 | 1443.55 | 1463.15 | SL hit (close>ema200) qty=0.50 sl=1443.55 alert=retest2 |

### Cycle 113 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 1510.50 | 1479.91 | 1475.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1520.00 | 1491.97 | 1482.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 1515.00 | 1517.07 | 1507.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 14:30:00 | 1518.90 | 1517.07 | 1507.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1553.05 | 1524.25 | 1512.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:45:00 | 1566.95 | 1532.77 | 1517.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 1566.30 | 1595.12 | 1596.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 1566.30 | 1595.12 | 1596.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 1560.55 | 1588.21 | 1593.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 1559.35 | 1544.42 | 1560.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 1559.35 | 1544.42 | 1560.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1559.35 | 1544.42 | 1560.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:15:00 | 1525.00 | 1540.29 | 1554.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 14:00:00 | 1526.60 | 1537.55 | 1551.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 1515.00 | 1533.04 | 1548.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 1549.70 | 1532.54 | 1531.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 1549.70 | 1532.54 | 1531.87 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 1527.55 | 1531.70 | 1531.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 1515.15 | 1528.39 | 1530.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 14:15:00 | 1529.60 | 1527.86 | 1529.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 15:00:00 | 1529.60 | 1527.86 | 1529.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 1530.15 | 1528.32 | 1529.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 1471.10 | 1528.32 | 1529.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1323.99 | 1484.36 | 1503.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 1483.60 | 1458.34 | 1457.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1539.40 | 1482.77 | 1470.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 1520.10 | 1526.04 | 1506.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 1520.10 | 1526.04 | 1506.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1494.60 | 1517.50 | 1508.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 1494.60 | 1517.50 | 1508.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1500.00 | 1514.00 | 1507.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:30:00 | 1504.00 | 1508.72 | 1506.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 14:15:00 | 1497.40 | 1504.76 | 1504.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 14:15:00 | 1497.40 | 1504.76 | 1504.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 15:15:00 | 1495.60 | 1502.93 | 1504.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 14:15:00 | 1502.00 | 1500.47 | 1502.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 14:15:00 | 1502.00 | 1500.47 | 1502.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 1502.00 | 1500.47 | 1502.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 14:45:00 | 1504.00 | 1500.47 | 1502.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 1503.80 | 1501.14 | 1502.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:15:00 | 1520.10 | 1501.14 | 1502.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 1515.00 | 1503.91 | 1503.38 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 14:15:00 | 1498.30 | 1503.19 | 1503.50 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 09:15:00 | 1507.90 | 1504.10 | 1503.86 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1448.60 | 1507.92 | 1511.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 1422.80 | 1447.46 | 1460.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 15:15:00 | 1423.00 | 1420.90 | 1432.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 09:15:00 | 1582.10 | 1420.90 | 1432.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 123 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1675.90 | 1471.90 | 1454.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 12:15:00 | 1688.00 | 1596.59 | 1569.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1561.30 | 1601.84 | 1582.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 1561.30 | 1601.84 | 1582.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1561.30 | 1601.84 | 1582.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 1561.30 | 1601.84 | 1582.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 1564.70 | 1594.41 | 1580.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:30:00 | 1559.60 | 1594.41 | 1580.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1570.80 | 1582.95 | 1578.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 13:30:00 | 1571.40 | 1582.95 | 1578.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 1587.90 | 1582.43 | 1578.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 1672.30 | 1582.43 | 1578.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-15 09:15:00 | 1839.53 | 1753.06 | 1725.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 1807.00 | 1817.39 | 1818.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 1783.80 | 1810.68 | 1815.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 1822.60 | 1808.28 | 1811.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 1822.60 | 1808.28 | 1811.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1822.60 | 1808.28 | 1811.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 1822.60 | 1808.28 | 1811.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1815.50 | 1809.72 | 1812.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:30:00 | 1811.30 | 1811.36 | 1812.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 1813.30 | 1810.89 | 1812.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 1806.30 | 1808.07 | 1810.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1929.90 | 1827.25 | 1817.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1929.90 | 1827.25 | 1817.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1987.90 | 1929.85 | 1899.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1991.60 | 2009.49 | 1981.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1991.60 | 2009.49 | 1981.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1991.60 | 2009.49 | 1981.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 1992.00 | 2009.49 | 1981.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1988.90 | 2005.37 | 1982.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 1992.60 | 2005.37 | 1982.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 1994.00 | 2001.05 | 1984.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1971.60 | 1993.39 | 1985.79 | SL hit (close<static) qty=1.00 sl=1980.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 1963.90 | 1980.04 | 1981.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 1951.00 | 1974.23 | 1978.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 1989.10 | 1973.81 | 1977.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 1989.10 | 1973.81 | 1977.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1989.10 | 1973.81 | 1977.34 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1995.90 | 1974.40 | 1973.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 1996.00 | 1982.16 | 1977.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 1987.10 | 2004.37 | 1994.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 1987.10 | 2004.37 | 1994.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1987.10 | 2004.37 | 1994.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1987.10 | 2004.37 | 1994.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1952.00 | 1993.90 | 1990.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 1923.70 | 1993.90 | 1990.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 1931.40 | 1981.40 | 1985.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1841.50 | 1896.92 | 1925.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 1769.30 | 1768.66 | 1805.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 1769.30 | 1768.66 | 1805.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1823.70 | 1783.34 | 1805.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1823.90 | 1783.34 | 1805.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1818.00 | 1790.27 | 1807.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1794.30 | 1790.27 | 1807.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1796.20 | 1793.35 | 1805.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 1807.30 | 1793.35 | 1805.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1769.00 | 1761.55 | 1775.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 1774.40 | 1761.55 | 1775.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1751.70 | 1739.52 | 1754.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 1741.60 | 1739.52 | 1754.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1757.90 | 1743.20 | 1755.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1757.90 | 1743.20 | 1755.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1757.50 | 1746.06 | 1755.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1746.00 | 1746.06 | 1755.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1762.90 | 1750.81 | 1755.33 | SL hit (close>static) qty=1.00 sl=1760.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1765.80 | 1758.90 | 1758.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1803.70 | 1769.63 | 1763.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1786.10 | 1786.61 | 1775.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 1786.10 | 1786.61 | 1775.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1770.00 | 1783.29 | 1774.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1770.00 | 1783.29 | 1774.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1775.30 | 1781.69 | 1774.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1836.20 | 1781.69 | 1774.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 1831.60 | 1851.94 | 1854.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 1831.60 | 1851.94 | 1854.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1819.00 | 1845.36 | 1851.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 1804.80 | 1796.65 | 1810.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:45:00 | 1809.00 | 1796.65 | 1810.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1809.80 | 1799.78 | 1809.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 1820.20 | 1799.78 | 1809.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1801.40 | 1800.11 | 1808.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 1810.80 | 1800.11 | 1808.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1803.00 | 1800.68 | 1808.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:30:00 | 1797.80 | 1800.27 | 1807.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1821.40 | 1805.03 | 1807.25 | SL hit (close>static) qty=1.00 sl=1808.20 alert=retest2 |

### Cycle 131 — BUY (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 11:15:00 | 1818.00 | 1810.56 | 1809.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 1929.80 | 1837.34 | 1822.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 1947.80 | 1962.57 | 1927.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:30:00 | 1955.30 | 1962.57 | 1927.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1959.70 | 1964.68 | 1953.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1960.70 | 1964.68 | 1953.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 1942.10 | 1960.17 | 1952.63 | SL hit (close<static) qty=1.00 sl=1950.10 alert=retest2 |

### Cycle 132 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 1931.00 | 1947.50 | 1948.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 1910.30 | 1932.81 | 1939.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1911.60 | 1909.32 | 1921.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1911.60 | 1909.32 | 1921.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1911.60 | 1909.32 | 1921.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 1910.20 | 1909.32 | 1921.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1882.90 | 1896.85 | 1908.32 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 1959.30 | 1917.86 | 1915.50 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 1923.00 | 1935.59 | 1935.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 1900.00 | 1927.55 | 1931.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 1920.90 | 1875.20 | 1896.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 1920.90 | 1875.20 | 1896.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1920.90 | 1875.20 | 1896.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 1920.90 | 1875.20 | 1896.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1931.10 | 1886.38 | 1899.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 1931.10 | 1886.38 | 1899.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 1952.60 | 1914.20 | 1910.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 1971.70 | 1925.70 | 1915.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 2117.80 | 2142.65 | 2079.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 2117.80 | 2142.65 | 2079.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 2199.00 | 2222.90 | 2192.41 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 2137.80 | 2173.19 | 2177.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 2130.30 | 2164.61 | 2172.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 2149.40 | 2148.94 | 2162.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 2149.40 | 2148.94 | 2162.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 2149.40 | 2148.94 | 2162.63 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 2150.00 | 2138.51 | 2137.46 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 2113.50 | 2132.82 | 2135.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 2089.00 | 2116.73 | 2126.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 2125.10 | 2104.59 | 2110.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 10:15:00 | 2125.10 | 2104.59 | 2110.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2125.10 | 2104.59 | 2110.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 2125.10 | 2104.59 | 2110.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2106.00 | 2104.88 | 2109.80 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 2129.10 | 2114.99 | 2113.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 2157.40 | 2123.47 | 2117.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 2125.00 | 2126.36 | 2119.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 2125.00 | 2126.36 | 2119.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 2139.40 | 2142.71 | 2135.76 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 2067.30 | 2123.15 | 2128.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 2051.90 | 2083.26 | 2104.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 11:15:00 | 2076.20 | 2073.25 | 2091.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 2076.20 | 2073.25 | 2091.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 2076.20 | 2073.25 | 2091.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 2081.60 | 2073.25 | 2091.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2130.80 | 2083.15 | 2089.09 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 2132.60 | 2100.09 | 2096.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 2171.20 | 2114.31 | 2103.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 10:15:00 | 2323.40 | 2327.61 | 2267.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 10:30:00 | 2327.20 | 2327.61 | 2267.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 2289.90 | 2310.80 | 2281.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 2279.20 | 2310.80 | 2281.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 2300.60 | 2308.76 | 2283.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 2301.90 | 2308.76 | 2283.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2281.10 | 2301.75 | 2284.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 2281.10 | 2301.75 | 2284.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 2277.20 | 2296.84 | 2283.57 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 2240.00 | 2273.61 | 2275.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 13:15:00 | 2208.60 | 2250.05 | 2262.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 2240.60 | 2231.01 | 2244.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 12:15:00 | 2240.60 | 2231.01 | 2244.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 2240.60 | 2231.01 | 2244.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 2240.60 | 2231.01 | 2244.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 2248.00 | 2234.41 | 2245.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 2231.00 | 2233.73 | 2243.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 2277.30 | 2246.46 | 2247.54 | SL hit (close>static) qty=1.00 sl=2265.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 2287.30 | 2254.62 | 2251.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 2382.20 | 2287.26 | 2268.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 2978.80 | 3047.61 | 2917.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:30:00 | 2981.00 | 3047.61 | 2917.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 3044.60 | 3087.80 | 3046.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 3044.60 | 3087.80 | 3046.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 3056.30 | 3081.50 | 3047.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 3039.10 | 3081.50 | 3047.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 3025.60 | 3070.32 | 3045.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 3025.60 | 3070.32 | 3045.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 3034.30 | 3063.12 | 3044.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 3034.30 | 3063.12 | 3044.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 3008.40 | 3052.17 | 3040.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 3008.40 | 3052.17 | 3040.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 2989.00 | 3039.54 | 3036.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 2955.70 | 3039.54 | 3036.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 2941.40 | 3019.91 | 3027.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 14:15:00 | 2902.00 | 2951.01 | 2986.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 2941.00 | 2919.63 | 2955.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 2941.00 | 2919.63 | 2955.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 2941.00 | 2919.63 | 2955.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 3003.30 | 2919.63 | 2955.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2884.60 | 2898.99 | 2932.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 2869.00 | 2891.35 | 2923.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:45:00 | 2864.90 | 2859.05 | 2883.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 2962.20 | 2901.30 | 2898.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 2962.20 | 2901.30 | 2898.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 3027.80 | 2926.60 | 2910.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 3410.00 | 3427.91 | 3294.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 11:00:00 | 3555.90 | 3453.51 | 3318.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 11:30:00 | 3565.80 | 3466.41 | 3336.68 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 12:45:00 | 3544.10 | 3478.52 | 3353.98 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 13:15:00 | 3547.50 | 3478.52 | 3353.98 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 3409.00 | 3468.40 | 3425.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 3409.00 | 3468.40 | 3425.42 | SL hit (close<ema400) qty=1.00 sl=3425.42 alert=retest1 |

### Cycle 146 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 4082.10 | 4248.20 | 4252.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 14:15:00 | 3996.50 | 4151.31 | 4203.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 14:15:00 | 3945.00 | 3882.29 | 3919.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 3945.00 | 3882.29 | 3919.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 3945.00 | 3882.29 | 3919.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 3945.00 | 3882.29 | 3919.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 3930.00 | 3891.83 | 3920.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 3900.20 | 3891.83 | 3920.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 4000.60 | 3863.25 | 3859.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 4000.60 | 3863.25 | 3859.84 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 3828.00 | 3877.69 | 3880.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 3809.50 | 3864.05 | 3873.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 3805.70 | 3757.10 | 3789.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 3805.70 | 3757.10 | 3789.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 3805.70 | 3757.10 | 3789.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 3805.70 | 3757.10 | 3789.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 3818.00 | 3769.28 | 3792.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 3818.00 | 3769.28 | 3792.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 3791.00 | 3773.62 | 3792.05 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 3890.00 | 3813.06 | 3806.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 15:15:00 | 3911.00 | 3832.64 | 3816.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 4010.80 | 4073.28 | 3988.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 4010.80 | 4073.28 | 3988.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 4010.80 | 4073.28 | 3988.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 4006.00 | 4073.28 | 3988.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 3920.50 | 4042.72 | 3982.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 3920.50 | 4042.72 | 3982.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 3945.60 | 4023.30 | 3979.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:45:00 | 3927.50 | 4023.30 | 3979.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 4095.40 | 4002.42 | 3979.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 4178.20 | 4002.42 | 3979.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 4123.40 | 4026.62 | 3992.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 4110.10 | 4066.16 | 4021.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 3815.90 | 3993.30 | 4008.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 3815.90 | 3993.30 | 4008.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 3739.00 | 3804.53 | 3877.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 3469.30 | 3420.11 | 3543.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 3469.30 | 3420.11 | 3543.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3500.00 | 3436.09 | 3539.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 3534.90 | 3436.09 | 3539.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 3490.00 | 3446.87 | 3534.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 3407.00 | 3438.90 | 3523.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 3461.90 | 3425.71 | 3451.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 3477.70 | 3446.56 | 3457.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 3500.00 | 3466.76 | 3464.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 3500.00 | 3466.76 | 3464.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 3582.50 | 3489.91 | 3475.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 3508.40 | 3528.65 | 3506.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 15:15:00 | 3508.40 | 3528.65 | 3506.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 3508.40 | 3528.65 | 3506.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 3478.90 | 3528.65 | 3506.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 3497.40 | 3522.40 | 3505.66 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 3472.40 | 3495.93 | 3497.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 3467.90 | 3490.32 | 3494.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 3353.90 | 3309.04 | 3354.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 3353.90 | 3309.04 | 3354.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 3353.90 | 3309.04 | 3354.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 3353.90 | 3309.04 | 3354.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 3314.30 | 3310.09 | 3350.79 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 3444.20 | 3364.91 | 3358.64 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 3331.00 | 3368.82 | 3370.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 3310.50 | 3346.71 | 3359.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 3293.80 | 3279.18 | 3312.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 3293.80 | 3279.18 | 3312.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 3304.00 | 3284.14 | 3311.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 3300.00 | 3284.14 | 3311.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3291.80 | 3285.67 | 3309.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 3263.30 | 3283.62 | 3306.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:30:00 | 3271.20 | 3280.83 | 3303.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 3326.00 | 3305.20 | 3303.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 3326.00 | 3305.20 | 3303.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 3347.50 | 3313.66 | 3307.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 3298.50 | 3319.04 | 3312.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 3298.50 | 3319.04 | 3312.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 3298.50 | 3319.04 | 3312.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 3298.50 | 3319.04 | 3312.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 3292.50 | 3313.73 | 3311.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 3297.70 | 3313.73 | 3311.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 3298.90 | 3307.43 | 3308.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 3288.80 | 3301.64 | 3305.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 3215.00 | 3206.87 | 3237.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:00:00 | 3215.00 | 3206.87 | 3237.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 3240.00 | 3214.40 | 3235.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 3239.60 | 3214.40 | 3235.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 3201.50 | 3211.82 | 3232.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 3180.50 | 3205.07 | 3227.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:00:00 | 3178.10 | 3205.07 | 3227.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 3113.90 | 3199.31 | 3215.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 3187.00 | 3194.83 | 3203.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 3187.00 | 3193.26 | 3202.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 3159.80 | 3193.26 | 3202.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 3150.40 | 3184.69 | 3197.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 3133.00 | 3184.69 | 3197.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:00:00 | 3143.00 | 3176.35 | 3192.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 3021.47 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 3019.19 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 2958.20 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 3027.65 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 2976.35 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 2985.85 | 3122.44 | 3158.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 3081.00 | 3053.48 | 3093.14 | SL hit (close>ema200) qty=0.50 sl=3053.48 alert=retest2 |

### Cycle 157 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 3109.80 | 3103.66 | 3103.02 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 3079.00 | 3098.32 | 3100.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 3051.00 | 3087.20 | 3095.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 3109.00 | 3091.56 | 3096.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 3109.00 | 3091.56 | 3096.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 3109.00 | 3091.56 | 3096.37 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 3149.10 | 3105.03 | 3101.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 3188.40 | 3155.79 | 3133.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 3258.90 | 3275.03 | 3226.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 3258.90 | 3275.03 | 3226.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3222.10 | 3263.63 | 3244.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 3224.30 | 3263.63 | 3244.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 3215.50 | 3254.00 | 3242.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 3209.70 | 3254.00 | 3242.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 3216.20 | 3241.80 | 3238.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 3214.70 | 3241.80 | 3238.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 14:15:00 | 3206.00 | 3230.34 | 3233.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 3175.00 | 3219.27 | 3228.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 3224.00 | 3167.99 | 3188.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 3224.00 | 3167.99 | 3188.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 3224.00 | 3167.99 | 3188.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 3222.00 | 3167.99 | 3188.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 3227.00 | 3179.80 | 3192.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 3227.00 | 3179.80 | 3192.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 3235.80 | 3200.33 | 3199.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 3259.90 | 3212.24 | 3204.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 3248.50 | 3257.34 | 3239.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 3248.50 | 3257.34 | 3239.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 3248.50 | 3257.34 | 3239.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:45:00 | 3257.00 | 3257.34 | 3239.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 3232.00 | 3250.70 | 3239.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 3236.10 | 3250.70 | 3239.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 3232.70 | 3247.10 | 3238.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 3232.70 | 3247.10 | 3238.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 3225.80 | 3242.84 | 3237.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 3225.80 | 3242.84 | 3237.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 3234.00 | 3238.26 | 3236.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 3243.40 | 3238.26 | 3236.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 3243.00 | 3239.21 | 3236.79 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 3206.80 | 3230.60 | 3233.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 3197.20 | 3216.98 | 3225.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 3106.90 | 3101.05 | 3126.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 3106.90 | 3101.05 | 3126.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 3090.60 | 3097.62 | 3118.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 3078.60 | 3094.04 | 3105.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 3079.30 | 3091.09 | 3103.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 3077.10 | 3087.14 | 3099.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 3161.30 | 3066.40 | 3073.82 | SL hit (close>static) qty=1.00 sl=3123.80 alert=retest2 |

### Cycle 163 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 3252.80 | 3103.68 | 3090.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 3347.60 | 3152.46 | 3113.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 3311.60 | 3325.25 | 3265.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 12:15:00 | 3391.40 | 3344.42 | 3284.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 3396.40 | 3363.09 | 3322.13 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 13:45:00 | 3415.50 | 3380.66 | 3341.34 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 14:30:00 | 3405.30 | 3383.35 | 3346.14 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3403.50 | 3390.06 | 3355.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 3366.80 | 3390.06 | 3355.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 3309.00 | 3374.31 | 3362.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 3309.00 | 3374.31 | 3362.04 | SL hit (close<ema400) qty=1.00 sl=3362.04 alert=retest1 |

### Cycle 164 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 3294.30 | 3347.81 | 3351.62 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 3359.10 | 3350.83 | 3350.25 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 3322.30 | 3345.12 | 3347.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 3315.90 | 3339.28 | 3344.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 3306.00 | 3253.04 | 3283.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 3306.00 | 3253.04 | 3283.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3306.00 | 3253.04 | 3283.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:30:00 | 3283.40 | 3253.04 | 3283.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 3295.00 | 3261.43 | 3284.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 3312.60 | 3261.43 | 3284.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 3294.00 | 3270.96 | 3285.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 3294.00 | 3270.96 | 3285.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 3273.30 | 3271.43 | 3283.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:30:00 | 3276.90 | 3271.43 | 3283.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 3348.10 | 3286.77 | 3289.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 3348.10 | 3286.77 | 3289.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 15:15:00 | 3377.90 | 3304.99 | 3297.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 09:15:00 | 3544.90 | 3352.97 | 3320.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 14:15:00 | 3413.90 | 3424.09 | 3375.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 15:00:00 | 3413.90 | 3424.09 | 3375.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 3412.00 | 3421.67 | 3378.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 3339.80 | 3421.67 | 3378.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 3311.50 | 3399.64 | 3372.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 3266.20 | 3399.64 | 3372.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 3276.80 | 3375.07 | 3363.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 3276.80 | 3375.07 | 3363.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 3290.50 | 3345.66 | 3351.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 3260.00 | 3328.53 | 3343.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 15:15:00 | 3156.90 | 3145.95 | 3187.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-23 09:15:00 | 3167.90 | 3145.95 | 3187.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 3149.30 | 3146.62 | 3183.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 3159.60 | 3146.62 | 3183.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3147.40 | 3072.98 | 3100.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 3147.40 | 3072.98 | 3100.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 3162.70 | 3090.92 | 3106.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 3162.70 | 3090.92 | 3106.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 3147.00 | 3121.10 | 3117.79 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 3079.50 | 3111.68 | 3114.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 13:15:00 | 3073.50 | 3093.39 | 3104.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 3161.50 | 3098.38 | 3103.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 3161.50 | 3098.38 | 3103.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 3161.50 | 3098.38 | 3103.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 3158.90 | 3098.38 | 3103.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 3150.40 | 3108.78 | 3107.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 3217.10 | 3147.78 | 3128.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 3195.00 | 3264.35 | 3214.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 10:15:00 | 3195.00 | 3264.35 | 3214.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 3195.00 | 3264.35 | 3214.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 3195.00 | 3264.35 | 3214.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 3180.80 | 3247.64 | 3211.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 3180.80 | 3247.64 | 3211.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 3152.50 | 3228.62 | 3205.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 3161.00 | 3228.62 | 3205.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 3205.00 | 3208.61 | 3200.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 3245.00 | 3208.61 | 3200.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 3229.00 | 3219.80 | 3207.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:15:00 | 3232.00 | 3212.87 | 3211.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 3144.00 | 3202.16 | 3207.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 3144.00 | 3202.16 | 3207.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 3102.20 | 3182.17 | 3197.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 3105.90 | 3100.50 | 3133.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 3105.90 | 3100.50 | 3133.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3105.90 | 3100.50 | 3133.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 3105.90 | 3100.50 | 3133.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 3225.00 | 3129.88 | 3138.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 3244.00 | 3129.88 | 3138.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 3212.30 | 3146.36 | 3145.65 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 3153.80 | 3172.23 | 3173.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 3066.70 | 3150.45 | 3163.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 12:15:00 | 3184.40 | 3142.04 | 3154.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 12:15:00 | 3184.40 | 3142.04 | 3154.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 3184.40 | 3142.04 | 3154.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:45:00 | 3179.40 | 3142.04 | 3154.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 3182.30 | 3150.09 | 3157.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:30:00 | 3184.00 | 3150.09 | 3157.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 3216.00 | 3163.27 | 3162.70 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 3120.10 | 3156.51 | 3159.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 3105.00 | 3131.82 | 3145.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 3087.00 | 3085.65 | 3107.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 3087.00 | 3085.65 | 3107.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 3087.00 | 3085.65 | 3107.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:45:00 | 3068.00 | 3081.92 | 3103.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:00:00 | 3073.50 | 3080.24 | 3100.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 3146.90 | 3101.10 | 3103.62 | SL hit (close>static) qty=1.00 sl=3119.90 alert=retest2 |

### Cycle 177 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 3289.10 | 3138.70 | 3120.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 3440.90 | 3199.14 | 3149.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 09:15:00 | 3579.00 | 3601.98 | 3512.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 3548.20 | 3585.04 | 3538.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 3548.20 | 3585.04 | 3538.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 3548.20 | 3585.04 | 3538.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 3558.00 | 3579.63 | 3540.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 3506.70 | 3579.63 | 3540.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 3477.00 | 3559.11 | 3534.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:45:00 | 3469.50 | 3559.11 | 3534.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 3485.00 | 3544.28 | 3529.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 12:00:00 | 3518.30 | 3539.09 | 3528.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 3522.70 | 3531.61 | 3526.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-27 10:15:00 | 3870.13 | 3724.69 | 3667.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 3624.50 | 3719.24 | 3729.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 3532.50 | 3681.89 | 3711.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 3195.00 | 3188.17 | 3271.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 14:45:00 | 3214.20 | 3188.17 | 3271.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3275.10 | 3209.82 | 3267.17 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 3338.80 | 3287.60 | 3287.13 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 3258.40 | 3285.66 | 3286.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 3248.00 | 3270.93 | 3279.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 14:15:00 | 3276.00 | 3271.95 | 3278.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 3276.00 | 3271.95 | 3278.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 3276.00 | 3271.95 | 3278.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 3276.00 | 3271.95 | 3278.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 3285.00 | 3274.56 | 3279.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 3243.30 | 3274.56 | 3279.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 3290.00 | 3223.28 | 3216.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 3290.00 | 3223.28 | 3216.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 3304.70 | 3265.92 | 3242.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 3299.00 | 3310.83 | 3282.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 15:00:00 | 3299.00 | 3310.83 | 3282.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 3290.00 | 3306.67 | 3282.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 3254.50 | 3306.67 | 3282.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3256.80 | 3296.69 | 3280.41 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 3256.80 | 3271.71 | 3272.80 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 3283.30 | 3272.88 | 3272.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 3314.50 | 3281.21 | 3276.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 3282.00 | 3286.35 | 3280.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 3282.00 | 3286.35 | 3280.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 3282.00 | 3286.35 | 3280.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 3282.00 | 3286.35 | 3280.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 3285.00 | 3286.08 | 3281.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 3197.00 | 3286.08 | 3281.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 3139.00 | 3256.66 | 3268.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 3109.10 | 3227.15 | 3253.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 3156.30 | 3137.80 | 3187.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 3156.30 | 3137.80 | 3187.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 3156.30 | 3137.80 | 3187.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 3143.30 | 3137.80 | 3187.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 3248.60 | 3187.63 | 3190.08 | SL hit (close>static) qty=1.00 sl=3232.30 alert=retest2 |

### Cycle 185 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 3250.00 | 3200.11 | 3195.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 3292.50 | 3218.59 | 3204.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3160.00 | 3227.62 | 3217.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 3160.00 | 3227.62 | 3217.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 3160.00 | 3227.62 | 3217.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 3160.00 | 3227.62 | 3217.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3153.00 | 3212.69 | 3211.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 3153.00 | 3212.69 | 3211.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 3176.60 | 3205.47 | 3208.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 3135.80 | 3171.58 | 3189.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3186.00 | 3144.58 | 3164.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3186.00 | 3144.58 | 3164.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3186.00 | 3144.58 | 3164.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 3161.90 | 3155.20 | 3166.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 3147.90 | 3160.21 | 3166.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 3182.80 | 3152.82 | 3149.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 3182.80 | 3152.82 | 3149.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 3213.00 | 3171.75 | 3159.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 3348.10 | 3374.37 | 3342.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 14:15:00 | 3348.10 | 3374.37 | 3342.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 3348.10 | 3374.37 | 3342.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:45:00 | 3357.00 | 3374.37 | 3342.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 3343.00 | 3368.09 | 3342.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 3290.30 | 3368.09 | 3342.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3285.00 | 3351.47 | 3337.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 3308.80 | 3342.38 | 3334.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 3308.80 | 3325.93 | 3328.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 3308.80 | 3325.93 | 3328.11 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 3460.00 | 3348.85 | 3337.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 3508.30 | 3380.74 | 3353.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 3773.00 | 3780.55 | 3695.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 3891.00 | 3806.42 | 3750.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:15:00 | 4085.55 | 3945.03 | 3862.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 3985.90 | 4005.18 | 3942.82 | SL hit (close<ema200) qty=0.50 sl=4005.18 alert=retest1 |

### Cycle 190 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 3795.00 | 3910.64 | 3921.54 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 3917.00 | 3900.08 | 3898.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 4065.00 | 3937.53 | 3916.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 3994.70 | 4016.16 | 3985.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:45:00 | 4002.60 | 4016.16 | 3985.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 4017.50 | 4016.43 | 3988.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 4060.00 | 4016.43 | 3988.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 10:30:00 | 4032.20 | 4019.27 | 3997.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:30:00 | 4041.60 | 4026.30 | 4002.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 3808.10 | 3998.14 | 4001.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 09:15:00 | 3808.10 | 3998.14 | 4001.01 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 4139.10 | 3988.68 | 3977.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 4174.00 | 4084.00 | 4031.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 13:15:00 | 4305.80 | 4330.56 | 4253.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:15:00 | 4436.20 | 4322.36 | 4262.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 4372.20 | 4348.60 | 4285.78 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-15 09:15:00 | 1560.00 | 2024-04-16 13:15:00 | 1626.00 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2024-04-19 13:45:00 | 1684.80 | 2024-04-24 11:15:00 | 1657.20 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-04-22 10:45:00 | 1679.95 | 2024-04-24 11:15:00 | 1657.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-04-23 09:30:00 | 1684.90 | 2024-04-24 11:15:00 | 1657.20 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-04-23 10:00:00 | 1682.00 | 2024-04-24 11:15:00 | 1657.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-04-24 09:15:00 | 1672.00 | 2024-04-24 11:15:00 | 1657.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-05-09 09:15:00 | 1967.90 | 2024-05-15 09:15:00 | 2164.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 11:00:00 | 1969.85 | 2024-05-15 09:15:00 | 2166.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-24 12:30:00 | 2134.70 | 2024-05-27 09:15:00 | 2170.65 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-05-31 11:30:00 | 2226.00 | 2024-06-03 09:15:00 | 2448.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 14:30:00 | 2448.25 | 2024-06-19 09:15:00 | 2667.23 | TARGET_HIT | 1.00 | 8.94% |
| BUY | retest2 | 2024-06-11 15:15:00 | 2424.75 | 2024-06-19 14:15:00 | 2693.08 | TARGET_HIT | 1.00 | 11.07% |
| BUY | retest2 | 2024-06-13 09:45:00 | 2447.90 | 2024-06-19 14:15:00 | 2692.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-28 09:15:00 | 2624.95 | 2024-06-28 11:15:00 | 2573.35 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-07-04 12:15:00 | 2617.00 | 2024-07-05 14:15:00 | 2647.10 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-07-05 09:30:00 | 2617.90 | 2024-07-05 14:15:00 | 2647.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-07-16 13:30:00 | 2474.10 | 2024-07-22 09:15:00 | 2350.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 13:30:00 | 2474.10 | 2024-07-22 09:15:00 | 2440.40 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2024-07-18 09:15:00 | 2465.00 | 2024-07-22 09:15:00 | 2341.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 09:15:00 | 2465.00 | 2024-07-22 09:15:00 | 2440.40 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2024-07-18 09:45:00 | 2468.35 | 2024-07-22 09:15:00 | 2344.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 09:45:00 | 2468.35 | 2024-07-22 09:15:00 | 2440.40 | STOP_HIT | 0.50 | 1.13% |
| SELL | retest2 | 2024-07-31 10:00:00 | 2324.00 | 2024-08-02 11:15:00 | 2386.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-07-31 10:30:00 | 2318.35 | 2024-08-02 11:15:00 | 2386.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-08-01 10:30:00 | 2325.05 | 2024-08-02 11:15:00 | 2386.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-08-01 11:30:00 | 2322.90 | 2024-08-02 11:15:00 | 2386.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-08-08 09:30:00 | 2250.10 | 2024-08-08 13:15:00 | 2261.15 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-08-23 10:15:00 | 2517.00 | 2024-08-29 10:15:00 | 2545.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2024-08-26 10:15:00 | 2507.10 | 2024-08-29 10:15:00 | 2545.00 | STOP_HIT | 1.00 | 1.51% |
| BUY | retest2 | 2024-08-26 11:15:00 | 2499.85 | 2024-08-29 10:15:00 | 2545.00 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2024-09-12 09:15:00 | 2720.00 | 2024-09-18 11:15:00 | 2689.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-09-12 10:30:00 | 2714.65 | 2024-09-18 11:15:00 | 2689.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-09-12 15:00:00 | 2724.60 | 2024-09-18 11:15:00 | 2689.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-09-13 15:15:00 | 2714.00 | 2024-09-18 11:15:00 | 2689.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-09-16 09:45:00 | 2795.00 | 2024-09-18 11:15:00 | 2689.00 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2024-09-23 15:15:00 | 2600.00 | 2024-09-25 12:15:00 | 2482.02 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2024-09-23 15:15:00 | 2600.00 | 2024-10-01 10:15:00 | 2524.45 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2024-09-24 09:30:00 | 2595.50 | 2024-10-01 11:15:00 | 2470.00 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2024-09-25 10:00:00 | 2612.65 | 2024-10-01 11:15:00 | 2465.72 | PARTIAL | 0.50 | 5.62% |
| SELL | retest2 | 2024-09-24 09:30:00 | 2595.50 | 2024-10-01 14:15:00 | 2509.85 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2024-09-25 10:00:00 | 2612.65 | 2024-10-01 14:15:00 | 2509.85 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-10-18 09:15:00 | 2503.65 | 2024-10-18 13:15:00 | 2651.30 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2024-10-29 10:30:00 | 2515.20 | 2024-10-30 09:15:00 | 2635.00 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2024-11-06 09:15:00 | 2799.90 | 2024-11-13 09:15:00 | 2729.05 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-11-21 13:00:00 | 2818.95 | 2024-11-28 14:15:00 | 2804.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-11-25 09:15:00 | 2876.90 | 2024-11-28 14:15:00 | 2804.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-11-25 12:30:00 | 2823.40 | 2024-11-28 14:15:00 | 2804.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-11-25 13:15:00 | 2814.50 | 2024-11-28 14:15:00 | 2804.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-12-06 09:15:00 | 2946.00 | 2024-12-11 10:15:00 | 2893.90 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-12-06 14:00:00 | 2915.05 | 2024-12-11 10:15:00 | 2893.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-12-06 15:15:00 | 2914.95 | 2024-12-11 10:15:00 | 2893.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-12-11 09:15:00 | 2921.10 | 2024-12-11 10:15:00 | 2893.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-01-07 12:15:00 | 2813.75 | 2025-01-07 14:15:00 | 2832.70 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-01-09 11:30:00 | 2777.95 | 2025-01-10 09:15:00 | 2639.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:00:00 | 2779.00 | 2025-01-10 09:15:00 | 2640.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:30:00 | 2777.95 | 2025-01-13 10:15:00 | 2500.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 12:00:00 | 2779.00 | 2025-01-13 10:15:00 | 2501.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-07 10:45:00 | 1817.20 | 2025-02-07 15:15:00 | 1755.00 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2025-02-07 12:15:00 | 1825.50 | 2025-02-07 15:15:00 | 1755.00 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2025-02-24 11:15:00 | 1554.95 | 2025-02-27 10:15:00 | 1512.70 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-02-24 14:30:00 | 1543.85 | 2025-02-27 10:15:00 | 1512.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-03-12 09:15:00 | 1514.15 | 2025-03-17 12:15:00 | 1438.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 09:45:00 | 1511.00 | 2025-03-17 12:15:00 | 1435.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 09:15:00 | 1514.15 | 2025-03-18 09:15:00 | 1491.00 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-03-12 09:45:00 | 1511.00 | 2025-03-18 09:15:00 | 1491.00 | STOP_HIT | 0.50 | 1.32% |
| BUY | retest2 | 2025-03-21 10:45:00 | 1566.95 | 2025-03-26 12:15:00 | 1566.30 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-03-28 13:15:00 | 1525.00 | 2025-04-02 14:15:00 | 1549.70 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-03-28 14:00:00 | 1526.60 | 2025-04-02 14:15:00 | 1549.70 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-03-28 15:00:00 | 1515.00 | 2025-04-02 14:15:00 | 1549.70 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-04-04 09:15:00 | 1471.10 | 2025-04-07 09:15:00 | 1323.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 12:30:00 | 1504.00 | 2025-04-17 14:15:00 | 1497.40 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-05-12 09:15:00 | 1672.30 | 2025-05-15 09:15:00 | 1839.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-22 11:30:00 | 1811.30 | 2025-05-26 09:15:00 | 1929.90 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2025-05-22 12:30:00 | 1813.30 | 2025-05-26 09:15:00 | 1929.90 | STOP_HIT | 1.00 | -6.43% |
| SELL | retest2 | 2025-05-23 09:45:00 | 1806.30 | 2025-05-26 09:15:00 | 1929.90 | STOP_HIT | 1.00 | -6.84% |
| BUY | retest2 | 2025-05-30 11:15:00 | 1992.60 | 2025-06-02 09:15:00 | 1971.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-30 13:00:00 | 1994.00 | 2025-06-02 09:15:00 | 1971.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1746.00 | 2025-06-20 14:15:00 | 1762.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1836.20 | 2025-07-04 10:15:00 | 1831.60 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-09 12:30:00 | 1797.80 | 2025-07-10 09:15:00 | 1821.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1960.70 | 2025-07-17 09:15:00 | 1942.10 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-09-01 15:00:00 | 2231.00 | 2025-09-02 10:15:00 | 2277.30 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-16 12:15:00 | 2869.00 | 2025-09-18 09:15:00 | 2962.20 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-09-17 13:45:00 | 2864.90 | 2025-09-18 09:15:00 | 2962.20 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest1 | 2025-09-23 11:00:00 | 3555.90 | 2025-09-24 14:15:00 | 3409.00 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest1 | 2025-09-23 11:30:00 | 3565.80 | 2025-09-24 14:15:00 | 3409.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest1 | 2025-09-23 12:45:00 | 3544.10 | 2025-09-24 14:15:00 | 3409.00 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest1 | 2025-09-23 13:15:00 | 3547.50 | 2025-09-24 14:15:00 | 3409.00 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-09-25 11:15:00 | 3564.80 | 2025-10-01 09:15:00 | 3921.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-25 11:45:00 | 3586.30 | 2025-10-01 09:15:00 | 3944.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 11:00:00 | 3560.70 | 2025-10-01 09:15:00 | 3916.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 12:15:00 | 3564.80 | 2025-10-01 09:15:00 | 3921.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-30 09:15:00 | 3733.10 | 2025-10-01 10:15:00 | 4106.41 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-16 09:15:00 | 3900.20 | 2025-10-20 09:15:00 | 4000.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-10-30 11:15:00 | 4178.20 | 2025-11-03 09:15:00 | 3815.90 | STOP_HIT | 1.00 | -8.67% |
| BUY | retest2 | 2025-10-30 12:00:00 | 4123.40 | 2025-11-03 09:15:00 | 3815.90 | STOP_HIT | 1.00 | -7.46% |
| BUY | retest2 | 2025-10-30 14:45:00 | 4110.10 | 2025-11-03 09:15:00 | 3815.90 | STOP_HIT | 1.00 | -7.16% |
| SELL | retest2 | 2025-11-07 15:00:00 | 3407.00 | 2025-11-11 15:15:00 | 3500.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-11-11 11:15:00 | 3461.90 | 2025-11-11 15:15:00 | 3500.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-11-11 12:45:00 | 3477.70 | 2025-11-11 15:15:00 | 3500.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-25 10:45:00 | 3263.30 | 2025-11-27 11:15:00 | 3326.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-11-25 11:30:00 | 3271.20 | 2025-11-27 11:15:00 | 3326.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-12-03 10:30:00 | 3180.50 | 2025-12-05 15:15:00 | 3021.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 11:00:00 | 3178.10 | 2025-12-05 15:15:00 | 3019.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 09:15:00 | 3113.90 | 2025-12-05 15:15:00 | 2958.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 15:15:00 | 3187.00 | 2025-12-05 15:15:00 | 3027.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 10:15:00 | 3133.00 | 2025-12-05 15:15:00 | 2976.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 11:00:00 | 3143.00 | 2025-12-05 15:15:00 | 2985.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 10:30:00 | 3180.50 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-12-03 11:00:00 | 3178.10 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2025-12-04 09:15:00 | 3113.90 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2025-12-04 15:15:00 | 3187.00 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2025-12-05 10:15:00 | 3133.00 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2025-12-05 11:00:00 | 3143.00 | 2025-12-09 10:15:00 | 3081.00 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-12-10 09:45:00 | 3141.00 | 2025-12-10 11:15:00 | 3109.80 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2026-01-01 12:00:00 | 3078.60 | 2026-01-05 09:15:00 | 3161.30 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-01-01 13:00:00 | 3079.30 | 2026-01-05 09:15:00 | 3161.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-01-01 15:00:00 | 3077.10 | 2026-01-05 09:15:00 | 3161.30 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest1 | 2026-01-07 12:15:00 | 3391.40 | 2026-01-09 14:15:00 | 3309.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest1 | 2026-01-08 11:15:00 | 3396.40 | 2026-01-09 14:15:00 | 3309.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest1 | 2026-01-08 13:45:00 | 3415.50 | 2026-01-09 14:15:00 | 3309.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest1 | 2026-01-08 14:30:00 | 3405.30 | 2026-01-09 14:15:00 | 3309.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-01-12 09:15:00 | 3358.60 | 2026-01-12 10:15:00 | 3294.30 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-02-03 09:15:00 | 3245.00 | 2026-02-05 09:15:00 | 3144.00 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2026-02-03 10:30:00 | 3229.00 | 2026-02-05 09:15:00 | 3144.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-02-04 15:15:00 | 3232.00 | 2026-02-05 09:15:00 | 3144.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-02-17 10:45:00 | 3068.00 | 2026-02-18 09:15:00 | 3146.90 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-02-17 12:00:00 | 3073.50 | 2026-02-18 09:15:00 | 3146.90 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-02-24 12:00:00 | 3518.30 | 2026-02-27 10:15:00 | 3870.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-24 13:15:00 | 3522.70 | 2026-02-27 10:15:00 | 3874.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 3243.30 | 2026-03-17 09:15:00 | 3290.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-03-24 10:15:00 | 3143.30 | 2026-03-25 09:15:00 | 3248.60 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2026-04-01 11:45:00 | 3161.90 | 2026-04-06 10:15:00 | 3182.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-04-01 13:30:00 | 3147.90 | 2026-04-06 10:15:00 | 3182.80 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-04-13 10:45:00 | 3308.80 | 2026-04-13 13:15:00 | 3308.80 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest1 | 2026-04-21 10:00:00 | 3891.00 | 2026-04-22 10:15:00 | 4085.55 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-21 10:00:00 | 3891.00 | 2026-04-23 10:15:00 | 3985.90 | STOP_HIT | 0.50 | 2.44% |
| BUY | retest2 | 2026-04-29 15:15:00 | 4060.00 | 2026-05-04 09:15:00 | 3808.10 | STOP_HIT | 1.00 | -6.20% |
| BUY | retest2 | 2026-04-30 10:30:00 | 4032.20 | 2026-05-04 09:15:00 | 3808.10 | STOP_HIT | 1.00 | -5.56% |
| BUY | retest2 | 2026-04-30 11:30:00 | 4041.60 | 2026-05-04 09:15:00 | 3808.10 | STOP_HIT | 1.00 | -5.78% |
