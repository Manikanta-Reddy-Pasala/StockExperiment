# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1669.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 205 |
| ALERT1 | 138 |
| ALERT2 | 135 |
| ALERT2_SKIP | 97 |
| ALERT3 | 274 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 121 |
| PARTIAL | 25 |
| TARGET_HIT | 6 |
| STOP_HIT | 118 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 149 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 74 / 75
- **Target hits / Stop hits / Partials:** 6 / 118 / 25
- **Avg / median % per leg:** 1.12% / -0.06%
- **Sum % (uncompounded):** 166.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 13 | 32.5% | 3 | 37 | 0 | -0.14% | -5.5% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.81% | -2.4% |
| BUY @ 3rd Alert (retest2) | 37 | 13 | 35.1% | 3 | 34 | 0 | -0.08% | -3.0% |
| SELL (all) | 109 | 61 | 56.0% | 3 | 81 | 25 | 1.58% | 172.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 109 | 61 | 56.0% | 3 | 81 | 25 | 1.58% | 172.2% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.81% | -2.4% |
| retest2 (combined) | 146 | 74 | 50.7% | 6 | 115 | 25 | 1.16% | 169.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 09:15:00 | 671.85 | 666.79 | 666.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 11:15:00 | 673.78 | 668.92 | 667.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 09:15:00 | 682.70 | 684.15 | 678.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 725.53 | 735.06 | 726.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 725.53 | 735.06 | 726.22 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 14:15:00 | 706.00 | 722.18 | 722.86 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 753.58 | 726.39 | 724.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 766.00 | 734.32 | 728.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 795.78 | 800.57 | 781.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 09:15:00 | 799.98 | 803.87 | 791.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 799.98 | 803.87 | 791.38 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 14:15:00 | 837.45 | 851.69 | 852.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 10:15:00 | 829.00 | 842.50 | 847.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-15 13:15:00 | 819.93 | 812.65 | 823.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 825.00 | 816.07 | 822.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 825.00 | 816.07 | 822.31 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 15:15:00 | 840.00 | 826.78 | 825.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 842.28 | 829.88 | 826.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 12:15:00 | 879.48 | 880.75 | 862.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 15:15:00 | 883.90 | 879.64 | 866.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 883.90 | 879.64 | 866.41 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 859.20 | 870.98 | 872.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 13:15:00 | 854.10 | 865.30 | 868.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 859.90 | 859.23 | 863.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 14:15:00 | 865.00 | 860.39 | 863.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 865.00 | 860.39 | 863.28 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-07-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 13:15:00 | 836.85 | 822.77 | 820.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 13:15:00 | 847.23 | 837.63 | 830.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 15:15:00 | 838.18 | 839.40 | 832.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 09:15:00 | 869.98 | 881.25 | 869.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 869.98 | 881.25 | 869.58 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 849.68 | 866.43 | 867.64 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 889.18 | 868.20 | 866.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 12:15:00 | 899.38 | 880.67 | 873.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 899.98 | 900.61 | 888.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 11:15:00 | 890.33 | 899.60 | 894.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 890.33 | 899.60 | 894.44 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 11:15:00 | 886.55 | 892.74 | 893.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 14:15:00 | 882.33 | 889.32 | 891.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 14:15:00 | 888.98 | 880.78 | 884.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 14:15:00 | 888.98 | 880.78 | 884.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 888.98 | 880.78 | 884.87 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 867.70 | 854.03 | 853.64 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 849.65 | 856.71 | 856.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 838.78 | 851.45 | 854.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 14:15:00 | 853.33 | 851.83 | 854.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 15:15:00 | 846.00 | 850.66 | 853.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 15:15:00 | 846.00 | 850.66 | 853.44 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 13:15:00 | 860.00 | 854.94 | 854.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 865.28 | 858.70 | 856.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 11:15:00 | 857.68 | 859.59 | 857.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 11:15:00 | 857.68 | 859.59 | 857.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 857.68 | 859.59 | 857.33 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 14:15:00 | 850.85 | 855.77 | 855.95 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 879.58 | 859.85 | 857.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 10:15:00 | 886.50 | 865.18 | 860.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 09:15:00 | 879.00 | 881.62 | 875.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 879.00 | 881.62 | 875.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 879.00 | 881.62 | 875.91 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 12:15:00 | 874.13 | 877.14 | 877.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 13:15:00 | 869.28 | 875.57 | 876.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 14:15:00 | 850.35 | 847.86 | 855.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 14:15:00 | 826.98 | 821.89 | 827.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 826.98 | 821.89 | 827.28 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 10:15:00 | 815.00 | 808.53 | 808.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 11:15:00 | 815.83 | 811.02 | 810.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 10:15:00 | 819.88 | 821.60 | 818.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 10:15:00 | 819.88 | 821.60 | 818.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 819.88 | 821.60 | 818.25 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 14:15:00 | 806.35 | 816.66 | 816.83 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 13:15:00 | 825.83 | 815.95 | 814.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 14:15:00 | 831.60 | 819.08 | 816.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 11:15:00 | 840.20 | 842.23 | 833.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 12:15:00 | 832.78 | 840.34 | 833.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 832.78 | 840.34 | 833.84 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 11:15:00 | 856.55 | 864.86 | 865.88 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 886.35 | 869.49 | 867.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 901.33 | 884.88 | 877.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 900.45 | 919.40 | 910.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 900.45 | 919.40 | 910.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 900.45 | 919.40 | 910.97 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 10:15:00 | 904.45 | 914.11 | 914.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 11:15:00 | 901.03 | 911.49 | 913.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 11:15:00 | 902.50 | 900.16 | 905.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 14:15:00 | 900.18 | 900.99 | 904.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 900.18 | 900.99 | 904.49 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 12:15:00 | 904.45 | 899.47 | 899.19 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 15:15:00 | 896.93 | 898.65 | 898.86 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 15:15:00 | 907.50 | 898.39 | 898.13 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 09:15:00 | 891.93 | 897.10 | 897.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 10:15:00 | 885.60 | 894.80 | 896.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 14:15:00 | 900.48 | 894.44 | 895.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 14:15:00 | 900.48 | 894.44 | 895.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 900.48 | 894.44 | 895.55 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 15:15:00 | 909.63 | 897.48 | 896.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 13:15:00 | 926.65 | 914.56 | 906.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 15:15:00 | 925.00 | 927.63 | 919.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 15:15:00 | 925.00 | 927.63 | 919.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 925.00 | 927.63 | 919.96 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 09:15:00 | 1055.50 | 1064.24 | 1064.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 10:15:00 | 1045.70 | 1060.53 | 1062.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 14:15:00 | 1027.38 | 1026.87 | 1035.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 14:15:00 | 1030.80 | 1020.19 | 1026.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 1030.80 | 1020.19 | 1026.78 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 12:15:00 | 977.13 | 968.84 | 968.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 13:15:00 | 983.98 | 971.87 | 970.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-27 15:15:00 | 967.50 | 971.72 | 970.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 15:15:00 | 967.50 | 971.72 | 970.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 15:15:00 | 967.50 | 971.72 | 970.39 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 10:15:00 | 976.50 | 984.00 | 984.09 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 14:15:00 | 987.73 | 983.85 | 983.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 995.35 | 986.74 | 985.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 983.20 | 988.78 | 987.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 14:15:00 | 983.20 | 988.78 | 987.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 983.20 | 988.78 | 987.21 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 13:15:00 | 1181.47 | 1197.00 | 1197.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 14:15:00 | 1171.60 | 1191.92 | 1195.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 1200.00 | 1190.43 | 1193.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 1200.00 | 1190.43 | 1193.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1200.00 | 1190.43 | 1193.93 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 11:15:00 | 1220.97 | 1199.08 | 1197.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 12:15:00 | 1296.00 | 1218.47 | 1206.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 09:15:00 | 1257.58 | 1260.78 | 1234.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 1266.55 | 1287.69 | 1277.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 1266.55 | 1287.69 | 1277.39 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 13:15:00 | 1258.45 | 1271.76 | 1272.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 14:15:00 | 1244.53 | 1266.32 | 1269.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 1275.22 | 1265.40 | 1268.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 1275.22 | 1265.40 | 1268.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 1275.22 | 1265.40 | 1268.51 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 11:15:00 | 1288.70 | 1272.20 | 1271.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 13:15:00 | 1307.50 | 1282.35 | 1276.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 1274.50 | 1286.45 | 1280.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 1274.50 | 1286.45 | 1280.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 1274.50 | 1286.45 | 1280.12 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 12:15:00 | 1300.30 | 1306.58 | 1307.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 15:15:00 | 1296.00 | 1302.37 | 1305.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 1294.25 | 1288.53 | 1294.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 1294.25 | 1288.53 | 1294.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 1294.25 | 1288.53 | 1294.72 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-12-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 15:15:00 | 1305.90 | 1298.54 | 1297.78 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 1275.43 | 1294.04 | 1296.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 14:15:00 | 1269.10 | 1285.82 | 1291.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 1286.63 | 1284.41 | 1289.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 1286.63 | 1284.41 | 1289.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 1286.63 | 1284.41 | 1289.99 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 1271.88 | 1265.73 | 1264.99 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 15:15:00 | 1245.80 | 1262.25 | 1263.68 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 10:15:00 | 1270.03 | 1265.04 | 1264.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 11:15:00 | 1290.15 | 1270.07 | 1267.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 14:15:00 | 1269.10 | 1274.64 | 1270.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 14:15:00 | 1269.10 | 1274.64 | 1270.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 1269.10 | 1274.64 | 1270.30 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 11:15:00 | 1262.08 | 1267.81 | 1268.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 12:15:00 | 1261.68 | 1266.58 | 1267.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 14:15:00 | 1265.95 | 1254.76 | 1259.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 14:15:00 | 1265.95 | 1254.76 | 1259.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 1265.95 | 1254.76 | 1259.01 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 10:15:00 | 1297.72 | 1268.32 | 1264.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 12:15:00 | 1305.45 | 1280.44 | 1270.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 1255.22 | 1275.40 | 1269.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 1255.22 | 1275.40 | 1269.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 1255.22 | 1275.40 | 1269.53 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 10:15:00 | 1247.13 | 1263.49 | 1265.39 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 1300.58 | 1268.91 | 1266.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 1308.90 | 1292.69 | 1281.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 12:15:00 | 1292.13 | 1292.71 | 1283.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 14:15:00 | 1311.03 | 1298.44 | 1288.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 1311.03 | 1298.44 | 1288.09 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 09:15:00 | 1285.58 | 1299.45 | 1300.84 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 11:15:00 | 1305.68 | 1302.39 | 1302.03 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 12:15:00 | 1294.18 | 1300.75 | 1301.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-01 14:15:00 | 1277.58 | 1295.03 | 1298.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 10:15:00 | 1282.03 | 1267.94 | 1277.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 10:15:00 | 1282.03 | 1267.94 | 1277.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 1282.03 | 1267.94 | 1277.25 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 1286.00 | 1281.39 | 1281.21 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 12:15:00 | 1272.65 | 1280.36 | 1280.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 09:15:00 | 1268.22 | 1276.24 | 1278.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 10:15:00 | 1280.00 | 1276.99 | 1278.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 10:15:00 | 1280.00 | 1276.99 | 1278.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 1280.00 | 1276.99 | 1278.68 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 13:15:00 | 1285.00 | 1279.79 | 1279.63 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 1265.28 | 1278.58 | 1279.38 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 1302.45 | 1279.91 | 1279.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 09:15:00 | 1315.48 | 1296.71 | 1291.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 13:15:00 | 1350.38 | 1356.01 | 1331.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 15:15:00 | 1380.00 | 1381.66 | 1371.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 1380.00 | 1381.66 | 1371.71 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 13:15:00 | 1348.50 | 1363.40 | 1365.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 1320.98 | 1351.42 | 1359.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 12:15:00 | 1328.68 | 1328.35 | 1338.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 14:15:00 | 1329.25 | 1327.01 | 1336.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 14:15:00 | 1329.25 | 1327.01 | 1336.23 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 1338.50 | 1323.31 | 1322.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 11:15:00 | 1345.98 | 1327.85 | 1324.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 11:15:00 | 1339.25 | 1341.85 | 1334.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 12:15:00 | 1324.88 | 1338.45 | 1333.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 1324.88 | 1338.45 | 1333.96 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 14:15:00 | 1342.60 | 1354.70 | 1354.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 1314.73 | 1344.45 | 1350.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 1325.00 | 1321.04 | 1332.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 1325.00 | 1321.04 | 1332.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 1325.00 | 1321.04 | 1332.45 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 14:15:00 | 1199.95 | 1169.17 | 1168.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-21 09:15:00 | 1230.15 | 1187.90 | 1177.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 15:15:00 | 1216.58 | 1216.95 | 1200.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 10:15:00 | 1234.50 | 1241.15 | 1231.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 1234.50 | 1241.15 | 1231.83 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 15:15:00 | 1220.00 | 1227.93 | 1228.03 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 09:15:00 | 1232.93 | 1228.93 | 1228.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 13:15:00 | 1243.95 | 1232.91 | 1230.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 11:15:00 | 1233.20 | 1240.80 | 1236.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 11:15:00 | 1233.20 | 1240.80 | 1236.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 1233.20 | 1240.80 | 1236.14 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 1242.90 | 1247.93 | 1248.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1234.80 | 1244.02 | 1245.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 1229.75 | 1222.45 | 1233.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 15:15:00 | 1214.85 | 1220.93 | 1231.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 1214.85 | 1220.93 | 1231.46 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 1224.55 | 1195.20 | 1192.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 15:15:00 | 1232.50 | 1205.88 | 1199.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 10:15:00 | 1195.90 | 1217.95 | 1212.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 10:15:00 | 1195.90 | 1217.95 | 1212.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1195.90 | 1217.95 | 1212.07 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 12:15:00 | 1175.00 | 1205.71 | 1207.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 13:15:00 | 1163.97 | 1197.37 | 1203.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 1193.55 | 1182.77 | 1191.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 1193.55 | 1182.77 | 1191.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 1193.55 | 1182.77 | 1191.22 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 1202.53 | 1192.76 | 1192.50 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 12:15:00 | 1186.45 | 1191.50 | 1191.95 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 13:15:00 | 1199.22 | 1193.04 | 1192.61 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 14:15:00 | 1172.47 | 1188.93 | 1190.78 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 13:15:00 | 1196.50 | 1190.78 | 1190.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 14:15:00 | 1211.53 | 1194.93 | 1192.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 13:15:00 | 1175.00 | 1195.57 | 1194.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 13:15:00 | 1175.00 | 1195.57 | 1194.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 1175.00 | 1195.57 | 1194.50 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 14:15:00 | 1169.00 | 1190.26 | 1192.18 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 15:15:00 | 1192.00 | 1191.27 | 1191.25 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 12:15:00 | 1184.08 | 1189.92 | 1190.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 13:15:00 | 1181.90 | 1188.32 | 1189.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 15:15:00 | 1192.50 | 1188.08 | 1189.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 15:15:00 | 1192.50 | 1188.08 | 1189.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 15:15:00 | 1192.50 | 1188.08 | 1189.45 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 1204.95 | 1191.45 | 1190.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 10:15:00 | 1212.50 | 1195.66 | 1192.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 13:15:00 | 1197.53 | 1198.55 | 1195.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 13:15:00 | 1197.53 | 1198.55 | 1195.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 13:15:00 | 1197.53 | 1198.55 | 1195.11 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 14:15:00 | 1188.80 | 1194.61 | 1194.79 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 09:15:00 | 1198.55 | 1195.34 | 1195.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 12:15:00 | 1208.90 | 1199.38 | 1197.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 10:15:00 | 1226.60 | 1233.76 | 1223.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 10:15:00 | 1226.60 | 1233.76 | 1223.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 1226.60 | 1233.76 | 1223.55 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 11:15:00 | 1233.65 | 1239.09 | 1239.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 1226.25 | 1236.52 | 1238.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 12:15:00 | 1229.00 | 1220.78 | 1227.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 12:15:00 | 1229.00 | 1220.78 | 1227.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 1229.00 | 1220.78 | 1227.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 12:45:00 | 1225.97 | 1220.78 | 1227.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 1230.03 | 1222.63 | 1227.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 09:15:00 | 1227.50 | 1225.45 | 1228.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 10:15:00 | 1237.63 | 1228.97 | 1229.39 | SL hit (close>static) qty=1.00 sl=1235.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 12:15:00 | 1230.97 | 1228.54 | 1228.53 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 1226.88 | 1228.21 | 1228.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 15:15:00 | 1221.50 | 1226.82 | 1227.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 1216.83 | 1214.36 | 1220.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 12:15:00 | 1216.83 | 1214.36 | 1220.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 1216.83 | 1214.36 | 1220.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 12:45:00 | 1212.53 | 1214.36 | 1220.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 1179.72 | 1176.30 | 1185.08 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 13:15:00 | 1217.47 | 1193.89 | 1191.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 09:15:00 | 1225.00 | 1202.93 | 1196.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 12:15:00 | 1219.35 | 1221.43 | 1213.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 12:45:00 | 1218.83 | 1221.43 | 1213.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 1216.40 | 1220.42 | 1213.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:45:00 | 1212.68 | 1220.42 | 1213.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 1214.90 | 1219.32 | 1213.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 14:30:00 | 1219.70 | 1219.32 | 1213.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 1218.33 | 1219.12 | 1214.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:15:00 | 1242.75 | 1219.12 | 1214.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 1233.75 | 1222.05 | 1216.12 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 13:15:00 | 1226.93 | 1229.21 | 1229.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 14:15:00 | 1213.13 | 1225.99 | 1227.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 09:15:00 | 1236.30 | 1227.74 | 1228.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 1236.30 | 1227.74 | 1228.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 1236.30 | 1227.74 | 1228.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:30:00 | 1233.20 | 1227.74 | 1228.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 1225.00 | 1227.19 | 1228.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 11:30:00 | 1222.58 | 1225.78 | 1227.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 13:15:00 | 1219.63 | 1225.66 | 1227.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 12:15:00 | 1161.45 | 1180.53 | 1195.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 13:15:00 | 1158.65 | 1178.32 | 1192.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-07 14:15:00 | 1194.50 | 1181.56 | 1192.95 | SL hit (close>ema200) qty=0.50 sl=1181.56 alert=retest2 |

### Cycle 79 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 1145.78 | 1131.66 | 1131.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 12:15:00 | 1155.50 | 1136.43 | 1133.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 1149.80 | 1179.77 | 1165.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 1149.80 | 1179.77 | 1165.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1149.80 | 1179.77 | 1165.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:30:00 | 1150.05 | 1179.77 | 1165.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 1143.53 | 1172.53 | 1163.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:45:00 | 1138.33 | 1172.53 | 1163.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 13:15:00 | 1142.68 | 1157.64 | 1157.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 15:15:00 | 1137.50 | 1150.55 | 1154.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 1180.00 | 1156.44 | 1156.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 1180.00 | 1156.44 | 1156.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 1180.00 | 1156.44 | 1156.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:45:00 | 1180.00 | 1156.44 | 1156.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 1184.50 | 1162.05 | 1159.35 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 14:15:00 | 1159.00 | 1165.43 | 1165.48 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 1169.75 | 1165.56 | 1165.42 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 1149.90 | 1162.42 | 1164.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 13:15:00 | 1140.00 | 1155.15 | 1160.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 1103.50 | 1094.35 | 1104.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 1103.50 | 1094.35 | 1104.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1103.50 | 1094.35 | 1104.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 1103.50 | 1094.35 | 1104.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1108.00 | 1097.08 | 1104.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:30:00 | 1110.00 | 1097.08 | 1104.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 1107.60 | 1099.18 | 1105.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 13:45:00 | 1104.43 | 1101.46 | 1105.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 1103.70 | 1103.52 | 1105.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 10:15:00 | 1113.50 | 1105.26 | 1105.96 | SL hit (close>static) qty=1.00 sl=1108.50 alert=retest2 |

### Cycle 85 — BUY (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 11:15:00 | 1115.93 | 1107.40 | 1106.87 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 1096.65 | 1106.08 | 1106.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 12:15:00 | 1094.97 | 1100.12 | 1103.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 09:15:00 | 1097.60 | 1097.52 | 1100.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 1086.50 | 1095.32 | 1099.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1086.50 | 1095.32 | 1099.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:30:00 | 1092.08 | 1095.32 | 1099.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1108.38 | 1094.87 | 1098.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:00:00 | 1108.38 | 1094.87 | 1098.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 1083.10 | 1092.51 | 1097.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:45:00 | 1080.00 | 1090.51 | 1095.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 09:30:00 | 1069.63 | 1088.04 | 1093.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 12:00:00 | 1079.97 | 1085.97 | 1091.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 10:15:00 | 1107.20 | 1085.10 | 1087.29 | SL hit (close>static) qty=1.00 sl=1105.72 alert=retest2 |

### Cycle 87 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 1103.40 | 1090.86 | 1089.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 1108.83 | 1095.92 | 1092.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 1199.53 | 1200.59 | 1181.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 1199.53 | 1200.59 | 1181.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 1200.00 | 1200.61 | 1195.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:15:00 | 1215.00 | 1197.51 | 1196.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 12:15:00 | 1192.13 | 1197.03 | 1196.29 | SL hit (close<static) qty=1.00 sl=1195.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 13:15:00 | 1188.45 | 1195.31 | 1195.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 15:15:00 | 1184.50 | 1191.60 | 1193.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 12:15:00 | 1191.97 | 1187.22 | 1190.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 12:15:00 | 1191.97 | 1187.22 | 1190.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 1191.97 | 1187.22 | 1190.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 1191.97 | 1187.22 | 1190.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 1187.83 | 1187.34 | 1190.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 11:00:00 | 1185.78 | 1188.59 | 1190.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1228.00 | 1194.84 | 1191.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1228.00 | 1194.84 | 1191.77 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 13:15:00 | 1176.08 | 1194.79 | 1197.01 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 11:15:00 | 1207.50 | 1194.54 | 1193.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 1220.97 | 1210.92 | 1207.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 1250.78 | 1254.60 | 1244.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 1250.78 | 1254.60 | 1244.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1250.78 | 1254.60 | 1244.52 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 10:15:00 | 1243.47 | 1248.22 | 1248.45 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 1254.95 | 1249.37 | 1248.87 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 1231.18 | 1245.73 | 1247.39 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 1260.38 | 1249.41 | 1248.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 1274.00 | 1256.18 | 1251.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 1257.53 | 1257.57 | 1253.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 1257.53 | 1257.57 | 1253.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1257.53 | 1257.57 | 1253.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 12:00:00 | 1307.40 | 1267.49 | 1259.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 1254.45 | 1262.21 | 1263.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 1254.45 | 1262.21 | 1263.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 13:15:00 | 1246.05 | 1256.11 | 1259.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 1248.55 | 1243.26 | 1250.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 1248.55 | 1243.26 | 1250.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1248.55 | 1243.26 | 1250.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 1248.55 | 1243.26 | 1250.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1238.95 | 1242.40 | 1249.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 15:15:00 | 1232.50 | 1242.40 | 1249.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:00:00 | 1229.30 | 1238.19 | 1245.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 13:15:00 | 1233.95 | 1235.21 | 1242.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 13:45:00 | 1235.10 | 1235.21 | 1241.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1200.83 | 1201.20 | 1211.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 1203.10 | 1201.20 | 1211.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1217.53 | 1204.46 | 1212.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 1217.53 | 1204.46 | 1212.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 1201.85 | 1203.94 | 1211.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 1195.53 | 1209.26 | 1211.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:45:00 | 1197.72 | 1196.80 | 1200.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 14:15:00 | 1196.88 | 1196.80 | 1200.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 10:30:00 | 1197.50 | 1197.02 | 1199.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 1201.58 | 1197.93 | 1199.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:30:00 | 1201.55 | 1197.93 | 1199.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 1192.75 | 1196.89 | 1198.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 13:15:00 | 1191.90 | 1196.89 | 1198.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 14:00:00 | 1192.47 | 1196.01 | 1198.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:15:00 | 1170.88 | 1188.14 | 1193.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:15:00 | 1172.25 | 1188.14 | 1193.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:15:00 | 1173.34 | 1188.14 | 1193.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-30 11:15:00 | 1195.30 | 1188.89 | 1193.07 | SL hit (close>ema200) qty=0.50 sl=1188.89 alert=retest2 |

### Cycle 97 — BUY (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 11:15:00 | 1200.47 | 1195.05 | 1194.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 10:15:00 | 1209.38 | 1200.23 | 1197.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 1213.03 | 1215.26 | 1207.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 1213.03 | 1215.26 | 1207.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1213.03 | 1215.26 | 1207.84 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1175.13 | 1204.15 | 1206.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 1167.50 | 1196.82 | 1202.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 13:15:00 | 1201.88 | 1195.61 | 1201.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 13:15:00 | 1201.88 | 1195.61 | 1201.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 13:15:00 | 1201.88 | 1195.61 | 1201.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 13:45:00 | 1206.70 | 1195.61 | 1201.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 1210.22 | 1198.53 | 1202.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 14:30:00 | 1209.20 | 1198.53 | 1202.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 1208.90 | 1200.61 | 1202.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 1227.90 | 1200.61 | 1202.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 1228.45 | 1206.18 | 1204.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 13:15:00 | 1239.05 | 1221.42 | 1213.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 1258.10 | 1266.77 | 1254.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 1258.10 | 1266.77 | 1254.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 1245.10 | 1262.43 | 1253.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:45:00 | 1246.00 | 1262.43 | 1253.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 1245.00 | 1258.95 | 1252.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:45:00 | 1230.47 | 1258.95 | 1252.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1260.03 | 1269.00 | 1263.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:30:00 | 1259.50 | 1269.00 | 1263.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1243.03 | 1263.81 | 1261.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1243.03 | 1263.81 | 1261.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 1242.47 | 1259.54 | 1259.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1204.00 | 1245.71 | 1253.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1211.88 | 1211.15 | 1227.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 11:15:00 | 1226.00 | 1215.49 | 1226.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 1226.00 | 1215.49 | 1226.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:00:00 | 1226.00 | 1215.49 | 1226.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 1242.03 | 1220.80 | 1228.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:00:00 | 1242.03 | 1220.80 | 1228.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 1241.55 | 1224.95 | 1229.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:30:00 | 1239.18 | 1224.95 | 1229.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 1253.50 | 1234.18 | 1233.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1297.10 | 1246.76 | 1238.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 1325.65 | 1328.77 | 1298.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 1325.65 | 1328.77 | 1298.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1352.18 | 1359.39 | 1346.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:45:00 | 1346.73 | 1359.39 | 1346.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1340.78 | 1355.67 | 1345.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 1340.78 | 1355.67 | 1345.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1346.98 | 1353.93 | 1345.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1347.50 | 1353.93 | 1345.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1348.28 | 1352.80 | 1346.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 1337.00 | 1352.80 | 1346.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1353.25 | 1352.89 | 1346.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 1351.50 | 1352.89 | 1346.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1354.93 | 1355.97 | 1350.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 1366.18 | 1355.97 | 1350.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 1413.65 | 1440.48 | 1443.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 1413.65 | 1440.48 | 1443.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 10:15:00 | 1403.33 | 1426.76 | 1435.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 13:15:00 | 1389.38 | 1386.82 | 1402.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 14:00:00 | 1389.38 | 1386.82 | 1402.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1391.65 | 1387.04 | 1398.49 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 1422.50 | 1400.71 | 1398.16 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 1384.53 | 1397.66 | 1397.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 1372.20 | 1392.57 | 1395.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 14:15:00 | 1350.00 | 1346.80 | 1359.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 14:15:00 | 1350.00 | 1346.80 | 1359.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1350.00 | 1346.80 | 1359.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 1350.00 | 1346.80 | 1359.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1347.30 | 1347.42 | 1357.84 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 14:15:00 | 1362.38 | 1355.39 | 1355.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 15:15:00 | 1364.50 | 1357.21 | 1355.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 10:15:00 | 1364.00 | 1365.27 | 1360.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 11:00:00 | 1364.00 | 1365.27 | 1360.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 1361.20 | 1364.45 | 1360.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 10:30:00 | 1375.95 | 1373.92 | 1366.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-25 11:15:00 | 1513.55 | 1403.44 | 1380.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 1489.55 | 1501.19 | 1501.54 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 1507.70 | 1502.49 | 1502.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 12:15:00 | 1515.85 | 1506.24 | 1503.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 15:15:00 | 1500.00 | 1507.87 | 1505.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 15:15:00 | 1500.00 | 1507.87 | 1505.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1500.00 | 1507.87 | 1505.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1528.60 | 1507.87 | 1505.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 11:15:00 | 1544.23 | 1508.50 | 1506.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 1532.45 | 1528.76 | 1520.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 14:15:00 | 1538.05 | 1550.07 | 1550.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 1538.05 | 1550.07 | 1550.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 15:15:00 | 1532.18 | 1546.49 | 1548.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 12:15:00 | 1551.50 | 1546.73 | 1548.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 12:15:00 | 1551.50 | 1546.73 | 1548.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 1551.50 | 1546.73 | 1548.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:45:00 | 1552.40 | 1546.73 | 1548.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 1543.28 | 1546.04 | 1547.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 1524.40 | 1545.40 | 1547.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:00:00 | 1540.40 | 1529.03 | 1531.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 14:15:00 | 1463.38 | 1483.48 | 1495.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 1448.18 | 1476.14 | 1489.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-24 11:15:00 | 1386.36 | 1413.16 | 1433.80 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 109 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 1413.88 | 1390.68 | 1387.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 1432.28 | 1399.00 | 1391.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 1469.10 | 1471.79 | 1444.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:00:00 | 1469.10 | 1471.79 | 1444.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 1453.90 | 1466.79 | 1453.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1482.13 | 1469.86 | 1455.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-06 09:15:00 | 1630.34 | 1525.88 | 1500.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1637.23 | 1664.35 | 1666.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 10:15:00 | 1609.03 | 1653.29 | 1660.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 1646.35 | 1613.88 | 1633.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 1646.35 | 1613.88 | 1633.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1646.35 | 1613.88 | 1633.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1646.35 | 1613.88 | 1633.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1648.15 | 1620.73 | 1634.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 1632.68 | 1620.73 | 1634.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 1635.00 | 1623.58 | 1634.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:15:00 | 1632.53 | 1623.58 | 1634.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 11:15:00 | 1665.00 | 1633.72 | 1633.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 11:15:00 | 1665.00 | 1633.72 | 1633.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 1683.45 | 1654.82 | 1644.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 11:15:00 | 1658.20 | 1659.08 | 1648.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 11:45:00 | 1662.43 | 1659.08 | 1648.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1662.98 | 1659.86 | 1650.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:30:00 | 1651.00 | 1659.86 | 1650.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1625.63 | 1653.02 | 1647.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 1625.63 | 1653.02 | 1647.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1634.85 | 1649.38 | 1646.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 1634.85 | 1649.38 | 1646.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 1660.03 | 1648.06 | 1646.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 1694.98 | 1650.43 | 1647.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-06 09:15:00 | 1864.48 | 1846.63 | 1820.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 1876.08 | 1903.42 | 1904.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 10:15:00 | 1856.05 | 1883.82 | 1893.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 14:15:00 | 1872.10 | 1871.45 | 1883.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 15:00:00 | 1872.10 | 1871.45 | 1883.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1861.95 | 1867.96 | 1879.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:30:00 | 1848.30 | 1861.45 | 1869.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:15:00 | 1755.88 | 1787.47 | 1807.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 1786.48 | 1780.06 | 1793.83 | SL hit (close>ema200) qty=0.50 sl=1780.06 alert=retest2 |

### Cycle 113 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 1825.03 | 1802.90 | 1800.11 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 1770.00 | 1794.31 | 1796.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 1739.53 | 1779.96 | 1789.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 15:15:00 | 1747.50 | 1746.04 | 1761.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 09:15:00 | 1748.80 | 1746.04 | 1761.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1705.50 | 1737.93 | 1756.17 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 1767.50 | 1747.38 | 1745.81 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1718.58 | 1743.09 | 1745.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 1707.30 | 1729.19 | 1738.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 14:15:00 | 1702.45 | 1694.84 | 1710.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 15:00:00 | 1702.45 | 1694.84 | 1710.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 1673.58 | 1651.89 | 1664.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:00:00 | 1673.58 | 1651.89 | 1664.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 1645.48 | 1650.61 | 1662.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:15:00 | 1629.80 | 1650.61 | 1662.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 11:00:00 | 1637.08 | 1639.42 | 1651.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:15:00 | 1639.40 | 1620.77 | 1634.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 09:45:00 | 1636.58 | 1604.01 | 1608.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 1629.33 | 1609.07 | 1610.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 11:15:00 | 1638.83 | 1609.07 | 1610.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 1626.45 | 1612.55 | 1611.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1626.45 | 1612.55 | 1611.77 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 1610.20 | 1615.57 | 1615.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 1601.00 | 1612.65 | 1614.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 1629.00 | 1611.64 | 1612.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 12:15:00 | 1629.00 | 1611.64 | 1612.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 1629.00 | 1611.64 | 1612.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 1629.00 | 1611.64 | 1612.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 1627.48 | 1614.81 | 1613.83 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 1606.50 | 1613.70 | 1614.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1577.23 | 1606.40 | 1610.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1575.25 | 1573.40 | 1590.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1575.25 | 1573.40 | 1590.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1575.25 | 1573.40 | 1590.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1575.25 | 1573.40 | 1590.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1580.65 | 1575.09 | 1588.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:30:00 | 1587.88 | 1575.09 | 1588.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1614.95 | 1583.06 | 1590.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 1614.95 | 1583.06 | 1590.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1601.53 | 1586.76 | 1591.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 1616.05 | 1586.76 | 1591.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 1589.33 | 1590.86 | 1592.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:30:00 | 1583.78 | 1590.86 | 1592.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 1590.48 | 1590.79 | 1592.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 1605.48 | 1590.79 | 1592.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1581.88 | 1589.01 | 1591.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 11:00:00 | 1567.13 | 1584.63 | 1589.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1488.77 | 1542.97 | 1564.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 1496.98 | 1486.61 | 1512.30 | SL hit (close>ema200) qty=0.50 sl=1486.61 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1560.68 | 1516.83 | 1514.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 1565.88 | 1526.64 | 1519.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 09:15:00 | 1516.53 | 1531.09 | 1523.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 09:15:00 | 1516.53 | 1531.09 | 1523.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1516.53 | 1531.09 | 1523.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 10:15:00 | 1529.03 | 1531.09 | 1523.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 13:15:00 | 1506.25 | 1517.38 | 1518.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1506.25 | 1517.38 | 1518.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 14:15:00 | 1491.20 | 1512.15 | 1515.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 13:15:00 | 1525.38 | 1500.94 | 1506.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 13:15:00 | 1525.38 | 1500.94 | 1506.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 1525.38 | 1500.94 | 1506.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:45:00 | 1524.75 | 1500.94 | 1506.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1535.78 | 1507.91 | 1509.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 15:00:00 | 1535.78 | 1507.91 | 1509.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 1557.98 | 1519.23 | 1514.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 10:15:00 | 1589.83 | 1533.35 | 1520.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 15:15:00 | 1587.80 | 1591.61 | 1571.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:15:00 | 1618.10 | 1591.61 | 1571.43 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 12:00:00 | 1611.75 | 1601.22 | 1581.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1591.43 | 1596.78 | 1586.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 1591.90 | 1596.78 | 1586.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1605.53 | 1620.98 | 1610.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-06 12:15:00 | 1605.53 | 1620.98 | 1610.84 | SL hit (close<ema400) qty=1.00 sl=1610.84 alert=retest1 |

### Cycle 124 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1590.83 | 1606.21 | 1607.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1561.93 | 1593.82 | 1601.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1529.25 | 1526.20 | 1549.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 1529.25 | 1526.20 | 1549.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1530.63 | 1530.92 | 1546.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 1519.38 | 1538.67 | 1542.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 14:15:00 | 1551.48 | 1516.83 | 1526.38 | SL hit (close>static) qty=1.00 sl=1546.83 alert=retest2 |

### Cycle 125 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 1538.08 | 1512.95 | 1512.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 10:15:00 | 1547.50 | 1519.86 | 1515.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 1553.15 | 1553.50 | 1540.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 13:00:00 | 1553.15 | 1553.50 | 1540.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1589.38 | 1561.90 | 1548.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 1610.10 | 1576.39 | 1557.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 11:15:00 | 1537.50 | 1554.98 | 1555.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 1537.50 | 1554.98 | 1555.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 1526.93 | 1541.33 | 1547.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 13:15:00 | 1331.73 | 1331.18 | 1368.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 14:00:00 | 1331.73 | 1331.18 | 1368.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1402.38 | 1345.52 | 1365.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 1403.05 | 1345.52 | 1365.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1393.48 | 1355.11 | 1368.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 11:15:00 | 1359.00 | 1355.11 | 1368.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 14:15:00 | 1396.58 | 1368.62 | 1365.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 14:15:00 | 1396.58 | 1368.62 | 1365.09 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 1353.05 | 1376.75 | 1377.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 1346.05 | 1366.90 | 1372.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 1323.85 | 1311.95 | 1331.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 1323.85 | 1311.95 | 1331.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1323.85 | 1311.95 | 1331.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:30:00 | 1337.50 | 1311.95 | 1331.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1284.40 | 1282.20 | 1297.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 1289.85 | 1282.20 | 1297.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1285.13 | 1271.65 | 1284.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1285.13 | 1271.65 | 1284.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1279.75 | 1273.27 | 1284.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 11:30:00 | 1277.40 | 1273.73 | 1283.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 13:00:00 | 1275.83 | 1274.15 | 1282.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 1305.00 | 1284.15 | 1285.02 | SL hit (close>static) qty=1.00 sl=1292.75 alert=retest2 |

### Cycle 129 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 10:15:00 | 1302.03 | 1287.73 | 1286.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 11:15:00 | 1322.98 | 1301.05 | 1294.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 1368.00 | 1370.53 | 1353.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 12:00:00 | 1368.00 | 1370.53 | 1353.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 1371.45 | 1377.18 | 1368.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 13:15:00 | 1377.38 | 1368.73 | 1367.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 15:00:00 | 1382.93 | 1372.18 | 1369.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 10:00:00 | 1379.03 | 1376.80 | 1372.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 10:30:00 | 1379.93 | 1376.09 | 1372.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 1380.55 | 1376.99 | 1372.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 14:15:00 | 1384.18 | 1377.07 | 1373.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 10:15:00 | 1361.95 | 1377.51 | 1375.47 | SL hit (close<static) qty=1.00 sl=1372.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 1367.18 | 1373.99 | 1374.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 15:15:00 | 1363.90 | 1371.97 | 1373.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 1373.50 | 1369.86 | 1371.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 12:15:00 | 1373.50 | 1369.86 | 1371.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1373.50 | 1369.86 | 1371.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 1373.50 | 1369.86 | 1371.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1373.58 | 1370.60 | 1371.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 1373.00 | 1370.60 | 1371.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1367.50 | 1370.60 | 1371.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 1394.20 | 1370.60 | 1371.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1367.50 | 1369.98 | 1371.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:30:00 | 1382.13 | 1369.98 | 1371.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1368.05 | 1369.60 | 1370.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:45:00 | 1373.50 | 1369.60 | 1370.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 1362.55 | 1368.19 | 1370.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:30:00 | 1369.35 | 1368.19 | 1370.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 1393.43 | 1371.72 | 1371.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 15:15:00 | 1395.03 | 1376.38 | 1373.25 | Break + close above crossover candle high |

### Cycle 132 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1343.70 | 1369.84 | 1370.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 1338.55 | 1359.47 | 1365.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 12:15:00 | 1254.38 | 1250.31 | 1277.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:30:00 | 1256.45 | 1250.31 | 1277.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1213.80 | 1215.05 | 1237.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:00:00 | 1206.30 | 1213.30 | 1234.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 1267.55 | 1220.72 | 1225.71 | SL hit (close>static) qty=1.00 sl=1261.28 alert=retest2 |

### Cycle 133 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 1251.05 | 1232.40 | 1230.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 10:15:00 | 1263.50 | 1248.51 | 1240.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1244.75 | 1254.13 | 1247.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1244.75 | 1254.13 | 1247.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1244.75 | 1254.13 | 1247.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 1259.40 | 1251.02 | 1248.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 1289.00 | 1301.85 | 1302.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1289.00 | 1301.85 | 1302.41 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 1298.50 | 1289.54 | 1288.66 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 1277.75 | 1287.18 | 1287.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 1262.50 | 1280.19 | 1284.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1262.00 | 1254.64 | 1262.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1262.00 | 1254.64 | 1262.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1262.00 | 1254.64 | 1262.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:45:00 | 1245.35 | 1250.59 | 1258.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1183.08 | 1217.31 | 1233.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 14:15:00 | 1208.10 | 1206.33 | 1220.98 | SL hit (close>ema200) qty=0.50 sl=1206.33 alert=retest2 |

### Cycle 137 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 1235.65 | 1226.17 | 1226.13 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 1222.45 | 1225.42 | 1225.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1196.30 | 1219.23 | 1222.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 1230.50 | 1215.43 | 1219.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 13:15:00 | 1230.50 | 1215.43 | 1219.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1230.50 | 1215.43 | 1219.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 1230.50 | 1215.43 | 1219.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 1220.20 | 1216.38 | 1219.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 1220.20 | 1216.38 | 1219.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 1219.50 | 1217.01 | 1219.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 1256.55 | 1217.01 | 1219.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1268.60 | 1227.32 | 1223.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 1279.40 | 1237.74 | 1228.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 15:15:00 | 1701.00 | 1702.23 | 1669.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 1694.50 | 1702.23 | 1669.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1681.10 | 1689.46 | 1679.23 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 13:15:00 | 1668.25 | 1679.71 | 1679.83 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 1690.90 | 1681.20 | 1680.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1736.40 | 1702.92 | 1692.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 1695.00 | 1714.82 | 1702.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 1695.00 | 1714.82 | 1702.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1695.00 | 1714.82 | 1702.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 1703.55 | 1714.82 | 1702.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1675.60 | 1706.98 | 1699.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 1675.60 | 1706.98 | 1699.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1685.05 | 1702.59 | 1698.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:45:00 | 1674.15 | 1702.59 | 1698.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1782.00 | 1765.22 | 1746.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:00:00 | 1787.00 | 1774.76 | 1767.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1794.90 | 1778.58 | 1770.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 1795.55 | 1825.63 | 1825.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1795.55 | 1825.63 | 1825.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1732.70 | 1807.05 | 1817.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 14:15:00 | 1838.60 | 1813.36 | 1819.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 14:15:00 | 1838.60 | 1813.36 | 1819.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1838.60 | 1813.36 | 1819.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 1838.60 | 1813.36 | 1819.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1826.50 | 1815.99 | 1820.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 1788.20 | 1815.99 | 1820.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 1816.85 | 1800.11 | 1800.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1816.85 | 1800.11 | 1800.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 12:15:00 | 1818.10 | 1806.00 | 1802.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 1801.50 | 1807.22 | 1804.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 15:15:00 | 1801.50 | 1807.22 | 1804.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 1801.50 | 1807.22 | 1804.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 1800.00 | 1807.22 | 1804.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1799.00 | 1805.57 | 1803.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 1799.90 | 1805.57 | 1803.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1795.05 | 1803.47 | 1803.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 1799.30 | 1803.47 | 1803.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1793.85 | 1801.55 | 1802.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 1759.50 | 1793.14 | 1798.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 1789.55 | 1788.72 | 1795.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:00:00 | 1789.55 | 1788.72 | 1795.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1824.50 | 1795.87 | 1797.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 1782.45 | 1795.87 | 1797.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1693.33 | 1727.07 | 1749.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1731.20 | 1725.57 | 1744.90 | SL hit (close>ema200) qty=0.50 sl=1725.57 alert=retest2 |

### Cycle 145 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1768.15 | 1750.59 | 1748.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1787.90 | 1762.02 | 1755.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 1765.00 | 1769.71 | 1761.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:00:00 | 1765.00 | 1769.71 | 1761.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1748.80 | 1765.52 | 1760.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:15:00 | 1749.00 | 1765.52 | 1760.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1752.05 | 1762.83 | 1759.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 1766.10 | 1761.26 | 1759.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 1746.60 | 1756.14 | 1757.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 1746.60 | 1756.14 | 1757.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 11:15:00 | 1732.60 | 1750.23 | 1753.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 14:15:00 | 1754.15 | 1746.03 | 1750.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1754.15 | 1746.03 | 1750.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1754.15 | 1746.03 | 1750.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:30:00 | 1758.90 | 1746.03 | 1750.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1761.55 | 1749.13 | 1751.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1744.75 | 1749.13 | 1751.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1745.40 | 1748.39 | 1750.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:30:00 | 1736.80 | 1746.25 | 1749.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 1739.90 | 1739.63 | 1743.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 1732.80 | 1737.75 | 1741.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:00:00 | 1739.35 | 1728.80 | 1730.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1715.00 | 1721.35 | 1726.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 1699.00 | 1715.73 | 1721.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:00:00 | 1701.45 | 1700.03 | 1709.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 1701.25 | 1701.84 | 1709.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 1700.65 | 1699.62 | 1706.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1678.45 | 1687.61 | 1696.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 1666.95 | 1687.01 | 1695.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 1677.65 | 1684.48 | 1690.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1674.85 | 1684.09 | 1690.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:45:00 | 1672.90 | 1683.38 | 1689.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1694.30 | 1685.56 | 1689.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1702.45 | 1692.89 | 1692.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 1702.45 | 1692.89 | 1692.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 1726.05 | 1701.46 | 1696.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 1777.35 | 1779.80 | 1761.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 15:00:00 | 1777.35 | 1779.80 | 1761.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1849.65 | 1849.93 | 1830.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1824.30 | 1849.93 | 1830.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1860.35 | 1864.86 | 1855.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1860.35 | 1864.86 | 1855.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1843.85 | 1860.66 | 1854.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 1843.85 | 1860.66 | 1854.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1848.55 | 1858.24 | 1853.82 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 1840.70 | 1850.65 | 1851.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1826.80 | 1844.18 | 1847.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 09:15:00 | 1894.50 | 1843.02 | 1843.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 1894.50 | 1843.02 | 1843.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1894.50 | 1843.02 | 1843.04 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 10:15:00 | 1850.50 | 1844.51 | 1843.72 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 1827.00 | 1842.49 | 1844.09 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 1893.00 | 1852.59 | 1848.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 1923.80 | 1889.92 | 1876.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 15:15:00 | 1885.00 | 1893.65 | 1883.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 09:15:00 | 1881.65 | 1893.65 | 1883.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1898.00 | 1894.52 | 1884.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1914.75 | 1886.87 | 1884.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1929.00 | 1905.18 | 1898.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 1991.95 | 2034.28 | 2038.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 1991.95 | 2034.28 | 2038.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 12:15:00 | 1968.30 | 1998.18 | 2008.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1996.50 | 1979.56 | 1994.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1996.50 | 1979.56 | 1994.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1996.50 | 1979.56 | 1994.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1996.50 | 1979.56 | 1994.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 2001.00 | 1983.85 | 1995.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 2003.60 | 1983.85 | 1995.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 2007.50 | 1988.58 | 1996.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 2007.50 | 1988.58 | 1996.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 2025.00 | 2004.21 | 2001.85 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 1993.20 | 2000.11 | 2000.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1987.00 | 1995.79 | 1998.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 14:15:00 | 1998.20 | 1994.11 | 1997.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 14:15:00 | 1998.20 | 1994.11 | 1997.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1998.20 | 1994.11 | 1997.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1998.20 | 1994.11 | 1997.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 2002.00 | 1995.69 | 1997.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1983.90 | 1995.69 | 1997.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1982.50 | 1993.05 | 1996.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 1971.40 | 1985.91 | 1992.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 2069.00 | 1986.62 | 1987.73 | SL hit (close>static) qty=1.00 sl=2016.45 alert=retest2 |

### Cycle 155 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 2086.40 | 2006.58 | 1996.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 11:15:00 | 2101.50 | 2025.56 | 2006.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 2139.85 | 2155.01 | 2109.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 14:15:00 | 2119.35 | 2139.78 | 2118.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 2119.35 | 2139.78 | 2118.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:45:00 | 2111.15 | 2139.78 | 2118.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 2119.00 | 2135.62 | 2118.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 2094.45 | 2135.62 | 2118.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 2073.00 | 2123.10 | 2114.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:00:00 | 2073.00 | 2123.10 | 2114.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 2102.50 | 2121.43 | 2116.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:00:00 | 2102.50 | 2121.43 | 2116.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 2090.55 | 2115.25 | 2113.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 2083.45 | 2115.25 | 2113.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 2101.50 | 2112.49 | 2112.91 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 2162.50 | 2122.49 | 2117.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 2204.40 | 2138.88 | 2125.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 2244.95 | 2255.53 | 2218.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 2230.75 | 2255.53 | 2218.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 2238.10 | 2252.04 | 2220.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:30:00 | 2248.00 | 2244.49 | 2219.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 2194.35 | 2234.46 | 2217.58 | SL hit (close<static) qty=1.00 sl=2220.15 alert=retest2 |

### Cycle 158 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 2184.95 | 2208.09 | 2209.22 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 2220.85 | 2211.33 | 2210.08 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 2191.70 | 2208.46 | 2209.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 2185.95 | 2201.00 | 2205.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 2180.90 | 2142.28 | 2159.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 2180.90 | 2142.28 | 2159.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2180.90 | 2142.28 | 2159.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 2185.00 | 2142.28 | 2159.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 2187.50 | 2151.33 | 2162.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 2187.50 | 2151.33 | 2162.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 2177.50 | 2156.56 | 2163.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:30:00 | 2175.00 | 2165.00 | 2166.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 2180.00 | 2169.94 | 2168.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 2180.00 | 2169.94 | 2168.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2222.00 | 2180.36 | 2173.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 2254.05 | 2255.15 | 2231.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 2254.05 | 2255.15 | 2231.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2239.30 | 2249.88 | 2233.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 2239.30 | 2249.88 | 2233.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2226.20 | 2245.14 | 2232.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 2223.50 | 2245.14 | 2232.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 2216.50 | 2239.41 | 2231.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:30:00 | 2221.90 | 2239.41 | 2231.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 2207.55 | 2225.94 | 2226.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 2188.80 | 2216.27 | 2222.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 2214.30 | 2197.59 | 2207.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 2214.30 | 2197.59 | 2207.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 2214.30 | 2197.59 | 2207.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:00:00 | 2183.25 | 2194.20 | 2204.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:30:00 | 2184.00 | 2191.73 | 2202.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:30:00 | 2179.45 | 2189.15 | 2197.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:00:00 | 2182.95 | 2188.35 | 2195.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 2195.85 | 2188.73 | 2193.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 2195.85 | 2188.73 | 2193.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 2200.00 | 2190.99 | 2194.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 2203.30 | 2190.99 | 2194.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 2224.40 | 2197.67 | 2197.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 2224.40 | 2197.67 | 2197.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 2254.20 | 2208.98 | 2202.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 2215.00 | 2215.14 | 2206.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 2215.00 | 2215.14 | 2206.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2195.95 | 2211.30 | 2205.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 2194.80 | 2211.30 | 2205.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 2200.10 | 2209.06 | 2205.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 2191.15 | 2209.06 | 2205.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 2205.60 | 2208.37 | 2205.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 2235.95 | 2212.41 | 2207.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 2179.15 | 2205.76 | 2204.78 | SL hit (close<static) qty=1.00 sl=2196.55 alert=retest2 |

### Cycle 164 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 2152.00 | 2195.01 | 2199.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 2132.60 | 2182.53 | 2193.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 2164.55 | 2156.23 | 2171.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 13:00:00 | 2164.55 | 2156.23 | 2171.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 2149.60 | 2147.76 | 2161.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 2147.55 | 2147.76 | 2161.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2111.20 | 2073.50 | 2100.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 2112.40 | 2073.50 | 2100.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 2139.40 | 2086.68 | 2103.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 2139.40 | 2086.68 | 2103.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 2136.70 | 2096.68 | 2106.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:45:00 | 2145.15 | 2096.68 | 2106.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 2108.70 | 2105.04 | 2108.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 2141.25 | 2105.04 | 2108.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 2105.50 | 2105.13 | 2108.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 2064.05 | 2105.13 | 2108.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:00:00 | 2092.75 | 2105.76 | 2107.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 15:00:00 | 2069.10 | 2096.50 | 2103.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:15:00 | 1988.11 | 2008.51 | 2032.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 2003.85 | 1995.35 | 2013.44 | SL hit (close>ema200) qty=0.50 sl=1995.35 alert=retest2 |

### Cycle 165 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 2040.60 | 2007.92 | 2003.54 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 1970.45 | 2006.42 | 2008.10 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 2027.90 | 2008.24 | 2006.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 2038.50 | 2014.29 | 2009.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 2054.70 | 2055.66 | 2040.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 12:15:00 | 2030.60 | 2050.44 | 2041.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 2030.60 | 2050.44 | 2041.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 2030.60 | 2050.44 | 2041.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 2020.95 | 2044.54 | 2039.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 2023.00 | 2044.54 | 2039.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 2015.25 | 2033.68 | 2034.88 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 2048.80 | 2036.70 | 2036.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 2074.50 | 2044.26 | 2039.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 2039.70 | 2047.27 | 2042.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 2039.70 | 2047.27 | 2042.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 2039.70 | 2047.27 | 2042.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 2037.35 | 2047.27 | 2042.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2044.00 | 2046.62 | 2042.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 2057.20 | 2045.29 | 2042.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 2048.05 | 2047.30 | 2043.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:00:00 | 2052.00 | 2048.24 | 2044.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 2037.00 | 2051.54 | 2049.97 | SL hit (close<static) qty=1.00 sl=2039.70 alert=retest2 |

### Cycle 170 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 2033.10 | 2047.85 | 2048.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 2032.60 | 2044.80 | 2047.00 | Break + close below crossover candle low |

### Cycle 171 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 2140.00 | 2063.84 | 2055.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 2174.45 | 2117.97 | 2088.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 2170.00 | 2171.91 | 2142.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 2170.00 | 2171.91 | 2142.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 2164.50 | 2171.76 | 2155.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 2156.05 | 2171.76 | 2155.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 2287.50 | 2340.64 | 2308.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 2266.65 | 2340.64 | 2308.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 2286.65 | 2329.84 | 2306.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 2283.35 | 2329.84 | 2306.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 2364.75 | 2366.87 | 2354.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:45:00 | 2353.35 | 2366.87 | 2354.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2364.95 | 2365.99 | 2357.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 2404.45 | 2374.64 | 2361.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 2399.95 | 2378.70 | 2364.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 2336.60 | 2365.08 | 2363.24 | SL hit (close<static) qty=1.00 sl=2352.50 alert=retest2 |

### Cycle 172 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 2329.30 | 2357.92 | 2360.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 2315.50 | 2344.50 | 2353.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 2191.50 | 2167.61 | 2213.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 2191.50 | 2167.61 | 2213.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2199.90 | 2173.25 | 2208.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 2171.30 | 2172.86 | 2204.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 2173.75 | 2171.47 | 2198.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 2240.00 | 2195.94 | 2198.93 | SL hit (close>static) qty=1.00 sl=2224.80 alert=retest2 |

### Cycle 173 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 2257.05 | 2208.17 | 2204.21 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 2203.65 | 2218.05 | 2218.15 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 2245.10 | 2219.77 | 2217.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 12:15:00 | 2265.50 | 2232.54 | 2223.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 2244.60 | 2253.06 | 2238.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 2244.60 | 2253.06 | 2238.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2244.60 | 2253.06 | 2238.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 2244.60 | 2253.06 | 2238.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 2237.95 | 2247.36 | 2239.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 2232.40 | 2247.36 | 2239.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 2233.95 | 2244.67 | 2238.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 2229.25 | 2244.67 | 2238.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2225.75 | 2237.63 | 2236.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 2250.75 | 2237.63 | 2236.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:45:00 | 2249.25 | 2242.22 | 2238.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 2221.30 | 2238.04 | 2237.10 | SL hit (close<static) qty=1.00 sl=2222.50 alert=retest2 |

### Cycle 176 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 2211.65 | 2232.76 | 2234.79 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 2248.45 | 2233.38 | 2232.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 14:15:00 | 2304.60 | 2247.63 | 2238.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 10:15:00 | 2243.55 | 2255.67 | 2245.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 2243.55 | 2255.67 | 2245.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 2243.55 | 2255.67 | 2245.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 2247.25 | 2255.67 | 2245.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 2247.50 | 2254.04 | 2245.84 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 2227.50 | 2241.58 | 2242.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 2219.35 | 2235.87 | 2238.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 14:15:00 | 2214.15 | 2212.04 | 2219.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 15:00:00 | 2214.15 | 2212.04 | 2219.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 2229.05 | 2215.76 | 2219.91 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 2241.25 | 2226.09 | 2224.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 2247.30 | 2230.33 | 2226.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 2231.25 | 2233.42 | 2228.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:15:00 | 2248.30 | 2233.42 | 2228.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2285.95 | 2243.93 | 2233.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:30:00 | 2296.40 | 2251.88 | 2238.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 2322.75 | 2266.83 | 2251.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 2353.90 | 2404.15 | 2405.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 2353.90 | 2404.15 | 2405.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 2347.60 | 2392.84 | 2400.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 2384.55 | 2374.64 | 2387.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 2384.55 | 2374.64 | 2387.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2384.55 | 2374.64 | 2387.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:45:00 | 2396.50 | 2374.64 | 2387.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 2373.40 | 2374.39 | 2386.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 2339.90 | 2374.39 | 2386.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 09:15:00 | 2222.91 | 2286.93 | 2326.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 2219.50 | 2212.02 | 2252.80 | SL hit (close>ema200) qty=0.50 sl=2212.02 alert=retest2 |

### Cycle 181 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 2285.90 | 2253.50 | 2252.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 2295.10 | 2261.82 | 2256.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2279.50 | 2292.45 | 2275.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 2279.50 | 2292.45 | 2275.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2276.85 | 2289.33 | 2276.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 2276.75 | 2289.33 | 2276.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 2267.40 | 2284.94 | 2275.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 2274.75 | 2284.94 | 2275.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 2257.05 | 2279.37 | 2273.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 2257.05 | 2279.37 | 2273.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 2242.00 | 2271.89 | 2270.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 2242.00 | 2271.89 | 2270.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 2230.50 | 2263.61 | 2267.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 2220.25 | 2240.55 | 2252.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 2244.00 | 2236.21 | 2242.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 2244.00 | 2236.21 | 2242.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2244.00 | 2236.21 | 2242.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 2253.00 | 2236.21 | 2242.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2254.05 | 2239.78 | 2243.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 2260.20 | 2239.78 | 2243.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2242.50 | 2240.32 | 2243.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 2249.45 | 2240.32 | 2243.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 2260.20 | 2243.86 | 2244.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 2259.20 | 2243.86 | 2244.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2262.00 | 2247.49 | 2246.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 2275.00 | 2252.99 | 2248.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 2305.35 | 2306.26 | 2284.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 2307.65 | 2306.26 | 2284.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2365.60 | 2380.49 | 2356.44 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 2325.10 | 2349.73 | 2350.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 2281.10 | 2324.39 | 2336.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 2294.25 | 2273.87 | 2294.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 2294.25 | 2273.87 | 2294.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2294.25 | 2273.87 | 2294.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 2294.25 | 2273.87 | 2294.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 2296.70 | 2278.44 | 2295.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 2297.50 | 2278.44 | 2295.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 2304.00 | 2283.55 | 2295.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 2304.00 | 2283.55 | 2295.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 2303.40 | 2287.52 | 2296.55 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 2349.50 | 2308.95 | 2305.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 2396.35 | 2336.76 | 2319.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 2373.05 | 2379.85 | 2353.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:45:00 | 2372.30 | 2379.85 | 2353.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 2361.10 | 2372.00 | 2356.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 2378.35 | 2367.42 | 2357.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 2370.90 | 2391.35 | 2393.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2370.90 | 2391.35 | 2393.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 2357.85 | 2375.11 | 2382.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 2308.00 | 2299.41 | 2322.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 2302.50 | 2299.41 | 2322.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2301.00 | 2299.73 | 2320.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 2278.25 | 2294.23 | 2311.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 2336.95 | 2306.51 | 2308.47 | SL hit (close>static) qty=1.00 sl=2325.00 alert=retest2 |

### Cycle 187 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 2343.95 | 2313.99 | 2311.69 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 2267.00 | 2309.89 | 2312.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 2238.50 | 2281.38 | 2297.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2220.70 | 2139.37 | 2172.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2220.70 | 2139.37 | 2172.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2220.70 | 2139.37 | 2172.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2237.00 | 2139.37 | 2172.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2221.30 | 2155.75 | 2177.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 2232.00 | 2155.75 | 2177.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 2179.05 | 2167.33 | 2179.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 2165.45 | 2167.33 | 2179.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 2175.40 | 2169.37 | 2177.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:15:00 | 2177.45 | 2173.88 | 2178.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 2230.00 | 2175.12 | 2170.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 2230.00 | 2175.12 | 2170.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 2285.65 | 2219.43 | 2198.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 2295.00 | 2302.24 | 2261.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 2295.00 | 2302.24 | 2261.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2371.50 | 2334.71 | 2313.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 2384.40 | 2334.71 | 2313.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:15:00 | 2388.70 | 2350.49 | 2326.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 2247.70 | 2383.52 | 2379.15 | SL hit (close<static) qty=1.00 sl=2282.95 alert=retest2 |

### Cycle 190 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 11:15:00 | 2247.50 | 2356.32 | 2367.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2173.60 | 2238.87 | 2278.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2147.35 | 2143.76 | 2201.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:00:00 | 2147.35 | 2143.76 | 2201.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1886.10 | 1862.29 | 1914.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 1888.15 | 1862.29 | 1914.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1901.75 | 1846.63 | 1866.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 1901.75 | 1846.63 | 1866.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1834.05 | 1844.11 | 1863.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:45:00 | 1816.30 | 1838.77 | 1857.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1817.80 | 1814.45 | 1825.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 1825.30 | 1816.65 | 1825.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 13:15:00 | 1734.03 | 1762.90 | 1781.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 1725.48 | 1742.81 | 1766.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 1726.91 | 1742.81 | 1766.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-25 12:15:00 | 1642.77 | 1676.36 | 1709.10 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 191 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1572.90 | 1538.44 | 1536.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1609.75 | 1565.12 | 1550.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1559.85 | 1571.83 | 1560.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 1559.85 | 1571.83 | 1560.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1559.85 | 1571.83 | 1560.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1559.85 | 1571.83 | 1560.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1559.00 | 1569.26 | 1560.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1544.80 | 1569.26 | 1560.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1529.50 | 1561.31 | 1557.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1522.20 | 1561.31 | 1557.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1579.70 | 1564.99 | 1559.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 1594.15 | 1572.83 | 1564.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1504.00 | 1562.50 | 1562.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1504.00 | 1562.50 | 1562.77 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 1604.00 | 1556.14 | 1555.72 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 1512.30 | 1547.37 | 1551.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 1504.70 | 1538.83 | 1547.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1533.10 | 1530.06 | 1539.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1534.00 | 1530.06 | 1539.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1522.00 | 1527.96 | 1537.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1488.20 | 1516.22 | 1529.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:15:00 | 1487.60 | 1511.24 | 1526.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:45:00 | 1483.10 | 1499.46 | 1518.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:00:00 | 1489.00 | 1496.00 | 1510.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1469.10 | 1478.69 | 1491.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 1480.80 | 1478.69 | 1491.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1479.10 | 1473.59 | 1485.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 1460.90 | 1472.33 | 1480.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1413.79 | 1444.90 | 1462.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1413.22 | 1444.90 | 1462.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1414.55 | 1444.90 | 1462.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 14:15:00 | 1408.94 | 1436.34 | 1456.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1425.30 | 1422.58 | 1442.63 | SL hit (close>ema200) qty=0.50 sl=1422.58 alert=retest2 |

### Cycle 195 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1482.60 | 1454.11 | 1452.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1498.00 | 1462.88 | 1456.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1432.40 | 1475.63 | 1468.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1432.40 | 1475.63 | 1468.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1432.40 | 1475.63 | 1468.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1432.40 | 1475.63 | 1468.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1428.00 | 1466.10 | 1465.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 1427.60 | 1466.10 | 1465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1473.80 | 1467.74 | 1466.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:30:00 | 1469.30 | 1467.74 | 1466.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1448.50 | 1463.89 | 1464.44 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 1470.00 | 1465.11 | 1464.95 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1407.00 | 1454.27 | 1460.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1397.50 | 1442.92 | 1454.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1446.00 | 1416.79 | 1432.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1446.00 | 1416.79 | 1432.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1446.00 | 1416.79 | 1432.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1446.00 | 1416.79 | 1432.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1414.30 | 1416.29 | 1430.99 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1486.10 | 1441.31 | 1438.74 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1418.80 | 1436.97 | 1437.77 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1449.10 | 1437.99 | 1437.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 1458.30 | 1445.24 | 1441.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 1493.00 | 1496.16 | 1482.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1513.70 | 1496.16 | 1482.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1494.60 | 1527.93 | 1512.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-09 09:15:00 | 1494.60 | 1527.93 | 1512.18 | SL hit (close<ema400) qty=1.00 sl=1512.18 alert=retest1 |

### Cycle 202 — SELL (started 2026-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 15:15:00 | 1492.00 | 1505.28 | 1505.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 1471.80 | 1493.23 | 1498.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1498.80 | 1472.20 | 1482.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1498.80 | 1472.20 | 1482.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1498.80 | 1472.20 | 1482.33 | EMA400 retest candle locked (from downside) |

### Cycle 203 — BUY (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 09:15:00 | 1555.10 | 1498.33 | 1491.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 1606.50 | 1555.71 | 1529.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 1598.30 | 1609.74 | 1576.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:30:00 | 1590.00 | 1609.74 | 1576.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1602.00 | 1618.23 | 1598.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:45:00 | 1603.50 | 1618.23 | 1598.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1596.80 | 1613.94 | 1598.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 1597.10 | 1613.94 | 1598.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1583.60 | 1607.87 | 1597.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 1583.60 | 1607.87 | 1597.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 1565.00 | 1589.55 | 1590.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1524.60 | 1573.11 | 1582.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1496.80 | 1474.20 | 1490.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1496.80 | 1474.20 | 1490.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1496.80 | 1474.20 | 1490.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1505.00 | 1474.20 | 1490.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1485.10 | 1476.38 | 1490.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:30:00 | 1484.60 | 1480.37 | 1489.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 1483.00 | 1480.37 | 1489.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 1482.90 | 1481.24 | 1489.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 15:15:00 | 1410.37 | 1435.62 | 1452.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 15:15:00 | 1408.85 | 1435.62 | 1452.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 15:15:00 | 1408.76 | 1435.62 | 1452.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 11:15:00 | 1442.30 | 1428.00 | 1443.59 | SL hit (close>ema200) qty=0.50 sl=1428.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 1447.70 | 1432.97 | 1431.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1499.20 | 1452.17 | 1441.47 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-16 09:15:00 | 1227.50 | 2024-04-16 10:15:00 | 1237.63 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-04-16 12:00:00 | 1227.90 | 2024-04-18 12:15:00 | 1230.97 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-04-16 13:00:00 | 1227.55 | 2024-04-18 12:15:00 | 1230.97 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-04-16 15:15:00 | 1217.45 | 2024-04-18 12:15:00 | 1230.97 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-05-03 11:30:00 | 1222.58 | 2024-05-07 12:15:00 | 1161.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 13:15:00 | 1219.63 | 2024-05-07 13:15:00 | 1158.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 11:30:00 | 1222.58 | 2024-05-07 14:15:00 | 1194.50 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2024-05-03 13:15:00 | 1219.63 | 2024-05-07 14:15:00 | 1194.50 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2024-05-29 13:45:00 | 1104.43 | 2024-05-30 10:15:00 | 1113.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-05-30 09:15:00 | 1103.70 | 2024-05-30 10:15:00 | 1113.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-06-04 14:45:00 | 1080.00 | 2024-06-06 10:15:00 | 1107.20 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-06-05 09:30:00 | 1069.63 | 2024-06-06 10:15:00 | 1107.20 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2024-06-05 12:00:00 | 1079.97 | 2024-06-06 10:15:00 | 1107.20 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-06-18 11:15:00 | 1215.00 | 2024-06-18 12:15:00 | 1192.13 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-06-20 11:00:00 | 1185.78 | 2024-06-21 09:15:00 | 1228.00 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-07-12 12:00:00 | 1307.40 | 2024-07-16 09:15:00 | 1254.45 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2024-07-18 15:15:00 | 1232.50 | 2024-07-30 09:15:00 | 1170.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 10:00:00 | 1229.30 | 2024-07-30 09:15:00 | 1172.25 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2024-07-19 13:15:00 | 1233.95 | 2024-07-30 09:15:00 | 1173.34 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2024-07-18 15:15:00 | 1232.50 | 2024-07-30 11:15:00 | 1195.30 | STOP_HIT | 0.50 | 3.02% |
| SELL | retest2 | 2024-07-19 10:00:00 | 1229.30 | 2024-07-30 11:15:00 | 1195.30 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2024-07-19 13:15:00 | 1233.95 | 2024-07-30 11:15:00 | 1195.30 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2024-07-19 13:45:00 | 1235.10 | 2024-07-31 11:15:00 | 1200.47 | STOP_HIT | 1.00 | 2.80% |
| SELL | retest2 | 2024-07-25 09:15:00 | 1195.53 | 2024-07-31 11:15:00 | 1200.47 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-07-26 13:45:00 | 1197.72 | 2024-07-31 11:15:00 | 1200.47 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-07-26 14:15:00 | 1196.88 | 2024-07-31 11:15:00 | 1200.47 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-07-29 10:30:00 | 1197.50 | 2024-07-31 11:15:00 | 1200.47 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-07-29 13:15:00 | 1191.90 | 2024-07-31 11:15:00 | 1200.47 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-07-29 14:00:00 | 1192.47 | 2024-07-31 11:15:00 | 1200.47 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-07-31 09:45:00 | 1190.50 | 2024-07-31 11:15:00 | 1200.47 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-08-26 09:15:00 | 1366.18 | 2024-09-06 14:15:00 | 1413.65 | STOP_HIT | 1.00 | 3.47% |
| BUY | retest2 | 2024-09-25 10:30:00 | 1375.95 | 2024-09-25 11:15:00 | 1513.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-07 09:15:00 | 1528.60 | 2024-10-10 14:15:00 | 1538.05 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2024-10-07 11:15:00 | 1544.23 | 2024-10-10 14:15:00 | 1538.05 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-10-08 10:00:00 | 1532.45 | 2024-10-10 14:15:00 | 1538.05 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2024-10-14 09:15:00 | 1524.40 | 2024-10-21 14:15:00 | 1463.38 | PARTIAL | 0.50 | 4.00% |
| SELL | retest2 | 2024-10-15 13:00:00 | 1540.40 | 2024-10-22 09:15:00 | 1448.18 | PARTIAL | 0.50 | 5.99% |
| SELL | retest2 | 2024-10-14 09:15:00 | 1524.40 | 2024-10-24 11:15:00 | 1386.36 | TARGET_HIT | 0.50 | 9.06% |
| SELL | retest2 | 2024-10-15 13:00:00 | 1540.40 | 2024-10-24 13:15:00 | 1371.96 | TARGET_HIT | 0.50 | 10.93% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1482.13 | 2024-11-06 09:15:00 | 1630.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-19 12:15:00 | 1632.53 | 2024-11-21 11:15:00 | 1665.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-11-26 09:15:00 | 1694.98 | 2024-12-06 09:15:00 | 1864.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-20 13:30:00 | 1848.30 | 2024-12-26 10:15:00 | 1755.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 13:30:00 | 1848.30 | 2024-12-27 09:15:00 | 1786.48 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2025-01-10 13:15:00 | 1629.80 | 2025-01-16 11:15:00 | 1626.45 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-01-13 11:00:00 | 1637.08 | 2025-01-16 11:15:00 | 1626.45 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2025-01-14 10:15:00 | 1639.40 | 2025-01-16 11:15:00 | 1626.45 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-01-16 09:45:00 | 1636.58 | 2025-01-16 11:15:00 | 1626.45 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-01-24 11:00:00 | 1567.13 | 2025-01-27 09:15:00 | 1488.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 11:00:00 | 1567.13 | 2025-01-28 12:15:00 | 1496.98 | STOP_HIT | 0.50 | 4.48% |
| BUY | retest2 | 2025-01-30 10:15:00 | 1529.03 | 2025-01-30 13:15:00 | 1506.25 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest1 | 2025-02-04 09:15:00 | 1618.10 | 2025-02-06 12:15:00 | 1605.53 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2025-02-04 12:00:00 | 1611.75 | 2025-02-06 12:15:00 | 1605.53 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-02-14 09:15:00 | 1519.38 | 2025-02-14 14:15:00 | 1551.48 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-02-17 10:00:00 | 1504.90 | 2025-02-19 09:15:00 | 1538.08 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-02-21 11:30:00 | 1610.10 | 2025-02-24 11:15:00 | 1537.50 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2025-03-05 11:15:00 | 1359.00 | 2025-03-06 14:15:00 | 1396.58 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-03-18 11:30:00 | 1277.40 | 2025-03-19 09:15:00 | 1305.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-03-18 13:00:00 | 1275.83 | 2025-03-19 09:15:00 | 1305.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-03-27 13:15:00 | 1377.38 | 2025-04-01 10:15:00 | 1361.95 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-03-27 15:00:00 | 1382.93 | 2025-04-01 14:15:00 | 1367.18 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-03-28 10:00:00 | 1379.03 | 2025-04-01 14:15:00 | 1367.18 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-03-28 10:30:00 | 1379.93 | 2025-04-01 14:15:00 | 1367.18 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-03-28 14:15:00 | 1384.18 | 2025-04-01 14:15:00 | 1367.18 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-04-11 11:00:00 | 1206.30 | 2025-04-15 10:15:00 | 1267.55 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest2 | 2025-04-21 09:15:00 | 1259.40 | 2025-04-25 11:15:00 | 1289.00 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-05-05 12:45:00 | 1245.35 | 2025-05-07 09:15:00 | 1183.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 12:45:00 | 1245.35 | 2025-05-07 14:15:00 | 1208.10 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2025-05-08 11:45:00 | 1239.95 | 2025-05-08 12:15:00 | 1235.65 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-05-08 12:15:00 | 1245.35 | 2025-05-08 12:15:00 | 1235.65 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-06-05 14:00:00 | 1787.00 | 2025-06-12 12:15:00 | 1795.55 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1794.90 | 2025-06-12 12:15:00 | 1795.55 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-06-13 09:15:00 | 1788.20 | 2025-06-17 10:15:00 | 1816.85 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-06-19 09:15:00 | 1782.45 | 2025-06-20 14:15:00 | 1693.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 09:15:00 | 1782.45 | 2025-06-23 09:15:00 | 1731.20 | STOP_HIT | 0.50 | 2.88% |
| BUY | retest2 | 2025-06-26 09:15:00 | 1766.10 | 2025-06-26 10:15:00 | 1746.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-30 10:30:00 | 1736.80 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2025-07-01 09:45:00 | 1739.90 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | 2.15% |
| SELL | retest2 | 2025-07-01 13:15:00 | 1732.80 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2025-07-03 13:00:00 | 1739.35 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2025-07-04 15:15:00 | 1699.00 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-07-07 14:00:00 | 1701.45 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-07-07 15:15:00 | 1701.25 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-07-08 09:30:00 | 1700.65 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-07-09 11:15:00 | 1666.95 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-07-09 14:30:00 | 1677.65 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-07-10 09:15:00 | 1674.85 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-10 09:45:00 | 1672.90 | 2025-07-10 14:15:00 | 1702.45 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-01 09:15:00 | 1914.75 | 2025-08-13 10:15:00 | 1991.95 | STOP_HIT | 1.00 | 4.03% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1929.00 | 2025-08-13 10:15:00 | 1991.95 | STOP_HIT | 1.00 | 3.26% |
| SELL | retest2 | 2025-08-22 12:00:00 | 1971.40 | 2025-08-25 09:15:00 | 2069.00 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest2 | 2025-09-03 10:30:00 | 2248.00 | 2025-09-03 11:15:00 | 2194.35 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-09-09 13:30:00 | 2175.00 | 2025-09-09 15:15:00 | 2180.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-09-16 12:00:00 | 2183.25 | 2025-09-18 09:15:00 | 2224.40 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-09-16 12:30:00 | 2184.00 | 2025-09-18 09:15:00 | 2224.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-09-17 09:30:00 | 2179.45 | 2025-09-18 09:15:00 | 2224.40 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-09-17 12:00:00 | 2182.95 | 2025-09-18 09:15:00 | 2224.40 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-09-19 09:30:00 | 2235.95 | 2025-09-19 10:15:00 | 2179.15 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2064.05 | 2025-10-01 10:15:00 | 1988.11 | PARTIAL | 0.50 | 3.68% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2064.05 | 2025-10-03 09:15:00 | 2003.85 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2025-09-26 13:00:00 | 2092.75 | 2025-10-03 13:15:00 | 1960.85 | PARTIAL | 0.50 | 6.30% |
| SELL | retest2 | 2025-09-26 15:00:00 | 2069.10 | 2025-10-03 13:15:00 | 1965.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 13:00:00 | 2092.75 | 2025-10-06 09:15:00 | 2003.95 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2025-09-26 15:00:00 | 2069.10 | 2025-10-06 09:15:00 | 2003.95 | STOP_HIT | 0.50 | 3.15% |
| BUY | retest2 | 2025-10-16 09:15:00 | 2057.20 | 2025-10-17 13:15:00 | 2037.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-16 11:00:00 | 2048.05 | 2025-10-17 13:15:00 | 2037.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-10-16 12:00:00 | 2052.00 | 2025-10-17 13:15:00 | 2037.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-04 10:30:00 | 2404.45 | 2025-11-06 09:15:00 | 2336.60 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-11-04 11:30:00 | 2399.95 | 2025-11-06 09:15:00 | 2336.60 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-11-11 11:00:00 | 2171.30 | 2025-11-12 11:15:00 | 2240.00 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-11-11 12:45:00 | 2173.75 | 2025-11-12 11:15:00 | 2240.00 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2025-11-19 09:15:00 | 2250.75 | 2025-11-19 11:15:00 | 2221.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-11-19 10:45:00 | 2249.25 | 2025-11-19 11:15:00 | 2221.30 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-28 10:30:00 | 2296.40 | 2025-12-08 09:15:00 | 2353.90 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2025-12-01 09:15:00 | 2322.75 | 2025-12-08 09:15:00 | 2353.90 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-12-09 09:15:00 | 2339.90 | 2025-12-10 09:15:00 | 2222.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-09 09:15:00 | 2339.90 | 2025-12-11 12:15:00 | 2219.50 | STOP_HIT | 0.50 | 5.15% |
| BUY | retest2 | 2026-01-05 10:15:00 | 2378.35 | 2026-01-08 09:15:00 | 2370.90 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-01-13 14:00:00 | 2278.25 | 2026-01-16 09:15:00 | 2336.95 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-01-22 13:15:00 | 2165.45 | 2026-01-27 15:15:00 | 2230.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2026-01-22 15:00:00 | 2175.40 | 2026-01-27 15:15:00 | 2230.00 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2177.45 | 2026-01-27 15:15:00 | 2230.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-02 10:15:00 | 2384.40 | 2026-02-04 10:15:00 | 2247.70 | STOP_HIT | 1.00 | -5.73% |
| BUY | retest2 | 2026-02-02 13:15:00 | 2388.70 | 2026-02-04 10:15:00 | 2247.70 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2026-02-17 12:45:00 | 1816.30 | 2026-02-23 13:15:00 | 1734.03 | PARTIAL | 0.50 | 4.53% |
| SELL | retest2 | 2026-02-19 09:30:00 | 1817.80 | 2026-02-24 09:15:00 | 1725.48 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2026-02-19 10:45:00 | 1825.30 | 2026-02-24 09:15:00 | 1726.91 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2026-02-17 12:45:00 | 1816.30 | 2026-02-25 12:15:00 | 1642.77 | TARGET_HIT | 0.50 | 9.55% |
| SELL | retest2 | 2026-02-19 09:30:00 | 1817.80 | 2026-02-26 09:15:00 | 1662.85 | STOP_HIT | 0.50 | 8.52% |
| SELL | retest2 | 2026-02-19 10:45:00 | 1825.30 | 2026-02-26 09:15:00 | 1662.85 | STOP_HIT | 0.50 | 8.90% |
| BUY | retest2 | 2026-03-12 13:00:00 | 1594.15 | 2026-03-13 09:15:00 | 1504.00 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1488.20 | 2026-03-23 13:15:00 | 1413.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 13:15:00 | 1487.60 | 2026-03-23 13:15:00 | 1413.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 14:45:00 | 1483.10 | 2026-03-23 13:15:00 | 1414.55 | PARTIAL | 0.50 | 4.62% |
| SELL | retest2 | 2026-03-18 12:00:00 | 1489.00 | 2026-03-23 14:15:00 | 1408.94 | PARTIAL | 0.50 | 5.38% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1488.20 | 2026-03-24 11:15:00 | 1425.30 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2026-03-17 13:15:00 | 1487.60 | 2026-03-24 11:15:00 | 1425.30 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2026-03-17 14:45:00 | 1483.10 | 2026-03-24 11:15:00 | 1425.30 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2026-03-18 12:00:00 | 1489.00 | 2026-03-24 11:15:00 | 1425.30 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2026-03-20 14:30:00 | 1460.90 | 2026-03-25 09:15:00 | 1482.60 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-03-24 13:30:00 | 1464.90 | 2026-03-25 09:15:00 | 1482.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-03-24 14:15:00 | 1464.40 | 2026-03-25 09:15:00 | 1482.60 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest1 | 2026-04-08 09:15:00 | 1513.70 | 2026-04-09 09:15:00 | 1494.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-04-27 12:30:00 | 1484.60 | 2026-04-29 15:15:00 | 1410.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:15:00 | 1483.00 | 2026-04-29 15:15:00 | 1408.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1482.90 | 2026-04-29 15:15:00 | 1408.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 12:30:00 | 1484.60 | 2026-04-30 11:15:00 | 1442.30 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-04-27 13:15:00 | 1483.00 | 2026-04-30 11:15:00 | 1442.30 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1482.90 | 2026-04-30 11:15:00 | 1442.30 | STOP_HIT | 0.50 | 2.74% |
