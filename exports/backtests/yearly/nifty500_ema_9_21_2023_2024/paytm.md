# One 97 Communications Ltd. (PAYTM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1188.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 199 |
| ALERT1 | 147 |
| ALERT2 | 148 |
| ALERT2_SKIP | 98 |
| ALERT3 | 329 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 142 |
| PARTIAL | 29 |
| TARGET_HIT | 19 |
| STOP_HIT | 124 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 172 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 79 / 93
- **Target hits / Stop hits / Partials:** 19 / 124 / 29
- **Avg / median % per leg:** 1.04% / -0.66%
- **Sum % (uncompounded):** 179.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 78 | 22 | 28.2% | 11 | 66 | 1 | 0.03% | 2.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.40% | 8.8% |
| BUY @ 3rd Alert (retest2) | 76 | 20 | 26.3% | 11 | 65 | 0 | -0.09% | -6.7% |
| SELL (all) | 94 | 57 | 60.6% | 8 | 58 | 28 | 1.89% | 177.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 94 | 57 | 60.6% | 8 | 58 | 28 | 1.89% | 177.2% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.40% | 8.8% |
| retest2 (combined) | 170 | 77 | 45.3% | 19 | 123 | 28 | 1.00% | 170.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 715.30 | 704.17 | 703.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 720.00 | 709.55 | 706.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 12:15:00 | 720.00 | 720.43 | 716.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 714.50 | 719.86 | 717.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 714.50 | 719.86 | 717.27 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 709.75 | 715.55 | 715.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 15:15:00 | 705.90 | 713.62 | 715.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 14:15:00 | 710.60 | 708.63 | 711.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 14:15:00 | 710.60 | 708.63 | 711.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 710.60 | 708.63 | 711.33 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 13:15:00 | 713.70 | 710.93 | 710.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 14:15:00 | 720.60 | 712.87 | 711.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 15:15:00 | 714.95 | 715.15 | 713.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 15:15:00 | 714.95 | 715.15 | 713.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 15:15:00 | 714.95 | 715.15 | 713.84 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 13:15:00 | 710.00 | 714.85 | 715.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 14:15:00 | 706.15 | 713.11 | 714.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 09:15:00 | 703.50 | 699.01 | 701.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-01 09:15:00 | 703.50 | 699.01 | 701.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 703.50 | 699.01 | 701.54 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 09:15:00 | 705.75 | 702.54 | 702.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 10:15:00 | 712.20 | 704.47 | 703.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 11:15:00 | 712.85 | 712.94 | 709.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 12:15:00 | 711.50 | 712.66 | 709.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 12:15:00 | 711.50 | 712.66 | 709.63 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 879.00 | 880.79 | 880.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 871.95 | 879.02 | 880.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 853.40 | 844.45 | 851.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 853.40 | 844.45 | 851.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 853.40 | 844.45 | 851.68 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 13:15:00 | 857.15 | 852.88 | 852.61 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 14:15:00 | 847.45 | 851.79 | 852.14 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 861.90 | 853.56 | 852.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 14:15:00 | 868.05 | 858.03 | 855.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 850.60 | 857.77 | 855.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 850.60 | 857.77 | 855.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 850.60 | 857.77 | 855.91 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 12:15:00 | 850.15 | 854.49 | 854.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 13:15:00 | 849.00 | 853.40 | 854.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 09:15:00 | 852.00 | 851.86 | 853.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 852.00 | 851.86 | 853.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 852.00 | 851.86 | 853.19 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 15:15:00 | 854.80 | 850.75 | 850.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 864.40 | 853.48 | 851.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 863.60 | 864.23 | 859.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 10:15:00 | 861.25 | 863.64 | 859.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 861.25 | 863.64 | 859.41 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 14:15:00 | 850.30 | 856.72 | 857.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 833.30 | 851.04 | 854.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 09:15:00 | 831.00 | 820.07 | 829.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 09:15:00 | 831.00 | 820.07 | 829.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 831.00 | 820.07 | 829.15 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 12:15:00 | 859.65 | 836.93 | 835.27 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 843.70 | 850.42 | 851.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 14:15:00 | 840.00 | 848.34 | 850.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 14:15:00 | 842.40 | 837.19 | 842.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 14:15:00 | 842.40 | 837.19 | 842.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 14:15:00 | 842.40 | 837.19 | 842.22 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 14:15:00 | 850.90 | 843.83 | 843.33 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 12:15:00 | 837.25 | 842.90 | 843.22 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 09:15:00 | 852.55 | 844.16 | 843.55 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 820.70 | 840.03 | 841.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 799.50 | 825.50 | 834.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 11:15:00 | 791.10 | 789.78 | 803.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 779.40 | 770.54 | 777.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 779.40 | 770.54 | 777.86 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 13:15:00 | 793.05 | 782.72 | 781.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 796.30 | 785.44 | 783.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 14:15:00 | 788.20 | 793.25 | 789.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 14:15:00 | 788.20 | 793.25 | 789.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 788.20 | 793.25 | 789.63 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 780.30 | 786.44 | 787.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 12:15:00 | 776.50 | 784.45 | 786.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 784.75 | 775.54 | 778.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 784.75 | 775.54 | 778.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 784.75 | 775.54 | 778.62 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 793.70 | 782.20 | 781.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 848.80 | 801.36 | 791.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 835.00 | 836.67 | 818.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 14:15:00 | 836.90 | 831.87 | 827.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 836.90 | 831.87 | 827.04 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 13:15:00 | 861.75 | 865.42 | 865.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 09:15:00 | 842.10 | 858.97 | 862.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 09:15:00 | 857.60 | 848.00 | 853.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 09:15:00 | 857.60 | 848.00 | 853.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 857.60 | 848.00 | 853.63 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 879.50 | 859.99 | 857.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 12:15:00 | 883.10 | 870.48 | 863.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 13:15:00 | 904.75 | 909.03 | 898.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 14:15:00 | 899.40 | 907.10 | 898.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 899.40 | 907.10 | 898.88 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 11:15:00 | 876.00 | 893.21 | 894.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 11:15:00 | 862.80 | 876.53 | 884.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 14:15:00 | 862.15 | 861.26 | 868.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 11:15:00 | 858.05 | 856.20 | 860.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 858.05 | 856.20 | 860.17 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 10:15:00 | 867.35 | 859.80 | 859.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 11:15:00 | 870.10 | 861.86 | 860.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 09:15:00 | 890.80 | 892.18 | 882.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 887.25 | 891.08 | 886.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 887.25 | 891.08 | 886.47 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 879.80 | 893.22 | 893.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 835.70 | 868.19 | 879.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 855.65 | 847.37 | 860.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 855.65 | 847.37 | 860.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 855.65 | 847.37 | 860.35 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 870.05 | 860.95 | 860.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 893.50 | 868.06 | 864.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 13:15:00 | 874.90 | 876.63 | 871.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 14:15:00 | 872.95 | 875.89 | 871.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 872.95 | 875.89 | 871.31 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 10:15:00 | 853.05 | 867.99 | 868.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 12:15:00 | 843.70 | 853.65 | 859.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 849.70 | 845.63 | 851.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 12:15:00 | 849.70 | 845.63 | 851.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 849.70 | 845.63 | 851.24 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 12:15:00 | 853.75 | 849.22 | 848.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 10:15:00 | 855.05 | 851.95 | 850.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 15:15:00 | 850.50 | 852.49 | 851.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 15:15:00 | 850.50 | 852.49 | 851.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 15:15:00 | 850.50 | 852.49 | 851.38 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 14:15:00 | 934.90 | 949.05 | 949.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 15:15:00 | 932.50 | 945.74 | 947.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 943.30 | 939.14 | 942.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 943.30 | 939.14 | 942.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 943.30 | 939.14 | 942.32 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 13:15:00 | 948.20 | 942.17 | 941.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 09:15:00 | 971.95 | 949.62 | 945.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 09:15:00 | 961.20 | 975.37 | 968.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 09:15:00 | 961.20 | 975.37 | 968.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 961.20 | 975.37 | 968.34 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 12:15:00 | 935.30 | 959.27 | 962.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 929.45 | 950.21 | 957.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 891.40 | 888.83 | 906.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 10:15:00 | 905.60 | 892.18 | 906.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 905.60 | 892.18 | 906.67 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 15:15:00 | 934.20 | 910.66 | 907.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 940.05 | 916.54 | 910.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 924.95 | 929.30 | 920.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 15:15:00 | 916.55 | 926.75 | 920.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 916.55 | 926.75 | 920.28 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 14:15:00 | 912.10 | 918.44 | 918.59 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 924.15 | 918.87 | 918.71 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 10:15:00 | 912.95 | 918.48 | 918.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 11:15:00 | 907.35 | 916.25 | 917.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 12:15:00 | 886.65 | 886.59 | 894.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 14:15:00 | 884.00 | 886.29 | 893.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 884.00 | 886.29 | 893.19 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 15:15:00 | 897.60 | 895.42 | 895.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 10:15:00 | 899.90 | 896.88 | 896.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 12:15:00 | 895.55 | 897.08 | 896.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 12:15:00 | 895.55 | 897.08 | 896.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 895.55 | 897.08 | 896.27 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-11-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 15:15:00 | 890.85 | 895.86 | 895.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 09:15:00 | 882.50 | 893.19 | 894.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 14:15:00 | 891.50 | 890.60 | 892.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 14:15:00 | 891.50 | 890.60 | 892.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 891.50 | 890.60 | 892.51 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 10:15:00 | 897.60 | 893.36 | 893.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 915.25 | 898.21 | 895.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 11:15:00 | 910.50 | 911.66 | 906.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 12:15:00 | 909.00 | 911.12 | 906.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 12:15:00 | 909.00 | 911.12 | 906.35 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 09:15:00 | 887.05 | 903.98 | 904.34 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 903.70 | 899.00 | 898.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 13:15:00 | 911.30 | 903.91 | 901.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 15:15:00 | 921.00 | 921.70 | 916.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 899.60 | 917.28 | 915.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 899.60 | 917.28 | 915.11 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 10:15:00 | 898.30 | 913.49 | 913.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 892.95 | 906.60 | 910.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 876.65 | 875.54 | 884.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 11:15:00 | 883.70 | 877.26 | 883.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 883.70 | 877.26 | 883.79 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 09:15:00 | 623.30 | 616.25 | 615.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 09:15:00 | 630.85 | 623.54 | 620.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 623.55 | 627.02 | 623.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 623.55 | 627.02 | 623.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 623.55 | 627.02 | 623.20 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 13:15:00 | 633.25 | 633.95 | 634.02 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 14:15:00 | 634.55 | 634.07 | 634.07 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 09:15:00 | 631.65 | 633.74 | 633.93 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 09:15:00 | 634.85 | 633.95 | 633.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 11:15:00 | 640.05 | 635.68 | 634.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 14:15:00 | 635.25 | 635.83 | 635.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 14:15:00 | 635.25 | 635.83 | 635.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 635.25 | 635.83 | 635.08 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 751.00 | 766.48 | 766.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 09:15:00 | 748.10 | 759.33 | 762.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 13:15:00 | 756.30 | 755.62 | 759.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 13:15:00 | 756.30 | 755.62 | 759.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 756.30 | 755.62 | 759.69 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 15:15:00 | 763.95 | 759.97 | 759.88 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 09:15:00 | 757.70 | 759.52 | 759.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 14:15:00 | 753.45 | 758.00 | 758.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 09:15:00 | 762.80 | 757.52 | 758.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 762.80 | 757.52 | 758.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 762.80 | 757.52 | 758.45 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 11:15:00 | 761.90 | 759.28 | 759.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 773.00 | 762.27 | 760.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 14:15:00 | 761.15 | 762.75 | 761.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 14:15:00 | 761.15 | 762.75 | 761.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 761.15 | 762.75 | 761.60 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 09:15:00 | 609.00 | 731.72 | 747.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 09:15:00 | 487.20 | 610.38 | 669.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 468.20 | 458.03 | 511.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 490.65 | 462.82 | 487.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 490.65 | 462.82 | 487.54 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 12:15:00 | 358.35 | 349.50 | 349.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 09:15:00 | 376.25 | 358.30 | 353.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-22 12:15:00 | 392.80 | 392.92 | 384.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 09:15:00 | 396.00 | 392.21 | 386.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 396.00 | 392.21 | 386.39 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 406.20 | 416.00 | 416.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 09:15:00 | 387.20 | 406.41 | 411.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 401.75 | 397.34 | 404.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 401.75 | 397.34 | 404.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 401.75 | 397.34 | 404.12 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 421.25 | 410.11 | 408.71 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 408.20 | 412.18 | 412.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 390.25 | 404.18 | 408.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 396.30 | 396.25 | 402.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 403.00 | 396.38 | 400.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 403.00 | 396.38 | 400.91 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 370.70 | 362.43 | 362.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 09:15:00 | 389.20 | 371.01 | 366.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 10:15:00 | 417.00 | 418.02 | 408.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 11:15:00 | 407.45 | 415.91 | 407.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 11:15:00 | 407.45 | 415.91 | 407.98 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-03-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 15:15:00 | 403.50 | 408.13 | 408.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 09:15:00 | 400.75 | 406.65 | 407.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 11:15:00 | 401.90 | 400.15 | 402.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 12:15:00 | 408.35 | 401.79 | 403.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 408.35 | 401.79 | 403.15 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 12:15:00 | 404.95 | 402.66 | 402.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 11:15:00 | 407.10 | 404.97 | 403.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 14:15:00 | 410.50 | 412.20 | 409.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 14:15:00 | 410.90 | 411.93 | 410.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 410.90 | 411.93 | 410.76 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 09:15:00 | 411.00 | 413.16 | 413.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 11:15:00 | 407.30 | 411.34 | 412.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 13:15:00 | 394.80 | 391.64 | 395.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 13:15:00 | 394.80 | 391.64 | 395.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 394.80 | 391.64 | 395.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 14:00:00 | 394.80 | 391.64 | 395.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 388.00 | 390.91 | 394.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:45:00 | 387.75 | 391.02 | 392.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 383.50 | 390.47 | 391.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 11:15:00 | 386.05 | 382.27 | 382.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 11:15:00 | 386.05 | 382.27 | 382.26 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 10:15:00 | 381.20 | 382.29 | 382.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 12:15:00 | 378.65 | 381.20 | 381.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 11:15:00 | 377.00 | 376.93 | 378.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 11:15:00 | 377.00 | 376.93 | 378.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 377.00 | 376.93 | 378.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:00:00 | 377.00 | 376.93 | 378.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 375.30 | 376.61 | 378.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 11:15:00 | 372.80 | 376.16 | 377.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 12:15:00 | 380.55 | 372.99 | 374.37 | SL hit (close>static) qty=1.00 sl=378.80 alert=retest2 |

### Cycle 63 — BUY (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 10:15:00 | 348.70 | 337.14 | 336.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 11:15:00 | 352.00 | 346.32 | 342.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 14:15:00 | 343.00 | 346.18 | 343.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 14:15:00 | 343.00 | 346.18 | 343.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 343.00 | 346.18 | 343.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 343.00 | 346.18 | 343.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 343.00 | 345.54 | 343.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 339.30 | 345.54 | 343.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 333.95 | 343.22 | 342.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:00:00 | 333.95 | 343.22 | 342.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 339.75 | 342.53 | 342.39 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 11:15:00 | 338.80 | 341.78 | 342.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 09:15:00 | 333.50 | 339.41 | 340.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 12:15:00 | 341.55 | 338.89 | 340.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 12:15:00 | 341.55 | 338.89 | 340.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 341.55 | 338.89 | 340.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:45:00 | 343.45 | 338.89 | 340.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 342.60 | 339.64 | 340.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 14:15:00 | 343.65 | 339.64 | 340.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 347.60 | 342.02 | 341.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 10:15:00 | 350.65 | 345.05 | 343.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 346.20 | 350.35 | 347.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 346.20 | 350.35 | 347.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 346.20 | 350.35 | 347.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 345.80 | 350.35 | 347.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 350.95 | 350.47 | 347.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:30:00 | 358.85 | 352.56 | 349.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 09:15:00 | 339.40 | 353.78 | 353.59 | SL hit (close<static) qty=1.00 sl=345.65 alert=retest2 |

### Cycle 66 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 339.65 | 350.95 | 352.32 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 357.15 | 350.88 | 350.27 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 346.85 | 350.57 | 350.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 341.95 | 347.97 | 349.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 359.45 | 349.47 | 349.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 359.45 | 349.47 | 349.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 359.45 | 349.47 | 349.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:30:00 | 359.45 | 349.47 | 349.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 359.45 | 351.47 | 350.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 09:15:00 | 377.40 | 360.95 | 356.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 11:15:00 | 367.80 | 375.43 | 368.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 11:15:00 | 367.80 | 375.43 | 368.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 367.80 | 375.43 | 368.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 367.80 | 375.43 | 368.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 366.25 | 373.60 | 368.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:30:00 | 365.85 | 373.60 | 368.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 375.10 | 373.90 | 369.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 363.90 | 373.90 | 369.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 359.25 | 370.97 | 368.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 359.25 | 370.97 | 368.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 363.00 | 369.37 | 367.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 378.75 | 369.37 | 367.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 357.40 | 371.58 | 370.79 | SL hit (close<static) qty=1.00 sl=358.55 alert=retest2 |

### Cycle 70 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 357.40 | 368.75 | 369.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 339.55 | 356.80 | 362.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 348.15 | 346.03 | 353.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 10:00:00 | 348.15 | 346.03 | 353.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 362.95 | 349.38 | 351.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:00:00 | 362.95 | 349.38 | 351.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 377.30 | 354.97 | 353.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 381.30 | 360.23 | 355.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 385.40 | 387.49 | 378.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 13:15:00 | 382.00 | 385.72 | 380.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 382.00 | 385.72 | 380.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 13:30:00 | 381.35 | 385.72 | 380.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 381.00 | 384.13 | 380.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 384.05 | 384.13 | 380.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:00:00 | 382.80 | 381.83 | 380.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-13 09:15:00 | 422.46 | 396.96 | 388.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 407.75 | 415.66 | 415.83 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 14:15:00 | 412.00 | 410.80 | 410.79 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 409.00 | 410.44 | 410.62 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 416.00 | 411.55 | 411.11 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 408.60 | 411.21 | 411.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 15:15:00 | 408.25 | 410.08 | 410.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 405.25 | 405.23 | 407.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 15:00:00 | 405.25 | 405.23 | 407.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 402.90 | 404.49 | 406.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 409.55 | 404.49 | 406.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 406.50 | 405.02 | 406.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 407.90 | 405.02 | 406.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 402.15 | 404.45 | 405.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:15:00 | 401.95 | 404.45 | 405.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 419.50 | 407.06 | 406.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 419.50 | 407.06 | 406.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 14:15:00 | 421.10 | 414.59 | 411.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 11:15:00 | 417.25 | 417.57 | 414.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 11:30:00 | 416.65 | 417.57 | 414.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 418.85 | 417.83 | 414.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:45:00 | 416.60 | 417.83 | 414.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 414.45 | 417.27 | 415.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 414.45 | 417.27 | 415.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 412.85 | 416.38 | 415.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 412.25 | 416.38 | 415.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 419.75 | 416.88 | 415.74 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 414.05 | 415.12 | 415.19 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 418.95 | 415.89 | 415.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 11:15:00 | 422.00 | 417.11 | 416.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 454.65 | 461.12 | 446.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:45:00 | 457.15 | 461.12 | 446.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 462.50 | 459.13 | 450.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 471.00 | 459.49 | 451.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 11:30:00 | 469.25 | 462.24 | 455.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:30:00 | 470.40 | 465.00 | 458.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 473.30 | 465.21 | 459.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 471.05 | 474.92 | 470.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 471.70 | 474.92 | 470.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 472.30 | 473.77 | 470.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:30:00 | 472.85 | 473.77 | 470.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 465.40 | 472.10 | 470.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 465.40 | 472.10 | 470.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 465.20 | 470.72 | 469.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 469.75 | 470.72 | 469.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 10:45:00 | 466.90 | 470.06 | 469.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 462.05 | 468.70 | 469.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 462.05 | 468.70 | 469.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 10:15:00 | 461.10 | 467.18 | 468.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 11:15:00 | 450.10 | 444.28 | 451.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 11:15:00 | 450.10 | 444.28 | 451.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 450.10 | 444.28 | 451.42 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 460.05 | 454.75 | 454.24 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 09:15:00 | 449.15 | 453.65 | 453.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 11:15:00 | 446.00 | 451.50 | 452.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 454.00 | 450.86 | 452.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 454.00 | 450.86 | 452.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 454.00 | 450.86 | 452.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 454.00 | 450.86 | 452.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 456.10 | 451.91 | 452.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 459.70 | 451.91 | 452.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 455.80 | 452.69 | 452.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 09:30:00 | 453.60 | 452.88 | 453.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 13:15:00 | 461.70 | 454.17 | 453.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 461.70 | 454.17 | 453.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 474.50 | 463.19 | 459.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 493.45 | 494.11 | 483.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 15:00:00 | 493.45 | 494.11 | 483.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 493.10 | 496.55 | 493.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 493.10 | 496.55 | 493.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 493.90 | 496.02 | 493.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 503.10 | 496.02 | 493.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 14:30:00 | 496.30 | 498.28 | 496.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:15:00 | 498.00 | 498.28 | 496.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 12:15:00 | 482.00 | 503.48 | 504.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 482.00 | 503.48 | 504.75 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 10:15:00 | 507.10 | 505.14 | 504.92 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 495.35 | 503.01 | 504.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 487.95 | 499.99 | 502.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 506.45 | 499.53 | 501.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 506.45 | 499.53 | 501.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 506.45 | 499.53 | 501.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 508.75 | 499.53 | 501.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 507.95 | 501.21 | 502.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:45:00 | 508.20 | 501.21 | 502.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 505.85 | 502.59 | 502.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 516.50 | 506.29 | 504.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 516.30 | 516.32 | 510.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 14:00:00 | 516.30 | 516.32 | 510.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 508.70 | 514.79 | 510.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:15:00 | 507.20 | 514.79 | 510.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 507.20 | 513.27 | 510.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 514.15 | 513.27 | 510.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 504.80 | 511.24 | 510.96 | SL hit (close<static) qty=1.00 sl=506.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 508.40 | 510.67 | 510.73 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 514.90 | 511.25 | 510.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 519.55 | 512.91 | 511.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 510.50 | 512.74 | 511.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 510.50 | 512.74 | 511.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 510.50 | 512.74 | 511.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 510.50 | 512.74 | 511.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 511.10 | 512.41 | 511.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:15:00 | 512.25 | 512.41 | 511.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 12:15:00 | 512.00 | 511.71 | 511.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 505.45 | 511.89 | 511.88 | SL hit (close<static) qty=1.00 sl=507.90 alert=retest2 |

### Cycle 90 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 508.00 | 511.11 | 511.53 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 10:15:00 | 518.60 | 513.05 | 512.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 11:15:00 | 525.50 | 515.54 | 513.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 566.15 | 572.25 | 557.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 15:00:00 | 566.15 | 572.25 | 557.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 571.45 | 575.17 | 571.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:00:00 | 571.45 | 575.17 | 571.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 568.85 | 573.91 | 571.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 568.85 | 573.91 | 571.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 13:15:00 | 550.10 | 569.14 | 569.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 13:15:00 | 523.30 | 550.52 | 557.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 544.65 | 541.45 | 547.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 14:15:00 | 544.65 | 541.45 | 547.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 544.65 | 541.45 | 547.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:45:00 | 546.40 | 541.45 | 547.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 546.00 | 542.36 | 547.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 552.85 | 542.36 | 547.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 555.60 | 545.01 | 548.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 555.00 | 545.01 | 548.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 554.20 | 546.84 | 548.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 554.20 | 546.84 | 548.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 538.50 | 546.95 | 548.33 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 555.00 | 548.16 | 547.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 618.85 | 563.41 | 554.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 590.90 | 590.95 | 575.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:00:00 | 590.90 | 590.95 | 575.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 600.40 | 599.95 | 592.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:15:00 | 613.15 | 603.05 | 596.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 595.40 | 609.13 | 609.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 595.40 | 609.13 | 609.66 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 12:15:00 | 613.40 | 609.98 | 609.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 13:15:00 | 616.25 | 611.24 | 610.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 10:15:00 | 661.25 | 665.37 | 654.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 11:00:00 | 661.25 | 665.37 | 654.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 662.90 | 663.81 | 655.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 665.10 | 663.81 | 655.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 14:15:00 | 663.85 | 664.13 | 660.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 14:15:00 | 655.15 | 662.33 | 660.16 | SL hit (close<static) qty=1.00 sl=655.50 alert=retest2 |

### Cycle 96 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 657.80 | 668.64 | 670.04 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 11:15:00 | 673.75 | 662.26 | 662.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 678.20 | 667.49 | 664.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 09:15:00 | 690.10 | 695.75 | 687.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 690.10 | 695.75 | 687.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 690.10 | 695.75 | 687.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 686.55 | 695.75 | 687.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 690.30 | 694.66 | 687.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 688.90 | 694.66 | 687.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 687.60 | 693.25 | 687.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:00:00 | 687.60 | 693.25 | 687.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 684.00 | 691.40 | 687.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:00:00 | 684.00 | 691.40 | 687.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 683.25 | 689.77 | 686.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 683.25 | 689.77 | 686.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 15:15:00 | 672.60 | 683.60 | 684.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 666.60 | 680.20 | 682.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 12:15:00 | 682.00 | 679.69 | 681.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 12:15:00 | 682.00 | 679.69 | 681.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 682.00 | 679.69 | 681.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:30:00 | 688.15 | 679.69 | 681.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 680.90 | 679.93 | 681.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:45:00 | 688.00 | 679.93 | 681.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 690.15 | 681.97 | 682.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 690.15 | 681.97 | 682.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 15:15:00 | 691.15 | 683.81 | 683.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 10:15:00 | 702.20 | 688.35 | 685.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 14:15:00 | 727.60 | 728.07 | 715.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 15:00:00 | 727.60 | 728.07 | 715.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 712.90 | 724.71 | 715.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 709.60 | 724.71 | 715.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 711.10 | 721.99 | 715.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:45:00 | 714.85 | 720.32 | 715.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 12:15:00 | 719.80 | 720.32 | 715.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 14:15:00 | 692.95 | 712.92 | 713.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 692.95 | 712.92 | 713.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 674.90 | 702.13 | 707.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 685.95 | 669.17 | 683.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 685.95 | 669.17 | 683.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 685.95 | 669.17 | 683.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:30:00 | 686.00 | 669.17 | 683.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 692.20 | 673.77 | 684.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 692.20 | 673.77 | 684.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 718.65 | 682.75 | 687.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 718.65 | 682.75 | 687.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 733.40 | 698.30 | 694.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 745.75 | 707.79 | 699.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 733.70 | 734.99 | 723.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:00:00 | 733.70 | 734.99 | 723.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 728.05 | 735.64 | 728.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 742.05 | 735.64 | 728.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 14:15:00 | 724.20 | 733.92 | 731.04 | SL hit (close<static) qty=1.00 sl=727.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 716.15 | 728.07 | 728.85 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 735.00 | 728.44 | 727.77 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 722.50 | 727.22 | 727.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 708.40 | 721.43 | 724.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 702.85 | 702.76 | 711.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:30:00 | 706.40 | 702.76 | 711.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 706.55 | 703.52 | 710.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 706.55 | 703.52 | 710.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 705.65 | 703.95 | 710.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 711.20 | 703.95 | 710.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 726.35 | 708.43 | 711.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 726.35 | 708.43 | 711.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 722.80 | 711.30 | 712.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 728.10 | 711.30 | 712.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 710.45 | 713.04 | 713.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:45:00 | 713.85 | 713.04 | 713.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 11:15:00 | 726.00 | 715.63 | 714.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 12:15:00 | 735.00 | 719.51 | 716.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 706.80 | 719.01 | 717.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 706.80 | 719.01 | 717.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 706.80 | 719.01 | 717.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 706.80 | 719.01 | 717.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 697.30 | 714.66 | 715.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 678.50 | 707.43 | 712.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 717.65 | 701.26 | 706.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 717.65 | 701.26 | 706.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 717.65 | 701.26 | 706.60 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 11:15:00 | 754.70 | 718.39 | 713.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 11:15:00 | 783.80 | 753.91 | 736.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 723.00 | 754.61 | 744.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 723.00 | 754.61 | 744.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 723.00 | 754.61 | 744.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 723.00 | 754.61 | 744.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 735.45 | 750.78 | 743.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 728.55 | 750.78 | 743.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 752.00 | 750.82 | 745.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 12:30:00 | 741.75 | 750.82 | 745.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 741.55 | 748.96 | 744.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 13:45:00 | 738.65 | 748.96 | 744.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 743.30 | 747.83 | 744.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 09:45:00 | 754.45 | 747.22 | 744.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 14:15:00 | 736.40 | 743.17 | 743.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 736.40 | 743.17 | 743.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 15:15:00 | 730.00 | 740.53 | 742.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 11:15:00 | 738.90 | 738.68 | 741.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 11:15:00 | 738.90 | 738.68 | 741.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 738.90 | 738.68 | 741.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:00:00 | 738.90 | 738.68 | 741.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 742.35 | 738.90 | 740.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:45:00 | 747.50 | 738.90 | 740.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 745.20 | 740.16 | 740.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 754.40 | 740.16 | 740.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 761.00 | 744.33 | 742.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 10:15:00 | 775.80 | 757.20 | 750.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 13:15:00 | 753.80 | 758.70 | 753.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 14:00:00 | 753.80 | 758.70 | 753.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 759.10 | 758.78 | 753.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:45:00 | 753.05 | 758.78 | 753.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 753.90 | 758.87 | 755.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 750.65 | 758.87 | 755.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 753.05 | 757.71 | 755.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 753.05 | 757.71 | 755.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 754.20 | 757.01 | 755.30 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 750.70 | 754.55 | 754.60 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 757.00 | 755.04 | 754.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 10:15:00 | 765.00 | 757.03 | 755.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 15:15:00 | 760.60 | 760.61 | 758.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:15:00 | 765.75 | 760.61 | 758.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 09:15:00 | 804.04 | 787.59 | 775.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-07 15:15:00 | 794.80 | 794.89 | 785.19 | SL hit (close<ema200) qty=0.50 sl=794.89 alert=retest1 |

### Cycle 112 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 794.95 | 813.77 | 815.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 788.50 | 808.71 | 812.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 791.00 | 773.44 | 786.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 791.00 | 773.44 | 786.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 791.00 | 773.44 | 786.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 791.00 | 773.44 | 786.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 781.10 | 774.97 | 786.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 792.80 | 774.97 | 786.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 788.35 | 777.65 | 786.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:45:00 | 785.60 | 777.65 | 786.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 781.70 | 778.46 | 785.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 774.15 | 776.88 | 784.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:30:00 | 776.90 | 774.91 | 780.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 814.25 | 786.35 | 784.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 814.25 | 786.35 | 784.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 11:15:00 | 828.20 | 815.68 | 804.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 13:15:00 | 883.60 | 886.45 | 866.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 14:00:00 | 883.60 | 886.45 | 866.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 884.95 | 893.03 | 881.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 884.95 | 893.03 | 881.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 924.95 | 928.91 | 914.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:45:00 | 924.40 | 928.91 | 914.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 923.85 | 929.16 | 919.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 921.70 | 929.16 | 919.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 905.75 | 924.48 | 918.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 905.75 | 924.48 | 918.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 907.70 | 921.12 | 917.07 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 15:15:00 | 900.45 | 914.07 | 914.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 09:15:00 | 894.20 | 910.10 | 912.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 896.65 | 895.99 | 903.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 896.65 | 895.99 | 903.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 894.90 | 895.73 | 902.11 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 919.90 | 904.94 | 904.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 11:15:00 | 937.70 | 914.00 | 908.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 946.20 | 950.68 | 938.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 09:45:00 | 942.00 | 950.68 | 938.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 962.80 | 970.95 | 964.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 962.80 | 970.95 | 964.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 967.30 | 970.22 | 964.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:30:00 | 964.50 | 970.22 | 964.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 966.95 | 970.55 | 966.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 966.95 | 970.55 | 966.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 968.00 | 970.04 | 966.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:15:00 | 971.20 | 970.04 | 966.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 958.60 | 967.75 | 965.90 | SL hit (close<static) qty=1.00 sl=965.35 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 11:15:00 | 956.80 | 964.16 | 964.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 12:15:00 | 953.70 | 962.06 | 963.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 09:15:00 | 960.65 | 959.42 | 961.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 960.65 | 959.42 | 961.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 960.65 | 959.42 | 961.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 960.65 | 959.42 | 961.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 962.80 | 960.10 | 961.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 966.35 | 960.10 | 961.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 953.35 | 958.75 | 960.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:30:00 | 949.15 | 955.87 | 958.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 13:15:00 | 977.20 | 959.98 | 959.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 977.20 | 959.98 | 959.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 982.50 | 964.49 | 961.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 1014.35 | 1016.67 | 1000.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 15:00:00 | 1014.35 | 1016.67 | 1000.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 1013.70 | 1014.95 | 1005.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:45:00 | 1003.15 | 1014.95 | 1005.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 1009.00 | 1013.19 | 1007.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:15:00 | 990.50 | 1013.19 | 1007.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1004.80 | 1011.51 | 1006.84 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 15:15:00 | 994.90 | 1003.96 | 1004.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 983.15 | 999.80 | 1002.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 973.20 | 969.02 | 982.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:00:00 | 973.20 | 969.02 | 982.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 975.00 | 967.70 | 974.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:00:00 | 975.00 | 967.70 | 974.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 975.00 | 969.16 | 974.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 975.00 | 969.16 | 974.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 991.50 | 973.63 | 975.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:00:00 | 991.50 | 973.63 | 975.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 991.40 | 977.18 | 977.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:15:00 | 985.95 | 977.18 | 977.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 983.20 | 978.39 | 977.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 983.20 | 978.39 | 977.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 1003.25 | 986.61 | 982.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 10:15:00 | 988.75 | 989.34 | 984.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 11:00:00 | 988.75 | 989.34 | 984.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 982.45 | 1010.46 | 1008.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 987.30 | 1010.46 | 1008.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 10:15:00 | 994.00 | 1007.17 | 1007.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 09:15:00 | 973.60 | 990.62 | 997.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 991.40 | 983.93 | 989.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 991.40 | 983.93 | 989.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 991.40 | 983.93 | 989.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:30:00 | 991.50 | 983.93 | 989.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 995.00 | 986.14 | 990.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 995.00 | 986.14 | 990.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 984.00 | 989.77 | 991.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 13:30:00 | 974.70 | 984.89 | 988.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 15:00:00 | 966.40 | 981.19 | 986.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 12:00:00 | 976.70 | 980.00 | 983.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 957.90 | 981.75 | 983.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 945.70 | 974.54 | 980.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 939.85 | 974.54 | 980.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:15:00 | 925.97 | 958.86 | 971.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:15:00 | 927.87 | 958.86 | 971.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 12:15:00 | 918.08 | 949.74 | 966.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 13:15:00 | 910.00 | 943.65 | 962.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:15:00 | 892.86 | 916.38 | 939.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-10 09:15:00 | 877.23 | 897.35 | 921.12 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 15:15:00 | 855.00 | 839.66 | 837.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 911.70 | 854.07 | 844.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 887.60 | 888.87 | 872.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 887.60 | 888.87 | 872.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 909.15 | 896.81 | 884.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:30:00 | 905.05 | 896.81 | 884.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 848.90 | 888.68 | 886.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 848.90 | 888.68 | 886.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 832.70 | 877.49 | 881.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 825.10 | 850.10 | 864.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 841.70 | 838.65 | 853.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 841.70 | 838.65 | 853.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 856.35 | 843.20 | 852.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 859.10 | 843.20 | 852.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 855.75 | 845.71 | 853.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 859.50 | 845.71 | 853.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 855.80 | 849.18 | 853.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 855.60 | 849.18 | 853.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 851.40 | 849.62 | 853.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:30:00 | 853.85 | 849.62 | 853.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 787.70 | 836.93 | 846.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 769.95 | 818.82 | 832.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 11:30:00 | 770.75 | 797.62 | 818.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:30:00 | 770.15 | 784.28 | 803.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:00:00 | 770.00 | 778.99 | 792.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 797.00 | 780.80 | 791.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 797.00 | 780.80 | 791.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 788.90 | 782.42 | 791.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-29 15:15:00 | 807.95 | 795.02 | 794.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 807.95 | 795.02 | 794.41 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 11:15:00 | 786.80 | 793.40 | 793.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 768.15 | 787.63 | 791.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 09:15:00 | 782.50 | 782.01 | 787.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:45:00 | 782.95 | 782.01 | 787.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 789.65 | 783.54 | 787.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:30:00 | 788.85 | 783.54 | 787.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 784.00 | 783.63 | 787.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 12:15:00 | 782.50 | 783.63 | 787.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 13:45:00 | 775.95 | 782.40 | 786.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 09:30:00 | 781.70 | 780.72 | 784.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 10:00:00 | 782.30 | 780.72 | 784.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 14:15:00 | 743.38 | 763.68 | 773.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 14:15:00 | 737.15 | 763.68 | 773.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 14:15:00 | 742.62 | 763.68 | 773.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 14:15:00 | 743.18 | 763.68 | 773.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 776.50 | 761.57 | 769.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 776.50 | 761.57 | 769.78 | SL hit (close>ema200) qty=0.50 sl=761.57 alert=retest2 |

### Cycle 125 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 785.00 | 772.81 | 772.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 807.50 | 783.79 | 778.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 12:15:00 | 803.95 | 804.44 | 795.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:00:00 | 803.95 | 804.44 | 795.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 797.95 | 802.96 | 796.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:15:00 | 797.60 | 802.96 | 796.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 797.60 | 801.89 | 796.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 801.95 | 801.89 | 796.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:00:00 | 809.35 | 803.38 | 797.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 778.95 | 803.18 | 801.73 | SL hit (close<static) qty=1.00 sl=795.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 779.20 | 798.39 | 799.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 769.10 | 792.53 | 796.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 754.10 | 749.19 | 761.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 754.10 | 749.19 | 761.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 759.50 | 750.07 | 757.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 759.50 | 750.07 | 757.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 770.70 | 754.19 | 758.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 771.95 | 754.19 | 758.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 756.60 | 756.89 | 759.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:45:00 | 758.35 | 756.89 | 759.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 742.25 | 753.94 | 757.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 736.40 | 753.94 | 757.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:00:00 | 734.20 | 730.96 | 741.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 11:00:00 | 735.90 | 731.95 | 740.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 13:15:00 | 746.15 | 732.17 | 731.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 746.15 | 732.17 | 731.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 747.65 | 735.26 | 732.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 743.05 | 757.83 | 751.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 743.05 | 757.83 | 751.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 743.05 | 757.83 | 751.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:00:00 | 743.05 | 757.83 | 751.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 745.20 | 755.30 | 751.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:30:00 | 741.15 | 755.30 | 751.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 753.00 | 754.54 | 752.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 757.00 | 754.54 | 752.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 771.60 | 757.95 | 753.99 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 742.55 | 751.94 | 752.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 730.60 | 742.79 | 747.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 729.80 | 715.59 | 721.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 729.80 | 715.59 | 721.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 729.80 | 715.59 | 721.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 729.80 | 715.59 | 721.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 726.60 | 717.79 | 722.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:00:00 | 716.90 | 720.52 | 722.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 12:15:00 | 681.05 | 699.90 | 704.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 693.40 | 677.08 | 685.33 | SL hit (close>ema200) qty=0.50 sl=677.08 alert=retest2 |

### Cycle 129 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 710.65 | 690.42 | 689.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 14:15:00 | 711.75 | 694.69 | 691.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 10:15:00 | 691.00 | 697.48 | 694.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 10:15:00 | 691.00 | 697.48 | 694.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 691.00 | 697.48 | 694.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 691.00 | 697.48 | 694.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 686.90 | 695.36 | 693.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 686.20 | 695.36 | 693.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 702.40 | 694.86 | 693.49 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 686.35 | 692.53 | 693.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 684.65 | 690.54 | 692.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 691.15 | 689.31 | 691.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 691.15 | 689.31 | 691.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 691.15 | 689.31 | 691.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 698.25 | 689.31 | 691.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 691.35 | 689.71 | 691.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 692.35 | 689.71 | 691.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 688.90 | 689.55 | 690.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 692.10 | 689.55 | 690.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 689.95 | 689.14 | 690.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 689.95 | 689.14 | 690.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 685.00 | 688.31 | 689.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 690.00 | 688.31 | 689.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 695.75 | 689.80 | 690.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 707.60 | 689.80 | 690.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 720.05 | 695.85 | 693.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 730.60 | 702.80 | 696.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 721.15 | 748.76 | 735.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 721.15 | 748.76 | 735.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 721.15 | 748.76 | 735.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 721.15 | 748.76 | 735.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 731.45 | 745.30 | 734.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 12:00:00 | 737.65 | 743.77 | 735.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 12:45:00 | 733.90 | 741.85 | 735.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:45:00 | 734.90 | 739.69 | 735.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 740.95 | 738.38 | 734.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 739.20 | 738.54 | 735.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:30:00 | 750.55 | 742.43 | 737.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:15:00 | 752.15 | 744.71 | 739.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 15:00:00 | 751.25 | 746.02 | 740.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:45:00 | 755.65 | 748.49 | 742.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 781.00 | 779.96 | 773.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 13:15:00 | 796.30 | 782.41 | 776.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-27 14:15:00 | 811.42 | 790.28 | 781.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 773.30 | 809.03 | 811.55 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 819.35 | 803.38 | 803.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 831.25 | 814.38 | 810.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 12:15:00 | 830.65 | 833.47 | 826.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 12:45:00 | 829.70 | 833.47 | 826.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 855.25 | 857.60 | 848.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 869.05 | 852.95 | 849.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 876.30 | 885.32 | 885.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 876.30 | 885.32 | 885.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 874.50 | 883.16 | 884.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 888.30 | 884.19 | 885.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 888.30 | 884.19 | 885.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 888.30 | 884.19 | 885.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 872.00 | 882.37 | 883.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:15:00 | 874.90 | 874.68 | 877.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:00:00 | 875.00 | 874.74 | 877.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:15:00 | 874.75 | 875.45 | 877.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 871.35 | 874.63 | 877.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 14:30:00 | 867.45 | 871.96 | 875.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:00:00 | 866.60 | 869.20 | 873.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 12:15:00 | 828.40 | 857.44 | 866.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 12:15:00 | 831.15 | 857.44 | 866.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 12:15:00 | 831.25 | 857.44 | 866.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 12:15:00 | 831.01 | 857.44 | 866.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 861.65 | 849.15 | 858.94 | SL hit (close>ema200) qty=0.50 sl=849.15 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 875.50 | 851.66 | 850.70 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 838.20 | 854.35 | 855.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 829.20 | 847.27 | 851.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 851.05 | 838.59 | 843.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 851.05 | 838.59 | 843.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 851.05 | 838.59 | 843.08 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 857.85 | 847.88 | 846.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 864.30 | 851.16 | 848.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 850.95 | 854.79 | 851.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 850.95 | 854.79 | 851.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 850.95 | 854.79 | 851.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 11:45:00 | 853.80 | 853.43 | 851.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 12:15:00 | 853.30 | 853.43 | 851.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:45:00 | 859.50 | 853.36 | 851.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 09:15:00 | 836.45 | 850.54 | 850.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 09:15:00 | 836.45 | 850.54 | 850.62 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 858.20 | 850.94 | 850.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 862.25 | 854.96 | 853.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 860.35 | 866.46 | 861.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 12:15:00 | 860.35 | 866.46 | 861.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 860.35 | 866.46 | 861.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 860.35 | 866.46 | 861.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 862.30 | 865.63 | 861.51 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 851.65 | 858.98 | 859.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 828.05 | 842.23 | 849.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 832.50 | 832.31 | 839.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 832.50 | 832.31 | 839.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 843.25 | 834.50 | 839.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 843.25 | 834.50 | 839.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 847.55 | 837.11 | 840.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 847.55 | 837.11 | 840.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 860.00 | 844.50 | 843.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 866.70 | 851.79 | 846.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 15:15:00 | 865.90 | 867.93 | 861.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:15:00 | 867.30 | 867.93 | 861.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 861.65 | 866.93 | 862.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 861.65 | 866.93 | 862.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 861.50 | 865.85 | 862.30 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 853.80 | 859.98 | 860.39 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 866.00 | 861.18 | 860.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 874.45 | 865.74 | 863.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 13:15:00 | 925.80 | 925.94 | 911.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:30:00 | 924.40 | 925.94 | 911.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 944.20 | 943.78 | 937.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 937.75 | 943.78 | 937.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 960.50 | 963.52 | 957.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 962.70 | 963.52 | 957.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 960.30 | 962.87 | 958.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 963.80 | 962.13 | 958.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 962.50 | 963.10 | 959.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 903.45 | 950.33 | 955.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 903.45 | 950.33 | 955.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 862.10 | 873.28 | 886.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 886.75 | 872.53 | 880.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 886.75 | 872.53 | 880.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 886.75 | 872.53 | 880.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 886.75 | 872.53 | 880.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 885.00 | 875.02 | 881.22 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 15:15:00 | 891.00 | 884.86 | 884.37 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 870.80 | 882.15 | 883.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 865.90 | 875.97 | 880.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 12:15:00 | 876.05 | 872.04 | 876.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 12:15:00 | 876.05 | 872.04 | 876.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 876.05 | 872.04 | 876.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 876.05 | 872.04 | 876.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 870.95 | 871.83 | 875.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 877.00 | 871.83 | 875.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 881.10 | 874.12 | 875.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 879.90 | 874.12 | 875.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 879.80 | 875.26 | 876.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:15:00 | 880.80 | 875.26 | 876.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 889.30 | 878.07 | 877.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 897.40 | 885.67 | 881.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 894.10 | 896.88 | 891.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:00:00 | 894.10 | 896.88 | 891.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 895.55 | 896.61 | 891.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:15:00 | 896.75 | 895.87 | 892.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:15:00 | 897.20 | 895.29 | 892.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 899.55 | 895.39 | 892.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 12:15:00 | 922.60 | 926.13 | 926.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 922.60 | 926.13 | 926.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 918.40 | 923.64 | 925.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 912.70 | 911.84 | 917.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 912.70 | 911.84 | 917.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 932.45 | 913.21 | 914.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 932.50 | 913.21 | 914.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 927.45 | 916.06 | 915.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 939.10 | 926.62 | 921.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 929.30 | 930.57 | 924.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 929.30 | 930.57 | 924.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 929.30 | 930.57 | 924.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 929.30 | 930.57 | 924.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 919.40 | 928.33 | 924.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 919.40 | 928.33 | 924.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 922.50 | 927.17 | 924.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 932.55 | 926.33 | 923.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-22 09:15:00 | 1025.81 | 1017.46 | 1008.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 1067.90 | 1081.73 | 1081.77 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1084.00 | 1076.24 | 1075.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 15:15:00 | 1091.00 | 1078.98 | 1076.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 13:15:00 | 1082.40 | 1086.29 | 1082.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 13:15:00 | 1082.40 | 1086.29 | 1082.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1082.40 | 1086.29 | 1082.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 1082.40 | 1086.29 | 1082.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1074.30 | 1083.89 | 1081.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1074.30 | 1083.89 | 1081.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1078.00 | 1082.71 | 1081.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1088.40 | 1082.71 | 1081.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 1088.10 | 1082.64 | 1081.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:45:00 | 1087.00 | 1084.21 | 1082.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 1068.10 | 1079.37 | 1080.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1068.10 | 1079.37 | 1080.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 14:15:00 | 1053.80 | 1066.53 | 1073.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 1061.80 | 1061.20 | 1068.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 12:00:00 | 1061.80 | 1061.20 | 1068.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1066.80 | 1062.32 | 1068.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 1065.20 | 1062.32 | 1068.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1062.50 | 1062.36 | 1067.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:30:00 | 1065.70 | 1062.36 | 1067.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1052.00 | 1058.32 | 1064.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1060.90 | 1058.32 | 1064.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1067.50 | 1058.80 | 1062.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 1067.50 | 1058.80 | 1062.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1066.90 | 1060.42 | 1062.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:30:00 | 1069.40 | 1060.42 | 1062.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1065.70 | 1062.06 | 1063.16 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 1067.70 | 1064.06 | 1063.93 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 1059.90 | 1063.38 | 1063.75 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 1070.30 | 1064.77 | 1064.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 1095.40 | 1072.04 | 1067.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1139.10 | 1147.76 | 1129.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1139.10 | 1147.76 | 1129.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1262.20 | 1270.63 | 1260.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1262.70 | 1270.63 | 1260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1257.80 | 1268.06 | 1260.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 1257.80 | 1268.06 | 1260.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1259.90 | 1266.43 | 1260.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 1254.40 | 1266.43 | 1260.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 1256.70 | 1262.18 | 1259.40 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 1244.80 | 1255.31 | 1256.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 1236.30 | 1251.51 | 1254.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 11:15:00 | 1226.70 | 1220.65 | 1229.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 12:00:00 | 1226.70 | 1220.65 | 1229.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1232.90 | 1223.10 | 1230.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 1231.80 | 1223.10 | 1230.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1233.10 | 1225.10 | 1230.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1229.50 | 1225.10 | 1230.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1235.00 | 1227.08 | 1230.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:15:00 | 1230.90 | 1227.08 | 1230.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1244.00 | 1231.08 | 1231.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1246.60 | 1231.08 | 1231.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1240.10 | 1232.88 | 1232.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 1254.40 | 1238.92 | 1235.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 1250.70 | 1269.78 | 1259.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 1250.70 | 1269.78 | 1259.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1250.70 | 1269.78 | 1259.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 1252.70 | 1269.78 | 1259.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1242.90 | 1264.40 | 1257.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 1240.00 | 1264.40 | 1257.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1238.30 | 1252.04 | 1253.42 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 1261.50 | 1253.98 | 1253.91 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 1250.20 | 1254.15 | 1254.30 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 1266.50 | 1256.23 | 1255.20 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 1224.00 | 1249.38 | 1252.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 1213.20 | 1224.76 | 1236.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 1225.20 | 1224.85 | 1235.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:00:00 | 1225.20 | 1224.85 | 1235.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1227.40 | 1225.93 | 1233.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:45:00 | 1227.40 | 1225.93 | 1233.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1233.70 | 1227.48 | 1233.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 1233.70 | 1227.48 | 1233.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1237.50 | 1229.48 | 1234.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 1237.50 | 1229.48 | 1234.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1236.70 | 1230.93 | 1234.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 1240.30 | 1230.93 | 1234.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1232.90 | 1231.77 | 1234.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 1231.00 | 1231.77 | 1234.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:15:00 | 1231.00 | 1231.63 | 1233.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 1227.10 | 1227.86 | 1230.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 1238.60 | 1227.81 | 1228.25 | SL hit (close>static) qty=1.00 sl=1237.70 alert=retest2 |

### Cycle 163 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 1228.80 | 1228.58 | 1228.56 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 14:15:00 | 1227.30 | 1228.33 | 1228.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 1223.80 | 1227.37 | 1227.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 14:15:00 | 1227.60 | 1225.27 | 1226.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 14:15:00 | 1227.60 | 1225.27 | 1226.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1227.60 | 1225.27 | 1226.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 1227.60 | 1225.27 | 1226.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1226.50 | 1225.52 | 1226.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1233.30 | 1225.52 | 1226.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1224.50 | 1225.32 | 1226.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 1219.30 | 1222.93 | 1224.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 1229.30 | 1226.07 | 1225.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1229.30 | 1226.07 | 1225.64 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 1223.20 | 1225.58 | 1225.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 1217.50 | 1223.97 | 1224.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 1199.00 | 1195.33 | 1207.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 1199.00 | 1195.33 | 1207.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1194.20 | 1196.61 | 1203.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 12:45:00 | 1187.10 | 1192.82 | 1200.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:30:00 | 1187.50 | 1191.72 | 1198.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:00:00 | 1187.30 | 1191.72 | 1198.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1128.12 | 1150.74 | 1163.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1127.74 | 1140.64 | 1153.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1127.93 | 1140.64 | 1153.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1123.40 | 1117.26 | 1126.86 | SL hit (close>ema200) qty=0.50 sl=1117.26 alert=retest2 |

### Cycle 167 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1146.80 | 1131.99 | 1130.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 1149.00 | 1137.76 | 1133.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1233.00 | 1233.89 | 1214.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 1233.00 | 1233.89 | 1214.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1235.00 | 1239.33 | 1231.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 1235.00 | 1239.33 | 1231.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1237.00 | 1240.15 | 1235.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1240.60 | 1240.15 | 1235.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1243.20 | 1240.76 | 1235.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:30:00 | 1247.70 | 1242.07 | 1236.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 1247.80 | 1243.22 | 1237.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:45:00 | 1246.30 | 1244.33 | 1239.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 1229.50 | 1242.41 | 1239.84 | SL hit (close<static) qty=1.00 sl=1231.10 alert=retest2 |

### Cycle 168 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1288.50 | 1291.80 | 1292.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 15:15:00 | 1285.20 | 1289.34 | 1290.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 1291.20 | 1289.71 | 1290.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 1291.20 | 1289.71 | 1290.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1291.20 | 1289.71 | 1290.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:15:00 | 1299.90 | 1289.71 | 1290.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1293.30 | 1290.43 | 1291.07 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 1303.90 | 1293.20 | 1292.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 1307.20 | 1296.00 | 1293.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1303.00 | 1307.86 | 1303.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1303.00 | 1307.86 | 1303.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1303.00 | 1307.86 | 1303.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1303.00 | 1307.86 | 1303.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1295.50 | 1305.39 | 1302.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 1296.50 | 1305.39 | 1302.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1311.50 | 1306.61 | 1303.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1318.50 | 1308.75 | 1307.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 15:15:00 | 1302.70 | 1307.14 | 1307.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1302.70 | 1307.14 | 1307.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1292.50 | 1304.22 | 1306.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 09:15:00 | 1311.10 | 1281.16 | 1285.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1311.10 | 1281.16 | 1285.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1311.10 | 1281.16 | 1285.78 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 1322.50 | 1289.43 | 1289.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 09:15:00 | 1344.20 | 1317.76 | 1305.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 11:15:00 | 1338.70 | 1339.62 | 1327.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 12:00:00 | 1338.70 | 1339.62 | 1327.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1329.70 | 1337.63 | 1327.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:45:00 | 1328.30 | 1337.63 | 1327.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 1334.30 | 1336.97 | 1328.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 14:15:00 | 1336.00 | 1336.97 | 1328.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1338.00 | 1334.46 | 1328.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1324.40 | 1332.45 | 1328.18 | SL hit (close<static) qty=1.00 sl=1325.40 alert=retest2 |

### Cycle 172 — SELL (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 10:15:00 | 1321.90 | 1326.18 | 1326.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 12:15:00 | 1316.20 | 1323.09 | 1324.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 12:15:00 | 1311.10 | 1310.03 | 1315.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 1311.10 | 1310.03 | 1315.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1311.10 | 1310.03 | 1315.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 1304.70 | 1309.17 | 1314.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 1304.80 | 1308.40 | 1313.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 1323.60 | 1306.51 | 1308.97 | SL hit (close>static) qty=1.00 sl=1316.80 alert=retest2 |

### Cycle 173 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 1327.10 | 1310.62 | 1310.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 1329.00 | 1314.30 | 1312.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1307.70 | 1320.63 | 1317.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1307.70 | 1320.63 | 1317.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1307.70 | 1320.63 | 1317.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:15:00 | 1305.10 | 1320.63 | 1317.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1304.70 | 1317.44 | 1315.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1304.50 | 1317.44 | 1315.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 1300.40 | 1314.03 | 1314.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1296.60 | 1308.12 | 1311.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 12:15:00 | 1289.00 | 1288.98 | 1295.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 12:30:00 | 1288.70 | 1288.98 | 1295.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1272.80 | 1283.77 | 1290.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 1268.00 | 1283.77 | 1290.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:30:00 | 1268.60 | 1273.65 | 1282.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:30:00 | 1269.40 | 1272.64 | 1281.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:30:00 | 1266.00 | 1272.05 | 1279.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1275.40 | 1272.72 | 1279.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 1276.00 | 1272.72 | 1279.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1261.00 | 1264.71 | 1271.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:30:00 | 1243.00 | 1257.46 | 1267.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 1274.00 | 1254.85 | 1260.10 | SL hit (close>static) qty=1.00 sl=1272.60 alert=retest2 |

### Cycle 175 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1275.90 | 1263.95 | 1263.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 1288.00 | 1274.50 | 1269.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 14:15:00 | 1364.40 | 1364.80 | 1347.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:45:00 | 1361.10 | 1364.80 | 1347.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1332.60 | 1358.39 | 1347.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 1332.60 | 1358.39 | 1347.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1325.30 | 1351.77 | 1345.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:45:00 | 1324.50 | 1351.77 | 1345.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 1326.80 | 1339.60 | 1341.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 1323.70 | 1332.99 | 1337.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 1334.80 | 1331.17 | 1334.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1334.80 | 1331.17 | 1334.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1334.80 | 1331.17 | 1334.69 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 1348.70 | 1338.36 | 1337.22 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1324.90 | 1337.27 | 1337.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1309.00 | 1331.62 | 1335.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 1318.70 | 1315.26 | 1322.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 1318.70 | 1315.26 | 1322.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1287.90 | 1281.95 | 1289.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 1286.50 | 1281.95 | 1289.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1286.00 | 1282.76 | 1289.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 1287.60 | 1282.76 | 1289.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1286.30 | 1283.47 | 1289.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 1293.00 | 1283.47 | 1289.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1292.30 | 1285.24 | 1289.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:00:00 | 1292.30 | 1285.24 | 1289.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1304.30 | 1289.05 | 1290.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 1304.30 | 1289.05 | 1290.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1303.30 | 1291.90 | 1291.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:15:00 | 1312.20 | 1291.90 | 1291.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1312.20 | 1295.96 | 1293.78 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 1285.00 | 1295.79 | 1296.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1282.20 | 1290.01 | 1293.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1271.10 | 1271.06 | 1279.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:45:00 | 1270.70 | 1271.06 | 1279.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1284.20 | 1273.69 | 1279.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 1286.00 | 1273.69 | 1279.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1285.00 | 1275.95 | 1280.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1285.00 | 1275.95 | 1280.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1283.10 | 1277.38 | 1280.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 1282.40 | 1277.38 | 1280.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 1306.00 | 1285.66 | 1283.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1306.00 | 1285.66 | 1283.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1321.50 | 1300.86 | 1292.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1329.90 | 1331.10 | 1316.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 14:45:00 | 1329.50 | 1331.10 | 1316.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1325.60 | 1335.32 | 1331.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1325.60 | 1335.32 | 1331.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1322.50 | 1332.76 | 1330.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1333.70 | 1332.76 | 1330.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1324.70 | 1329.97 | 1329.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1324.70 | 1329.97 | 1329.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1327.00 | 1329.38 | 1329.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:30:00 | 1324.20 | 1329.38 | 1329.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1324.20 | 1328.34 | 1328.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1315.80 | 1325.01 | 1327.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 10:15:00 | 1324.80 | 1323.52 | 1325.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 10:15:00 | 1324.80 | 1323.52 | 1325.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1324.80 | 1323.52 | 1325.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1325.00 | 1323.52 | 1325.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1318.90 | 1322.60 | 1325.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:45:00 | 1324.80 | 1322.60 | 1325.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1310.80 | 1303.15 | 1308.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1310.80 | 1303.15 | 1308.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1302.50 | 1303.02 | 1308.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:00:00 | 1297.10 | 1302.07 | 1306.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 1297.00 | 1301.66 | 1306.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 1297.70 | 1299.48 | 1304.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 1292.20 | 1299.92 | 1303.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1312.90 | 1300.32 | 1302.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1312.90 | 1300.32 | 1302.18 | SL hit (close>static) qty=1.00 sl=1312.00 alert=retest2 |

### Cycle 183 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1316.30 | 1305.29 | 1304.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1327.30 | 1309.69 | 1306.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1332.30 | 1336.22 | 1325.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 1332.30 | 1336.22 | 1325.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1328.60 | 1335.21 | 1327.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1326.30 | 1335.21 | 1327.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1327.00 | 1333.57 | 1327.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 1332.80 | 1333.06 | 1327.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:45:00 | 1334.50 | 1333.25 | 1328.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 1333.70 | 1331.82 | 1328.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 1333.30 | 1331.24 | 1328.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1320.50 | 1329.09 | 1328.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 1320.50 | 1329.09 | 1328.18 | SL hit (close<static) qty=1.00 sl=1323.00 alert=retest2 |

### Cycle 184 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 1319.60 | 1327.19 | 1327.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 1310.20 | 1319.96 | 1323.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1285.90 | 1274.79 | 1286.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1285.90 | 1274.79 | 1286.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1285.90 | 1274.79 | 1286.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1272.70 | 1276.73 | 1283.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 1304.70 | 1283.79 | 1285.53 | SL hit (close>static) qty=1.00 sl=1298.50 alert=retest2 |

### Cycle 185 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1315.00 | 1290.03 | 1288.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1319.60 | 1299.92 | 1293.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1326.70 | 1333.22 | 1320.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:00:00 | 1326.70 | 1333.22 | 1320.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1325.70 | 1331.72 | 1320.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 15:00:00 | 1334.40 | 1325.76 | 1320.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 1317.90 | 1324.67 | 1321.58 | SL hit (close<static) qty=1.00 sl=1318.10 alert=retest2 |

### Cycle 186 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1303.10 | 1318.43 | 1319.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1290.90 | 1312.92 | 1316.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1263.60 | 1255.18 | 1276.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 1263.60 | 1255.18 | 1276.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1293.10 | 1264.47 | 1270.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 1298.90 | 1264.47 | 1270.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1251.90 | 1261.95 | 1268.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 1278.60 | 1261.95 | 1268.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1177.20 | 1165.78 | 1177.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 1177.20 | 1165.78 | 1177.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1178.50 | 1168.32 | 1177.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 1179.10 | 1168.32 | 1177.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1175.90 | 1169.84 | 1177.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 1182.80 | 1169.84 | 1177.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1165.40 | 1168.95 | 1176.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:00:00 | 1160.00 | 1167.16 | 1174.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1124.30 | 1167.12 | 1172.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-01 12:15:00 | 1044.00 | 1151.53 | 1154.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 1181.50 | 1157.52 | 1156.56 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 1176.90 | 1191.90 | 1193.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 09:15:00 | 1162.60 | 1179.15 | 1184.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 11:15:00 | 1166.90 | 1161.94 | 1170.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:00:00 | 1166.90 | 1161.94 | 1170.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1143.50 | 1127.27 | 1132.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 1143.50 | 1127.27 | 1132.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1150.00 | 1131.81 | 1133.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1154.90 | 1131.81 | 1133.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1150.10 | 1135.47 | 1135.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 1171.80 | 1147.37 | 1141.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1163.20 | 1182.51 | 1168.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1163.20 | 1182.51 | 1168.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1163.20 | 1182.51 | 1168.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 1163.20 | 1182.51 | 1168.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1160.60 | 1178.13 | 1167.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 1160.60 | 1178.13 | 1167.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1163.90 | 1172.22 | 1166.74 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 1143.00 | 1160.83 | 1162.47 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1172.00 | 1159.81 | 1159.78 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 1141.30 | 1158.11 | 1159.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 1136.50 | 1153.79 | 1157.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1140.50 | 1140.19 | 1148.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:15:00 | 1142.00 | 1140.19 | 1148.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1140.70 | 1140.29 | 1147.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 1136.50 | 1140.29 | 1147.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1136.30 | 1134.31 | 1140.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1079.67 | 1101.71 | 1115.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1079.48 | 1101.71 | 1115.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 1049.10 | 1048.79 | 1070.64 | SL hit (close>ema200) qty=0.50 sl=1048.79 alert=retest2 |

### Cycle 193 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 1020.30 | 1001.31 | 1001.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1043.90 | 1020.44 | 1012.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1052.70 | 1057.71 | 1039.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1052.70 | 1057.71 | 1039.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1052.70 | 1057.71 | 1039.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 1057.00 | 1057.11 | 1040.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 1057.50 | 1057.11 | 1040.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 1056.40 | 1056.57 | 1043.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1055.10 | 1050.01 | 1043.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1066.90 | 1052.08 | 1045.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 1001.80 | 1043.89 | 1045.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1001.80 | 1043.89 | 1045.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 993.70 | 1033.85 | 1040.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1009.90 | 1007.47 | 1021.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 1022.30 | 1011.02 | 1020.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1022.30 | 1011.02 | 1020.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 1022.30 | 1011.02 | 1020.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1031.10 | 1015.04 | 1021.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1035.80 | 1015.04 | 1021.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1034.20 | 1018.87 | 1022.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1034.20 | 1018.87 | 1022.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1051.50 | 1029.94 | 1027.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1056.80 | 1035.31 | 1029.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1033.50 | 1050.33 | 1041.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1033.50 | 1050.33 | 1041.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1033.50 | 1050.33 | 1041.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1033.50 | 1050.33 | 1041.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1021.00 | 1044.46 | 1039.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1021.00 | 1044.46 | 1039.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1019.50 | 1034.92 | 1036.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1010.40 | 1027.23 | 1032.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 993.60 | 982.40 | 1000.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 993.60 | 982.40 | 1000.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 993.60 | 982.40 | 1000.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 972.50 | 995.01 | 1000.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 1007.55 | 991.43 | 994.51 | SL hit (close>static) qty=1.00 sl=1001.85 alert=retest2 |

### Cycle 197 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1019.00 | 996.85 | 995.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1025.00 | 1002.48 | 998.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1095.85 | 1105.28 | 1079.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 1097.20 | 1105.28 | 1079.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1096.25 | 1111.53 | 1099.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 1101.80 | 1110.17 | 1099.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1102.70 | 1107.46 | 1100.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 1103.70 | 1107.46 | 1100.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1148.90 | 1162.10 | 1162.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1148.90 | 1162.10 | 1162.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 1146.90 | 1159.06 | 1160.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 14:15:00 | 1157.40 | 1155.09 | 1157.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 1157.40 | 1155.09 | 1157.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1157.40 | 1155.09 | 1157.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 1157.40 | 1155.09 | 1157.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 1164.00 | 1156.87 | 1158.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1162.30 | 1156.87 | 1158.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1158.85 | 1157.27 | 1158.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 1161.95 | 1157.27 | 1158.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1148.45 | 1155.50 | 1157.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:30:00 | 1146.15 | 1154.59 | 1157.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 1146.10 | 1152.18 | 1155.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 1147.35 | 1152.18 | 1155.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 15:00:00 | 1146.10 | 1150.96 | 1154.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1153.00 | 1151.37 | 1154.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1061.60 | 1151.37 | 1154.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 1088.84 | 1142.12 | 1149.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 1088.79 | 1142.12 | 1149.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 1089.98 | 1142.12 | 1149.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 1088.79 | 1142.12 | 1149.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 1140.70 | 1134.62 | 1143.90 | SL hit (close>ema200) qty=0.50 sl=1134.62 alert=retest2 |

### Cycle 199 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1108.40 | 1106.63 | 1106.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1173.20 | 1120.48 | 1112.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1186.60 | 1191.98 | 1168.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 1186.60 | 1191.98 | 1168.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-18 14:45:00 | 387.75 | 2024-04-24 11:15:00 | 386.05 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-04-19 09:15:00 | 383.50 | 2024-04-24 11:15:00 | 386.05 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-04-30 11:15:00 | 372.80 | 2024-05-02 12:15:00 | 380.55 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-05-02 14:15:00 | 373.25 | 2024-05-06 09:15:00 | 354.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 09:30:00 | 372.70 | 2024-05-06 09:15:00 | 354.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 14:15:00 | 373.25 | 2024-05-07 09:15:00 | 335.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-03 09:30:00 | 372.70 | 2024-05-07 09:15:00 | 335.43 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-05-22 13:30:00 | 358.85 | 2024-05-24 09:15:00 | 339.40 | STOP_HIT | 1.00 | -5.42% |
| BUY | retest2 | 2024-06-03 09:15:00 | 378.75 | 2024-06-04 09:15:00 | 357.40 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest2 | 2024-06-12 09:15:00 | 384.05 | 2024-06-13 09:15:00 | 422.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 14:00:00 | 382.80 | 2024-06-13 09:15:00 | 421.08 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-28 15:15:00 | 401.95 | 2024-07-01 09:15:00 | 419.50 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2024-07-10 09:15:00 | 471.00 | 2024-07-16 09:15:00 | 462.05 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-07-10 11:30:00 | 469.25 | 2024-07-16 09:15:00 | 462.05 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-07-10 14:30:00 | 470.40 | 2024-07-16 09:15:00 | 462.05 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-07-11 09:15:00 | 473.30 | 2024-07-16 09:15:00 | 462.05 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-07-15 09:15:00 | 469.75 | 2024-07-16 09:15:00 | 462.05 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-07-15 10:45:00 | 466.90 | 2024-07-16 09:15:00 | 462.05 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-07-24 09:30:00 | 453.60 | 2024-07-24 13:15:00 | 461.70 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-08-01 09:15:00 | 503.10 | 2024-08-05 12:15:00 | 482.00 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-08-01 14:30:00 | 496.30 | 2024-08-05 12:15:00 | 482.00 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-08-01 15:15:00 | 498.00 | 2024-08-05 12:15:00 | 482.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-08-09 09:15:00 | 514.15 | 2024-08-12 09:15:00 | 504.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-08-13 11:15:00 | 512.25 | 2024-08-13 14:15:00 | 505.45 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-08-13 12:15:00 | 512.00 | 2024-08-13 14:15:00 | 505.45 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-09-04 14:15:00 | 613.15 | 2024-09-09 09:15:00 | 595.40 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-09-12 13:15:00 | 665.10 | 2024-09-13 14:15:00 | 655.15 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-09-13 14:15:00 | 663.85 | 2024-09-13 14:15:00 | 655.15 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-09-16 09:15:00 | 665.55 | 2024-09-19 11:15:00 | 657.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-09-16 09:45:00 | 670.25 | 2024-09-19 11:15:00 | 657.80 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-09-17 15:15:00 | 667.00 | 2024-09-19 11:15:00 | 657.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-10-04 11:45:00 | 714.85 | 2024-10-04 14:15:00 | 692.95 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-10-04 12:15:00 | 719.80 | 2024-10-04 14:15:00 | 692.95 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-10-11 09:15:00 | 742.05 | 2024-10-11 14:15:00 | 724.20 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-10-28 09:45:00 | 754.45 | 2024-10-28 14:15:00 | 736.40 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest1 | 2024-11-06 09:15:00 | 765.75 | 2024-11-07 09:15:00 | 804.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-11-06 09:15:00 | 765.75 | 2024-11-07 15:15:00 | 794.80 | STOP_HIT | 0.50 | 3.79% |
| BUY | retest2 | 2024-11-08 12:45:00 | 836.55 | 2024-11-12 13:15:00 | 794.95 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest2 | 2024-11-11 10:00:00 | 833.90 | 2024-11-12 13:15:00 | 794.95 | STOP_HIT | 1.00 | -4.67% |
| BUY | retest2 | 2024-11-11 13:00:00 | 835.55 | 2024-11-12 13:15:00 | 794.95 | STOP_HIT | 1.00 | -4.86% |
| SELL | retest2 | 2024-11-14 13:30:00 | 774.15 | 2024-11-19 09:15:00 | 814.25 | STOP_HIT | 1.00 | -5.18% |
| SELL | retest2 | 2024-11-18 11:30:00 | 776.90 | 2024-11-19 09:15:00 | 814.25 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2024-12-11 09:15:00 | 971.20 | 2024-12-11 09:15:00 | 958.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-12-13 09:30:00 | 949.15 | 2024-12-13 13:15:00 | 977.20 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-12-24 14:15:00 | 985.95 | 2024-12-24 14:15:00 | 983.20 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-01-06 13:30:00 | 974.70 | 2025-01-08 11:15:00 | 925.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 15:00:00 | 966.40 | 2025-01-08 11:15:00 | 927.87 | PARTIAL | 0.50 | 3.99% |
| SELL | retest2 | 2025-01-07 12:00:00 | 976.70 | 2025-01-08 12:15:00 | 918.08 | PARTIAL | 0.50 | 6.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 957.90 | 2025-01-08 13:15:00 | 910.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:15:00 | 939.85 | 2025-01-09 11:15:00 | 892.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 13:30:00 | 974.70 | 2025-01-10 09:15:00 | 877.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 15:00:00 | 966.40 | 2025-01-10 09:15:00 | 869.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-07 12:00:00 | 976.70 | 2025-01-10 09:15:00 | 879.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 957.90 | 2025-01-10 09:15:00 | 862.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 10:15:00 | 939.85 | 2025-01-10 12:15:00 | 845.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 769.95 | 2025-01-29 15:15:00 | 807.95 | STOP_HIT | 1.00 | -4.94% |
| SELL | retest2 | 2025-01-27 11:30:00 | 770.75 | 2025-01-29 15:15:00 | 807.95 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest2 | 2025-01-28 09:30:00 | 770.15 | 2025-01-29 15:15:00 | 807.95 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2025-01-28 15:00:00 | 770.00 | 2025-01-29 15:15:00 | 807.95 | STOP_HIT | 1.00 | -4.93% |
| SELL | retest2 | 2025-01-31 12:15:00 | 782.50 | 2025-02-01 14:15:00 | 743.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 13:45:00 | 775.95 | 2025-02-01 14:15:00 | 737.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 09:30:00 | 781.70 | 2025-02-01 14:15:00 | 742.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 10:00:00 | 782.30 | 2025-02-01 14:15:00 | 743.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-31 12:15:00 | 782.50 | 2025-02-03 10:15:00 | 776.50 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2025-01-31 13:45:00 | 775.95 | 2025-02-03 10:15:00 | 776.50 | STOP_HIT | 0.50 | -0.07% |
| SELL | retest2 | 2025-02-01 09:30:00 | 781.70 | 2025-02-03 10:15:00 | 776.50 | STOP_HIT | 0.50 | 0.67% |
| SELL | retest2 | 2025-02-01 10:00:00 | 782.30 | 2025-02-03 10:15:00 | 776.50 | STOP_HIT | 0.50 | 0.74% |
| SELL | retest2 | 2025-02-03 14:30:00 | 770.85 | 2025-02-04 09:15:00 | 785.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-02-07 09:15:00 | 801.95 | 2025-02-10 09:15:00 | 778.95 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-02-07 10:00:00 | 809.35 | 2025-02-10 09:15:00 | 778.95 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-02-14 10:15:00 | 736.40 | 2025-02-19 13:15:00 | 746.15 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-02-17 10:00:00 | 734.20 | 2025-02-19 13:15:00 | 746.15 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-02-17 11:00:00 | 735.90 | 2025-02-19 13:15:00 | 746.15 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-03-04 10:00:00 | 716.90 | 2025-03-07 12:15:00 | 681.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-04 10:00:00 | 716.90 | 2025-03-11 09:15:00 | 693.40 | STOP_HIT | 0.50 | 3.28% |
| BUY | retest2 | 2025-03-20 12:00:00 | 737.65 | 2025-03-27 14:15:00 | 811.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 12:45:00 | 733.90 | 2025-03-27 14:15:00 | 807.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 14:45:00 | 734.90 | 2025-03-27 14:15:00 | 808.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 09:15:00 | 740.95 | 2025-03-28 09:15:00 | 815.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 11:30:00 | 750.55 | 2025-04-03 13:15:00 | 825.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 14:15:00 | 752.15 | 2025-04-03 13:15:00 | 827.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 15:00:00 | 751.25 | 2025-04-03 13:15:00 | 826.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-24 09:45:00 | 755.65 | 2025-04-03 13:15:00 | 831.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-27 13:15:00 | 796.30 | 2025-04-07 09:15:00 | 773.30 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-04-01 11:15:00 | 791.00 | 2025-04-07 09:15:00 | 773.30 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-04-21 09:15:00 | 869.05 | 2025-04-25 14:15:00 | 876.30 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2025-04-29 10:15:00 | 872.00 | 2025-05-02 12:15:00 | 828.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 10:15:00 | 874.90 | 2025-05-02 12:15:00 | 831.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 11:00:00 | 875.00 | 2025-05-02 12:15:00 | 831.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 12:15:00 | 874.75 | 2025-05-02 12:15:00 | 831.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 10:15:00 | 872.00 | 2025-05-05 09:15:00 | 861.65 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2025-04-30 10:15:00 | 874.90 | 2025-05-05 09:15:00 | 861.65 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2025-04-30 11:00:00 | 875.00 | 2025-05-05 09:15:00 | 861.65 | STOP_HIT | 0.50 | 1.53% |
| SELL | retest2 | 2025-04-30 12:15:00 | 874.75 | 2025-05-05 09:15:00 | 861.65 | STOP_HIT | 0.50 | 1.50% |
| SELL | retest2 | 2025-04-30 14:30:00 | 867.45 | 2025-05-06 11:15:00 | 824.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 10:00:00 | 866.60 | 2025-05-06 11:15:00 | 823.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 14:30:00 | 867.45 | 2025-05-07 09:15:00 | 861.65 | STOP_HIT | 0.50 | 0.67% |
| SELL | retest2 | 2025-05-02 10:00:00 | 866.60 | 2025-05-07 09:15:00 | 861.65 | STOP_HIT | 0.50 | 0.57% |
| BUY | retest2 | 2025-05-13 11:45:00 | 853.80 | 2025-05-14 09:15:00 | 836.45 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-05-13 12:15:00 | 853.30 | 2025-05-14 09:15:00 | 836.45 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-05-13 14:45:00 | 859.50 | 2025-05-14 09:15:00 | 836.45 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-06-11 09:15:00 | 963.80 | 2025-06-12 09:15:00 | 903.45 | STOP_HIT | 1.00 | -6.26% |
| BUY | retest2 | 2025-06-11 10:30:00 | 962.50 | 2025-06-12 09:15:00 | 903.45 | STOP_HIT | 1.00 | -6.14% |
| BUY | retest2 | 2025-06-25 15:15:00 | 896.75 | 2025-07-07 12:15:00 | 922.60 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2025-06-26 10:15:00 | 897.20 | 2025-07-07 12:15:00 | 922.60 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-06-26 11:15:00 | 899.55 | 2025-07-07 12:15:00 | 922.60 | STOP_HIT | 1.00 | 2.56% |
| BUY | retest2 | 2025-07-11 14:15:00 | 932.55 | 2025-07-22 09:15:00 | 1025.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1088.40 | 2025-08-05 09:15:00 | 1068.10 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1088.10 | 2025-08-05 09:15:00 | 1068.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-08-04 13:45:00 | 1087.00 | 2025-08-05 09:15:00 | 1068.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-11 11:15:00 | 1231.00 | 2025-09-15 11:15:00 | 1238.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-11 12:15:00 | 1231.00 | 2025-09-15 11:15:00 | 1238.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-12 10:45:00 | 1227.10 | 2025-09-15 11:15:00 | 1238.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-18 09:30:00 | 1219.30 | 2025-09-18 14:15:00 | 1229.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1187.10 | 2025-09-26 09:15:00 | 1128.12 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-09-23 13:30:00 | 1187.50 | 2025-09-26 14:15:00 | 1127.74 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-09-23 14:00:00 | 1187.30 | 2025-09-26 14:15:00 | 1127.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1187.10 | 2025-09-30 14:15:00 | 1123.40 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2025-09-23 13:30:00 | 1187.50 | 2025-09-30 14:15:00 | 1123.40 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2025-09-23 14:00:00 | 1187.30 | 2025-09-30 14:15:00 | 1123.40 | STOP_HIT | 0.50 | 5.38% |
| BUY | retest2 | 2025-10-13 10:30:00 | 1247.70 | 2025-10-14 09:15:00 | 1229.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-13 12:00:00 | 1247.80 | 2025-10-14 09:15:00 | 1229.50 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-13 13:45:00 | 1246.30 | 2025-10-14 09:15:00 | 1229.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-14 14:45:00 | 1248.20 | 2025-10-24 12:15:00 | 1288.50 | STOP_HIT | 1.00 | 3.23% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1258.70 | 2025-10-24 12:15:00 | 1288.50 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1318.50 | 2025-10-31 15:15:00 | 1302.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-10 14:15:00 | 1336.00 | 2025-11-11 09:15:00 | 1324.40 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-11-11 09:15:00 | 1338.00 | 2025-11-11 09:15:00 | 1324.40 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1336.30 | 2025-11-12 09:15:00 | 1323.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-11-13 14:45:00 | 1304.70 | 2025-11-17 09:15:00 | 1323.60 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-11-14 10:00:00 | 1304.80 | 2025-11-17 09:15:00 | 1323.60 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-11-21 10:15:00 | 1268.00 | 2025-11-26 10:15:00 | 1274.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-11-21 13:30:00 | 1268.60 | 2025-11-26 12:15:00 | 1275.90 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-21 14:30:00 | 1269.40 | 2025-11-26 12:15:00 | 1275.90 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-24 09:30:00 | 1266.00 | 2025-11-26 12:15:00 | 1275.90 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-25 11:30:00 | 1243.00 | 2025-11-26 12:15:00 | 1275.90 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-12-18 14:15:00 | 1282.40 | 2025-12-19 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-12-31 14:00:00 | 1297.10 | 2026-01-02 09:15:00 | 1312.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-31 15:15:00 | 1297.00 | 2026-01-02 09:15:00 | 1312.90 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-01 09:30:00 | 1297.70 | 2026-01-02 09:15:00 | 1312.90 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-01 12:15:00 | 1292.20 | 2026-01-02 09:15:00 | 1312.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-01-06 12:15:00 | 1332.80 | 2026-01-07 11:15:00 | 1320.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-01-06 12:45:00 | 1334.50 | 2026-01-07 11:15:00 | 1320.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-01-06 15:00:00 | 1333.70 | 2026-01-07 11:15:00 | 1320.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-01-07 10:45:00 | 1333.30 | 2026-01-07 11:15:00 | 1320.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1272.70 | 2026-01-14 09:15:00 | 1304.70 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-01-19 15:00:00 | 1334.40 | 2026-01-20 10:15:00 | 1317.90 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-29 12:00:00 | 1160.00 | 2026-02-01 12:15:00 | 1044.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1124.30 | 2026-02-01 12:15:00 | 1068.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1124.30 | 2026-02-01 12:15:00 | 1227.30 | STOP_HIT | 0.50 | -9.16% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1136.50 | 2026-03-02 09:15:00 | 1079.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:45:00 | 1136.30 | 2026-03-02 09:15:00 | 1079.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1136.50 | 2026-03-04 13:15:00 | 1049.10 | STOP_HIT | 0.50 | 7.69% |
| SELL | retest2 | 2026-02-26 09:45:00 | 1136.30 | 2026-03-04 13:15:00 | 1049.10 | STOP_HIT | 0.50 | 7.67% |
| BUY | retest2 | 2026-03-19 10:30:00 | 1057.00 | 2026-03-23 09:15:00 | 1001.80 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest2 | 2026-03-19 11:15:00 | 1057.50 | 2026-03-23 09:15:00 | 1001.80 | STOP_HIT | 1.00 | -5.27% |
| BUY | retest2 | 2026-03-19 12:45:00 | 1056.40 | 2026-03-23 09:15:00 | 1001.80 | STOP_HIT | 1.00 | -5.17% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1055.10 | 2026-03-23 09:15:00 | 1001.80 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2026-04-02 09:15:00 | 972.50 | 2026-04-02 14:15:00 | 1007.55 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2026-04-06 11:00:00 | 978.60 | 2026-04-06 12:15:00 | 1019.00 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2026-04-13 10:30:00 | 1101.80 | 2026-04-23 09:15:00 | 1148.90 | STOP_HIT | 1.00 | 4.27% |
| BUY | retest2 | 2026-04-13 12:30:00 | 1102.70 | 2026-04-23 09:15:00 | 1148.90 | STOP_HIT | 1.00 | 4.19% |
| BUY | retest2 | 2026-04-13 13:00:00 | 1103.70 | 2026-04-23 09:15:00 | 1148.90 | STOP_HIT | 1.00 | 4.10% |
| SELL | retest2 | 2026-04-24 11:30:00 | 1146.15 | 2026-04-27 09:15:00 | 1088.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 13:30:00 | 1146.10 | 2026-04-27 09:15:00 | 1088.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 14:00:00 | 1147.35 | 2026-04-27 09:15:00 | 1089.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 15:00:00 | 1146.10 | 2026-04-27 09:15:00 | 1088.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 11:30:00 | 1146.15 | 2026-04-27 12:15:00 | 1140.70 | STOP_HIT | 0.50 | 0.48% |
| SELL | retest2 | 2026-04-24 13:30:00 | 1146.10 | 2026-04-27 12:15:00 | 1140.70 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2026-04-24 14:00:00 | 1147.35 | 2026-04-27 12:15:00 | 1140.70 | STOP_HIT | 0.50 | 0.58% |
| SELL | retest2 | 2026-04-24 15:00:00 | 1146.10 | 2026-04-27 12:15:00 | 1140.70 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2026-04-27 09:15:00 | 1061.60 | 2026-05-06 14:15:00 | 1108.40 | STOP_HIT | 1.00 | -4.41% |
