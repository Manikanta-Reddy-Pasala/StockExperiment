# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2025-01-03 15:15:00 (3131 bars)
- **Last close:** 736.78
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 108 |
| ALERT1 | 68 |
| ALERT2 | 66 |
| ALERT2_SKIP | 49 |
| ALERT3 | 120 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 70 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 48
- **Target hits / Stop hits / Partials:** 2 / 68 / 0
- **Avg / median % per leg:** -0.75% / -0.72%
- **Sum % (uncompounded):** -52.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 6 | 21.4% | 2 | 26 | 0 | -0.31% | -8.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 6 | 21.4% | 2 | 26 | 0 | -0.31% | -8.7% |
| SELL (all) | 42 | 16 | 38.1% | 0 | 42 | 0 | -1.04% | -43.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 16 | 38.1% | 0 | 42 | 0 | -1.04% | -43.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 70 | 22 | 31.4% | 2 | 68 | 0 | -0.75% | -52.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 14:15:00 | 670.00 | 672.70 | 672.93 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 09:15:00 | 680.38 | 674.11 | 673.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 11:15:00 | 684.00 | 680.91 | 679.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 677.51 | 681.08 | 679.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 14:15:00 | 677.51 | 681.08 | 679.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 677.51 | 681.08 | 679.82 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 09:15:00 | 678.60 | 679.76 | 679.85 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 683.64 | 679.73 | 679.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 09:15:00 | 687.97 | 681.98 | 680.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 10:15:00 | 697.81 | 697.86 | 694.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 11:15:00 | 693.90 | 697.07 | 694.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 693.90 | 697.07 | 694.40 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 701.64 | 706.46 | 706.57 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 09:15:00 | 710.35 | 705.14 | 705.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 714.79 | 709.74 | 707.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 09:15:00 | 710.98 | 714.28 | 711.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 710.98 | 714.28 | 711.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 710.98 | 714.28 | 711.80 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 09:15:00 | 709.61 | 710.83 | 710.90 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 15:15:00 | 712.30 | 710.90 | 710.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 09:15:00 | 718.00 | 712.32 | 711.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 724.52 | 729.75 | 724.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 10:15:00 | 722.50 | 728.30 | 723.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 722.50 | 728.30 | 723.96 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 11:15:00 | 721.70 | 723.09 | 723.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 09:15:00 | 711.61 | 719.83 | 721.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 700.09 | 699.04 | 703.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 698.30 | 699.05 | 702.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 698.30 | 699.05 | 702.69 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 711.48 | 703.90 | 703.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 14:15:00 | 716.44 | 711.28 | 708.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 09:15:00 | 778.63 | 780.90 | 768.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 769.23 | 775.31 | 771.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 769.23 | 775.31 | 771.05 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 764.16 | 769.05 | 769.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 14:15:00 | 761.83 | 767.61 | 768.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 15:15:00 | 756.20 | 754.79 | 759.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 761.20 | 756.07 | 759.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 761.20 | 756.07 | 759.78 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 14:15:00 | 751.37 | 748.04 | 747.72 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 738.70 | 746.63 | 747.26 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 750.93 | 747.06 | 746.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 12:15:00 | 755.95 | 748.84 | 747.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 09:15:00 | 751.93 | 752.78 | 750.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 09:15:00 | 755.67 | 756.76 | 753.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 755.67 | 756.76 | 753.87 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 14:15:00 | 743.80 | 759.38 | 759.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 14:15:00 | 728.67 | 743.64 | 750.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 14:15:00 | 737.50 | 733.85 | 740.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 712.35 | 713.09 | 718.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 712.35 | 713.09 | 718.26 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 15:15:00 | 720.30 | 714.67 | 714.09 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 711.25 | 713.59 | 713.67 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 15:15:00 | 717.20 | 713.48 | 713.16 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 11:15:00 | 711.30 | 712.73 | 712.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 706.15 | 711.28 | 712.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 13:15:00 | 705.21 | 704.04 | 707.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 697.10 | 697.73 | 701.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 697.10 | 697.73 | 701.06 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 12:15:00 | 704.00 | 695.75 | 694.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 13:15:00 | 704.98 | 697.59 | 695.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 707.20 | 707.38 | 703.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 701.72 | 705.92 | 703.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 701.72 | 705.92 | 703.07 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 719.60 | 724.24 | 724.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 717.08 | 721.55 | 722.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 11:15:00 | 726.30 | 721.19 | 722.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 11:15:00 | 726.30 | 721.19 | 722.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 726.30 | 721.19 | 722.15 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 13:15:00 | 729.07 | 723.72 | 723.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 731.81 | 725.34 | 723.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 13:15:00 | 729.00 | 729.45 | 727.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 14:15:00 | 727.10 | 728.98 | 727.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 727.10 | 728.98 | 727.06 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 734.40 | 737.23 | 737.28 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-12 11:15:00 | 738.57 | 737.50 | 737.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-12 12:15:00 | 742.89 | 738.58 | 737.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 09:15:00 | 744.68 | 744.73 | 742.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 15:15:00 | 749.15 | 750.28 | 747.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 749.15 | 750.28 | 747.89 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 11:15:00 | 747.70 | 751.86 | 752.24 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 754.80 | 751.87 | 751.62 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 14:15:00 | 747.00 | 750.90 | 751.22 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 09:15:00 | 776.30 | 755.43 | 753.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 10:15:00 | 780.58 | 760.46 | 755.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 09:15:00 | 779.22 | 783.10 | 775.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 10:15:00 | 778.80 | 782.24 | 776.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 778.80 | 782.24 | 776.08 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 773.50 | 777.07 | 777.33 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 783.60 | 778.38 | 777.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 11:15:00 | 784.53 | 781.68 | 780.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 789.09 | 790.58 | 786.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 12:15:00 | 785.18 | 789.50 | 786.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 785.18 | 789.50 | 786.45 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 803.34 | 806.39 | 806.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 14:15:00 | 801.50 | 805.41 | 806.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 12:15:00 | 804.10 | 803.63 | 804.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 12:15:00 | 804.10 | 803.63 | 804.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 804.10 | 803.63 | 804.83 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 09:15:00 | 811.25 | 804.79 | 804.63 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 791.65 | 803.25 | 804.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 786.91 | 795.57 | 800.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 12:15:00 | 788.89 | 787.35 | 793.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 09:15:00 | 783.03 | 779.73 | 784.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 783.03 | 779.73 | 784.26 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 12:15:00 | 752.96 | 747.21 | 747.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 13:15:00 | 755.67 | 748.90 | 747.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 745.82 | 751.59 | 750.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 11:15:00 | 745.82 | 751.59 | 750.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 745.82 | 751.59 | 750.04 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 09:15:00 | 743.35 | 748.97 | 749.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 10:15:00 | 740.89 | 745.22 | 746.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 741.34 | 740.99 | 743.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 10:15:00 | 742.61 | 741.31 | 743.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 742.61 | 741.31 | 743.49 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 15:15:00 | 713.00 | 709.07 | 708.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 714.31 | 710.12 | 709.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 710.22 | 711.83 | 710.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 710.22 | 711.83 | 710.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 710.22 | 711.83 | 710.85 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 729.28 | 735.99 | 736.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 724.70 | 730.19 | 731.60 | Break + close below crossover candle low |

### Cycle 38 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 747.82 | 730.43 | 730.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 749.14 | 736.90 | 733.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 755.07 | 755.79 | 750.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 13:15:00 | 751.68 | 760.00 | 757.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 751.68 | 760.00 | 757.36 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 748.10 | 755.78 | 755.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 09:15:00 | 744.69 | 753.56 | 754.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 09:15:00 | 725.05 | 723.28 | 730.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 09:15:00 | 725.05 | 723.28 | 730.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 725.05 | 723.28 | 730.66 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 12:15:00 | 730.70 | 726.97 | 726.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 13:15:00 | 731.07 | 727.79 | 727.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 10:15:00 | 729.44 | 729.60 | 728.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 10:15:00 | 729.44 | 729.60 | 728.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 729.44 | 729.60 | 728.49 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 15:15:00 | 767.50 | 769.91 | 770.18 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 774.63 | 770.85 | 770.59 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 14:15:00 | 766.74 | 770.85 | 770.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 11:15:00 | 764.30 | 767.84 | 769.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 756.10 | 754.04 | 759.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 756.10 | 754.04 | 759.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 756.10 | 754.04 | 759.43 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 13:15:00 | 719.44 | 713.90 | 713.16 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 09:15:00 | 687.81 | 710.32 | 711.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 682.25 | 693.08 | 701.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 11:15:00 | 690.36 | 690.28 | 697.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 10:15:00 | 685.04 | 680.65 | 685.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 685.04 | 680.65 | 685.31 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 664.59 | 661.30 | 661.29 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 12:15:00 | 660.30 | 661.20 | 661.32 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 14:15:00 | 662.02 | 661.50 | 661.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 663.30 | 661.94 | 661.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 662.00 | 662.05 | 661.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 11:15:00 | 662.00 | 662.05 | 661.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 662.00 | 662.05 | 661.76 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 661.60 | 669.51 | 670.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 10:15:00 | 658.94 | 667.40 | 669.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 666.86 | 664.17 | 666.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 666.86 | 664.17 | 666.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 666.86 | 664.17 | 666.92 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 15:15:00 | 669.20 | 667.62 | 667.62 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 665.27 | 667.23 | 667.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 12:15:00 | 663.99 | 666.35 | 666.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 15:15:00 | 651.17 | 650.74 | 653.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 15:15:00 | 651.17 | 650.74 | 653.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 651.17 | 650.74 | 653.76 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 14:15:00 | 656.64 | 655.18 | 655.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 659.15 | 656.36 | 655.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 12:15:00 | 658.11 | 658.47 | 657.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 12:15:00 | 658.11 | 658.47 | 657.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 658.11 | 658.47 | 657.19 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 12:15:00 | 641.98 | 655.22 | 656.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 13:15:00 | 631.09 | 650.39 | 654.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 641.55 | 633.26 | 639.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 641.55 | 633.26 | 639.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 641.55 | 633.26 | 639.25 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 10:15:00 | 642.29 | 641.02 | 641.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 12:15:00 | 647.22 | 642.79 | 641.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 14:15:00 | 643.50 | 643.61 | 642.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 15:15:00 | 643.80 | 643.65 | 642.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 643.80 | 643.65 | 642.55 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-03-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 12:15:00 | 640.15 | 644.30 | 644.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 14:15:00 | 638.09 | 642.51 | 643.63 | Break + close below crossover candle low |

### Cycle 56 — BUY (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 09:15:00 | 654.20 | 644.13 | 644.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 11:15:00 | 658.00 | 654.03 | 650.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 13:15:00 | 724.80 | 725.16 | 715.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 715.26 | 723.05 | 716.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 715.26 | 723.05 | 716.69 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 11:15:00 | 716.76 | 724.14 | 724.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 713.64 | 717.89 | 719.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 14:15:00 | 718.47 | 717.68 | 719.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 14:15:00 | 718.47 | 717.68 | 719.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 718.47 | 717.68 | 719.06 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 14:15:00 | 722.03 | 719.93 | 719.66 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 714.92 | 720.10 | 720.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 707.80 | 715.34 | 717.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 13:15:00 | 702.85 | 702.74 | 709.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 14:00:00 | 702.85 | 702.74 | 709.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 706.35 | 694.38 | 697.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:00:00 | 706.35 | 694.38 | 697.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 712.34 | 697.97 | 698.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:30:00 | 712.27 | 697.97 | 698.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 15:15:00 | 711.00 | 700.57 | 699.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 723.00 | 705.06 | 701.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 725.82 | 726.10 | 719.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 15:00:00 | 725.82 | 726.10 | 719.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 721.56 | 729.26 | 725.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 721.56 | 729.26 | 725.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 722.25 | 727.86 | 725.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:30:00 | 727.90 | 727.21 | 725.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 12:45:00 | 725.42 | 727.20 | 725.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 680.84 | 719.36 | 722.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 09:15:00 | 680.84 | 719.36 | 722.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 10:15:00 | 675.80 | 710.65 | 718.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 13:15:00 | 681.80 | 680.28 | 692.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 14:00:00 | 681.80 | 680.28 | 692.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 688.50 | 683.58 | 690.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:45:00 | 688.00 | 683.58 | 690.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 689.46 | 684.76 | 689.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:00:00 | 689.46 | 684.76 | 689.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 688.33 | 685.47 | 689.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:30:00 | 690.00 | 685.47 | 689.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 692.50 | 686.88 | 690.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 13:45:00 | 692.93 | 686.88 | 690.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 691.01 | 687.70 | 690.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:45:00 | 696.82 | 687.70 | 690.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 691.76 | 688.93 | 690.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:15:00 | 690.53 | 688.93 | 690.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:45:00 | 689.91 | 688.72 | 690.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 09:15:00 | 725.20 | 696.07 | 692.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 725.20 | 696.07 | 692.83 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 687.72 | 693.78 | 694.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 14:15:00 | 686.20 | 691.40 | 693.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 12:15:00 | 690.32 | 687.84 | 690.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 12:15:00 | 690.32 | 687.84 | 690.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 690.32 | 687.84 | 690.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:30:00 | 691.53 | 687.84 | 690.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 687.16 | 687.70 | 690.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 14:15:00 | 684.89 | 687.70 | 690.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 684.31 | 687.84 | 689.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 10:15:00 | 684.71 | 687.71 | 689.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 15:15:00 | 673.10 | 669.16 | 668.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 673.10 | 669.16 | 668.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 676.56 | 671.06 | 669.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 672.76 | 673.24 | 671.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 672.76 | 673.24 | 671.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 676.57 | 674.25 | 672.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:15:00 | 678.20 | 674.25 | 672.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 677.55 | 675.85 | 674.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 12:30:00 | 681.00 | 677.04 | 675.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 13:15:00 | 683.78 | 687.26 | 687.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 683.78 | 687.26 | 687.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 681.12 | 686.03 | 686.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 672.79 | 670.07 | 675.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:45:00 | 673.90 | 670.07 | 675.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 675.20 | 671.10 | 675.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:30:00 | 671.25 | 671.73 | 675.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 13:15:00 | 672.17 | 671.73 | 675.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:45:00 | 671.98 | 671.78 | 674.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 690.61 | 675.78 | 676.09 | SL hit (close>static) qty=1.00 sl=677.20 alert=retest2 |

### Cycle 66 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 695.55 | 679.74 | 677.86 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 653.66 | 675.58 | 678.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 650.76 | 666.68 | 673.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 668.28 | 664.76 | 670.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 668.28 | 664.76 | 670.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 672.80 | 666.37 | 670.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 672.80 | 666.37 | 670.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 677.90 | 668.68 | 671.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:45:00 | 677.96 | 668.68 | 671.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 683.07 | 673.30 | 673.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 696.94 | 679.68 | 676.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 13:15:00 | 712.77 | 713.28 | 705.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:15:00 | 710.15 | 713.28 | 705.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 712.30 | 711.75 | 706.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:30:00 | 713.30 | 712.23 | 708.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 725.38 | 729.39 | 729.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 725.38 | 729.39 | 729.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 13:15:00 | 722.60 | 728.03 | 728.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 12:15:00 | 721.37 | 721.12 | 724.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 13:00:00 | 721.37 | 721.12 | 724.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 719.29 | 720.75 | 724.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:30:00 | 718.31 | 719.34 | 722.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 717.78 | 710.89 | 710.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 717.78 | 710.89 | 710.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 720.00 | 713.94 | 712.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 711.10 | 714.63 | 713.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 711.10 | 714.63 | 713.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 711.10 | 714.63 | 713.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 711.10 | 714.63 | 713.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 710.23 | 713.75 | 712.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 710.23 | 713.75 | 712.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 716.88 | 714.38 | 713.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 711.21 | 714.38 | 713.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 718.16 | 715.55 | 713.95 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 711.41 | 713.76 | 713.94 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 717.86 | 714.38 | 714.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 724.20 | 716.35 | 715.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 714.21 | 718.93 | 716.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 714.21 | 718.93 | 716.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 714.21 | 718.93 | 716.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 714.21 | 718.93 | 716.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 718.40 | 718.82 | 716.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:45:00 | 721.69 | 718.28 | 717.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 723.18 | 717.55 | 716.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:45:00 | 721.61 | 718.26 | 717.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 720.43 | 722.11 | 720.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 716.70 | 721.03 | 719.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 716.70 | 721.03 | 719.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 713.90 | 719.60 | 719.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-04 13:15:00 | 713.30 | 718.34 | 718.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 13:15:00 | 713.30 | 718.34 | 718.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 15:15:00 | 710.90 | 715.71 | 717.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 714.30 | 713.20 | 715.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 15:00:00 | 714.30 | 713.20 | 715.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 708.40 | 712.25 | 714.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 14:00:00 | 703.51 | 706.52 | 708.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 14:45:00 | 704.40 | 706.47 | 708.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 15:15:00 | 704.50 | 706.47 | 708.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 10:15:00 | 704.30 | 701.30 | 701.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 704.88 | 702.01 | 702.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:15:00 | 703.28 | 702.01 | 702.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 11:15:00 | 705.50 | 702.71 | 702.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 705.50 | 702.71 | 702.36 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 700.72 | 703.49 | 703.80 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 13:15:00 | 709.15 | 704.97 | 704.44 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 700.19 | 704.60 | 704.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 696.26 | 702.16 | 703.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 10:15:00 | 663.20 | 663.13 | 671.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 10:45:00 | 663.17 | 663.13 | 671.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 673.75 | 666.25 | 669.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:30:00 | 673.53 | 666.25 | 669.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 676.02 | 668.21 | 669.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 676.02 | 668.21 | 669.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 679.00 | 671.81 | 671.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 679.95 | 675.35 | 673.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 681.50 | 682.93 | 680.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 681.50 | 682.93 | 680.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 682.59 | 682.57 | 680.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 682.80 | 682.57 | 680.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 680.04 | 682.16 | 681.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 680.04 | 682.16 | 681.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 680.95 | 681.92 | 681.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 683.42 | 681.92 | 681.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 09:15:00 | 678.87 | 681.31 | 680.85 | SL hit (close<static) qty=1.00 sl=679.30 alert=retest2 |

### Cycle 79 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 677.98 | 680.58 | 680.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 677.32 | 679.60 | 680.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 663.00 | 662.58 | 667.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 663.00 | 662.58 | 667.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 663.00 | 662.58 | 667.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 660.21 | 662.58 | 667.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 659.47 | 660.96 | 665.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:00:00 | 662.00 | 659.20 | 663.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:30:00 | 662.20 | 661.15 | 663.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 663.74 | 661.67 | 663.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 663.74 | 661.67 | 663.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 664.01 | 662.14 | 663.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:15:00 | 663.40 | 662.14 | 663.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:00:00 | 662.06 | 662.32 | 663.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 662.00 | 662.32 | 663.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 662.74 | 662.82 | 663.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 660.56 | 662.37 | 662.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:45:00 | 660.53 | 662.37 | 662.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 661.31 | 660.84 | 662.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 659.62 | 660.84 | 662.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:00:00 | 659.95 | 660.89 | 661.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 659.60 | 661.18 | 661.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 14:45:00 | 660.01 | 661.39 | 661.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 660.00 | 661.11 | 661.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 658.00 | 661.11 | 661.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 658.29 | 660.55 | 661.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:45:00 | 656.20 | 659.02 | 660.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 14:15:00 | 659.45 | 652.50 | 652.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 659.45 | 652.50 | 652.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 660.99 | 655.18 | 653.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 671.72 | 673.79 | 671.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 671.72 | 673.79 | 671.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 671.72 | 673.79 | 671.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 671.50 | 673.79 | 671.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 673.11 | 673.65 | 671.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:15:00 | 673.80 | 673.65 | 671.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 677.80 | 673.86 | 672.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-02 10:15:00 | 741.18 | 723.32 | 711.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 725.00 | 727.10 | 727.36 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 10:15:00 | 731.80 | 727.64 | 727.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 11:15:00 | 735.27 | 729.16 | 728.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 728.86 | 730.02 | 729.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 728.86 | 730.02 | 729.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 728.86 | 730.02 | 729.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 728.86 | 730.02 | 729.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 732.43 | 730.50 | 729.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 12:30:00 | 734.34 | 731.55 | 730.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 14:45:00 | 734.10 | 732.26 | 730.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 724.18 | 730.96 | 730.32 | SL hit (close<static) qty=1.00 sl=726.28 alert=retest2 |

### Cycle 83 — SELL (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 10:15:00 | 723.40 | 729.44 | 729.69 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 741.10 | 730.08 | 729.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 743.72 | 736.82 | 733.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 736.23 | 750.66 | 745.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 736.23 | 750.66 | 745.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 736.23 | 750.66 | 745.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 736.23 | 750.66 | 745.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 736.40 | 747.81 | 744.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:45:00 | 739.28 | 747.81 | 744.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 737.80 | 745.81 | 743.72 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 735.56 | 742.11 | 742.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 734.20 | 739.48 | 741.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 09:15:00 | 741.49 | 739.88 | 741.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 741.49 | 739.88 | 741.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 741.49 | 739.88 | 741.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 741.49 | 739.88 | 741.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 738.50 | 739.61 | 740.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 14:45:00 | 735.30 | 738.85 | 740.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 752.70 | 741.18 | 740.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 09:15:00 | 752.70 | 741.18 | 740.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 10:15:00 | 755.00 | 743.95 | 742.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 758.72 | 759.24 | 752.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 757.30 | 759.00 | 755.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 757.30 | 759.00 | 755.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 759.72 | 759.00 | 755.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 756.69 | 759.00 | 756.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 753.50 | 759.00 | 756.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 759.20 | 759.04 | 756.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 15:15:00 | 759.80 | 759.04 | 756.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 13:45:00 | 760.00 | 757.73 | 756.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 14:15:00 | 760.10 | 757.73 | 756.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 752.67 | 757.07 | 756.84 | SL hit (close<static) qty=1.00 sl=752.71 alert=retest2 |

### Cycle 87 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 753.79 | 756.42 | 756.57 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 10:15:00 | 759.81 | 756.26 | 756.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 14:15:00 | 763.50 | 759.19 | 757.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 13:15:00 | 774.33 | 775.52 | 772.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 13:30:00 | 773.98 | 775.52 | 772.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 770.22 | 774.46 | 772.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 770.22 | 774.46 | 772.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 769.79 | 773.53 | 771.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 767.22 | 773.53 | 771.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 778.36 | 774.49 | 772.56 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 763.60 | 771.33 | 772.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 759.01 | 768.87 | 770.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 737.02 | 725.41 | 729.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 737.02 | 725.41 | 729.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 737.02 | 725.41 | 729.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 737.02 | 725.41 | 729.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 740.54 | 728.43 | 730.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 740.54 | 728.43 | 730.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 736.00 | 731.43 | 731.32 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 728.90 | 730.93 | 731.10 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 735.82 | 731.90 | 731.51 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 09:15:00 | 727.30 | 731.39 | 731.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 722.35 | 726.00 | 728.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 691.74 | 690.61 | 696.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 691.74 | 690.61 | 696.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 682.00 | 688.68 | 693.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:45:00 | 678.75 | 686.50 | 691.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:30:00 | 679.42 | 679.38 | 685.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:00:00 | 679.44 | 679.39 | 684.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 700.77 | 678.78 | 681.36 | SL hit (close>static) qty=1.00 sl=695.08 alert=retest2 |

### Cycle 94 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 10:15:00 | 705.88 | 684.20 | 683.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 09:15:00 | 708.52 | 698.98 | 692.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 696.55 | 701.02 | 697.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 696.55 | 701.02 | 697.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 696.55 | 701.02 | 697.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 696.55 | 701.02 | 697.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 699.40 | 700.70 | 697.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 693.20 | 700.70 | 697.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 696.56 | 699.97 | 697.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 13:00:00 | 696.56 | 699.97 | 697.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 693.71 | 698.72 | 697.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:00:00 | 693.71 | 698.72 | 697.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 690.50 | 695.75 | 696.08 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 701.75 | 696.72 | 696.45 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 12:15:00 | 691.28 | 695.68 | 696.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 684.30 | 691.77 | 693.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 12:15:00 | 696.43 | 691.98 | 693.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 12:15:00 | 696.43 | 691.98 | 693.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 696.43 | 691.98 | 693.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:45:00 | 696.73 | 691.98 | 693.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 703.78 | 694.34 | 694.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 703.78 | 694.34 | 694.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 703.77 | 696.23 | 695.21 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 692.55 | 695.45 | 695.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 691.09 | 693.98 | 694.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 693.08 | 692.03 | 693.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 693.08 | 692.03 | 693.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 693.08 | 692.03 | 693.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 693.08 | 692.03 | 693.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 691.11 | 691.85 | 693.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:30:00 | 693.08 | 691.85 | 693.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 682.86 | 690.05 | 692.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:15:00 | 680.50 | 690.05 | 692.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 09:30:00 | 681.32 | 683.76 | 687.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 10:30:00 | 678.91 | 683.07 | 686.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 13:15:00 | 697.52 | 683.66 | 685.73 | SL hit (close>static) qty=1.00 sl=697.50 alert=retest2 |

### Cycle 100 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 692.16 | 687.51 | 687.18 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 686.42 | 690.33 | 690.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 681.76 | 686.78 | 688.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 660.41 | 658.06 | 662.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 660.41 | 658.06 | 662.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 660.41 | 658.06 | 662.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 657.94 | 658.03 | 661.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 657.71 | 658.69 | 661.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:30:00 | 657.75 | 659.48 | 660.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:00:00 | 657.47 | 659.48 | 660.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 659.38 | 659.46 | 660.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:45:00 | 660.25 | 659.46 | 660.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 658.00 | 659.17 | 660.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 655.10 | 659.17 | 660.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 650.36 | 657.41 | 659.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 11:30:00 | 648.50 | 654.21 | 657.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:00:00 | 648.16 | 654.21 | 657.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 14:00:00 | 648.50 | 652.12 | 655.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 12:15:00 | 662.47 | 655.05 | 655.44 | SL hit (close>static) qty=1.00 sl=661.40 alert=retest2 |

### Cycle 102 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 665.57 | 657.16 | 656.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 667.63 | 659.25 | 657.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 13:15:00 | 669.09 | 669.38 | 664.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:45:00 | 668.40 | 669.38 | 664.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 669.15 | 668.99 | 665.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:45:00 | 670.87 | 666.12 | 665.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:30:00 | 669.49 | 666.89 | 665.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 669.21 | 667.12 | 666.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 10:00:00 | 671.40 | 669.70 | 667.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 663.52 | 668.47 | 667.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 664.50 | 668.47 | 667.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 662.58 | 667.29 | 667.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-28 11:15:00 | 662.58 | 667.29 | 667.07 | SL hit (close<static) qty=1.00 sl=663.10 alert=retest2 |

### Cycle 103 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 660.00 | 665.83 | 666.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 656.89 | 664.04 | 665.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 659.20 | 658.89 | 662.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:45:00 | 660.20 | 658.89 | 662.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 650.98 | 656.31 | 659.41 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 664.70 | 660.88 | 660.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 666.38 | 661.98 | 660.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 12:15:00 | 664.17 | 664.22 | 662.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 13:00:00 | 664.17 | 664.22 | 662.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 670.91 | 671.63 | 668.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 670.17 | 671.63 | 668.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 686.21 | 688.18 | 685.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 686.07 | 688.18 | 685.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 692.36 | 689.02 | 686.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:45:00 | 690.20 | 689.02 | 686.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 705.74 | 709.94 | 704.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:15:00 | 702.70 | 709.94 | 704.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 702.54 | 708.46 | 704.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 703.14 | 708.46 | 704.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 711.88 | 709.14 | 705.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:15:00 | 714.50 | 709.14 | 705.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:45:00 | 714.97 | 710.08 | 706.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:15:00 | 716.10 | 710.08 | 706.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:45:00 | 718.40 | 719.70 | 715.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 713.50 | 718.31 | 715.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 713.50 | 718.31 | 715.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 713.20 | 717.29 | 715.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:30:00 | 717.41 | 715.07 | 714.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 710.90 | 714.24 | 714.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 710.90 | 714.24 | 714.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 708.24 | 712.47 | 713.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 693.03 | 688.78 | 694.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 693.03 | 688.78 | 694.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 693.03 | 688.78 | 694.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 691.97 | 688.78 | 694.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 695.32 | 690.09 | 694.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 695.32 | 690.09 | 694.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 690.59 | 690.19 | 694.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:00:00 | 689.16 | 689.98 | 693.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 694.81 | 687.03 | 686.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 694.81 | 687.03 | 686.72 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 681.24 | 688.26 | 688.69 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 689.50 | 687.00 | 686.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 692.70 | 688.97 | 687.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 736.78 | 737.74 | 725.28 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-25 11:30:00 | 727.90 | 2024-04-26 09:15:00 | 680.84 | STOP_HIT | 1.00 | -6.47% |
| BUY | retest2 | 2024-04-25 12:45:00 | 725.42 | 2024-04-26 09:15:00 | 680.84 | STOP_HIT | 1.00 | -6.15% |
| SELL | retest2 | 2024-05-02 10:15:00 | 690.53 | 2024-05-03 09:15:00 | 725.20 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2024-05-02 10:45:00 | 689.91 | 2024-05-03 09:15:00 | 725.20 | STOP_HIT | 1.00 | -5.12% |
| SELL | retest2 | 2024-05-07 14:15:00 | 684.89 | 2024-05-16 15:15:00 | 673.10 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2024-05-08 09:15:00 | 684.31 | 2024-05-16 15:15:00 | 673.10 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2024-05-08 10:15:00 | 684.71 | 2024-05-16 15:15:00 | 673.10 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2024-05-21 10:15:00 | 678.20 | 2024-05-29 13:15:00 | 683.78 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2024-05-23 11:15:00 | 677.55 | 2024-05-29 13:15:00 | 683.78 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2024-05-23 12:30:00 | 681.00 | 2024-05-29 13:15:00 | 683.78 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2024-05-31 12:30:00 | 671.25 | 2024-06-03 09:15:00 | 690.61 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-05-31 13:15:00 | 672.17 | 2024-06-03 09:15:00 | 690.61 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-05-31 14:45:00 | 671.98 | 2024-06-03 09:15:00 | 690.61 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-06-11 12:30:00 | 713.30 | 2024-06-19 12:15:00 | 725.38 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2024-06-21 09:30:00 | 718.31 | 2024-06-26 13:15:00 | 717.78 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-07-02 13:45:00 | 721.69 | 2024-07-04 13:15:00 | 713.30 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-07-03 09:15:00 | 723.18 | 2024-07-04 13:15:00 | 713.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-07-03 10:45:00 | 721.61 | 2024-07-04 13:15:00 | 713.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-07-04 11:15:00 | 720.43 | 2024-07-04 13:15:00 | 713.30 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-10 14:00:00 | 703.51 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-07-10 14:45:00 | 704.40 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-07-10 15:15:00 | 704.50 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-07-15 10:15:00 | 704.30 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-07-15 11:15:00 | 703.28 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-08-01 09:15:00 | 683.42 | 2024-08-01 09:15:00 | 678.87 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-08-01 11:00:00 | 682.60 | 2024-08-01 12:15:00 | 677.98 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-08-06 10:15:00 | 660.21 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-08-06 13:30:00 | 659.47 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-08-07 10:00:00 | 662.00 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-08-07 12:30:00 | 662.20 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-08-07 15:15:00 | 663.40 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2024-08-08 10:00:00 | 662.06 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-08-08 10:30:00 | 662.00 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-08-08 13:00:00 | 662.74 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2024-08-09 10:15:00 | 659.62 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-08-09 12:00:00 | 659.95 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-08-12 09:15:00 | 659.60 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-08-12 14:45:00 | 660.01 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-08-13 12:45:00 | 656.20 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-08-23 11:15:00 | 673.80 | 2024-09-02 10:15:00 | 741.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 09:15:00 | 677.80 | 2024-09-03 09:15:00 | 745.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-09 12:30:00 | 734.34 | 2024-09-10 09:15:00 | 724.18 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-09-09 14:45:00 | 734.10 | 2024-09-10 09:15:00 | 724.18 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-09-17 14:45:00 | 735.30 | 2024-09-18 09:15:00 | 752.70 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-09-20 15:15:00 | 759.80 | 2024-09-24 09:15:00 | 752.67 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-09-23 13:45:00 | 760.00 | 2024-09-24 09:15:00 | 752.67 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-09-23 14:15:00 | 760.10 | 2024-09-24 09:15:00 | 752.67 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-10-21 10:45:00 | 678.75 | 2024-10-23 09:15:00 | 700.77 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-10-22 09:30:00 | 679.42 | 2024-10-23 09:15:00 | 700.77 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-10-22 11:00:00 | 679.44 | 2024-10-23 09:15:00 | 700.77 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-11-04 10:15:00 | 680.50 | 2024-11-05 13:15:00 | 697.52 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-11-05 09:30:00 | 681.32 | 2024-11-05 13:15:00 | 697.52 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-11-05 10:30:00 | 678.91 | 2024-11-05 13:15:00 | 697.52 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-11-18 11:00:00 | 657.94 | 2024-11-22 12:15:00 | 662.47 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-11-18 14:00:00 | 657.71 | 2024-11-22 12:15:00 | 662.47 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-11-19 13:30:00 | 657.75 | 2024-11-22 12:15:00 | 662.47 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-11-19 14:00:00 | 657.47 | 2024-11-22 13:15:00 | 665.57 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-11-21 11:30:00 | 648.50 | 2024-11-22 13:15:00 | 665.57 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-11-21 12:00:00 | 648.16 | 2024-11-22 13:15:00 | 665.57 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-11-21 14:00:00 | 648.50 | 2024-11-22 13:15:00 | 665.57 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-11-27 09:45:00 | 670.87 | 2024-11-28 11:15:00 | 662.58 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-11-27 10:30:00 | 669.49 | 2024-11-28 11:15:00 | 662.58 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-11-27 12:15:00 | 669.21 | 2024-11-28 11:15:00 | 662.58 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-11-28 10:00:00 | 671.40 | 2024-11-28 11:15:00 | 662.58 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-12-13 12:15:00 | 714.50 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-13 12:45:00 | 714.97 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-12-13 13:15:00 | 716.10 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-12-17 10:45:00 | 718.40 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-18 09:30:00 | 717.41 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-12-23 13:00:00 | 689.16 | 2024-12-27 10:15:00 | 694.81 | STOP_HIT | 1.00 | -0.82% |
