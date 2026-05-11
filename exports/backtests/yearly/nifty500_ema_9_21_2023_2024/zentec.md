# Zen Technologies Ltd. (ZENTEC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5310 bars)
- **Last close:** 1626.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 227 |
| ALERT1 | 145 |
| ALERT2 | 146 |
| ALERT2_SKIP | 97 |
| ALERT3 | 278 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 113 |
| PARTIAL | 8 |
| TARGET_HIT | 10 |
| STOP_HIT | 106 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 124 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 89
- **Target hits / Stop hits / Partials:** 10 / 106 / 8
- **Avg / median % per leg:** -0.39% / -1.47%
- **Sum % (uncompounded):** -48.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 16 | 29.1% | 8 | 47 | 0 | 0.00% | 0.2% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.64% | 0.6% |
| BUY @ 3rd Alert (retest2) | 54 | 15 | 27.8% | 8 | 46 | 0 | -0.01% | -0.4% |
| SELL (all) | 69 | 19 | 27.5% | 2 | 59 | 8 | -0.70% | -48.5% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 1 | 1 | 0 | 3.85% | 7.7% |
| SELL @ 3rd Alert (retest2) | 67 | 18 | 26.9% | 1 | 58 | 8 | -0.84% | -56.2% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 2 | 0 | 2.78% | 8.3% |
| retest2 (combined) | 121 | 33 | 27.3% | 9 | 104 | 8 | -0.47% | -56.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 310.85 | 305.02 | 304.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 09:15:00 | 333.30 | 312.82 | 308.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 14:15:00 | 337.50 | 338.85 | 330.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 09:15:00 | 331.55 | 337.27 | 331.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 331.55 | 337.27 | 331.15 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 10:15:00 | 330.05 | 332.44 | 332.57 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 09:15:00 | 336.40 | 332.99 | 332.66 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 15:15:00 | 330.40 | 332.30 | 332.51 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 09:15:00 | 338.30 | 333.50 | 333.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 10:15:00 | 344.20 | 335.64 | 334.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 09:15:00 | 408.10 | 412.63 | 401.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 09:15:00 | 421.55 | 426.81 | 420.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 421.55 | 426.81 | 420.91 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 10:15:00 | 414.95 | 422.60 | 423.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 09:15:00 | 405.50 | 409.02 | 411.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 413.40 | 407.63 | 408.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 413.40 | 407.63 | 408.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 413.40 | 407.63 | 408.93 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 412.00 | 409.63 | 409.60 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 405.45 | 409.09 | 409.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 388.00 | 403.52 | 406.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 409.50 | 395.65 | 399.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 409.50 | 395.65 | 399.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 409.50 | 395.65 | 399.75 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 12:15:00 | 414.45 | 404.32 | 403.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 419.40 | 409.12 | 405.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 14:15:00 | 412.90 | 413.06 | 409.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 415.10 | 415.39 | 413.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 415.10 | 415.39 | 413.00 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 15:15:00 | 413.90 | 417.70 | 417.90 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 418.30 | 418.07 | 418.05 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 11:15:00 | 416.15 | 417.69 | 417.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 12:15:00 | 414.75 | 417.10 | 417.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 09:15:00 | 417.95 | 416.01 | 416.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 09:15:00 | 417.95 | 416.01 | 416.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 417.95 | 416.01 | 416.77 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 12:15:00 | 418.50 | 417.28 | 417.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 14:15:00 | 419.75 | 417.92 | 417.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 09:15:00 | 424.85 | 426.56 | 423.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 424.85 | 426.56 | 423.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 424.85 | 426.56 | 423.56 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 13:15:00 | 622.70 | 628.17 | 628.37 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 14:15:00 | 630.10 | 628.55 | 628.53 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 15:15:00 | 628.00 | 628.44 | 628.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-31 09:15:00 | 625.90 | 627.93 | 628.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-01 14:15:00 | 629.70 | 621.79 | 623.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 14:15:00 | 629.70 | 621.79 | 623.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 629.70 | 621.79 | 623.14 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 10:15:00 | 625.80 | 624.21 | 624.06 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 586.40 | 617.73 | 621.26 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 674.85 | 623.96 | 617.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 09:15:00 | 729.55 | 675.12 | 650.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 796.00 | 813.16 | 784.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 14:15:00 | 789.45 | 799.99 | 788.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 789.45 | 799.99 | 788.18 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 12:15:00 | 824.90 | 848.91 | 849.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 09:15:00 | 814.40 | 834.94 | 841.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 09:15:00 | 840.00 | 823.02 | 830.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 09:15:00 | 840.00 | 823.02 | 830.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 840.00 | 823.02 | 830.34 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 13:15:00 | 843.95 | 834.79 | 834.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 15:15:00 | 845.90 | 838.41 | 836.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 09:15:00 | 835.05 | 837.74 | 836.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 835.05 | 837.74 | 836.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 835.05 | 837.74 | 836.05 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 09:15:00 | 832.95 | 835.28 | 835.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 12:15:00 | 824.00 | 830.57 | 833.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 12:15:00 | 820.00 | 816.20 | 822.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 12:15:00 | 820.00 | 816.20 | 822.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 12:15:00 | 820.00 | 816.20 | 822.89 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2023-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 15:15:00 | 812.00 | 809.86 | 809.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 09:15:00 | 826.00 | 813.09 | 811.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 13:15:00 | 834.45 | 834.66 | 827.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 09:15:00 | 819.85 | 830.99 | 827.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 819.85 | 830.99 | 827.39 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 14:15:00 | 820.50 | 824.76 | 825.17 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 10:15:00 | 837.80 | 827.14 | 826.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 11:15:00 | 839.90 | 829.69 | 827.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 11:15:00 | 835.40 | 835.77 | 832.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 12:15:00 | 835.00 | 835.62 | 832.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 835.00 | 835.62 | 832.47 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 09:15:00 | 831.00 | 833.10 | 833.16 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 11:15:00 | 834.00 | 833.26 | 833.22 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 13:15:00 | 829.00 | 832.52 | 832.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 09:15:00 | 816.10 | 828.29 | 830.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 774.00 | 768.60 | 781.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 09:15:00 | 763.10 | 770.36 | 776.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 763.10 | 770.36 | 776.83 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 09:15:00 | 756.20 | 717.28 | 717.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 784.00 | 753.60 | 738.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 13:15:00 | 774.50 | 778.45 | 766.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 764.00 | 774.82 | 767.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 764.00 | 774.82 | 767.61 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 11:15:00 | 759.50 | 765.44 | 766.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 14:15:00 | 753.00 | 759.31 | 762.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 748.25 | 741.30 | 748.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 748.25 | 741.30 | 748.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 748.25 | 741.30 | 748.44 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 745.15 | 731.06 | 731.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 756.40 | 741.63 | 736.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 741.10 | 744.31 | 739.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 14:15:00 | 737.80 | 743.01 | 739.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 737.80 | 743.01 | 739.58 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 11:15:00 | 732.00 | 737.99 | 738.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 12:15:00 | 727.00 | 735.79 | 737.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 15:15:00 | 733.90 | 733.69 | 735.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 739.00 | 734.75 | 735.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 739.00 | 734.75 | 735.95 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 12:15:00 | 735.95 | 732.46 | 732.42 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 731.40 | 732.25 | 732.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 09:15:00 | 728.00 | 731.25 | 731.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 695.00 | 689.93 | 700.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 695.00 | 689.93 | 700.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 695.00 | 689.93 | 700.07 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-10-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-26 14:15:00 | 707.05 | 694.61 | 693.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 09:15:00 | 738.00 | 705.28 | 698.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 09:15:00 | 705.05 | 726.80 | 716.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-30 09:15:00 | 705.05 | 726.80 | 716.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 705.05 | 726.80 | 716.22 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 15:15:00 | 705.05 | 710.75 | 711.36 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 718.60 | 712.32 | 712.02 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 710.40 | 713.94 | 714.33 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 718.00 | 714.59 | 714.51 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 10:15:00 | 712.20 | 715.13 | 715.19 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 718.00 | 715.71 | 715.44 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 711.35 | 714.84 | 715.07 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 13:15:00 | 718.95 | 715.66 | 715.42 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 14:15:00 | 712.00 | 714.93 | 715.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 10:15:00 | 708.85 | 712.74 | 713.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 711.10 | 706.09 | 708.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 711.10 | 706.09 | 708.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 711.10 | 706.09 | 708.17 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 13:15:00 | 711.90 | 709.80 | 709.56 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 14:15:00 | 707.50 | 709.34 | 709.37 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 09:15:00 | 717.00 | 710.51 | 709.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 10:15:00 | 720.00 | 712.41 | 710.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 12:15:00 | 728.65 | 729.58 | 723.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 12:15:00 | 731.00 | 737.09 | 731.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 12:15:00 | 731.00 | 737.09 | 731.43 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 09:15:00 | 727.00 | 736.17 | 736.78 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 10:15:00 | 758.55 | 736.46 | 735.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 09:15:00 | 796.45 | 760.34 | 748.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 09:15:00 | 761.85 | 775.60 | 764.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 761.85 | 775.60 | 764.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 761.85 | 775.60 | 764.66 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 11:15:00 | 755.55 | 760.53 | 761.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 12:15:00 | 754.00 | 759.22 | 760.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 14:15:00 | 757.00 | 751.47 | 754.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 14:15:00 | 757.00 | 751.47 | 754.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 757.00 | 751.47 | 754.71 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 763.80 | 749.09 | 748.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 780.20 | 763.82 | 756.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 772.05 | 773.34 | 764.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 14:15:00 | 773.75 | 776.01 | 770.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 773.75 | 776.01 | 770.48 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 12:15:00 | 755.70 | 766.34 | 767.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 13:15:00 | 753.90 | 763.85 | 766.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 14:15:00 | 742.95 | 742.56 | 746.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 747.60 | 743.79 | 746.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 747.60 | 743.79 | 746.61 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2023-12-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 15:15:00 | 741.15 | 734.99 | 734.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 09:15:00 | 760.50 | 740.09 | 736.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 754.00 | 760.95 | 755.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 754.00 | 760.95 | 755.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 754.00 | 760.95 | 755.78 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 750.00 | 756.37 | 756.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 730.00 | 751.10 | 754.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 09:15:00 | 746.50 | 746.10 | 750.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 15:15:00 | 745.00 | 742.39 | 746.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 745.00 | 742.39 | 746.44 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 778.60 | 753.34 | 750.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 810.35 | 776.70 | 765.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 799.00 | 800.06 | 785.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 792.00 | 802.23 | 794.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 792.00 | 802.23 | 794.25 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 13:15:00 | 788.00 | 793.61 | 794.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 09:15:00 | 772.00 | 788.09 | 791.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 14:15:00 | 780.00 | 777.63 | 783.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 778.90 | 775.92 | 779.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 778.90 | 775.92 | 779.00 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 773.60 | 761.11 | 760.83 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 12:15:00 | 764.90 | 767.91 | 768.02 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 13:15:00 | 772.10 | 768.75 | 768.39 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 744.45 | 763.43 | 766.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 724.30 | 744.35 | 749.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 714.25 | 711.71 | 724.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 12:15:00 | 714.25 | 711.71 | 724.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 714.25 | 711.71 | 724.48 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 762.30 | 728.31 | 724.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 800.40 | 762.79 | 746.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 15:15:00 | 849.50 | 850.40 | 829.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 09:15:00 | 842.00 | 848.72 | 830.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 842.00 | 848.72 | 830.90 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2024-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 15:15:00 | 828.00 | 832.98 | 833.27 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 09:15:00 | 838.20 | 834.02 | 833.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 10:15:00 | 846.00 | 836.42 | 834.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 12:15:00 | 835.20 | 836.75 | 835.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 12:15:00 | 835.20 | 836.75 | 835.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 835.20 | 836.75 | 835.29 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 825.25 | 834.58 | 834.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 804.60 | 827.02 | 830.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 12:15:00 | 830.90 | 826.29 | 829.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 12:15:00 | 830.90 | 826.29 | 829.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 12:15:00 | 830.90 | 826.29 | 829.59 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 10:15:00 | 836.10 | 831.83 | 831.33 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 11:15:00 | 826.20 | 830.70 | 830.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 12:15:00 | 822.35 | 829.03 | 830.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 829.00 | 814.94 | 819.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 829.00 | 814.94 | 819.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 829.00 | 814.94 | 819.91 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 12:15:00 | 829.00 | 823.36 | 822.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 839.80 | 827.68 | 825.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 12:15:00 | 829.15 | 829.92 | 826.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 12:15:00 | 829.15 | 829.92 | 826.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 829.15 | 829.92 | 826.95 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 10:15:00 | 809.80 | 824.93 | 825.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-16 11:15:00 | 802.30 | 820.40 | 823.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 09:15:00 | 816.85 | 815.41 | 819.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 816.85 | 815.41 | 819.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 816.85 | 815.41 | 819.41 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 11:15:00 | 826.05 | 818.40 | 817.83 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 09:15:00 | 811.55 | 817.73 | 818.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 10:15:00 | 800.00 | 814.19 | 816.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 11:15:00 | 823.00 | 815.95 | 816.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 11:15:00 | 823.00 | 815.95 | 816.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 11:15:00 | 823.00 | 815.95 | 816.98 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 12:15:00 | 809.35 | 808.75 | 808.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 09:15:00 | 814.05 | 810.06 | 809.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 12:15:00 | 820.00 | 828.35 | 821.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 12:15:00 | 820.00 | 828.35 | 821.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 820.00 | 828.35 | 821.81 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 887.15 | 921.95 | 924.92 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 12:15:00 | 948.60 | 922.29 | 919.92 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 15:15:00 | 909.95 | 926.22 | 926.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 902.85 | 921.55 | 924.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 13:15:00 | 906.35 | 899.46 | 911.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 14:15:00 | 898.00 | 899.17 | 909.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 898.00 | 899.17 | 909.89 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 10:15:00 | 897.90 | 872.22 | 870.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 13:15:00 | 905.05 | 889.67 | 883.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 09:15:00 | 883.00 | 891.75 | 886.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 883.00 | 891.75 | 886.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 883.00 | 891.75 | 886.35 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 12:15:00 | 889.15 | 893.86 | 894.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 14:15:00 | 881.55 | 891.07 | 892.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 902.95 | 892.64 | 893.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 902.95 | 892.64 | 893.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 902.95 | 892.64 | 893.28 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 10:15:00 | 902.95 | 894.70 | 894.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 09:15:00 | 930.00 | 901.43 | 897.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 09:15:00 | 945.60 | 951.13 | 937.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 14:15:00 | 945.75 | 947.22 | 940.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 945.75 | 947.22 | 940.28 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 12:15:00 | 1010.55 | 1025.26 | 1026.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 1008.65 | 1021.94 | 1025.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 1008.00 | 1006.89 | 1013.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 1008.00 | 1006.89 | 1013.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1008.00 | 1006.89 | 1013.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 15:15:00 | 992.00 | 1005.95 | 1010.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 14:15:00 | 994.05 | 1004.59 | 1007.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 14:45:00 | 994.25 | 1002.47 | 1006.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 1035.00 | 1008.28 | 1008.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 09:15:00 | 1035.00 | 1008.28 | 1008.07 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 11:15:00 | 997.00 | 1010.96 | 1012.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 982.40 | 996.72 | 1004.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 14:15:00 | 985.75 | 983.71 | 993.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 14:30:00 | 982.25 | 983.71 | 993.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 1024.00 | 991.34 | 995.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:45:00 | 1019.90 | 991.34 | 995.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 10:15:00 | 1025.90 | 998.25 | 997.90 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 14:15:00 | 1003.20 | 1009.96 | 1010.84 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 11:15:00 | 1014.65 | 1011.72 | 1011.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 13:15:00 | 1022.40 | 1013.96 | 1012.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 10:15:00 | 1014.50 | 1019.23 | 1015.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 10:15:00 | 1014.50 | 1019.23 | 1015.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 1014.50 | 1019.23 | 1015.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 11:00:00 | 1014.50 | 1019.23 | 1015.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 1018.70 | 1019.13 | 1016.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 12:15:00 | 1028.90 | 1019.13 | 1016.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 1031.10 | 1020.98 | 1018.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 09:15:00 | 1033.90 | 1070.45 | 1072.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 1033.90 | 1070.45 | 1072.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 982.20 | 1031.23 | 1049.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 927.00 | 916.18 | 944.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 09:30:00 | 923.90 | 916.18 | 944.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 924.25 | 928.40 | 938.40 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 946.65 | 938.36 | 938.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 968.00 | 944.29 | 940.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 948.40 | 949.70 | 944.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 948.40 | 949.70 | 944.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 955.00 | 955.17 | 950.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 955.00 | 955.17 | 950.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 952.05 | 954.57 | 951.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 952.05 | 954.57 | 951.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 952.25 | 954.11 | 951.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 953.35 | 954.11 | 951.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 947.00 | 952.69 | 950.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 948.40 | 952.69 | 950.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 949.75 | 952.10 | 950.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 948.60 | 952.10 | 950.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 09:15:00 | 940.25 | 949.73 | 949.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 10:15:00 | 937.05 | 947.19 | 948.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 960.00 | 945.73 | 946.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 960.00 | 945.73 | 946.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 960.00 | 945.73 | 946.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:30:00 | 964.90 | 945.73 | 946.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 964.50 | 949.48 | 948.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 974.05 | 964.36 | 958.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 12:15:00 | 985.80 | 986.52 | 976.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 12:45:00 | 987.40 | 986.52 | 976.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 973.50 | 984.76 | 979.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 973.50 | 984.76 | 979.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 984.55 | 984.71 | 979.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:45:00 | 989.95 | 986.24 | 980.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 970.85 | 980.14 | 979.77 | SL hit (close<static) qty=1.00 sl=972.45 alert=retest2 |

### Cycle 88 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 959.20 | 975.96 | 977.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 11:15:00 | 955.70 | 971.90 | 975.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 937.20 | 933.57 | 943.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 13:45:00 | 934.40 | 933.57 | 943.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 917.60 | 928.95 | 935.33 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 962.95 | 939.94 | 938.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 15:15:00 | 971.95 | 946.34 | 941.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 953.95 | 975.66 | 963.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 953.95 | 975.66 | 963.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 953.95 | 975.66 | 963.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 941.00 | 975.66 | 963.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 941.00 | 968.73 | 961.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:30:00 | 941.00 | 968.73 | 961.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 13:15:00 | 941.00 | 955.20 | 956.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 893.95 | 938.86 | 948.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 936.05 | 913.47 | 926.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 936.05 | 913.47 | 926.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 936.05 | 913.47 | 926.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 936.05 | 913.47 | 926.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 937.95 | 918.36 | 927.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 941.00 | 918.36 | 927.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 922.05 | 920.71 | 927.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 919.90 | 919.58 | 926.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 978.30 | 931.46 | 929.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 978.30 | 931.46 | 929.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1000.00 | 969.23 | 953.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 994.00 | 994.62 | 982.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 994.00 | 994.62 | 982.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 1159.10 | 1180.50 | 1169.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 1166.85 | 1180.50 | 1169.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 1158.40 | 1176.08 | 1168.48 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 1144.00 | 1163.37 | 1163.98 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 1202.95 | 1168.49 | 1166.03 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 1161.05 | 1180.45 | 1181.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 1146.95 | 1173.75 | 1178.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 1172.80 | 1168.57 | 1174.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 1172.80 | 1168.57 | 1174.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1172.80 | 1168.57 | 1174.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 1172.80 | 1168.57 | 1174.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1195.00 | 1173.85 | 1176.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 1195.00 | 1173.85 | 1176.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1190.00 | 1177.08 | 1177.30 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 13:15:00 | 1187.20 | 1179.11 | 1178.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 1261.50 | 1195.40 | 1185.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 11:15:00 | 1394.45 | 1395.94 | 1350.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 12:00:00 | 1394.45 | 1395.94 | 1350.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1366.80 | 1407.45 | 1393.73 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 1364.35 | 1388.27 | 1389.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 10:15:00 | 1324.80 | 1375.58 | 1383.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 1328.85 | 1324.22 | 1342.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 15:00:00 | 1328.85 | 1324.22 | 1342.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1332.85 | 1327.03 | 1340.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 1320.75 | 1327.03 | 1340.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:30:00 | 1311.10 | 1318.71 | 1329.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 09:15:00 | 1360.95 | 1313.61 | 1319.21 | SL hit (close>static) qty=1.00 sl=1355.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 1360.95 | 1323.08 | 1323.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 1365.15 | 1351.86 | 1339.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 1355.30 | 1356.12 | 1345.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:30:00 | 1353.80 | 1356.12 | 1345.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1349.85 | 1353.89 | 1346.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 1354.05 | 1353.89 | 1346.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1364.90 | 1356.27 | 1348.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1365.85 | 1356.27 | 1348.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1350.00 | 1355.01 | 1348.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:30:00 | 1368.10 | 1357.96 | 1350.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 1341.35 | 1367.17 | 1361.15 | SL hit (close<static) qty=1.00 sl=1342.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 1641.35 | 1679.95 | 1682.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 14:15:00 | 1636.50 | 1660.61 | 1671.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1668.00 | 1659.11 | 1669.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1668.00 | 1659.11 | 1669.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1668.00 | 1659.11 | 1669.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 1615.80 | 1652.14 | 1664.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 1724.25 | 1649.34 | 1642.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 1724.25 | 1649.34 | 1642.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 1731.60 | 1702.09 | 1682.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 14:15:00 | 1720.00 | 1729.07 | 1710.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 15:00:00 | 1720.00 | 1729.07 | 1710.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1695.00 | 1720.04 | 1709.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:45:00 | 1685.45 | 1720.04 | 1709.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1691.80 | 1714.39 | 1707.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 1689.10 | 1714.39 | 1707.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 1681.90 | 1703.56 | 1703.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1660.70 | 1694.99 | 1699.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 1664.50 | 1661.36 | 1677.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:00:00 | 1664.50 | 1661.36 | 1677.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1706.00 | 1671.92 | 1677.56 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1698.55 | 1682.15 | 1681.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 1718.00 | 1700.88 | 1692.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 1702.85 | 1705.40 | 1697.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 14:15:00 | 1702.85 | 1705.40 | 1697.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 1702.85 | 1705.40 | 1697.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 1702.85 | 1705.40 | 1697.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 1702.00 | 1704.72 | 1697.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 1711.35 | 1704.72 | 1697.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 1686.15 | 1701.01 | 1696.67 | SL hit (close<static) qty=1.00 sl=1695.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 11:15:00 | 1682.90 | 1693.38 | 1693.71 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 14:15:00 | 1698.15 | 1693.70 | 1693.70 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 15:15:00 | 1692.00 | 1693.36 | 1693.54 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 1703.90 | 1695.47 | 1694.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 1725.65 | 1701.50 | 1697.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 1753.85 | 1754.77 | 1731.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 1753.85 | 1754.77 | 1731.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1753.85 | 1754.77 | 1731.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:45:00 | 1863.75 | 1784.97 | 1765.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 1876.70 | 1784.97 | 1765.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 13:15:00 | 1798.00 | 1826.64 | 1828.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 13:15:00 | 1798.00 | 1826.64 | 1828.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 1737.20 | 1806.58 | 1818.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 10:15:00 | 1692.40 | 1690.08 | 1710.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-03 10:30:00 | 1696.25 | 1690.08 | 1710.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1688.00 | 1685.29 | 1698.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:30:00 | 1692.05 | 1685.29 | 1698.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1669.65 | 1667.60 | 1681.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:15:00 | 1654.00 | 1667.60 | 1681.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1661.05 | 1664.16 | 1672.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 1578.00 | 1627.10 | 1646.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 1601.25 | 1599.61 | 1619.62 | SL hit (close>ema200) qty=0.50 sl=1599.61 alert=retest2 |

### Cycle 107 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 1620.10 | 1615.79 | 1615.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1632.85 | 1620.96 | 1618.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 1692.85 | 1694.31 | 1680.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 09:30:00 | 1696.90 | 1694.31 | 1680.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1675.00 | 1690.00 | 1681.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:45:00 | 1679.10 | 1690.00 | 1681.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 1671.00 | 1686.20 | 1680.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 1671.00 | 1686.20 | 1680.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 1670.00 | 1680.79 | 1678.95 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 1641.50 | 1672.93 | 1675.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 1615.55 | 1661.46 | 1670.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1680.50 | 1645.39 | 1657.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 1680.50 | 1645.39 | 1657.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 1680.50 | 1645.39 | 1657.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 1680.50 | 1645.39 | 1657.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 1680.00 | 1652.31 | 1659.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 1691.90 | 1652.31 | 1659.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 1687.00 | 1665.44 | 1664.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 1703.25 | 1676.84 | 1670.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 1692.05 | 1692.17 | 1681.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 12:00:00 | 1692.05 | 1692.17 | 1681.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1707.00 | 1704.62 | 1695.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 1723.20 | 1704.69 | 1696.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 1693.25 | 1703.16 | 1698.12 | SL hit (close<static) qty=1.00 sl=1695.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 1678.40 | 1697.67 | 1697.96 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 13:15:00 | 1701.80 | 1698.49 | 1698.31 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 14:15:00 | 1692.00 | 1697.19 | 1697.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 15:15:00 | 1674.00 | 1692.56 | 1695.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 1696.95 | 1686.60 | 1690.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 1696.95 | 1686.60 | 1690.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1696.95 | 1686.60 | 1690.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 1696.95 | 1686.60 | 1690.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 1750.00 | 1699.28 | 1695.93 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1688.35 | 1694.06 | 1694.11 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 12:15:00 | 1699.10 | 1695.07 | 1694.56 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 1677.00 | 1701.98 | 1702.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 1650.75 | 1662.24 | 1675.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 14:15:00 | 1672.95 | 1662.83 | 1673.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 1672.95 | 1662.83 | 1673.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1672.95 | 1662.83 | 1673.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 1672.95 | 1662.83 | 1673.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1665.00 | 1663.26 | 1672.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1648.00 | 1663.26 | 1672.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 1689.40 | 1646.90 | 1644.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 1689.40 | 1646.90 | 1644.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 1705.50 | 1658.62 | 1649.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 11:15:00 | 1884.45 | 1892.60 | 1827.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 12:15:00 | 1888.80 | 1892.60 | 1827.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1923.10 | 1934.49 | 1913.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 1920.00 | 1934.49 | 1913.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1876.75 | 1918.29 | 1913.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1876.75 | 1918.29 | 1913.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1878.85 | 1910.40 | 1910.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:15:00 | 1868.55 | 1910.40 | 1910.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 1882.85 | 1904.89 | 1907.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 1857.25 | 1880.19 | 1890.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 1784.00 | 1777.52 | 1807.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 13:15:00 | 1784.00 | 1777.52 | 1807.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1784.00 | 1777.52 | 1807.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 1784.00 | 1777.52 | 1807.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1718.45 | 1701.95 | 1727.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 1699.00 | 1709.45 | 1721.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 1764.60 | 1727.01 | 1725.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 1764.60 | 1727.01 | 1725.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1822.45 | 1750.60 | 1737.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1846.20 | 1855.85 | 1826.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 1846.20 | 1855.85 | 1826.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1872.40 | 1867.53 | 1848.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:45:00 | 1887.15 | 1867.20 | 1853.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 1911.70 | 1871.45 | 1857.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 1885.55 | 1890.31 | 1880.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:00:00 | 1883.95 | 1887.59 | 1880.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1875.50 | 1885.17 | 1880.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 1875.50 | 1885.17 | 1880.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1869.00 | 1881.94 | 1879.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1871.85 | 1881.94 | 1879.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1866.80 | 1878.43 | 1878.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:30:00 | 1867.75 | 1878.43 | 1878.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 1848.10 | 1872.36 | 1875.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1848.10 | 1872.36 | 1875.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 1824.15 | 1856.32 | 1866.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1755.00 | 1733.78 | 1763.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1755.00 | 1733.78 | 1763.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1755.00 | 1733.78 | 1763.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1755.00 | 1733.78 | 1763.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1758.20 | 1738.67 | 1763.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1765.45 | 1738.67 | 1763.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1784.00 | 1747.73 | 1765.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 1784.00 | 1747.73 | 1765.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1757.00 | 1749.59 | 1764.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 1742.00 | 1747.07 | 1761.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1744.20 | 1753.57 | 1759.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1797.90 | 1762.90 | 1762.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 1797.90 | 1762.90 | 1762.39 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 11:15:00 | 1761.40 | 1772.81 | 1773.54 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1817.50 | 1778.68 | 1775.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 1877.35 | 1829.05 | 1811.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 09:15:00 | 1887.05 | 1891.20 | 1858.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:00:00 | 1887.05 | 1891.20 | 1858.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1897.85 | 1897.85 | 1887.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:30:00 | 1895.75 | 1897.85 | 1887.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1833.90 | 1890.87 | 1889.37 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 10:15:00 | 1823.65 | 1877.43 | 1883.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 14:15:00 | 1803.85 | 1838.82 | 1861.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 1838.50 | 1834.77 | 1855.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 1838.50 | 1834.77 | 1855.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1838.50 | 1834.77 | 1855.07 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 1891.85 | 1855.56 | 1851.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 1951.00 | 1880.48 | 1863.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 2077.40 | 2078.69 | 2046.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 12:00:00 | 2077.40 | 2078.69 | 2046.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 2328.15 | 2367.28 | 2319.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:00:00 | 2328.15 | 2367.28 | 2319.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 2322.00 | 2358.22 | 2319.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:45:00 | 2308.85 | 2358.22 | 2319.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 2360.30 | 2358.64 | 2323.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 15:00:00 | 2400.25 | 2366.96 | 2330.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 2441.10 | 2366.57 | 2333.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 2478.95 | 2418.79 | 2396.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 15:15:00 | 2461.00 | 2490.88 | 2494.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 2461.00 | 2490.88 | 2494.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 09:15:00 | 2368.50 | 2466.41 | 2482.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 13:15:00 | 2447.20 | 2397.62 | 2423.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 13:15:00 | 2447.20 | 2397.62 | 2423.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 2447.20 | 2397.62 | 2423.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 2447.20 | 2397.62 | 2423.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 2429.05 | 2403.91 | 2423.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 2429.05 | 2403.91 | 2423.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 2425.00 | 2408.13 | 2423.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 2419.90 | 2408.13 | 2423.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 2353.30 | 2397.16 | 2417.57 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 2480.00 | 2426.78 | 2422.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 2538.50 | 2497.61 | 2475.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 2528.00 | 2530.78 | 2506.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:15:00 | 2500.30 | 2530.78 | 2506.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2526.00 | 2529.83 | 2507.93 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 2434.70 | 2487.15 | 2492.41 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 2499.00 | 2490.90 | 2490.79 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 2463.95 | 2488.96 | 2490.41 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 10:15:00 | 2553.95 | 2495.35 | 2488.91 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 2456.00 | 2490.93 | 2495.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 2434.00 | 2479.54 | 2489.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 11:15:00 | 2210.10 | 2187.58 | 2250.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 11:45:00 | 2205.00 | 2187.58 | 2250.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 2242.00 | 2187.86 | 2225.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 2248.00 | 2187.86 | 2225.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 2241.00 | 2198.49 | 2227.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:45:00 | 2261.05 | 2198.49 | 2227.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 2242.00 | 2220.26 | 2231.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 2214.45 | 2227.62 | 2232.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 12:15:00 | 2270.60 | 2221.07 | 2219.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 2270.60 | 2221.07 | 2219.34 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 13:15:00 | 2205.00 | 2217.86 | 2218.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 2158.95 | 2203.06 | 2210.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1668.00 | 1633.85 | 1708.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1668.00 | 1633.85 | 1708.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1668.00 | 1633.85 | 1708.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 1660.75 | 1633.85 | 1708.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 1696.95 | 1656.74 | 1692.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 1696.95 | 1656.74 | 1692.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 1694.00 | 1664.19 | 1692.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 1654.90 | 1664.19 | 1692.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1722.00 | 1675.75 | 1694.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 1727.60 | 1675.75 | 1694.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1717.85 | 1684.17 | 1696.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:45:00 | 1683.95 | 1689.40 | 1696.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 1756.30 | 1700.51 | 1699.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 1756.30 | 1700.51 | 1699.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 1859.95 | 1754.94 | 1729.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1751.95 | 1773.67 | 1746.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1751.95 | 1773.67 | 1746.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1751.95 | 1773.67 | 1746.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1768.15 | 1773.67 | 1746.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1792.00 | 1777.34 | 1750.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1792.05 | 1777.34 | 1750.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 1819.70 | 1785.81 | 1756.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1677.20 | 1765.24 | 1752.48 | SL hit (close<static) qty=1.00 sl=1725.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 1667.90 | 1733.63 | 1739.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 13:15:00 | 1653.15 | 1707.34 | 1725.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 15:15:00 | 1649.95 | 1649.13 | 1676.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 09:15:00 | 1696.95 | 1649.13 | 1676.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1693.20 | 1657.94 | 1677.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 1697.65 | 1657.94 | 1677.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1694.00 | 1665.15 | 1679.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 12:45:00 | 1679.20 | 1672.86 | 1680.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:15:00 | 1595.24 | 1635.91 | 1648.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-10 14:15:00 | 1621.70 | 1608.59 | 1627.94 | SL hit (close>ema200) qty=0.50 sl=1608.59 alert=retest2 |

### Cycle 137 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 1114.75 | 1096.39 | 1094.97 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 1074.35 | 1094.23 | 1096.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 1051.00 | 1085.59 | 1090.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 1064.00 | 1031.06 | 1049.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 1064.00 | 1031.06 | 1049.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 1064.00 | 1031.06 | 1049.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 1064.00 | 1031.06 | 1049.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 1066.65 | 1038.18 | 1051.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:00:00 | 1037.25 | 1037.99 | 1050.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 11:15:00 | 1068.65 | 1056.60 | 1055.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 1068.65 | 1056.60 | 1055.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 1110.95 | 1076.05 | 1066.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1228.00 | 1240.22 | 1203.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 10:00:00 | 1228.00 | 1240.22 | 1203.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1202.35 | 1225.67 | 1205.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 1202.35 | 1225.67 | 1205.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 1220.00 | 1224.53 | 1206.79 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 1191.95 | 1198.31 | 1198.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 1175.05 | 1191.01 | 1194.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 1201.80 | 1185.48 | 1189.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 1201.80 | 1185.48 | 1189.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1201.80 | 1185.48 | 1189.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1201.80 | 1185.48 | 1189.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 15:15:00 | 1221.00 | 1192.58 | 1192.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 1242.50 | 1202.57 | 1196.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 1211.75 | 1213.29 | 1205.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 14:45:00 | 1215.15 | 1213.29 | 1205.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1228.45 | 1216.28 | 1208.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 1248.00 | 1227.37 | 1217.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 13:45:00 | 1246.10 | 1240.82 | 1229.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 10:15:00 | 1372.80 | 1309.92 | 1283.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 1446.35 | 1460.89 | 1462.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 1430.50 | 1451.56 | 1457.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 1435.00 | 1433.62 | 1444.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 12:15:00 | 1435.00 | 1433.62 | 1444.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1435.00 | 1433.62 | 1444.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:30:00 | 1442.40 | 1433.62 | 1444.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1444.70 | 1436.21 | 1444.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:45:00 | 1445.40 | 1436.21 | 1444.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1445.00 | 1437.97 | 1444.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 1454.40 | 1437.97 | 1444.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1448.30 | 1440.03 | 1444.51 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 1455.75 | 1448.69 | 1447.73 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1391.75 | 1437.64 | 1442.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 1388.05 | 1411.79 | 1427.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 14:15:00 | 1449.90 | 1419.41 | 1429.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 14:15:00 | 1449.90 | 1419.41 | 1429.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 1449.90 | 1419.41 | 1429.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 15:00:00 | 1449.90 | 1419.41 | 1429.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 1446.00 | 1424.73 | 1431.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 1377.20 | 1424.73 | 1431.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 1437.00 | 1408.66 | 1406.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 1437.00 | 1408.66 | 1406.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 14:15:00 | 1452.00 | 1424.56 | 1415.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 15:15:00 | 1455.10 | 1455.87 | 1440.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:15:00 | 1477.70 | 1455.87 | 1440.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1504.30 | 1506.10 | 1493.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1509.00 | 1506.10 | 1493.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 1505.20 | 1505.92 | 1494.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 09:15:00 | 1487.20 | 1498.47 | 1494.74 | SL hit (close<ema400) qty=1.00 sl=1494.74 alert=retest1 |

### Cycle 146 — SELL (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 12:15:00 | 1475.00 | 1489.29 | 1491.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 13:15:00 | 1469.70 | 1485.37 | 1489.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 10:15:00 | 1444.50 | 1442.05 | 1453.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 10:45:00 | 1443.00 | 1442.05 | 1453.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 1419.00 | 1392.15 | 1413.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:00:00 | 1419.00 | 1392.15 | 1413.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 1417.00 | 1397.12 | 1413.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 09:15:00 | 1400.10 | 1397.12 | 1413.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 1449.00 | 1407.49 | 1416.82 | SL hit (close>static) qty=1.00 sl=1420.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 1457.90 | 1428.07 | 1425.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 1479.40 | 1447.50 | 1436.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 1446.40 | 1459.37 | 1449.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1446.40 | 1459.37 | 1449.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1446.40 | 1459.37 | 1449.54 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 1422.90 | 1441.43 | 1443.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 1407.00 | 1434.54 | 1440.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 09:15:00 | 1393.20 | 1383.21 | 1395.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 1393.20 | 1383.21 | 1395.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1393.20 | 1383.21 | 1395.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 1391.10 | 1383.21 | 1395.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1388.70 | 1384.31 | 1395.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 1379.00 | 1382.82 | 1393.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:45:00 | 1376.00 | 1381.20 | 1391.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 1404.00 | 1369.11 | 1371.81 | SL hit (close>static) qty=1.00 sl=1396.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1398.50 | 1374.99 | 1374.24 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 1342.20 | 1368.89 | 1372.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 1337.00 | 1362.51 | 1368.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 09:15:00 | 1406.40 | 1371.29 | 1372.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 1406.40 | 1371.29 | 1372.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1406.40 | 1371.29 | 1372.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:30:00 | 1406.40 | 1371.29 | 1372.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 10:15:00 | 1406.40 | 1378.31 | 1375.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 1476.70 | 1413.10 | 1395.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 15:15:00 | 1898.00 | 1901.26 | 1853.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:15:00 | 1816.30 | 1901.26 | 1853.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1870.00 | 1895.01 | 1855.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1850.00 | 1895.01 | 1855.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1865.00 | 1889.00 | 1855.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 1857.40 | 1889.00 | 1855.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1842.10 | 1879.62 | 1854.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:45:00 | 1846.30 | 1879.62 | 1854.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1887.00 | 1881.10 | 1857.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:45:00 | 1944.00 | 1888.56 | 1882.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:45:00 | 1916.00 | 1893.45 | 1885.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:30:00 | 1916.10 | 1899.10 | 1888.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 13:00:00 | 1916.00 | 1902.48 | 1891.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 2107.60 | 1996.40 | 1950.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 2084.80 | 2160.13 | 2166.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 09:15:00 | 2032.30 | 2095.19 | 2127.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 2038.00 | 2035.95 | 2074.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:45:00 | 2045.50 | 2035.95 | 2074.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1967.00 | 2015.47 | 2044.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 1962.30 | 1996.60 | 2030.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 1961.30 | 1988.40 | 2023.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1945.10 | 1975.24 | 2011.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 1956.00 | 1968.43 | 2001.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1989.60 | 1941.07 | 1967.22 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1864.18 | 1941.07 | 1967.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1989.60 | 1941.07 | 1967.22 | SL hit (close>static) qty=0.50 sl=1941.07 alert=retest2 |

### Cycle 153 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1975.00 | 1970.72 | 1970.69 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 10:15:00 | 1955.80 | 1968.24 | 1969.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 1938.00 | 1962.19 | 1966.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 1966.40 | 1954.85 | 1960.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1966.40 | 1954.85 | 1960.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1966.40 | 1954.85 | 1960.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 1971.80 | 1954.85 | 1960.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1949.00 | 1953.68 | 1959.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 1934.10 | 1952.34 | 1957.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 1935.00 | 1948.88 | 1955.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:00:00 | 1932.00 | 1943.60 | 1951.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1979.10 | 1917.97 | 1921.11 | SL hit (close>static) qty=1.00 sl=1966.40 alert=retest2 |

### Cycle 155 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1994.60 | 1933.30 | 1927.79 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 1922.40 | 1942.90 | 1942.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 09:15:00 | 1883.70 | 1926.54 | 1934.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 10:15:00 | 1909.70 | 1903.17 | 1915.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 10:15:00 | 1909.70 | 1903.17 | 1915.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 1909.70 | 1903.17 | 1915.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 1909.70 | 1903.17 | 1915.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1911.00 | 1904.73 | 1914.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 1911.00 | 1904.73 | 1914.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1900.20 | 1903.83 | 1913.42 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 15:15:00 | 1929.80 | 1914.43 | 1913.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 1968.50 | 1925.24 | 1918.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 11:15:00 | 1973.90 | 1974.38 | 1955.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 12:00:00 | 1973.90 | 1974.38 | 1955.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1953.50 | 1970.19 | 1960.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1953.50 | 1970.19 | 1960.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1954.80 | 1967.11 | 1959.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 1954.80 | 1967.11 | 1959.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 1936.60 | 1952.91 | 1954.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 1923.60 | 1938.82 | 1946.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 1961.90 | 1939.76 | 1944.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 1961.90 | 1939.76 | 1944.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1961.90 | 1939.76 | 1944.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 1972.00 | 1939.76 | 1944.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1953.20 | 1942.45 | 1945.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 1942.00 | 1943.74 | 1945.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 12:15:00 | 1941.10 | 1943.74 | 1945.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 1942.20 | 1930.08 | 1933.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 1953.00 | 1934.23 | 1933.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1953.00 | 1934.23 | 1933.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 1982.10 | 1943.80 | 1937.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 1952.90 | 1954.31 | 1945.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 15:00:00 | 1952.90 | 1954.31 | 1945.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1912.00 | 1946.72 | 1944.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1912.00 | 1946.72 | 1944.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 1911.00 | 1939.58 | 1941.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 1907.00 | 1933.06 | 1937.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1848.60 | 1832.32 | 1854.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1848.60 | 1832.32 | 1854.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1848.60 | 1832.32 | 1854.13 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 1853.40 | 1842.47 | 1842.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 1872.00 | 1851.10 | 1846.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 1850.10 | 1850.90 | 1846.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 10:15:00 | 1850.10 | 1850.90 | 1846.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1850.10 | 1850.90 | 1846.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 1851.70 | 1850.90 | 1846.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1850.30 | 1850.78 | 1846.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 1845.10 | 1850.78 | 1846.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1847.30 | 1850.09 | 1846.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 1844.00 | 1850.09 | 1846.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1854.50 | 1850.97 | 1847.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:15:00 | 1856.70 | 1850.97 | 1847.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 1859.50 | 1858.87 | 1852.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:00:00 | 1856.60 | 1858.42 | 1853.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1900.80 | 1864.19 | 1857.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1876.00 | 1882.25 | 1874.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:00:00 | 1880.80 | 1881.96 | 1875.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 14:00:00 | 1883.10 | 1881.89 | 1876.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 1870.00 | 1876.94 | 1877.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 15:15:00 | 1861.00 | 1873.75 | 1875.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 1518.00 | 1495.74 | 1543.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:45:00 | 1528.70 | 1495.74 | 1543.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1511.00 | 1522.18 | 1537.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:45:00 | 1483.50 | 1496.69 | 1512.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 1409.33 | 1441.51 | 1465.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 1459.00 | 1435.13 | 1450.12 | SL hit (close>ema200) qty=0.50 sl=1435.13 alert=retest2 |

### Cycle 163 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1441.10 | 1415.76 | 1415.08 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 1419.40 | 1425.89 | 1426.26 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 1437.30 | 1428.05 | 1427.09 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 1419.80 | 1425.30 | 1426.00 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 1461.70 | 1431.54 | 1428.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 14:15:00 | 1492.60 | 1452.82 | 1440.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1490.20 | 1501.26 | 1478.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1490.20 | 1501.26 | 1478.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1490.20 | 1501.26 | 1478.93 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 1461.80 | 1481.55 | 1481.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 15:15:00 | 1450.00 | 1475.24 | 1479.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 14:15:00 | 1461.20 | 1458.94 | 1468.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 15:00:00 | 1461.20 | 1458.94 | 1468.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1477.50 | 1463.11 | 1468.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1486.10 | 1463.11 | 1468.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1478.60 | 1466.21 | 1469.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1478.60 | 1466.21 | 1469.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 1501.00 | 1475.63 | 1473.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 1523.90 | 1499.82 | 1487.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1497.30 | 1499.32 | 1488.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 1497.30 | 1499.32 | 1488.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1492.00 | 1498.69 | 1490.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:15:00 | 1488.90 | 1498.69 | 1490.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1481.20 | 1495.19 | 1489.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 1483.10 | 1495.19 | 1489.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1484.50 | 1493.05 | 1489.33 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 1481.10 | 1486.48 | 1487.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 1478.00 | 1484.78 | 1486.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 12:15:00 | 1482.90 | 1481.60 | 1484.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 1482.90 | 1481.60 | 1484.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1482.90 | 1481.60 | 1484.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 1486.40 | 1481.60 | 1484.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1470.00 | 1477.15 | 1481.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 1458.00 | 1472.46 | 1476.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 1464.20 | 1455.39 | 1459.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 1479.40 | 1462.95 | 1462.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 1479.40 | 1462.95 | 1462.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 1499.30 | 1470.22 | 1465.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1477.90 | 1478.07 | 1470.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 1477.90 | 1478.07 | 1470.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1472.40 | 1477.31 | 1471.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:00:00 | 1472.40 | 1477.31 | 1471.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1463.50 | 1474.55 | 1470.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 1463.50 | 1474.55 | 1470.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1463.70 | 1472.38 | 1469.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 1463.90 | 1472.38 | 1469.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 1463.80 | 1467.88 | 1468.13 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 1470.10 | 1468.49 | 1468.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1491.50 | 1480.49 | 1475.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 1488.50 | 1489.41 | 1481.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:00:00 | 1488.50 | 1489.41 | 1481.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1483.90 | 1490.14 | 1484.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1475.20 | 1490.14 | 1484.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1480.70 | 1488.25 | 1483.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1480.70 | 1488.25 | 1483.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1479.60 | 1486.52 | 1483.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:45:00 | 1476.00 | 1486.52 | 1483.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1470.00 | 1480.34 | 1481.22 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1493.80 | 1482.10 | 1481.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1533.40 | 1493.96 | 1487.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1591.00 | 1597.87 | 1563.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 15:15:00 | 1602.00 | 1605.48 | 1599.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1602.00 | 1605.48 | 1599.57 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1568.40 | 1592.11 | 1594.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 1540.70 | 1575.55 | 1585.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 15:15:00 | 1505.30 | 1502.47 | 1525.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:15:00 | 1503.40 | 1502.47 | 1525.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1463.70 | 1443.43 | 1461.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 1466.20 | 1443.43 | 1461.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1482.30 | 1451.21 | 1463.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 1482.30 | 1451.21 | 1463.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1482.60 | 1457.48 | 1465.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:45:00 | 1487.20 | 1457.48 | 1465.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1469.70 | 1461.27 | 1465.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 1469.70 | 1461.27 | 1465.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1466.60 | 1462.34 | 1465.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:15:00 | 1475.00 | 1462.34 | 1465.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1475.00 | 1464.87 | 1466.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1491.20 | 1464.87 | 1466.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1485.00 | 1468.90 | 1468.24 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1447.20 | 1468.66 | 1470.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 1441.00 | 1454.55 | 1461.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 1446.70 | 1444.93 | 1452.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 1446.70 | 1444.93 | 1452.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1410.40 | 1413.68 | 1424.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 1428.50 | 1413.68 | 1424.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1427.00 | 1410.97 | 1418.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 1427.00 | 1410.97 | 1418.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1419.90 | 1412.76 | 1418.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1420.00 | 1412.76 | 1418.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1401.90 | 1410.59 | 1417.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:30:00 | 1397.40 | 1407.07 | 1414.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 1378.00 | 1372.19 | 1371.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1378.00 | 1372.19 | 1371.72 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 1363.50 | 1370.11 | 1371.00 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 1375.00 | 1371.59 | 1371.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1400.60 | 1377.39 | 1374.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1382.40 | 1387.67 | 1382.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 13:15:00 | 1382.40 | 1387.67 | 1382.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1382.40 | 1387.67 | 1382.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 1382.40 | 1387.67 | 1382.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1387.30 | 1387.60 | 1382.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1390.90 | 1387.08 | 1382.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 1329.90 | 1382.21 | 1384.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 1329.90 | 1382.21 | 1384.93 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 1359.00 | 1352.62 | 1352.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 1377.60 | 1357.62 | 1354.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1364.90 | 1369.30 | 1363.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 1364.90 | 1369.30 | 1363.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1364.90 | 1369.30 | 1363.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1364.90 | 1369.30 | 1363.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1358.20 | 1367.08 | 1362.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 1359.50 | 1367.08 | 1362.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 1363.90 | 1366.44 | 1363.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1401.80 | 1362.43 | 1361.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 1370.50 | 1395.19 | 1393.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1369.90 | 1390.13 | 1391.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 1369.90 | 1390.13 | 1391.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1349.40 | 1375.44 | 1383.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1362.00 | 1361.38 | 1371.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:15:00 | 1359.60 | 1361.38 | 1371.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1363.70 | 1361.84 | 1370.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1368.50 | 1361.84 | 1370.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1371.80 | 1363.83 | 1370.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1365.50 | 1363.83 | 1370.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1366.40 | 1364.35 | 1370.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:30:00 | 1361.70 | 1363.04 | 1369.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 1363.90 | 1364.00 | 1368.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 1357.00 | 1364.00 | 1368.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1382.80 | 1366.64 | 1368.80 | SL hit (close>static) qty=1.00 sl=1371.90 alert=retest2 |

### Cycle 185 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 1389.50 | 1371.21 | 1370.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 1411.00 | 1395.25 | 1387.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1392.30 | 1396.41 | 1390.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:45:00 | 1394.80 | 1396.41 | 1390.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1385.70 | 1394.27 | 1389.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 1385.70 | 1394.27 | 1389.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1371.10 | 1389.64 | 1388.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1371.10 | 1389.64 | 1388.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 1368.00 | 1385.31 | 1386.43 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 1433.40 | 1394.93 | 1390.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 1452.90 | 1406.52 | 1396.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 13:15:00 | 1450.10 | 1452.37 | 1433.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:00:00 | 1450.10 | 1452.37 | 1433.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1407.00 | 1442.65 | 1433.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1407.00 | 1442.65 | 1433.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1407.90 | 1435.70 | 1431.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1410.00 | 1435.70 | 1431.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1406.30 | 1425.71 | 1427.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1398.80 | 1417.59 | 1423.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1452.80 | 1411.05 | 1412.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1452.80 | 1411.05 | 1412.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1452.80 | 1411.05 | 1412.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1452.80 | 1411.05 | 1412.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1434.00 | 1415.64 | 1414.64 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 1404.00 | 1418.80 | 1420.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 1377.70 | 1406.95 | 1414.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1390.30 | 1387.48 | 1398.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:30:00 | 1395.30 | 1387.48 | 1398.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1391.40 | 1388.26 | 1397.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 1391.40 | 1388.26 | 1397.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1395.20 | 1389.65 | 1397.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:00:00 | 1395.20 | 1389.65 | 1397.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 1394.40 | 1390.60 | 1397.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:30:00 | 1397.00 | 1390.60 | 1397.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1401.80 | 1392.84 | 1397.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 1401.80 | 1392.84 | 1397.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1397.70 | 1393.81 | 1397.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1397.70 | 1393.81 | 1397.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1396.50 | 1394.35 | 1397.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 1433.40 | 1394.35 | 1397.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1415.00 | 1398.48 | 1399.10 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 1414.70 | 1401.72 | 1400.52 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1401.90 | 1405.97 | 1406.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 13:15:00 | 1398.90 | 1403.82 | 1405.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 1405.50 | 1404.16 | 1405.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 1405.50 | 1404.16 | 1405.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1405.50 | 1404.16 | 1405.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 1405.50 | 1404.16 | 1405.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1403.60 | 1404.05 | 1405.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1410.50 | 1404.05 | 1405.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1400.60 | 1403.36 | 1404.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:15:00 | 1396.90 | 1401.13 | 1403.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 1397.00 | 1400.30 | 1402.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:45:00 | 1396.40 | 1400.24 | 1402.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1392.20 | 1400.24 | 1402.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1380.00 | 1394.91 | 1399.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:45:00 | 1375.00 | 1386.24 | 1391.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 1373.90 | 1384.19 | 1390.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:45:00 | 1374.20 | 1380.79 | 1387.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 10:15:00 | 1375.50 | 1379.03 | 1384.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1380.80 | 1379.39 | 1384.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1381.40 | 1379.39 | 1384.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1401.00 | 1381.28 | 1382.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 1386.80 | 1381.28 | 1382.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 1396.00 | 1384.22 | 1383.74 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1368.50 | 1384.74 | 1385.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 1365.20 | 1380.83 | 1383.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 1367.60 | 1367.16 | 1374.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:15:00 | 1336.20 | 1367.16 | 1374.43 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1367.00 | 1351.91 | 1359.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1367.00 | 1351.91 | 1359.41 | SL hit (close>ema400) qty=1.00 sl=1359.41 alert=retest1 |

### Cycle 195 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1374.00 | 1357.80 | 1356.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1392.00 | 1368.45 | 1362.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 1372.10 | 1372.32 | 1366.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:45:00 | 1373.90 | 1372.32 | 1366.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1368.80 | 1371.99 | 1368.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 1368.80 | 1371.99 | 1368.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1367.50 | 1371.09 | 1368.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 1359.10 | 1371.09 | 1368.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1359.10 | 1368.69 | 1367.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1362.40 | 1368.69 | 1367.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1356.90 | 1366.33 | 1366.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1356.90 | 1366.33 | 1366.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1356.10 | 1364.29 | 1365.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 1351.20 | 1358.99 | 1361.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1363.00 | 1355.64 | 1358.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 1363.00 | 1355.64 | 1358.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1363.00 | 1355.64 | 1358.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 1363.50 | 1355.64 | 1358.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1360.30 | 1356.57 | 1359.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1352.30 | 1356.57 | 1359.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1365.90 | 1354.24 | 1353.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1365.90 | 1354.24 | 1353.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1373.10 | 1360.05 | 1356.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 1389.30 | 1394.35 | 1384.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 1389.30 | 1394.35 | 1384.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1390.00 | 1394.79 | 1390.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1401.00 | 1394.79 | 1390.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1395.00 | 1394.83 | 1390.71 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 1378.90 | 1388.65 | 1389.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 1363.80 | 1379.30 | 1384.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 1388.90 | 1379.04 | 1383.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 1388.90 | 1379.04 | 1383.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1388.90 | 1379.04 | 1383.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1388.90 | 1379.04 | 1383.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1394.40 | 1382.11 | 1384.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1374.90 | 1382.11 | 1384.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1360.20 | 1360.34 | 1369.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1349.20 | 1359.07 | 1367.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 1373.00 | 1360.50 | 1364.79 | SL hit (close>static) qty=1.00 sl=1370.00 alert=retest2 |

### Cycle 199 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1384.20 | 1365.16 | 1362.96 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1351.70 | 1363.49 | 1364.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1346.20 | 1358.26 | 1361.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 1331.00 | 1330.38 | 1341.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 1320.00 | 1327.36 | 1334.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1320.00 | 1327.36 | 1334.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:00:00 | 1306.90 | 1320.91 | 1329.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1241.56 | 1301.90 | 1317.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 1247.70 | 1246.59 | 1265.84 | SL hit (close>ema200) qty=0.50 sl=1246.59 alert=retest2 |

### Cycle 201 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 1322.10 | 1269.21 | 1265.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 15:15:00 | 1336.60 | 1311.06 | 1297.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1302.00 | 1309.24 | 1297.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 1302.00 | 1309.24 | 1297.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1295.20 | 1306.44 | 1297.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 1295.20 | 1306.44 | 1297.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1288.10 | 1302.77 | 1296.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 1288.10 | 1302.77 | 1296.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1284.60 | 1299.13 | 1295.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 1284.60 | 1299.13 | 1295.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1291.70 | 1295.27 | 1294.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1279.20 | 1295.27 | 1294.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1294.60 | 1295.14 | 1294.27 | EMA400 retest candle locked (from upside) |

### Cycle 202 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1279.00 | 1291.91 | 1292.88 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 1318.50 | 1296.10 | 1293.91 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 1281.30 | 1296.35 | 1297.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 1272.90 | 1291.66 | 1295.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1292.00 | 1286.39 | 1290.44 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 1309.00 | 1290.02 | 1288.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1339.80 | 1303.81 | 1295.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1380.20 | 1401.50 | 1378.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1380.20 | 1401.50 | 1378.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1346.00 | 1390.40 | 1375.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1356.80 | 1390.40 | 1375.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1320.80 | 1376.48 | 1370.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 1320.80 | 1376.48 | 1370.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1317.00 | 1357.23 | 1362.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 13:15:00 | 1312.90 | 1323.26 | 1332.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1333.60 | 1325.33 | 1332.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 1341.70 | 1325.33 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 1338.00 | 1327.86 | 1332.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 1319.30 | 1326.33 | 1331.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 10:30:00 | 1319.90 | 1325.98 | 1331.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 1341.20 | 1331.20 | 1330.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 1341.20 | 1331.20 | 1330.38 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 10:15:00 | 1322.80 | 1328.84 | 1329.52 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 1345.10 | 1327.60 | 1327.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 13:15:00 | 1352.70 | 1339.61 | 1333.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1311.10 | 1335.92 | 1333.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1313.60 | 1335.92 | 1333.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1315.10 | 1331.76 | 1332.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 1305.00 | 1320.12 | 1325.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1332.10 | 1318.98 | 1322.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 1334.90 | 1318.98 | 1322.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1326.00 | 1320.39 | 1322.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1333.20 | 1320.39 | 1322.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 1354.40 | 1327.19 | 1325.29 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1325.50 | 1331.52 | 1331.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1319.00 | 1329.02 | 1330.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1341.00 | 1328.69 | 1329.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 1366.90 | 1328.69 | 1329.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 1345.50 | 1332.05 | 1330.75 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 1322.00 | 1329.75 | 1330.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 1313.10 | 1326.42 | 1328.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1322.00 | 1318.08 | 1322.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 1323.90 | 1318.08 | 1322.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1320.00 | 1318.46 | 1322.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1309.40 | 1318.46 | 1322.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1319.80 | 1319.61 | 1320.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1331.20 | 1321.93 | 1321.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1331.20 | 1321.93 | 1321.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1339.30 | 1326.25 | 1323.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 1346.00 | 1348.70 | 1340.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:00:00 | 1346.00 | 1348.70 | 1340.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1366.30 | 1355.66 | 1347.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:30:00 | 1414.80 | 1365.32 | 1355.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 10:00:00 | 1426.00 | 1365.32 | 1355.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 15:15:00 | 1414.90 | 1387.02 | 1371.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:45:00 | 1411.60 | 1400.51 | 1380.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1408.70 | 1420.36 | 1410.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1448.00 | 1420.88 | 1413.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:45:00 | 1432.20 | 1425.10 | 1417.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 1429.80 | 1430.98 | 1424.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 12:00:00 | 1424.00 | 1428.64 | 1424.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1419.60 | 1426.83 | 1424.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1419.60 | 1426.83 | 1424.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1420.50 | 1425.57 | 1423.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1418.90 | 1425.57 | 1423.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1407.00 | 1421.85 | 1422.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1391.00 | 1410.05 | 1414.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1407.20 | 1364.75 | 1378.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 1407.20 | 1364.75 | 1378.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1419.00 | 1375.60 | 1382.39 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1412.80 | 1388.87 | 1387.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 1426.00 | 1400.38 | 1393.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 1432.70 | 1440.20 | 1428.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 1432.70 | 1440.20 | 1428.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1416.80 | 1435.52 | 1427.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1416.80 | 1435.52 | 1427.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 1424.40 | 1433.29 | 1427.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1439.60 | 1433.29 | 1427.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 11:15:00 | 1404.40 | 1423.62 | 1424.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 1404.40 | 1423.62 | 1424.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 12:15:00 | 1402.20 | 1419.34 | 1422.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1375.00 | 1367.02 | 1381.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1375.00 | 1367.02 | 1381.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1362.00 | 1367.64 | 1379.06 | EMA400 retest candle locked (from downside) |

### Cycle 219 — BUY (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 15:15:00 | 1387.00 | 1380.56 | 1380.26 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1351.40 | 1374.73 | 1377.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1344.60 | 1368.70 | 1374.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1351.30 | 1316.07 | 1334.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1381.90 | 1316.07 | 1334.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1329.50 | 1318.75 | 1333.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1300.90 | 1335.68 | 1337.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1362.00 | 1335.17 | 1335.22 | SL hit (close>static) qty=1.00 sl=1352.10 alert=retest2 |

### Cycle 221 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 1368.30 | 1341.80 | 1338.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 1372.10 | 1357.52 | 1347.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 1539.90 | 1542.05 | 1509.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 15:00:00 | 1539.90 | 1542.05 | 1509.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1499.30 | 1533.17 | 1510.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 1524.00 | 1529.07 | 1512.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 1521.00 | 1524.91 | 1513.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 1523.90 | 1525.43 | 1514.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1545.70 | 1524.34 | 1515.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 1519.60 | 1530.72 | 1523.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:00:00 | 1519.60 | 1530.72 | 1523.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 1520.30 | 1528.64 | 1523.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 1529.80 | 1528.64 | 1523.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 1505.80 | 1524.07 | 1521.76 | SL hit (close<static) qty=1.00 sl=1513.40 alert=retest2 |

### Cycle 222 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 1509.20 | 1518.52 | 1519.59 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1534.30 | 1522.01 | 1520.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 1623.40 | 1552.33 | 1536.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 1739.00 | 1751.66 | 1718.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 1719.10 | 1745.15 | 1718.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1719.10 | 1745.15 | 1718.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 1719.10 | 1745.15 | 1718.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 1729.00 | 1741.92 | 1719.12 | EMA400 retest candle locked (from upside) |

### Cycle 224 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1672.60 | 1708.05 | 1711.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1654.40 | 1697.32 | 1706.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1697.00 | 1686.66 | 1696.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1697.00 | 1686.66 | 1696.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1709.90 | 1691.31 | 1697.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:45:00 | 1709.00 | 1691.31 | 1697.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1718.50 | 1696.75 | 1699.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 1718.50 | 1696.75 | 1699.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1727.00 | 1702.80 | 1701.78 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1683.90 | 1700.00 | 1702.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 11:15:00 | 1680.70 | 1690.74 | 1696.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 15:15:00 | 1690.00 | 1688.48 | 1693.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1659.00 | 1688.48 | 1693.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1647.90 | 1680.36 | 1689.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 1640.70 | 1680.36 | 1689.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1491.40 | 1670.22 | 1678.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-04 09:15:00 | 1493.10 | 1636.13 | 1662.17 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 227 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1613.70 | 1566.18 | 1563.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 1665.50 | 1616.99 | 1594.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1626.00 | 1635.56 | 1614.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:45:00 | 1623.80 | 1635.56 | 1614.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 15:15:00 | 992.00 | 2024-04-16 09:15:00 | 1035.00 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2024-04-15 14:15:00 | 994.05 | 2024-04-16 09:15:00 | 1035.00 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2024-04-15 14:45:00 | 994.25 | 2024-04-16 09:15:00 | 1035.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2024-04-26 12:15:00 | 1028.90 | 2024-05-06 09:15:00 | 1033.90 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2024-04-29 09:15:00 | 1031.10 | 2024-05-06 09:15:00 | 1033.90 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2024-05-24 11:45:00 | 989.95 | 2024-05-27 09:15:00 | 970.85 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-06-06 13:45:00 | 919.90 | 2024-06-07 09:15:00 | 978.30 | STOP_HIT | 1.00 | -6.35% |
| SELL | retest2 | 2024-07-11 10:15:00 | 1320.75 | 2024-07-15 09:15:00 | 1360.95 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-07-12 09:30:00 | 1311.10 | 2024-07-15 09:15:00 | 1360.95 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2024-07-18 11:30:00 | 1368.10 | 2024-07-19 10:15:00 | 1341.35 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-07-19 12:30:00 | 1369.80 | 2024-07-26 09:15:00 | 1506.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-19 13:30:00 | 1396.65 | 2024-07-29 09:15:00 | 1536.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 10:30:00 | 1615.80 | 2024-08-08 09:15:00 | 1724.25 | STOP_HIT | 1.00 | -6.71% |
| BUY | retest2 | 2024-08-20 09:15:00 | 1711.35 | 2024-08-20 09:15:00 | 1686.15 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-08-26 09:45:00 | 1863.75 | 2024-08-28 13:15:00 | 1798.00 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2024-08-26 10:15:00 | 1876.70 | 2024-08-28 13:15:00 | 1798.00 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2024-09-05 10:15:00 | 1654.00 | 2024-09-09 09:15:00 | 1578.00 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2024-09-05 10:15:00 | 1654.00 | 2024-09-10 09:15:00 | 1601.25 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2024-09-06 09:15:00 | 1661.05 | 2024-09-12 13:15:00 | 1620.10 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2024-09-25 09:15:00 | 1723.20 | 2024-09-25 11:15:00 | 1693.25 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-09-26 09:15:00 | 1736.30 | 2024-09-26 12:15:00 | 1678.40 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-10-07 09:15:00 | 1648.00 | 2024-10-09 09:15:00 | 1689.40 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-10-29 10:30:00 | 1699.00 | 2024-10-29 14:15:00 | 1764.60 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2024-11-05 13:45:00 | 1887.15 | 2024-11-08 11:15:00 | 1848.10 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1911.70 | 2024-11-08 11:15:00 | 1848.10 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2024-11-07 12:15:00 | 1885.55 | 2024-11-08 11:15:00 | 1848.10 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-11-07 14:00:00 | 1883.95 | 2024-11-08 11:15:00 | 1848.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-11-14 13:30:00 | 1742.00 | 2024-11-19 09:15:00 | 1797.90 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2024-11-18 14:15:00 | 1744.20 | 2024-11-19 09:15:00 | 1797.90 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2024-12-18 15:00:00 | 2400.25 | 2024-12-26 15:15:00 | 2461.00 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2024-12-19 09:15:00 | 2441.10 | 2024-12-26 15:15:00 | 2461.00 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2024-12-20 15:00:00 | 2478.95 | 2024-12-26 15:15:00 | 2461.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-01-17 09:15:00 | 2214.45 | 2025-01-20 12:15:00 | 2270.60 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-01-30 13:45:00 | 1683.95 | 2025-01-31 09:15:00 | 1756.30 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1792.05 | 2025-02-03 09:15:00 | 1677.20 | STOP_HIT | 1.00 | -6.41% |
| BUY | retest2 | 2025-02-01 15:00:00 | 1819.70 | 2025-02-03 09:15:00 | 1677.20 | STOP_HIT | 1.00 | -7.83% |
| SELL | retest2 | 2025-02-05 12:45:00 | 1679.20 | 2025-02-10 09:15:00 | 1595.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 12:45:00 | 1679.20 | 2025-02-10 14:15:00 | 1621.70 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2025-03-03 15:00:00 | 1037.25 | 2025-03-04 11:15:00 | 1068.65 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-03-18 09:15:00 | 1248.00 | 2025-03-21 10:15:00 | 1372.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 13:45:00 | 1246.10 | 2025-03-21 10:15:00 | 1370.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1377.20 | 2025-04-09 11:15:00 | 1437.00 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest1 | 2025-04-15 09:15:00 | 1477.70 | 2025-04-21 09:15:00 | 1487.20 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-04-17 11:15:00 | 1509.00 | 2025-04-21 09:15:00 | 1487.20 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-04-17 12:00:00 | 1505.20 | 2025-04-21 09:15:00 | 1487.20 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-04-28 09:15:00 | 1400.10 | 2025-04-28 09:15:00 | 1449.00 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-05-06 11:30:00 | 1379.00 | 2025-05-08 09:15:00 | 1404.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-05-06 12:45:00 | 1376.00 | 2025-05-08 09:15:00 | 1404.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-05-27 09:45:00 | 1944.00 | 2025-05-29 09:15:00 | 2107.60 | TARGET_HIT | 1.00 | 8.42% |
| BUY | retest2 | 2025-05-27 10:45:00 | 1916.00 | 2025-05-29 09:15:00 | 2107.71 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-05-27 11:30:00 | 1916.10 | 2025-05-29 09:15:00 | 2107.60 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2025-05-27 13:00:00 | 1916.00 | 2025-05-30 09:15:00 | 2138.40 | TARGET_HIT | 1.00 | 11.61% |
| BUY | retest2 | 2025-06-03 09:15:00 | 2160.00 | 2025-06-06 10:15:00 | 2084.80 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-06-03 10:30:00 | 2158.00 | 2025-06-06 10:15:00 | 2084.80 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-06-04 09:15:00 | 2213.10 | 2025-06-06 10:15:00 | 2084.80 | STOP_HIT | 1.00 | -5.80% |
| SELL | retest2 | 2025-06-11 12:00:00 | 1962.30 | 2025-06-13 09:15:00 | 1864.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:00:00 | 1962.30 | 2025-06-13 09:15:00 | 1989.60 | STOP_HIT | 0.50 | -1.39% |
| SELL | retest2 | 2025-06-11 12:30:00 | 1961.30 | 2025-06-13 09:15:00 | 1863.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 12:30:00 | 1961.30 | 2025-06-13 09:15:00 | 1989.60 | STOP_HIT | 0.50 | -1.44% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1945.10 | 2025-06-13 09:15:00 | 1847.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1945.10 | 2025-06-13 09:15:00 | 1989.60 | STOP_HIT | 0.50 | -2.29% |
| SELL | retest2 | 2025-06-12 10:15:00 | 1956.00 | 2025-06-13 09:15:00 | 1858.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 10:15:00 | 1956.00 | 2025-06-13 09:15:00 | 1989.60 | STOP_HIT | 0.50 | -1.72% |
| SELL | retest2 | 2025-06-18 14:15:00 | 1934.10 | 2025-06-23 09:15:00 | 1979.10 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-06-18 15:00:00 | 1935.00 | 2025-06-23 09:15:00 | 1979.10 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-06-19 10:00:00 | 1932.00 | 2025-06-23 09:15:00 | 1979.10 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-07-04 11:30:00 | 1942.00 | 2025-07-08 15:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-04 12:15:00 | 1941.10 | 2025-07-08 15:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-08 09:45:00 | 1942.20 | 2025-07-08 15:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-18 14:15:00 | 1856.70 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-07-21 10:15:00 | 1859.50 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-07-21 11:00:00 | 1856.60 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-07-21 14:45:00 | 1900.80 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-23 12:00:00 | 1880.80 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-23 14:00:00 | 1883.10 | 2025-07-24 14:15:00 | 1870.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-05 11:45:00 | 1483.50 | 2025-08-07 09:15:00 | 1409.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:45:00 | 1483.50 | 2025-08-07 15:15:00 | 1459.00 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-09-08 15:00:00 | 1458.00 | 2025-09-10 13:15:00 | 1479.40 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-10 11:30:00 | 1464.20 | 2025-09-10 13:15:00 | 1479.40 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-13 10:30:00 | 1397.40 | 2025-10-17 13:15:00 | 1378.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1390.90 | 2025-10-27 09:15:00 | 1329.90 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1401.80 | 2025-11-06 11:15:00 | 1369.90 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-11-06 10:30:00 | 1370.50 | 2025-11-06 11:15:00 | 1369.90 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-10 12:30:00 | 1361.70 | 2025-11-11 09:15:00 | 1382.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-10 14:45:00 | 1363.90 | 2025-11-11 09:15:00 | 1382.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-10 15:15:00 | 1357.00 | 2025-11-11 09:15:00 | 1382.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-12-01 13:15:00 | 1396.90 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-12-01 14:00:00 | 1397.00 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-12-01 14:45:00 | 1396.40 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1392.20 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-12-03 09:45:00 | 1375.00 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-03 11:15:00 | 1373.90 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-12-03 12:45:00 | 1374.20 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-12-04 10:15:00 | 1375.50 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-12-05 10:15:00 | 1386.80 | 2025-12-05 10:15:00 | 1396.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-12-09 09:15:00 | 1336.20 | 2025-12-10 09:15:00 | 1367.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-10 11:15:00 | 1342.50 | 2025-12-11 14:15:00 | 1374.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1347.00 | 2025-12-11 14:15:00 | 1374.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1352.30 | 2025-12-19 14:15:00 | 1365.90 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1349.20 | 2025-12-31 15:15:00 | 1373.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-01-09 14:00:00 | 1306.90 | 2026-01-12 09:15:00 | 1241.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 14:00:00 | 1306.90 | 2026-01-14 09:15:00 | 1247.70 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2026-02-06 09:30:00 | 1319.30 | 2026-02-09 14:15:00 | 1341.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-06 10:30:00 | 1319.90 | 2026-02-09 14:15:00 | 1341.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1309.40 | 2026-02-25 10:15:00 | 1331.20 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-25 09:30:00 | 1319.80 | 2026-02-25 10:15:00 | 1331.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-03-04 09:30:00 | 1414.80 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-03-04 10:00:00 | 1426.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-03-04 15:15:00 | 1414.90 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-03-05 09:45:00 | 1411.60 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-03-09 14:30:00 | 1448.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-10 10:45:00 | 1432.20 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-03-11 10:15:00 | 1429.80 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-03-11 12:00:00 | 1424.00 | 2026-03-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1439.60 | 2026-03-20 11:15:00 | 1404.40 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1300.90 | 2026-04-02 13:15:00 | 1362.00 | STOP_HIT | 1.00 | -4.70% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1524.00 | 2026-04-16 09:15:00 | 1505.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-04-13 13:30:00 | 1521.00 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-04-13 14:45:00 | 1523.90 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1545.70 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-04-16 09:15:00 | 1529.80 | 2026-04-16 12:15:00 | 1509.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2026-04-30 09:15:00 | 1659.00 | 2026-05-04 09:15:00 | 1493.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 10:15:00 | 1640.70 | 2026-05-04 09:15:00 | 1476.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 09:15:00 | 1491.40 | 2026-05-07 09:15:00 | 1613.70 | STOP_HIT | 1.00 | -8.20% |
