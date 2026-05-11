# NATCO Pharma Ltd. (NATCOPHARM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1174.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 221 |
| ALERT1 | 150 |
| ALERT2 | 148 |
| ALERT2_SKIP | 94 |
| ALERT3 | 316 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 123 |
| PARTIAL | 5 |
| TARGET_HIT | 14 |
| STOP_HIT | 113 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 88
- **Target hits / Stop hits / Partials:** 14 / 113 / 5
- **Avg / median % per leg:** 0.44% / -0.70%
- **Sum % (uncompounded):** 57.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 23 | 33.8% | 12 | 56 | 0 | 0.97% | 65.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.54% | -4.6% |
| BUY @ 3rd Alert (retest2) | 65 | 23 | 35.4% | 12 | 53 | 0 | 1.08% | 70.4% |
| SELL (all) | 64 | 21 | 32.8% | 2 | 57 | 5 | -0.13% | -8.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 62 | 19 | 30.6% | 1 | 57 | 4 | -0.37% | -23.2% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 2.08% | 10.4% |
| retest2 (combined) | 127 | 42 | 33.1% | 13 | 110 | 4 | 0.37% | 47.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 635.20 | 627.66 | 627.02 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 09:15:00 | 623.95 | 627.54 | 627.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 10:15:00 | 622.00 | 626.43 | 627.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 14:15:00 | 622.65 | 622.60 | 624.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 626.55 | 623.20 | 624.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 626.55 | 623.20 | 624.69 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 13:15:00 | 622.85 | 620.68 | 620.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 12:15:00 | 634.65 | 623.76 | 621.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 14:15:00 | 632.20 | 634.44 | 629.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 14:15:00 | 632.20 | 634.44 | 629.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 632.20 | 634.44 | 629.92 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 14:15:00 | 617.40 | 627.49 | 628.44 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 09:15:00 | 630.65 | 624.99 | 624.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 637.65 | 629.66 | 627.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 11:15:00 | 629.60 | 630.06 | 627.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 12:15:00 | 629.45 | 629.93 | 628.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 12:15:00 | 629.45 | 629.93 | 628.12 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 10:15:00 | 620.30 | 627.42 | 627.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 14:15:00 | 619.90 | 624.03 | 625.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 09:15:00 | 623.95 | 623.37 | 625.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 623.95 | 623.37 | 625.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 623.95 | 623.37 | 625.15 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 10:15:00 | 629.35 | 622.74 | 621.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 11:15:00 | 635.05 | 625.20 | 623.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 11:15:00 | 631.85 | 633.61 | 629.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 14:15:00 | 630.50 | 633.12 | 630.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 630.50 | 633.12 | 630.35 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 626.85 | 629.94 | 630.07 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 10:15:00 | 632.45 | 629.83 | 629.76 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 14:15:00 | 623.40 | 628.89 | 629.41 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 11:15:00 | 630.00 | 628.80 | 628.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 12:15:00 | 633.95 | 629.83 | 629.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 12:15:00 | 686.50 | 687.76 | 673.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 14:15:00 | 693.25 | 699.38 | 694.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 14:15:00 | 693.25 | 699.38 | 694.99 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 12:15:00 | 682.45 | 692.45 | 693.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 14:15:00 | 678.85 | 688.38 | 691.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 12:15:00 | 689.75 | 686.89 | 689.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 12:15:00 | 689.75 | 686.89 | 689.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 12:15:00 | 689.75 | 686.89 | 689.44 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-07-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 15:15:00 | 697.45 | 691.12 | 690.92 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 10:15:00 | 684.00 | 689.52 | 690.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 11:15:00 | 681.55 | 687.92 | 689.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 09:15:00 | 688.95 | 685.77 | 687.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 688.95 | 685.77 | 687.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 688.95 | 685.77 | 687.48 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 12:15:00 | 693.10 | 689.19 | 688.76 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 14:15:00 | 684.00 | 688.24 | 688.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 09:15:00 | 679.85 | 685.80 | 687.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 09:15:00 | 675.60 | 674.39 | 677.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 09:15:00 | 675.60 | 674.39 | 677.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 675.60 | 674.39 | 677.73 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 15:15:00 | 684.65 | 678.44 | 678.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 09:15:00 | 696.55 | 682.06 | 680.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 13:15:00 | 716.10 | 716.25 | 704.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 12:15:00 | 709.30 | 715.70 | 709.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 709.30 | 715.70 | 709.82 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 812.05 | 822.79 | 823.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 11:15:00 | 804.10 | 819.05 | 821.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 808.35 | 808.21 | 814.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 09:15:00 | 808.35 | 808.21 | 814.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 808.35 | 808.21 | 814.57 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 825.00 | 812.93 | 812.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 13:15:00 | 839.15 | 822.48 | 817.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 15:15:00 | 885.00 | 885.67 | 870.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 10:15:00 | 874.00 | 881.73 | 871.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 874.00 | 881.73 | 871.20 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 11:15:00 | 854.85 | 870.04 | 870.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 12:15:00 | 848.95 | 865.82 | 868.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 838.80 | 834.79 | 841.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 838.80 | 834.79 | 841.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 838.80 | 834.79 | 841.73 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 14:15:00 | 854.75 | 846.44 | 845.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 15:15:00 | 859.85 | 849.12 | 846.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 14:15:00 | 860.90 | 864.04 | 859.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 15:15:00 | 861.00 | 863.43 | 860.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 861.00 | 863.43 | 860.02 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 09:15:00 | 894.05 | 904.88 | 905.35 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 10:15:00 | 911.90 | 904.66 | 904.30 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 13:15:00 | 892.80 | 902.34 | 903.34 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 09:15:00 | 926.20 | 905.36 | 904.30 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 11:15:00 | 882.30 | 902.88 | 905.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 09:15:00 | 844.90 | 882.21 | 893.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 09:15:00 | 832.00 | 827.24 | 843.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 846.80 | 832.72 | 843.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 846.80 | 832.72 | 843.46 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 15:15:00 | 856.00 | 849.15 | 849.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 10:15:00 | 865.20 | 853.57 | 851.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 10:15:00 | 875.45 | 875.83 | 868.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 11:15:00 | 870.00 | 874.67 | 869.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 870.00 | 874.67 | 869.00 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 12:15:00 | 859.80 | 868.21 | 868.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 853.30 | 862.41 | 865.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 15:15:00 | 853.50 | 851.94 | 855.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 875.30 | 856.61 | 857.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 875.30 | 856.61 | 857.66 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 875.00 | 860.29 | 859.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 09:15:00 | 879.60 | 870.50 | 865.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 13:15:00 | 876.00 | 881.57 | 876.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 13:15:00 | 876.00 | 881.57 | 876.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 876.00 | 881.57 | 876.66 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 09:15:00 | 869.90 | 877.46 | 877.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 11:15:00 | 857.60 | 864.17 | 868.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 13:15:00 | 865.00 | 863.27 | 867.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 14:15:00 | 866.10 | 863.83 | 867.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 866.10 | 863.83 | 867.08 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 11:15:00 | 861.75 | 856.31 | 856.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 12:15:00 | 863.00 | 857.65 | 856.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 10:15:00 | 861.65 | 862.77 | 860.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 11:15:00 | 857.00 | 861.62 | 859.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 11:15:00 | 857.00 | 861.62 | 859.74 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 14:15:00 | 850.80 | 857.48 | 858.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 15:15:00 | 850.00 | 855.98 | 857.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 847.75 | 845.49 | 850.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 847.75 | 845.49 | 850.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 847.75 | 845.49 | 850.05 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 10:15:00 | 766.95 | 749.01 | 748.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 11:15:00 | 774.00 | 754.01 | 750.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 09:15:00 | 779.20 | 785.03 | 775.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 11:15:00 | 769.95 | 781.21 | 775.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 11:15:00 | 769.95 | 781.21 | 775.02 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 15:15:00 | 759.50 | 769.95 | 771.08 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 10:15:00 | 776.50 | 772.47 | 772.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 09:15:00 | 781.60 | 775.27 | 773.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 13:15:00 | 780.25 | 781.19 | 778.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 12:15:00 | 779.95 | 782.74 | 780.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 12:15:00 | 779.95 | 782.74 | 780.94 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 10:15:00 | 783.00 | 790.32 | 790.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 11:15:00 | 781.50 | 788.55 | 789.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 10:15:00 | 779.75 | 777.48 | 780.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 10:15:00 | 779.75 | 777.48 | 780.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 779.75 | 777.48 | 780.74 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 792.15 | 782.97 | 782.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 793.95 | 788.18 | 785.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 10:15:00 | 790.10 | 790.70 | 787.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 12:15:00 | 787.05 | 790.01 | 787.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 787.05 | 790.01 | 787.50 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 12:15:00 | 781.90 | 788.07 | 788.77 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 10:15:00 | 789.60 | 787.09 | 787.04 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 14:15:00 | 782.35 | 786.47 | 786.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 778.75 | 783.82 | 785.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 09:15:00 | 763.65 | 762.61 | 769.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 762.70 | 760.78 | 764.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 762.70 | 760.78 | 764.74 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 773.55 | 766.36 | 765.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 09:15:00 | 780.00 | 774.07 | 770.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 777.95 | 778.00 | 774.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 774.45 | 777.61 | 774.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 774.45 | 777.61 | 774.74 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 766.35 | 775.75 | 776.71 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 803.30 | 779.71 | 777.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 813.35 | 804.34 | 796.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 802.45 | 804.63 | 798.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 14:15:00 | 805.00 | 803.96 | 799.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 805.00 | 803.96 | 799.27 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 10:15:00 | 844.80 | 848.86 | 849.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 11:15:00 | 842.50 | 847.59 | 848.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 827.60 | 824.28 | 830.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 15:15:00 | 829.90 | 825.25 | 829.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 829.90 | 825.25 | 829.02 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 835.75 | 831.08 | 830.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 13:15:00 | 835.80 | 832.02 | 831.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 828.80 | 832.53 | 831.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 828.80 | 832.53 | 831.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 828.80 | 832.53 | 831.87 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 829.45 | 831.48 | 831.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 13:15:00 | 827.00 | 830.03 | 830.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-20 14:15:00 | 831.20 | 830.26 | 830.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 14:15:00 | 831.20 | 830.26 | 830.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 831.20 | 830.26 | 830.82 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 09:15:00 | 840.15 | 831.56 | 831.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 09:15:00 | 853.70 | 840.84 | 836.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 10:15:00 | 874.40 | 876.66 | 869.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 12:15:00 | 876.90 | 876.34 | 870.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 876.90 | 876.34 | 870.44 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 15:15:00 | 865.00 | 870.72 | 871.03 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 12:15:00 | 874.35 | 871.68 | 871.38 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 15:15:00 | 869.00 | 870.81 | 871.05 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 10:15:00 | 874.45 | 871.65 | 871.39 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 12:15:00 | 864.20 | 870.37 | 870.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 13:15:00 | 861.95 | 868.68 | 870.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 09:15:00 | 869.95 | 866.68 | 868.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 09:15:00 | 869.95 | 866.68 | 868.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 869.95 | 866.68 | 868.58 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 14:15:00 | 856.20 | 845.42 | 844.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 09:15:00 | 871.70 | 852.69 | 848.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 14:15:00 | 887.50 | 887.65 | 876.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 10:15:00 | 1024.60 | 1044.31 | 1028.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 1024.60 | 1044.31 | 1028.52 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 13:15:00 | 1015.20 | 1024.45 | 1025.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 15:15:00 | 1012.00 | 1020.43 | 1023.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 11:15:00 | 1019.75 | 1018.23 | 1021.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 11:15:00 | 1019.75 | 1018.23 | 1021.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 1019.75 | 1018.23 | 1021.40 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 15:15:00 | 1027.00 | 1023.05 | 1022.94 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 09:15:00 | 1019.90 | 1023.23 | 1023.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 12:15:00 | 1014.70 | 1019.79 | 1021.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 09:15:00 | 996.65 | 991.02 | 1000.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 09:15:00 | 996.65 | 991.02 | 1000.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 996.65 | 991.02 | 1000.79 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 09:15:00 | 1015.60 | 996.34 | 995.18 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 990.25 | 995.79 | 995.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 10:15:00 | 985.05 | 993.65 | 994.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 09:15:00 | 1002.75 | 989.84 | 991.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 1002.75 | 989.84 | 991.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 1002.75 | 989.84 | 991.54 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 10:15:00 | 1010.75 | 994.02 | 993.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 14:15:00 | 1020.60 | 1005.72 | 999.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-07 12:15:00 | 1016.00 | 1017.75 | 1008.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 14:15:00 | 1015.45 | 1016.70 | 1009.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 1015.45 | 1016.70 | 1009.89 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 09:15:00 | 996.05 | 1009.45 | 1009.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 983.00 | 1004.16 | 1007.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 974.95 | 956.41 | 971.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 974.95 | 956.41 | 971.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 974.95 | 956.41 | 971.42 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 965.00 | 956.97 | 956.62 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 15:15:00 | 950.00 | 955.59 | 956.25 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 13:15:00 | 961.00 | 956.34 | 956.22 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-03-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 14:15:00 | 951.90 | 955.45 | 955.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 15:15:00 | 950.00 | 954.36 | 955.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 955.65 | 954.62 | 955.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 955.65 | 954.62 | 955.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 955.65 | 954.62 | 955.33 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 12:15:00 | 960.00 | 956.60 | 956.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 13:15:00 | 966.90 | 958.66 | 957.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 15:15:00 | 955.00 | 958.10 | 957.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 15:15:00 | 955.00 | 958.10 | 957.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 955.00 | 958.10 | 957.15 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 14:15:00 | 954.25 | 961.69 | 962.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 15:15:00 | 952.95 | 959.94 | 961.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 975.00 | 962.96 | 962.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 975.00 | 962.96 | 962.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 975.00 | 962.96 | 962.99 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 974.75 | 965.31 | 964.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 09:15:00 | 987.65 | 974.18 | 969.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 15:15:00 | 995.00 | 997.91 | 990.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 13:15:00 | 994.40 | 996.79 | 992.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 994.40 | 996.79 | 992.73 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 10:15:00 | 984.95 | 996.84 | 996.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 09:15:00 | 962.50 | 983.97 | 990.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 09:15:00 | 975.40 | 966.52 | 975.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 975.40 | 966.52 | 975.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 975.40 | 966.52 | 975.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:30:00 | 973.35 | 966.52 | 975.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 968.30 | 966.88 | 975.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 964.95 | 971.82 | 974.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 983.25 | 974.11 | 975.57 | SL hit (close>static) qty=1.00 sl=977.85 alert=retest2 |

### Cycle 69 — BUY (started 2024-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 11:15:00 | 983.75 | 977.06 | 976.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 09:15:00 | 987.40 | 981.98 | 979.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 14:15:00 | 985.50 | 986.54 | 983.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-16 15:00:00 | 985.50 | 986.54 | 983.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 993.45 | 999.25 | 992.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 15:00:00 | 993.45 | 999.25 | 992.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 996.85 | 998.77 | 993.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 990.05 | 998.77 | 993.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1004.75 | 999.96 | 994.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 1013.00 | 999.97 | 996.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 10:45:00 | 1005.20 | 1003.12 | 998.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 15:15:00 | 1009.90 | 1005.44 | 1001.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 15:15:00 | 1000.00 | 1006.78 | 1007.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 15:15:00 | 1000.00 | 1006.78 | 1007.67 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 09:15:00 | 1024.95 | 1010.42 | 1009.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 1057.00 | 1031.44 | 1021.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 13:15:00 | 1032.75 | 1035.49 | 1026.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 13:45:00 | 1031.70 | 1035.49 | 1026.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 1028.30 | 1034.05 | 1026.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 1028.30 | 1034.05 | 1026.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 1030.00 | 1033.24 | 1027.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:15:00 | 1015.80 | 1033.24 | 1027.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 1014.45 | 1029.48 | 1025.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 09:45:00 | 1013.90 | 1029.48 | 1025.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 1008.00 | 1025.19 | 1024.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:00:00 | 1008.00 | 1025.19 | 1024.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 11:15:00 | 1009.50 | 1022.05 | 1022.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 13:15:00 | 1006.25 | 1016.80 | 1020.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 1028.50 | 1016.92 | 1019.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 1028.50 | 1016.92 | 1019.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 1028.50 | 1016.92 | 1019.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:45:00 | 1029.00 | 1016.92 | 1019.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 1027.45 | 1019.03 | 1019.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:30:00 | 1028.65 | 1019.03 | 1019.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 11:15:00 | 1027.85 | 1020.79 | 1020.69 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 15:15:00 | 1019.20 | 1020.49 | 1020.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 09:15:00 | 1011.40 | 1018.67 | 1019.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 09:15:00 | 1034.65 | 1017.82 | 1018.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 09:15:00 | 1034.65 | 1017.82 | 1018.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 1034.65 | 1017.82 | 1018.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 09:30:00 | 1037.55 | 1017.82 | 1018.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 10:15:00 | 1036.60 | 1021.57 | 1019.79 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 14:15:00 | 1015.85 | 1023.41 | 1023.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 997.65 | 1015.84 | 1019.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 984.50 | 977.91 | 986.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 984.50 | 977.91 | 986.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 984.50 | 977.91 | 986.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:00:00 | 984.50 | 977.91 | 986.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 970.70 | 977.58 | 984.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 12:15:00 | 968.15 | 977.58 | 984.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 12:45:00 | 968.85 | 975.86 | 983.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 988.95 | 980.32 | 980.48 | SL hit (close>static) qty=1.00 sl=985.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 990.00 | 982.25 | 981.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 999.50 | 988.37 | 986.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 12:15:00 | 993.25 | 996.32 | 993.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 12:15:00 | 993.25 | 996.32 | 993.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 993.25 | 996.32 | 993.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:45:00 | 992.00 | 996.32 | 993.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 992.20 | 995.50 | 993.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:00:00 | 992.20 | 995.50 | 993.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 995.80 | 995.56 | 993.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:30:00 | 992.10 | 995.56 | 993.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 993.95 | 995.24 | 993.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 999.55 | 995.24 | 993.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 999.20 | 997.07 | 994.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:45:00 | 998.40 | 997.91 | 995.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 998.55 | 999.51 | 997.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1002.30 | 1002.39 | 999.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 1001.00 | 1002.39 | 999.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1000.55 | 1002.31 | 1000.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:30:00 | 999.85 | 1002.31 | 1000.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1006.30 | 1003.11 | 1000.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 1000.30 | 1003.11 | 1000.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1023.00 | 1024.50 | 1017.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1023.00 | 1024.50 | 1017.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1020.10 | 1023.62 | 1017.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 1033.50 | 1023.62 | 1017.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-28 09:15:00 | 1099.51 | 1039.21 | 1030.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 1027.00 | 1035.07 | 1035.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 1022.75 | 1029.93 | 1032.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 09:15:00 | 1014.00 | 1005.12 | 1010.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1014.00 | 1005.12 | 1010.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1014.00 | 1005.12 | 1010.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:00:00 | 1014.00 | 1005.12 | 1010.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 974.85 | 999.07 | 1006.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 12:15:00 | 957.85 | 994.29 | 1004.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 15:00:00 | 969.80 | 985.22 | 997.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1022.90 | 990.61 | 997.34 | SL hit (close>static) qty=1.00 sl=1015.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 1029.10 | 1003.35 | 1002.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 1030.50 | 1008.78 | 1004.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 12:15:00 | 1205.05 | 1206.58 | 1190.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 13:00:00 | 1205.05 | 1206.58 | 1190.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1192.80 | 1203.01 | 1194.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1192.25 | 1203.01 | 1194.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1189.95 | 1200.40 | 1193.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 1192.00 | 1200.40 | 1193.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 1184.50 | 1197.22 | 1192.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:45:00 | 1184.50 | 1197.22 | 1192.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1171.80 | 1186.85 | 1188.83 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 1200.00 | 1184.98 | 1184.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 12:15:00 | 1205.50 | 1189.08 | 1186.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 13:15:00 | 1200.00 | 1200.42 | 1194.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 13:30:00 | 1198.00 | 1200.42 | 1194.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 1194.00 | 1199.14 | 1194.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:45:00 | 1194.05 | 1199.14 | 1194.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 1200.00 | 1199.31 | 1195.29 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 1185.90 | 1193.73 | 1193.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 1164.00 | 1179.08 | 1185.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 1164.75 | 1162.49 | 1171.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 11:00:00 | 1164.75 | 1162.49 | 1171.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1171.70 | 1164.87 | 1171.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:45:00 | 1170.05 | 1164.87 | 1171.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 1172.70 | 1166.44 | 1171.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:00:00 | 1172.70 | 1166.44 | 1171.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1167.50 | 1166.65 | 1170.93 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 1212.85 | 1176.11 | 1174.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 1229.25 | 1201.91 | 1194.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 11:15:00 | 1216.00 | 1218.01 | 1209.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 12:00:00 | 1216.00 | 1218.01 | 1209.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1218.00 | 1217.67 | 1212.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 1236.70 | 1217.67 | 1212.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1222.00 | 1218.54 | 1213.10 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 1196.00 | 1211.82 | 1213.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 10:15:00 | 1184.25 | 1201.24 | 1207.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 13:15:00 | 1208.50 | 1200.26 | 1205.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 13:15:00 | 1208.50 | 1200.26 | 1205.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 1208.50 | 1200.26 | 1205.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 1208.50 | 1200.26 | 1205.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 1198.85 | 1199.98 | 1204.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:00:00 | 1186.00 | 1197.12 | 1202.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:30:00 | 1194.10 | 1190.00 | 1194.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 1193.05 | 1190.00 | 1194.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:45:00 | 1191.70 | 1190.40 | 1194.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 1192.00 | 1190.72 | 1193.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 1192.00 | 1190.72 | 1193.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 1191.90 | 1190.96 | 1193.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:00:00 | 1191.90 | 1190.96 | 1193.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1190.75 | 1189.39 | 1192.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-15 10:15:00 | 1216.95 | 1193.56 | 1192.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 1216.95 | 1193.56 | 1192.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 12:15:00 | 1233.00 | 1205.30 | 1198.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 1207.15 | 1209.38 | 1201.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 1207.15 | 1209.38 | 1201.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1208.00 | 1209.11 | 1202.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 1215.00 | 1209.11 | 1202.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 10:00:00 | 1213.00 | 1209.89 | 1203.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 1219.20 | 1238.14 | 1239.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 1219.20 | 1238.14 | 1239.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 1216.30 | 1233.77 | 1237.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1242.65 | 1233.03 | 1235.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1242.65 | 1233.03 | 1235.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1242.65 | 1233.03 | 1235.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 1242.65 | 1233.03 | 1235.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1245.00 | 1235.43 | 1236.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 1252.95 | 1235.43 | 1236.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 1226.30 | 1229.68 | 1233.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 1224.75 | 1229.68 | 1233.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1224.95 | 1228.63 | 1232.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1175.05 | 1228.69 | 1231.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 1256.00 | 1231.53 | 1231.74 | SL hit (close>static) qty=1.00 sl=1242.30 alert=retest2 |

### Cycle 87 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 1251.20 | 1235.46 | 1233.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 1266.85 | 1241.74 | 1236.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 1346.00 | 1349.67 | 1330.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 1346.00 | 1349.67 | 1330.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1348.40 | 1351.51 | 1343.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 1355.65 | 1351.51 | 1343.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:15:00 | 1355.75 | 1353.48 | 1346.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:00:00 | 1357.10 | 1359.88 | 1355.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 1316.60 | 1350.97 | 1354.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1316.60 | 1350.97 | 1354.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 14:15:00 | 1309.90 | 1328.86 | 1341.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1347.70 | 1329.93 | 1339.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1347.70 | 1329.93 | 1339.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1347.70 | 1329.93 | 1339.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 1347.70 | 1329.93 | 1339.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1349.30 | 1333.81 | 1340.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 1349.30 | 1333.81 | 1340.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 1366.25 | 1340.29 | 1342.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:45:00 | 1366.00 | 1340.29 | 1342.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 1360.00 | 1344.24 | 1344.33 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 13:15:00 | 1347.65 | 1344.92 | 1344.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 09:15:00 | 1401.95 | 1357.67 | 1350.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 11:15:00 | 1433.95 | 1436.03 | 1416.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 12:00:00 | 1433.95 | 1436.03 | 1416.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1434.05 | 1439.10 | 1425.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:45:00 | 1454.85 | 1444.03 | 1429.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 13:15:00 | 1463.30 | 1466.06 | 1466.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 13:15:00 | 1463.30 | 1466.06 | 1466.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 14:15:00 | 1459.00 | 1464.05 | 1465.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1457.40 | 1450.23 | 1455.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 1457.40 | 1450.23 | 1455.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1457.40 | 1450.23 | 1455.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:45:00 | 1463.95 | 1450.23 | 1455.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 1448.00 | 1449.78 | 1454.61 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 14:15:00 | 1471.95 | 1458.87 | 1457.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 1490.85 | 1466.73 | 1461.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1526.85 | 1529.08 | 1503.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:45:00 | 1523.75 | 1529.08 | 1503.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1547.00 | 1556.25 | 1550.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 1550.20 | 1556.25 | 1550.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 1548.35 | 1554.67 | 1550.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:45:00 | 1547.90 | 1554.67 | 1550.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1537.35 | 1549.61 | 1549.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 1537.35 | 1549.61 | 1549.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 1543.00 | 1548.29 | 1548.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 1523.60 | 1543.35 | 1546.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 1553.95 | 1519.46 | 1528.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 1553.95 | 1519.46 | 1528.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1553.95 | 1519.46 | 1528.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 1553.95 | 1519.46 | 1528.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1543.90 | 1524.35 | 1530.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:15:00 | 1538.90 | 1524.35 | 1530.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:45:00 | 1536.30 | 1526.96 | 1530.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 13:15:00 | 1549.90 | 1534.91 | 1533.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 1549.90 | 1534.91 | 1533.88 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 13:15:00 | 1532.85 | 1536.02 | 1536.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 15:15:00 | 1525.00 | 1533.13 | 1534.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 12:15:00 | 1532.95 | 1529.60 | 1532.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 12:15:00 | 1532.95 | 1529.60 | 1532.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 1532.95 | 1529.60 | 1532.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:45:00 | 1535.10 | 1529.60 | 1532.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 1551.00 | 1533.88 | 1533.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:00:00 | 1551.00 | 1533.88 | 1533.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 1554.50 | 1538.00 | 1535.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 1568.25 | 1547.25 | 1540.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 14:15:00 | 1554.65 | 1560.07 | 1550.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 1554.65 | 1560.07 | 1550.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1554.65 | 1560.07 | 1550.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 1554.65 | 1560.07 | 1550.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 1555.90 | 1559.23 | 1551.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:45:00 | 1543.85 | 1555.43 | 1550.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 1541.60 | 1552.66 | 1549.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:45:00 | 1564.00 | 1555.91 | 1551.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 10:30:00 | 1557.75 | 1555.08 | 1553.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 11:15:00 | 1539.10 | 1551.89 | 1552.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 1539.10 | 1551.89 | 1552.02 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 1574.80 | 1553.77 | 1552.31 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 13:15:00 | 1544.75 | 1551.03 | 1551.59 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 1601.90 | 1559.54 | 1555.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 1604.95 | 1568.62 | 1559.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 1579.15 | 1587.18 | 1575.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 1579.15 | 1587.18 | 1575.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1579.15 | 1587.18 | 1575.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:45:00 | 1618.70 | 1593.79 | 1579.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 15:15:00 | 1582.50 | 1587.48 | 1587.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 15:15:00 | 1582.50 | 1587.48 | 1587.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 09:15:00 | 1569.90 | 1583.96 | 1585.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 14:15:00 | 1576.95 | 1574.33 | 1579.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 14:15:00 | 1576.95 | 1574.33 | 1579.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1576.95 | 1574.33 | 1579.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:30:00 | 1579.00 | 1574.33 | 1579.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1567.00 | 1572.97 | 1577.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 15:00:00 | 1560.15 | 1567.34 | 1572.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 13:15:00 | 1482.14 | 1510.47 | 1524.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-26 09:15:00 | 1404.14 | 1414.87 | 1431.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 101 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 1435.10 | 1416.51 | 1415.09 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 1404.30 | 1416.86 | 1416.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 14:15:00 | 1400.70 | 1408.75 | 1412.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 1383.70 | 1379.20 | 1390.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 11:00:00 | 1383.70 | 1379.20 | 1390.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1403.20 | 1375.23 | 1382.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:45:00 | 1403.05 | 1375.23 | 1382.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 1364.25 | 1373.03 | 1380.95 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 14:15:00 | 1398.00 | 1385.01 | 1384.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 1424.55 | 1402.64 | 1394.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 1437.35 | 1447.44 | 1428.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 10:00:00 | 1437.35 | 1447.44 | 1428.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1434.20 | 1439.78 | 1432.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 1429.20 | 1439.78 | 1432.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1433.45 | 1438.51 | 1432.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 1433.45 | 1438.51 | 1432.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 1433.35 | 1437.48 | 1432.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:45:00 | 1440.75 | 1438.09 | 1433.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:30:00 | 1437.65 | 1437.45 | 1433.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 1419.50 | 1433.39 | 1432.69 | SL hit (close<static) qty=1.00 sl=1431.15 alert=retest2 |

### Cycle 104 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 1425.75 | 1431.86 | 1432.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 09:15:00 | 1384.80 | 1417.20 | 1424.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 1407.35 | 1396.44 | 1407.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 1407.35 | 1396.44 | 1407.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1407.35 | 1396.44 | 1407.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 1408.45 | 1396.44 | 1407.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 1409.00 | 1398.95 | 1407.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 1409.10 | 1398.95 | 1407.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1397.00 | 1398.56 | 1406.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 1385.95 | 1399.35 | 1404.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1388.35 | 1391.02 | 1392.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 1318.93 | 1347.63 | 1363.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 1316.65 | 1345.61 | 1361.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 1347.00 | 1345.89 | 1359.81 | SL hit (close>ema200) qty=0.50 sl=1345.89 alert=retest2 |

### Cycle 105 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 1343.95 | 1325.00 | 1324.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1366.25 | 1341.43 | 1333.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 1400.05 | 1400.53 | 1383.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 11:00:00 | 1400.05 | 1400.53 | 1383.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1425.15 | 1433.38 | 1426.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 1425.15 | 1433.38 | 1426.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1437.40 | 1434.18 | 1427.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1434.25 | 1434.18 | 1427.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1432.00 | 1433.75 | 1428.19 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 1413.40 | 1425.46 | 1425.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 1407.90 | 1421.95 | 1424.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1429.60 | 1407.94 | 1412.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1429.60 | 1407.94 | 1412.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1429.60 | 1407.94 | 1412.92 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 12:15:00 | 1418.85 | 1416.46 | 1416.15 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 1409.90 | 1415.15 | 1415.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 1388.40 | 1409.80 | 1413.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 10:15:00 | 1408.80 | 1403.06 | 1408.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 10:15:00 | 1408.80 | 1403.06 | 1408.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1408.80 | 1403.06 | 1408.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:45:00 | 1427.30 | 1403.06 | 1408.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 1422.70 | 1406.99 | 1409.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 12:00:00 | 1422.70 | 1406.99 | 1409.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 12:15:00 | 1387.00 | 1402.99 | 1407.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:45:00 | 1368.45 | 1392.57 | 1402.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 1367.50 | 1381.24 | 1390.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:30:00 | 1368.00 | 1379.00 | 1387.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:15:00 | 1370.00 | 1378.68 | 1386.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1377.15 | 1369.80 | 1378.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1377.15 | 1369.80 | 1378.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1371.95 | 1370.23 | 1377.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:45:00 | 1371.70 | 1370.23 | 1377.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1366.80 | 1369.13 | 1374.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:30:00 | 1369.75 | 1369.13 | 1374.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1384.10 | 1370.07 | 1374.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:45:00 | 1384.95 | 1370.07 | 1374.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1379.90 | 1372.03 | 1374.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 14:15:00 | 1371.50 | 1374.56 | 1375.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 09:30:00 | 1366.10 | 1373.18 | 1374.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 09:45:00 | 1370.15 | 1363.30 | 1367.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 13:15:00 | 1368.50 | 1355.91 | 1355.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 1368.50 | 1355.91 | 1355.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 1375.80 | 1359.89 | 1357.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 1432.30 | 1434.30 | 1419.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 12:00:00 | 1432.30 | 1434.30 | 1419.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 1478.05 | 1483.54 | 1476.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:00:00 | 1478.05 | 1483.54 | 1476.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1461.85 | 1479.20 | 1475.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 1461.85 | 1479.20 | 1475.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 1462.00 | 1475.76 | 1474.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 1464.05 | 1475.76 | 1474.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 1443.95 | 1469.40 | 1471.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 14:15:00 | 1442.10 | 1453.35 | 1460.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 1431.75 | 1429.97 | 1440.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 1431.75 | 1429.97 | 1440.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1431.75 | 1429.97 | 1440.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:30:00 | 1439.45 | 1429.97 | 1440.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 1437.20 | 1431.69 | 1438.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 1441.50 | 1431.69 | 1438.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1429.55 | 1431.26 | 1438.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:15:00 | 1427.65 | 1431.26 | 1438.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 14:15:00 | 1443.00 | 1433.61 | 1438.49 | SL hit (close>static) qty=1.00 sl=1438.15 alert=retest2 |

### Cycle 111 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 1458.80 | 1441.24 | 1440.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 10:15:00 | 1465.65 | 1447.91 | 1444.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 1459.05 | 1465.65 | 1456.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 1459.05 | 1465.65 | 1456.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1459.05 | 1465.65 | 1456.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 1459.05 | 1465.65 | 1456.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 1458.70 | 1464.26 | 1457.10 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 1412.00 | 1446.77 | 1450.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 1406.85 | 1435.46 | 1444.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 1369.55 | 1368.88 | 1382.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 09:45:00 | 1374.00 | 1368.88 | 1382.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1377.50 | 1371.38 | 1381.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 1379.40 | 1371.38 | 1381.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1388.00 | 1374.71 | 1381.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 1388.00 | 1374.71 | 1381.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1392.50 | 1378.26 | 1382.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 1391.00 | 1378.26 | 1382.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 1392.05 | 1381.02 | 1383.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 1398.90 | 1381.02 | 1383.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 1400.85 | 1386.26 | 1385.73 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 1388.00 | 1392.52 | 1392.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 13:15:00 | 1386.50 | 1391.32 | 1392.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1334.05 | 1331.31 | 1350.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:45:00 | 1330.95 | 1331.31 | 1350.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 1245.55 | 1237.78 | 1250.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:45:00 | 1246.50 | 1237.78 | 1250.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1253.10 | 1241.76 | 1249.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 1253.10 | 1241.76 | 1249.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 1260.00 | 1245.41 | 1250.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 1242.25 | 1245.41 | 1250.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 1245.00 | 1244.84 | 1249.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 1247.35 | 1244.84 | 1249.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1234.80 | 1234.53 | 1241.68 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 1248.75 | 1240.70 | 1240.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 1258.20 | 1244.20 | 1241.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 1240.65 | 1249.09 | 1245.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 1240.65 | 1249.09 | 1245.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1240.65 | 1249.09 | 1245.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1240.65 | 1249.09 | 1245.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1248.40 | 1248.96 | 1246.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:30:00 | 1259.70 | 1246.90 | 1245.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 10:15:00 | 1259.30 | 1246.90 | 1245.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 12:15:00 | 1239.80 | 1246.06 | 1245.84 | SL hit (close<static) qty=1.00 sl=1240.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 1243.65 | 1245.58 | 1245.64 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 1263.75 | 1249.21 | 1247.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 1276.40 | 1256.06 | 1250.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 1261.25 | 1280.38 | 1268.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 1261.25 | 1280.38 | 1268.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1261.25 | 1280.38 | 1268.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:45:00 | 1264.00 | 1280.38 | 1268.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1266.20 | 1277.54 | 1268.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:30:00 | 1269.90 | 1275.49 | 1268.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:15:00 | 1268.80 | 1275.49 | 1268.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 15:15:00 | 1253.65 | 1265.16 | 1265.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 1253.65 | 1265.16 | 1265.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1209.50 | 1254.03 | 1260.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1180.00 | 1163.94 | 1187.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:45:00 | 1182.20 | 1163.94 | 1187.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1173.05 | 1165.76 | 1186.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:45:00 | 1179.95 | 1165.76 | 1186.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 1179.80 | 1168.54 | 1181.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 1179.80 | 1168.54 | 1181.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 1165.00 | 1167.83 | 1179.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 1175.00 | 1167.83 | 1179.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1187.40 | 1171.75 | 1180.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 1187.90 | 1171.75 | 1180.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1164.80 | 1170.36 | 1178.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:00:00 | 1160.05 | 1166.71 | 1174.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:45:00 | 1160.55 | 1166.36 | 1174.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 1180.95 | 1173.49 | 1173.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 1180.95 | 1173.49 | 1173.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 1183.80 | 1176.08 | 1174.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 15:15:00 | 1175.10 | 1177.91 | 1175.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 15:15:00 | 1175.10 | 1177.91 | 1175.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 1175.10 | 1177.91 | 1175.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 1166.75 | 1177.91 | 1175.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1168.95 | 1176.12 | 1175.32 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 1156.65 | 1172.22 | 1173.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 1156.00 | 1168.98 | 1172.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 1165.30 | 1163.92 | 1168.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 15:00:00 | 1165.30 | 1163.92 | 1168.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 121 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 1205.00 | 1171.73 | 1171.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 10:15:00 | 1213.00 | 1179.98 | 1175.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 13:15:00 | 1311.95 | 1312.48 | 1289.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 14:00:00 | 1311.95 | 1312.48 | 1289.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1297.10 | 1311.38 | 1295.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:45:00 | 1298.35 | 1311.38 | 1295.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1292.95 | 1307.70 | 1295.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1292.95 | 1307.70 | 1295.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1288.75 | 1303.91 | 1294.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 1288.75 | 1303.91 | 1294.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 1283.60 | 1299.85 | 1293.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 13:00:00 | 1283.60 | 1299.85 | 1293.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 1288.40 | 1297.56 | 1293.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:15:00 | 1283.60 | 1297.56 | 1293.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 1248.50 | 1283.67 | 1287.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1241.15 | 1267.56 | 1278.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 1223.50 | 1221.42 | 1241.09 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 09:15:00 | 1029.00 | 1221.42 | 1241.09 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 10:15:00 | 977.55 | 1141.45 | 1199.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-14 09:15:00 | 926.10 | 1004.29 | 1095.93 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 123 — BUY (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 14:15:00 | 824.85 | 816.38 | 815.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 09:15:00 | 841.65 | 821.94 | 818.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 14:15:00 | 823.35 | 828.10 | 823.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 14:15:00 | 823.35 | 828.10 | 823.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 823.35 | 828.10 | 823.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 823.35 | 828.10 | 823.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 821.90 | 826.86 | 823.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 825.40 | 826.86 | 823.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 808.05 | 823.10 | 821.67 | SL hit (close<static) qty=1.00 sl=817.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 806.30 | 819.74 | 820.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 804.60 | 816.71 | 818.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 783.75 | 778.65 | 790.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 783.75 | 778.65 | 790.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 783.75 | 778.65 | 790.62 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 811.00 | 796.82 | 795.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 818.00 | 802.98 | 800.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 822.80 | 825.24 | 816.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 822.80 | 825.24 | 816.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 814.95 | 822.92 | 816.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:30:00 | 828.30 | 823.53 | 817.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:30:00 | 827.95 | 823.01 | 817.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:15:00 | 826.90 | 822.39 | 818.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 09:15:00 | 810.85 | 820.59 | 818.75 | SL hit (close<static) qty=1.00 sl=812.85 alert=retest2 |

### Cycle 126 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 813.45 | 817.25 | 817.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 809.70 | 815.74 | 816.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 789.90 | 787.26 | 793.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 10:15:00 | 792.95 | 787.26 | 793.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 791.40 | 788.09 | 793.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:45:00 | 790.75 | 788.09 | 793.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 800.90 | 788.20 | 790.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 802.80 | 788.20 | 790.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 800.55 | 790.67 | 791.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 801.55 | 790.67 | 791.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 803.75 | 793.29 | 792.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 807.65 | 799.91 | 796.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 13:15:00 | 803.20 | 803.59 | 799.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 13:15:00 | 803.20 | 803.59 | 799.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 803.20 | 803.59 | 799.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:30:00 | 803.15 | 803.59 | 799.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 826.40 | 826.08 | 822.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:30:00 | 822.25 | 826.08 | 822.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 821.55 | 829.77 | 827.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 821.55 | 829.77 | 827.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 824.00 | 828.62 | 827.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 821.95 | 828.62 | 827.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 819.90 | 825.50 | 826.07 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 11:15:00 | 829.15 | 826.57 | 826.30 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 822.55 | 825.84 | 826.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 809.90 | 821.65 | 823.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 822.70 | 818.59 | 821.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 822.70 | 818.59 | 821.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 822.70 | 818.59 | 821.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 822.70 | 818.59 | 821.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 821.05 | 819.08 | 821.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 827.05 | 819.08 | 821.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 818.25 | 818.91 | 820.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:45:00 | 816.00 | 818.17 | 820.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 829.20 | 810.68 | 809.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 829.20 | 810.68 | 809.67 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 796.90 | 813.35 | 814.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 794.50 | 809.58 | 812.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 778.70 | 771.63 | 783.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 778.70 | 771.63 | 783.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 775.10 | 773.39 | 782.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 780.65 | 773.39 | 782.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 758.30 | 756.14 | 765.36 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 778.00 | 768.51 | 768.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 795.20 | 773.85 | 770.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 836.50 | 837.67 | 831.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 09:30:00 | 839.25 | 837.67 | 831.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 835.90 | 837.32 | 831.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:30:00 | 832.45 | 837.32 | 831.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 889.75 | 850.77 | 840.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:15:00 | 900.50 | 850.77 | 840.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 13:45:00 | 889.95 | 888.36 | 877.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 12:30:00 | 891.10 | 888.19 | 882.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 891.25 | 886.30 | 882.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 886.55 | 886.35 | 883.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:45:00 | 885.20 | 886.35 | 883.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 886.50 | 886.38 | 883.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 883.70 | 886.38 | 883.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 880.75 | 885.25 | 883.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:30:00 | 882.50 | 885.25 | 883.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 877.85 | 883.77 | 882.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:30:00 | 878.70 | 883.77 | 882.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 884.95 | 883.36 | 882.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 869.25 | 883.36 | 882.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 874.35 | 881.56 | 881.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 874.35 | 881.56 | 881.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 857.95 | 870.86 | 876.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 852.00 | 849.88 | 859.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 852.00 | 849.88 | 859.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 852.00 | 849.88 | 859.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 852.00 | 849.88 | 859.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 854.10 | 851.82 | 856.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:45:00 | 858.80 | 851.82 | 856.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 855.00 | 852.45 | 856.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 845.00 | 852.45 | 856.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 14:15:00 | 802.75 | 812.96 | 822.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 13:15:00 | 806.85 | 802.95 | 812.19 | SL hit (close>ema200) qty=0.50 sl=802.95 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 828.80 | 816.35 | 815.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 837.80 | 826.54 | 821.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 826.60 | 826.80 | 822.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:00:00 | 826.60 | 826.80 | 822.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 825.00 | 826.84 | 824.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 828.80 | 826.84 | 824.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:00:00 | 830.80 | 829.52 | 826.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:00:00 | 828.05 | 828.51 | 826.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:45:00 | 828.00 | 828.74 | 827.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 825.00 | 827.99 | 827.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 825.00 | 827.99 | 827.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 824.50 | 827.29 | 826.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 824.50 | 827.29 | 826.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-16 09:15:00 | 825.00 | 826.50 | 826.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 09:15:00 | 825.00 | 826.50 | 826.61 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 845.15 | 830.09 | 828.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 864.95 | 842.02 | 835.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 861.90 | 863.11 | 854.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 13:00:00 | 861.90 | 863.11 | 854.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 850.80 | 859.76 | 854.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 850.80 | 859.76 | 854.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 854.55 | 858.72 | 854.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 871.85 | 858.72 | 854.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 868.75 | 860.72 | 855.42 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 867.30 | 874.43 | 875.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 863.90 | 869.45 | 872.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 15:15:00 | 868.00 | 867.80 | 871.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 09:15:00 | 878.55 | 867.80 | 871.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 883.95 | 871.03 | 872.38 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 881.85 | 874.80 | 873.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 888.75 | 877.59 | 875.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 884.30 | 884.43 | 879.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 883.80 | 884.80 | 881.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 883.80 | 884.80 | 881.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 884.40 | 884.80 | 881.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 882.30 | 884.30 | 881.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 876.00 | 884.30 | 881.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 876.50 | 882.74 | 881.36 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 877.45 | 880.29 | 880.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 867.40 | 877.05 | 878.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 864.05 | 862.88 | 867.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:45:00 | 864.50 | 862.88 | 867.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 866.95 | 862.75 | 866.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 854.40 | 863.25 | 865.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:45:00 | 858.60 | 861.43 | 863.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 859.85 | 861.43 | 863.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 14:15:00 | 859.05 | 860.81 | 863.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 865.95 | 861.29 | 862.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 865.00 | 863.71 | 863.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 865.00 | 863.71 | 863.55 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 09:15:00 | 860.95 | 863.26 | 863.41 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 884.05 | 867.42 | 865.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 11:15:00 | 902.00 | 884.16 | 876.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 919.50 | 925.11 | 908.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 919.50 | 925.11 | 908.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 916.35 | 922.71 | 911.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 922.50 | 922.71 | 911.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 897.15 | 909.81 | 910.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 10:15:00 | 897.15 | 909.81 | 910.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 888.80 | 899.97 | 904.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 15:15:00 | 878.95 | 878.13 | 885.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-19 09:15:00 | 880.80 | 878.13 | 885.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 881.40 | 874.77 | 880.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 881.40 | 874.77 | 880.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 874.50 | 874.72 | 879.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:45:00 | 871.95 | 874.98 | 879.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 872.85 | 874.98 | 879.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 872.85 | 874.95 | 878.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 870.50 | 874.76 | 878.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 875.75 | 874.00 | 877.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 875.75 | 874.00 | 877.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 873.10 | 870.35 | 872.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 872.70 | 870.35 | 872.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 868.85 | 870.05 | 872.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 882.85 | 874.43 | 873.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 882.85 | 874.43 | 873.41 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 12:15:00 | 870.50 | 874.23 | 874.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 866.80 | 872.12 | 873.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 13:15:00 | 877.60 | 873.21 | 873.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 13:15:00 | 877.60 | 873.21 | 873.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 877.60 | 873.21 | 873.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:00:00 | 877.60 | 873.21 | 873.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 881.80 | 874.93 | 874.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 909.70 | 883.61 | 878.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 918.25 | 924.62 | 913.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 09:45:00 | 919.20 | 924.62 | 913.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 961.90 | 931.85 | 922.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 14:00:00 | 970.55 | 950.58 | 935.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 971.70 | 974.41 | 971.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 975.70 | 994.36 | 995.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 975.70 | 994.36 | 995.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 971.80 | 989.85 | 992.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 975.15 | 974.47 | 981.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 975.15 | 974.47 | 981.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 978.20 | 975.21 | 981.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:30:00 | 980.55 | 975.21 | 981.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 978.45 | 975.86 | 981.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:30:00 | 981.65 | 975.86 | 981.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 977.15 | 976.12 | 980.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:30:00 | 981.00 | 976.12 | 980.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 977.00 | 976.79 | 980.21 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1011.05 | 986.81 | 984.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1018.45 | 1002.75 | 993.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 14:15:00 | 1033.40 | 1036.12 | 1027.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 15:00:00 | 1033.40 | 1036.12 | 1027.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1021.30 | 1032.98 | 1027.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1022.80 | 1032.98 | 1027.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1025.00 | 1031.38 | 1027.24 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 1013.95 | 1024.00 | 1024.56 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 1030.75 | 1025.18 | 1024.63 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1020.65 | 1024.23 | 1024.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1016.60 | 1022.71 | 1023.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 1030.05 | 1022.29 | 1023.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 1030.05 | 1022.29 | 1023.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1030.05 | 1022.29 | 1023.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 1030.05 | 1022.29 | 1023.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 1031.65 | 1024.16 | 1023.79 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 1017.45 | 1022.82 | 1023.22 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 1035.90 | 1024.83 | 1024.02 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 1009.80 | 1023.37 | 1023.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 1004.55 | 1019.61 | 1021.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 959.40 | 953.71 | 967.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 959.40 | 953.71 | 967.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 959.95 | 956.54 | 966.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 966.70 | 956.54 | 966.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 964.85 | 958.76 | 965.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:00:00 | 958.80 | 961.87 | 965.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 15:15:00 | 981.00 | 967.51 | 967.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 981.00 | 967.51 | 967.42 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 956.15 | 965.24 | 966.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 940.00 | 955.53 | 960.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 936.15 | 935.87 | 944.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 936.15 | 935.87 | 944.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 927.35 | 933.97 | 940.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 923.00 | 929.58 | 935.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 11:15:00 | 943.90 | 930.55 | 930.89 | SL hit (close>static) qty=1.00 sl=943.35 alert=retest2 |

### Cycle 159 — BUY (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 12:15:00 | 944.25 | 933.29 | 932.10 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 923.00 | 932.74 | 932.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 920.75 | 928.70 | 930.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 918.00 | 907.50 | 914.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 918.00 | 907.50 | 914.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 918.00 | 907.50 | 914.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 918.00 | 907.50 | 914.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 915.60 | 909.12 | 914.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:15:00 | 903.05 | 909.12 | 914.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 15:15:00 | 889.05 | 887.60 | 887.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 15:15:00 | 889.05 | 887.60 | 887.41 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 881.95 | 886.47 | 886.92 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 891.15 | 887.40 | 887.30 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 885.00 | 887.13 | 887.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 884.15 | 886.53 | 886.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 886.45 | 884.92 | 885.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 886.45 | 884.92 | 885.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 886.45 | 884.92 | 885.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 880.45 | 883.83 | 884.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 880.85 | 883.23 | 884.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 878.80 | 867.85 | 866.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 878.80 | 867.85 | 866.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 880.50 | 870.38 | 868.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 864.30 | 870.28 | 868.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 864.30 | 870.28 | 868.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 864.30 | 870.28 | 868.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 864.30 | 870.28 | 868.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 867.95 | 869.81 | 868.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 865.00 | 869.81 | 868.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 865.00 | 868.85 | 868.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 865.50 | 868.85 | 868.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 863.30 | 867.74 | 867.95 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 871.20 | 868.17 | 867.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 15:15:00 | 874.10 | 869.36 | 868.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 868.25 | 869.46 | 868.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 868.25 | 869.46 | 868.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 868.25 | 869.46 | 868.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 868.25 | 869.46 | 868.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 864.35 | 868.43 | 868.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 864.60 | 868.43 | 868.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 862.30 | 867.21 | 867.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 858.85 | 865.54 | 866.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 863.40 | 862.39 | 864.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 863.40 | 862.39 | 864.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 863.40 | 862.39 | 864.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 863.40 | 862.39 | 864.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 858.60 | 861.63 | 864.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:30:00 | 855.45 | 859.99 | 863.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 856.50 | 856.01 | 859.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 852.00 | 847.35 | 848.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 857.15 | 850.63 | 850.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 857.15 | 850.63 | 850.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 861.00 | 855.04 | 852.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 856.25 | 857.93 | 855.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 856.25 | 857.93 | 855.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 856.25 | 857.93 | 855.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 855.00 | 857.93 | 855.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 850.40 | 856.43 | 854.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 850.40 | 856.43 | 854.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 856.80 | 856.50 | 854.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 858.50 | 854.35 | 854.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 852.70 | 854.27 | 854.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 852.70 | 854.27 | 854.35 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 855.00 | 854.42 | 854.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 858.45 | 855.22 | 854.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 854.95 | 856.61 | 855.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 854.95 | 856.61 | 855.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 854.95 | 856.61 | 855.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 854.95 | 856.61 | 855.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 856.85 | 856.66 | 855.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 858.45 | 856.66 | 855.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 865.00 | 873.56 | 873.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 865.00 | 873.56 | 873.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 857.80 | 869.04 | 871.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 12:15:00 | 853.30 | 852.92 | 856.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:00:00 | 853.30 | 852.92 | 856.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 807.70 | 800.61 | 808.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 809.75 | 800.61 | 808.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 805.05 | 801.50 | 808.40 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 819.00 | 810.48 | 810.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 822.80 | 815.31 | 812.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 814.45 | 816.30 | 813.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 814.45 | 816.30 | 813.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 814.45 | 816.30 | 813.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 814.45 | 816.30 | 813.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 812.45 | 815.53 | 813.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 811.90 | 815.53 | 813.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 810.50 | 814.52 | 813.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 810.75 | 814.52 | 813.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 815.50 | 814.07 | 813.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 815.50 | 814.07 | 813.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 812.15 | 814.40 | 813.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 812.15 | 814.40 | 813.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 811.75 | 813.87 | 813.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 816.55 | 814.40 | 813.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 813.80 | 814.28 | 813.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 15:15:00 | 811.95 | 813.34 | 813.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 811.95 | 813.34 | 813.43 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 818.70 | 814.41 | 813.91 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 807.40 | 812.46 | 813.08 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 813.00 | 812.18 | 812.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 843.60 | 818.83 | 815.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 15:15:00 | 839.00 | 839.06 | 833.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:15:00 | 837.75 | 839.06 | 833.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 818.15 | 834.87 | 831.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 818.15 | 834.87 | 831.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 816.35 | 831.17 | 830.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:15:00 | 814.10 | 831.17 | 830.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 808.40 | 826.62 | 828.34 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 833.85 | 822.70 | 821.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 842.85 | 834.42 | 829.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 836.20 | 836.91 | 833.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 836.20 | 836.91 | 833.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 832.85 | 836.10 | 833.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 832.85 | 836.10 | 833.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 832.60 | 835.40 | 833.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 832.60 | 835.40 | 833.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 836.30 | 835.58 | 833.37 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 826.25 | 831.58 | 832.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 14:15:00 | 823.90 | 829.05 | 830.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 831.05 | 829.24 | 830.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 831.05 | 829.24 | 830.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 831.05 | 829.24 | 830.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 830.10 | 829.24 | 830.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 824.25 | 828.24 | 829.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 822.60 | 827.46 | 829.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:15:00 | 823.25 | 826.67 | 828.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 828.50 | 828.12 | 828.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 828.50 | 828.12 | 828.07 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 826.40 | 827.73 | 827.90 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 833.55 | 828.89 | 828.35 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 821.00 | 827.24 | 827.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 819.60 | 825.71 | 827.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 827.30 | 824.25 | 825.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 11:15:00 | 827.30 | 824.25 | 825.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 827.30 | 824.25 | 825.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 827.30 | 824.25 | 825.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 829.25 | 825.25 | 825.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:45:00 | 829.30 | 825.25 | 825.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 832.05 | 826.61 | 826.51 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 820.00 | 826.80 | 826.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 812.60 | 820.10 | 823.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 802.80 | 801.80 | 807.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:45:00 | 803.65 | 801.80 | 807.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 801.25 | 801.80 | 806.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:30:00 | 799.10 | 801.37 | 806.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:45:00 | 799.30 | 801.00 | 805.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:30:00 | 799.85 | 799.08 | 802.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 811.95 | 798.29 | 799.89 | SL hit (close>static) qty=1.00 sl=808.10 alert=retest2 |

### Cycle 187 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 816.55 | 803.64 | 802.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 824.00 | 809.25 | 805.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 822.35 | 824.12 | 818.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:30:00 | 825.20 | 824.12 | 818.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 818.50 | 823.63 | 818.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 815.95 | 823.63 | 818.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 823.20 | 823.54 | 819.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:15:00 | 818.40 | 823.54 | 819.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 816.90 | 822.21 | 819.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 811.45 | 822.21 | 819.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 813.85 | 820.54 | 818.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:30:00 | 814.80 | 820.54 | 818.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 813.00 | 819.03 | 818.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 816.75 | 819.03 | 818.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 814.70 | 818.07 | 817.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 814.70 | 818.07 | 817.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 818.75 | 818.21 | 817.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:30:00 | 824.10 | 819.28 | 818.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 14:15:00 | 821.80 | 819.43 | 818.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 817.70 | 818.38 | 818.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 817.70 | 818.38 | 818.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 815.00 | 817.70 | 818.08 | Break + close below crossover candle low |

### Cycle 189 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 832.60 | 819.50 | 818.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 862.00 | 835.38 | 827.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 10:15:00 | 860.40 | 862.45 | 850.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:45:00 | 859.70 | 862.45 | 850.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 861.80 | 864.03 | 856.73 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 842.00 | 853.66 | 853.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 836.00 | 848.07 | 851.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 847.05 | 846.83 | 850.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 847.05 | 846.83 | 850.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 876.00 | 849.19 | 849.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 876.00 | 849.19 | 849.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 879.30 | 855.22 | 852.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 894.55 | 863.08 | 856.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 889.75 | 895.81 | 877.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:00:00 | 889.75 | 895.81 | 877.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 878.85 | 887.19 | 878.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 877.90 | 887.19 | 878.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 879.50 | 885.65 | 878.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 887.00 | 884.94 | 879.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 12:45:00 | 886.20 | 886.30 | 881.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:45:00 | 885.10 | 885.83 | 881.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:30:00 | 884.80 | 884.89 | 881.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 879.00 | 883.72 | 881.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 917.95 | 883.72 | 881.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 912.90 | 931.82 | 931.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 912.90 | 931.82 | 931.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 906.00 | 926.65 | 929.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 903.50 | 885.71 | 894.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 903.50 | 885.71 | 894.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 903.50 | 885.71 | 894.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 903.50 | 885.71 | 894.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 911.80 | 890.93 | 896.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 911.80 | 890.93 | 896.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 916.30 | 896.00 | 897.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 916.35 | 896.00 | 897.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 925.00 | 901.80 | 900.42 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 914.50 | 917.21 | 917.48 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 929.70 | 918.98 | 918.19 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 920.95 | 922.81 | 922.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 917.00 | 921.65 | 922.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 902.85 | 898.95 | 904.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 902.85 | 898.95 | 904.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 902.85 | 898.95 | 904.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 887.05 | 901.93 | 902.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 910.80 | 898.14 | 897.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 910.80 | 898.14 | 897.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 925.70 | 909.77 | 904.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 938.90 | 941.91 | 931.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 938.90 | 941.91 | 931.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 938.90 | 941.91 | 931.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 930.15 | 941.91 | 931.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 923.25 | 938.18 | 931.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 923.25 | 938.18 | 931.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 917.55 | 934.06 | 929.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 917.55 | 934.06 | 929.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 911.55 | 924.31 | 925.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 898.10 | 914.08 | 919.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 873.15 | 870.10 | 886.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:15:00 | 876.50 | 870.10 | 886.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 877.05 | 872.56 | 879.63 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 893.00 | 882.23 | 882.19 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 875.85 | 881.81 | 882.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 872.85 | 877.76 | 879.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 836.95 | 834.89 | 846.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 840.00 | 836.82 | 842.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 840.00 | 836.82 | 842.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 843.00 | 836.82 | 842.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 838.20 | 837.09 | 841.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 835.95 | 837.09 | 841.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 835.45 | 835.32 | 838.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 837.00 | 836.35 | 838.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 830.50 | 819.88 | 819.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 830.50 | 819.88 | 819.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 835.40 | 822.99 | 821.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 827.65 | 829.65 | 825.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 827.65 | 829.65 | 825.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 827.65 | 829.65 | 825.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:00:00 | 838.90 | 831.50 | 826.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 824.80 | 830.12 | 827.39 | SL hit (close<static) qty=1.00 sl=825.20 alert=retest2 |

### Cycle 202 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 824.55 | 825.65 | 825.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 11:15:00 | 816.00 | 822.62 | 824.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 823.40 | 821.70 | 823.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 823.40 | 821.70 | 823.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 823.40 | 821.70 | 823.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 823.40 | 821.70 | 823.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 823.00 | 821.96 | 823.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 849.10 | 821.96 | 823.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 848.45 | 827.26 | 825.58 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 831.45 | 838.73 | 839.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 828.15 | 833.75 | 836.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 15:15:00 | 824.10 | 823.94 | 828.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:15:00 | 826.55 | 823.94 | 828.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 832.00 | 825.55 | 829.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 832.00 | 825.55 | 829.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 840.85 | 828.61 | 830.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 840.85 | 828.61 | 830.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 842.25 | 833.34 | 832.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 845.55 | 837.22 | 834.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 840.10 | 841.74 | 838.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:30:00 | 841.00 | 841.74 | 838.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 857.00 | 851.82 | 847.04 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 833.55 | 846.62 | 846.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 826.65 | 842.63 | 844.93 | Break + close below crossover candle low |

### Cycle 207 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 917.00 | 847.54 | 844.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 10:15:00 | 927.30 | 863.49 | 851.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 15:15:00 | 878.20 | 881.18 | 866.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:15:00 | 887.40 | 881.18 | 866.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:00:00 | 886.00 | 882.14 | 868.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 15:15:00 | 886.00 | 882.02 | 873.53 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 872.80 | 880.81 | 874.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 872.80 | 880.81 | 874.50 | SL hit (close<ema400) qty=1.00 sl=874.50 alert=retest1 |

### Cycle 208 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 944.45 | 963.71 | 964.78 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 09:15:00 | 986.00 | 964.66 | 964.41 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 960.50 | 963.83 | 964.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 951.10 | 961.28 | 962.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 982.30 | 960.77 | 961.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 982.30 | 960.77 | 961.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 982.30 | 960.77 | 961.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:15:00 | 990.45 | 960.77 | 961.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 997.85 | 968.18 | 964.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 1012.45 | 993.34 | 980.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1004.95 | 1011.41 | 998.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1004.95 | 1011.41 | 998.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1004.95 | 1011.41 | 998.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:15:00 | 997.30 | 1011.41 | 998.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 1000.50 | 1009.23 | 998.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 998.80 | 1009.23 | 998.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 997.95 | 1006.98 | 998.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:00:00 | 997.95 | 1006.98 | 998.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 1010.50 | 1007.68 | 999.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 1047.10 | 1005.95 | 1000.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 1013.20 | 1017.92 | 1017.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 984.60 | 1012.59 | 1015.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 984.60 | 1012.59 | 1015.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 976.50 | 1005.37 | 1012.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 944.35 | 942.95 | 956.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 15:00:00 | 944.35 | 942.95 | 956.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 952.00 | 945.17 | 955.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 954.50 | 945.17 | 955.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 970.30 | 950.20 | 956.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 970.30 | 950.20 | 956.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 970.25 | 954.21 | 957.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:00:00 | 967.45 | 959.99 | 960.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:30:00 | 965.60 | 959.55 | 959.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 962.50 | 960.34 | 960.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 09:15:00 | 962.50 | 960.34 | 960.18 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 951.60 | 958.59 | 959.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 942.55 | 953.79 | 956.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 960.70 | 951.63 | 954.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 960.70 | 951.63 | 954.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 960.70 | 951.63 | 954.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 962.85 | 951.63 | 954.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 962.20 | 953.75 | 955.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 964.00 | 953.75 | 955.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 958.00 | 956.67 | 956.51 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 941.75 | 954.52 | 955.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 936.90 | 951.00 | 953.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 940.20 | 935.57 | 942.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 940.20 | 935.57 | 942.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 951.75 | 938.81 | 943.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 951.75 | 938.81 | 943.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 953.00 | 941.65 | 944.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 952.50 | 941.65 | 944.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 945.00 | 943.78 | 944.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 978.65 | 943.78 | 944.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 972.70 | 949.57 | 947.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 09:15:00 | 1004.45 | 978.75 | 969.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 13:15:00 | 985.30 | 989.58 | 978.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 13:45:00 | 985.00 | 989.58 | 978.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 975.85 | 986.84 | 978.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 975.85 | 986.84 | 978.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 975.00 | 984.47 | 977.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1010.00 | 984.47 | 977.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 12:15:00 | 1111.00 | 1090.24 | 1063.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 1082.05 | 1097.28 | 1097.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 13:15:00 | 1080.25 | 1091.55 | 1095.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 13:15:00 | 1085.25 | 1083.89 | 1088.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 13:45:00 | 1085.00 | 1083.89 | 1088.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 1092.30 | 1085.57 | 1088.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 15:00:00 | 1092.30 | 1085.57 | 1088.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 1092.00 | 1086.86 | 1089.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:15:00 | 1086.10 | 1086.86 | 1089.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1080.40 | 1085.57 | 1088.32 | EMA400 retest candle locked (from downside) |

### Cycle 219 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1123.00 | 1092.02 | 1089.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 1128.20 | 1115.74 | 1105.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 10:15:00 | 1113.80 | 1115.35 | 1105.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 11:00:00 | 1113.80 | 1115.35 | 1105.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 11:15:00 | 1107.00 | 1113.68 | 1106.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:30:00 | 1109.25 | 1113.68 | 1106.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 1105.75 | 1112.10 | 1106.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:00:00 | 1105.75 | 1112.10 | 1106.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 1101.10 | 1109.90 | 1105.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:30:00 | 1101.90 | 1109.90 | 1105.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 1102.45 | 1108.41 | 1105.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:15:00 | 1100.00 | 1108.41 | 1105.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1100.00 | 1106.73 | 1104.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 1077.00 | 1106.73 | 1104.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 1083.75 | 1102.13 | 1102.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 1061.95 | 1087.00 | 1095.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 11:15:00 | 1072.85 | 1072.01 | 1082.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 12:00:00 | 1072.85 | 1072.01 | 1082.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1092.50 | 1075.96 | 1080.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 1092.50 | 1075.96 | 1080.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1081.80 | 1077.13 | 1080.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:15:00 | 1079.30 | 1077.13 | 1080.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 1078.40 | 1075.83 | 1078.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:30:00 | 1078.00 | 1075.68 | 1077.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 1094.00 | 1080.21 | 1079.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 13:15:00 | 1094.00 | 1080.21 | 1079.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 1102.00 | 1088.01 | 1083.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 14:15:00 | 1090.15 | 1091.33 | 1087.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 14:45:00 | 1089.95 | 1091.33 | 1087.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 1092.00 | 1091.47 | 1087.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 1099.70 | 1091.47 | 1087.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:45:00 | 1094.35 | 1093.87 | 1089.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 1096.50 | 1094.39 | 1090.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:30:00 | 1094.10 | 1093.98 | 1090.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1095.15 | 1096.40 | 1093.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:30:00 | 1094.80 | 1096.40 | 1093.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1084.00 | 1093.92 | 1092.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 1084.00 | 1093.92 | 1092.70 | SL hit (close<static) qty=1.00 sl=1086.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-15 09:15:00 | 964.95 | 2024-04-15 09:15:00 | 983.25 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-04-22 09:15:00 | 1013.00 | 2024-04-24 15:15:00 | 1000.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-04-22 10:45:00 | 1005.20 | 2024-04-24 15:15:00 | 1000.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-04-22 15:15:00 | 1009.90 | 2024-04-24 15:15:00 | 1000.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-05-10 12:15:00 | 968.15 | 2024-05-14 10:15:00 | 988.95 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-05-10 12:45:00 | 968.85 | 2024-05-14 10:15:00 | 988.95 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-05-18 09:15:00 | 999.55 | 2024-05-28 09:15:00 | 1099.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-21 09:15:00 | 999.20 | 2024-05-28 09:15:00 | 1099.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-21 10:45:00 | 998.40 | 2024-05-28 09:15:00 | 1098.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-22 09:45:00 | 998.55 | 2024-05-28 09:15:00 | 1098.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-27 09:15:00 | 1033.50 | 2024-05-30 10:15:00 | 1027.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-05-30 10:15:00 | 1026.35 | 2024-05-30 10:15:00 | 1027.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-06-04 12:15:00 | 957.85 | 2024-06-05 09:15:00 | 1022.90 | STOP_HIT | 1.00 | -6.79% |
| SELL | retest2 | 2024-06-04 15:00:00 | 969.80 | 2024-06-05 09:15:00 | 1022.90 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2024-07-10 10:00:00 | 1186.00 | 2024-07-15 10:15:00 | 1216.95 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-07-11 10:30:00 | 1194.10 | 2024-07-15 10:15:00 | 1216.95 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-07-11 11:00:00 | 1193.05 | 2024-07-15 10:15:00 | 1216.95 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-07-11 11:45:00 | 1191.70 | 2024-07-15 10:15:00 | 1216.95 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-07-16 09:15:00 | 1215.00 | 2024-07-19 12:15:00 | 1219.20 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-07-16 10:00:00 | 1213.00 | 2024-07-19 12:15:00 | 1219.20 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1175.05 | 2024-07-23 14:15:00 | 1256.00 | STOP_HIT | 1.00 | -6.89% |
| BUY | retest2 | 2024-07-31 09:15:00 | 1355.65 | 2024-08-05 10:15:00 | 1316.60 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-07-31 12:15:00 | 1355.75 | 2024-08-05 10:15:00 | 1316.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-08-02 10:00:00 | 1357.10 | 2024-08-05 10:15:00 | 1316.60 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-08-12 10:45:00 | 1454.85 | 2024-08-16 13:15:00 | 1463.30 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2024-08-30 11:15:00 | 1538.90 | 2024-08-30 13:15:00 | 1549.90 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-08-30 11:45:00 | 1536.30 | 2024-08-30 13:15:00 | 1549.90 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-09-06 11:45:00 | 1564.00 | 2024-09-09 11:15:00 | 1539.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-09-09 10:30:00 | 1557.75 | 2024-09-09 11:15:00 | 1539.10 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-12 10:45:00 | 1618.70 | 2024-09-13 15:15:00 | 1582.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-09-17 15:00:00 | 1560.15 | 2024-09-20 13:15:00 | 1482.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 15:00:00 | 1560.15 | 2024-09-26 09:15:00 | 1404.14 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-11 12:45:00 | 1440.75 | 2024-10-14 09:15:00 | 1419.50 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-10-11 14:30:00 | 1437.65 | 2024-10-14 09:15:00 | 1419.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-10-17 09:15:00 | 1385.95 | 2024-10-22 15:15:00 | 1318.93 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1388.35 | 2024-10-23 09:15:00 | 1316.65 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2024-10-17 09:15:00 | 1385.95 | 2024-10-23 10:15:00 | 1347.00 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1388.35 | 2024-10-23 10:15:00 | 1347.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2024-11-13 14:45:00 | 1368.45 | 2024-11-29 13:15:00 | 1368.50 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2024-11-18 09:15:00 | 1367.50 | 2024-11-29 13:15:00 | 1368.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-11-18 10:30:00 | 1368.00 | 2024-11-29 13:15:00 | 1368.50 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-11-18 12:15:00 | 1370.00 | 2024-11-29 13:15:00 | 1368.50 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-11-21 14:15:00 | 1371.50 | 2024-11-29 13:15:00 | 1368.50 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2024-11-22 09:30:00 | 1366.10 | 2024-11-29 13:15:00 | 1368.50 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-11-25 09:45:00 | 1370.15 | 2024-11-29 13:15:00 | 1368.50 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-12-17 14:15:00 | 1427.65 | 2024-12-17 14:15:00 | 1443.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-01-22 09:30:00 | 1259.70 | 2025-01-22 12:15:00 | 1239.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-01-22 10:15:00 | 1259.30 | 2025-01-22 12:15:00 | 1239.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-01-24 11:30:00 | 1269.90 | 2025-01-24 15:15:00 | 1253.65 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-01-24 12:15:00 | 1268.80 | 2025-01-24 15:15:00 | 1253.65 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-01-30 14:00:00 | 1160.05 | 2025-02-01 09:15:00 | 1180.95 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-01-30 14:45:00 | 1160.55 | 2025-02-01 09:15:00 | 1180.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest1 | 2025-02-13 09:15:00 | 1029.00 | 2025-02-13 10:15:00 | 977.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 09:15:00 | 1029.00 | 2025-02-14 09:15:00 | 926.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-24 14:00:00 | 824.85 | 2025-02-24 14:15:00 | 824.85 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-02-27 09:15:00 | 825.40 | 2025-02-27 09:15:00 | 808.05 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-03-07 10:30:00 | 828.30 | 2025-03-10 09:15:00 | 810.85 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-03-07 11:30:00 | 827.95 | 2025-03-10 09:15:00 | 810.85 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-03-07 14:15:00 | 826.90 | 2025-03-10 09:15:00 | 810.85 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-03-28 11:45:00 | 816.00 | 2025-04-03 09:15:00 | 829.20 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-04-24 10:15:00 | 900.50 | 2025-04-30 09:15:00 | 874.35 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-04-25 13:45:00 | 889.95 | 2025-04-30 09:15:00 | 874.35 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-04-28 12:30:00 | 891.10 | 2025-04-30 09:15:00 | 874.35 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-04-29 09:15:00 | 891.25 | 2025-04-30 09:15:00 | 874.35 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-05-06 09:15:00 | 845.00 | 2025-05-08 14:15:00 | 802.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 845.00 | 2025-05-09 13:15:00 | 806.85 | STOP_HIT | 0.50 | 4.51% |
| BUY | retest2 | 2025-05-14 09:15:00 | 828.80 | 2025-05-16 09:15:00 | 825.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-05-14 13:00:00 | 830.80 | 2025-05-16 09:15:00 | 825.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-05-14 15:00:00 | 828.05 | 2025-05-16 09:15:00 | 825.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-05-15 12:45:00 | 828.00 | 2025-05-16 09:15:00 | 825.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-06-06 09:15:00 | 854.40 | 2025-06-09 12:15:00 | 865.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-06-06 10:45:00 | 858.60 | 2025-06-09 12:15:00 | 865.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-06 11:15:00 | 859.85 | 2025-06-09 12:15:00 | 865.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-06 14:15:00 | 859.05 | 2025-06-09 12:15:00 | 865.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-06-13 10:15:00 | 922.50 | 2025-06-16 10:15:00 | 897.15 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-06-20 09:45:00 | 871.95 | 2025-06-25 10:15:00 | 882.85 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-20 10:15:00 | 872.85 | 2025-06-25 10:15:00 | 882.85 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-20 11:15:00 | 872.85 | 2025-06-25 10:15:00 | 882.85 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-20 12:15:00 | 870.50 | 2025-06-25 10:15:00 | 882.85 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-03 14:00:00 | 970.55 | 2025-07-11 10:15:00 | 975.70 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-07-08 11:45:00 | 971.70 | 2025-07-11 10:15:00 | 975.70 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-07-30 14:00:00 | 958.80 | 2025-07-30 15:15:00 | 981.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-08-06 09:30:00 | 923.00 | 2025-08-07 11:15:00 | 943.90 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-08-12 11:15:00 | 903.05 | 2025-08-20 15:15:00 | 889.05 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2025-08-25 09:15:00 | 880.45 | 2025-09-02 09:15:00 | 878.80 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-08-25 10:00:00 | 880.85 | 2025-09-02 09:15:00 | 878.80 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-09-05 11:30:00 | 855.45 | 2025-09-11 10:15:00 | 857.15 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-09-08 09:45:00 | 856.50 | 2025-09-11 10:15:00 | 857.15 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-09-10 14:45:00 | 852.00 | 2025-09-11 10:15:00 | 857.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-09-16 09:15:00 | 858.50 | 2025-09-16 14:15:00 | 852.70 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-17 15:15:00 | 858.45 | 2025-09-22 14:15:00 | 865.00 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-10-07 13:00:00 | 816.55 | 2025-10-07 15:15:00 | 811.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-07 14:00:00 | 813.80 | 2025-10-07 15:15:00 | 811.95 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-10-27 12:15:00 | 822.60 | 2025-10-29 12:15:00 | 828.50 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-27 13:15:00 | 823.25 | 2025-10-29 12:15:00 | 828.50 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-10 10:30:00 | 799.10 | 2025-11-12 09:15:00 | 811.95 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-11-10 11:45:00 | 799.30 | 2025-11-12 09:15:00 | 811.95 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-11-11 10:30:00 | 799.85 | 2025-11-12 09:15:00 | 811.95 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-11-17 12:30:00 | 824.10 | 2025-11-18 11:15:00 | 817.70 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-17 14:15:00 | 821.80 | 2025-11-18 11:15:00 | 817.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-11-28 10:15:00 | 887.00 | 2025-12-08 09:15:00 | 912.90 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-11-28 12:45:00 | 886.20 | 2025-12-08 09:15:00 | 912.90 | STOP_HIT | 1.00 | 3.01% |
| BUY | retest2 | 2025-11-28 13:45:00 | 885.10 | 2025-12-08 09:15:00 | 912.90 | STOP_HIT | 1.00 | 3.14% |
| BUY | retest2 | 2025-11-28 14:30:00 | 884.80 | 2025-12-08 09:15:00 | 912.90 | STOP_HIT | 1.00 | 3.18% |
| BUY | retest2 | 2025-12-01 09:15:00 | 917.95 | 2025-12-08 09:15:00 | 912.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-01-01 09:15:00 | 887.05 | 2026-01-05 09:15:00 | 910.80 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-01-22 10:15:00 | 835.95 | 2026-01-30 10:15:00 | 830.50 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2026-01-23 09:30:00 | 835.45 | 2026-01-30 10:15:00 | 830.50 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2026-01-23 12:00:00 | 837.00 | 2026-01-30 10:15:00 | 830.50 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2026-02-01 11:00:00 | 838.90 | 2026-02-01 13:15:00 | 824.80 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2026-02-17 09:15:00 | 887.40 | 2026-02-18 09:15:00 | 872.80 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest1 | 2026-02-17 10:00:00 | 886.00 | 2026-02-18 09:15:00 | 872.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest1 | 2026-02-17 15:15:00 | 886.00 | 2026-02-18 09:15:00 | 872.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-18 11:45:00 | 884.30 | 2026-02-25 09:15:00 | 972.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-20 10:15:00 | 890.05 | 2026-02-25 09:15:00 | 979.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-10 09:15:00 | 1047.10 | 2026-03-13 09:15:00 | 984.60 | STOP_HIT | 1.00 | -5.97% |
| BUY | retest2 | 2026-03-12 11:00:00 | 1013.20 | 2026-03-13 09:15:00 | 984.60 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2026-03-18 14:00:00 | 967.45 | 2026-03-19 09:15:00 | 962.50 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2026-03-18 14:30:00 | 965.60 | 2026-03-19 09:15:00 | 962.50 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1010.00 | 2026-04-08 12:15:00 | 1111.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-23 11:15:00 | 1079.30 | 2026-04-24 13:15:00 | 1094.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-23 13:30:00 | 1078.40 | 2026-04-24 13:15:00 | 1094.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-04-24 09:30:00 | 1078.00 | 2026-04-24 13:15:00 | 1094.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-04-28 09:15:00 | 1099.70 | 2026-04-29 13:15:00 | 1084.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-04-28 10:45:00 | 1094.35 | 2026-04-29 13:15:00 | 1084.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-04-28 12:00:00 | 1096.50 | 2026-04-29 13:15:00 | 1084.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-04-28 14:30:00 | 1094.10 | 2026-04-29 13:15:00 | 1084.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-04-30 09:30:00 | 1105.00 | 2026-05-08 09:15:00 | 1215.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 10:30:00 | 1098.00 | 2026-05-08 09:15:00 | 1207.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 11:30:00 | 1100.45 | 2026-05-08 09:15:00 | 1210.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 13:30:00 | 1099.45 | 2026-05-08 09:15:00 | 1209.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1102.20 | 2026-05-08 09:15:00 | 1212.42 | TARGET_HIT | 1.00 | 10.00% |
