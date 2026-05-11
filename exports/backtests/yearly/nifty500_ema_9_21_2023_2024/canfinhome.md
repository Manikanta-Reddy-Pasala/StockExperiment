# Can Fin Homes Ltd. (CANFINHOME)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 878.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 234 |
| ALERT1 | 178 |
| ALERT2 | 177 |
| ALERT2_SKIP | 90 |
| ALERT3 | 497 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 205 |
| PARTIAL | 14 |
| TARGET_HIT | 11 |
| STOP_HIT | 205 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 228 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 70 / 158
- **Target hits / Stop hits / Partials:** 11 / 203 / 14
- **Avg / median % per leg:** 0.13% / -0.85%
- **Sum % (uncompounded):** 30.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 119 | 39 | 32.8% | 11 | 108 | 0 | 0.22% | 25.8% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 6 | 0 | -0.13% | -0.8% |
| BUY @ 3rd Alert (retest2) | 113 | 38 | 33.6% | 11 | 102 | 0 | 0.24% | 26.6% |
| SELL (all) | 109 | 31 | 28.4% | 0 | 95 | 14 | 0.04% | 4.3% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 0.79% | 2.4% |
| SELL @ 3rd Alert (retest2) | 106 | 28 | 26.4% | 0 | 92 | 14 | 0.02% | 1.9% |
| retest1 (combined) | 9 | 4 | 44.4% | 0 | 9 | 0 | 0.18% | 1.6% |
| retest2 (combined) | 219 | 66 | 30.1% | 11 | 194 | 14 | 0.13% | 28.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 649.85 | 646.63 | 646.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 12:15:00 | 654.30 | 648.97 | 647.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 13:15:00 | 660.00 | 660.03 | 655.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-16 14:00:00 | 660.00 | 660.03 | 655.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 15:15:00 | 660.75 | 660.11 | 656.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 09:15:00 | 652.30 | 660.11 | 656.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 652.60 | 658.61 | 655.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 09:45:00 | 652.30 | 658.61 | 655.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 652.10 | 657.31 | 655.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 10:30:00 | 651.50 | 657.31 | 655.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 12:15:00 | 660.70 | 657.61 | 655.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-17 13:45:00 | 665.00 | 659.76 | 656.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 11:00:00 | 663.55 | 671.60 | 669.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 11:30:00 | 664.00 | 669.96 | 668.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 13:15:00 | 663.45 | 668.49 | 667.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 667.75 | 667.84 | 667.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:45:00 | 665.30 | 667.84 | 667.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 666.95 | 667.66 | 667.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 658.90 | 667.66 | 667.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-05-22 09:15:00 | 664.00 | 666.93 | 667.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 09:15:00 | 664.00 | 666.93 | 667.33 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 673.00 | 668.35 | 667.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 09:15:00 | 680.40 | 671.49 | 669.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 12:15:00 | 674.95 | 677.37 | 674.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 12:15:00 | 674.95 | 677.37 | 674.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 674.95 | 677.37 | 674.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 13:00:00 | 674.95 | 677.37 | 674.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 679.85 | 677.87 | 675.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 09:30:00 | 681.65 | 677.88 | 676.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 10:00:00 | 681.65 | 677.88 | 676.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 10:30:00 | 682.80 | 678.64 | 677.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 09:30:00 | 687.10 | 679.15 | 677.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 687.65 | 688.28 | 684.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:30:00 | 685.75 | 688.28 | 684.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 686.25 | 689.89 | 687.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 10:00:00 | 686.25 | 689.89 | 687.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 10:15:00 | 692.45 | 690.41 | 687.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:00:00 | 702.10 | 692.78 | 689.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-07 10:15:00 | 749.82 | 727.98 | 721.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 13:15:00 | 735.80 | 746.50 | 746.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 14:15:00 | 733.45 | 743.89 | 745.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 749.45 | 743.58 | 744.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 749.45 | 743.58 | 744.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 749.45 | 743.58 | 744.93 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 11:15:00 | 754.90 | 747.36 | 746.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 759.65 | 751.44 | 749.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 12:15:00 | 750.05 | 752.35 | 750.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 12:15:00 | 750.05 | 752.35 | 750.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 12:15:00 | 750.05 | 752.35 | 750.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:00:00 | 750.05 | 752.35 | 750.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 745.00 | 750.88 | 749.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 14:00:00 | 745.00 | 750.88 | 749.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 14:15:00 | 744.95 | 749.69 | 749.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 14:30:00 | 745.65 | 749.69 | 749.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 748.95 | 749.92 | 749.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 11:00:00 | 748.95 | 749.92 | 749.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 749.65 | 749.87 | 749.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 13:00:00 | 751.35 | 750.16 | 749.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 14:45:00 | 750.60 | 749.99 | 749.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 12:15:00 | 750.05 | 750.23 | 749.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 13:15:00 | 746.85 | 749.33 | 749.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 13:15:00 | 746.85 | 749.33 | 749.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 14:15:00 | 743.00 | 748.07 | 748.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 09:15:00 | 748.75 | 747.74 | 748.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 748.75 | 747.74 | 748.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 748.75 | 747.74 | 748.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:30:00 | 750.90 | 747.74 | 748.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 746.95 | 747.59 | 748.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:30:00 | 749.95 | 747.59 | 748.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 737.55 | 735.73 | 738.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:30:00 | 740.95 | 735.73 | 738.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 741.05 | 736.79 | 738.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 741.05 | 736.79 | 738.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 736.85 | 736.81 | 738.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 13:00:00 | 735.70 | 736.58 | 738.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 15:15:00 | 745.00 | 739.20 | 739.24 | SL hit (close>static) qty=1.00 sl=743.30 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 765.90 | 744.54 | 741.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 10:15:00 | 770.50 | 749.73 | 744.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 12:15:00 | 773.80 | 780.26 | 773.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-30 13:00:00 | 773.80 | 780.26 | 773.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 13:15:00 | 777.35 | 779.68 | 773.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 14:15:00 | 777.80 | 779.68 | 773.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 15:15:00 | 779.00 | 779.21 | 773.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 15:15:00 | 778.55 | 780.64 | 777.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 09:30:00 | 781.25 | 780.55 | 778.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 781.15 | 789.27 | 785.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:45:00 | 780.70 | 789.27 | 785.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 11:15:00 | 775.40 | 786.49 | 784.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 12:00:00 | 775.40 | 786.49 | 784.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 12:15:00 | 771.30 | 783.45 | 783.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 12:45:00 | 769.20 | 783.45 | 783.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-05 13:15:00 | 769.35 | 780.63 | 782.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 13:15:00 | 769.35 | 780.63 | 782.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 10:15:00 | 765.20 | 773.26 | 777.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 09:15:00 | 784.40 | 772.18 | 774.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 784.40 | 772.18 | 774.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 784.40 | 772.18 | 774.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:00:00 | 784.40 | 772.18 | 774.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 774.90 | 772.72 | 774.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:30:00 | 776.10 | 772.72 | 774.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 771.85 | 772.55 | 774.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 12:15:00 | 769.50 | 772.55 | 774.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 09:15:00 | 784.35 | 774.48 | 774.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 784.35 | 774.48 | 774.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 09:15:00 | 788.90 | 779.15 | 776.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 791.00 | 793.68 | 789.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 13:15:00 | 791.00 | 793.68 | 789.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 791.00 | 793.68 | 789.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 13:45:00 | 790.15 | 793.68 | 789.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 792.60 | 793.47 | 789.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 15:00:00 | 792.60 | 793.47 | 789.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 15:15:00 | 792.00 | 793.17 | 790.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:15:00 | 798.00 | 793.17 | 790.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 14:00:00 | 795.00 | 795.46 | 792.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 11:45:00 | 796.15 | 803.71 | 801.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 12:15:00 | 796.65 | 803.71 | 801.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 795.75 | 802.12 | 800.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 13:00:00 | 795.75 | 802.12 | 800.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-18 14:15:00 | 797.00 | 800.14 | 800.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 14:15:00 | 797.00 | 800.14 | 800.20 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 809.50 | 801.33 | 800.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 14:15:00 | 818.00 | 809.62 | 805.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 09:15:00 | 841.00 | 870.53 | 855.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 841.00 | 870.53 | 855.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 841.00 | 870.53 | 855.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:00:00 | 841.00 | 870.53 | 855.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 835.00 | 863.42 | 853.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:45:00 | 836.65 | 863.42 | 853.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 851.05 | 853.24 | 851.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 14:30:00 | 846.95 | 853.24 | 851.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 850.90 | 852.77 | 851.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:15:00 | 794.70 | 852.77 | 851.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 09:15:00 | 787.45 | 839.71 | 845.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 10:15:00 | 784.35 | 828.64 | 839.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 748.00 | 739.88 | 757.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-31 10:00:00 | 748.00 | 739.88 | 757.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 728.60 | 726.56 | 731.02 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 738.00 | 732.75 | 732.38 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 11:15:00 | 731.20 | 736.03 | 736.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 12:15:00 | 730.05 | 734.83 | 735.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-10 12:15:00 | 752.10 | 734.43 | 734.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 12:15:00 | 752.10 | 734.43 | 734.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 12:15:00 | 752.10 | 734.43 | 734.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 12:45:00 | 754.00 | 734.43 | 734.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 13:15:00 | 751.70 | 737.89 | 736.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-11 09:15:00 | 759.25 | 745.78 | 740.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 11:15:00 | 747.25 | 749.15 | 743.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-11 11:30:00 | 753.00 | 749.15 | 743.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 13:15:00 | 731.00 | 745.31 | 742.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 13:45:00 | 734.50 | 745.31 | 742.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 724.55 | 741.16 | 740.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 15:00:00 | 724.55 | 741.16 | 740.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 15:15:00 | 724.40 | 737.81 | 739.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 712.95 | 732.83 | 736.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 12:15:00 | 725.00 | 724.03 | 731.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 13:00:00 | 725.00 | 724.03 | 731.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 735.05 | 726.77 | 731.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 14:30:00 | 737.80 | 726.77 | 731.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 732.95 | 728.00 | 731.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 726.25 | 728.00 | 731.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 12:15:00 | 735.40 | 729.26 | 730.77 | SL hit (close>static) qty=1.00 sl=735.35 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 14:15:00 | 735.60 | 732.11 | 731.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 09:15:00 | 743.70 | 735.08 | 733.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 11:15:00 | 735.00 | 736.15 | 734.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 11:15:00 | 735.00 | 736.15 | 734.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 735.00 | 736.15 | 734.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:00:00 | 735.00 | 736.15 | 734.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 734.15 | 735.75 | 734.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:45:00 | 734.10 | 735.75 | 734.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 738.75 | 736.35 | 734.60 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 730.55 | 733.60 | 733.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 11:15:00 | 727.00 | 732.28 | 733.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 734.60 | 729.72 | 731.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 734.60 | 729.72 | 731.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 734.60 | 729.72 | 731.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:30:00 | 733.60 | 729.72 | 731.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 736.80 | 731.14 | 731.70 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 11:15:00 | 737.70 | 732.45 | 732.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 12:15:00 | 744.40 | 734.84 | 733.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 14:15:00 | 747.00 | 747.35 | 744.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 15:00:00 | 747.00 | 747.35 | 744.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 747.70 | 751.09 | 748.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 15:00:00 | 747.70 | 751.09 | 748.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 747.70 | 750.41 | 748.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 752.05 | 750.41 | 748.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 748.35 | 750.00 | 748.20 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 738.10 | 745.63 | 746.41 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 749.00 | 744.40 | 744.20 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 12:15:00 | 743.05 | 744.06 | 744.10 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 13:15:00 | 744.90 | 744.23 | 744.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 748.00 | 744.98 | 744.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 750.05 | 750.91 | 748.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 15:00:00 | 750.05 | 750.91 | 748.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 749.00 | 750.53 | 748.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 750.55 | 750.53 | 748.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 753.95 | 751.21 | 748.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 10:45:00 | 756.40 | 752.17 | 749.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 11:30:00 | 755.90 | 752.93 | 750.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 11:15:00 | 757.50 | 751.91 | 750.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 12:00:00 | 763.25 | 754.18 | 751.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 774.35 | 779.17 | 776.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:00:00 | 774.35 | 779.17 | 776.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 774.00 | 778.14 | 776.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:30:00 | 773.70 | 778.14 | 776.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 13:15:00 | 773.50 | 777.21 | 776.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 14:45:00 | 775.75 | 776.61 | 775.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 09:15:00 | 778.05 | 775.78 | 775.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-08 15:15:00 | 772.15 | 775.29 | 775.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 15:15:00 | 772.15 | 775.29 | 775.64 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 09:15:00 | 784.55 | 777.15 | 776.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 13:15:00 | 787.75 | 781.65 | 778.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 758.80 | 778.89 | 778.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 758.80 | 778.89 | 778.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 758.80 | 778.89 | 778.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 764.20 | 778.89 | 778.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 762.35 | 775.58 | 777.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 751.00 | 767.32 | 772.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 765.70 | 760.64 | 766.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 11:00:00 | 765.70 | 760.64 | 766.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 768.40 | 762.19 | 766.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:00:00 | 768.40 | 762.19 | 766.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 774.00 | 764.56 | 767.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:45:00 | 774.20 | 764.56 | 767.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 776.60 | 766.96 | 768.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:45:00 | 775.65 | 766.96 | 768.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 15:15:00 | 775.00 | 769.92 | 769.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 10:15:00 | 780.00 | 772.43 | 770.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 12:15:00 | 782.90 | 783.22 | 778.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 12:45:00 | 782.60 | 783.22 | 778.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 13:15:00 | 782.00 | 782.98 | 778.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 13:30:00 | 780.55 | 782.98 | 778.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 782.65 | 782.53 | 779.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:15:00 | 775.85 | 782.53 | 779.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 771.85 | 780.39 | 778.65 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 761.35 | 775.02 | 776.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 14:15:00 | 759.00 | 766.54 | 770.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 11:15:00 | 766.00 | 764.38 | 767.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 12:00:00 | 766.00 | 764.38 | 767.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 768.40 | 765.19 | 767.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 13:15:00 | 760.70 | 765.19 | 767.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 14:45:00 | 763.25 | 764.72 | 767.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 14:15:00 | 772.35 | 762.66 | 764.05 | SL hit (close>static) qty=1.00 sl=770.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 770.20 | 765.66 | 765.27 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 12:15:00 | 757.70 | 764.48 | 764.91 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 774.65 | 765.69 | 765.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 12:15:00 | 776.10 | 769.21 | 766.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 14:15:00 | 769.90 | 769.96 | 767.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-27 14:30:00 | 770.00 | 769.96 | 767.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 768.20 | 769.61 | 767.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:45:00 | 766.70 | 769.61 | 767.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 761.70 | 768.03 | 767.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 761.70 | 768.03 | 767.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 762.50 | 766.92 | 766.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:30:00 | 759.00 | 766.92 | 766.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 12:15:00 | 757.50 | 765.04 | 766.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 751.25 | 762.28 | 764.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 765.65 | 758.96 | 762.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 765.65 | 758.96 | 762.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 765.65 | 758.96 | 762.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:00:00 | 765.65 | 758.96 | 762.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 758.55 | 758.88 | 761.85 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 15:15:00 | 765.00 | 763.32 | 763.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 13:15:00 | 767.30 | 764.25 | 763.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 15:15:00 | 763.40 | 764.28 | 763.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 15:15:00 | 763.40 | 764.28 | 763.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 15:15:00 | 763.40 | 764.28 | 763.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:15:00 | 755.20 | 764.28 | 763.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 751.80 | 761.78 | 762.73 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 11:15:00 | 765.10 | 760.55 | 760.50 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 14:15:00 | 757.95 | 760.34 | 760.44 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 770.30 | 762.38 | 761.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 14:15:00 | 774.25 | 767.19 | 764.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 758.65 | 766.42 | 764.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 758.65 | 766.42 | 764.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 758.65 | 766.42 | 764.37 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 752.30 | 761.13 | 762.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 750.00 | 756.47 | 759.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 759.35 | 756.97 | 759.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 10:15:00 | 759.35 | 756.97 | 759.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 759.35 | 756.97 | 759.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:00:00 | 759.35 | 756.97 | 759.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 758.95 | 757.37 | 759.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:30:00 | 759.60 | 757.37 | 759.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 759.45 | 757.78 | 759.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:30:00 | 759.40 | 757.78 | 759.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 757.80 | 757.79 | 759.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 13:30:00 | 760.00 | 757.79 | 759.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 14:15:00 | 759.00 | 758.03 | 759.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 15:00:00 | 759.00 | 758.03 | 759.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 758.65 | 758.15 | 759.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:15:00 | 766.05 | 758.15 | 759.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 761.15 | 758.75 | 759.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 11:15:00 | 759.50 | 759.04 | 759.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 11:15:00 | 764.15 | 760.06 | 759.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 764.15 | 760.06 | 759.80 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 12:15:00 | 753.60 | 758.77 | 759.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 13:15:00 | 750.10 | 755.08 | 756.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 11:15:00 | 754.65 | 753.86 | 755.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-13 11:45:00 | 752.15 | 753.86 | 755.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 750.50 | 753.19 | 754.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 12:30:00 | 752.95 | 753.19 | 754.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 748.30 | 742.95 | 746.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:30:00 | 750.00 | 742.95 | 746.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 750.40 | 744.44 | 747.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:30:00 | 750.70 | 744.44 | 747.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 755.90 | 748.36 | 748.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:30:00 | 753.90 | 748.36 | 748.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 13:15:00 | 761.40 | 750.96 | 749.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 14:15:00 | 768.20 | 754.41 | 751.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 754.45 | 756.78 | 753.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 10:15:00 | 754.45 | 756.78 | 753.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 754.45 | 756.78 | 753.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:45:00 | 757.10 | 756.78 | 753.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 764.65 | 760.26 | 756.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 13:30:00 | 768.05 | 762.12 | 758.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 770.90 | 762.63 | 759.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 10:15:00 | 750.25 | 759.95 | 760.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 10:15:00 | 750.25 | 759.95 | 760.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 13:15:00 | 748.10 | 755.72 | 758.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 13:15:00 | 716.35 | 713.51 | 726.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-26 14:00:00 | 716.35 | 713.51 | 726.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 730.00 | 718.42 | 725.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 730.00 | 718.42 | 725.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 738.45 | 722.42 | 726.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 738.45 | 722.42 | 726.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 738.55 | 730.35 | 729.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 14:15:00 | 740.05 | 732.29 | 730.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 12:15:00 | 761.95 | 764.10 | 756.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 12:30:00 | 763.05 | 764.10 | 756.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 758.05 | 762.67 | 757.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 15:00:00 | 758.05 | 762.67 | 757.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 771.20 | 774.51 | 771.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 10:00:00 | 771.20 | 774.51 | 771.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 10:15:00 | 765.70 | 772.75 | 771.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 11:00:00 | 765.70 | 772.75 | 771.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2023-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 11:15:00 | 760.25 | 770.25 | 770.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 12:15:00 | 757.70 | 767.74 | 769.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 11:15:00 | 762.40 | 761.24 | 764.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-08 12:00:00 | 762.40 | 761.24 | 764.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 763.00 | 761.59 | 764.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:30:00 | 762.65 | 761.59 | 764.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 766.60 | 762.72 | 764.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:00:00 | 766.60 | 762.72 | 764.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 763.35 | 762.85 | 763.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 11:30:00 | 760.65 | 762.70 | 763.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 12:15:00 | 760.80 | 762.70 | 763.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 12:00:00 | 756.65 | 753.93 | 754.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 09:15:00 | 763.00 | 753.28 | 753.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 763.00 | 753.28 | 753.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 09:15:00 | 777.75 | 766.06 | 760.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 10:15:00 | 774.50 | 776.17 | 772.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-22 10:45:00 | 773.75 | 776.17 | 772.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 771.00 | 775.13 | 772.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 11:45:00 | 772.75 | 775.13 | 772.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 765.85 | 773.28 | 771.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 13:00:00 | 765.85 | 773.28 | 771.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2023-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 15:15:00 | 767.25 | 769.75 | 770.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 10:15:00 | 763.55 | 765.35 | 767.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 12:15:00 | 759.90 | 757.76 | 761.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-28 13:00:00 | 759.90 | 757.76 | 761.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 761.30 | 757.86 | 760.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:00:00 | 761.30 | 757.86 | 760.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 761.10 | 758.51 | 760.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:15:00 | 759.20 | 758.51 | 760.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 758.85 | 758.57 | 760.00 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 14:15:00 | 770.75 | 762.47 | 761.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 781.40 | 766.66 | 763.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 825.30 | 825.41 | 812.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-05 11:45:00 | 825.50 | 825.41 | 812.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 807.05 | 826.21 | 818.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 09:45:00 | 802.70 | 826.21 | 818.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 10:15:00 | 809.00 | 822.77 | 817.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 11:00:00 | 809.00 | 822.77 | 817.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2023-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 13:15:00 | 792.75 | 810.29 | 812.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 14:15:00 | 789.00 | 806.04 | 810.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 12:15:00 | 806.50 | 802.06 | 806.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 12:15:00 | 806.50 | 802.06 | 806.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 12:15:00 | 806.50 | 802.06 | 806.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 13:00:00 | 806.50 | 802.06 | 806.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 13:15:00 | 800.90 | 801.83 | 805.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-07 15:00:00 | 798.00 | 801.06 | 805.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 09:30:00 | 796.20 | 799.31 | 803.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 10:00:00 | 797.30 | 793.18 | 797.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 12:30:00 | 798.30 | 796.29 | 797.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 13:15:00 | 778.85 | 792.80 | 796.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 13:45:00 | 796.50 | 792.80 | 796.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 14:15:00 | 758.10 | 785.24 | 792.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 14:15:00 | 756.39 | 785.24 | 792.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 14:15:00 | 757.43 | 785.24 | 792.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 14:15:00 | 758.38 | 785.24 | 792.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-12 14:15:00 | 768.55 | 758.47 | 772.12 | SL hit (close>ema200) qty=0.50 sl=758.47 alert=retest2 |

### Cycle 49 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 786.15 | 775.24 | 775.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 792.45 | 781.84 | 778.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 783.00 | 785.08 | 781.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 10:15:00 | 783.00 | 785.08 | 781.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 783.00 | 785.08 | 781.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 10:45:00 | 781.25 | 785.08 | 781.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 782.75 | 784.24 | 782.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 782.75 | 784.24 | 782.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 780.00 | 783.39 | 781.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:15:00 | 783.50 | 783.39 | 781.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 784.15 | 783.54 | 782.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 11:45:00 | 786.15 | 784.89 | 783.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 09:15:00 | 787.85 | 783.92 | 783.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 10:45:00 | 787.30 | 784.39 | 783.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 12:15:00 | 786.05 | 784.63 | 783.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 783.40 | 784.64 | 783.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 15:00:00 | 783.40 | 784.64 | 783.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 784.10 | 784.53 | 783.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:30:00 | 787.25 | 785.24 | 784.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 12:15:00 | 783.05 | 784.73 | 784.34 | SL hit (close<static) qty=1.00 sl=783.25 alert=retest2 |

### Cycle 50 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 764.80 | 780.74 | 782.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 753.70 | 775.33 | 779.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 14:15:00 | 764.80 | 759.09 | 763.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 14:15:00 | 764.80 | 759.09 | 763.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 764.80 | 759.09 | 763.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 15:00:00 | 764.80 | 759.09 | 763.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 15:15:00 | 764.55 | 760.18 | 763.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:15:00 | 774.50 | 760.18 | 763.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 781.10 | 764.37 | 765.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 10:00:00 | 781.10 | 764.37 | 765.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 774.85 | 766.46 | 766.32 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 764.95 | 774.45 | 775.61 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 11:15:00 | 775.50 | 771.31 | 771.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 13:15:00 | 780.50 | 773.97 | 772.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 10:15:00 | 781.05 | 781.14 | 776.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-05 11:00:00 | 781.05 | 781.14 | 776.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 784.75 | 781.86 | 777.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 788.15 | 780.88 | 778.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 10:30:00 | 788.20 | 781.99 | 779.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 14:15:00 | 768.35 | 777.62 | 778.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 768.35 | 777.62 | 778.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 766.95 | 773.64 | 775.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 774.00 | 764.83 | 768.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 774.00 | 764.83 | 768.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 774.00 | 764.83 | 768.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:30:00 | 774.60 | 764.83 | 768.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 768.95 | 765.65 | 768.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 11:45:00 | 767.35 | 765.83 | 768.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 14:00:00 | 767.60 | 766.97 | 768.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 09:15:00 | 763.30 | 767.86 | 768.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 14:00:00 | 767.05 | 768.54 | 768.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 766.95 | 768.22 | 768.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 15:00:00 | 766.95 | 768.22 | 768.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 767.55 | 768.09 | 768.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:15:00 | 769.75 | 768.09 | 768.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 762.70 | 767.01 | 767.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 768.65 | 763.39 | 763.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 09:15:00 | 768.65 | 763.39 | 763.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 10:15:00 | 776.95 | 766.10 | 764.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 13:15:00 | 785.10 | 792.07 | 785.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 13:15:00 | 785.10 | 792.07 | 785.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 785.10 | 792.07 | 785.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 14:00:00 | 785.10 | 792.07 | 785.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 766.95 | 787.05 | 783.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 766.95 | 787.05 | 783.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 769.00 | 783.44 | 782.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 750.25 | 783.44 | 782.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 09:15:00 | 748.50 | 776.45 | 779.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 741.05 | 761.14 | 770.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 745.20 | 744.27 | 757.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 11:00:00 | 745.20 | 744.27 | 757.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 755.10 | 745.93 | 752.04 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 13:15:00 | 765.70 | 755.61 | 755.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 14:15:00 | 770.45 | 758.58 | 756.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 13:15:00 | 765.70 | 765.76 | 761.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-29 13:45:00 | 766.75 | 765.76 | 761.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 764.80 | 769.99 | 766.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 764.80 | 769.99 | 766.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 767.00 | 769.39 | 766.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 10:45:00 | 776.75 | 772.00 | 768.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 13:45:00 | 776.15 | 773.33 | 769.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 14:30:00 | 777.35 | 773.80 | 770.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 15:15:00 | 778.00 | 773.80 | 770.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 802.90 | 816.55 | 808.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 15:00:00 | 802.90 | 816.55 | 808.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 804.30 | 814.10 | 808.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 09:15:00 | 805.85 | 814.10 | 808.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 10:30:00 | 809.20 | 812.97 | 808.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-07 09:15:00 | 854.43 | 820.71 | 813.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-02-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 14:15:00 | 812.50 | 827.14 | 827.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 801.85 | 820.46 | 824.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 14:15:00 | 809.65 | 808.34 | 815.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-09 14:45:00 | 808.65 | 808.34 | 815.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 798.35 | 806.45 | 813.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 10:15:00 | 797.30 | 806.45 | 813.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 10:45:00 | 797.10 | 804.32 | 811.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 14:00:00 | 797.00 | 793.06 | 798.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 14:30:00 | 795.20 | 794.13 | 798.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 798.95 | 795.09 | 798.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 785.90 | 795.09 | 798.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 11:30:00 | 789.95 | 793.05 | 796.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 12:00:00 | 791.00 | 793.05 | 796.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 12:30:00 | 791.35 | 792.94 | 796.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 797.95 | 793.94 | 796.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 797.95 | 793.94 | 796.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 805.50 | 796.25 | 797.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-14 14:15:00 | 805.50 | 796.25 | 797.47 | SL hit (close>static) qty=1.00 sl=799.50 alert=retest2 |

### Cycle 59 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 809.00 | 800.36 | 799.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 11:15:00 | 820.05 | 805.55 | 801.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 10:15:00 | 811.70 | 813.39 | 808.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 11:00:00 | 811.70 | 813.39 | 808.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 807.25 | 811.62 | 809.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 15:00:00 | 807.25 | 811.62 | 809.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 805.90 | 810.48 | 808.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 813.85 | 810.48 | 808.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 10:00:00 | 808.60 | 810.10 | 808.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 12:45:00 | 807.85 | 809.38 | 808.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 14:15:00 | 800.80 | 807.22 | 807.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 800.80 | 807.22 | 807.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 09:15:00 | 799.40 | 804.74 | 806.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 798.00 | 797.81 | 801.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 798.00 | 797.81 | 801.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 798.00 | 797.81 | 801.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:30:00 | 802.55 | 797.81 | 801.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 799.70 | 798.19 | 801.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 11:00:00 | 799.70 | 798.19 | 801.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 791.85 | 792.47 | 795.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:00:00 | 791.85 | 792.47 | 795.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 797.00 | 793.75 | 795.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 807.20 | 793.75 | 795.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 807.15 | 796.43 | 796.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:30:00 | 807.80 | 796.43 | 796.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 804.40 | 798.02 | 797.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 09:15:00 | 807.65 | 801.40 | 799.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 11:15:00 | 799.05 | 801.64 | 800.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 11:15:00 | 799.05 | 801.64 | 800.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 799.05 | 801.64 | 800.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 12:00:00 | 799.05 | 801.64 | 800.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 801.50 | 801.61 | 800.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 14:15:00 | 802.85 | 801.39 | 800.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-26 14:15:00 | 797.90 | 800.69 | 799.99 | SL hit (close<static) qty=1.00 sl=798.05 alert=retest2 |

### Cycle 62 — SELL (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 10:15:00 | 796.30 | 799.08 | 799.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 11:15:00 | 791.10 | 797.49 | 798.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 772.95 | 771.96 | 778.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 772.95 | 771.96 | 778.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 784.65 | 774.67 | 778.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:00:00 | 784.65 | 774.67 | 778.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 787.50 | 777.23 | 779.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:30:00 | 789.00 | 777.23 | 779.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 790.85 | 782.10 | 781.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 794.05 | 787.71 | 784.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 15:15:00 | 787.55 | 790.35 | 788.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 15:15:00 | 787.55 | 790.35 | 788.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 787.55 | 790.35 | 788.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 09:15:00 | 792.90 | 790.35 | 788.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 10:45:00 | 791.65 | 790.57 | 788.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 11:30:00 | 794.05 | 791.38 | 789.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 15:15:00 | 796.00 | 792.42 | 790.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 796.00 | 793.14 | 790.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:15:00 | 789.90 | 793.14 | 790.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 768.40 | 788.19 | 788.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 768.40 | 788.19 | 788.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 765.30 | 780.99 | 785.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 785.00 | 780.65 | 783.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 14:15:00 | 785.00 | 780.65 | 783.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 785.00 | 780.65 | 783.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 785.00 | 780.65 | 783.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 783.30 | 781.18 | 783.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 786.10 | 781.18 | 783.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 784.80 | 781.90 | 783.99 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 12:15:00 | 788.10 | 785.23 | 785.19 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 15:15:00 | 783.35 | 785.05 | 785.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 781.95 | 784.43 | 784.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 13:15:00 | 782.10 | 779.81 | 782.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 13:15:00 | 782.10 | 779.81 | 782.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 782.10 | 779.81 | 782.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 14:00:00 | 782.10 | 779.81 | 782.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 774.95 | 778.84 | 781.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 15:15:00 | 768.50 | 778.84 | 781.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 12:15:00 | 730.07 | 748.00 | 760.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-14 10:15:00 | 733.05 | 732.92 | 746.91 | SL hit (close>ema200) qty=0.50 sl=732.92 alert=retest2 |

### Cycle 67 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 737.65 | 719.99 | 718.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 10:15:00 | 738.30 | 723.65 | 720.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 15:15:00 | 754.00 | 755.00 | 749.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 09:15:00 | 755.20 | 755.00 | 749.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 753.85 | 754.80 | 750.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:30:00 | 750.50 | 754.80 | 750.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 750.20 | 753.88 | 750.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:30:00 | 751.65 | 753.88 | 750.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 750.05 | 753.11 | 750.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 12:45:00 | 749.95 | 753.11 | 750.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 754.95 | 753.48 | 750.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 765.85 | 753.64 | 751.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 10:15:00 | 799.10 | 807.40 | 808.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 10:15:00 | 799.10 | 807.40 | 808.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 794.00 | 803.21 | 806.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 772.00 | 771.75 | 779.57 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 10:30:00 | 768.80 | 771.46 | 778.73 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 774.35 | 772.33 | 777.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 772.00 | 772.33 | 777.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 09:15:00 | 764.00 | 751.01 | 754.88 | SL hit (close>ema400) qty=1.00 sl=754.88 alert=retest1 |

### Cycle 69 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 762.30 | 745.67 | 745.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 09:15:00 | 782.75 | 763.12 | 755.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 09:15:00 | 772.35 | 773.43 | 765.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 09:30:00 | 773.30 | 773.43 | 765.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 764.35 | 771.59 | 766.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:45:00 | 767.35 | 771.59 | 766.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 768.60 | 770.99 | 766.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 14:30:00 | 773.35 | 770.17 | 766.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 09:15:00 | 762.00 | 768.34 | 766.67 | SL hit (close<static) qty=1.00 sl=765.25 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 14:15:00 | 759.20 | 764.70 | 765.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 749.30 | 760.71 | 763.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 745.25 | 745.21 | 752.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 745.25 | 745.21 | 752.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 747.05 | 746.63 | 750.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:30:00 | 749.05 | 746.63 | 750.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 733.95 | 729.91 | 734.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 733.95 | 729.91 | 734.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 736.00 | 731.13 | 734.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 732.50 | 731.13 | 734.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 13:15:00 | 731.00 | 731.33 | 733.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 15:15:00 | 739.70 | 734.77 | 734.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 739.70 | 734.77 | 734.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 741.90 | 736.20 | 735.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 752.50 | 754.91 | 750.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:00:00 | 752.50 | 754.91 | 750.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 754.80 | 756.04 | 753.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:00:00 | 754.80 | 756.04 | 753.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 758.35 | 756.23 | 753.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:30:00 | 759.95 | 757.10 | 754.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 10:15:00 | 752.45 | 756.15 | 755.09 | SL hit (close<static) qty=1.00 sl=753.60 alert=retest2 |

### Cycle 72 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 748.55 | 754.01 | 754.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 744.35 | 752.08 | 753.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 12:15:00 | 740.60 | 740.23 | 743.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 13:15:00 | 740.95 | 740.37 | 743.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 740.95 | 740.37 | 743.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:45:00 | 740.65 | 740.37 | 743.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 742.85 | 740.87 | 743.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 742.85 | 740.87 | 743.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 744.00 | 741.49 | 743.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 745.75 | 741.49 | 743.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 741.25 | 741.44 | 743.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 14:15:00 | 738.05 | 740.29 | 742.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:30:00 | 737.55 | 738.85 | 740.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 747.95 | 742.17 | 741.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 747.95 | 742.17 | 741.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 751.35 | 744.10 | 742.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 11:15:00 | 743.75 | 745.29 | 743.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 11:15:00 | 743.75 | 745.29 | 743.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 743.75 | 745.29 | 743.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 743.75 | 745.29 | 743.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 743.80 | 744.99 | 743.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 742.75 | 744.99 | 743.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 743.95 | 744.78 | 743.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:30:00 | 743.95 | 744.78 | 743.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 745.75 | 744.98 | 743.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 745.75 | 744.98 | 743.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 745.00 | 744.98 | 744.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 741.30 | 744.98 | 744.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 739.60 | 743.91 | 743.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:15:00 | 735.00 | 743.91 | 743.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 729.05 | 740.93 | 742.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 13:15:00 | 726.50 | 735.24 | 739.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 727.80 | 725.17 | 729.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 727.80 | 725.17 | 729.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 727.80 | 725.17 | 729.64 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 742.00 | 732.27 | 731.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 753.10 | 736.44 | 733.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 729.00 | 742.72 | 738.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 729.00 | 742.72 | 738.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 729.00 | 742.72 | 738.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 720.70 | 742.72 | 738.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 699.45 | 734.06 | 734.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 683.95 | 724.04 | 730.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 723.25 | 713.87 | 721.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 723.25 | 713.87 | 721.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 723.25 | 713.87 | 721.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 723.25 | 713.87 | 721.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 732.25 | 717.54 | 722.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 732.25 | 717.54 | 722.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 731.95 | 720.42 | 723.67 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 743.70 | 727.36 | 726.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 745.20 | 730.93 | 728.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 775.70 | 776.09 | 767.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 777.70 | 776.09 | 767.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 770.50 | 774.97 | 767.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 781.50 | 774.97 | 767.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 859.65 | 834.77 | 819.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 876.00 | 910.05 | 911.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 867.50 | 879.76 | 884.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 11:15:00 | 880.50 | 878.77 | 882.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 11:15:00 | 880.50 | 878.77 | 882.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 880.50 | 878.77 | 882.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:45:00 | 882.90 | 878.77 | 882.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 881.70 | 879.36 | 882.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:00:00 | 881.70 | 879.36 | 882.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 880.35 | 879.55 | 882.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:30:00 | 880.70 | 879.55 | 882.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 882.90 | 880.22 | 882.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:45:00 | 882.65 | 880.22 | 882.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 884.00 | 880.98 | 882.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 884.70 | 880.98 | 882.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 885.40 | 881.86 | 882.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:30:00 | 879.00 | 881.46 | 882.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:00:00 | 879.65 | 881.10 | 882.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 889.65 | 877.35 | 876.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 889.65 | 877.35 | 876.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 895.70 | 881.02 | 877.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 881.20 | 886.77 | 883.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 881.20 | 886.77 | 883.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 881.20 | 886.77 | 883.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 875.40 | 886.77 | 883.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 868.20 | 883.06 | 881.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 868.20 | 883.06 | 881.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 872.30 | 880.91 | 880.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 867.95 | 880.91 | 880.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 871.75 | 879.07 | 880.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 861.00 | 875.34 | 878.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 847.80 | 839.03 | 851.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 15:00:00 | 847.80 | 839.03 | 851.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 849.70 | 841.17 | 851.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 849.65 | 841.17 | 851.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 846.60 | 842.25 | 851.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:45:00 | 839.45 | 842.84 | 850.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 797.48 | 840.40 | 848.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 13:15:00 | 844.40 | 841.20 | 847.92 | SL hit (close>ema200) qty=0.50 sl=841.20 alert=retest2 |

### Cycle 81 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 863.30 | 846.21 | 844.17 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 849.65 | 852.35 | 852.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 848.30 | 851.54 | 852.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 12:15:00 | 836.35 | 834.48 | 840.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 13:00:00 | 836.35 | 834.48 | 840.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 837.55 | 835.10 | 839.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:45:00 | 838.90 | 835.10 | 839.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 838.80 | 835.84 | 839.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:30:00 | 837.55 | 835.84 | 839.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 840.00 | 836.67 | 839.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 800.00 | 836.67 | 839.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 10:15:00 | 795.85 | 793.16 | 792.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 795.85 | 793.16 | 792.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 802.90 | 796.57 | 794.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 808.30 | 812.23 | 806.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 10:00:00 | 808.30 | 812.23 | 806.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 808.40 | 811.46 | 806.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 804.70 | 811.46 | 806.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 811.00 | 811.37 | 806.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:45:00 | 807.25 | 811.37 | 806.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 806.35 | 810.37 | 806.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 808.65 | 810.37 | 806.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 805.35 | 809.36 | 806.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:30:00 | 802.20 | 809.36 | 806.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 802.60 | 808.01 | 806.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 802.60 | 808.01 | 806.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 805.80 | 807.57 | 806.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 797.50 | 807.57 | 806.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 804.00 | 806.85 | 806.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 802.50 | 806.85 | 806.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 803.60 | 806.20 | 805.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:45:00 | 801.15 | 806.20 | 805.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 803.15 | 805.59 | 805.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 13:15:00 | 799.00 | 804.04 | 804.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 14:15:00 | 805.45 | 804.32 | 804.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 14:15:00 | 805.45 | 804.32 | 804.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 805.45 | 804.32 | 804.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:45:00 | 805.75 | 804.32 | 804.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 805.50 | 804.55 | 804.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:15:00 | 814.05 | 804.55 | 804.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 816.55 | 806.95 | 806.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 10:15:00 | 820.80 | 809.72 | 807.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 11:15:00 | 847.50 | 847.51 | 840.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 12:00:00 | 847.50 | 847.51 | 840.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 845.30 | 846.82 | 842.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:45:00 | 844.65 | 846.82 | 842.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 848.25 | 849.20 | 846.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 845.50 | 849.20 | 846.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 847.20 | 849.45 | 847.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 847.20 | 849.45 | 847.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 850.45 | 849.65 | 847.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 851.95 | 849.55 | 848.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 10:15:00 | 845.45 | 848.73 | 847.76 | SL hit (close<static) qty=1.00 sl=847.05 alert=retest2 |

### Cycle 86 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 843.45 | 847.16 | 847.24 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 853.80 | 848.49 | 847.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 12:15:00 | 862.30 | 852.18 | 849.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 12:15:00 | 879.15 | 879.26 | 870.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 13:00:00 | 879.15 | 879.26 | 870.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 873.55 | 879.49 | 875.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:30:00 | 874.60 | 879.49 | 875.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 872.90 | 878.17 | 874.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:00:00 | 872.90 | 878.17 | 874.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 861.60 | 874.86 | 873.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 861.60 | 874.86 | 873.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 863.00 | 872.49 | 872.69 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 880.55 | 874.10 | 873.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 886.00 | 879.90 | 877.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 15:15:00 | 881.40 | 883.06 | 880.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 15:15:00 | 881.40 | 883.06 | 880.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 881.40 | 883.06 | 880.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 889.30 | 883.06 | 880.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 887.65 | 883.98 | 880.99 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 869.00 | 884.22 | 884.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 10:15:00 | 865.85 | 880.55 | 882.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 881.30 | 880.70 | 882.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 12:00:00 | 881.30 | 880.70 | 882.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 870.60 | 878.68 | 881.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:45:00 | 867.35 | 876.80 | 880.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 15:15:00 | 867.00 | 876.80 | 880.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 11:30:00 | 865.55 | 871.26 | 875.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 12:15:00 | 865.35 | 871.26 | 875.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 859.25 | 867.45 | 872.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 854.50 | 864.02 | 869.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:00:00 | 854.60 | 862.14 | 868.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 882.00 | 866.50 | 865.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 882.00 | 866.50 | 865.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 11:15:00 | 890.40 | 873.65 | 868.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 10:15:00 | 893.00 | 918.08 | 906.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 10:15:00 | 893.00 | 918.08 | 906.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 893.00 | 918.08 | 906.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 893.00 | 918.08 | 906.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 889.35 | 912.34 | 904.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:30:00 | 886.40 | 912.34 | 904.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 906.10 | 910.39 | 904.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 906.10 | 910.39 | 904.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 904.65 | 909.24 | 904.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:30:00 | 904.25 | 909.24 | 904.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 903.05 | 908.00 | 904.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 895.10 | 908.00 | 904.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 890.25 | 904.45 | 903.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 896.00 | 904.45 | 903.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 888.00 | 901.16 | 902.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 14:15:00 | 881.50 | 892.31 | 897.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 896.05 | 891.44 | 895.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 896.05 | 891.44 | 895.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 896.05 | 891.44 | 895.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 896.05 | 891.44 | 895.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 882.05 | 889.56 | 894.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:15:00 | 881.30 | 889.56 | 894.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:45:00 | 881.00 | 885.38 | 891.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:30:00 | 872.15 | 877.01 | 885.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 837.23 | 866.20 | 878.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 836.95 | 866.20 | 878.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 10:15:00 | 861.40 | 856.16 | 866.83 | SL hit (close>ema200) qty=0.50 sl=856.16 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 883.30 | 869.93 | 868.71 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 861.65 | 868.56 | 869.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 861.15 | 867.08 | 868.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 10:15:00 | 867.05 | 865.19 | 867.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 10:15:00 | 867.05 | 865.19 | 867.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 867.05 | 865.19 | 867.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 867.05 | 865.19 | 867.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 862.00 | 864.56 | 866.81 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 13:15:00 | 884.65 | 869.96 | 868.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 887.50 | 873.47 | 869.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 874.55 | 876.27 | 873.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 12:15:00 | 874.55 | 876.27 | 873.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 874.55 | 876.27 | 873.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:45:00 | 873.50 | 876.27 | 873.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 876.30 | 879.07 | 875.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 875.40 | 879.07 | 875.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 878.00 | 878.85 | 875.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 877.60 | 878.85 | 875.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 892.10 | 899.04 | 892.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:30:00 | 889.75 | 899.04 | 892.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 877.50 | 894.73 | 891.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 877.50 | 894.73 | 891.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 875.10 | 890.81 | 889.64 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 876.50 | 887.94 | 888.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 872.15 | 884.79 | 886.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 15:15:00 | 891.05 | 885.40 | 886.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 15:15:00 | 891.05 | 885.40 | 886.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 891.05 | 885.40 | 886.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 857.25 | 885.40 | 886.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 873.60 | 850.11 | 849.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 873.60 | 850.11 | 849.81 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 857.55 | 860.47 | 860.62 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 867.40 | 861.86 | 861.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 15:15:00 | 872.00 | 863.89 | 862.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 862.20 | 863.55 | 862.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 862.20 | 863.55 | 862.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 862.20 | 863.55 | 862.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 862.20 | 863.55 | 862.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 869.60 | 864.76 | 862.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:15:00 | 871.20 | 864.76 | 862.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 10:15:00 | 869.85 | 879.01 | 877.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 862.50 | 875.71 | 876.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 862.50 | 875.71 | 876.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 860.00 | 872.57 | 874.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 862.45 | 859.73 | 864.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:45:00 | 862.40 | 859.73 | 864.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 862.05 | 860.19 | 864.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 862.40 | 860.19 | 864.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 857.55 | 859.63 | 863.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 843.60 | 856.17 | 859.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 868.10 | 848.83 | 852.70 | SL hit (close>static) qty=1.00 sl=865.45 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 10:15:00 | 882.70 | 855.60 | 855.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 11:15:00 | 889.45 | 862.37 | 858.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-23 15:15:00 | 867.40 | 869.31 | 863.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-24 09:15:00 | 862.90 | 869.31 | 863.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 884.90 | 872.43 | 865.63 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 14:15:00 | 844.85 | 862.21 | 862.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 837.60 | 848.81 | 853.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 13:15:00 | 853.80 | 848.77 | 851.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 13:15:00 | 853.80 | 848.77 | 851.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 853.80 | 848.77 | 851.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 853.80 | 848.77 | 851.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 863.75 | 851.77 | 853.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 863.75 | 851.77 | 853.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 861.55 | 853.72 | 853.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 877.25 | 853.72 | 853.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 894.50 | 861.88 | 857.53 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 852.00 | 866.75 | 868.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 849.55 | 857.35 | 861.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 869.25 | 858.69 | 861.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 869.25 | 858.69 | 861.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 869.25 | 858.69 | 861.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 869.25 | 858.69 | 861.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 865.85 | 860.12 | 861.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:30:00 | 868.35 | 860.12 | 861.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 870.70 | 863.03 | 862.78 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 860.10 | 864.02 | 864.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 15:15:00 | 857.95 | 861.14 | 862.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 844.15 | 842.99 | 848.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 13:15:00 | 835.55 | 841.32 | 846.31 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 14:00:00 | 834.65 | 839.99 | 845.25 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 823.35 | 822.87 | 829.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 826.30 | 822.87 | 829.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 829.00 | 824.10 | 829.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 829.00 | 824.10 | 829.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 827.65 | 824.81 | 829.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 824.95 | 825.05 | 829.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:15:00 | 823.55 | 825.05 | 829.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 824.05 | 824.96 | 828.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 827.80 | 824.18 | 827.01 | SL hit (close>ema400) qty=1.00 sl=827.01 alert=retest1 |

### Cycle 107 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 848.00 | 832.01 | 830.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 849.25 | 837.95 | 833.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 832.00 | 838.64 | 834.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 14:15:00 | 832.00 | 838.64 | 834.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 832.00 | 838.64 | 834.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:00:00 | 832.00 | 838.64 | 834.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 829.00 | 836.71 | 834.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 822.10 | 836.71 | 834.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 816.75 | 832.72 | 832.77 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 12:15:00 | 840.35 | 830.94 | 830.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 845.60 | 833.87 | 831.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 13:15:00 | 844.55 | 847.79 | 841.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 13:15:00 | 844.55 | 847.79 | 841.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 844.55 | 847.79 | 841.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:45:00 | 843.45 | 847.79 | 841.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 844.00 | 847.03 | 841.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:30:00 | 844.85 | 847.03 | 841.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 830.00 | 842.85 | 840.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 830.00 | 842.85 | 840.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 822.25 | 838.73 | 839.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 821.20 | 829.51 | 834.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 825.40 | 825.10 | 829.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 12:45:00 | 825.50 | 825.10 | 829.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 827.55 | 825.59 | 829.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:45:00 | 830.30 | 825.59 | 829.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 826.80 | 825.83 | 829.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:30:00 | 827.90 | 825.83 | 829.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 826.00 | 825.89 | 828.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:15:00 | 825.20 | 825.89 | 828.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 10:00:00 | 819.30 | 822.33 | 825.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 10:00:00 | 822.40 | 822.48 | 823.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 11:45:00 | 824.85 | 822.61 | 823.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 828.90 | 823.86 | 824.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 14:00:00 | 822.80 | 823.65 | 823.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 10:15:00 | 829.00 | 823.59 | 823.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 829.00 | 823.59 | 823.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 834.30 | 826.10 | 824.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 12:15:00 | 839.70 | 840.11 | 836.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 13:15:00 | 836.85 | 840.11 | 836.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 837.15 | 839.52 | 836.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:30:00 | 836.60 | 839.52 | 836.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 834.25 | 838.47 | 835.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 834.00 | 838.47 | 835.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 836.00 | 837.97 | 835.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 828.15 | 837.97 | 835.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 811.80 | 832.74 | 833.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 11:15:00 | 806.85 | 825.27 | 830.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 15:15:00 | 805.30 | 805.12 | 812.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 09:15:00 | 816.00 | 805.12 | 812.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 816.65 | 807.43 | 812.96 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 819.00 | 815.56 | 815.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 825.05 | 817.46 | 816.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 14:15:00 | 820.70 | 821.24 | 818.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 15:00:00 | 820.70 | 821.24 | 818.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 818.00 | 820.59 | 818.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 822.70 | 820.59 | 818.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 816.55 | 819.78 | 818.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 816.55 | 819.78 | 818.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 814.25 | 818.68 | 818.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 814.25 | 818.68 | 818.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 812.30 | 817.40 | 817.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 811.50 | 815.07 | 816.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 15:15:00 | 804.80 | 804.64 | 808.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:15:00 | 804.15 | 804.64 | 808.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 806.45 | 805.00 | 808.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 807.50 | 805.00 | 808.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 801.90 | 803.90 | 806.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 801.90 | 803.90 | 806.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 744.55 | 728.59 | 731.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 744.55 | 728.59 | 731.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 749.55 | 732.78 | 733.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 747.65 | 732.78 | 733.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 751.00 | 736.43 | 735.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 754.80 | 747.74 | 742.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 15:15:00 | 753.00 | 753.12 | 747.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 09:15:00 | 753.90 | 753.12 | 747.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 752.70 | 753.03 | 748.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:45:00 | 759.00 | 755.00 | 750.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:30:00 | 757.70 | 754.28 | 751.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 740.00 | 750.46 | 750.17 | SL hit (close<static) qty=1.00 sl=746.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 12:15:00 | 736.50 | 747.67 | 748.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 14:15:00 | 731.00 | 742.36 | 746.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 14:15:00 | 729.05 | 728.65 | 735.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-02 14:45:00 | 728.65 | 728.65 | 735.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 720.70 | 723.07 | 728.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:45:00 | 712.65 | 719.83 | 726.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 13:45:00 | 713.60 | 710.00 | 714.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 15:00:00 | 713.65 | 710.73 | 714.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 695.70 | 711.68 | 714.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 703.80 | 710.11 | 713.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 685.40 | 697.80 | 703.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 677.02 | 693.91 | 700.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 677.92 | 693.91 | 700.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 677.97 | 693.91 | 700.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 660.91 | 673.48 | 684.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 651.13 | 666.09 | 678.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 665.05 | 663.20 | 674.16 | SL hit (close>ema200) qty=0.50 sl=663.20 alert=retest2 |

### Cycle 117 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 686.30 | 675.14 | 674.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 712.00 | 690.24 | 684.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 696.50 | 702.90 | 695.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 696.50 | 702.90 | 695.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 696.50 | 702.90 | 695.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 696.50 | 702.90 | 695.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 696.20 | 701.56 | 695.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 698.05 | 701.56 | 695.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 697.75 | 700.80 | 696.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 694.75 | 700.80 | 696.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 693.85 | 699.41 | 695.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 691.90 | 699.41 | 695.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 692.40 | 698.01 | 695.51 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 682.05 | 692.87 | 693.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 676.65 | 683.85 | 686.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 673.60 | 671.99 | 676.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 673.60 | 671.99 | 676.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 675.70 | 669.97 | 674.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 675.70 | 669.97 | 674.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 670.40 | 670.06 | 673.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 674.80 | 670.06 | 673.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 667.25 | 667.94 | 670.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:15:00 | 666.70 | 667.94 | 670.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 13:45:00 | 665.50 | 663.52 | 664.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 674.60 | 666.69 | 666.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 674.60 | 666.69 | 666.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 10:15:00 | 691.60 | 671.67 | 668.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 663.80 | 671.60 | 668.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 663.80 | 671.60 | 668.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 663.80 | 671.60 | 668.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 663.80 | 671.60 | 668.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 665.25 | 670.33 | 668.64 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 660.15 | 666.72 | 667.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 646.45 | 662.67 | 665.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 657.30 | 652.31 | 657.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 657.30 | 652.31 | 657.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 657.30 | 652.31 | 657.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 657.30 | 652.31 | 657.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 659.60 | 653.77 | 657.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 659.60 | 653.77 | 657.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 658.50 | 654.71 | 657.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:30:00 | 658.00 | 654.71 | 657.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 659.20 | 655.61 | 657.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:00:00 | 659.20 | 655.61 | 657.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 660.65 | 656.62 | 657.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:00:00 | 660.65 | 656.62 | 657.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 662.75 | 657.84 | 658.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:30:00 | 663.10 | 657.84 | 658.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 663.10 | 658.90 | 658.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 678.70 | 662.86 | 660.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 669.70 | 671.58 | 667.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 669.70 | 671.58 | 667.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 666.15 | 670.49 | 667.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 666.15 | 670.49 | 667.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 665.65 | 669.52 | 667.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 665.65 | 669.52 | 667.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 663.75 | 668.37 | 667.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 664.00 | 668.37 | 667.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 666.60 | 668.01 | 667.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 664.00 | 668.01 | 667.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 666.00 | 667.61 | 666.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 665.75 | 667.61 | 666.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 667.20 | 667.53 | 667.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 657.20 | 667.53 | 667.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 676.25 | 669.27 | 667.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 677.90 | 669.27 | 667.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 651.85 | 665.10 | 666.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 651.85 | 665.10 | 666.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 650.35 | 662.15 | 664.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 618.15 | 617.62 | 626.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:45:00 | 617.80 | 617.62 | 626.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 607.30 | 604.45 | 609.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 607.85 | 604.45 | 609.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 609.35 | 605.43 | 609.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 609.35 | 605.43 | 609.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 609.90 | 606.32 | 609.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 599.30 | 606.32 | 609.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 611.50 | 603.60 | 603.63 | SL hit (close>static) qty=1.00 sl=609.90 alert=retest2 |

### Cycle 123 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 11:15:00 | 605.30 | 603.94 | 603.79 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 600.85 | 603.88 | 603.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 12:15:00 | 597.00 | 601.80 | 602.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 10:15:00 | 600.00 | 599.55 | 601.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 10:15:00 | 600.00 | 599.55 | 601.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 600.00 | 599.55 | 601.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 598.45 | 599.55 | 601.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 602.75 | 600.19 | 601.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 603.40 | 600.19 | 601.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 603.80 | 600.91 | 601.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:45:00 | 604.00 | 600.91 | 601.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 603.00 | 601.33 | 601.69 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 14:15:00 | 605.65 | 602.19 | 602.05 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 592.45 | 600.28 | 601.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 573.20 | 594.86 | 598.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 586.45 | 584.54 | 591.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 586.45 | 584.54 | 591.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 586.45 | 584.54 | 591.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:30:00 | 576.60 | 580.83 | 586.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 572.40 | 580.26 | 585.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:30:00 | 572.65 | 577.88 | 583.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 14:15:00 | 575.00 | 577.76 | 582.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 571.60 | 576.35 | 580.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-04 10:15:00 | 586.65 | 579.64 | 579.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 586.65 | 579.64 | 579.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 591.85 | 584.39 | 582.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 09:15:00 | 579.00 | 584.33 | 582.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 579.00 | 584.33 | 582.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 579.00 | 584.33 | 582.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 11:00:00 | 583.95 | 584.26 | 582.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 11:30:00 | 585.75 | 584.70 | 582.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-10 09:15:00 | 642.35 | 625.78 | 614.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 14:15:00 | 612.70 | 616.52 | 616.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 607.05 | 614.06 | 615.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 607.30 | 605.61 | 609.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 607.30 | 605.61 | 609.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 607.30 | 605.61 | 609.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 607.30 | 605.61 | 609.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 614.40 | 604.99 | 606.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 614.65 | 604.99 | 606.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 611.40 | 606.27 | 607.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:15:00 | 616.10 | 606.27 | 607.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 621.80 | 609.38 | 608.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 12:15:00 | 622.75 | 612.05 | 609.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 671.65 | 680.60 | 674.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 671.65 | 680.60 | 674.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 671.65 | 680.60 | 674.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 671.65 | 680.60 | 674.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 677.40 | 679.96 | 674.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 15:00:00 | 680.00 | 678.80 | 675.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 670.55 | 674.60 | 674.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 670.55 | 674.60 | 674.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 666.75 | 673.03 | 674.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 670.10 | 669.52 | 671.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 670.10 | 669.52 | 671.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 670.10 | 669.52 | 671.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:30:00 | 664.10 | 667.91 | 669.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 14:00:00 | 664.55 | 665.93 | 668.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 15:00:00 | 663.00 | 665.34 | 667.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 673.80 | 668.45 | 667.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 673.80 | 668.45 | 667.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 680.10 | 670.78 | 669.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 669.20 | 675.20 | 672.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 669.20 | 675.20 | 672.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 669.20 | 675.20 | 672.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 669.20 | 675.20 | 672.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 668.45 | 673.85 | 672.50 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 666.45 | 671.24 | 671.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 14:15:00 | 663.70 | 669.25 | 670.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 665.00 | 651.79 | 657.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 665.00 | 651.79 | 657.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 665.00 | 651.79 | 657.40 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 661.05 | 660.00 | 659.99 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 659.10 | 659.82 | 659.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-11 10:15:00 | 649.60 | 654.79 | 657.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 664.50 | 654.73 | 655.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 664.50 | 654.73 | 655.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 664.50 | 654.73 | 655.75 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 664.70 | 656.72 | 656.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 670.00 | 664.65 | 661.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 729.45 | 735.89 | 725.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 729.45 | 735.89 | 725.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 727.45 | 739.48 | 733.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:30:00 | 722.00 | 739.48 | 733.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 725.40 | 736.66 | 732.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 725.40 | 736.66 | 732.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 715.55 | 727.35 | 728.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 14:15:00 | 706.00 | 723.08 | 726.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 12:15:00 | 721.00 | 715.15 | 720.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 12:15:00 | 721.00 | 715.15 | 720.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 12:15:00 | 721.00 | 715.15 | 720.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:00:00 | 721.00 | 715.15 | 720.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 720.05 | 716.13 | 720.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:45:00 | 720.40 | 716.13 | 720.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 722.20 | 717.34 | 720.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:00:00 | 722.20 | 717.34 | 720.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 722.00 | 718.27 | 720.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 721.10 | 718.27 | 720.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 725.80 | 719.78 | 721.21 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 729.95 | 722.59 | 722.29 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 10:15:00 | 715.60 | 722.99 | 723.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 710.10 | 718.14 | 720.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 11:15:00 | 716.75 | 716.09 | 718.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 11:15:00 | 716.75 | 716.09 | 718.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 716.75 | 716.09 | 718.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 716.70 | 716.09 | 718.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 727.90 | 717.84 | 718.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 729.70 | 717.84 | 718.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 724.00 | 719.07 | 718.89 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 10:15:00 | 714.70 | 718.76 | 719.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 708.05 | 712.23 | 714.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 713.60 | 710.62 | 713.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 713.60 | 710.62 | 713.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 713.60 | 710.62 | 713.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:45:00 | 711.40 | 710.62 | 713.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 714.95 | 711.49 | 713.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 715.15 | 711.49 | 713.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 711.60 | 711.51 | 713.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 713.95 | 711.51 | 713.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 714.85 | 712.18 | 713.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 714.85 | 712.18 | 713.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 718.00 | 713.34 | 713.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 718.00 | 713.34 | 713.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 718.00 | 714.27 | 714.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 727.75 | 716.97 | 715.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 722.00 | 723.46 | 719.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 722.00 | 723.46 | 719.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 722.00 | 723.46 | 719.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:30:00 | 722.45 | 723.46 | 719.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 714.90 | 721.74 | 719.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 716.45 | 721.74 | 719.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 704.00 | 718.20 | 717.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 702.80 | 718.20 | 717.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 704.55 | 715.47 | 716.56 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 726.70 | 715.89 | 714.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 731.00 | 724.98 | 722.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 15:15:00 | 742.00 | 742.10 | 736.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:15:00 | 749.25 | 742.10 | 736.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 11:00:00 | 744.90 | 743.13 | 738.08 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 748.60 | 748.66 | 743.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 742.30 | 748.66 | 743.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 745.65 | 748.81 | 745.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 14:15:00 | 745.65 | 748.81 | 745.73 | SL hit (close<ema400) qty=1.00 sl=745.73 alert=retest1 |

### Cycle 144 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 741.05 | 744.44 | 744.77 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 750.65 | 745.30 | 745.08 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 740.45 | 745.75 | 746.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 739.00 | 744.02 | 745.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 746.05 | 741.52 | 742.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 746.05 | 741.52 | 742.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 746.05 | 741.52 | 742.88 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 749.00 | 744.03 | 743.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 752.00 | 746.56 | 745.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 766.65 | 767.40 | 759.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:00:00 | 766.65 | 767.40 | 759.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 776.70 | 772.01 | 766.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 14:45:00 | 778.30 | 774.12 | 768.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:45:00 | 780.75 | 775.73 | 771.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:00:00 | 780.70 | 781.73 | 777.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 779.60 | 784.35 | 782.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 785.05 | 784.49 | 783.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 788.05 | 784.49 | 783.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 782.15 | 784.79 | 783.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 782.15 | 784.79 | 783.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 782.00 | 784.23 | 783.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 782.05 | 783.16 | 783.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 12:15:00 | 782.05 | 783.16 | 783.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 778.40 | 781.80 | 782.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 786.05 | 782.04 | 782.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 786.05 | 782.04 | 782.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 786.05 | 782.04 | 782.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 786.05 | 782.04 | 782.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 780.00 | 781.63 | 782.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 793.60 | 781.63 | 782.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 785.00 | 782.31 | 782.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 785.00 | 782.31 | 782.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 786.60 | 783.16 | 782.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 808.90 | 788.43 | 785.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 804.50 | 808.65 | 803.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:45:00 | 805.65 | 808.65 | 803.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 802.10 | 807.34 | 803.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 802.10 | 807.34 | 803.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 804.90 | 806.85 | 803.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:30:00 | 804.05 | 806.85 | 803.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 803.05 | 805.89 | 803.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 804.10 | 805.89 | 803.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 816.20 | 807.95 | 804.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 803.00 | 807.95 | 804.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 811.35 | 808.81 | 805.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 804.35 | 808.81 | 805.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 805.45 | 808.39 | 805.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 805.45 | 808.39 | 805.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 811.50 | 809.01 | 806.44 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 796.30 | 803.65 | 804.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 793.95 | 801.71 | 803.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 793.00 | 785.31 | 789.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 793.00 | 785.31 | 789.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 793.00 | 785.31 | 789.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 793.00 | 785.31 | 789.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 790.85 | 786.42 | 790.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 792.20 | 786.42 | 790.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 789.75 | 787.46 | 789.90 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 801.10 | 791.84 | 791.32 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 777.00 | 789.04 | 790.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 774.00 | 786.03 | 789.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 10:15:00 | 768.20 | 768.00 | 773.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 11:00:00 | 768.20 | 768.00 | 773.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 763.70 | 767.76 | 771.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 763.95 | 767.76 | 771.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 767.50 | 767.18 | 770.86 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 789.35 | 774.83 | 773.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 795.75 | 784.61 | 779.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 787.55 | 788.25 | 782.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:00:00 | 787.55 | 788.25 | 782.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 783.35 | 786.71 | 782.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 785.35 | 786.19 | 782.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:15:00 | 785.60 | 785.53 | 783.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 790.20 | 786.47 | 783.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 788.55 | 790.66 | 790.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 788.55 | 790.66 | 790.69 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 792.50 | 790.64 | 790.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 799.00 | 792.32 | 791.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 812.95 | 813.08 | 807.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 12:45:00 | 815.00 | 813.76 | 808.91 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 12:30:00 | 816.05 | 814.99 | 812.16 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 815.00 | 817.73 | 815.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 815.00 | 817.73 | 815.84 | SL hit (close<ema400) qty=1.00 sl=815.84 alert=retest1 |

### Cycle 156 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 812.05 | 817.19 | 817.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 803.00 | 814.35 | 815.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 800.25 | 797.08 | 802.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:15:00 | 800.85 | 797.08 | 802.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 802.70 | 797.76 | 800.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:15:00 | 796.00 | 799.29 | 800.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:30:00 | 796.00 | 798.23 | 799.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 796.00 | 798.00 | 799.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 805.55 | 800.31 | 800.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 805.55 | 800.31 | 800.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 14:15:00 | 808.35 | 801.92 | 800.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 811.80 | 813.04 | 808.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:30:00 | 812.60 | 813.04 | 808.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 807.90 | 812.01 | 808.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:45:00 | 807.40 | 812.01 | 808.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 814.25 | 812.46 | 809.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 812.00 | 812.46 | 809.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 803.55 | 812.09 | 810.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 801.95 | 812.09 | 810.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 802.85 | 810.24 | 809.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:15:00 | 801.50 | 810.24 | 809.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 803.45 | 808.89 | 809.19 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 812.45 | 809.72 | 809.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 10:15:00 | 817.05 | 811.64 | 810.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 12:15:00 | 803.05 | 810.29 | 810.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 12:15:00 | 803.05 | 810.29 | 810.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 803.05 | 810.29 | 810.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 803.05 | 810.29 | 810.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 799.90 | 808.21 | 809.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 15:15:00 | 795.20 | 804.22 | 807.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 09:15:00 | 779.00 | 777.57 | 785.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:00:00 | 779.00 | 777.57 | 785.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 741.60 | 749.58 | 758.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 740.50 | 749.58 | 758.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 740.05 | 748.29 | 757.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:45:00 | 740.00 | 746.83 | 755.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 740.15 | 741.21 | 749.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 751.60 | 743.10 | 748.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 737.15 | 745.13 | 747.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:00:00 | 738.85 | 743.88 | 746.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 755.00 | 745.46 | 746.58 | SL hit (close>static) qty=1.00 sl=753.50 alert=retest2 |

### Cycle 161 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 753.60 | 748.41 | 747.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 757.45 | 751.06 | 749.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 750.55 | 751.96 | 749.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 10:00:00 | 750.55 | 751.96 | 749.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 749.55 | 751.48 | 749.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 748.55 | 751.48 | 749.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 748.95 | 750.97 | 749.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:45:00 | 748.70 | 750.97 | 749.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 746.55 | 749.46 | 749.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 746.55 | 749.46 | 749.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 752.30 | 752.08 | 750.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 749.50 | 752.08 | 750.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 756.05 | 752.88 | 751.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:30:00 | 753.10 | 752.88 | 751.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 748.00 | 751.90 | 750.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:45:00 | 748.20 | 751.90 | 750.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 747.65 | 751.05 | 750.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 747.30 | 751.05 | 750.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 745.40 | 749.92 | 750.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 743.15 | 747.74 | 749.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 747.00 | 743.26 | 745.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 747.00 | 743.26 | 745.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 747.00 | 743.26 | 745.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 747.00 | 743.26 | 745.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 746.00 | 743.81 | 745.18 | EMA400 retest candle locked (from downside) |

### Cycle 163 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 747.95 | 746.14 | 746.03 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 744.25 | 745.76 | 745.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 739.00 | 744.41 | 745.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 750.50 | 745.63 | 745.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 750.50 | 745.63 | 745.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 750.50 | 745.63 | 745.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 749.50 | 745.63 | 745.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 752.15 | 746.93 | 746.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 754.70 | 750.76 | 748.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 751.05 | 751.26 | 749.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 751.05 | 751.26 | 749.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 751.05 | 751.26 | 749.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 753.50 | 751.26 | 749.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 748.20 | 750.65 | 749.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 748.20 | 750.65 | 749.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 746.30 | 749.78 | 748.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 746.30 | 749.78 | 748.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 739.90 | 747.03 | 747.70 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 750.85 | 747.50 | 747.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 759.45 | 750.60 | 748.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 779.85 | 782.79 | 776.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 779.50 | 782.13 | 777.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 779.50 | 782.13 | 777.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 777.70 | 782.13 | 777.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 779.85 | 781.08 | 777.40 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 770.00 | 775.59 | 775.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 766.30 | 770.29 | 772.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 11:15:00 | 721.10 | 719.77 | 728.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:30:00 | 722.00 | 719.77 | 728.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 736.00 | 723.51 | 726.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 736.00 | 723.51 | 726.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 746.00 | 728.01 | 728.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 746.70 | 728.01 | 728.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 745.10 | 731.43 | 729.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 747.45 | 734.63 | 731.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 759.50 | 763.20 | 755.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:00:00 | 759.50 | 763.20 | 755.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 754.60 | 761.06 | 755.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 754.60 | 761.06 | 755.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 762.00 | 761.25 | 756.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 762.50 | 760.73 | 756.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 749.20 | 758.42 | 755.95 | SL hit (close<static) qty=1.00 sl=751.55 alert=retest2 |

### Cycle 170 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 750.10 | 754.05 | 754.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 746.00 | 752.44 | 753.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 744.50 | 743.41 | 746.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 744.50 | 743.41 | 746.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 744.50 | 743.41 | 746.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 740.00 | 742.84 | 745.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 750.05 | 742.17 | 742.30 | SL hit (close>static) qty=1.00 sl=748.75 alert=retest2 |

### Cycle 171 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 749.70 | 743.68 | 742.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 774.90 | 754.05 | 749.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 777.10 | 777.89 | 771.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 11:00:00 | 777.10 | 777.89 | 771.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 774.20 | 775.97 | 772.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 777.45 | 775.81 | 772.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 777.70 | 774.01 | 773.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 776.50 | 774.71 | 773.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 771.65 | 772.90 | 773.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 771.65 | 772.90 | 773.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 763.20 | 770.96 | 772.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 774.50 | 770.00 | 771.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 774.50 | 770.00 | 771.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 774.50 | 770.00 | 771.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 774.50 | 770.00 | 771.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 780.20 | 772.04 | 772.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 781.00 | 772.04 | 772.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 780.75 | 773.78 | 772.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 789.50 | 777.95 | 774.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 13:15:00 | 779.50 | 779.92 | 777.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 14:00:00 | 779.50 | 779.92 | 777.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 778.00 | 779.54 | 777.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 778.00 | 779.54 | 777.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 773.00 | 778.23 | 776.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 767.20 | 776.01 | 775.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 764.40 | 773.69 | 774.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 763.85 | 770.28 | 772.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 751.85 | 749.82 | 757.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 11:15:00 | 751.25 | 749.39 | 756.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 751.25 | 749.39 | 756.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 754.45 | 749.39 | 756.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 753.60 | 750.96 | 754.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 754.85 | 750.96 | 754.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 760.40 | 752.85 | 754.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 760.40 | 752.85 | 754.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 760.00 | 754.28 | 755.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:30:00 | 760.70 | 754.28 | 755.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 763.20 | 756.94 | 756.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 781.15 | 764.65 | 760.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 785.95 | 787.15 | 778.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:15:00 | 790.40 | 787.15 | 778.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 798.00 | 800.49 | 794.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-08 11:15:00 | 790.25 | 797.59 | 794.30 | SL hit (close<ema400) qty=1.00 sl=794.30 alert=retest1 |

### Cycle 176 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 789.65 | 793.21 | 793.51 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 801.85 | 793.25 | 792.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 12:15:00 | 802.45 | 795.09 | 793.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 796.85 | 798.42 | 796.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 796.85 | 798.42 | 796.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 796.85 | 798.42 | 796.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:30:00 | 796.30 | 798.42 | 796.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 796.40 | 798.02 | 796.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:30:00 | 796.25 | 798.02 | 796.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 785.00 | 795.41 | 795.35 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 785.45 | 793.42 | 794.45 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 802.05 | 795.65 | 794.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 803.50 | 798.33 | 796.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 12:15:00 | 799.50 | 799.84 | 797.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 12:15:00 | 799.50 | 799.84 | 797.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 799.50 | 799.84 | 797.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 799.35 | 799.84 | 797.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 804.40 | 802.34 | 799.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 10:30:00 | 805.95 | 801.89 | 799.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 818.00 | 801.65 | 800.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 864.55 | 867.74 | 868.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 15:15:00 | 864.55 | 867.74 | 868.01 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 872.75 | 868.74 | 868.44 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 862.20 | 867.44 | 867.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 15:15:00 | 859.00 | 865.01 | 866.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 864.45 | 863.49 | 865.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 864.45 | 863.49 | 865.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 864.45 | 863.49 | 865.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 864.45 | 863.49 | 865.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 867.10 | 864.21 | 865.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 865.00 | 864.21 | 865.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 874.85 | 866.34 | 866.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 874.85 | 866.34 | 866.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 13:15:00 | 878.95 | 868.86 | 867.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 882.00 | 875.07 | 872.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 877.65 | 877.92 | 874.48 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:15:00 | 888.65 | 877.92 | 874.48 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 887.45 | 893.18 | 887.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 887.45 | 893.18 | 887.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 889.40 | 892.43 | 887.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:45:00 | 886.95 | 892.43 | 887.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 886.35 | 891.57 | 887.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-12 14:15:00 | 886.35 | 891.57 | 887.91 | SL hit (close<ema400) qty=1.00 sl=887.91 alert=retest1 |

### Cycle 184 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 874.70 | 884.99 | 885.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 869.95 | 879.63 | 882.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 09:15:00 | 887.00 | 876.57 | 880.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 887.00 | 876.57 | 880.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 887.00 | 876.57 | 880.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:30:00 | 893.20 | 876.57 | 880.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 889.05 | 879.07 | 880.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 889.70 | 879.07 | 880.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 888.65 | 882.73 | 882.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 912.00 | 889.82 | 885.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 11:15:00 | 888.90 | 891.96 | 887.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 888.90 | 891.96 | 887.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 888.90 | 891.96 | 887.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 887.15 | 891.96 | 887.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 889.25 | 891.42 | 887.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:30:00 | 888.00 | 891.42 | 887.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 889.00 | 890.93 | 887.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:45:00 | 888.00 | 890.93 | 887.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 892.75 | 891.30 | 888.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 905.00 | 891.26 | 888.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 896.70 | 900.30 | 897.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 12:15:00 | 887.10 | 895.28 | 895.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 887.10 | 895.28 | 895.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 880.25 | 885.06 | 888.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 888.70 | 885.79 | 888.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 888.70 | 885.79 | 888.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 888.70 | 885.79 | 888.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 888.45 | 885.79 | 888.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 887.65 | 886.16 | 888.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:15:00 | 886.00 | 886.16 | 888.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:00:00 | 886.00 | 886.19 | 888.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 884.70 | 886.24 | 887.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 12:15:00 | 891.30 | 887.89 | 888.06 | SL hit (close>static) qty=1.00 sl=891.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 13:15:00 | 892.40 | 888.79 | 888.46 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 884.85 | 887.68 | 888.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 14:15:00 | 882.45 | 885.15 | 886.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 15:15:00 | 886.40 | 885.40 | 886.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 891.60 | 885.40 | 886.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 189 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 903.15 | 888.95 | 887.97 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 883.10 | 889.08 | 889.53 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 896.80 | 889.33 | 888.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 11:15:00 | 900.80 | 891.62 | 889.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 12:15:00 | 902.15 | 906.29 | 900.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:00:00 | 902.15 | 906.29 | 900.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 897.05 | 904.44 | 899.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:45:00 | 896.60 | 904.44 | 899.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 907.85 | 905.12 | 900.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 917.75 | 903.63 | 900.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 891.60 | 902.03 | 900.63 | SL hit (close<static) qty=1.00 sl=896.75 alert=retest2 |

### Cycle 192 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 884.00 | 898.43 | 899.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 882.55 | 895.25 | 897.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 895.50 | 895.30 | 897.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 895.50 | 895.30 | 897.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 895.50 | 895.30 | 897.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 895.50 | 895.30 | 897.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 891.15 | 890.65 | 893.59 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 903.30 | 896.00 | 895.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 917.00 | 903.37 | 899.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 15:15:00 | 910.75 | 913.14 | 907.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-09 09:15:00 | 892.75 | 913.14 | 907.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 896.80 | 909.87 | 906.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 896.00 | 909.87 | 906.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 908.70 | 909.64 | 906.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 899.40 | 909.64 | 906.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 910.75 | 909.86 | 907.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:45:00 | 905.10 | 909.86 | 907.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 904.75 | 909.12 | 907.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 904.75 | 909.12 | 907.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 908.30 | 908.96 | 907.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 911.95 | 908.96 | 907.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 14:15:00 | 901.80 | 910.54 | 909.65 | SL hit (close<static) qty=1.00 sl=902.00 alert=retest2 |

### Cycle 194 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 902.10 | 908.35 | 908.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 898.75 | 906.43 | 907.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 906.70 | 906.48 | 907.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 15:00:00 | 906.70 | 906.48 | 907.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 906.75 | 906.54 | 907.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 912.50 | 906.54 | 907.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 914.95 | 908.22 | 908.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 916.35 | 908.22 | 908.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 916.45 | 909.87 | 909.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 923.40 | 915.65 | 912.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 919.85 | 920.58 | 916.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 919.85 | 920.58 | 916.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 915.00 | 919.46 | 916.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 914.85 | 918.72 | 916.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 922.75 | 919.53 | 917.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 926.20 | 920.33 | 918.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 923.90 | 922.58 | 920.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:00:00 | 923.85 | 923.26 | 921.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 14:45:00 | 924.20 | 925.52 | 923.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 923.00 | 925.02 | 923.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 913.15 | 925.02 | 923.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 920.90 | 924.19 | 923.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 922.00 | 924.19 | 923.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 926.45 | 932.29 | 932.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 926.45 | 932.29 | 932.75 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 937.75 | 933.70 | 933.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 11:15:00 | 939.75 | 934.91 | 933.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 11:15:00 | 935.40 | 942.92 | 939.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 11:15:00 | 935.40 | 942.92 | 939.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 935.40 | 942.92 | 939.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 935.40 | 942.92 | 939.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 928.55 | 940.04 | 938.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 928.55 | 940.04 | 938.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 925.00 | 937.03 | 937.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 920.85 | 929.49 | 933.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 915.00 | 911.24 | 919.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 12:45:00 | 914.30 | 911.24 | 919.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 931.95 | 915.38 | 920.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 931.95 | 915.38 | 920.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 926.25 | 917.55 | 920.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 926.25 | 917.55 | 920.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 936.15 | 922.21 | 922.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 936.15 | 922.21 | 922.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 936.50 | 925.07 | 923.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 940.85 | 928.22 | 925.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 929.60 | 930.84 | 927.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 15:00:00 | 929.60 | 930.84 | 927.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 930.00 | 930.67 | 927.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 930.15 | 930.54 | 927.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:15:00 | 930.40 | 930.54 | 927.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 10:15:00 | 926.65 | 929.76 | 927.76 | SL hit (close<static) qty=1.00 sl=927.45 alert=retest2 |

### Cycle 200 — SELL (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 13:15:00 | 917.75 | 928.21 | 928.57 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 944.40 | 931.45 | 930.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 952.50 | 935.93 | 932.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 938.20 | 939.68 | 935.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 12:15:00 | 938.20 | 939.68 | 935.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 938.20 | 939.68 | 935.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:45:00 | 932.95 | 939.68 | 935.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 942.60 | 941.12 | 937.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 945.95 | 941.12 | 937.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:45:00 | 947.45 | 949.71 | 945.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:30:00 | 945.25 | 948.26 | 945.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 12:15:00 | 933.10 | 943.37 | 943.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 933.10 | 943.37 | 943.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 14:15:00 | 928.80 | 938.98 | 941.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 891.00 | 888.68 | 899.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 891.00 | 888.68 | 899.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 893.30 | 889.58 | 894.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 893.30 | 889.58 | 894.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 895.80 | 890.83 | 894.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 898.15 | 890.83 | 894.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 898.90 | 892.44 | 895.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 897.10 | 892.44 | 895.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 893.25 | 892.60 | 894.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:45:00 | 900.00 | 892.60 | 894.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 898.85 | 893.85 | 895.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 898.85 | 893.85 | 895.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 906.05 | 896.29 | 896.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 906.05 | 896.29 | 896.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 913.00 | 899.63 | 897.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 918.85 | 903.48 | 899.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 915.00 | 917.93 | 911.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 915.00 | 917.93 | 911.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 915.00 | 917.93 | 911.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 887.15 | 917.93 | 911.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 896.40 | 913.63 | 909.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 11:45:00 | 930.15 | 914.52 | 910.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:45:00 | 926.25 | 926.45 | 920.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 14:30:00 | 929.40 | 921.25 | 919.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 898.55 | 915.31 | 916.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 898.55 | 915.31 | 916.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 893.75 | 911.00 | 914.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 909.60 | 903.55 | 908.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 909.60 | 903.55 | 908.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 909.60 | 903.55 | 908.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 915.50 | 903.55 | 908.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 902.65 | 903.37 | 907.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 906.65 | 903.37 | 907.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 911.60 | 905.02 | 908.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:00:00 | 911.60 | 905.02 | 908.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 914.80 | 906.97 | 908.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:30:00 | 912.25 | 906.97 | 908.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 912.05 | 907.99 | 909.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 912.05 | 907.99 | 909.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 911.60 | 909.80 | 909.73 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 903.50 | 908.54 | 909.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 901.95 | 907.22 | 908.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 893.00 | 890.28 | 896.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 893.00 | 890.28 | 896.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 888.30 | 889.88 | 896.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 897.30 | 889.88 | 896.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 900.80 | 892.06 | 896.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 901.60 | 892.06 | 896.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 902.00 | 894.05 | 897.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 902.00 | 894.05 | 897.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 908.55 | 899.09 | 898.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 910.45 | 904.09 | 901.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 905.15 | 905.66 | 902.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 905.15 | 905.66 | 902.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 901.70 | 904.87 | 902.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:45:00 | 901.90 | 904.87 | 902.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 901.00 | 904.10 | 902.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 901.00 | 904.10 | 902.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 900.05 | 903.29 | 902.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 900.05 | 903.29 | 902.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 903.00 | 903.23 | 902.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 890.40 | 903.23 | 902.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 908.00 | 904.18 | 902.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:15:00 | 909.55 | 906.35 | 904.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 893.30 | 906.39 | 907.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 893.30 | 906.39 | 907.71 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 09:15:00 | 921.65 | 909.44 | 908.98 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 893.65 | 907.43 | 908.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 13:15:00 | 890.80 | 904.11 | 906.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 912.65 | 905.81 | 907.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 912.65 | 905.81 | 907.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 912.65 | 905.81 | 907.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 912.65 | 905.81 | 907.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 925.00 | 909.65 | 908.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 935.75 | 914.87 | 911.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 924.10 | 926.99 | 920.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:15:00 | 932.35 | 926.99 | 920.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 937.95 | 929.18 | 921.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 941.40 | 935.36 | 928.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 940.90 | 938.23 | 934.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 954.30 | 936.29 | 935.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 11:15:00 | 925.85 | 937.93 | 939.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 925.85 | 937.93 | 939.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 12:15:00 | 924.80 | 935.30 | 937.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 10:15:00 | 902.10 | 901.98 | 912.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:00:00 | 902.10 | 901.98 | 912.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 898.00 | 887.79 | 893.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 898.00 | 887.79 | 893.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 897.95 | 889.82 | 893.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:45:00 | 897.90 | 889.82 | 893.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 905.05 | 895.52 | 895.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 910.50 | 901.20 | 898.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 13:15:00 | 902.20 | 906.00 | 902.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 13:15:00 | 902.20 | 906.00 | 902.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 902.20 | 906.00 | 902.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 902.20 | 906.00 | 902.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 904.85 | 905.77 | 902.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 905.35 | 905.77 | 902.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 905.95 | 905.80 | 902.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 905.75 | 905.80 | 902.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 902.05 | 905.05 | 902.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 902.05 | 905.05 | 902.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 902.00 | 904.44 | 902.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 902.45 | 904.44 | 902.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 13:15:00 | 897.05 | 901.83 | 901.77 | SL hit (close<static) qty=1.00 sl=898.15 alert=retest2 |

### Cycle 214 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 887.75 | 899.01 | 900.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 886.00 | 896.41 | 899.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 894.50 | 894.31 | 897.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:00:00 | 894.50 | 894.31 | 897.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 899.05 | 895.25 | 897.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 899.05 | 895.25 | 897.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 886.00 | 893.40 | 896.29 | EMA400 retest candle locked (from downside) |

### Cycle 215 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 908.00 | 896.45 | 896.22 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 886.45 | 895.97 | 896.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 872.95 | 883.82 | 887.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 820.90 | 819.43 | 833.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 15:15:00 | 832.50 | 825.20 | 830.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 832.50 | 825.20 | 830.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 837.00 | 825.20 | 830.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 830.00 | 826.16 | 830.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 803.55 | 827.85 | 829.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 11:15:00 | 834.00 | 821.52 | 821.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 834.00 | 821.52 | 821.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 844.40 | 828.68 | 824.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 857.40 | 860.78 | 849.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 857.40 | 860.78 | 849.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 857.40 | 860.78 | 849.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:00:00 | 867.75 | 862.17 | 851.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 829.60 | 856.11 | 856.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 829.60 | 856.11 | 856.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 820.70 | 849.03 | 853.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 829.15 | 827.09 | 837.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 829.15 | 827.09 | 837.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 840.30 | 829.73 | 837.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 846.55 | 829.73 | 837.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 845.40 | 832.87 | 838.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 845.00 | 832.87 | 838.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 850.70 | 836.43 | 839.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 850.70 | 836.43 | 839.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 861.85 | 837.75 | 838.15 | EMA400 retest candle locked (from downside) |

### Cycle 219 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 873.30 | 844.86 | 841.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 884.00 | 852.69 | 845.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 13:15:00 | 849.95 | 855.39 | 847.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 14:00:00 | 849.95 | 855.39 | 847.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 841.00 | 852.51 | 847.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:30:00 | 836.10 | 852.51 | 847.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 849.90 | 851.99 | 847.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 835.80 | 851.99 | 847.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 829.60 | 847.51 | 845.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 831.95 | 847.51 | 845.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 833.80 | 844.77 | 844.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 824.15 | 833.61 | 838.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 817.55 | 812.39 | 821.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 15:00:00 | 817.55 | 812.39 | 821.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 834.00 | 816.71 | 822.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 830.35 | 816.71 | 822.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 822.75 | 817.92 | 822.49 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 833.05 | 825.35 | 824.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 838.40 | 830.02 | 827.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 13:15:00 | 838.95 | 839.33 | 833.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:00:00 | 838.95 | 839.33 | 833.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 832.35 | 837.93 | 833.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:30:00 | 834.10 | 837.93 | 833.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 836.50 | 837.65 | 833.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 835.05 | 837.65 | 833.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 823.20 | 834.76 | 832.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 823.20 | 834.76 | 832.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 822.80 | 832.37 | 831.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 823.05 | 832.37 | 831.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 822.75 | 830.44 | 831.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 809.00 | 825.03 | 828.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 826.75 | 806.19 | 813.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 826.75 | 806.19 | 813.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 826.75 | 806.19 | 813.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 798.30 | 813.68 | 814.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:45:00 | 796.80 | 809.95 | 813.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 824.35 | 813.88 | 813.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 824.35 | 813.88 | 813.41 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 806.35 | 812.87 | 813.06 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 820.40 | 812.56 | 812.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 835.85 | 818.77 | 815.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 15:15:00 | 841.50 | 842.30 | 837.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 09:15:00 | 817.80 | 842.30 | 837.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 821.20 | 838.08 | 836.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 826.70 | 835.91 | 835.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 828.00 | 834.33 | 834.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 828.00 | 834.33 | 834.77 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 12:15:00 | 840.90 | 835.65 | 835.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 843.70 | 837.26 | 836.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 853.65 | 855.34 | 849.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 853.65 | 855.34 | 849.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 853.65 | 855.34 | 849.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:30:00 | 855.00 | 855.34 | 849.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 873.30 | 866.16 | 860.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 897.55 | 869.02 | 864.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 880.85 | 892.77 | 892.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 880.85 | 892.77 | 892.85 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 12:15:00 | 904.35 | 895.08 | 893.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 910.70 | 899.03 | 895.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 11:15:00 | 904.20 | 904.54 | 900.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:30:00 | 904.60 | 904.54 | 900.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 901.05 | 904.52 | 900.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 901.05 | 904.52 | 900.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 916.45 | 906.91 | 902.27 | EMA400 retest candle locked (from upside) |

### Cycle 230 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 886.05 | 901.32 | 901.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 14:15:00 | 882.85 | 897.62 | 900.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 868.70 | 868.50 | 877.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:45:00 | 869.55 | 868.50 | 877.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 889.80 | 872.49 | 877.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 889.80 | 872.49 | 877.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 894.55 | 876.90 | 878.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 888.00 | 876.90 | 878.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 886.25 | 878.77 | 879.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 890.90 | 881.20 | 880.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 890.90 | 881.20 | 880.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 892.00 | 883.36 | 881.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 881.45 | 886.87 | 884.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 881.45 | 886.87 | 884.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 881.45 | 886.87 | 884.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 882.30 | 886.87 | 884.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 875.35 | 884.57 | 883.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 875.05 | 884.57 | 883.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 877.05 | 881.53 | 882.01 | EMA200 below EMA400 |

### Cycle 233 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 890.20 | 881.86 | 881.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 894.20 | 885.78 | 883.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 890.00 | 891.72 | 887.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:30:00 | 891.00 | 891.72 | 887.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 886.50 | 890.38 | 887.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:45:00 | 887.75 | 890.38 | 887.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 884.40 | 889.18 | 887.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 884.40 | 889.18 | 887.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 884.45 | 888.23 | 886.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:00:00 | 884.45 | 888.23 | 886.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 886.05 | 887.48 | 886.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 890.20 | 887.48 | 886.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 892.95 | 888.68 | 887.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 882.65 | 887.69 | 887.12 | SL hit (close<static) qty=1.00 sl=885.00 alert=retest2 |

### Cycle 234 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 882.55 | 886.66 | 886.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 878.90 | 885.11 | 886.00 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-17 13:45:00 | 665.00 | 2023-05-22 09:15:00 | 664.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2023-05-19 11:00:00 | 663.55 | 2023-05-22 09:15:00 | 664.00 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2023-05-19 11:30:00 | 664.00 | 2023-05-22 09:15:00 | 664.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2023-05-19 13:15:00 | 663.45 | 2023-05-22 09:15:00 | 664.00 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2023-05-26 09:30:00 | 681.65 | 2023-06-07 10:15:00 | 749.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-26 10:00:00 | 681.65 | 2023-06-07 10:15:00 | 749.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-26 10:30:00 | 682.80 | 2023-06-07 10:15:00 | 751.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-29 09:30:00 | 687.10 | 2023-06-12 13:15:00 | 755.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-31 14:00:00 | 702.10 | 2023-06-15 13:15:00 | 735.80 | STOP_HIT | 1.00 | 4.80% |
| BUY | retest2 | 2023-06-20 13:00:00 | 751.35 | 2023-06-21 13:15:00 | 746.85 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-06-20 14:45:00 | 750.60 | 2023-06-21 13:15:00 | 746.85 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-06-21 12:15:00 | 750.05 | 2023-06-21 13:15:00 | 746.85 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2023-06-26 13:00:00 | 735.70 | 2023-06-26 15:15:00 | 745.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-06-30 14:15:00 | 777.80 | 2023-07-05 13:15:00 | 769.35 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-06-30 15:15:00 | 779.00 | 2023-07-05 13:15:00 | 769.35 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-07-03 15:15:00 | 778.55 | 2023-07-05 13:15:00 | 769.35 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-07-04 09:30:00 | 781.25 | 2023-07-05 13:15:00 | 769.35 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2023-07-07 12:15:00 | 769.50 | 2023-07-10 09:15:00 | 784.35 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2023-07-14 09:15:00 | 798.00 | 2023-07-18 14:15:00 | 797.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2023-07-14 14:00:00 | 795.00 | 2023-07-18 14:15:00 | 797.00 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2023-07-18 11:45:00 | 796.15 | 2023-07-18 14:15:00 | 797.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-07-18 12:15:00 | 796.65 | 2023-07-18 14:15:00 | 797.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2023-08-16 09:15:00 | 726.25 | 2023-08-16 12:15:00 | 735.40 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-08-31 10:45:00 | 756.40 | 2023-09-08 15:15:00 | 772.15 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2023-08-31 11:30:00 | 755.90 | 2023-09-08 15:15:00 | 772.15 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2023-09-01 11:15:00 | 757.50 | 2023-09-08 15:15:00 | 772.15 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2023-09-01 12:00:00 | 763.25 | 2023-09-08 15:15:00 | 772.15 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2023-09-07 14:45:00 | 775.75 | 2023-09-08 15:15:00 | 772.15 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-09-08 09:15:00 | 778.05 | 2023-09-08 15:15:00 | 772.15 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-09-22 13:15:00 | 760.70 | 2023-09-25 14:15:00 | 772.35 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2023-09-22 14:45:00 | 763.25 | 2023-09-25 14:15:00 | 772.35 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-10-11 11:15:00 | 759.50 | 2023-10-11 11:15:00 | 764.15 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-10-19 13:30:00 | 768.05 | 2023-10-23 10:15:00 | 750.25 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2023-10-20 09:15:00 | 770.90 | 2023-10-23 10:15:00 | 750.25 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2023-11-09 11:30:00 | 760.65 | 2023-11-17 09:15:00 | 763.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2023-11-09 12:15:00 | 760.80 | 2023-11-17 09:15:00 | 763.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2023-11-15 12:00:00 | 756.65 | 2023-11-17 09:15:00 | 763.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-12-07 15:00:00 | 798.00 | 2023-12-11 14:15:00 | 758.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-08 09:30:00 | 796.20 | 2023-12-11 14:15:00 | 756.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-11 10:00:00 | 797.30 | 2023-12-11 14:15:00 | 757.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-11 12:30:00 | 798.30 | 2023-12-11 14:15:00 | 758.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-07 15:00:00 | 798.00 | 2023-12-12 14:15:00 | 768.55 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2023-12-08 09:30:00 | 796.20 | 2023-12-12 14:15:00 | 768.55 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2023-12-11 10:00:00 | 797.30 | 2023-12-12 14:15:00 | 768.55 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2023-12-11 12:30:00 | 798.30 | 2023-12-12 14:15:00 | 768.55 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2023-12-13 10:15:00 | 775.35 | 2023-12-14 09:15:00 | 786.15 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-12-13 14:00:00 | 775.40 | 2023-12-14 09:15:00 | 786.15 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-12-13 15:15:00 | 775.00 | 2023-12-14 09:15:00 | 786.15 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2023-12-18 11:45:00 | 786.15 | 2023-12-20 12:15:00 | 783.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-12-19 09:15:00 | 787.85 | 2023-12-20 13:15:00 | 764.80 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2023-12-19 10:45:00 | 787.30 | 2023-12-20 13:15:00 | 764.80 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2023-12-19 12:15:00 | 786.05 | 2023-12-20 13:15:00 | 764.80 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2023-12-20 09:30:00 | 787.25 | 2023-12-20 13:15:00 | 764.80 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2023-12-20 12:45:00 | 786.00 | 2023-12-20 13:15:00 | 764.80 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-01-08 09:15:00 | 788.15 | 2024-01-08 14:15:00 | 768.35 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-01-08 10:30:00 | 788.20 | 2024-01-08 14:15:00 | 768.35 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2024-01-11 11:45:00 | 767.35 | 2024-01-18 09:15:00 | 768.65 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-01-11 14:00:00 | 767.60 | 2024-01-18 09:15:00 | 768.65 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-01-12 09:15:00 | 763.30 | 2024-01-18 09:15:00 | 768.65 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-01-12 14:00:00 | 767.05 | 2024-01-18 09:15:00 | 768.65 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-01-31 10:45:00 | 776.75 | 2024-02-07 09:15:00 | 854.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 13:45:00 | 776.15 | 2024-02-07 09:15:00 | 853.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 14:30:00 | 777.35 | 2024-02-07 09:15:00 | 855.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 15:15:00 | 778.00 | 2024-02-07 09:15:00 | 855.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-06 09:15:00 | 805.85 | 2024-02-08 14:15:00 | 812.50 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2024-02-06 10:30:00 | 809.20 | 2024-02-08 14:15:00 | 812.50 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2024-02-12 10:15:00 | 797.30 | 2024-02-14 14:15:00 | 805.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-02-12 10:45:00 | 797.10 | 2024-02-14 14:15:00 | 805.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-02-13 14:00:00 | 797.00 | 2024-02-14 14:15:00 | 805.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-02-13 14:30:00 | 795.20 | 2024-02-14 14:15:00 | 805.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-02-14 09:15:00 | 785.90 | 2024-02-15 09:15:00 | 809.00 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-02-14 11:30:00 | 789.95 | 2024-02-15 09:15:00 | 809.00 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-02-14 12:00:00 | 791.00 | 2024-02-15 09:15:00 | 809.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-02-14 12:30:00 | 791.35 | 2024-02-15 09:15:00 | 809.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-02-19 09:15:00 | 813.85 | 2024-02-19 14:15:00 | 800.80 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-02-19 10:00:00 | 808.60 | 2024-02-19 14:15:00 | 800.80 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-02-19 12:45:00 | 807.85 | 2024-02-19 14:15:00 | 800.80 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-02-26 14:15:00 | 802.85 | 2024-02-26 14:15:00 | 797.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-02-27 09:15:00 | 802.00 | 2024-02-27 10:15:00 | 796.30 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-03-05 09:15:00 | 792.90 | 2024-03-06 09:15:00 | 768.40 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2024-03-05 10:45:00 | 791.65 | 2024-03-06 09:15:00 | 768.40 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2024-03-05 11:30:00 | 794.05 | 2024-03-06 09:15:00 | 768.40 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-03-05 15:15:00 | 796.00 | 2024-03-06 09:15:00 | 768.40 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-03-11 15:15:00 | 768.50 | 2024-03-13 12:15:00 | 730.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 15:15:00 | 768.50 | 2024-03-14 10:15:00 | 733.05 | STOP_HIT | 0.50 | 4.61% |
| BUY | retest2 | 2024-04-01 09:15:00 | 765.85 | 2024-04-12 10:15:00 | 799.10 | STOP_HIT | 1.00 | 4.34% |
| SELL | retest1 | 2024-04-18 10:30:00 | 768.80 | 2024-04-24 09:15:00 | 764.00 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-04-18 13:15:00 | 772.00 | 2024-04-29 10:15:00 | 733.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 13:15:00 | 772.00 | 2024-04-29 13:15:00 | 742.55 | STOP_HIT | 0.50 | 3.81% |
| BUY | retest2 | 2024-05-03 14:30:00 | 773.35 | 2024-05-06 09:15:00 | 762.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-05-13 09:15:00 | 732.50 | 2024-05-13 15:15:00 | 739.70 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-05-13 13:15:00 | 731.00 | 2024-05-13 15:15:00 | 739.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-05-18 09:30:00 | 759.95 | 2024-05-21 10:15:00 | 752.45 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-05-24 14:15:00 | 738.05 | 2024-05-27 13:15:00 | 747.95 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-05-27 09:30:00 | 737.55 | 2024-05-27 13:15:00 | 747.95 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-06-11 09:15:00 | 781.50 | 2024-06-18 09:15:00 | 859.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-11 10:30:00 | 879.00 | 2024-07-16 09:15:00 | 889.65 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-07-11 12:00:00 | 879.65 | 2024-07-16 09:15:00 | 889.65 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-07-23 11:45:00 | 839.45 | 2024-07-23 12:15:00 | 797.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 11:45:00 | 839.45 | 2024-07-23 13:15:00 | 844.40 | STOP_HIT | 0.50 | -0.59% |
| SELL | retest2 | 2024-07-24 09:45:00 | 839.10 | 2024-07-26 11:15:00 | 863.30 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-07-25 09:15:00 | 830.55 | 2024-07-26 11:15:00 | 863.30 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2024-08-05 09:15:00 | 800.00 | 2024-08-09 10:15:00 | 795.85 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2024-08-26 09:30:00 | 851.95 | 2024-08-26 10:15:00 | 845.45 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-09-06 14:45:00 | 867.35 | 2024-09-12 09:15:00 | 882.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-09-06 15:15:00 | 867.00 | 2024-09-12 09:15:00 | 882.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-09-09 11:30:00 | 865.55 | 2024-09-12 09:15:00 | 882.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-09-09 12:15:00 | 865.35 | 2024-09-12 09:15:00 | 882.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-10 12:15:00 | 854.50 | 2024-09-12 09:15:00 | 882.00 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2024-09-10 13:00:00 | 854.60 | 2024-09-12 09:15:00 | 882.00 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2024-09-18 11:15:00 | 881.30 | 2024-09-19 11:15:00 | 837.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-18 12:45:00 | 881.00 | 2024-09-19 11:15:00 | 836.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-18 11:15:00 | 881.30 | 2024-09-20 10:15:00 | 861.40 | STOP_HIT | 0.50 | 2.26% |
| SELL | retest2 | 2024-09-18 12:45:00 | 881.00 | 2024-09-20 10:15:00 | 861.40 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2024-09-19 09:30:00 | 872.15 | 2024-09-23 12:15:00 | 883.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-10-04 09:15:00 | 857.25 | 2024-10-09 09:15:00 | 873.60 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-10-14 11:15:00 | 871.20 | 2024-10-17 10:15:00 | 862.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-10-17 10:15:00 | 869.85 | 2024-10-17 10:15:00 | 862.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-10-22 10:15:00 | 843.60 | 2024-10-23 09:15:00 | 868.10 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest1 | 2024-11-12 13:15:00 | 835.55 | 2024-11-18 11:15:00 | 827.80 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest1 | 2024-11-12 14:00:00 | 834.65 | 2024-11-18 11:15:00 | 827.80 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2024-11-14 13:30:00 | 824.95 | 2024-11-19 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2024-11-14 14:15:00 | 823.55 | 2024-11-19 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-11-14 14:45:00 | 824.05 | 2024-11-19 09:15:00 | 848.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-11-28 10:15:00 | 825.20 | 2024-12-03 10:15:00 | 829.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-11-29 10:00:00 | 819.30 | 2024-12-03 10:15:00 | 829.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-12-02 10:00:00 | 822.40 | 2024-12-03 10:15:00 | 829.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-12-02 11:45:00 | 824.85 | 2024-12-03 10:15:00 | 829.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-12-02 14:00:00 | 822.80 | 2024-12-03 10:15:00 | 829.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-12-31 13:45:00 | 759.00 | 2025-01-01 11:15:00 | 740.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-01-01 09:30:00 | 757.70 | 2025-01-01 11:15:00 | 740.00 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-01-06 10:45:00 | 712.65 | 2025-01-10 09:15:00 | 677.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 13:45:00 | 713.60 | 2025-01-10 09:15:00 | 677.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 15:00:00 | 713.65 | 2025-01-10 09:15:00 | 677.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 695.70 | 2025-01-13 12:15:00 | 660.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 685.40 | 2025-01-13 14:15:00 | 651.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 10:45:00 | 712.65 | 2025-01-14 10:15:00 | 665.05 | STOP_HIT | 0.50 | 6.68% |
| SELL | retest2 | 2025-01-07 13:45:00 | 713.60 | 2025-01-14 10:15:00 | 665.05 | STOP_HIT | 0.50 | 6.80% |
| SELL | retest2 | 2025-01-07 15:00:00 | 713.65 | 2025-01-14 10:15:00 | 665.05 | STOP_HIT | 0.50 | 6.81% |
| SELL | retest2 | 2025-01-08 09:15:00 | 695.70 | 2025-01-14 10:15:00 | 665.05 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-01-10 09:15:00 | 685.40 | 2025-01-14 10:15:00 | 665.05 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2025-01-30 10:15:00 | 666.70 | 2025-02-01 09:15:00 | 674.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-01-31 13:45:00 | 665.50 | 2025-02-01 09:15:00 | 674.60 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-02-07 11:15:00 | 677.90 | 2025-02-10 09:15:00 | 651.85 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-02-18 09:15:00 | 599.30 | 2025-02-20 10:15:00 | 611.50 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-02-27 14:30:00 | 576.60 | 2025-03-04 10:15:00 | 586.65 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-02-28 09:15:00 | 572.40 | 2025-03-04 10:15:00 | 586.65 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-02-28 12:30:00 | 572.65 | 2025-03-04 10:15:00 | 586.65 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-02-28 14:15:00 | 575.00 | 2025-03-04 10:15:00 | 586.65 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-03-05 11:00:00 | 583.95 | 2025-03-10 09:15:00 | 642.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-05 11:30:00 | 585.75 | 2025-03-10 09:15:00 | 644.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-25 15:00:00 | 680.00 | 2025-03-26 14:15:00 | 670.55 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-04-01 10:30:00 | 664.10 | 2025-04-02 15:15:00 | 673.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-04-01 14:00:00 | 664.55 | 2025-04-02 15:15:00 | 673.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-04-01 15:00:00 | 663.00 | 2025-04-02 15:15:00 | 673.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest1 | 2025-05-16 09:15:00 | 749.25 | 2025-05-19 14:15:00 | 745.65 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-05-16 11:00:00 | 744.90 | 2025-05-19 14:15:00 | 745.65 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-05-29 14:45:00 | 778.30 | 2025-06-05 12:15:00 | 782.05 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-05-30 12:45:00 | 780.75 | 2025-06-05 12:15:00 | 782.05 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-06-02 13:00:00 | 780.70 | 2025-06-05 12:15:00 | 782.05 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-06-04 09:30:00 | 779.60 | 2025-06-05 12:15:00 | 782.05 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-06-26 09:15:00 | 785.35 | 2025-07-01 11:15:00 | 788.55 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-06-26 12:15:00 | 785.60 | 2025-07-01 11:15:00 | 788.55 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-06-26 13:00:00 | 790.20 | 2025-07-01 11:15:00 | 788.55 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-04 12:45:00 | 815.00 | 2025-07-08 14:15:00 | 815.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest1 | 2025-07-07 12:30:00 | 816.05 | 2025-07-08 14:15:00 | 815.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-07-09 09:15:00 | 819.40 | 2025-07-10 09:15:00 | 811.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-15 15:15:00 | 796.00 | 2025-07-16 13:15:00 | 805.55 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-16 09:30:00 | 796.00 | 2025-07-16 13:15:00 | 805.55 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-16 11:15:00 | 796.00 | 2025-07-16 13:15:00 | 805.55 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-30 10:15:00 | 740.50 | 2025-08-04 10:15:00 | 755.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-07-30 11:15:00 | 740.05 | 2025-08-04 10:15:00 | 755.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-07-30 11:45:00 | 740.00 | 2025-08-04 12:15:00 | 753.60 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-07-31 09:45:00 | 740.15 | 2025-08-04 12:15:00 | 753.60 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-08-01 14:15:00 | 737.15 | 2025-08-04 12:15:00 | 753.60 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-08-01 15:00:00 | 738.85 | 2025-08-04 12:15:00 | 753.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-09-05 09:45:00 | 762.50 | 2025-09-05 10:15:00 | 749.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-10 13:00:00 | 740.00 | 2025-09-12 11:15:00 | 750.05 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-09-19 09:15:00 | 777.45 | 2025-09-22 15:15:00 | 771.65 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-22 09:15:00 | 777.70 | 2025-09-22 15:15:00 | 771.65 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-22 10:45:00 | 776.50 | 2025-09-22 15:15:00 | 771.65 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2025-10-06 10:15:00 | 790.40 | 2025-10-08 11:15:00 | 790.25 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-10-17 10:30:00 | 805.95 | 2025-11-03 15:15:00 | 864.55 | STOP_HIT | 1.00 | 7.27% |
| BUY | retest2 | 2025-10-20 09:15:00 | 818.00 | 2025-11-03 15:15:00 | 864.55 | STOP_HIT | 1.00 | 5.69% |
| BUY | retest1 | 2025-11-11 09:15:00 | 888.65 | 2025-11-12 14:15:00 | 886.35 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-11-18 09:15:00 | 905.00 | 2025-11-19 12:15:00 | 887.10 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-11-19 11:15:00 | 896.70 | 2025-11-19 12:15:00 | 887.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-11-21 11:15:00 | 886.00 | 2025-11-24 12:15:00 | 891.30 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-21 13:00:00 | 886.00 | 2025-11-24 12:15:00 | 891.30 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-24 10:15:00 | 884.70 | 2025-11-24 12:15:00 | 891.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-03 09:15:00 | 917.75 | 2025-12-03 11:15:00 | 891.60 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-12-10 09:15:00 | 911.95 | 2025-12-10 14:15:00 | 901.80 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-11 09:15:00 | 912.95 | 2025-12-11 12:15:00 | 902.10 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-11 10:00:00 | 911.80 | 2025-12-11 12:15:00 | 902.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-11 11:45:00 | 909.80 | 2025-12-11 12:15:00 | 902.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-17 10:30:00 | 926.20 | 2025-12-23 15:15:00 | 926.45 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-12-17 14:00:00 | 923.90 | 2025-12-23 15:15:00 | 926.45 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-12-18 10:00:00 | 923.85 | 2025-12-23 15:15:00 | 926.45 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-12-18 14:45:00 | 924.20 | 2025-12-23 15:15:00 | 926.45 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2025-12-19 10:15:00 | 922.00 | 2025-12-23 15:15:00 | 926.45 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2026-01-01 09:30:00 | 930.15 | 2026-01-01 10:15:00 | 926.65 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2026-01-01 10:15:00 | 930.40 | 2026-01-01 10:15:00 | 926.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-01-01 14:00:00 | 931.10 | 2026-01-02 12:15:00 | 922.45 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-01 15:15:00 | 934.00 | 2026-01-02 12:15:00 | 922.45 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-01-02 09:15:00 | 934.40 | 2026-01-02 12:15:00 | 922.45 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-02 09:45:00 | 936.25 | 2026-01-02 12:15:00 | 922.45 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-02 12:15:00 | 934.40 | 2026-01-02 12:15:00 | 922.45 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-06 09:15:00 | 945.95 | 2026-01-07 12:15:00 | 933.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-01-07 09:45:00 | 947.45 | 2026-01-07 12:15:00 | 933.10 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-07 10:30:00 | 945.25 | 2026-01-07 12:15:00 | 933.10 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-19 11:45:00 | 930.15 | 2026-01-21 09:15:00 | 898.55 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-01-20 10:45:00 | 926.25 | 2026-01-21 09:15:00 | 898.55 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2026-01-20 14:30:00 | 929.40 | 2026-01-21 09:15:00 | 898.55 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-01-30 14:15:00 | 909.55 | 2026-02-01 15:15:00 | 893.30 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-02-05 09:15:00 | 941.40 | 2026-02-10 11:15:00 | 925.85 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-02-06 11:00:00 | 940.90 | 2026-02-10 11:15:00 | 925.85 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-09 09:15:00 | 954.30 | 2026-02-10 11:15:00 | 925.85 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2026-02-19 11:15:00 | 902.45 | 2026-02-19 13:15:00 | 897.05 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-03-09 09:15:00 | 803.55 | 2026-03-10 11:15:00 | 834.00 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2026-03-12 11:00:00 | 867.75 | 2026-03-13 12:15:00 | 829.60 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2026-04-02 09:15:00 | 798.30 | 2026-04-02 14:15:00 | 824.35 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2026-04-02 09:45:00 | 796.80 | 2026-04-02 14:15:00 | 824.35 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2026-04-13 10:45:00 | 826.70 | 2026-04-13 11:15:00 | 828.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2026-04-21 09:15:00 | 897.55 | 2026-04-24 11:15:00 | 880.85 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-05-04 11:15:00 | 888.00 | 2026-05-04 12:15:00 | 890.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-05-04 12:00:00 | 886.25 | 2026-05-04 12:15:00 | 890.90 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2026-05-08 09:15:00 | 890.20 | 2026-05-08 11:15:00 | 882.65 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-05-08 09:45:00 | 892.95 | 2026-05-08 11:15:00 | 882.65 | STOP_HIT | 1.00 | -1.15% |
