# Chennai Petroleum Corporation Ltd. (CHENNPETRO)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1079.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 60 |
| ALERT1 | 46 |
| ALERT2 | 47 |
| ALERT2_SKIP | 25 |
| ALERT3 | 104 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 66 |
| PARTIAL | 6 |
| TARGET_HIT | 8 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 24 / 49
- **Target hits / Stop hits / Partials:** 8 / 59 / 6
- **Avg / median % per leg:** 0.54% / -1.02%
- **Sum % (uncompounded):** 39.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 12 | 30.0% | 6 | 34 | 0 | 0.34% | 13.5% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.36% | -4.1% |
| BUY @ 3rd Alert (retest2) | 37 | 11 | 29.7% | 6 | 31 | 0 | 0.47% | 17.6% |
| SELL (all) | 33 | 12 | 36.4% | 2 | 25 | 6 | 0.79% | 26.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.15% | -1.1% |
| SELL @ 3rd Alert (retest2) | 32 | 12 | 37.5% | 2 | 24 | 6 | 0.85% | 27.3% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.31% | -5.2% |
| retest2 (combined) | 69 | 23 | 33.3% | 8 | 55 | 6 | 0.65% | 44.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 618.55 | 609.26 | 608.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 622.65 | 615.26 | 611.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 645.00 | 649.58 | 644.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 13:15:00 | 645.00 | 649.58 | 644.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 645.00 | 649.58 | 644.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 645.00 | 649.58 | 644.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 653.40 | 650.35 | 644.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:00:00 | 671.50 | 654.60 | 647.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 672.80 | 702.20 | 704.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 672.80 | 702.20 | 704.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 668.05 | 695.37 | 700.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 654.30 | 654.13 | 663.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:45:00 | 653.70 | 654.13 | 663.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 659.95 | 656.12 | 661.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 655.00 | 656.84 | 660.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:00:00 | 654.15 | 656.30 | 659.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 653.90 | 651.89 | 654.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 657.25 | 655.68 | 655.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 657.25 | 655.68 | 655.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 657.25 | 655.68 | 655.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 657.25 | 655.68 | 655.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 662.55 | 657.05 | 656.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 672.00 | 679.42 | 672.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 672.00 | 679.42 | 672.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 672.00 | 679.42 | 672.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 672.00 | 679.42 | 672.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 670.85 | 677.71 | 672.50 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 664.65 | 669.41 | 669.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 651.85 | 665.90 | 668.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 642.40 | 633.99 | 642.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 642.40 | 633.99 | 642.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 642.40 | 633.99 | 642.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 641.20 | 633.99 | 642.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 637.50 | 635.53 | 641.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:30:00 | 633.50 | 635.96 | 639.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 633.00 | 635.96 | 639.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 632.35 | 634.49 | 638.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 643.80 | 629.86 | 628.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 643.80 | 629.86 | 628.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 643.80 | 629.86 | 628.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 643.80 | 629.86 | 628.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 673.90 | 638.67 | 632.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 679.10 | 680.27 | 664.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:00:00 | 679.10 | 680.27 | 664.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 668.35 | 676.26 | 670.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:00:00 | 668.35 | 676.26 | 670.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 667.50 | 674.51 | 670.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:30:00 | 667.00 | 674.51 | 670.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 671.50 | 673.35 | 671.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 671.50 | 673.35 | 671.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 672.65 | 673.21 | 671.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:30:00 | 671.05 | 673.21 | 671.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 672.20 | 673.01 | 671.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 673.30 | 673.01 | 671.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 669.55 | 672.32 | 671.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 674.45 | 672.74 | 671.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 11:15:00 | 741.90 | 724.96 | 711.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 727.00 | 735.23 | 735.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 720.45 | 730.15 | 732.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 722.70 | 720.45 | 724.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 722.70 | 720.45 | 724.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 722.05 | 720.77 | 724.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 719.65 | 720.79 | 724.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 719.80 | 721.03 | 724.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 727.15 | 722.06 | 723.99 | SL hit (close>static) qty=1.00 sl=725.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 727.15 | 722.06 | 723.99 | SL hit (close>static) qty=1.00 sl=725.60 alert=retest2 |

### Cycle 7 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 734.00 | 725.87 | 725.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 739.45 | 728.58 | 726.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 745.50 | 748.04 | 743.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:45:00 | 745.50 | 748.04 | 743.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 754.05 | 749.19 | 745.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:15:00 | 759.50 | 750.19 | 745.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:45:00 | 771.20 | 754.51 | 749.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 762.40 | 771.62 | 771.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 762.40 | 771.62 | 771.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 762.40 | 771.62 | 771.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 759.50 | 769.20 | 770.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 13:15:00 | 727.80 | 717.45 | 733.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:00:00 | 727.80 | 717.45 | 733.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 696.25 | 705.69 | 716.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 688.30 | 698.77 | 707.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 653.88 | 668.74 | 681.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 640.95 | 637.60 | 646.31 | SL hit (close>ema200) qty=0.50 sl=637.60 alert=retest2 |

### Cycle 9 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 652.15 | 643.45 | 642.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 656.00 | 648.50 | 645.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 658.75 | 658.76 | 654.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:30:00 | 658.50 | 658.76 | 654.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 656.90 | 658.39 | 654.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 655.15 | 658.39 | 654.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 656.30 | 657.67 | 655.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 655.60 | 657.67 | 655.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 653.50 | 656.83 | 654.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 653.50 | 656.83 | 654.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 651.90 | 655.85 | 654.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 651.90 | 655.85 | 654.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 641.70 | 652.24 | 653.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 639.40 | 649.67 | 651.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 639.05 | 633.93 | 638.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 639.05 | 633.93 | 638.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 639.05 | 633.93 | 638.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:15:00 | 642.70 | 633.93 | 638.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 644.10 | 635.96 | 639.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 644.10 | 635.96 | 639.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 652.30 | 639.23 | 640.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 652.30 | 639.23 | 640.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 649.00 | 642.59 | 641.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 656.25 | 645.32 | 643.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 662.55 | 665.49 | 659.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 662.55 | 665.49 | 659.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 659.30 | 663.72 | 660.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 656.30 | 663.72 | 660.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 658.20 | 662.62 | 659.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 658.20 | 662.62 | 659.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 660.50 | 662.19 | 659.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:00:00 | 662.15 | 662.19 | 660.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 663.80 | 660.99 | 660.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 662.65 | 661.15 | 660.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 655.55 | 660.02 | 660.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 655.55 | 660.02 | 660.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 655.55 | 660.02 | 660.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 655.55 | 660.02 | 660.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 650.00 | 655.27 | 657.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 648.40 | 648.30 | 651.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 648.40 | 648.30 | 651.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 652.60 | 647.57 | 649.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 653.35 | 647.57 | 649.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 668.55 | 651.77 | 651.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 683.00 | 658.01 | 654.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 696.15 | 697.80 | 687.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 696.15 | 697.80 | 687.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 697.45 | 699.10 | 693.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 14:30:00 | 698.95 | 698.18 | 693.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 690.00 | 696.55 | 693.48 | SL hit (close<static) qty=1.00 sl=692.10 alert=retest2 |

### Cycle 14 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 687.50 | 691.65 | 691.90 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 700.00 | 693.32 | 692.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 14:15:00 | 702.65 | 695.19 | 693.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 702.50 | 703.07 | 699.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 702.50 | 703.07 | 699.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 736.65 | 740.92 | 735.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 736.65 | 740.92 | 735.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 732.60 | 739.26 | 735.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 733.60 | 739.26 | 735.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 733.45 | 738.10 | 734.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 733.80 | 738.10 | 734.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 749.00 | 739.69 | 736.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 753.40 | 743.87 | 739.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 753.90 | 746.30 | 741.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:30:00 | 755.60 | 750.83 | 744.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 754.55 | 752.25 | 747.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 747.25 | 751.35 | 748.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 747.25 | 751.35 | 748.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 747.10 | 750.50 | 748.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 747.10 | 750.50 | 748.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 748.00 | 750.00 | 748.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 748.25 | 750.00 | 748.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 741.50 | 748.30 | 747.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 741.50 | 748.30 | 747.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 739.40 | 746.52 | 746.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 739.40 | 746.52 | 746.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 739.40 | 746.52 | 746.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 739.40 | 746.52 | 746.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 739.40 | 746.52 | 746.82 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 752.00 | 747.14 | 746.87 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 744.10 | 746.53 | 746.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 15:15:00 | 744.00 | 745.55 | 746.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 746.50 | 745.74 | 746.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 746.50 | 745.74 | 746.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 746.50 | 745.74 | 746.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 746.40 | 745.74 | 746.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 747.15 | 746.02 | 746.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 747.15 | 746.02 | 746.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 744.10 | 745.64 | 746.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:15:00 | 743.00 | 745.76 | 746.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 754.90 | 746.95 | 746.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 754.90 | 746.95 | 746.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 762.50 | 751.30 | 748.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 749.00 | 754.37 | 751.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 749.00 | 754.37 | 751.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 749.00 | 754.37 | 751.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 749.00 | 754.37 | 751.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 746.00 | 752.69 | 751.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 746.90 | 752.69 | 751.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 754.40 | 752.64 | 751.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 759.50 | 753.14 | 751.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 756.70 | 754.03 | 752.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 759.75 | 753.83 | 752.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:30:00 | 757.85 | 758.28 | 756.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 760.50 | 762.24 | 759.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 766.30 | 762.24 | 759.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 765.10 | 767.11 | 764.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 754.55 | 761.47 | 762.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 764.85 | 761.17 | 762.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 764.85 | 761.17 | 762.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 764.85 | 761.17 | 762.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 764.85 | 761.17 | 762.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 758.00 | 760.54 | 761.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:00:00 | 750.95 | 758.62 | 760.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 765.60 | 760.35 | 760.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 765.60 | 760.35 | 760.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 773.85 | 763.05 | 761.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 804.80 | 806.71 | 796.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 15:15:00 | 804.80 | 806.71 | 796.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 804.80 | 806.71 | 796.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 811.00 | 806.71 | 796.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 811.75 | 806.77 | 797.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 809.95 | 803.57 | 799.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 785.00 | 798.32 | 799.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 785.00 | 798.32 | 799.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 785.00 | 798.32 | 799.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 785.00 | 798.32 | 799.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 775.00 | 787.00 | 792.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 772.50 | 770.93 | 778.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 772.50 | 770.93 | 778.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 744.75 | 760.95 | 770.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:45:00 | 740.90 | 754.33 | 765.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 743.15 | 752.91 | 763.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:30:00 | 741.85 | 750.09 | 761.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 777.10 | 746.67 | 745.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 777.10 | 746.67 | 745.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 777.10 | 746.67 | 745.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 777.10 | 746.67 | 745.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 780.70 | 764.00 | 755.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 748.95 | 763.23 | 756.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 748.95 | 763.23 | 756.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 748.95 | 763.23 | 756.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 740.65 | 763.23 | 756.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 743.60 | 759.30 | 755.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 740.70 | 759.30 | 755.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 743.95 | 751.80 | 752.54 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 765.00 | 752.73 | 752.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 775.80 | 759.43 | 755.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 11:15:00 | 764.00 | 771.75 | 765.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 764.00 | 771.75 | 765.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 764.00 | 771.75 | 765.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 757.50 | 771.75 | 765.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 796.70 | 776.74 | 768.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 13:45:00 | 835.35 | 797.25 | 782.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 846.80 | 820.23 | 807.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-31 09:15:00 | 918.89 | 871.24 | 842.94 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-31 10:15:00 | 931.48 | 884.62 | 851.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1036.20 | 1067.51 | 1070.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1031.00 | 1048.58 | 1059.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 1018.45 | 1004.04 | 1017.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 1018.45 | 1004.04 | 1017.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1018.45 | 1004.04 | 1017.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 1021.00 | 1004.04 | 1017.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1019.45 | 1007.12 | 1017.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1019.45 | 1007.12 | 1017.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1015.00 | 1008.70 | 1017.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 976.85 | 1008.70 | 1017.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 12:15:00 | 928.01 | 969.58 | 994.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 938.00 | 933.15 | 960.79 | SL hit (close>ema200) qty=0.50 sl=933.15 alert=retest2 |

### Cycle 27 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 933.70 | 928.33 | 927.79 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 914.60 | 926.09 | 926.90 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 937.55 | 927.03 | 926.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 14:15:00 | 943.15 | 932.27 | 929.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 13:15:00 | 931.45 | 938.79 | 934.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 13:15:00 | 931.45 | 938.79 | 934.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 931.45 | 938.79 | 934.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 931.45 | 938.79 | 934.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 932.60 | 937.55 | 934.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:15:00 | 926.50 | 937.55 | 934.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 932.00 | 934.11 | 933.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 934.45 | 934.11 | 933.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 927.95 | 932.88 | 933.09 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 09:15:00 | 945.00 | 933.67 | 933.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 10:15:00 | 945.60 | 936.06 | 934.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 930.00 | 935.69 | 934.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 13:15:00 | 930.00 | 935.69 | 934.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 930.00 | 935.69 | 934.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 930.00 | 935.69 | 934.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 933.45 | 935.24 | 934.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 928.10 | 935.24 | 934.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 928.00 | 933.79 | 933.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 901.55 | 927.34 | 931.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 926.00 | 923.72 | 927.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 926.00 | 923.72 | 927.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 947.30 | 927.29 | 928.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 940.90 | 927.29 | 928.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 947.00 | 931.23 | 929.97 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 920.25 | 932.50 | 932.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 907.50 | 916.58 | 921.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 917.70 | 915.71 | 919.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 12:15:00 | 917.70 | 915.71 | 919.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 917.70 | 915.71 | 919.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 910.10 | 915.01 | 918.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 912.50 | 914.85 | 918.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 13:15:00 | 864.60 | 877.95 | 886.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 13:15:00 | 866.88 | 877.95 | 886.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-29 12:15:00 | 819.09 | 829.16 | 842.44 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-29 12:15:00 | 821.25 | 829.16 | 842.44 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 35 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 846.05 | 826.05 | 824.94 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 14:15:00 | 825.95 | 832.63 | 833.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 817.00 | 829.08 | 831.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 813.30 | 809.32 | 814.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 813.30 | 809.32 | 814.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 813.30 | 809.32 | 814.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 814.10 | 809.32 | 814.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 809.50 | 809.35 | 814.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 808.25 | 809.35 | 814.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 817.80 | 810.85 | 813.78 | SL hit (close>static) qty=1.00 sl=815.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 808.50 | 811.98 | 814.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 768.07 | 783.39 | 792.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 794.05 | 782.57 | 789.57 | SL hit (close>ema200) qty=0.50 sl=782.57 alert=retest2 |

### Cycle 37 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 813.90 | 794.63 | 793.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 822.40 | 807.35 | 801.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 852.70 | 857.97 | 838.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:00:00 | 852.70 | 857.97 | 838.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 854.10 | 859.04 | 847.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 851.15 | 859.04 | 847.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 849.50 | 857.13 | 848.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 849.10 | 857.13 | 848.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 849.60 | 855.63 | 848.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:30:00 | 849.85 | 855.63 | 848.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 848.50 | 854.20 | 848.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:30:00 | 846.65 | 854.20 | 848.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 852.05 | 853.77 | 848.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:45:00 | 848.45 | 853.77 | 848.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 847.35 | 852.49 | 848.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 847.35 | 852.49 | 848.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 843.05 | 850.60 | 847.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 837.70 | 850.60 | 847.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 827.00 | 845.88 | 846.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 814.85 | 839.67 | 843.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 820.60 | 818.81 | 829.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:00:00 | 820.60 | 818.81 | 829.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 810.55 | 817.15 | 827.61 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 858.50 | 832.57 | 831.61 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 15:15:00 | 830.00 | 831.94 | 832.14 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 843.70 | 834.29 | 833.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 866.25 | 842.18 | 837.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 14:15:00 | 836.40 | 843.81 | 840.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 836.40 | 843.81 | 840.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 836.40 | 843.81 | 840.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 834.90 | 843.81 | 840.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 845.50 | 844.15 | 840.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 861.00 | 844.15 | 840.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:30:00 | 850.20 | 847.29 | 842.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 839.90 | 858.78 | 859.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 839.90 | 858.78 | 859.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 839.90 | 858.78 | 859.84 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 875.00 | 850.32 | 849.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 882.10 | 856.68 | 852.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 888.55 | 889.46 | 878.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:15:00 | 903.70 | 889.46 | 878.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 921.30 | 924.59 | 919.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 925.25 | 924.59 | 919.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 918.65 | 923.40 | 919.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 918.65 | 923.40 | 919.73 | SL hit (close<ema400) qty=1.00 sl=919.73 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 918.65 | 923.40 | 919.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 893.10 | 917.34 | 917.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 893.10 | 917.34 | 917.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 902.30 | 914.33 | 915.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 877.80 | 902.48 | 909.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 889.05 | 885.37 | 892.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 889.05 | 885.37 | 892.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 883.05 | 885.07 | 890.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 879.55 | 885.07 | 890.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 910.35 | 890.20 | 889.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 910.35 | 890.20 | 889.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 918.85 | 907.59 | 900.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 909.75 | 911.02 | 904.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 14:00:00 | 909.75 | 911.02 | 904.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 896.55 | 908.13 | 904.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 896.55 | 908.13 | 904.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 891.90 | 904.88 | 902.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 903.30 | 902.72 | 902.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 897.65 | 902.72 | 902.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 894.45 | 901.06 | 901.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 894.45 | 901.06 | 901.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 894.45 | 901.06 | 901.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 885.55 | 897.96 | 900.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 10:15:00 | 895.60 | 889.96 | 894.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 895.60 | 889.96 | 894.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 895.60 | 889.96 | 894.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 895.60 | 889.96 | 894.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 892.15 | 890.40 | 894.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:30:00 | 896.45 | 890.40 | 894.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 886.95 | 889.71 | 893.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:00:00 | 885.95 | 888.96 | 892.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:30:00 | 884.15 | 887.79 | 891.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 900.50 | 890.12 | 892.18 | SL hit (close>static) qty=1.00 sl=894.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 900.50 | 890.12 | 892.18 | SL hit (close>static) qty=1.00 sl=894.85 alert=retest2 |

### Cycle 47 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 909.25 | 893.95 | 893.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 13:15:00 | 912.00 | 901.70 | 897.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 14:15:00 | 915.95 | 917.09 | 909.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 915.95 | 917.09 | 909.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 917.00 | 917.08 | 911.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 921.80 | 918.26 | 912.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 09:30:00 | 938.00 | 920.82 | 915.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:45:00 | 924.15 | 941.03 | 935.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-05 09:15:00 | 1013.98 | 990.83 | 967.10 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-05 09:15:00 | 1031.80 | 990.83 | 967.10 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-05 09:15:00 | 1016.57 | 990.83 | 967.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 944.50 | 986.32 | 987.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 941.95 | 965.69 | 977.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 947.25 | 923.39 | 934.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 947.25 | 923.39 | 934.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 947.25 | 923.39 | 934.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 947.25 | 923.39 | 934.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 954.10 | 929.53 | 936.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:15:00 | 954.25 | 929.53 | 936.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 953.15 | 934.26 | 937.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:30:00 | 946.80 | 939.14 | 939.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 899.46 | 923.98 | 929.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 927.95 | 923.98 | 929.92 | SL hit (close>static) qty=0.50 sl=923.98 alert=retest2 |

### Cycle 49 — BUY (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 13:15:00 | 951.00 | 934.71 | 933.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 14:15:00 | 986.95 | 945.15 | 938.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 09:15:00 | 1016.75 | 1024.98 | 993.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 09:45:00 | 1019.90 | 1024.98 | 993.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1034.50 | 1020.26 | 1005.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1076.90 | 1029.57 | 1017.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:45:00 | 1065.10 | 1037.58 | 1022.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1007.35 | 1027.65 | 1030.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1007.35 | 1027.65 | 1030.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1007.35 | 1027.65 | 1030.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 09:15:00 | 960.80 | 1010.17 | 1021.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 14:15:00 | 991.50 | 985.97 | 1002.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 14:15:00 | 991.50 | 985.97 | 1002.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 991.50 | 985.97 | 1002.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 991.50 | 985.97 | 1002.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1022.00 | 993.87 | 1003.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1022.00 | 993.87 | 1003.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1023.40 | 999.78 | 1005.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 1019.70 | 1003.82 | 1006.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 1019.40 | 1003.82 | 1006.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 13:15:00 | 1020.00 | 1010.10 | 1009.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 13:15:00 | 1020.00 | 1010.10 | 1009.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1020.00 | 1010.10 | 1009.19 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 1000.00 | 1007.77 | 1008.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 949.95 | 996.21 | 1002.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 09:15:00 | 982.75 | 968.80 | 981.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 982.75 | 968.80 | 981.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 982.75 | 968.80 | 981.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:30:00 | 997.30 | 968.80 | 981.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 974.70 | 969.98 | 980.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 15:00:00 | 965.80 | 973.77 | 979.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 993.00 | 977.33 | 980.17 | SL hit (close>static) qty=1.00 sl=982.90 alert=retest2 |

### Cycle 53 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1008.85 | 985.98 | 983.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 1015.20 | 991.83 | 986.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 985.35 | 999.37 | 992.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 985.35 | 999.37 | 992.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 985.35 | 999.37 | 992.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 985.35 | 999.37 | 992.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 1000.70 | 999.63 | 993.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 1013.00 | 1003.14 | 995.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 1013.25 | 1014.97 | 1005.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 991.40 | 1000.11 | 1000.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 991.40 | 1000.11 | 1000.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 13:15:00 | 991.40 | 1000.11 | 1000.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 14:15:00 | 981.00 | 996.29 | 998.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1008.90 | 983.83 | 987.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1008.90 | 983.83 | 987.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1008.90 | 983.83 | 987.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 1008.90 | 983.83 | 987.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 993.25 | 985.71 | 988.38 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 999.00 | 990.88 | 990.42 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 11:15:00 | 988.05 | 990.14 | 990.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 12:15:00 | 986.00 | 989.31 | 989.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 15:15:00 | 974.40 | 974.37 | 980.02 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 955.45 | 974.37 | 980.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 953.20 | 953.54 | 963.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 966.40 | 956.11 | 964.16 | SL hit (close>ema400) qty=1.00 sl=964.16 alert=retest1 |

### Cycle 57 — BUY (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 15:15:00 | 972.70 | 968.73 | 968.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 989.10 | 972.81 | 970.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 980.80 | 981.90 | 976.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 15:00:00 | 980.80 | 981.90 | 976.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1008.45 | 1011.70 | 1005.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1019.90 | 1011.70 | 1005.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 1019.30 | 1059.16 | 1055.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:30:00 | 1016.10 | 1054.33 | 1053.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 1037.50 | 1050.96 | 1052.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 1037.50 | 1050.96 | 1052.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 1037.50 | 1050.96 | 1052.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 1037.50 | 1050.96 | 1052.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 09:15:00 | 998.00 | 1040.37 | 1047.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 09:15:00 | 1048.80 | 1017.07 | 1027.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 1048.80 | 1017.07 | 1027.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1048.80 | 1017.07 | 1027.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 1067.00 | 1017.07 | 1027.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1078.20 | 1029.30 | 1032.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 1078.20 | 1029.30 | 1032.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 1058.70 | 1035.18 | 1034.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1117.50 | 1064.36 | 1049.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 1113.40 | 1117.45 | 1093.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 12:30:00 | 1124.05 | 1118.58 | 1097.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 14:30:00 | 1130.45 | 1121.47 | 1102.81 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1122.20 | 1122.02 | 1106.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 1094.90 | 1111.11 | 1107.98 | SL hit (close<ema400) qty=1.00 sl=1107.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 1094.90 | 1111.11 | 1107.98 | SL hit (close<ema400) qty=1.00 sl=1107.98 alert=retest1 |

### Cycle 60 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 1089.50 | 1103.52 | 1104.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 1074.10 | 1089.04 | 1096.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 10:15:00 | 1074.10 | 1074.04 | 1082.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 10:15:00 | 1074.10 | 1074.04 | 1082.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1074.10 | 1074.04 | 1082.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:30:00 | 1080.00 | 1074.04 | 1082.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1088.60 | 1078.43 | 1081.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 12:45:00 | 1081.40 | 1081.49 | 1082.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 13:30:00 | 1080.50 | 1080.59 | 1081.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 14:45:00 | 1079.10 | 1079.85 | 1081.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 10:00:00 | 671.50 | 2025-05-30 14:15:00 | 672.80 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-06-05 13:30:00 | 655.00 | 2025-06-10 10:15:00 | 657.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-05 15:00:00 | 654.15 | 2025-06-10 10:15:00 | 657.25 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-06-09 10:45:00 | 653.90 | 2025-06-10 10:15:00 | 657.25 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-06-18 09:30:00 | 633.50 | 2025-06-23 09:15:00 | 643.80 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-18 10:15:00 | 633.00 | 2025-06-23 09:15:00 | 643.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-06-18 11:45:00 | 632.35 | 2025-06-23 09:15:00 | 643.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-06-27 10:00:00 | 674.45 | 2025-07-04 11:15:00 | 741.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-14 13:45:00 | 719.65 | 2025-07-15 09:15:00 | 727.15 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-14 15:15:00 | 719.80 | 2025-07-15 09:15:00 | 727.15 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-18 11:15:00 | 759.50 | 2025-07-24 11:15:00 | 762.40 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-07-18 13:45:00 | 771.20 | 2025-07-24 11:15:00 | 762.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-07-31 10:00:00 | 688.30 | 2025-08-01 15:15:00 | 653.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 10:00:00 | 688.30 | 2025-08-06 12:15:00 | 640.95 | STOP_HIT | 0.50 | 6.88% |
| BUY | retest2 | 2025-08-22 13:00:00 | 662.15 | 2025-08-26 09:15:00 | 655.55 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-08-25 10:15:00 | 663.80 | 2025-08-26 09:15:00 | 655.55 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-25 11:15:00 | 662.65 | 2025-08-26 09:15:00 | 655.55 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-04 14:30:00 | 698.95 | 2025-09-04 15:15:00 | 690.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-15 13:15:00 | 753.40 | 2025-09-17 14:15:00 | 739.40 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-09-15 15:15:00 | 753.90 | 2025-09-17 14:15:00 | 739.40 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-09-16 10:30:00 | 755.60 | 2025-09-17 14:15:00 | 739.40 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-09-16 14:15:00 | 754.55 | 2025-09-17 14:15:00 | 739.40 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-09-19 14:15:00 | 743.00 | 2025-09-22 09:15:00 | 754.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-24 10:15:00 | 759.50 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-09-24 12:45:00 | 756.70 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-09-25 09:15:00 | 759.75 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-26 09:30:00 | 757.85 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-09-29 09:15:00 | 766.30 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-09-30 09:30:00 | 765.10 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-10-01 12:00:00 | 750.95 | 2025-10-03 11:15:00 | 765.60 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-08 09:15:00 | 811.00 | 2025-10-13 09:15:00 | 785.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-10-08 10:15:00 | 811.75 | 2025-10-13 09:15:00 | 785.00 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2025-10-09 09:15:00 | 809.95 | 2025-10-13 09:15:00 | 785.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-10-16 11:45:00 | 740.90 | 2025-10-20 12:15:00 | 777.10 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2025-10-16 12:30:00 | 743.15 | 2025-10-20 12:15:00 | 777.10 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-10-16 13:30:00 | 741.85 | 2025-10-20 12:15:00 | 777.10 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2025-10-28 13:45:00 | 835.35 | 2025-10-31 09:15:00 | 918.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 09:30:00 | 846.80 | 2025-10-31 10:15:00 | 931.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-26 09:15:00 | 976.85 | 2025-11-26 12:15:00 | 928.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 09:15:00 | 976.85 | 2025-11-27 11:15:00 | 938.00 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2025-12-17 13:30:00 | 910.10 | 2025-12-23 13:15:00 | 864.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-17 15:15:00 | 912.50 | 2025-12-23 13:15:00 | 866.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-17 13:30:00 | 910.10 | 2025-12-29 12:15:00 | 819.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-17 15:15:00 | 912.50 | 2025-12-29 12:15:00 | 821.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 12:15:00 | 808.25 | 2026-01-07 14:15:00 | 817.80 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-08 09:15:00 | 808.50 | 2026-01-12 09:15:00 | 768.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 808.50 | 2026-01-12 12:15:00 | 794.05 | STOP_HIT | 0.50 | 1.79% |
| BUY | retest2 | 2026-01-28 09:15:00 | 861.00 | 2026-02-01 12:15:00 | 839.90 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-28 10:30:00 | 850.20 | 2026-02-01 12:15:00 | 839.90 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2026-02-06 09:15:00 | 903.70 | 2026-02-12 10:15:00 | 918.65 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2026-02-17 10:15:00 | 879.55 | 2026-02-18 09:15:00 | 910.35 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2026-02-20 09:30:00 | 903.30 | 2026-02-20 10:15:00 | 894.45 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-02-20 10:15:00 | 897.65 | 2026-02-20 10:15:00 | 894.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-02-23 14:00:00 | 885.95 | 2026-02-24 09:15:00 | 900.50 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-23 14:30:00 | 884.15 | 2026-02-24 09:15:00 | 900.50 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-02-26 10:30:00 | 921.80 | 2026-03-05 09:15:00 | 1013.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-27 09:30:00 | 938.00 | 2026-03-05 09:15:00 | 1031.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 09:45:00 | 924.15 | 2026-03-05 09:15:00 | 1016.57 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 946.80 | 2026-03-16 09:15:00 | 899.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 946.80 | 2026-03-16 09:15:00 | 927.95 | STOP_HIT | 0.50 | 1.99% |
| BUY | retest2 | 2026-03-20 09:30:00 | 1076.90 | 2026-03-23 14:15:00 | 1007.35 | STOP_HIT | 1.00 | -6.46% |
| BUY | retest2 | 2026-03-20 10:45:00 | 1065.10 | 2026-03-23 14:15:00 | 1007.35 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2026-03-25 11:30:00 | 1019.70 | 2026-03-25 13:15:00 | 1020.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-03-25 12:15:00 | 1019.40 | 2026-03-25 13:15:00 | 1020.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2026-03-30 15:00:00 | 965.80 | 2026-04-01 09:15:00 | 993.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-04-02 11:30:00 | 1013.00 | 2026-04-06 13:15:00 | 991.40 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-04-06 09:30:00 | 1013.25 | 2026-04-06 13:15:00 | 991.40 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest1 | 2026-04-13 09:15:00 | 955.45 | 2026-04-15 10:15:00 | 966.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-04-21 09:15:00 | 1019.90 | 2026-04-24 15:15:00 | 1037.50 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2026-04-24 13:30:00 | 1019.30 | 2026-04-24 15:15:00 | 1037.50 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2026-04-24 14:30:00 | 1016.10 | 2026-04-24 15:15:00 | 1037.50 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest1 | 2026-04-30 12:30:00 | 1124.05 | 2026-05-05 09:15:00 | 1094.90 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest1 | 2026-04-30 14:30:00 | 1130.45 | 2026-05-05 09:15:00 | 1094.90 | STOP_HIT | 1.00 | -3.14% |
