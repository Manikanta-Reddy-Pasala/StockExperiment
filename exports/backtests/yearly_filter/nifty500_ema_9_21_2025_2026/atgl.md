# Adani Total Gas Ltd. (ATGL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 632.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 64 |
| ALERT1 | 43 |
| ALERT2 | 42 |
| ALERT2_SKIP | 19 |
| ALERT3 | 108 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 66 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 44
- **Target hits / Stop hits / Partials:** 5 / 63 / 11
- **Avg / median % per leg:** 1.43% / -0.51%
- **Sum % (uncompounded):** 113.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 10 | 47.6% | 5 | 14 | 2 | 2.82% | 59.3% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.38% | 17.5% |
| BUY @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 5 | 12 | 0 | 2.45% | 41.7% |
| SELL (all) | 58 | 25 | 43.1% | 0 | 49 | 9 | 0.93% | 54.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 58 | 25 | 43.1% | 0 | 49 | 9 | 0.93% | 54.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 4.38% | 17.5% |
| retest2 (combined) | 75 | 31 | 41.3% | 5 | 61 | 9 | 1.28% | 95.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 628.80 | 618.35 | 617.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 631.75 | 622.86 | 619.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 652.00 | 652.25 | 645.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 652.00 | 652.25 | 645.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 672.55 | 675.64 | 670.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 671.00 | 675.64 | 670.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 669.55 | 674.43 | 670.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 669.55 | 674.43 | 670.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 669.00 | 673.34 | 670.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 669.00 | 673.34 | 670.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 667.10 | 672.09 | 670.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 666.20 | 672.09 | 670.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 659.85 | 667.98 | 668.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 655.50 | 660.58 | 663.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 660.25 | 660.01 | 662.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 660.25 | 660.01 | 662.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 660.25 | 660.01 | 662.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 665.90 | 660.01 | 662.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 661.80 | 660.37 | 662.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 661.55 | 660.37 | 662.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 662.65 | 660.83 | 662.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 662.65 | 660.83 | 662.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 664.30 | 661.52 | 662.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 663.75 | 661.52 | 662.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 658.55 | 661.00 | 662.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:30:00 | 662.10 | 661.00 | 662.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 668.05 | 661.82 | 662.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 670.05 | 661.82 | 662.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 673.85 | 664.23 | 663.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 679.00 | 671.01 | 667.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 675.50 | 676.68 | 673.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:45:00 | 676.00 | 676.68 | 673.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 672.00 | 675.33 | 673.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 669.35 | 675.33 | 673.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 672.00 | 674.66 | 673.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 671.65 | 674.66 | 673.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 673.20 | 674.37 | 673.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:45:00 | 671.55 | 674.37 | 673.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 671.50 | 673.80 | 673.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:45:00 | 672.30 | 673.80 | 673.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 681.35 | 675.31 | 673.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 671.30 | 675.31 | 673.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 679.40 | 676.09 | 674.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 684.00 | 676.09 | 674.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 682.35 | 677.29 | 675.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 682.00 | 678.03 | 676.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 681.75 | 678.03 | 676.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 685.35 | 683.35 | 680.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:30:00 | 686.70 | 684.42 | 680.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:00:00 | 688.70 | 684.42 | 680.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 691.10 | 687.01 | 683.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 15:15:00 | 678.50 | 682.04 | 682.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 678.50 | 682.04 | 682.18 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 688.55 | 682.91 | 682.31 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 679.75 | 682.05 | 682.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 11:15:00 | 677.35 | 681.11 | 681.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 15:15:00 | 681.50 | 680.83 | 681.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 15:15:00 | 681.50 | 680.83 | 681.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 681.50 | 680.83 | 681.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 685.80 | 680.83 | 681.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 685.90 | 681.84 | 681.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 685.90 | 681.84 | 681.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 686.80 | 682.83 | 682.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 697.30 | 689.67 | 686.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 707.10 | 707.39 | 699.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 11:00:00 | 707.10 | 707.39 | 699.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 694.95 | 704.19 | 699.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 695.65 | 704.19 | 699.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 698.00 | 702.95 | 699.74 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 685.05 | 696.77 | 697.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 683.95 | 694.21 | 696.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 660.25 | 659.67 | 668.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 660.25 | 659.67 | 668.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 659.70 | 652.74 | 658.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 659.70 | 652.74 | 658.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 659.00 | 653.99 | 658.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 652.60 | 654.87 | 658.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 619.97 | 625.18 | 634.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 624.35 | 624.35 | 632.53 | SL hit (close>ema200) qty=0.50 sl=624.35 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 636.55 | 632.37 | 632.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 644.00 | 639.09 | 636.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 672.75 | 673.30 | 660.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 672.75 | 673.30 | 660.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 670.95 | 674.52 | 669.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 670.95 | 674.52 | 669.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 666.55 | 672.93 | 668.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 666.55 | 672.93 | 668.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 666.80 | 671.70 | 668.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 665.75 | 671.70 | 668.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 664.50 | 667.94 | 667.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 662.60 | 667.94 | 667.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 665.00 | 667.58 | 667.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 663.70 | 666.80 | 667.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 667.50 | 666.12 | 666.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 667.50 | 666.12 | 666.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 667.50 | 666.12 | 666.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 667.50 | 666.12 | 666.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 665.50 | 665.99 | 666.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:15:00 | 664.00 | 665.99 | 666.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 663.85 | 665.56 | 666.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 663.65 | 665.56 | 665.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 654.85 | 649.02 | 648.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 654.85 | 649.02 | 648.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 14:15:00 | 659.55 | 652.71 | 650.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 657.45 | 659.81 | 656.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 657.45 | 659.81 | 656.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 656.40 | 659.12 | 656.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 654.25 | 659.12 | 656.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 655.65 | 658.43 | 656.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:00:00 | 655.65 | 658.43 | 656.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 655.95 | 657.93 | 656.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 657.75 | 656.52 | 656.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 657.65 | 657.40 | 656.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:00:00 | 657.65 | 657.68 | 656.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:30:00 | 659.00 | 657.68 | 656.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 658.55 | 657.86 | 657.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 654.00 | 657.08 | 656.82 | SL hit (close<static) qty=1.00 sl=654.80 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 651.30 | 655.93 | 656.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 648.00 | 652.62 | 654.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 651.65 | 651.53 | 653.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 651.65 | 651.53 | 653.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 651.65 | 651.53 | 653.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 646.95 | 649.83 | 651.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:30:00 | 646.95 | 649.33 | 651.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 646.00 | 649.33 | 651.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:15:00 | 614.60 | 623.12 | 626.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:15:00 | 614.60 | 623.12 | 626.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 613.70 | 617.53 | 623.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 604.50 | 603.31 | 608.72 | SL hit (close>ema200) qty=0.50 sl=603.31 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 596.70 | 592.45 | 591.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 605.80 | 595.69 | 593.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 607.70 | 607.96 | 602.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 11:00:00 | 607.70 | 607.96 | 602.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 607.35 | 609.04 | 607.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 607.60 | 609.04 | 607.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 608.70 | 608.97 | 607.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 611.70 | 608.97 | 607.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 618.00 | 624.34 | 624.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 618.00 | 624.34 | 624.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 616.25 | 620.26 | 622.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 624.90 | 620.14 | 621.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 624.90 | 620.14 | 621.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 624.90 | 620.14 | 621.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 624.05 | 620.14 | 621.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 625.60 | 621.23 | 621.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 627.10 | 621.23 | 621.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 626.90 | 622.37 | 622.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 12:15:00 | 631.30 | 624.15 | 623.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 15:15:00 | 633.00 | 635.00 | 631.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:15:00 | 617.05 | 635.00 | 631.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 609.25 | 629.85 | 629.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 608.20 | 629.85 | 629.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 619.95 | 627.87 | 628.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 593.20 | 614.75 | 621.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 596.30 | 595.30 | 603.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:45:00 | 596.25 | 595.30 | 603.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 598.50 | 596.11 | 599.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 597.40 | 597.23 | 599.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:30:00 | 597.80 | 596.95 | 598.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:30:00 | 597.80 | 597.04 | 598.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 597.65 | 597.21 | 598.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 598.25 | 597.42 | 598.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 598.00 | 597.42 | 598.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 596.35 | 597.21 | 598.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 595.00 | 597.17 | 597.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 599.95 | 594.10 | 595.11 | SL hit (close>static) qty=1.00 sl=598.45 alert=retest2 |

### Cycle 17 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 600.85 | 595.45 | 595.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 609.40 | 599.99 | 597.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 606.00 | 606.72 | 603.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 11:00:00 | 606.00 | 606.72 | 603.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 605.40 | 605.78 | 604.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 611.65 | 605.90 | 605.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 609.80 | 607.69 | 606.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 667.40 | 607.14 | 606.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-19 09:15:00 | 672.82 | 617.98 | 611.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 678.20 | 702.18 | 705.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 666.65 | 695.08 | 701.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 630.15 | 628.34 | 640.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 637.85 | 628.34 | 640.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 642.15 | 632.16 | 637.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 642.15 | 632.16 | 637.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 641.20 | 633.96 | 637.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:30:00 | 644.05 | 633.96 | 637.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 636.25 | 635.94 | 637.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 635.90 | 635.94 | 637.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 634.65 | 635.69 | 637.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 633.90 | 635.69 | 637.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:30:00 | 633.60 | 635.35 | 636.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 634.00 | 634.08 | 635.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 633.80 | 634.08 | 635.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 641.40 | 635.50 | 635.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 641.40 | 635.50 | 635.94 | SL hit (close>static) qty=1.00 sl=639.05 alert=retest2 |

### Cycle 19 — BUY (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 11:15:00 | 637.20 | 636.41 | 636.31 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 629.95 | 635.16 | 635.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 629.20 | 633.97 | 635.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 629.85 | 629.20 | 630.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 629.85 | 629.20 | 630.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 629.85 | 629.20 | 630.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 631.65 | 629.20 | 630.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 628.20 | 629.00 | 630.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 623.30 | 628.67 | 629.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 626.65 | 626.05 | 627.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 625.55 | 622.29 | 621.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 625.55 | 622.29 | 621.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 636.95 | 625.22 | 623.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 624.90 | 625.15 | 623.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:00:00 | 624.90 | 625.15 | 623.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 622.00 | 624.52 | 623.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 623.50 | 624.52 | 623.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 622.95 | 624.21 | 623.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 622.50 | 624.21 | 623.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 620.05 | 623.38 | 622.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 620.05 | 623.38 | 622.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 619.55 | 622.61 | 622.66 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 625.05 | 622.22 | 622.13 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 620.05 | 622.61 | 622.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 12:15:00 | 619.00 | 620.64 | 621.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 620.60 | 620.06 | 620.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 620.60 | 620.06 | 620.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 620.60 | 620.06 | 620.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 622.20 | 620.06 | 620.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 618.95 | 619.84 | 620.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 620.20 | 619.84 | 620.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 621.40 | 620.15 | 620.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 621.40 | 620.15 | 620.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 620.75 | 620.27 | 620.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 619.35 | 620.18 | 620.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 619.55 | 620.18 | 620.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 616.50 | 620.74 | 620.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 626.40 | 621.87 | 621.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 626.40 | 621.87 | 621.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 658.10 | 629.12 | 624.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 634.30 | 637.95 | 631.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 634.30 | 637.95 | 631.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 638.00 | 637.12 | 631.88 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 631.20 | 633.48 | 633.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 14:15:00 | 626.15 | 629.79 | 631.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 629.50 | 628.78 | 630.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 629.50 | 628.78 | 630.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 629.50 | 628.78 | 630.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 628.45 | 628.78 | 630.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 630.15 | 628.28 | 629.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 630.15 | 628.28 | 629.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 628.15 | 628.26 | 629.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 621.50 | 628.26 | 629.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 629.55 | 615.38 | 614.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 629.55 | 615.38 | 614.65 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 618.30 | 622.00 | 622.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 613.45 | 620.29 | 621.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 615.80 | 612.58 | 614.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 615.80 | 612.58 | 614.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 615.80 | 612.58 | 614.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 608.05 | 613.63 | 614.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 606.20 | 605.66 | 605.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 606.20 | 605.66 | 605.60 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 604.05 | 605.38 | 605.48 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 611.25 | 606.34 | 605.71 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 601.70 | 605.69 | 606.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 601.30 | 604.23 | 605.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 594.60 | 593.49 | 595.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 594.60 | 593.49 | 595.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 594.60 | 593.49 | 595.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:30:00 | 594.45 | 593.49 | 595.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 582.30 | 591.15 | 593.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 579.85 | 587.15 | 591.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 579.25 | 587.15 | 591.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 594.60 | 587.06 | 586.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 594.60 | 587.06 | 586.62 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 585.75 | 586.98 | 587.01 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 587.80 | 587.15 | 587.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 591.50 | 588.20 | 587.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 592.10 | 592.44 | 591.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 592.10 | 592.44 | 591.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 587.95 | 591.63 | 590.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 588.50 | 591.63 | 590.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 588.80 | 591.07 | 590.76 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 587.05 | 590.26 | 590.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 586.00 | 588.13 | 589.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 582.25 | 580.89 | 583.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 582.25 | 580.89 | 583.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 582.45 | 581.21 | 583.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 583.60 | 581.21 | 583.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 578.00 | 580.56 | 582.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 576.00 | 578.86 | 581.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 15:00:00 | 572.05 | 577.30 | 580.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 576.85 | 569.61 | 569.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 576.85 | 569.61 | 569.14 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 566.60 | 569.37 | 569.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 566.05 | 568.71 | 569.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 565.50 | 564.96 | 566.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 565.50 | 564.96 | 566.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 563.90 | 564.75 | 566.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 565.75 | 564.75 | 566.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 566.45 | 565.13 | 566.14 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 567.60 | 566.72 | 566.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 613.60 | 576.15 | 571.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 591.00 | 591.59 | 582.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 09:15:00 | 589.05 | 591.59 | 582.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 578.80 | 586.91 | 584.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 578.00 | 586.91 | 584.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 578.80 | 585.29 | 584.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 578.80 | 585.29 | 584.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 579.15 | 582.95 | 583.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 577.20 | 581.80 | 582.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 571.30 | 571.15 | 574.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 571.30 | 571.15 | 574.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 572.95 | 571.66 | 573.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:45:00 | 574.65 | 571.66 | 573.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 555.75 | 554.51 | 558.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 557.40 | 554.51 | 558.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 559.00 | 555.41 | 558.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 558.40 | 555.41 | 558.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 557.55 | 555.84 | 558.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 554.85 | 556.13 | 558.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 554.60 | 555.85 | 557.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 554.35 | 555.47 | 557.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 554.70 | 555.54 | 556.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 554.70 | 555.37 | 556.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 555.15 | 555.37 | 556.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 549.90 | 554.12 | 555.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 547.00 | 550.28 | 553.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 527.11 | 536.50 | 541.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 526.87 | 536.50 | 541.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 526.63 | 536.50 | 541.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 526.97 | 536.50 | 541.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 539.05 | 532.27 | 536.14 | SL hit (close>ema200) qty=0.50 sl=532.27 alert=retest2 |

### Cycle 41 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 546.15 | 538.91 | 538.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 551.00 | 541.33 | 539.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 538.00 | 540.66 | 539.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 538.00 | 540.66 | 539.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 538.00 | 540.66 | 539.30 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 524.00 | 536.09 | 537.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 517.95 | 532.46 | 535.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 535.35 | 528.89 | 532.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 535.35 | 528.89 | 532.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 535.35 | 528.89 | 532.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 535.35 | 528.89 | 532.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 526.00 | 528.31 | 532.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:00:00 | 524.70 | 527.96 | 531.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 524.35 | 526.22 | 529.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 524.25 | 525.02 | 528.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 524.40 | 527.13 | 528.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 532.45 | 526.94 | 527.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:45:00 | 530.80 | 526.94 | 527.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 528.30 | 527.21 | 527.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:15:00 | 525.05 | 527.21 | 527.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 12:15:00 | 527.00 | 527.26 | 527.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 14:15:00 | 527.00 | 527.04 | 527.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 15:15:00 | 529.90 | 527.59 | 527.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 529.90 | 527.59 | 527.52 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 525.50 | 527.32 | 527.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 522.95 | 526.45 | 527.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 518.35 | 517.31 | 520.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 518.35 | 517.31 | 520.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 45 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 546.35 | 523.53 | 522.99 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 543.00 | 545.16 | 545.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 540.20 | 544.17 | 544.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 542.25 | 542.20 | 543.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 15:00:00 | 542.25 | 542.20 | 543.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 536.50 | 541.07 | 542.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:15:00 | 535.00 | 538.34 | 540.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 533.20 | 529.91 | 529.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 533.20 | 529.91 | 529.80 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 528.50 | 530.52 | 530.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 526.80 | 529.78 | 530.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 525.40 | 522.69 | 524.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 525.40 | 522.69 | 524.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 525.40 | 522.69 | 524.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 527.50 | 522.69 | 524.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 525.00 | 523.15 | 524.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 525.60 | 523.15 | 524.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 523.60 | 523.24 | 524.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 520.35 | 524.15 | 524.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 494.33 | 509.73 | 513.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 482.10 | 482.02 | 487.79 | SL hit (close>ema200) qty=0.50 sl=482.02 alert=retest2 |

### Cycle 49 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 537.45 | 485.36 | 480.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 551.15 | 507.28 | 491.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 13:15:00 | 576.45 | 601.60 | 579.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 13:15:00 | 576.45 | 601.60 | 579.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 576.45 | 601.60 | 579.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:00:00 | 576.45 | 601.60 | 579.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 574.00 | 596.08 | 579.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 574.00 | 596.08 | 579.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 572.50 | 591.36 | 578.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 550.30 | 591.36 | 578.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 532.50 | 572.62 | 571.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:00:00 | 532.50 | 572.62 | 571.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 527.00 | 563.50 | 567.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 519.10 | 554.62 | 563.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 09:15:00 | 553.50 | 523.99 | 527.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 553.50 | 523.99 | 527.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 553.50 | 523.99 | 527.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 568.30 | 523.99 | 527.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 10:15:00 | 564.55 | 532.10 | 531.10 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 520.40 | 542.23 | 543.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 515.50 | 533.37 | 539.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 519.45 | 519.13 | 527.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 518.55 | 519.13 | 527.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 526.05 | 521.63 | 526.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 15:00:00 | 521.50 | 523.24 | 525.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 538.80 | 526.13 | 526.34 | SL hit (close>static) qty=1.00 sl=531.40 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 10:15:00 | 531.45 | 527.20 | 526.80 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 515.10 | 525.43 | 526.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 513.10 | 520.44 | 523.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 517.05 | 516.08 | 520.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 517.05 | 516.08 | 520.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 517.05 | 516.08 | 520.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 512.05 | 519.86 | 520.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 10:15:00 | 513.10 | 518.85 | 520.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 521.60 | 519.96 | 519.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 521.60 | 519.96 | 519.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 523.40 | 520.65 | 520.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 521.30 | 522.22 | 521.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 10:15:00 | 521.30 | 522.22 | 521.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 521.30 | 522.22 | 521.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 521.30 | 522.22 | 521.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 521.20 | 522.01 | 521.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 521.20 | 522.01 | 521.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 520.75 | 521.76 | 521.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:45:00 | 520.40 | 521.76 | 521.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 518.90 | 521.19 | 521.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 518.30 | 521.19 | 521.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 519.00 | 520.75 | 520.85 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 531.00 | 522.84 | 521.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 566.15 | 531.50 | 525.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 548.00 | 549.10 | 539.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 558.40 | 550.20 | 544.06 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:00:00 | 554.70 | 551.10 | 545.03 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:15:00 | 582.44 | 557.39 | 548.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 580.50 | 572.63 | 560.68 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:15:00 | 586.32 | 575.53 | 563.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 587.10 | 575.53 | 563.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 12:15:00 | 577.50 | 577.70 | 566.34 | SL hit (close<ema200) qty=0.50 sl=577.70 alert=retest1 |

### Cycle 58 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 612.35 | 621.60 | 621.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 610.10 | 619.30 | 620.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 626.00 | 618.09 | 619.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 626.00 | 618.09 | 619.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 626.00 | 618.09 | 619.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 627.50 | 618.09 | 619.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 625.00 | 619.48 | 620.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 627.70 | 619.48 | 620.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 633.15 | 622.21 | 621.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 643.00 | 626.37 | 623.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 657.30 | 657.42 | 649.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 646.65 | 657.42 | 649.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 629.15 | 651.77 | 647.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 629.15 | 651.77 | 647.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 628.95 | 647.20 | 646.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 628.65 | 647.20 | 646.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 632.20 | 644.20 | 644.81 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 648.00 | 643.48 | 642.96 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 638.80 | 642.35 | 642.78 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 648.00 | 643.73 | 643.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 650.20 | 645.78 | 644.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 646.90 | 647.14 | 645.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 11:15:00 | 646.90 | 647.14 | 645.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 646.90 | 647.14 | 645.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 645.60 | 647.14 | 645.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 646.35 | 646.94 | 645.82 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 641.60 | 645.18 | 645.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 635.65 | 641.10 | 643.16 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-29 15:15:00 | 684.00 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-05-30 11:15:00 | 682.35 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-05-30 11:45:00 | 682.00 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-05-30 12:15:00 | 681.75 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-06-02 12:30:00 | 686.70 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-02 13:00:00 | 688.70 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-03 09:45:00 | 691.10 | 2025-06-03 15:15:00 | 678.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-06-18 12:15:00 | 652.60 | 2025-06-20 14:15:00 | 619.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 12:15:00 | 652.60 | 2025-06-23 09:15:00 | 624.35 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2025-07-03 11:15:00 | 664.00 | 2025-07-16 09:15:00 | 654.85 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-07-03 12:00:00 | 663.85 | 2025-07-16 09:15:00 | 654.85 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2025-07-04 11:30:00 | 663.65 | 2025-07-16 09:15:00 | 654.85 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2025-07-21 10:00:00 | 657.75 | 2025-07-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-21 11:45:00 | 657.65 | 2025-07-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-21 14:00:00 | 657.65 | 2025-07-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-21 14:30:00 | 659.00 | 2025-07-22 09:15:00 | 654.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-24 12:45:00 | 646.95 | 2025-07-31 11:15:00 | 614.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 13:30:00 | 646.95 | 2025-07-31 11:15:00 | 614.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 14:15:00 | 646.00 | 2025-07-31 14:15:00 | 613.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 12:45:00 | 646.95 | 2025-08-04 13:15:00 | 604.50 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2025-07-24 13:30:00 | 646.95 | 2025-08-04 13:15:00 | 604.50 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2025-07-24 14:15:00 | 646.00 | 2025-08-04 13:15:00 | 604.50 | STOP_HIT | 0.50 | 6.42% |
| BUY | retest2 | 2025-08-18 09:15:00 | 611.70 | 2025-08-21 14:15:00 | 618.00 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2025-09-02 13:15:00 | 597.40 | 2025-09-08 09:15:00 | 599.95 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-09-03 09:30:00 | 597.80 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-09-03 10:30:00 | 597.80 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-09-03 11:45:00 | 597.65 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-04 15:15:00 | 595.00 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-08 13:30:00 | 595.20 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-09-08 14:45:00 | 593.50 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-09 09:30:00 | 594.80 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-09-09 13:45:00 | 592.30 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-09-09 14:15:00 | 593.00 | 2025-09-10 09:15:00 | 600.85 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-09-17 09:15:00 | 611.65 | 2025-09-19 09:15:00 | 672.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-18 11:00:00 | 609.80 | 2025-09-19 09:15:00 | 670.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-19 09:15:00 | 667.40 | 2025-09-22 10:15:00 | 734.14 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-03 12:15:00 | 633.90 | 2025-10-07 09:15:00 | 641.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-06 10:30:00 | 633.60 | 2025-10-07 09:15:00 | 641.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-06 14:30:00 | 634.00 | 2025-10-07 09:15:00 | 641.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-10-06 15:15:00 | 633.80 | 2025-10-07 09:15:00 | 641.40 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-10-13 09:15:00 | 623.30 | 2025-10-17 09:15:00 | 625.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-14 09:45:00 | 626.65 | 2025-10-17 09:15:00 | 625.55 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-10-28 13:30:00 | 619.35 | 2025-10-29 09:15:00 | 626.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-28 14:15:00 | 619.55 | 2025-10-29 09:15:00 | 626.40 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-29 09:15:00 | 616.50 | 2025-10-29 09:15:00 | 626.40 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-11-06 09:15:00 | 621.50 | 2025-11-12 09:15:00 | 629.55 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-11-20 15:15:00 | 608.05 | 2025-11-26 14:15:00 | 606.20 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-12-08 11:30:00 | 579.85 | 2025-12-10 09:15:00 | 594.60 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-12-08 12:00:00 | 579.25 | 2025-12-10 09:15:00 | 594.60 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-12-19 12:30:00 | 576.00 | 2025-12-26 11:15:00 | 576.85 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-12-19 15:00:00 | 572.05 | 2025-12-26 11:15:00 | 576.85 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-01-13 12:00:00 | 554.85 | 2026-01-21 09:15:00 | 527.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 554.60 | 2026-01-21 09:15:00 | 526.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 554.35 | 2026-01-21 09:15:00 | 526.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:45:00 | 554.70 | 2026-01-21 09:15:00 | 526.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 554.85 | 2026-01-22 09:15:00 | 539.05 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-01-13 14:15:00 | 554.60 | 2026-01-22 09:15:00 | 539.05 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-01-14 10:00:00 | 554.35 | 2026-01-22 09:15:00 | 539.05 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2026-01-14 13:45:00 | 554.70 | 2026-01-22 09:15:00 | 539.05 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2026-01-16 15:00:00 | 547.00 | 2026-01-22 14:15:00 | 546.15 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2026-01-27 13:00:00 | 524.70 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-01-28 09:15:00 | 524.35 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-28 12:15:00 | 524.25 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-29 10:15:00 | 524.40 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-29 15:15:00 | 525.05 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-30 12:15:00 | 527.00 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-01-30 14:15:00 | 527.00 | 2026-01-30 15:15:00 | 529.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-02-12 15:15:00 | 535.00 | 2026-02-17 12:15:00 | 533.20 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2026-02-24 09:15:00 | 520.35 | 2026-03-02 09:15:00 | 494.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 520.35 | 2026-03-05 14:15:00 | 482.10 | STOP_HIT | 0.50 | 7.35% |
| SELL | retest2 | 2026-03-25 15:00:00 | 521.50 | 2026-03-27 09:15:00 | 538.80 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2026-04-02 09:15:00 | 512.05 | 2026-04-06 11:15:00 | 521.60 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-04-02 10:15:00 | 513.10 | 2026-04-06 11:15:00 | 521.60 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2026-04-10 09:15:00 | 558.40 | 2026-04-10 11:15:00 | 582.44 | PARTIAL | 0.50 | 4.30% |
| BUY | retest1 | 2026-04-10 10:00:00 | 554.70 | 2026-04-13 10:15:00 | 586.32 | PARTIAL | 0.50 | 5.70% |
| BUY | retest1 | 2026-04-10 09:15:00 | 558.40 | 2026-04-13 12:15:00 | 577.50 | STOP_HIT | 0.50 | 3.42% |
| BUY | retest1 | 2026-04-10 10:00:00 | 554.70 | 2026-04-13 12:15:00 | 577.50 | STOP_HIT | 0.50 | 4.11% |
| BUY | retest2 | 2026-04-13 11:00:00 | 587.10 | 2026-04-17 14:15:00 | 645.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 10:00:00 | 587.15 | 2026-04-17 14:15:00 | 645.87 | TARGET_HIT | 1.00 | 10.00% |
