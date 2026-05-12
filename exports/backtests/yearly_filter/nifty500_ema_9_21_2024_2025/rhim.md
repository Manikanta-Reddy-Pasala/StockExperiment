# RHI MAGNESITA INDIA LTD. (RHIM)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3708 bars)
- **Last close:** 409.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 97 |
| ALERT2 | 96 |
| ALERT2_SKIP | 48 |
| ALERT3 | 247 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 127 |
| PARTIAL | 13 |
| TARGET_HIT | 4 |
| STOP_HIT | 124 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 107
- **Target hits / Stop hits / Partials:** 4 / 124 / 13
- **Avg / median % per leg:** -0.34% / -1.30%
- **Sum % (uncompounded):** -47.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 5 | 9.8% | 2 | 49 | 0 | -1.06% | -54.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 51 | 5 | 9.8% | 2 | 49 | 0 | -1.06% | -54.1% |
| SELL (all) | 90 | 29 | 32.2% | 2 | 75 | 13 | 0.07% | 6.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.52% | -2.5% |
| SELL @ 3rd Alert (retest2) | 89 | 29 | 32.6% | 2 | 74 | 13 | 0.10% | 9.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.52% | -2.5% |
| retest2 (combined) | 140 | 34 | 24.3% | 4 | 123 | 13 | -0.32% | -45.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 620.00 | 612.16 | 611.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 624.30 | 614.59 | 612.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 640.15 | 640.72 | 634.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 11:00:00 | 640.15 | 640.72 | 634.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 663.75 | 666.56 | 662.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 665.20 | 666.56 | 662.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 668.00 | 666.85 | 662.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 658.35 | 666.85 | 662.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 670.95 | 668.77 | 665.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:00:00 | 675.50 | 670.12 | 666.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 14:45:00 | 672.05 | 673.34 | 669.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:15:00 | 674.00 | 672.43 | 669.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 12:15:00 | 675.10 | 672.32 | 670.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 672.25 | 672.31 | 670.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 660.35 | 669.87 | 669.82 | SL hit (close<static) qty=1.00 sl=660.80 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 10:15:00 | 660.00 | 667.90 | 668.93 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 09:15:00 | 679.15 | 668.21 | 667.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 10:15:00 | 709.00 | 676.37 | 670.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 12:15:00 | 711.90 | 712.29 | 698.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-31 13:00:00 | 711.90 | 712.29 | 698.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 730.00 | 715.21 | 703.12 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 669.00 | 700.61 | 703.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 659.65 | 678.78 | 689.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 673.65 | 667.10 | 676.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 673.65 | 667.10 | 676.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 673.65 | 667.10 | 676.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 676.05 | 667.10 | 676.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 679.10 | 669.50 | 677.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 679.10 | 669.50 | 677.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 671.40 | 669.88 | 676.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:00:00 | 669.10 | 669.72 | 675.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:15:00 | 668.65 | 670.28 | 675.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 15:15:00 | 669.10 | 670.53 | 675.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 10:15:00 | 679.25 | 673.43 | 675.48 | SL hit (close>static) qty=1.00 sl=679.10 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 10:15:00 | 684.20 | 676.59 | 676.00 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 14:15:00 | 672.95 | 676.11 | 676.43 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 10:15:00 | 678.45 | 676.64 | 676.56 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 675.20 | 676.35 | 676.44 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 677.90 | 676.66 | 676.57 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 13:15:00 | 675.35 | 676.40 | 676.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 12:15:00 | 671.00 | 674.41 | 675.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 13:15:00 | 676.25 | 674.77 | 675.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 13:15:00 | 676.25 | 674.77 | 675.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 676.25 | 674.77 | 675.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:00:00 | 676.25 | 674.77 | 675.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 675.95 | 675.01 | 675.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:45:00 | 677.60 | 675.01 | 675.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 677.65 | 675.54 | 675.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 683.00 | 675.54 | 675.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 684.20 | 677.27 | 676.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 10:15:00 | 695.70 | 680.96 | 678.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 689.50 | 689.88 | 684.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 10:00:00 | 689.50 | 689.88 | 684.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 687.05 | 689.42 | 686.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:00:00 | 687.05 | 689.42 | 686.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 687.95 | 689.13 | 686.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:30:00 | 686.55 | 689.13 | 686.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 689.00 | 689.10 | 686.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 678.95 | 689.10 | 686.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 676.05 | 686.49 | 685.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 677.45 | 686.49 | 685.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 669.50 | 683.09 | 684.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 11:15:00 | 668.00 | 680.08 | 682.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 12:15:00 | 663.35 | 662.47 | 667.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 12:30:00 | 662.30 | 662.47 | 667.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 674.20 | 664.41 | 667.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 674.20 | 664.41 | 667.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 671.00 | 665.73 | 667.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 665.15 | 665.73 | 667.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 668.35 | 666.48 | 667.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 668.20 | 666.48 | 667.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 661.40 | 665.46 | 666.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:15:00 | 654.70 | 665.46 | 666.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 643.75 | 639.60 | 639.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 643.75 | 639.60 | 639.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 653.40 | 643.73 | 641.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 641.20 | 644.28 | 642.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 641.20 | 644.28 | 642.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 641.20 | 644.28 | 642.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 641.20 | 644.28 | 642.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 637.80 | 642.98 | 642.04 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 15:15:00 | 637.00 | 640.91 | 641.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 13:15:00 | 634.20 | 639.10 | 640.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 15:15:00 | 615.75 | 614.35 | 619.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 09:15:00 | 620.55 | 614.35 | 619.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 608.10 | 613.10 | 618.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:30:00 | 607.00 | 612.00 | 617.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 15:15:00 | 576.65 | 592.76 | 599.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-15 10:15:00 | 597.40 | 591.36 | 597.56 | SL hit (close>ema200) qty=0.50 sl=591.36 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 15:15:00 | 612.00 | 601.62 | 600.75 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 593.20 | 600.14 | 600.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 15:15:00 | 590.00 | 597.47 | 599.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 11:15:00 | 590.05 | 588.25 | 592.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 12:00:00 | 590.05 | 588.25 | 592.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 584.15 | 585.55 | 589.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 587.05 | 585.55 | 589.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 588.20 | 586.00 | 588.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 588.20 | 586.00 | 588.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 590.50 | 586.90 | 588.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 589.05 | 586.90 | 588.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 589.80 | 587.48 | 588.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 576.00 | 588.01 | 588.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 09:30:00 | 584.65 | 584.25 | 586.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:00:00 | 586.70 | 586.08 | 586.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:45:00 | 586.20 | 586.56 | 586.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 14:15:00 | 589.40 | 587.13 | 587.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 589.40 | 587.13 | 587.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 592.00 | 588.10 | 587.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 589.95 | 597.35 | 593.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 14:15:00 | 589.95 | 597.35 | 593.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 589.95 | 597.35 | 593.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 589.95 | 597.35 | 593.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 590.00 | 595.88 | 593.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 593.35 | 595.88 | 593.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 10:15:00 | 599.55 | 605.80 | 606.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 10:15:00 | 599.55 | 605.80 | 606.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 585.60 | 598.59 | 601.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 591.50 | 587.56 | 592.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 591.50 | 587.56 | 592.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 591.50 | 587.56 | 592.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 591.50 | 587.56 | 592.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 585.30 | 587.11 | 591.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:15:00 | 584.10 | 587.11 | 591.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:00:00 | 584.60 | 586.61 | 591.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:15:00 | 584.10 | 583.77 | 586.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 588.45 | 587.73 | 587.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 588.45 | 587.73 | 587.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 13:15:00 | 590.00 | 588.52 | 588.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 13:15:00 | 593.55 | 596.81 | 594.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 13:15:00 | 593.55 | 596.81 | 594.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 593.55 | 596.81 | 594.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 593.55 | 596.81 | 594.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 592.75 | 596.00 | 594.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 592.75 | 596.00 | 594.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 594.95 | 595.79 | 594.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 594.30 | 595.79 | 594.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 594.50 | 595.53 | 594.55 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 592.50 | 594.26 | 594.36 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 13:15:00 | 595.10 | 594.37 | 594.37 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 591.60 | 593.82 | 594.12 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 640.50 | 602.86 | 598.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 10:15:00 | 662.50 | 614.79 | 604.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 14:15:00 | 626.95 | 627.61 | 614.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-16 14:30:00 | 627.20 | 627.61 | 614.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 633.15 | 632.29 | 628.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 11:15:00 | 634.00 | 632.33 | 628.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 644.00 | 632.54 | 630.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 639.00 | 634.52 | 632.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:45:00 | 634.10 | 634.20 | 632.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 633.85 | 634.13 | 632.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:15:00 | 632.50 | 634.13 | 632.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 631.30 | 633.57 | 632.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 631.30 | 633.57 | 632.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 630.55 | 632.96 | 632.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:45:00 | 630.50 | 632.96 | 632.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-23 13:15:00 | 629.00 | 632.17 | 632.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 13:15:00 | 629.00 | 632.17 | 632.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 09:15:00 | 623.15 | 629.48 | 630.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 11:15:00 | 630.85 | 629.34 | 630.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 11:15:00 | 630.85 | 629.34 | 630.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 630.85 | 629.34 | 630.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:30:00 | 631.00 | 629.34 | 630.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 633.00 | 630.07 | 630.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 14:15:00 | 629.75 | 630.08 | 630.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 15:15:00 | 629.95 | 630.26 | 630.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:15:00 | 598.26 | 602.61 | 604.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:15:00 | 598.45 | 602.61 | 604.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 10:15:00 | 591.00 | 588.05 | 592.78 | SL hit (close>ema200) qty=0.50 sl=588.05 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 15:15:00 | 585.95 | 582.51 | 582.05 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 577.30 | 581.47 | 581.62 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 10:15:00 | 585.30 | 582.24 | 581.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 11:15:00 | 590.00 | 583.79 | 582.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 13:15:00 | 589.25 | 589.34 | 586.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:45:00 | 589.55 | 589.34 | 586.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 588.40 | 589.41 | 587.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 583.00 | 588.13 | 586.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 580.60 | 586.62 | 586.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 580.60 | 586.62 | 586.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 583.95 | 586.09 | 586.19 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 12:15:00 | 589.00 | 586.67 | 586.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 13:15:00 | 590.35 | 587.41 | 586.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 595.55 | 596.04 | 592.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 10:00:00 | 595.55 | 596.04 | 592.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 598.75 | 596.72 | 594.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 14:30:00 | 598.95 | 597.58 | 594.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 15:00:00 | 601.00 | 597.58 | 594.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 15:15:00 | 593.55 | 596.77 | 594.66 | SL hit (close<static) qty=1.00 sl=594.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 612.80 | 617.02 | 617.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 607.80 | 614.77 | 615.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 611.50 | 608.87 | 611.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 611.50 | 608.87 | 611.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 611.50 | 608.87 | 611.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 611.50 | 608.87 | 611.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 612.45 | 609.58 | 611.45 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 623.50 | 613.31 | 612.88 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 613.00 | 615.00 | 615.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 609.45 | 613.30 | 614.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 602.20 | 592.82 | 596.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 602.20 | 592.82 | 596.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 602.20 | 592.82 | 596.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:45:00 | 591.80 | 593.84 | 596.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:15:00 | 589.90 | 594.69 | 595.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:15:00 | 591.70 | 591.19 | 593.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 595.15 | 593.82 | 593.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 595.15 | 593.82 | 593.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 635.95 | 606.14 | 600.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 618.70 | 619.17 | 611.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 11:15:00 | 615.10 | 619.17 | 611.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 619.30 | 622.21 | 616.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 619.30 | 622.21 | 616.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 614.75 | 620.28 | 616.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:45:00 | 615.55 | 620.28 | 616.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 616.40 | 619.51 | 616.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 13:30:00 | 618.30 | 618.42 | 616.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 14:15:00 | 609.35 | 616.61 | 615.81 | SL hit (close<static) qty=1.00 sl=613.80 alert=retest2 |

### Cycle 34 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 609.80 | 615.25 | 615.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 10:15:00 | 605.95 | 612.04 | 613.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 15:15:00 | 614.20 | 609.50 | 611.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 15:15:00 | 614.20 | 609.50 | 611.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 614.20 | 609.50 | 611.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 604.90 | 609.50 | 611.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 574.65 | 587.20 | 596.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 584.50 | 582.95 | 590.61 | SL hit (close>ema200) qty=0.50 sl=582.95 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 574.55 | 572.22 | 572.19 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 571.00 | 572.06 | 572.12 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 583.30 | 574.33 | 573.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 10:15:00 | 588.95 | 577.26 | 574.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 580.40 | 589.25 | 583.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 580.40 | 589.25 | 583.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 580.40 | 589.25 | 583.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:15:00 | 577.80 | 589.25 | 583.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 577.65 | 586.93 | 583.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 578.30 | 586.93 | 583.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 582.30 | 584.10 | 582.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:30:00 | 581.90 | 584.10 | 582.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 585.00 | 584.28 | 583.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 578.35 | 584.28 | 583.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 580.85 | 583.60 | 582.94 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 14:15:00 | 582.10 | 582.58 | 582.61 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 583.00 | 582.67 | 582.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 586.50 | 583.43 | 583.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 573.45 | 589.12 | 588.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 573.45 | 589.12 | 588.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 573.45 | 589.12 | 588.76 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 573.00 | 585.90 | 587.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 562.70 | 575.23 | 580.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 11:15:00 | 521.85 | 521.83 | 527.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 11:30:00 | 522.95 | 521.83 | 527.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 510.85 | 508.35 | 511.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 511.05 | 508.35 | 511.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 512.55 | 509.19 | 511.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 517.60 | 509.19 | 511.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 518.90 | 511.13 | 512.57 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 518.05 | 513.95 | 513.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 11:15:00 | 522.20 | 518.07 | 516.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 523.30 | 527.65 | 525.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 523.30 | 527.65 | 525.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 523.30 | 527.65 | 525.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 524.25 | 527.65 | 525.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 522.45 | 526.61 | 524.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 522.45 | 526.61 | 524.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 518.60 | 525.01 | 524.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:30:00 | 519.40 | 525.01 | 524.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 517.50 | 523.51 | 523.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 516.20 | 522.05 | 522.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 518.75 | 516.20 | 519.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 518.75 | 516.20 | 519.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 518.75 | 516.20 | 519.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:45:00 | 519.50 | 516.20 | 519.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 522.90 | 517.54 | 519.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 522.90 | 517.54 | 519.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 522.00 | 518.43 | 519.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 519.55 | 518.43 | 519.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 522.30 | 519.20 | 519.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 522.30 | 519.20 | 519.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 529.50 | 521.26 | 520.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 533.10 | 524.53 | 522.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 553.00 | 556.31 | 547.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:45:00 | 553.80 | 556.31 | 547.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 552.15 | 554.77 | 548.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 549.00 | 554.77 | 548.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 553.30 | 553.46 | 549.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 550.00 | 553.46 | 549.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 550.90 | 552.95 | 549.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 550.40 | 552.95 | 549.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 551.40 | 552.64 | 549.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:30:00 | 552.90 | 552.64 | 549.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 553.70 | 552.85 | 550.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 14:45:00 | 560.80 | 554.22 | 551.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 558.45 | 554.38 | 551.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 559.95 | 554.70 | 552.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 11:15:00 | 564.15 | 558.08 | 556.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 565.70 | 559.61 | 556.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:00:00 | 571.05 | 564.65 | 560.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 14:15:00 | 557.55 | 564.69 | 564.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 557.55 | 564.69 | 564.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 548.00 | 560.25 | 562.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 557.50 | 556.81 | 559.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 557.50 | 556.81 | 559.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 561.90 | 557.83 | 559.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 560.60 | 557.83 | 559.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 562.45 | 558.75 | 560.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 562.50 | 558.75 | 560.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 555.50 | 558.10 | 559.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 553.70 | 557.22 | 559.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:45:00 | 553.95 | 555.49 | 557.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 567.00 | 553.35 | 554.32 | SL hit (close>static) qty=1.00 sl=562.45 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 570.45 | 556.77 | 555.79 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 549.90 | 555.35 | 555.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 534.45 | 550.24 | 553.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 510.25 | 504.74 | 508.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 510.25 | 504.74 | 508.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 510.25 | 504.74 | 508.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 510.50 | 504.74 | 508.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 510.95 | 505.98 | 508.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:45:00 | 510.50 | 505.98 | 508.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 510.90 | 508.77 | 509.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 510.70 | 508.77 | 509.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 511.80 | 509.38 | 509.31 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 09:15:00 | 505.90 | 508.68 | 509.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 11:15:00 | 500.50 | 506.00 | 507.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 499.00 | 495.54 | 499.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 12:15:00 | 499.00 | 495.54 | 499.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 499.00 | 495.54 | 499.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 500.00 | 495.54 | 499.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 502.25 | 496.88 | 500.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 502.25 | 496.88 | 500.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 502.10 | 497.92 | 500.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:15:00 | 502.95 | 497.92 | 500.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 499.05 | 499.44 | 500.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 15:15:00 | 496.10 | 498.98 | 499.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 11:15:00 | 504.55 | 500.41 | 500.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 504.55 | 500.41 | 500.28 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 495.00 | 500.04 | 500.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 492.50 | 495.49 | 496.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 492.90 | 490.97 | 493.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 492.90 | 490.97 | 493.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 492.90 | 490.97 | 493.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:15:00 | 493.95 | 490.97 | 493.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 495.10 | 491.79 | 493.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:00:00 | 495.10 | 491.79 | 493.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 493.35 | 492.11 | 493.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:30:00 | 491.00 | 492.03 | 493.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 492.10 | 492.46 | 493.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 505.65 | 495.04 | 494.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 505.65 | 495.04 | 494.28 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 488.55 | 494.50 | 495.30 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 492.70 | 486.68 | 486.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 492.80 | 489.19 | 487.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 487.55 | 488.92 | 488.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 487.55 | 488.92 | 488.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 487.55 | 488.92 | 488.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 486.15 | 488.92 | 488.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 491.55 | 489.44 | 488.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:30:00 | 494.80 | 491.16 | 489.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:00:00 | 494.35 | 492.91 | 491.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:45:00 | 494.30 | 493.14 | 491.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 14:15:00 | 494.40 | 493.14 | 491.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 488.50 | 492.34 | 491.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 488.50 | 492.34 | 491.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 491.85 | 492.24 | 491.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 492.30 | 492.24 | 491.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 492.95 | 492.38 | 491.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 492.95 | 492.38 | 491.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 492.00 | 492.30 | 491.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:00:00 | 492.00 | 492.30 | 491.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 492.00 | 492.24 | 491.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 492.00 | 492.24 | 491.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 490.05 | 491.80 | 491.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 483.95 | 491.80 | 491.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 483.65 | 490.17 | 490.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 483.65 | 490.17 | 490.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 479.95 | 487.06 | 489.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 484.95 | 484.34 | 487.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 484.95 | 484.34 | 487.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 485.55 | 484.51 | 486.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 485.75 | 484.51 | 486.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 485.75 | 484.76 | 486.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:00:00 | 485.05 | 484.82 | 486.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 460.80 | 473.47 | 478.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 462.65 | 461.72 | 467.62 | SL hit (close>ema200) qty=0.50 sl=461.72 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 471.30 | 469.21 | 468.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 475.50 | 470.62 | 469.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 467.15 | 470.58 | 470.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 467.15 | 470.58 | 470.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 467.15 | 470.58 | 470.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 467.15 | 470.58 | 470.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 467.55 | 469.97 | 469.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 467.95 | 469.97 | 469.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 15:15:00 | 466.10 | 469.20 | 469.46 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 14:15:00 | 474.05 | 469.72 | 469.48 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 466.40 | 469.65 | 469.65 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 471.55 | 469.84 | 469.72 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 467.50 | 469.25 | 469.47 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 471.55 | 469.31 | 469.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 477.95 | 471.33 | 470.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 11:15:00 | 471.30 | 472.34 | 470.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 11:15:00 | 471.30 | 472.34 | 470.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 471.30 | 472.34 | 470.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 471.30 | 472.34 | 470.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 472.50 | 472.37 | 471.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 14:45:00 | 472.95 | 472.11 | 471.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:30:00 | 475.60 | 472.12 | 471.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 473.00 | 479.71 | 479.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 473.00 | 479.71 | 479.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 469.35 | 477.64 | 478.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 453.85 | 452.96 | 461.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 453.85 | 452.96 | 461.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 451.95 | 453.52 | 460.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 13:15:00 | 446.10 | 453.52 | 460.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 423.80 | 433.49 | 443.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-17 13:15:00 | 401.49 | 412.12 | 423.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 11:15:00 | 406.00 | 400.39 | 400.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 12:15:00 | 413.40 | 402.99 | 401.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 402.05 | 405.42 | 403.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 402.05 | 405.42 | 403.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 402.05 | 405.42 | 403.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:45:00 | 400.15 | 405.42 | 403.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 400.65 | 404.46 | 403.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:15:00 | 400.05 | 404.46 | 403.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 394.10 | 401.28 | 401.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 390.75 | 399.17 | 400.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 390.35 | 386.14 | 389.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 390.35 | 386.14 | 389.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 390.35 | 386.14 | 389.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:45:00 | 395.25 | 386.14 | 389.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 388.00 | 386.51 | 389.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 381.50 | 386.51 | 389.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 11:45:00 | 387.60 | 386.94 | 389.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 391.60 | 388.77 | 389.44 | SL hit (close>static) qty=1.00 sl=390.50 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 12:15:00 | 394.00 | 390.57 | 390.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 395.20 | 392.87 | 391.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 420.00 | 422.70 | 415.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 14:00:00 | 420.00 | 422.70 | 415.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 416.00 | 420.49 | 415.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 412.90 | 420.49 | 415.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 411.80 | 418.76 | 415.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 411.80 | 418.76 | 415.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 414.85 | 417.97 | 415.51 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 407.70 | 413.38 | 413.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 404.70 | 411.65 | 412.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 397.00 | 396.30 | 401.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 13:15:00 | 397.00 | 396.30 | 401.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 397.00 | 396.30 | 401.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:30:00 | 399.10 | 396.30 | 401.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 395.20 | 395.63 | 399.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 390.80 | 394.25 | 397.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 390.40 | 391.55 | 392.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 15:15:00 | 400.00 | 394.26 | 393.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 400.00 | 394.26 | 393.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 404.30 | 396.26 | 394.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 09:15:00 | 462.05 | 462.75 | 448.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 09:30:00 | 462.75 | 462.75 | 448.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 462.10 | 464.54 | 460.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 461.05 | 464.54 | 460.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 460.50 | 463.73 | 460.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 460.50 | 463.73 | 460.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 463.30 | 463.64 | 460.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:30:00 | 462.00 | 463.64 | 460.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 512.25 | 474.24 | 466.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 13:30:00 | 514.20 | 494.36 | 479.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:45:00 | 515.35 | 499.35 | 483.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 528.35 | 500.18 | 485.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 14:30:00 | 516.45 | 508.76 | 497.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 521.00 | 511.21 | 499.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:30:00 | 533.00 | 514.87 | 501.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 11:00:00 | 527.05 | 517.31 | 504.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 11:45:00 | 532.25 | 520.05 | 506.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 506.30 | 512.32 | 513.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 506.30 | 512.32 | 513.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 489.25 | 504.12 | 508.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 467.60 | 465.64 | 478.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 471.85 | 465.64 | 478.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 461.75 | 464.87 | 477.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:00:00 | 454.50 | 464.90 | 470.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:45:00 | 460.15 | 463.66 | 468.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:15:00 | 458.80 | 463.66 | 468.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 13:15:00 | 459.70 | 463.13 | 466.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 468.30 | 463.99 | 466.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:45:00 | 468.85 | 463.99 | 466.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 465.75 | 464.34 | 466.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 09:15:00 | 474.80 | 464.34 | 466.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 480.05 | 467.48 | 467.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 480.05 | 467.48 | 467.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 482.95 | 473.80 | 470.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 484.65 | 485.40 | 480.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 484.65 | 485.40 | 480.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 478.15 | 483.54 | 481.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:00:00 | 478.15 | 483.54 | 481.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 476.40 | 482.11 | 480.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:00:00 | 476.40 | 482.11 | 480.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 14:15:00 | 474.65 | 479.41 | 479.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 15:15:00 | 472.90 | 478.11 | 478.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 12:15:00 | 480.05 | 477.23 | 478.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 12:15:00 | 480.05 | 477.23 | 478.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 480.05 | 477.23 | 478.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:00:00 | 480.05 | 477.23 | 478.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 480.20 | 477.82 | 478.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 14:00:00 | 480.20 | 477.82 | 478.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 15:15:00 | 478.90 | 478.67 | 478.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 484.80 | 479.89 | 479.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 13:15:00 | 480.60 | 480.98 | 480.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:00:00 | 480.60 | 480.98 | 480.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 476.25 | 480.03 | 479.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 14:45:00 | 475.85 | 480.03 | 479.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 477.50 | 479.53 | 479.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 478.10 | 479.53 | 479.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 471.70 | 477.96 | 478.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 471.70 | 477.96 | 478.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 10:15:00 | 470.55 | 476.48 | 478.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 10:15:00 | 471.00 | 470.66 | 473.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-24 11:00:00 | 471.00 | 470.66 | 473.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 454.95 | 451.84 | 457.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 13:15:00 | 449.70 | 452.15 | 456.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:30:00 | 448.25 | 449.00 | 453.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:00:00 | 449.20 | 449.07 | 450.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 15:15:00 | 455.95 | 447.81 | 447.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 15:15:00 | 455.95 | 447.81 | 447.68 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 14:15:00 | 444.45 | 447.58 | 447.77 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 448.30 | 447.81 | 447.79 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 447.35 | 447.72 | 447.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 445.00 | 447.18 | 447.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 15:15:00 | 446.70 | 446.51 | 447.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 09:15:00 | 441.65 | 446.51 | 447.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 439.15 | 443.62 | 445.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:30:00 | 444.65 | 443.62 | 445.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 444.35 | 443.37 | 444.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 444.35 | 443.37 | 444.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 444.00 | 443.49 | 444.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 449.00 | 443.49 | 444.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 452.80 | 445.35 | 445.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 452.80 | 445.35 | 445.46 | SL hit (close>ema400) qty=1.00 sl=445.46 alert=retest1 |

### Cycle 77 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 453.20 | 446.92 | 446.16 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 437.90 | 445.11 | 445.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 434.45 | 442.98 | 444.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 449.80 | 441.62 | 442.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 449.80 | 441.62 | 442.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 449.80 | 441.62 | 442.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 449.05 | 441.62 | 442.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 449.00 | 443.10 | 443.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 457.15 | 447.20 | 445.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 449.30 | 450.37 | 448.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 449.30 | 450.37 | 448.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 449.70 | 450.10 | 448.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 449.70 | 450.10 | 448.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 448.55 | 449.79 | 448.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 463.00 | 449.79 | 448.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 460.60 | 466.03 | 466.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 460.60 | 466.03 | 466.76 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 12:15:00 | 469.75 | 466.11 | 465.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 471.45 | 467.79 | 466.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 10:15:00 | 464.80 | 467.77 | 467.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 464.80 | 467.77 | 467.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 464.80 | 467.77 | 467.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 464.80 | 467.77 | 467.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 466.65 | 467.55 | 467.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 466.25 | 467.55 | 467.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 12:15:00 | 461.80 | 466.40 | 466.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 13:15:00 | 460.00 | 465.12 | 465.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 465.25 | 464.47 | 465.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 465.25 | 464.47 | 465.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 465.25 | 464.47 | 465.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:15:00 | 467.60 | 464.47 | 465.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 467.60 | 465.10 | 465.60 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 466.85 | 466.04 | 465.96 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 463.70 | 465.77 | 465.86 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 471.40 | 466.25 | 465.84 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 456.95 | 465.24 | 465.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 445.65 | 455.38 | 458.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 445.65 | 441.33 | 444.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 445.65 | 441.33 | 444.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 445.65 | 441.33 | 444.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 445.65 | 441.33 | 444.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 454.00 | 443.86 | 445.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:30:00 | 460.75 | 443.86 | 445.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 457.00 | 446.49 | 446.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:45:00 | 456.65 | 446.49 | 446.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 454.00 | 447.99 | 447.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 14:15:00 | 460.80 | 455.48 | 453.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 10:15:00 | 512.90 | 513.47 | 502.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 10:30:00 | 514.85 | 513.47 | 502.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 507.80 | 510.62 | 505.50 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 495.55 | 504.17 | 504.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 492.40 | 501.81 | 503.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 493.00 | 492.88 | 497.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:45:00 | 489.40 | 492.88 | 497.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 494.10 | 493.12 | 497.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 497.20 | 493.12 | 497.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 491.20 | 490.17 | 493.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 493.95 | 490.17 | 493.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 490.40 | 490.59 | 492.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:30:00 | 491.85 | 490.59 | 492.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 491.35 | 490.74 | 492.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 485.00 | 490.49 | 491.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 486.20 | 487.04 | 489.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 13:15:00 | 461.89 | 468.06 | 471.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 474.95 | 468.98 | 470.81 | SL hit (close>ema200) qty=0.50 sl=468.98 alert=retest2 |

### Cycle 89 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 477.50 | 471.92 | 471.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 13:15:00 | 482.50 | 475.11 | 473.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 478.25 | 478.45 | 475.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 478.25 | 478.45 | 475.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 475.85 | 477.93 | 475.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 476.20 | 477.93 | 475.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 476.35 | 477.61 | 475.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:30:00 | 478.25 | 477.51 | 476.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 09:15:00 | 475.15 | 477.16 | 476.22 | SL hit (close<static) qty=1.00 sl=475.65 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 472.55 | 475.22 | 475.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 470.65 | 473.84 | 474.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 15:15:00 | 469.00 | 467.91 | 470.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:15:00 | 469.80 | 467.91 | 470.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 470.85 | 468.50 | 470.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 469.45 | 468.50 | 470.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 471.10 | 469.02 | 470.53 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 476.20 | 471.65 | 471.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 479.70 | 474.44 | 472.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 15:15:00 | 481.00 | 481.28 | 478.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:15:00 | 480.40 | 481.28 | 478.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 481.95 | 481.41 | 478.41 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 475.25 | 477.51 | 477.65 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 480.00 | 477.60 | 477.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 485.80 | 481.11 | 479.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 482.00 | 482.19 | 480.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 15:15:00 | 482.00 | 482.19 | 480.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 482.00 | 482.19 | 480.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 481.35 | 482.09 | 481.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 479.70 | 481.61 | 480.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 480.05 | 481.61 | 480.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 478.70 | 481.03 | 480.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 478.70 | 481.03 | 480.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 478.60 | 480.36 | 480.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 475.00 | 478.79 | 479.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 482.70 | 477.58 | 478.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 482.70 | 477.58 | 478.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 482.70 | 477.58 | 478.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:30:00 | 482.90 | 477.58 | 478.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 481.05 | 478.28 | 478.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 479.95 | 478.28 | 478.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 486.60 | 479.94 | 479.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 486.60 | 479.94 | 479.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 513.65 | 488.78 | 484.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 513.00 | 519.81 | 511.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 513.00 | 519.81 | 511.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 513.00 | 519.81 | 511.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 512.60 | 519.81 | 511.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 507.00 | 517.25 | 511.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 507.85 | 517.25 | 511.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 506.00 | 515.00 | 510.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 509.80 | 510.83 | 509.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 508.00 | 510.83 | 509.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 501.60 | 508.53 | 508.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 501.60 | 508.53 | 508.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 497.40 | 506.31 | 507.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 512.20 | 501.79 | 504.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 512.20 | 501.79 | 504.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 512.20 | 501.79 | 504.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 506.85 | 501.79 | 504.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 509.00 | 503.23 | 504.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:15:00 | 505.75 | 503.23 | 504.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 15:15:00 | 500.00 | 497.57 | 497.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 15:15:00 | 500.00 | 497.57 | 497.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 517.70 | 501.59 | 499.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 514.60 | 516.96 | 509.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 514.60 | 516.96 | 509.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 509.45 | 515.46 | 509.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:30:00 | 521.55 | 517.44 | 511.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 504.95 | 519.02 | 518.79 | SL hit (close<static) qty=1.00 sl=505.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 506.00 | 516.42 | 517.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 500.20 | 508.80 | 513.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 510.35 | 509.11 | 512.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:45:00 | 508.75 | 509.11 | 512.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 510.20 | 503.75 | 507.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:30:00 | 512.20 | 503.75 | 507.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 504.50 | 503.90 | 507.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:30:00 | 499.55 | 503.22 | 506.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 495.00 | 502.88 | 505.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 474.57 | 496.55 | 502.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 490.10 | 489.63 | 496.48 | SL hit (close>ema200) qty=0.50 sl=489.63 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 501.55 | 496.36 | 495.96 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 493.30 | 495.28 | 495.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 489.75 | 494.17 | 494.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 493.70 | 491.81 | 493.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 493.70 | 491.81 | 493.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 493.70 | 491.81 | 493.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:30:00 | 489.00 | 490.74 | 492.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:00:00 | 489.60 | 490.51 | 492.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 489.00 | 490.46 | 492.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 500.00 | 493.24 | 492.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 500.00 | 493.24 | 492.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 501.75 | 494.95 | 493.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 501.95 | 503.02 | 499.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 12:00:00 | 501.95 | 503.02 | 499.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 499.50 | 502.32 | 499.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 499.50 | 502.32 | 499.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 499.25 | 501.70 | 499.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:45:00 | 497.30 | 501.70 | 499.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 499.15 | 501.19 | 499.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 499.15 | 501.19 | 499.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 497.60 | 500.47 | 499.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:30:00 | 505.05 | 501.95 | 499.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 499.65 | 501.62 | 500.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 499.95 | 501.15 | 500.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 492.35 | 499.00 | 499.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 492.35 | 499.00 | 499.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 489.90 | 497.18 | 498.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 476.50 | 476.25 | 481.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:30:00 | 477.10 | 476.25 | 481.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 474.20 | 473.77 | 476.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 477.00 | 473.77 | 476.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 478.50 | 474.72 | 476.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 481.85 | 474.72 | 476.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 478.25 | 475.42 | 476.54 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 480.45 | 477.55 | 477.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 483.40 | 478.89 | 478.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 477.60 | 480.96 | 479.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 477.60 | 480.96 | 479.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 477.60 | 480.96 | 479.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 477.60 | 480.96 | 479.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 480.80 | 480.93 | 479.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 485.70 | 480.54 | 479.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 477.55 | 479.21 | 479.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 477.55 | 479.21 | 479.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 475.25 | 478.40 | 478.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 474.45 | 473.77 | 475.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 474.45 | 473.77 | 475.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 474.45 | 473.77 | 475.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 473.00 | 474.70 | 475.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 472.30 | 470.75 | 471.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 471.40 | 470.75 | 471.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 476.30 | 471.75 | 471.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 476.30 | 471.75 | 471.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 479.30 | 476.20 | 474.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 477.60 | 478.93 | 477.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 477.60 | 478.93 | 477.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 477.60 | 478.93 | 477.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 477.60 | 478.93 | 477.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 478.95 | 478.93 | 477.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 479.40 | 478.93 | 477.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 479.25 | 479.00 | 477.42 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 10:15:00 | 475.20 | 476.73 | 476.93 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 480.05 | 476.99 | 476.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 490.15 | 480.87 | 478.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 14:15:00 | 483.95 | 484.01 | 481.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:45:00 | 483.50 | 484.01 | 481.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 477.75 | 483.04 | 481.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 477.75 | 483.04 | 481.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 478.55 | 482.14 | 481.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 479.00 | 482.14 | 481.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 480.25 | 481.58 | 481.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 478.40 | 481.17 | 481.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 478.40 | 481.17 | 481.26 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 485.00 | 481.37 | 481.05 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 479.60 | 480.83 | 480.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 476.80 | 479.96 | 480.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 448.70 | 446.97 | 453.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 448.70 | 446.97 | 453.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 444.30 | 445.80 | 450.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 437.85 | 445.08 | 449.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 439.00 | 443.87 | 448.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:15:00 | 439.90 | 443.56 | 446.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 439.75 | 444.10 | 445.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 437.90 | 442.86 | 444.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:00:00 | 436.70 | 441.63 | 443.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 454.75 | 443.13 | 443.31 | SL hit (close>static) qty=1.00 sl=450.60 alert=retest2 |

### Cycle 111 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 493.05 | 453.11 | 447.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 11:15:00 | 501.50 | 462.79 | 452.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 475.45 | 477.04 | 465.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:45:00 | 474.10 | 477.04 | 465.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 466.40 | 473.32 | 466.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:00:00 | 466.40 | 473.32 | 466.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 461.15 | 470.89 | 465.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 461.15 | 470.89 | 465.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 457.75 | 468.26 | 465.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 457.40 | 468.26 | 465.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 447.70 | 462.47 | 462.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 444.55 | 448.38 | 452.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 448.35 | 445.18 | 448.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 448.35 | 445.18 | 448.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 448.35 | 445.18 | 448.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 450.25 | 445.18 | 448.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 446.50 | 445.44 | 448.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 446.85 | 445.44 | 448.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 446.45 | 445.78 | 447.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:30:00 | 449.15 | 445.78 | 447.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 453.25 | 447.27 | 448.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 454.35 | 447.27 | 448.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 447.00 | 447.22 | 447.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 446.10 | 447.37 | 447.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 451.25 | 448.83 | 448.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 451.25 | 448.83 | 448.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 453.10 | 449.68 | 448.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 15:15:00 | 447.10 | 449.21 | 448.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 15:15:00 | 447.10 | 449.21 | 448.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 447.10 | 449.21 | 448.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 452.65 | 449.21 | 448.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 10:30:00 | 452.45 | 449.82 | 449.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 15:15:00 | 446.55 | 450.02 | 450.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 446.55 | 450.02 | 450.36 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 453.95 | 450.83 | 450.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 13:15:00 | 454.90 | 452.93 | 452.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 450.70 | 452.48 | 451.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 450.70 | 452.48 | 451.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 450.70 | 452.48 | 451.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 450.70 | 452.48 | 451.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 449.00 | 451.78 | 451.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 451.00 | 451.78 | 451.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 449.25 | 451.21 | 451.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 449.25 | 451.21 | 451.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 447.90 | 448.85 | 449.34 | Break + close below crossover candle low |

### Cycle 117 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 460.60 | 450.90 | 450.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 472.15 | 463.62 | 459.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 476.70 | 478.27 | 473.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:45:00 | 475.95 | 478.27 | 473.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 469.35 | 475.89 | 474.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 469.35 | 475.89 | 474.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 466.80 | 474.07 | 473.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 467.05 | 474.07 | 473.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 464.20 | 472.10 | 472.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 460.50 | 468.93 | 471.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 469.00 | 468.94 | 470.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 469.00 | 468.94 | 470.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 471.30 | 469.42 | 470.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 471.30 | 469.42 | 470.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 470.85 | 469.70 | 470.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 470.75 | 470.10 | 470.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 473.10 | 464.37 | 466.46 | SL hit (close>static) qty=1.00 sl=472.60 alert=retest2 |

### Cycle 119 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 481.60 | 469.40 | 468.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 492.05 | 479.40 | 473.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 486.00 | 488.08 | 482.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:00:00 | 486.00 | 488.08 | 482.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 482.15 | 486.94 | 483.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 482.15 | 486.94 | 483.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 481.80 | 485.91 | 482.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 481.20 | 485.91 | 482.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 485.10 | 485.75 | 483.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 486.15 | 485.00 | 483.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 481.60 | 485.99 | 485.61 | SL hit (close<static) qty=1.00 sl=481.80 alert=retest2 |

### Cycle 120 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 481.60 | 485.11 | 485.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 479.00 | 481.92 | 483.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 483.65 | 482.27 | 483.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 483.65 | 482.27 | 483.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 483.65 | 482.27 | 483.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 483.65 | 482.27 | 483.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 481.90 | 482.20 | 483.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:45:00 | 480.45 | 481.74 | 482.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 479.80 | 481.27 | 482.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 456.43 | 464.01 | 469.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 455.81 | 464.01 | 469.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 458.65 | 458.19 | 464.28 | SL hit (close>ema200) qty=0.50 sl=458.19 alert=retest2 |

### Cycle 121 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 469.90 | 463.44 | 463.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 473.00 | 466.39 | 464.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 469.75 | 469.89 | 467.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:30:00 | 469.85 | 469.89 | 467.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 467.75 | 469.30 | 467.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 467.75 | 469.30 | 467.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 467.05 | 468.85 | 467.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:45:00 | 465.35 | 468.85 | 467.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 467.00 | 468.48 | 467.51 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 464.00 | 466.56 | 466.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 462.00 | 463.64 | 464.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 463.10 | 462.09 | 463.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 13:15:00 | 463.10 | 462.09 | 463.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 463.10 | 462.09 | 463.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 464.75 | 462.09 | 463.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 465.00 | 462.67 | 463.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 465.00 | 462.67 | 463.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 465.00 | 463.14 | 463.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 462.05 | 463.14 | 463.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 438.95 | 445.42 | 449.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 445.50 | 443.47 | 446.38 | SL hit (close>ema200) qty=0.50 sl=443.47 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 445.95 | 442.95 | 442.79 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 440.20 | 442.54 | 442.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 438.80 | 441.79 | 442.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 440.90 | 440.42 | 441.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 440.90 | 440.42 | 441.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 440.90 | 440.42 | 441.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 442.85 | 440.42 | 441.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 438.90 | 440.11 | 441.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:00:00 | 436.40 | 439.37 | 440.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:45:00 | 436.80 | 438.71 | 439.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 437.00 | 438.46 | 439.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 436.75 | 438.25 | 439.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 440.80 | 438.52 | 439.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 446.00 | 440.41 | 439.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 446.00 | 440.41 | 439.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 458.40 | 445.22 | 442.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 449.20 | 450.57 | 446.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:45:00 | 449.05 | 450.57 | 446.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 450.85 | 452.04 | 450.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 453.75 | 452.04 | 450.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 452.65 | 452.14 | 450.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 11:15:00 | 448.50 | 451.14 | 450.27 | SL hit (close<static) qty=1.00 sl=450.10 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 447.00 | 449.61 | 449.68 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 451.50 | 449.69 | 449.56 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 447.90 | 449.39 | 449.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 444.95 | 448.20 | 448.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 456.75 | 447.73 | 447.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 456.75 | 447.73 | 447.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 456.75 | 447.73 | 447.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 456.75 | 447.73 | 447.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 459.60 | 450.11 | 448.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 466.00 | 457.22 | 455.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 13:15:00 | 479.45 | 480.60 | 475.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:30:00 | 478.95 | 480.60 | 475.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 475.35 | 479.71 | 476.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 475.35 | 479.71 | 476.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 473.35 | 478.44 | 475.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:15:00 | 470.85 | 478.44 | 475.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 468.20 | 476.39 | 475.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 468.20 | 476.39 | 475.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 467.00 | 473.23 | 473.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 463.65 | 471.31 | 472.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 447.00 | 446.12 | 450.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 447.00 | 446.12 | 450.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 450.95 | 446.94 | 449.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 450.95 | 446.94 | 449.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 459.10 | 449.37 | 450.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:15:00 | 467.10 | 449.37 | 450.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 464.05 | 452.31 | 451.94 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 446.85 | 452.59 | 453.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 445.90 | 450.13 | 451.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 430.45 | 425.81 | 432.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 430.45 | 425.81 | 432.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 430.85 | 426.82 | 432.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 430.35 | 426.82 | 432.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 430.10 | 427.47 | 432.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 426.95 | 428.65 | 431.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 443.05 | 418.91 | 417.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 443.05 | 418.91 | 417.89 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 422.35 | 428.70 | 429.02 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 438.00 | 429.24 | 428.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 442.70 | 435.50 | 432.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 436.50 | 437.53 | 434.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 13:00:00 | 436.50 | 437.53 | 434.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 437.45 | 440.48 | 437.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 437.40 | 440.48 | 437.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 439.60 | 440.31 | 437.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:30:00 | 437.80 | 440.31 | 437.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 440.00 | 440.37 | 438.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 436.70 | 440.37 | 438.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 431.55 | 438.61 | 437.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 431.55 | 438.61 | 437.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 431.35 | 437.16 | 437.25 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 443.80 | 437.55 | 436.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 450.00 | 441.91 | 439.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 463.15 | 464.71 | 457.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 13:00:00 | 463.15 | 464.71 | 457.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 457.00 | 461.23 | 458.12 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 442.00 | 454.67 | 455.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 10:15:00 | 440.00 | 451.74 | 454.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 463.50 | 446.39 | 449.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 463.50 | 446.39 | 449.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 463.50 | 446.39 | 449.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 463.50 | 446.39 | 449.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 462.40 | 449.59 | 450.76 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 459.70 | 451.61 | 451.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 465.50 | 457.52 | 454.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 470.00 | 470.23 | 464.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:15:00 | 473.25 | 470.23 | 464.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 470.90 | 476.86 | 472.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 470.90 | 476.86 | 472.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 472.00 | 475.89 | 472.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 470.95 | 475.89 | 472.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 464.05 | 470.88 | 471.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 460.70 | 468.85 | 470.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 442.40 | 437.93 | 442.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 442.40 | 437.93 | 442.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 442.40 | 437.93 | 442.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 442.40 | 437.93 | 442.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 439.50 | 438.24 | 442.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 437.80 | 438.90 | 441.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:15:00 | 415.91 | 424.91 | 431.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 09:15:00 | 394.02 | 402.09 | 412.68 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 379.30 | 372.30 | 372.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 380.70 | 375.60 | 373.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 373.70 | 376.66 | 374.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 373.70 | 376.66 | 374.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 373.70 | 376.66 | 374.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 374.15 | 376.66 | 374.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 371.50 | 375.63 | 374.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 371.50 | 375.63 | 374.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 373.00 | 375.10 | 374.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:30:00 | 372.30 | 375.10 | 374.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 371.60 | 374.40 | 374.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 371.60 | 374.40 | 374.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 367.85 | 373.09 | 373.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 366.00 | 371.67 | 372.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 372.85 | 371.16 | 372.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 372.85 | 371.16 | 372.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 372.85 | 371.16 | 372.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 371.20 | 371.16 | 372.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 373.75 | 371.68 | 372.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 374.00 | 371.68 | 372.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 372.90 | 371.92 | 372.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 374.05 | 371.92 | 372.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 373.00 | 372.14 | 372.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:45:00 | 372.80 | 372.14 | 372.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 373.30 | 372.37 | 372.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 374.50 | 372.37 | 372.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 356.85 | 348.14 | 352.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:30:00 | 345.00 | 350.63 | 352.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 327.75 | 337.06 | 343.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 340.70 | 332.20 | 338.16 | SL hit (close>ema200) qty=0.50 sl=332.20 alert=retest2 |

### Cycle 143 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 354.00 | 341.60 | 341.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 362.30 | 345.74 | 343.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 354.65 | 356.38 | 350.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 10:15:00 | 352.20 | 356.38 | 350.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 381.85 | 383.68 | 379.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 380.35 | 383.68 | 379.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 381.70 | 383.28 | 380.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 381.70 | 383.28 | 380.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 378.05 | 382.24 | 379.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 386.45 | 382.24 | 379.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 383.15 | 385.88 | 383.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 09:15:00 | 421.47 | 410.58 | 406.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 403.65 | 408.55 | 409.02 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 414.45 | 407.74 | 407.61 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 407.25 | 409.96 | 409.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 406.15 | 409.20 | 409.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 407.50 | 404.92 | 406.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 407.50 | 404.92 | 406.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 407.50 | 404.92 | 406.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 403.75 | 406.29 | 406.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 404.00 | 405.83 | 406.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 404.50 | 403.51 | 404.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 402.70 | 403.86 | 404.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 402.35 | 403.56 | 404.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 409.95 | 405.66 | 405.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 409.95 | 405.66 | 405.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 411.00 | 406.72 | 405.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 409.95 | 410.54 | 408.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 11:15:00 | 409.15 | 410.54 | 408.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 409.35 | 410.30 | 408.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 409.20 | 410.30 | 408.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 409.35 | 410.11 | 408.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:30:00 | 407.80 | 410.11 | 408.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 409.20 | 409.93 | 408.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 408.05 | 409.93 | 408.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:15:00 | 618.40 | 2024-05-13 11:15:00 | 620.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-05-24 11:00:00 | 675.50 | 2024-05-28 09:15:00 | 660.35 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-05-24 14:45:00 | 672.05 | 2024-05-28 09:15:00 | 660.35 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-05-27 11:15:00 | 674.00 | 2024-05-28 09:15:00 | 660.35 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-05-27 12:15:00 | 675.10 | 2024-05-28 09:15:00 | 660.35 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-06-06 13:00:00 | 669.10 | 2024-06-07 10:15:00 | 679.25 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-06-06 14:15:00 | 668.65 | 2024-06-07 10:15:00 | 679.25 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-06-06 15:15:00 | 669.10 | 2024-06-07 10:15:00 | 679.25 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-06-24 12:15:00 | 654.70 | 2024-07-01 11:15:00 | 643.75 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2024-07-10 10:30:00 | 607.00 | 2024-07-12 15:15:00 | 576.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-10 10:30:00 | 607.00 | 2024-07-15 10:15:00 | 597.40 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2024-07-23 12:15:00 | 576.00 | 2024-07-24 14:15:00 | 589.40 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-07-24 09:30:00 | 584.65 | 2024-07-24 14:15:00 | 589.40 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-07-24 13:00:00 | 586.70 | 2024-07-24 14:15:00 | 589.40 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-07-24 13:45:00 | 586.20 | 2024-07-24 14:15:00 | 589.40 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-07-26 09:15:00 | 593.35 | 2024-07-31 10:15:00 | 599.55 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2024-08-06 11:15:00 | 584.10 | 2024-08-08 10:15:00 | 588.45 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-08-06 12:00:00 | 584.60 | 2024-08-08 10:15:00 | 588.45 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-08-07 13:15:00 | 584.10 | 2024-08-08 10:15:00 | 588.45 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-08-21 11:15:00 | 634.00 | 2024-08-23 13:15:00 | 629.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-08-22 09:15:00 | 644.00 | 2024-08-23 13:15:00 | 629.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-08-23 09:15:00 | 639.00 | 2024-08-23 13:15:00 | 629.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-08-23 09:45:00 | 634.10 | 2024-08-23 13:15:00 | 629.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-08-26 14:15:00 | 629.75 | 2024-09-06 10:15:00 | 598.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-26 15:15:00 | 629.95 | 2024-09-06 10:15:00 | 598.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-26 14:15:00 | 629.75 | 2024-09-10 10:15:00 | 591.00 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2024-08-26 15:15:00 | 629.95 | 2024-09-10 10:15:00 | 591.00 | STOP_HIT | 0.50 | 6.18% |
| BUY | retest2 | 2024-09-23 14:30:00 | 598.95 | 2024-09-23 15:15:00 | 593.55 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-09-23 15:00:00 | 601.00 | 2024-09-23 15:15:00 | 593.55 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-09-24 09:45:00 | 603.00 | 2024-09-27 12:15:00 | 612.80 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2024-10-09 12:45:00 | 591.80 | 2024-10-14 09:15:00 | 595.15 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-10-10 12:15:00 | 589.90 | 2024-10-14 09:15:00 | 595.15 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-10-11 10:15:00 | 591.70 | 2024-10-14 09:15:00 | 595.15 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-10-17 13:30:00 | 618.30 | 2024-10-17 14:15:00 | 609.35 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-10-21 09:15:00 | 604.90 | 2024-10-22 12:15:00 | 574.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 604.90 | 2024-10-23 10:15:00 | 584.50 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest2 | 2024-12-06 14:45:00 | 560.80 | 2024-12-12 14:15:00 | 557.55 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-12-09 09:15:00 | 558.45 | 2024-12-12 14:15:00 | 557.55 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-12-09 10:15:00 | 559.95 | 2024-12-12 14:15:00 | 557.55 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-12-10 11:15:00 | 564.15 | 2024-12-12 14:15:00 | 557.55 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-12-11 10:00:00 | 571.05 | 2024-12-12 14:15:00 | 557.55 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-12-16 12:00:00 | 553.70 | 2024-12-18 09:15:00 | 567.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-12-16 14:45:00 | 553.95 | 2024-12-18 09:15:00 | 567.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-01-01 15:15:00 | 496.10 | 2025-01-02 11:15:00 | 504.55 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-01-08 12:30:00 | 491.00 | 2025-01-09 09:15:00 | 505.65 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-01-08 15:15:00 | 492.10 | 2025-01-09 09:15:00 | 505.65 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-01-17 14:30:00 | 494.80 | 2025-01-22 09:15:00 | 483.65 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-01-20 13:00:00 | 494.35 | 2025-01-22 09:15:00 | 483.65 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-01-20 13:45:00 | 494.30 | 2025-01-22 09:15:00 | 483.65 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-01-20 14:15:00 | 494.40 | 2025-01-22 09:15:00 | 483.65 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-01-23 12:00:00 | 485.05 | 2025-01-27 09:15:00 | 460.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:00:00 | 485.05 | 2025-01-28 11:15:00 | 462.65 | STOP_HIT | 0.50 | 4.62% |
| BUY | retest2 | 2025-02-04 14:45:00 | 472.95 | 2025-02-10 10:15:00 | 473.00 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-02-05 09:30:00 | 475.60 | 2025-02-10 10:15:00 | 473.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-02-12 13:15:00 | 446.10 | 2025-02-14 09:15:00 | 423.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 13:15:00 | 446.10 | 2025-02-17 13:15:00 | 401.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 381.50 | 2025-02-28 14:15:00 | 391.60 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-02-28 11:45:00 | 387.60 | 2025-02-28 14:15:00 | 391.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-03-03 09:15:00 | 382.45 | 2025-03-03 12:15:00 | 394.00 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-03-17 09:15:00 | 390.80 | 2025-03-18 15:15:00 | 400.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-03-18 12:30:00 | 390.40 | 2025-03-18 15:15:00 | 400.00 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-03-27 13:30:00 | 514.20 | 2025-04-03 10:15:00 | 506.30 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-03-27 14:45:00 | 515.35 | 2025-04-03 10:15:00 | 506.30 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-03-28 09:15:00 | 528.35 | 2025-04-03 10:15:00 | 506.30 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2025-03-28 14:30:00 | 516.45 | 2025-04-03 10:15:00 | 506.30 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-04-01 09:30:00 | 533.00 | 2025-04-03 10:15:00 | 506.30 | STOP_HIT | 1.00 | -5.01% |
| BUY | retest2 | 2025-04-01 11:00:00 | 527.05 | 2025-04-03 10:15:00 | 506.30 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2025-04-01 11:45:00 | 532.25 | 2025-04-03 10:15:00 | 506.30 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2025-04-09 13:00:00 | 454.50 | 2025-04-15 09:15:00 | 480.05 | STOP_HIT | 1.00 | -5.62% |
| SELL | retest2 | 2025-04-09 14:45:00 | 460.15 | 2025-04-15 09:15:00 | 480.05 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2025-04-09 15:15:00 | 458.80 | 2025-04-15 09:15:00 | 480.05 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2025-04-11 13:15:00 | 459.70 | 2025-04-15 09:15:00 | 480.05 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest2 | 2025-04-23 09:15:00 | 478.10 | 2025-04-23 09:15:00 | 471.70 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-04-28 13:15:00 | 449.70 | 2025-05-02 15:15:00 | 455.95 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-04-29 09:30:00 | 448.25 | 2025-05-02 15:15:00 | 455.95 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-04-30 12:00:00 | 449.20 | 2025-05-02 15:15:00 | 455.95 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest1 | 2025-05-07 09:15:00 | 441.65 | 2025-05-08 09:15:00 | 452.80 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-14 09:15:00 | 463.00 | 2025-05-21 11:15:00 | 460.60 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-06-24 12:30:00 | 485.00 | 2025-07-02 13:15:00 | 461.89 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2025-06-24 12:30:00 | 485.00 | 2025-07-03 09:15:00 | 474.95 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2025-06-25 09:30:00 | 486.20 | 2025-07-03 11:15:00 | 477.50 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2025-07-04 14:30:00 | 478.25 | 2025-07-07 09:15:00 | 475.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-21 09:15:00 | 479.95 | 2025-07-21 09:15:00 | 486.60 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-24 14:30:00 | 509.80 | 2025-07-25 09:15:00 | 501.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-24 15:15:00 | 508.00 | 2025-07-25 09:15:00 | 501.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-28 11:15:00 | 505.75 | 2025-07-31 15:15:00 | 500.00 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-08-04 10:30:00 | 521.55 | 2025-08-06 10:15:00 | 504.95 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-08-08 12:30:00 | 499.55 | 2025-08-11 09:15:00 | 474.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 12:30:00 | 499.55 | 2025-08-11 13:15:00 | 490.10 | STOP_HIT | 0.50 | 1.89% |
| SELL | retest2 | 2025-08-08 15:15:00 | 495.00 | 2025-08-14 09:15:00 | 501.55 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-12 11:00:00 | 499.60 | 2025-08-14 09:15:00 | 501.55 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-08-12 12:00:00 | 496.70 | 2025-08-14 09:15:00 | 501.55 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-12 14:45:00 | 491.05 | 2025-08-14 09:15:00 | 501.55 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-08-18 11:30:00 | 489.00 | 2025-08-19 10:15:00 | 500.00 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-08-18 13:00:00 | 489.60 | 2025-08-19 10:15:00 | 500.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-08-18 14:45:00 | 489.00 | 2025-08-19 10:15:00 | 500.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-08-21 09:30:00 | 505.05 | 2025-08-22 09:15:00 | 492.35 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-08-21 13:15:00 | 499.65 | 2025-08-22 09:15:00 | 492.35 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-08-21 15:00:00 | 499.95 | 2025-08-22 09:15:00 | 492.35 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-09-03 09:15:00 | 485.70 | 2025-09-03 15:15:00 | 477.55 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-08 10:30:00 | 473.00 | 2025-09-10 13:15:00 | 476.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-10 10:30:00 | 472.30 | 2025-09-10 13:15:00 | 476.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-10 11:15:00 | 471.40 | 2025-09-10 13:15:00 | 476.30 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-09-18 13:15:00 | 479.00 | 2025-09-19 09:15:00 | 478.40 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-09-18 14:30:00 | 480.25 | 2025-09-19 09:15:00 | 478.40 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-09-30 09:15:00 | 437.85 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-09-30 10:00:00 | 439.00 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-09-30 14:15:00 | 439.90 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-10-06 09:15:00 | 439.75 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-10-06 11:00:00 | 436.70 | 2025-10-07 09:15:00 | 454.75 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest2 | 2025-10-15 10:15:00 | 446.10 | 2025-10-15 12:15:00 | 451.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-16 09:15:00 | 452.65 | 2025-10-17 15:15:00 | 446.55 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-16 10:30:00 | 452.45 | 2025-10-17 15:15:00 | 446.55 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-24 09:15:00 | 451.00 | 2025-10-24 10:15:00 | 449.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-11-07 15:00:00 | 470.75 | 2025-11-11 10:15:00 | 473.10 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-11 11:15:00 | 467.55 | 2025-11-12 11:15:00 | 481.60 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-11-12 11:00:00 | 469.25 | 2025-11-12 11:15:00 | 481.60 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-11-17 09:15:00 | 486.15 | 2025-11-18 11:15:00 | 481.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-19 14:45:00 | 480.45 | 2025-11-24 11:15:00 | 456.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:45:00 | 479.80 | 2025-11-24 11:15:00 | 455.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 14:45:00 | 480.45 | 2025-11-25 09:15:00 | 458.65 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2025-11-20 11:45:00 | 479.80 | 2025-11-25 09:15:00 | 458.65 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-12-03 09:15:00 | 462.05 | 2025-12-08 13:15:00 | 438.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 462.05 | 2025-12-09 12:15:00 | 445.50 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-17 12:00:00 | 436.40 | 2025-12-19 11:15:00 | 446.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-12-18 09:45:00 | 436.80 | 2025-12-19 11:15:00 | 446.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-12-18 14:15:00 | 437.00 | 2025-12-19 11:15:00 | 446.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-12-18 15:15:00 | 436.75 | 2025-12-19 11:15:00 | 446.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-24 09:15:00 | 453.75 | 2025-12-24 11:15:00 | 448.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-24 09:45:00 | 452.65 | 2025-12-24 11:15:00 | 448.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-23 10:45:00 | 426.95 | 2026-01-30 09:15:00 | 443.05 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2026-02-26 15:15:00 | 437.80 | 2026-03-02 12:15:00 | 415.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 437.80 | 2026-03-05 09:15:00 | 394.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-27 09:30:00 | 345.00 | 2026-03-30 09:15:00 | 327.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:30:00 | 345.00 | 2026-03-30 14:15:00 | 340.70 | STOP_HIT | 0.50 | 1.25% |
| BUY | retest2 | 2026-04-10 09:15:00 | 386.45 | 2026-04-22 09:15:00 | 421.47 | TARGET_HIT | 1.00 | 9.06% |
| BUY | retest2 | 2026-04-13 09:45:00 | 383.15 | 2026-04-22 10:15:00 | 425.10 | TARGET_HIT | 1.00 | 10.95% |
| SELL | retest2 | 2026-05-05 09:15:00 | 403.75 | 2026-05-06 15:15:00 | 409.95 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-05-05 10:00:00 | 404.00 | 2026-05-06 15:15:00 | 409.95 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-05-06 10:30:00 | 404.50 | 2026-05-06 15:15:00 | 409.95 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-05-06 12:15:00 | 402.70 | 2026-05-06 15:15:00 | 409.95 | STOP_HIT | 1.00 | -1.80% |
