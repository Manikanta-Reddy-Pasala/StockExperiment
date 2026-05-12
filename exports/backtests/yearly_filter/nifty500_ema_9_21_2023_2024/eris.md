# Eris Lifesciences Ltd. (ERIS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1389.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 132 |
| ALERT2 | 131 |
| ALERT2_SKIP | 86 |
| ALERT3 | 353 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 227 |
| PARTIAL | 17 |
| TARGET_HIT | 28 |
| STOP_HIT | 208 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 247 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 96 / 151
- **Target hits / Stop hits / Partials:** 28 / 202 / 17
- **Avg / median % per leg:** 0.83% / -0.76%
- **Sum % (uncompounded):** 204.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 142 | 45 | 31.7% | 25 | 117 | 0 | 0.94% | 133.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.34% | -2.3% |
| BUY @ 3rd Alert (retest2) | 141 | 45 | 31.9% | 25 | 116 | 0 | 0.97% | 136.2% |
| SELL (all) | 105 | 51 | 48.6% | 3 | 85 | 17 | 0.67% | 70.8% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.59% | -1.2% |
| SELL @ 3rd Alert (retest2) | 103 | 50 | 48.5% | 3 | 83 | 17 | 0.70% | 72.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.17% | -3.5% |
| retest2 (combined) | 244 | 95 | 38.9% | 28 | 199 | 17 | 0.85% | 208.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 09:15:00 | 629.60 | 632.00 | 632.02 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 636.55 | 632.23 | 631.85 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 11:15:00 | 629.70 | 631.37 | 631.50 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 10:15:00 | 634.90 | 631.93 | 631.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 14:15:00 | 637.00 | 632.85 | 632.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 09:15:00 | 631.75 | 633.45 | 632.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 631.75 | 633.45 | 632.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 631.75 | 633.45 | 632.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:30:00 | 631.95 | 633.45 | 632.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 632.40 | 633.24 | 632.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:30:00 | 632.00 | 633.24 | 632.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 12:15:00 | 632.10 | 632.76 | 632.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-18 13:45:00 | 633.65 | 632.69 | 632.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-18 15:15:00 | 630.65 | 632.06 | 632.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-05-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 15:15:00 | 630.65 | 632.06 | 632.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 625.00 | 630.65 | 631.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 09:15:00 | 613.65 | 613.52 | 618.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-23 09:45:00 | 612.85 | 613.52 | 618.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 620.25 | 615.74 | 617.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 15:00:00 | 620.25 | 615.74 | 617.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 621.00 | 616.79 | 618.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:15:00 | 620.45 | 616.79 | 618.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 620.80 | 618.09 | 618.55 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 12:15:00 | 621.10 | 619.16 | 618.98 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 10:15:00 | 616.50 | 618.57 | 618.86 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-05-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 13:15:00 | 621.35 | 619.22 | 619.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 15:15:00 | 623.45 | 620.17 | 619.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 09:15:00 | 617.60 | 619.66 | 619.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 09:15:00 | 617.60 | 619.66 | 619.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 617.60 | 619.66 | 619.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:30:00 | 618.25 | 619.66 | 619.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 622.50 | 620.22 | 619.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 11:15:00 | 624.40 | 620.22 | 619.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 14:15:00 | 645.90 | 646.34 | 646.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 14:15:00 | 645.90 | 646.34 | 646.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 15:15:00 | 645.00 | 646.07 | 646.24 | Break + close below crossover candle low |

### Cycle 10 — BUY (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 09:15:00 | 666.20 | 650.10 | 648.05 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 13:15:00 | 656.90 | 659.77 | 659.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 14:15:00 | 652.90 | 658.40 | 659.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 11:15:00 | 660.65 | 656.36 | 657.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 11:15:00 | 660.65 | 656.36 | 657.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 11:15:00 | 660.65 | 656.36 | 657.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 12:00:00 | 660.65 | 656.36 | 657.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 12:15:00 | 661.35 | 657.36 | 658.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 12:30:00 | 662.05 | 657.36 | 658.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 659.75 | 658.18 | 658.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 15:00:00 | 659.75 | 658.18 | 658.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 658.00 | 658.14 | 658.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-19 09:15:00 | 656.20 | 658.14 | 658.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 09:15:00 | 666.80 | 659.88 | 659.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 666.80 | 659.88 | 659.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 10:15:00 | 671.95 | 662.29 | 660.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 15:15:00 | 674.00 | 678.76 | 673.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 15:15:00 | 674.00 | 678.76 | 673.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 674.00 | 678.76 | 673.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:15:00 | 670.90 | 678.76 | 673.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 670.90 | 677.19 | 673.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:30:00 | 670.05 | 677.19 | 673.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 676.10 | 676.97 | 673.50 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 668.80 | 671.73 | 671.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 10:15:00 | 665.75 | 668.78 | 670.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 672.15 | 667.47 | 668.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 672.15 | 667.47 | 668.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 672.15 | 667.47 | 668.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:00:00 | 672.15 | 667.47 | 668.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 670.05 | 667.98 | 668.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:00:00 | 670.05 | 667.98 | 668.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 673.25 | 669.04 | 669.15 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 12:15:00 | 672.20 | 669.67 | 669.43 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 14:15:00 | 667.35 | 669.07 | 669.19 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 674.40 | 670.13 | 669.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 10:15:00 | 675.00 | 671.54 | 670.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 10:15:00 | 676.55 | 676.57 | 674.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 11:00:00 | 676.55 | 676.57 | 674.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 12:15:00 | 671.10 | 675.23 | 673.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 13:00:00 | 671.10 | 675.23 | 673.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 674.60 | 675.10 | 673.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 700.75 | 675.45 | 674.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 09:15:00 | 708.10 | 715.41 | 715.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 09:15:00 | 708.10 | 715.41 | 715.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 14:15:00 | 698.05 | 705.49 | 710.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 09:15:00 | 707.60 | 704.54 | 708.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 09:15:00 | 707.60 | 704.54 | 708.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 707.60 | 704.54 | 708.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 10:00:00 | 707.60 | 704.54 | 708.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 712.70 | 703.43 | 705.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:45:00 | 714.75 | 703.43 | 705.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 712.65 | 705.27 | 706.35 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 12:15:00 | 714.05 | 708.26 | 707.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 09:15:00 | 726.65 | 714.22 | 710.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 15:15:00 | 720.10 | 721.72 | 717.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 15:15:00 | 720.10 | 721.72 | 717.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 720.10 | 721.72 | 717.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:15:00 | 725.95 | 721.72 | 717.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:45:00 | 726.50 | 722.66 | 717.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 11:15:00 | 714.65 | 721.12 | 718.01 | SL hit (close<static) qty=1.00 sl=716.35 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 14:15:00 | 822.10 | 824.49 | 824.64 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 15:15:00 | 831.00 | 825.79 | 825.22 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 10:15:00 | 817.15 | 824.70 | 824.86 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 15:15:00 | 829.90 | 824.79 | 824.61 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 09:15:00 | 814.45 | 822.72 | 823.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 11:15:00 | 812.15 | 819.22 | 821.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 09:15:00 | 808.00 | 800.76 | 807.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 09:15:00 | 808.00 | 800.76 | 807.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 808.00 | 800.76 | 807.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 09:30:00 | 804.30 | 800.76 | 807.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 809.75 | 802.56 | 807.39 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 13:15:00 | 831.90 | 811.56 | 810.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 10:15:00 | 835.55 | 824.11 | 817.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 13:15:00 | 835.25 | 837.53 | 831.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 14:00:00 | 835.25 | 837.53 | 831.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 841.05 | 838.43 | 833.10 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 11:15:00 | 823.05 | 833.60 | 833.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 12:15:00 | 822.60 | 831.40 | 832.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 12:15:00 | 804.65 | 803.02 | 809.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 12:30:00 | 806.70 | 803.02 | 809.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 803.00 | 802.52 | 807.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 09:15:00 | 812.70 | 802.52 | 807.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 818.95 | 805.81 | 808.70 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 11:15:00 | 824.90 | 811.68 | 811.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 12:15:00 | 828.00 | 814.95 | 812.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 09:15:00 | 817.90 | 819.68 | 816.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 817.90 | 819.68 | 816.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 817.90 | 819.68 | 816.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:45:00 | 816.05 | 819.68 | 816.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 817.80 | 819.31 | 816.17 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 15:15:00 | 804.05 | 815.06 | 815.32 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 827.00 | 812.29 | 812.21 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 13:15:00 | 821.20 | 822.21 | 822.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 14:15:00 | 818.55 | 821.48 | 821.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-12 15:15:00 | 810.00 | 808.30 | 813.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 09:15:00 | 808.00 | 808.30 | 813.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 808.45 | 808.33 | 812.88 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 11:15:00 | 817.05 | 812.80 | 812.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 825.35 | 817.19 | 814.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 821.50 | 824.28 | 820.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 15:00:00 | 821.50 | 824.28 | 820.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 822.00 | 823.82 | 820.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 09:30:00 | 826.10 | 824.33 | 820.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 11:45:00 | 825.55 | 824.27 | 821.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-20 09:15:00 | 817.00 | 819.68 | 819.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 817.00 | 819.68 | 819.94 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 14:15:00 | 825.65 | 820.32 | 820.05 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 15:15:00 | 816.60 | 819.57 | 819.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 812.00 | 818.06 | 819.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 813.05 | 812.57 | 814.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 13:00:00 | 813.05 | 812.57 | 814.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 812.00 | 812.52 | 814.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 14:30:00 | 818.70 | 812.52 | 814.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 822.80 | 814.17 | 814.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:45:00 | 822.40 | 814.17 | 814.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 10:15:00 | 823.45 | 816.02 | 815.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 14:15:00 | 828.70 | 822.60 | 819.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 09:15:00 | 821.40 | 822.90 | 819.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 821.40 | 822.90 | 819.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 821.40 | 822.90 | 819.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 14:30:00 | 833.95 | 826.33 | 822.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 09:15:00 | 834.00 | 827.46 | 823.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-29 09:15:00 | 917.35 | 855.00 | 841.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 881.85 | 888.63 | 889.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 877.15 | 885.10 | 887.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 886.05 | 885.29 | 887.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 886.05 | 885.29 | 887.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 886.05 | 885.29 | 887.53 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 898.00 | 888.80 | 888.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 12:15:00 | 901.20 | 891.28 | 889.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-10 15:15:00 | 892.05 | 892.94 | 891.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 15:15:00 | 892.05 | 892.94 | 891.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 892.05 | 892.94 | 891.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 09:15:00 | 902.00 | 892.94 | 891.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 10:00:00 | 897.85 | 893.92 | 891.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 10:30:00 | 899.00 | 894.74 | 892.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 12:00:00 | 897.90 | 895.37 | 892.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 892.70 | 895.06 | 893.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 14:00:00 | 892.70 | 895.06 | 893.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 896.50 | 895.34 | 893.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 911.10 | 895.60 | 893.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 14:15:00 | 891.70 | 900.79 | 898.48 | SL hit (close<static) qty=1.00 sl=892.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 10:15:00 | 893.00 | 896.45 | 896.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 14:15:00 | 890.15 | 894.36 | 895.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 09:15:00 | 872.25 | 870.81 | 876.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 09:15:00 | 872.25 | 870.81 | 876.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 872.25 | 870.81 | 876.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:00:00 | 872.25 | 870.81 | 876.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 877.55 | 871.59 | 873.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:30:00 | 873.40 | 871.59 | 873.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 870.85 | 871.44 | 873.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 14:15:00 | 868.60 | 871.37 | 873.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:00:00 | 868.90 | 871.45 | 872.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 864.90 | 871.71 | 872.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 825.17 | 851.78 | 861.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 14:15:00 | 825.45 | 851.78 | 861.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 15:15:00 | 821.65 | 847.57 | 858.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 838.95 | 822.34 | 830.72 | SL hit (close>ema200) qty=0.50 sl=822.34 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 14:15:00 | 840.95 | 834.47 | 834.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 15:15:00 | 842.00 | 835.98 | 835.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 861.90 | 862.23 | 855.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 15:00:00 | 861.90 | 862.23 | 855.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 853.20 | 860.42 | 854.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 10:15:00 | 862.65 | 860.25 | 855.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 12:15:00 | 863.90 | 860.70 | 856.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 13:30:00 | 865.35 | 861.90 | 857.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 15:15:00 | 865.10 | 865.94 | 862.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 865.10 | 865.77 | 863.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 878.20 | 865.77 | 863.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 12:45:00 | 868.85 | 869.84 | 868.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-06 14:15:00 | 859.90 | 867.32 | 867.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 14:15:00 | 859.90 | 867.32 | 867.69 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 10:15:00 | 874.35 | 868.29 | 867.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 09:15:00 | 882.35 | 871.53 | 869.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 15:15:00 | 908.00 | 913.87 | 900.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 09:30:00 | 920.45 | 914.21 | 902.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 908.75 | 910.91 | 906.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:45:00 | 907.65 | 910.91 | 906.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 908.00 | 910.33 | 906.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:30:00 | 907.95 | 910.33 | 906.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 11:15:00 | 898.90 | 908.04 | 905.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-13 11:15:00 | 898.90 | 908.04 | 905.98 | SL hit (close<ema400) qty=1.00 sl=905.98 alert=retest1 |

### Cycle 41 — SELL (started 2023-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 13:15:00 | 896.85 | 903.36 | 904.06 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 10:15:00 | 915.60 | 906.42 | 905.28 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 14:15:00 | 894.50 | 903.97 | 904.54 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 15:15:00 | 909.00 | 904.45 | 904.09 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 10:15:00 | 900.00 | 903.13 | 903.53 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 911.40 | 904.78 | 904.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 12:15:00 | 913.80 | 906.58 | 905.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 13:15:00 | 920.85 | 921.27 | 915.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 15:15:00 | 918.00 | 919.77 | 915.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 918.00 | 919.77 | 915.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:15:00 | 924.25 | 919.77 | 915.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 10:30:00 | 919.60 | 919.34 | 916.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 11:30:00 | 919.10 | 919.57 | 916.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 13:45:00 | 922.00 | 920.01 | 917.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 921.00 | 921.67 | 918.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 09:15:00 | 935.60 | 921.67 | 918.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 10:30:00 | 931.05 | 925.26 | 920.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 15:15:00 | 924.00 | 934.09 | 935.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 15:15:00 | 924.00 | 934.09 | 935.42 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 949.60 | 937.19 | 936.71 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 14:15:00 | 934.50 | 942.44 | 942.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-01 12:15:00 | 928.15 | 935.11 | 938.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 09:15:00 | 921.60 | 918.44 | 925.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 921.60 | 918.44 | 925.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 921.60 | 918.44 | 925.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-05 10:00:00 | 921.60 | 918.44 | 925.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 920.00 | 918.75 | 924.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 09:45:00 | 915.70 | 918.46 | 922.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 11:15:00 | 915.05 | 918.02 | 921.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 14:15:00 | 914.95 | 916.58 | 919.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 09:15:00 | 909.00 | 898.49 | 898.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 909.00 | 898.49 | 898.26 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 11:15:00 | 888.65 | 897.64 | 897.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 12:15:00 | 886.00 | 895.31 | 896.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 09:15:00 | 873.80 | 873.45 | 880.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 11:15:00 | 864.70 | 872.41 | 879.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 864.70 | 872.41 | 879.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-15 11:15:00 | 858.00 | 867.63 | 873.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-18 09:15:00 | 890.30 | 873.59 | 873.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 09:15:00 | 890.30 | 873.59 | 873.59 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 09:15:00 | 873.40 | 881.08 | 881.66 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 882.20 | 879.13 | 879.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 908.00 | 887.70 | 883.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 10:15:00 | 903.05 | 903.14 | 895.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 11:00:00 | 903.05 | 903.14 | 895.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 12:15:00 | 901.95 | 906.65 | 902.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 13:00:00 | 901.95 | 906.65 | 902.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 13:15:00 | 903.15 | 905.95 | 902.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:45:00 | 913.55 | 907.71 | 904.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:30:00 | 917.10 | 908.88 | 907.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 14:15:00 | 913.10 | 916.02 | 914.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 14:45:00 | 916.20 | 915.14 | 914.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 15:15:00 | 919.00 | 915.91 | 915.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:15:00 | 920.50 | 915.91 | 915.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 925.55 | 917.84 | 915.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 13:00:00 | 927.15 | 919.68 | 917.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 13:45:00 | 927.95 | 921.14 | 918.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 09:15:00 | 926.50 | 922.03 | 919.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 11:30:00 | 926.50 | 924.13 | 920.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 928.05 | 927.75 | 924.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:30:00 | 930.65 | 927.75 | 924.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 923.15 | 927.03 | 924.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 12:00:00 | 923.15 | 927.03 | 924.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 922.35 | 926.09 | 924.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-08 15:15:00 | 913.70 | 921.75 | 922.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 15:15:00 | 913.70 | 921.75 | 922.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 908.05 | 914.92 | 918.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 14:15:00 | 910.70 | 909.40 | 913.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 14:15:00 | 910.70 | 909.40 | 913.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 14:15:00 | 910.70 | 909.40 | 913.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 15:00:00 | 910.70 | 909.40 | 913.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 910.80 | 909.58 | 913.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:45:00 | 914.45 | 909.58 | 913.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 914.50 | 910.57 | 913.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 11:00:00 | 914.50 | 910.57 | 913.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 11:15:00 | 918.15 | 912.08 | 913.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 12:00:00 | 918.15 | 912.08 | 913.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 12:15:00 | 917.90 | 913.25 | 914.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 13:15:00 | 918.35 | 913.25 | 914.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-01-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 14:15:00 | 917.65 | 914.73 | 914.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 10:15:00 | 923.05 | 917.81 | 916.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 13:15:00 | 919.00 | 919.57 | 917.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 13:15:00 | 919.00 | 919.57 | 917.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 13:15:00 | 919.00 | 919.57 | 917.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 13:45:00 | 918.30 | 919.57 | 917.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 920.00 | 919.65 | 918.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 14:45:00 | 919.15 | 919.65 | 918.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 15:15:00 | 921.00 | 919.92 | 918.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 09:15:00 | 923.40 | 919.92 | 918.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 10:15:00 | 925.00 | 920.08 | 918.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 12:15:00 | 914.95 | 918.76 | 918.27 | SL hit (close<static) qty=1.00 sl=917.50 alert=retest2 |

### Cycle 57 — SELL (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 13:15:00 | 912.95 | 917.60 | 917.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 15:15:00 | 907.00 | 914.39 | 916.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 909.25 | 899.29 | 902.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 909.25 | 899.29 | 902.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 909.25 | 899.29 | 902.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:45:00 | 909.35 | 899.29 | 902.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 894.70 | 898.37 | 902.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 13:00:00 | 892.45 | 896.86 | 900.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 10:00:00 | 892.90 | 893.70 | 897.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 11:30:00 | 891.60 | 892.45 | 896.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 13:30:00 | 893.95 | 893.19 | 896.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 893.25 | 893.42 | 895.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:45:00 | 894.60 | 893.42 | 895.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 887.85 | 884.24 | 888.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:00:00 | 887.85 | 884.24 | 888.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 892.10 | 885.81 | 888.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:45:00 | 891.55 | 885.81 | 888.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 892.40 | 887.13 | 889.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 13:45:00 | 886.35 | 887.10 | 888.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 14:30:00 | 889.55 | 888.47 | 889.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 09:45:00 | 886.80 | 888.60 | 889.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:30:00 | 885.00 | 888.42 | 889.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 885.90 | 887.92 | 888.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:30:00 | 888.05 | 887.92 | 888.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 900.00 | 888.36 | 888.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 905.05 | 891.70 | 890.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 905.05 | 891.70 | 890.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 10:15:00 | 922.20 | 897.80 | 892.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 917.45 | 921.10 | 912.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 917.45 | 921.10 | 912.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 917.45 | 921.10 | 912.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:30:00 | 914.35 | 921.10 | 912.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 914.00 | 921.24 | 914.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 11:45:00 | 937.90 | 921.36 | 915.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 13:15:00 | 907.00 | 914.20 | 914.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 907.00 | 914.20 | 914.59 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 10:15:00 | 925.70 | 915.95 | 915.10 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 13:15:00 | 907.00 | 913.78 | 914.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 14:15:00 | 898.50 | 910.72 | 912.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 893.90 | 892.73 | 900.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 893.90 | 892.73 | 900.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 893.90 | 892.73 | 900.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:30:00 | 897.80 | 892.73 | 900.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 897.95 | 888.84 | 893.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 10:00:00 | 897.95 | 888.84 | 893.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 894.75 | 890.02 | 893.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:30:00 | 891.45 | 890.31 | 893.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 12:00:00 | 891.50 | 890.31 | 893.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 14:15:00 | 907.15 | 894.82 | 894.84 | SL hit (close>static) qty=1.00 sl=898.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 15:15:00 | 912.00 | 898.26 | 896.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 09:15:00 | 923.90 | 903.39 | 898.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 925.55 | 928.63 | 916.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 925.55 | 928.63 | 916.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 925.55 | 928.63 | 916.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:45:00 | 920.15 | 928.63 | 916.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 916.30 | 926.16 | 916.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 916.30 | 926.16 | 916.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 917.00 | 924.33 | 916.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 09:15:00 | 939.05 | 917.49 | 915.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 10:30:00 | 927.85 | 922.09 | 917.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-12 15:15:00 | 906.00 | 921.57 | 919.81 | SL hit (close<static) qty=1.00 sl=914.30 alert=retest2 |

### Cycle 63 — SELL (started 2024-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 09:15:00 | 901.90 | 919.51 | 920.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 10:15:00 | 900.00 | 915.61 | 918.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 09:15:00 | 882.05 | 868.76 | 881.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 882.05 | 868.76 | 881.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 882.05 | 868.76 | 881.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:00:00 | 882.05 | 868.76 | 881.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 893.00 | 873.61 | 882.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:00:00 | 893.00 | 873.61 | 882.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 909.75 | 880.84 | 885.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 11:30:00 | 913.50 | 880.84 | 885.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 889.00 | 885.52 | 886.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:15:00 | 876.85 | 885.52 | 886.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 871.00 | 882.62 | 884.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-19 10:30:00 | 868.50 | 878.90 | 883.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 15:00:00 | 867.40 | 860.20 | 860.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 15:15:00 | 870.90 | 862.34 | 861.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 870.90 | 862.34 | 861.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 871.75 | 864.22 | 862.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 11:15:00 | 865.00 | 865.07 | 863.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-23 11:45:00 | 864.70 | 865.07 | 863.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 870.05 | 877.44 | 873.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:00:00 | 870.05 | 877.44 | 873.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 870.05 | 875.96 | 873.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:30:00 | 870.10 | 875.96 | 873.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 870.80 | 874.30 | 873.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:45:00 | 870.95 | 874.30 | 873.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 870.15 | 873.47 | 872.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:00:00 | 870.15 | 873.47 | 872.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 15:15:00 | 868.00 | 872.30 | 872.34 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 877.15 | 873.27 | 872.78 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 868.45 | 872.31 | 872.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 865.70 | 870.99 | 871.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 13:15:00 | 875.05 | 871.39 | 871.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 13:15:00 | 875.05 | 871.39 | 871.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 875.05 | 871.39 | 871.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 14:00:00 | 875.05 | 871.39 | 871.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 14:15:00 | 872.35 | 871.58 | 871.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 14:30:00 | 875.45 | 871.58 | 871.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 870.00 | 871.27 | 871.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 09:15:00 | 871.20 | 871.27 | 871.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 870.55 | 871.12 | 871.58 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 11:15:00 | 875.95 | 872.23 | 872.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 14:15:00 | 883.75 | 874.52 | 873.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 11:15:00 | 876.10 | 876.76 | 874.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 11:15:00 | 876.10 | 876.76 | 874.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 876.10 | 876.76 | 874.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 15:00:00 | 886.00 | 878.28 | 875.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 09:15:00 | 882.05 | 878.58 | 877.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 10:00:00 | 893.40 | 881.55 | 878.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 12:00:00 | 882.75 | 884.94 | 882.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 888.95 | 885.74 | 883.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 13:15:00 | 890.00 | 885.74 | 883.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 15:00:00 | 890.00 | 887.43 | 884.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 09:15:00 | 879.75 | 885.51 | 884.20 | SL hit (close<static) qty=1.00 sl=880.20 alert=retest2 |

### Cycle 69 — SELL (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 11:15:00 | 880.60 | 896.65 | 897.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 845.70 | 869.89 | 882.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 856.90 | 849.92 | 865.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 09:45:00 | 853.15 | 849.92 | 865.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 857.70 | 851.48 | 865.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 857.70 | 851.48 | 865.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 857.45 | 852.67 | 864.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:45:00 | 861.00 | 852.67 | 864.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 858.00 | 854.67 | 862.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 858.00 | 854.67 | 862.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 857.15 | 855.17 | 861.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:45:00 | 849.90 | 853.33 | 860.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 14:15:00 | 849.95 | 847.13 | 850.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-26 12:15:00 | 847.45 | 843.93 | 843.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 12:15:00 | 847.45 | 843.93 | 843.50 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 10:15:00 | 842.15 | 843.46 | 843.47 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 11:15:00 | 854.10 | 843.15 | 842.85 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 15:15:00 | 836.10 | 842.59 | 842.82 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 847.65 | 843.60 | 843.26 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 11:15:00 | 838.80 | 846.43 | 847.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 15:15:00 | 838.10 | 842.58 | 844.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 851.60 | 844.38 | 845.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 851.60 | 844.38 | 845.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 851.60 | 844.38 | 845.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:00:00 | 851.60 | 844.38 | 845.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 854.70 | 846.45 | 846.35 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 09:15:00 | 843.55 | 848.42 | 848.67 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 847.30 | 846.18 | 846.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 11:15:00 | 849.95 | 846.94 | 846.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 15:15:00 | 845.00 | 847.53 | 847.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 15:15:00 | 845.00 | 847.53 | 847.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 845.00 | 847.53 | 847.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 831.00 | 847.53 | 847.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 842.70 | 846.56 | 846.62 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 12:15:00 | 852.00 | 846.99 | 846.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-15 14:15:00 | 855.00 | 849.07 | 847.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 14:15:00 | 870.45 | 871.56 | 864.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-18 15:00:00 | 870.45 | 871.56 | 864.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 872.00 | 871.65 | 865.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 862.90 | 871.65 | 865.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 865.55 | 870.43 | 865.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 10:15:00 | 874.90 | 870.43 | 865.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 11:00:00 | 872.00 | 870.74 | 865.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 11:30:00 | 872.00 | 878.49 | 875.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 15:15:00 | 870.05 | 876.39 | 876.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 15:15:00 | 870.05 | 876.39 | 876.86 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 10:15:00 | 883.90 | 878.40 | 877.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 887.40 | 883.59 | 881.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 894.10 | 896.29 | 893.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 14:15:00 | 894.10 | 896.29 | 893.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 894.10 | 896.29 | 893.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 894.10 | 896.29 | 893.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 895.00 | 896.03 | 893.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 11:30:00 | 899.65 | 895.09 | 893.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 15:15:00 | 892.00 | 892.74 | 892.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 15:15:00 | 892.00 | 892.74 | 892.82 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 898.00 | 893.80 | 893.29 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 10:15:00 | 884.85 | 892.01 | 892.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 13:15:00 | 876.45 | 885.75 | 889.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 09:15:00 | 876.80 | 875.13 | 879.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 876.80 | 875.13 | 879.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 876.80 | 875.13 | 879.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:00:00 | 876.80 | 875.13 | 879.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 866.20 | 873.34 | 878.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 11:15:00 | 856.40 | 873.34 | 878.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 13:15:00 | 864.05 | 869.87 | 875.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 863.30 | 870.65 | 874.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 11:00:00 | 863.85 | 869.43 | 871.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 857.85 | 857.06 | 861.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 857.55 | 857.06 | 861.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 851.30 | 849.11 | 853.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 858.15 | 849.11 | 853.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 856.65 | 850.62 | 853.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 858.45 | 850.62 | 853.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 862.40 | 852.98 | 854.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:15:00 | 866.35 | 852.98 | 854.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-14 11:15:00 | 870.35 | 856.45 | 855.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 870.35 | 856.45 | 855.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 875.65 | 862.52 | 858.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 878.30 | 880.16 | 874.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 15:15:00 | 877.50 | 879.90 | 875.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 877.50 | 879.90 | 875.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:30:00 | 872.00 | 878.69 | 875.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 876.95 | 878.34 | 875.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:15:00 | 879.50 | 878.34 | 875.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 14:15:00 | 887.95 | 893.54 | 894.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 14:15:00 | 887.95 | 893.54 | 894.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 15:15:00 | 883.40 | 891.51 | 893.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 12:15:00 | 889.20 | 887.89 | 890.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 12:15:00 | 889.20 | 887.89 | 890.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 889.20 | 887.89 | 890.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:00:00 | 889.20 | 887.89 | 890.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 888.00 | 885.00 | 888.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 889.05 | 885.00 | 888.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 888.20 | 885.97 | 888.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:30:00 | 886.60 | 885.97 | 888.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 884.15 | 885.61 | 887.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:00:00 | 880.95 | 884.67 | 887.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:15:00 | 883.25 | 885.08 | 886.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 11:15:00 | 896.15 | 887.29 | 887.36 | SL hit (close>static) qty=1.00 sl=888.20 alert=retest2 |

### Cycle 88 — BUY (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 12:15:00 | 895.50 | 888.93 | 888.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 13:15:00 | 902.40 | 891.63 | 889.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 09:15:00 | 909.40 | 911.54 | 907.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 909.40 | 911.54 | 907.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 909.40 | 911.54 | 907.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 909.40 | 911.54 | 907.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 910.60 | 911.35 | 907.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:30:00 | 908.80 | 911.35 | 907.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 913.75 | 914.19 | 910.96 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 899.60 | 909.04 | 909.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 15:15:00 | 871.00 | 902.01 | 906.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 919.55 | 905.52 | 907.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 919.55 | 905.52 | 907.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 919.55 | 905.52 | 907.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 919.55 | 905.52 | 907.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 946.10 | 913.64 | 910.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 976.00 | 932.30 | 922.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 13:15:00 | 1015.00 | 1018.60 | 993.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:00:00 | 1015.00 | 1018.60 | 993.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 1015.75 | 1018.93 | 1001.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:00:00 | 1015.75 | 1018.93 | 1001.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 999.90 | 1013.90 | 1002.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:30:00 | 999.25 | 1013.90 | 1002.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 998.70 | 1010.86 | 1002.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:00:00 | 998.70 | 1010.86 | 1002.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1013.80 | 1015.87 | 1011.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:45:00 | 1011.40 | 1015.87 | 1011.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 1010.55 | 1014.34 | 1011.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:45:00 | 1013.30 | 1013.96 | 1011.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:30:00 | 1014.15 | 1013.84 | 1011.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:45:00 | 1014.00 | 1016.08 | 1012.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:00:00 | 1013.50 | 1015.00 | 1014.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1030.55 | 1017.63 | 1015.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:45:00 | 1044.00 | 1021.10 | 1017.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:15:00 | 1038.95 | 1021.10 | 1017.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 13:30:00 | 1037.50 | 1029.55 | 1022.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 1035.70 | 1041.97 | 1036.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1034.00 | 1040.38 | 1036.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:30:00 | 1034.95 | 1040.38 | 1036.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1031.05 | 1038.51 | 1035.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 1031.05 | 1038.51 | 1035.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 1033.95 | 1037.23 | 1035.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:00:00 | 1033.95 | 1037.23 | 1035.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1032.75 | 1036.33 | 1035.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:30:00 | 1032.70 | 1036.33 | 1035.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1030.00 | 1035.07 | 1035.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1047.20 | 1035.07 | 1035.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 10:15:00 | 1033.25 | 1038.15 | 1038.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 1033.25 | 1038.15 | 1038.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 11:15:00 | 1021.85 | 1034.89 | 1036.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 13:15:00 | 1032.90 | 1020.52 | 1025.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 13:15:00 | 1032.90 | 1020.52 | 1025.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 1032.90 | 1020.52 | 1025.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:45:00 | 1022.95 | 1020.52 | 1025.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 1040.00 | 1024.42 | 1027.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:30:00 | 1051.05 | 1024.42 | 1027.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 1047.15 | 1031.46 | 1029.95 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 11:15:00 | 1028.40 | 1033.09 | 1033.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 1020.20 | 1029.36 | 1031.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 1031.15 | 1029.66 | 1031.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 1031.15 | 1029.66 | 1031.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1031.15 | 1029.66 | 1031.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:15:00 | 1051.00 | 1029.66 | 1031.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1039.00 | 1031.53 | 1031.81 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 1045.70 | 1034.36 | 1033.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 1054.20 | 1040.15 | 1036.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 1043.00 | 1050.87 | 1043.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 11:15:00 | 1043.00 | 1050.87 | 1043.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1043.00 | 1050.87 | 1043.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 1043.00 | 1050.87 | 1043.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 1033.25 | 1047.35 | 1042.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 1030.25 | 1047.35 | 1042.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1042.75 | 1046.43 | 1042.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:30:00 | 1047.80 | 1045.70 | 1042.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:30:00 | 1047.20 | 1044.08 | 1042.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:00:00 | 1044.95 | 1046.73 | 1044.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 1025.05 | 1042.51 | 1043.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 1025.05 | 1042.51 | 1043.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 12:15:00 | 1020.05 | 1033.06 | 1036.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 1028.75 | 1024.60 | 1030.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 1028.75 | 1024.60 | 1030.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1028.75 | 1024.60 | 1030.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 1028.75 | 1024.60 | 1030.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 1028.50 | 1024.82 | 1029.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:30:00 | 1030.95 | 1024.82 | 1029.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 1027.60 | 1025.37 | 1029.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:45:00 | 1029.65 | 1025.37 | 1029.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 1030.90 | 1026.48 | 1029.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 1030.90 | 1026.48 | 1029.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 1032.85 | 1027.75 | 1029.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 1030.85 | 1027.75 | 1029.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 1044.90 | 1031.18 | 1030.93 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 1029.45 | 1033.92 | 1033.99 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 1038.10 | 1034.41 | 1033.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 1048.70 | 1038.28 | 1035.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 1037.15 | 1039.07 | 1036.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 11:15:00 | 1037.15 | 1039.07 | 1036.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 1037.15 | 1039.07 | 1036.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 1037.15 | 1039.07 | 1036.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 1035.60 | 1038.38 | 1036.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 13:30:00 | 1039.15 | 1038.80 | 1037.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 09:15:00 | 1014.80 | 1035.39 | 1036.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 1014.80 | 1035.39 | 1036.03 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 12:15:00 | 1033.70 | 1023.49 | 1022.79 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1009.00 | 1022.34 | 1022.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 1005.65 | 1019.00 | 1021.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 1012.85 | 1011.76 | 1015.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 1012.85 | 1011.76 | 1015.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1012.85 | 1011.76 | 1015.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 1012.85 | 1011.76 | 1015.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1012.10 | 1011.83 | 1015.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 1017.10 | 1011.83 | 1015.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1016.60 | 1012.79 | 1015.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 1016.70 | 1012.79 | 1015.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1015.45 | 1013.32 | 1015.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 1018.90 | 1013.32 | 1015.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 1015.85 | 1013.82 | 1015.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 1015.85 | 1013.82 | 1015.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 1016.00 | 1014.26 | 1015.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 1016.10 | 1014.26 | 1015.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 1027.70 | 1016.95 | 1016.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 1032.00 | 1019.96 | 1017.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 1100.00 | 1101.59 | 1084.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:45:00 | 1099.40 | 1101.59 | 1084.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1122.10 | 1126.03 | 1119.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 1128.70 | 1123.29 | 1119.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:00:00 | 1126.90 | 1124.01 | 1120.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:45:00 | 1127.50 | 1124.07 | 1120.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 11:30:00 | 1127.60 | 1124.45 | 1121.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 1119.45 | 1125.71 | 1122.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:30:00 | 1124.70 | 1125.71 | 1122.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 1127.95 | 1126.16 | 1123.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 1087.45 | 1126.16 | 1123.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1104.65 | 1121.86 | 1121.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 1104.65 | 1121.86 | 1121.44 | SL hit (close<static) qty=1.00 sl=1118.55 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1098.00 | 1117.08 | 1119.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 1096.00 | 1112.87 | 1117.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1116.70 | 1100.52 | 1107.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1116.70 | 1100.52 | 1107.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1116.70 | 1100.52 | 1107.85 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 1125.80 | 1111.93 | 1111.79 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 1101.55 | 1111.55 | 1111.74 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 1135.25 | 1116.38 | 1113.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 11:15:00 | 1148.95 | 1122.89 | 1117.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 14:15:00 | 1122.15 | 1129.53 | 1122.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 14:15:00 | 1122.15 | 1129.53 | 1122.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1122.15 | 1129.53 | 1122.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 1122.15 | 1129.53 | 1122.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 1120.00 | 1127.63 | 1121.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 1149.80 | 1127.63 | 1121.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 12:15:00 | 1172.80 | 1189.15 | 1190.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 12:15:00 | 1172.80 | 1189.15 | 1190.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 13:15:00 | 1166.25 | 1184.57 | 1187.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1198.60 | 1181.87 | 1185.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1198.60 | 1181.87 | 1185.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1198.60 | 1181.87 | 1185.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 1197.35 | 1181.87 | 1185.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 1213.10 | 1188.11 | 1187.88 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 09:15:00 | 1191.70 | 1194.69 | 1194.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 10:15:00 | 1187.00 | 1193.15 | 1194.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 12:15:00 | 1192.55 | 1190.50 | 1192.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 12:15:00 | 1192.55 | 1190.50 | 1192.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 1192.55 | 1190.50 | 1192.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 1193.45 | 1190.50 | 1192.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 1203.10 | 1193.02 | 1193.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:45:00 | 1205.00 | 1193.02 | 1193.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 14:15:00 | 1209.55 | 1196.33 | 1194.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 15:15:00 | 1212.60 | 1199.58 | 1196.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1212.40 | 1227.22 | 1218.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 1212.40 | 1227.22 | 1218.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1212.40 | 1227.22 | 1218.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 1212.40 | 1227.22 | 1218.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1211.00 | 1223.97 | 1218.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:15:00 | 1209.55 | 1223.97 | 1218.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 1224.95 | 1221.61 | 1218.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 14:45:00 | 1263.60 | 1228.30 | 1221.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-29 09:15:00 | 1389.96 | 1310.12 | 1291.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 11:15:00 | 1449.45 | 1456.65 | 1457.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 12:15:00 | 1440.15 | 1453.35 | 1455.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 14:15:00 | 1453.35 | 1450.93 | 1454.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 1453.35 | 1450.93 | 1454.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1453.35 | 1450.93 | 1454.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1453.35 | 1450.93 | 1454.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1435.60 | 1447.86 | 1452.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 1449.80 | 1447.86 | 1452.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1448.80 | 1448.05 | 1452.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:30:00 | 1424.25 | 1435.72 | 1443.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 11:15:00 | 1353.04 | 1412.80 | 1431.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-13 13:15:00 | 1417.10 | 1408.50 | 1425.99 | SL hit (close>ema200) qty=0.50 sl=1408.50 alert=retest2 |

### Cycle 112 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 1438.00 | 1398.45 | 1397.14 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 10:15:00 | 1388.50 | 1400.69 | 1400.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 11:15:00 | 1384.45 | 1397.44 | 1399.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 15:15:00 | 1321.45 | 1314.81 | 1333.85 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 1304.65 | 1311.13 | 1330.45 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1279.80 | 1268.48 | 1277.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-30 09:15:00 | 1279.80 | 1268.48 | 1277.04 | SL hit (close>ema400) qty=1.00 sl=1277.04 alert=retest1 |

### Cycle 114 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 11:15:00 | 1316.70 | 1283.62 | 1282.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 12:15:00 | 1328.00 | 1292.49 | 1286.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 14:15:00 | 1325.20 | 1330.32 | 1315.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 15:00:00 | 1325.20 | 1330.32 | 1315.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 1321.00 | 1337.06 | 1324.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:45:00 | 1325.45 | 1337.06 | 1324.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1318.95 | 1333.44 | 1324.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 1318.95 | 1333.44 | 1324.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1321.85 | 1331.12 | 1324.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:00:00 | 1321.85 | 1331.12 | 1324.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1315.00 | 1327.90 | 1323.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 1332.95 | 1332.06 | 1325.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 1338.70 | 1347.36 | 1339.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 1338.70 | 1347.36 | 1339.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 1344.20 | 1346.73 | 1339.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 12:00:00 | 1344.20 | 1346.73 | 1339.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 1333.20 | 1344.02 | 1339.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 1333.20 | 1344.02 | 1339.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 1335.15 | 1342.25 | 1338.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:45:00 | 1330.55 | 1342.25 | 1338.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1316.95 | 1337.19 | 1336.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 15:00:00 | 1316.95 | 1337.19 | 1336.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 15:15:00 | 1322.75 | 1334.30 | 1335.52 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 1372.55 | 1343.34 | 1339.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1421.90 | 1369.76 | 1354.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 1403.10 | 1407.22 | 1388.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:30:00 | 1400.30 | 1407.22 | 1388.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 1390.80 | 1400.83 | 1391.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 1395.85 | 1400.83 | 1391.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1400.90 | 1400.85 | 1391.98 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 1383.55 | 1390.32 | 1390.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 11:15:00 | 1380.50 | 1388.36 | 1389.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1393.75 | 1380.78 | 1384.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1393.75 | 1380.78 | 1384.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1393.75 | 1380.78 | 1384.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:45:00 | 1393.00 | 1380.78 | 1384.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1387.50 | 1382.13 | 1384.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 11:45:00 | 1381.95 | 1380.84 | 1383.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 1384.90 | 1376.88 | 1380.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 10:15:00 | 1379.65 | 1376.88 | 1380.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 1312.85 | 1352.83 | 1362.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 1315.65 | 1352.83 | 1362.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 15:15:00 | 1368.00 | 1349.12 | 1355.20 | SL hit (close>ema200) qty=0.50 sl=1349.12 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 1297.20 | 1286.50 | 1285.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 15:15:00 | 1335.15 | 1296.23 | 1289.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 1293.75 | 1295.73 | 1290.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 1293.75 | 1295.73 | 1290.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1293.75 | 1295.73 | 1290.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:30:00 | 1323.05 | 1298.75 | 1294.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 12:30:00 | 1313.90 | 1304.32 | 1297.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:00:00 | 1314.65 | 1304.32 | 1297.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 14:15:00 | 1314.00 | 1305.55 | 1298.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1311.20 | 1321.16 | 1311.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1311.20 | 1321.16 | 1311.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1312.25 | 1319.38 | 1311.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 1324.95 | 1317.17 | 1312.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 1298.50 | 1312.91 | 1311.10 | SL hit (close<static) qty=1.00 sl=1306.95 alert=retest2 |

### Cycle 119 — SELL (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 11:15:00 | 1297.90 | 1307.67 | 1308.89 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 1313.10 | 1309.23 | 1309.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1323.00 | 1312.10 | 1310.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 13:15:00 | 1307.55 | 1314.50 | 1312.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 13:15:00 | 1307.55 | 1314.50 | 1312.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 1307.55 | 1314.50 | 1312.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 1307.55 | 1314.50 | 1312.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 1305.10 | 1312.62 | 1311.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 1305.10 | 1312.62 | 1311.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 15:15:00 | 1304.00 | 1310.89 | 1311.14 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 1306.35 | 1305.38 | 1305.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 14:15:00 | 1325.40 | 1311.38 | 1308.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 1317.05 | 1347.01 | 1334.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 1317.05 | 1347.01 | 1334.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1317.05 | 1347.01 | 1334.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:45:00 | 1324.70 | 1347.01 | 1334.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1333.20 | 1344.25 | 1333.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 10:00:00 | 1354.50 | 1339.14 | 1334.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 1347.70 | 1344.92 | 1339.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 11:15:00 | 1348.75 | 1344.19 | 1340.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-25 14:15:00 | 1482.47 | 1417.91 | 1396.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 1423.90 | 1448.80 | 1449.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 1418.75 | 1442.79 | 1446.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1456.00 | 1437.83 | 1442.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 1456.00 | 1437.83 | 1442.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1456.00 | 1437.83 | 1442.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 1456.00 | 1437.83 | 1442.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1447.30 | 1439.72 | 1442.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 13:00:00 | 1437.65 | 1439.31 | 1442.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 1488.65 | 1447.92 | 1445.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 1488.65 | 1447.92 | 1445.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 1510.40 | 1475.18 | 1465.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 1457.30 | 1471.60 | 1464.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 1457.30 | 1471.60 | 1464.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1457.30 | 1471.60 | 1464.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 1457.30 | 1471.60 | 1464.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1466.05 | 1470.49 | 1464.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:15:00 | 1472.95 | 1470.49 | 1464.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:45:00 | 1470.65 | 1471.12 | 1465.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 1476.00 | 1469.46 | 1465.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 1444.40 | 1462.83 | 1463.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 1444.40 | 1462.83 | 1463.44 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 1468.65 | 1458.34 | 1458.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 14:15:00 | 1505.10 | 1479.03 | 1469.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 13:15:00 | 1500.00 | 1502.49 | 1487.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 13:45:00 | 1499.00 | 1502.49 | 1487.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 1518.80 | 1519.37 | 1503.33 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1467.85 | 1505.64 | 1507.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 1456.90 | 1495.89 | 1502.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 09:15:00 | 1411.15 | 1405.99 | 1420.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 10:00:00 | 1411.15 | 1405.99 | 1420.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1337.95 | 1340.64 | 1353.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 1335.20 | 1340.64 | 1353.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 1362.55 | 1351.16 | 1351.29 | SL hit (close>static) qty=1.00 sl=1358.90 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 1353.35 | 1351.35 | 1351.34 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 1346.45 | 1350.37 | 1350.89 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1383.70 | 1356.97 | 1353.80 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 1330.40 | 1359.52 | 1361.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 11:15:00 | 1327.55 | 1346.29 | 1354.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1302.60 | 1289.68 | 1303.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:00:00 | 1302.60 | 1289.68 | 1303.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1297.15 | 1291.17 | 1302.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:30:00 | 1291.55 | 1291.17 | 1302.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1300.25 | 1292.99 | 1302.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1290.35 | 1299.24 | 1302.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:00:00 | 1290.25 | 1297.44 | 1301.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:15:00 | 1295.80 | 1294.56 | 1299.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 13:45:00 | 1295.80 | 1295.25 | 1299.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1296.00 | 1295.40 | 1298.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1296.00 | 1295.40 | 1298.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1298.30 | 1295.98 | 1298.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 1293.05 | 1295.98 | 1298.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 1309.55 | 1299.50 | 1299.90 | SL hit (close>static) qty=1.00 sl=1304.30 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 1304.45 | 1300.49 | 1300.31 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 1291.00 | 1300.51 | 1300.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 1278.45 | 1296.10 | 1298.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 1294.75 | 1289.15 | 1293.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 15:15:00 | 1294.75 | 1289.15 | 1293.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 1294.75 | 1289.15 | 1293.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 1275.10 | 1289.15 | 1293.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:30:00 | 1279.45 | 1271.92 | 1272.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 11:15:00 | 1297.15 | 1277.08 | 1274.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 1297.15 | 1277.08 | 1274.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 1301.55 | 1284.83 | 1279.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 12:15:00 | 1296.35 | 1297.41 | 1288.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-16 13:15:00 | 1298.45 | 1297.41 | 1288.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 1297.00 | 1297.84 | 1291.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 1286.00 | 1294.98 | 1290.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 1288.60 | 1293.70 | 1290.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 1283.70 | 1293.70 | 1290.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 1294.30 | 1293.82 | 1290.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 1286.85 | 1293.82 | 1290.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 1295.90 | 1294.24 | 1291.13 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 1273.40 | 1288.38 | 1289.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 13:15:00 | 1270.45 | 1281.55 | 1285.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 15:15:00 | 1289.20 | 1282.33 | 1285.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 15:15:00 | 1289.20 | 1282.33 | 1285.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 1289.20 | 1282.33 | 1285.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:00:00 | 1273.45 | 1280.56 | 1283.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 14:15:00 | 1209.78 | 1228.34 | 1240.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 10:15:00 | 1146.11 | 1200.98 | 1224.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 136 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 1200.65 | 1178.73 | 1177.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 1213.30 | 1189.46 | 1182.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 14:15:00 | 1186.75 | 1190.29 | 1184.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 15:00:00 | 1186.75 | 1190.29 | 1184.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 1202.10 | 1192.65 | 1186.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:30:00 | 1218.20 | 1203.60 | 1191.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:30:00 | 1213.75 | 1219.18 | 1209.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:00:00 | 1217.40 | 1218.82 | 1209.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:30:00 | 1234.30 | 1220.25 | 1211.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1240.00 | 1231.40 | 1219.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 15:15:00 | 1250.00 | 1235.18 | 1225.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 1291.55 | 1229.42 | 1227.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-05 10:15:00 | 1340.02 | 1265.88 | 1245.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 10:15:00 | 1341.05 | 1390.15 | 1395.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1329.15 | 1370.24 | 1384.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 1248.00 | 1243.78 | 1262.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:45:00 | 1246.35 | 1243.78 | 1262.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1219.00 | 1242.56 | 1257.47 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 1270.25 | 1254.58 | 1253.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 1274.25 | 1258.52 | 1255.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 1266.05 | 1273.22 | 1266.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 12:15:00 | 1266.05 | 1273.22 | 1266.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 1266.05 | 1273.22 | 1266.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 1266.05 | 1273.22 | 1266.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 1271.70 | 1272.92 | 1266.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 15:00:00 | 1277.75 | 1273.89 | 1267.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 10:30:00 | 1280.30 | 1274.08 | 1269.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 1277.70 | 1275.22 | 1270.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:30:00 | 1277.10 | 1277.01 | 1272.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 1260.00 | 1273.61 | 1271.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-21 15:15:00 | 1260.00 | 1273.61 | 1271.33 | SL hit (close<static) qty=1.00 sl=1266.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 1261.20 | 1270.90 | 1271.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 1257.60 | 1268.24 | 1269.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 10:15:00 | 1269.15 | 1267.48 | 1269.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 10:15:00 | 1269.15 | 1267.48 | 1269.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1269.15 | 1267.48 | 1269.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:00:00 | 1269.15 | 1267.48 | 1269.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 1274.45 | 1268.87 | 1269.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:00:00 | 1274.45 | 1268.87 | 1269.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 1253.15 | 1265.73 | 1268.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:00:00 | 1251.65 | 1262.91 | 1266.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 12:15:00 | 1189.07 | 1223.98 | 1243.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 15:15:00 | 1220.05 | 1219.28 | 1236.05 | SL hit (close>ema200) qty=0.50 sl=1219.28 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1225.00 | 1194.76 | 1192.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 1234.75 | 1207.84 | 1199.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 1262.90 | 1263.41 | 1243.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:00:00 | 1262.90 | 1263.41 | 1243.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1278.90 | 1272.53 | 1260.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 12:30:00 | 1283.15 | 1275.97 | 1264.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:15:00 | 1283.20 | 1275.97 | 1264.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 1282.95 | 1277.25 | 1269.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 09:15:00 | 1243.30 | 1267.58 | 1267.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 1243.30 | 1267.58 | 1267.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 1229.30 | 1257.18 | 1262.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 1261.45 | 1256.07 | 1261.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 13:15:00 | 1261.45 | 1256.07 | 1261.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 1261.45 | 1256.07 | 1261.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:00:00 | 1261.45 | 1256.07 | 1261.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1270.05 | 1258.87 | 1262.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1254.80 | 1258.68 | 1261.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:00:00 | 1254.55 | 1257.86 | 1261.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 1250.00 | 1253.20 | 1257.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 1252.70 | 1253.10 | 1257.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1247.20 | 1251.92 | 1256.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-17 12:15:00 | 1268.70 | 1258.41 | 1257.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 12:15:00 | 1268.70 | 1258.41 | 1257.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1281.95 | 1264.23 | 1260.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 14:15:00 | 1364.40 | 1368.26 | 1338.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 1364.40 | 1368.26 | 1338.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1323.95 | 1357.59 | 1338.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:00:00 | 1323.95 | 1357.59 | 1338.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 1338.15 | 1353.70 | 1338.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:00:00 | 1349.55 | 1352.87 | 1339.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 10:45:00 | 1348.20 | 1357.52 | 1356.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 11:15:00 | 1347.50 | 1357.52 | 1356.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 12:00:00 | 1345.85 | 1355.19 | 1355.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 1334.80 | 1351.11 | 1353.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 1334.80 | 1351.11 | 1353.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 1326.50 | 1342.15 | 1347.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 1355.30 | 1328.88 | 1336.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 1355.30 | 1328.88 | 1336.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1355.30 | 1328.88 | 1336.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 1351.95 | 1328.88 | 1336.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1376.85 | 1338.47 | 1340.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 1376.85 | 1338.47 | 1340.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 1358.15 | 1342.41 | 1341.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 1389.25 | 1356.66 | 1348.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 1401.20 | 1402.62 | 1383.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 1401.20 | 1402.62 | 1383.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1401.20 | 1402.62 | 1383.84 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 12:15:00 | 1372.00 | 1383.56 | 1384.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 11:15:00 | 1368.95 | 1374.91 | 1378.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 1304.15 | 1283.61 | 1304.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 11:15:00 | 1304.15 | 1283.61 | 1304.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 1304.15 | 1283.61 | 1304.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 1304.15 | 1283.61 | 1304.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1327.10 | 1292.31 | 1306.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:45:00 | 1328.15 | 1292.31 | 1306.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1318.00 | 1297.44 | 1307.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1289.30 | 1302.60 | 1308.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 1305.60 | 1304.74 | 1307.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:00:00 | 1306.80 | 1305.15 | 1307.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 14:15:00 | 1353.95 | 1314.86 | 1311.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 1353.95 | 1314.86 | 1311.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 1378.80 | 1350.39 | 1333.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 1367.60 | 1372.84 | 1358.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 09:45:00 | 1367.30 | 1372.84 | 1358.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 1373.00 | 1372.87 | 1360.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 11:15:00 | 1380.10 | 1372.87 | 1360.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 12:00:00 | 1381.80 | 1374.66 | 1362.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 1376.60 | 1383.86 | 1373.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 1380.20 | 1382.11 | 1373.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1362.90 | 1379.04 | 1375.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:45:00 | 1394.40 | 1381.42 | 1377.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 12:30:00 | 1401.60 | 1385.42 | 1380.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-24 09:15:00 | 1518.11 | 1464.38 | 1432.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 14:15:00 | 1464.20 | 1477.04 | 1477.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 1452.90 | 1469.57 | 1474.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1497.80 | 1455.67 | 1461.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1497.80 | 1455.67 | 1461.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1497.80 | 1455.67 | 1461.42 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 1493.30 | 1469.38 | 1467.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 1517.90 | 1488.58 | 1478.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 1489.80 | 1499.92 | 1490.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 1489.80 | 1499.92 | 1490.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1489.80 | 1499.92 | 1490.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 1489.80 | 1499.92 | 1490.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1487.10 | 1497.36 | 1490.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 12:15:00 | 1496.90 | 1496.09 | 1490.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 1464.20 | 1488.14 | 1488.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 1464.20 | 1488.14 | 1488.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 1417.80 | 1468.29 | 1478.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 1467.50 | 1459.73 | 1469.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 1467.50 | 1459.73 | 1469.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1467.50 | 1459.73 | 1469.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1467.50 | 1459.73 | 1469.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1476.00 | 1462.98 | 1470.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1479.80 | 1462.98 | 1470.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1493.10 | 1469.01 | 1472.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 1484.30 | 1469.01 | 1472.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1479.20 | 1471.04 | 1472.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:30:00 | 1488.70 | 1471.04 | 1472.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1471.80 | 1470.71 | 1472.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1471.80 | 1470.71 | 1472.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1465.00 | 1469.57 | 1471.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 1422.80 | 1469.57 | 1471.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 1474.00 | 1457.56 | 1460.92 | SL hit (close>static) qty=1.00 sl=1473.70 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1456.90 | 1447.20 | 1446.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 1477.80 | 1453.32 | 1449.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1458.40 | 1466.24 | 1458.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1458.40 | 1466.24 | 1458.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1458.40 | 1466.24 | 1458.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 1454.60 | 1466.24 | 1458.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1476.80 | 1468.35 | 1460.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:30:00 | 1460.70 | 1468.35 | 1460.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1470.30 | 1476.42 | 1468.66 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1454.00 | 1465.02 | 1466.14 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1475.70 | 1461.86 | 1461.78 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1457.60 | 1461.01 | 1461.40 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 1469.10 | 1462.63 | 1462.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1525.00 | 1475.90 | 1468.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 12:15:00 | 1474.90 | 1485.33 | 1475.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 12:15:00 | 1474.90 | 1485.33 | 1475.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1474.90 | 1485.33 | 1475.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:00:00 | 1474.90 | 1485.33 | 1475.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 1504.20 | 1489.10 | 1478.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 1563.90 | 1512.36 | 1499.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1534.30 | 1510.13 | 1505.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1520.40 | 1516.42 | 1509.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1558.90 | 1586.25 | 1588.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 1558.90 | 1586.25 | 1588.42 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 12:15:00 | 1608.90 | 1584.23 | 1582.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1621.00 | 1606.34 | 1598.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 1629.20 | 1636.72 | 1622.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 1629.20 | 1636.72 | 1622.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 1625.50 | 1633.03 | 1623.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:30:00 | 1628.80 | 1633.03 | 1623.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 1632.80 | 1632.99 | 1624.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:30:00 | 1622.50 | 1632.99 | 1624.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1628.80 | 1633.22 | 1626.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 1622.20 | 1633.22 | 1626.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1632.00 | 1632.97 | 1627.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 1634.30 | 1632.97 | 1627.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:45:00 | 1634.90 | 1633.69 | 1628.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:15:00 | 1633.90 | 1633.53 | 1628.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-12 10:15:00 | 1797.73 | 1738.80 | 1701.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 1778.50 | 1788.75 | 1790.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1767.90 | 1784.58 | 1788.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1613.80 | 1610.64 | 1637.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-25 10:15:00 | 1623.00 | 1610.64 | 1637.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1625.70 | 1618.49 | 1634.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 1634.00 | 1618.49 | 1634.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1629.00 | 1623.26 | 1632.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 1626.60 | 1623.26 | 1632.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1617.60 | 1622.13 | 1631.50 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 1676.40 | 1638.24 | 1633.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 15:15:00 | 1695.00 | 1667.02 | 1651.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 1663.00 | 1668.12 | 1654.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 10:15:00 | 1663.00 | 1668.12 | 1654.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1663.00 | 1668.12 | 1654.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 1663.00 | 1668.12 | 1654.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1690.60 | 1672.61 | 1657.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:30:00 | 1660.10 | 1672.61 | 1657.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1663.20 | 1673.55 | 1665.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1663.20 | 1673.55 | 1665.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1653.00 | 1669.44 | 1664.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 1653.00 | 1669.44 | 1664.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1685.00 | 1666.91 | 1664.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1657.40 | 1666.91 | 1664.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 1640.60 | 1661.64 | 1662.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 1637.20 | 1656.76 | 1659.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1658.00 | 1642.39 | 1649.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1658.00 | 1642.39 | 1649.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1658.00 | 1642.39 | 1649.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 1656.80 | 1642.39 | 1649.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1667.20 | 1647.35 | 1650.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:15:00 | 1670.00 | 1647.35 | 1650.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 1702.40 | 1658.36 | 1655.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 15:15:00 | 1715.00 | 1688.94 | 1672.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1720.00 | 1727.14 | 1712.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1720.00 | 1727.14 | 1712.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1720.00 | 1727.14 | 1712.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1720.00 | 1727.14 | 1712.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1713.30 | 1724.37 | 1712.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:30:00 | 1727.40 | 1716.26 | 1712.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 1724.80 | 1726.72 | 1721.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:30:00 | 1721.80 | 1726.66 | 1725.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 12:15:00 | 1714.20 | 1724.17 | 1724.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1714.20 | 1724.17 | 1724.87 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 1741.60 | 1726.28 | 1725.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 1760.90 | 1735.99 | 1729.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 1770.10 | 1770.23 | 1758.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:15:00 | 1774.40 | 1770.23 | 1758.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1764.90 | 1768.95 | 1760.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 1764.90 | 1768.95 | 1760.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1776.30 | 1771.73 | 1763.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 1776.30 | 1771.73 | 1763.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1770.20 | 1774.12 | 1767.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 1773.30 | 1774.12 | 1767.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1771.10 | 1773.51 | 1767.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 1769.80 | 1773.51 | 1767.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1757.40 | 1770.29 | 1767.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 1757.40 | 1770.29 | 1767.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1754.50 | 1767.13 | 1765.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:15:00 | 1758.70 | 1767.13 | 1765.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 1754.00 | 1764.49 | 1764.89 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 1769.80 | 1765.42 | 1765.24 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 1763.10 | 1765.05 | 1765.11 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 14:15:00 | 1768.70 | 1765.39 | 1765.23 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 1763.50 | 1765.01 | 1765.07 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1766.50 | 1765.31 | 1765.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 1791.90 | 1773.84 | 1769.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 12:15:00 | 1769.60 | 1780.06 | 1774.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 12:15:00 | 1769.60 | 1780.06 | 1774.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 1769.60 | 1780.06 | 1774.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 1769.60 | 1780.06 | 1774.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 1770.00 | 1778.04 | 1773.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 1770.00 | 1778.04 | 1773.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1790.00 | 1776.10 | 1773.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1814.90 | 1794.76 | 1787.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 1763.70 | 1790.21 | 1788.45 | SL hit (close<static) qty=1.00 sl=1772.60 alert=retest2 |

### Cycle 169 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 1778.20 | 1785.96 | 1786.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 1762.50 | 1777.14 | 1782.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 1791.60 | 1774.87 | 1778.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 14:15:00 | 1791.60 | 1774.87 | 1778.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1791.60 | 1774.87 | 1778.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1791.60 | 1774.87 | 1778.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1795.00 | 1778.90 | 1780.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1813.80 | 1778.90 | 1780.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1816.50 | 1786.42 | 1783.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 1827.80 | 1794.70 | 1787.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1811.20 | 1813.73 | 1800.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 1811.20 | 1813.73 | 1800.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1798.90 | 1810.84 | 1801.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:30:00 | 1823.70 | 1808.56 | 1802.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:00:00 | 1831.40 | 1811.10 | 1805.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:45:00 | 1827.50 | 1815.46 | 1808.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 1771.00 | 1804.56 | 1805.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 1771.00 | 1804.56 | 1805.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1697.50 | 1775.64 | 1786.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 1719.90 | 1701.99 | 1723.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 1719.90 | 1701.99 | 1723.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1722.10 | 1706.01 | 1723.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 1714.00 | 1707.61 | 1722.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 13:15:00 | 1732.90 | 1715.00 | 1723.32 | SL hit (close>static) qty=1.00 sl=1729.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1785.80 | 1701.12 | 1692.86 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 11:15:00 | 1762.90 | 1786.77 | 1789.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 1739.40 | 1761.72 | 1767.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1714.80 | 1712.51 | 1721.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1714.80 | 1712.51 | 1721.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1714.80 | 1712.51 | 1721.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 1699.20 | 1708.00 | 1716.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 1725.10 | 1712.13 | 1716.46 | SL hit (close>static) qty=1.00 sl=1722.60 alert=retest2 |

### Cycle 174 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 1665.00 | 1650.23 | 1650.09 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1640.50 | 1649.28 | 1650.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1634.90 | 1641.53 | 1644.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1647.10 | 1634.45 | 1637.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1647.10 | 1634.45 | 1637.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1647.10 | 1634.45 | 1637.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1647.10 | 1634.45 | 1637.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1637.70 | 1635.10 | 1637.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 1630.10 | 1634.28 | 1636.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:45:00 | 1630.00 | 1633.44 | 1636.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 1606.80 | 1595.47 | 1594.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1606.80 | 1595.47 | 1594.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1623.90 | 1603.73 | 1598.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 09:15:00 | 1605.20 | 1608.23 | 1601.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 1605.20 | 1608.23 | 1601.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1605.20 | 1608.23 | 1601.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 1605.20 | 1608.23 | 1601.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1603.10 | 1607.20 | 1602.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 1607.80 | 1607.32 | 1602.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 1592.20 | 1604.91 | 1602.69 | SL hit (close<static) qty=1.00 sl=1593.80 alert=retest2 |

### Cycle 177 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1590.80 | 1599.67 | 1600.73 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1607.20 | 1601.35 | 1601.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 1617.50 | 1604.58 | 1602.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 1591.70 | 1606.35 | 1604.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 13:15:00 | 1591.70 | 1606.35 | 1604.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1591.70 | 1606.35 | 1604.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1591.70 | 1606.35 | 1604.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1599.70 | 1605.02 | 1604.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 1592.30 | 1605.02 | 1604.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1589.90 | 1602.00 | 1602.82 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 1611.40 | 1599.10 | 1598.76 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1597.40 | 1598.76 | 1598.77 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 1602.00 | 1599.41 | 1599.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 1606.50 | 1600.83 | 1599.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 13:15:00 | 1598.30 | 1601.39 | 1600.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 1598.30 | 1601.39 | 1600.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1598.30 | 1601.39 | 1600.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:30:00 | 1600.00 | 1601.39 | 1600.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1608.20 | 1602.75 | 1600.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 1600.00 | 1602.75 | 1600.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1594.90 | 1602.34 | 1601.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 1592.40 | 1602.34 | 1601.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1593.10 | 1600.49 | 1600.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:15:00 | 1584.20 | 1600.49 | 1600.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1578.20 | 1596.03 | 1598.41 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1604.20 | 1598.80 | 1598.18 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 1586.80 | 1596.76 | 1597.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 1577.90 | 1591.20 | 1594.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 13:15:00 | 1588.20 | 1586.40 | 1590.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 13:15:00 | 1588.20 | 1586.40 | 1590.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1588.20 | 1586.40 | 1590.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 1588.20 | 1586.40 | 1590.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1587.30 | 1586.58 | 1590.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 1587.60 | 1586.58 | 1590.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1589.00 | 1587.06 | 1590.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 1579.80 | 1587.06 | 1590.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 1595.50 | 1589.51 | 1590.54 | SL hit (close>static) qty=1.00 sl=1592.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1599.00 | 1591.40 | 1591.31 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1571.00 | 1589.02 | 1590.42 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 1604.90 | 1593.47 | 1591.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 1606.00 | 1597.64 | 1594.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 13:15:00 | 1580.60 | 1594.24 | 1592.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1580.60 | 1594.24 | 1592.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1580.60 | 1594.24 | 1592.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1581.60 | 1594.24 | 1592.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1586.90 | 1592.77 | 1592.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 1588.30 | 1592.77 | 1592.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1588.00 | 1591.81 | 1592.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 1588.00 | 1591.81 | 1592.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 1583.80 | 1589.39 | 1590.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 1590.50 | 1589.61 | 1590.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1590.50 | 1589.61 | 1590.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1590.50 | 1589.61 | 1590.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1590.50 | 1589.61 | 1590.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1584.60 | 1588.61 | 1589.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 15:15:00 | 1575.00 | 1587.71 | 1589.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 1593.40 | 1586.81 | 1588.57 | SL hit (close>static) qty=1.00 sl=1591.90 alert=retest2 |

### Cycle 190 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1602.50 | 1591.27 | 1590.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1627.00 | 1598.42 | 1593.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 1605.00 | 1606.38 | 1599.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:00:00 | 1605.00 | 1606.38 | 1599.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1607.60 | 1606.63 | 1599.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1620.40 | 1610.26 | 1603.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:45:00 | 1621.90 | 1617.76 | 1614.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1618.00 | 1618.20 | 1615.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1596.80 | 1611.55 | 1612.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1596.80 | 1611.55 | 1612.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1590.90 | 1607.42 | 1610.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1610.60 | 1602.35 | 1606.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1610.60 | 1602.35 | 1606.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1610.60 | 1602.35 | 1606.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1610.60 | 1602.35 | 1606.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1620.80 | 1606.04 | 1608.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 1620.80 | 1606.04 | 1608.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1592.80 | 1605.49 | 1607.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:15:00 | 1591.70 | 1600.98 | 1604.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 1589.20 | 1597.97 | 1601.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:00:00 | 1592.20 | 1594.00 | 1598.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:30:00 | 1591.40 | 1588.87 | 1592.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1594.20 | 1589.94 | 1592.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:00:00 | 1584.20 | 1588.62 | 1591.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 1584.30 | 1588.38 | 1590.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1579.00 | 1585.10 | 1586.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 1584.20 | 1586.19 | 1586.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1574.00 | 1583.75 | 1585.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1561.60 | 1580.63 | 1583.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1512.12 | 1550.85 | 1566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1509.74 | 1550.85 | 1566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1512.59 | 1550.85 | 1566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1511.83 | 1550.85 | 1566.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1540.00 | 1536.68 | 1550.75 | SL hit (close>ema200) qty=0.50 sl=1536.68 alert=retest2 |

### Cycle 192 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 1560.40 | 1552.75 | 1552.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 1566.90 | 1555.58 | 1554.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1553.90 | 1556.75 | 1554.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1553.90 | 1556.75 | 1554.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1553.90 | 1556.75 | 1554.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1553.90 | 1556.75 | 1554.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1566.00 | 1558.60 | 1555.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 1571.80 | 1561.50 | 1557.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:45:00 | 1586.30 | 1570.70 | 1565.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1616.30 | 1644.06 | 1644.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 1616.30 | 1644.06 | 1644.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 10:15:00 | 1612.30 | 1637.71 | 1641.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 1622.30 | 1621.36 | 1630.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1622.30 | 1621.36 | 1630.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1622.30 | 1621.36 | 1630.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1611.50 | 1621.36 | 1630.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:30:00 | 1605.70 | 1615.84 | 1624.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 1609.00 | 1614.47 | 1623.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 1530.92 | 1544.69 | 1553.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 1528.55 | 1544.69 | 1553.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1608.00 | 1552.66 | 1553.88 | SL hit (close>ema200) qty=0.50 sl=1552.66 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 1606.00 | 1563.33 | 1558.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 1640.00 | 1586.29 | 1570.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 09:15:00 | 1629.00 | 1634.44 | 1601.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:30:00 | 1624.80 | 1634.44 | 1601.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1593.00 | 1616.98 | 1606.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1582.00 | 1616.98 | 1606.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1593.10 | 1612.21 | 1605.20 | EMA400 retest candle locked (from upside) |

### Cycle 195 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 1582.70 | 1599.12 | 1600.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 1572.00 | 1587.56 | 1593.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1576.10 | 1574.99 | 1582.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:45:00 | 1578.60 | 1574.99 | 1582.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1581.70 | 1576.56 | 1581.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 1581.70 | 1576.56 | 1581.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1582.20 | 1577.68 | 1581.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:15:00 | 1592.00 | 1577.68 | 1581.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1592.00 | 1580.55 | 1582.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 1579.40 | 1580.55 | 1582.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1584.00 | 1581.24 | 1582.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:45:00 | 1574.20 | 1578.99 | 1581.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1571.40 | 1564.41 | 1563.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1571.40 | 1564.41 | 1563.89 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 1552.10 | 1563.44 | 1564.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 1550.80 | 1555.64 | 1559.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1505.00 | 1504.77 | 1516.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 1503.00 | 1504.77 | 1516.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1508.10 | 1501.97 | 1511.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1508.10 | 1501.97 | 1511.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1506.10 | 1503.67 | 1510.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:15:00 | 1514.30 | 1503.67 | 1510.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1514.30 | 1505.79 | 1510.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 1499.70 | 1505.79 | 1510.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1501.90 | 1505.01 | 1509.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 1492.80 | 1500.90 | 1507.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1520.00 | 1507.88 | 1507.93 | SL hit (close>static) qty=1.00 sl=1517.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1528.20 | 1511.95 | 1509.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 1533.20 | 1520.08 | 1514.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 1550.00 | 1550.23 | 1542.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 1550.00 | 1550.23 | 1542.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1558.50 | 1553.40 | 1547.63 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 1532.10 | 1543.17 | 1544.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 1522.60 | 1534.99 | 1539.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1482.60 | 1481.99 | 1495.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 1482.60 | 1481.99 | 1495.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1416.80 | 1393.09 | 1407.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1420.00 | 1393.09 | 1407.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1415.80 | 1397.63 | 1407.83 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1440.00 | 1417.58 | 1414.87 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 1382.60 | 1410.59 | 1411.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 1360.90 | 1385.45 | 1396.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 12:15:00 | 1351.20 | 1349.14 | 1365.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 13:00:00 | 1351.20 | 1349.14 | 1365.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1350.00 | 1350.45 | 1361.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1338.10 | 1350.45 | 1361.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 1382.10 | 1360.45 | 1358.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 1382.10 | 1360.45 | 1358.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 15:15:00 | 1385.00 | 1370.52 | 1363.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1374.70 | 1383.31 | 1376.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1374.70 | 1383.31 | 1376.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1374.70 | 1383.31 | 1376.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:15:00 | 1395.10 | 1376.07 | 1374.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 1398.10 | 1413.34 | 1414.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 1398.10 | 1413.34 | 1414.04 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 1426.90 | 1414.74 | 1414.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 1430.80 | 1417.95 | 1415.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 11:15:00 | 1451.80 | 1458.16 | 1444.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 12:00:00 | 1451.80 | 1458.16 | 1444.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1452.20 | 1460.14 | 1451.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:45:00 | 1484.20 | 1471.94 | 1461.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 1440.60 | 1472.85 | 1474.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 1440.60 | 1472.85 | 1474.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 1414.10 | 1461.10 | 1469.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 1346.20 | 1343.21 | 1363.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 1346.20 | 1343.21 | 1363.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1347.90 | 1339.91 | 1347.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 1347.00 | 1339.91 | 1347.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1341.70 | 1340.27 | 1347.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 1333.00 | 1342.04 | 1346.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:00:00 | 1335.50 | 1340.73 | 1345.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 1334.50 | 1340.15 | 1344.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1349.10 | 1342.66 | 1344.49 | SL hit (close>static) qty=1.00 sl=1348.20 alert=retest2 |

### Cycle 206 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 1358.00 | 1345.73 | 1345.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 1359.50 | 1348.49 | 1346.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 09:15:00 | 1373.00 | 1373.42 | 1362.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:15:00 | 1371.30 | 1373.42 | 1362.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1384.90 | 1379.07 | 1371.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 1376.60 | 1379.07 | 1371.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1360.00 | 1375.36 | 1370.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 1363.10 | 1375.36 | 1370.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 1358.00 | 1371.89 | 1369.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 1360.00 | 1371.89 | 1369.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 1361.90 | 1367.73 | 1368.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1354.30 | 1362.69 | 1364.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 1364.70 | 1358.58 | 1361.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 1364.70 | 1358.58 | 1361.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 1364.70 | 1358.58 | 1361.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 1364.70 | 1358.58 | 1361.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1367.20 | 1360.30 | 1362.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 15:00:00 | 1367.20 | 1360.30 | 1362.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1363.00 | 1360.84 | 1362.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 1343.40 | 1360.84 | 1362.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 1352.40 | 1359.05 | 1361.31 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 10:15:00 | 1365.00 | 1361.71 | 1361.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 1371.10 | 1364.62 | 1362.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 09:15:00 | 1362.70 | 1365.09 | 1363.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 1362.70 | 1365.09 | 1363.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1362.70 | 1365.09 | 1363.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 1361.70 | 1365.09 | 1363.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1363.10 | 1364.70 | 1363.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 1364.60 | 1364.70 | 1363.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1366.70 | 1365.10 | 1363.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:45:00 | 1364.70 | 1365.10 | 1363.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1365.00 | 1365.35 | 1364.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:45:00 | 1365.00 | 1365.35 | 1364.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1362.30 | 1364.74 | 1363.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 1362.30 | 1364.74 | 1363.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1359.00 | 1363.59 | 1363.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 1320.80 | 1363.59 | 1363.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1319.80 | 1354.83 | 1359.51 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 1358.00 | 1348.37 | 1348.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1358.40 | 1350.37 | 1349.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 1350.50 | 1353.45 | 1351.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 13:15:00 | 1350.50 | 1353.45 | 1351.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1350.50 | 1353.45 | 1351.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 1350.50 | 1353.45 | 1351.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1349.90 | 1352.74 | 1351.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1349.90 | 1352.74 | 1351.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1349.70 | 1352.13 | 1351.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1342.70 | 1352.13 | 1351.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1339.40 | 1349.58 | 1349.99 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 15:15:00 | 1352.00 | 1350.01 | 1349.79 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1338.70 | 1347.75 | 1348.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 1335.30 | 1345.26 | 1347.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1298.30 | 1295.51 | 1307.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 10:00:00 | 1298.30 | 1295.51 | 1307.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 1310.90 | 1301.31 | 1305.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 1310.90 | 1301.31 | 1305.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1301.00 | 1301.25 | 1305.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 1296.10 | 1301.19 | 1304.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 11:00:00 | 1299.40 | 1301.19 | 1304.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1261.20 | 1288.02 | 1292.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1310.10 | 1285.43 | 1283.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1310.10 | 1285.43 | 1283.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1335.90 | 1312.74 | 1298.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1316.50 | 1322.55 | 1307.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 1304.20 | 1316.58 | 1308.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1304.20 | 1316.58 | 1308.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 1304.20 | 1316.58 | 1308.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 1334.90 | 1320.24 | 1310.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 1341.20 | 1324.43 | 1313.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 1289.90 | 1319.26 | 1314.04 | SL hit (close<static) qty=1.00 sl=1303.90 alert=retest2 |

### Cycle 215 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 1289.00 | 1308.21 | 1309.61 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1320.40 | 1305.83 | 1305.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 1334.80 | 1311.63 | 1307.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 1299.50 | 1312.48 | 1309.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 1299.50 | 1312.48 | 1309.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1299.50 | 1312.48 | 1309.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 1298.20 | 1312.48 | 1309.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1310.40 | 1312.06 | 1309.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:30:00 | 1317.20 | 1312.91 | 1309.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 1317.90 | 1313.63 | 1310.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 1334.50 | 1313.63 | 1310.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 1325.90 | 1317.14 | 1313.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1315.80 | 1318.32 | 1315.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:00:00 | 1315.80 | 1318.32 | 1315.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 1313.20 | 1317.29 | 1314.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:00:00 | 1313.20 | 1317.29 | 1314.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 1316.40 | 1317.11 | 1314.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:30:00 | 1310.00 | 1317.11 | 1314.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 1316.00 | 1316.89 | 1315.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:45:00 | 1330.80 | 1320.13 | 1316.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 1448.92 | 1403.65 | 1382.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 1428.70 | 1440.33 | 1441.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 1370.90 | 1421.51 | 1432.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 10:15:00 | 1368.70 | 1366.25 | 1385.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:30:00 | 1352.30 | 1363.30 | 1382.13 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1394.00 | 1370.79 | 1380.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 1394.00 | 1370.79 | 1380.92 | SL hit (close>ema400) qty=1.00 sl=1380.92 alert=retest1 |

### Cycle 218 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1352.60 | 1336.41 | 1335.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 1356.50 | 1347.19 | 1341.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 1388.70 | 1389.39 | 1380.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 1385.60 | 1389.39 | 1380.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-18 13:45:00 | 633.65 | 2023-05-18 15:15:00 | 630.65 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2023-05-26 11:15:00 | 624.40 | 2023-06-09 14:15:00 | 645.90 | STOP_HIT | 1.00 | 3.44% |
| SELL | retest2 | 2023-06-19 09:15:00 | 656.20 | 2023-06-19 09:15:00 | 666.80 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-06-30 09:15:00 | 700.75 | 2023-07-12 09:15:00 | 708.10 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2023-07-18 09:15:00 | 725.95 | 2023-07-18 11:15:00 | 714.65 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2023-07-18 09:45:00 | 726.50 | 2023-07-18 11:15:00 | 714.65 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2023-07-18 13:45:00 | 725.65 | 2023-07-31 14:15:00 | 798.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-18 09:30:00 | 826.10 | 2023-09-20 09:15:00 | 817.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-09-18 11:45:00 | 825.55 | 2023-09-20 09:15:00 | 817.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-09-27 14:30:00 | 833.95 | 2023-09-29 09:15:00 | 917.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-28 09:15:00 | 834.00 | 2023-09-29 09:15:00 | 917.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-11 09:15:00 | 902.00 | 2023-10-12 14:15:00 | 891.70 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-10-11 10:00:00 | 897.85 | 2023-10-12 15:15:00 | 887.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-10-11 10:30:00 | 899.00 | 2023-10-12 15:15:00 | 887.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-10-11 12:00:00 | 897.90 | 2023-10-12 15:15:00 | 887.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-10-12 09:15:00 | 911.10 | 2023-10-12 15:15:00 | 887.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2023-10-13 09:30:00 | 899.10 | 2023-10-13 10:15:00 | 893.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2023-10-19 14:15:00 | 868.60 | 2023-10-23 14:15:00 | 825.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:00:00 | 868.90 | 2023-10-23 14:15:00 | 825.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 09:15:00 | 864.90 | 2023-10-23 15:15:00 | 821.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 14:15:00 | 868.60 | 2023-10-27 09:15:00 | 838.95 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2023-10-20 12:00:00 | 868.90 | 2023-10-27 09:15:00 | 838.95 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2023-10-23 09:15:00 | 864.90 | 2023-10-27 09:15:00 | 838.95 | STOP_HIT | 0.50 | 3.00% |
| BUY | retest2 | 2023-11-01 10:15:00 | 862.65 | 2023-11-06 14:15:00 | 859.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2023-11-01 12:15:00 | 863.90 | 2023-11-06 14:15:00 | 859.90 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2023-11-01 13:30:00 | 865.35 | 2023-11-06 14:15:00 | 859.90 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-11-02 15:15:00 | 865.10 | 2023-11-06 14:15:00 | 859.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-11-03 09:15:00 | 878.20 | 2023-11-06 14:15:00 | 859.90 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2023-11-06 12:45:00 | 868.85 | 2023-11-06 14:15:00 | 859.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest1 | 2023-11-10 09:30:00 | 920.45 | 2023-11-13 11:15:00 | 898.90 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2023-11-21 09:15:00 | 924.25 | 2023-11-24 15:15:00 | 924.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2023-11-21 10:30:00 | 919.60 | 2023-11-24 15:15:00 | 924.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2023-11-21 11:30:00 | 919.10 | 2023-11-24 15:15:00 | 924.00 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2023-11-21 13:45:00 | 922.00 | 2023-11-24 15:15:00 | 924.00 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2023-11-22 09:15:00 | 935.60 | 2023-11-24 15:15:00 | 924.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-11-22 10:30:00 | 931.05 | 2023-11-24 15:15:00 | 924.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-12-06 09:45:00 | 915.70 | 2023-12-12 09:15:00 | 909.00 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2023-12-06 11:15:00 | 915.05 | 2023-12-12 09:15:00 | 909.00 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2023-12-06 14:15:00 | 914.95 | 2023-12-12 09:15:00 | 909.00 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2023-12-15 11:15:00 | 858.00 | 2023-12-18 09:15:00 | 890.30 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2023-12-29 11:45:00 | 913.55 | 2024-01-08 15:15:00 | 913.70 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2024-01-02 09:30:00 | 917.10 | 2024-01-08 15:15:00 | 913.70 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-01-03 14:15:00 | 913.10 | 2024-01-08 15:15:00 | 913.70 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-01-03 14:45:00 | 916.20 | 2024-01-08 15:15:00 | 913.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-01-04 13:00:00 | 927.15 | 2024-01-08 15:15:00 | 913.70 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-01-04 13:45:00 | 927.95 | 2024-01-08 15:15:00 | 913.70 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-01-05 09:15:00 | 926.50 | 2024-01-08 15:15:00 | 913.70 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-01-05 11:30:00 | 926.50 | 2024-01-08 15:15:00 | 913.70 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-01-16 09:15:00 | 923.40 | 2024-01-16 12:15:00 | 914.95 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-01-16 10:15:00 | 925.00 | 2024-01-16 12:15:00 | 914.95 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-01-19 13:00:00 | 892.45 | 2024-01-29 09:15:00 | 905.05 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-01-20 10:00:00 | 892.90 | 2024-01-29 09:15:00 | 905.05 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-01-20 11:30:00 | 891.60 | 2024-01-29 09:15:00 | 905.05 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-01-20 13:30:00 | 893.95 | 2024-01-29 09:15:00 | 905.05 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-01-24 13:45:00 | 886.35 | 2024-01-29 09:15:00 | 905.05 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-01-24 14:30:00 | 889.55 | 2024-01-29 09:15:00 | 905.05 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-01-25 09:45:00 | 886.80 | 2024-01-29 09:15:00 | 905.05 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-01-25 10:30:00 | 885.00 | 2024-01-29 09:15:00 | 905.05 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-01-31 11:45:00 | 937.90 | 2024-02-01 13:15:00 | 907.00 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-02-07 11:30:00 | 891.45 | 2024-02-07 14:15:00 | 907.15 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-02-07 12:00:00 | 891.50 | 2024-02-07 14:15:00 | 907.15 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-02-12 09:15:00 | 939.05 | 2024-02-12 15:15:00 | 906.00 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2024-02-12 10:30:00 | 927.85 | 2024-02-12 15:15:00 | 906.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-02-13 10:00:00 | 924.35 | 2024-02-14 09:15:00 | 901.90 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-02-13 12:30:00 | 927.05 | 2024-02-14 09:15:00 | 901.90 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-02-19 10:30:00 | 868.50 | 2024-02-22 15:15:00 | 870.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-02-22 15:00:00 | 867.40 | 2024-02-22 15:15:00 | 870.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-03-01 15:00:00 | 886.00 | 2024-03-06 09:15:00 | 879.75 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-03-04 09:15:00 | 882.05 | 2024-03-06 09:15:00 | 879.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-03-04 10:00:00 | 893.40 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-03-05 12:00:00 | 882.75 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-03-05 13:15:00 | 890.00 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-03-05 15:00:00 | 890.00 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-03-06 12:30:00 | 901.00 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-03-07 10:15:00 | 889.60 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-03-11 09:15:00 | 931.70 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -5.48% |
| BUY | retest2 | 2024-03-11 11:15:00 | 917.60 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2024-03-11 13:15:00 | 922.00 | 2024-03-12 11:15:00 | 880.60 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2024-03-15 09:45:00 | 849.90 | 2024-03-26 12:15:00 | 847.45 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2024-03-18 14:15:00 | 849.95 | 2024-03-26 12:15:00 | 847.45 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2024-04-19 10:15:00 | 874.90 | 2024-04-24 15:15:00 | 870.05 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-04-19 11:00:00 | 872.00 | 2024-04-24 15:15:00 | 870.05 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-04-23 11:30:00 | 872.00 | 2024-04-24 15:15:00 | 870.05 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-05-02 11:30:00 | 899.65 | 2024-05-02 15:15:00 | 892.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-05-07 11:15:00 | 856.40 | 2024-05-14 11:15:00 | 870.35 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-05-07 13:15:00 | 864.05 | 2024-05-14 11:15:00 | 870.35 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-05-08 09:15:00 | 863.30 | 2024-05-14 11:15:00 | 870.35 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-05-09 11:00:00 | 863.85 | 2024-05-14 11:15:00 | 870.35 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-05-17 11:15:00 | 879.50 | 2024-05-23 14:15:00 | 887.95 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2024-05-27 14:00:00 | 880.95 | 2024-05-28 11:15:00 | 896.15 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-05-28 11:15:00 | 883.25 | 2024-05-28 11:15:00 | 896.15 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-06-13 12:45:00 | 1013.30 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2024-06-13 13:30:00 | 1014.15 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2024-06-13 14:45:00 | 1014.00 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2024-06-14 15:00:00 | 1013.50 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2024-06-18 10:45:00 | 1044.00 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-06-18 11:15:00 | 1038.95 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-06-18 13:30:00 | 1037.50 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-06-20 10:15:00 | 1035.70 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-06-21 09:15:00 | 1047.20 | 2024-06-25 10:15:00 | 1033.25 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-07-02 14:30:00 | 1047.80 | 2024-07-04 09:15:00 | 1025.05 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-07-03 09:30:00 | 1047.20 | 2024-07-04 09:15:00 | 1025.05 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-07-03 15:00:00 | 1044.95 | 2024-07-04 09:15:00 | 1025.05 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-07-12 13:30:00 | 1039.15 | 2024-07-15 09:15:00 | 1014.80 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-08-02 09:15:00 | 1128.70 | 2024-08-05 09:15:00 | 1104.65 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-08-02 10:00:00 | 1126.90 | 2024-08-05 09:15:00 | 1104.65 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-08-02 10:45:00 | 1127.50 | 2024-08-05 09:15:00 | 1104.65 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-08-02 11:30:00 | 1127.60 | 2024-08-05 09:15:00 | 1104.65 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-08-08 09:15:00 | 1149.80 | 2024-08-14 12:15:00 | 1172.80 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2024-08-23 14:45:00 | 1263.60 | 2024-08-29 09:15:00 | 1389.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-13 09:30:00 | 1424.25 | 2024-09-13 11:15:00 | 1353.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 09:30:00 | 1424.25 | 2024-09-13 13:15:00 | 1417.10 | STOP_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2024-09-25 09:30:00 | 1304.65 | 2024-09-30 09:15:00 | 1279.80 | STOP_HIT | 1.00 | 1.90% |
| SELL | retest2 | 2024-10-15 11:45:00 | 1381.95 | 2024-10-18 09:15:00 | 1312.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:30:00 | 1384.90 | 2024-10-18 09:15:00 | 1315.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 11:45:00 | 1381.95 | 2024-10-18 15:15:00 | 1368.00 | STOP_HIT | 0.50 | 1.01% |
| SELL | retest2 | 2024-10-16 09:30:00 | 1384.90 | 2024-10-18 15:15:00 | 1368.00 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2024-10-16 10:15:00 | 1379.65 | 2024-10-22 10:15:00 | 1310.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 10:15:00 | 1379.65 | 2024-10-23 10:15:00 | 1292.85 | STOP_HIT | 0.50 | 6.29% |
| BUY | retest2 | 2024-10-31 10:30:00 | 1323.05 | 2024-11-05 09:15:00 | 1298.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-10-31 12:30:00 | 1313.90 | 2024-11-05 11:15:00 | 1297.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-10-31 13:00:00 | 1314.65 | 2024-11-05 11:15:00 | 1297.90 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-10-31 14:15:00 | 1314.00 | 2024-11-05 11:15:00 | 1297.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-11-04 15:00:00 | 1324.95 | 2024-11-05 11:15:00 | 1297.90 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-11-14 10:00:00 | 1354.50 | 2024-11-25 14:15:00 | 1482.47 | TARGET_HIT | 1.00 | 9.45% |
| BUY | retest2 | 2024-11-14 14:45:00 | 1347.70 | 2024-11-25 14:15:00 | 1483.63 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2024-11-18 11:15:00 | 1348.75 | 2024-11-26 09:15:00 | 1489.95 | TARGET_HIT | 1.00 | 10.47% |
| SELL | retest2 | 2024-11-29 13:00:00 | 1437.65 | 2024-12-02 09:15:00 | 1488.65 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2024-12-04 13:15:00 | 1472.95 | 2024-12-05 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-12-04 13:45:00 | 1470.65 | 2024-12-05 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-12-05 09:15:00 | 1476.00 | 2024-12-05 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-12-26 10:15:00 | 1335.20 | 2024-12-30 09:15:00 | 1362.55 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1290.35 | 2025-01-09 10:15:00 | 1309.55 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-01-08 10:00:00 | 1290.25 | 2025-01-09 10:15:00 | 1309.55 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-01-08 13:15:00 | 1295.80 | 2025-01-09 10:15:00 | 1309.55 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-01-08 13:45:00 | 1295.80 | 2025-01-09 10:15:00 | 1309.55 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1293.05 | 2025-01-09 11:15:00 | 1304.45 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1275.10 | 2025-01-15 11:15:00 | 1297.15 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-01-15 09:30:00 | 1279.45 | 2025-01-15 11:15:00 | 1297.15 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-01-21 10:00:00 | 1273.45 | 2025-01-24 14:15:00 | 1209.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:00:00 | 1273.45 | 2025-01-27 10:15:00 | 1146.11 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-31 09:30:00 | 1218.20 | 2025-02-05 10:15:00 | 1340.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-01 10:30:00 | 1213.75 | 2025-02-05 10:15:00 | 1335.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-01 12:00:00 | 1217.40 | 2025-02-05 10:15:00 | 1339.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-01 13:30:00 | 1234.30 | 2025-02-05 10:15:00 | 1357.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-03 15:15:00 | 1250.00 | 2025-02-05 10:15:00 | 1375.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-05 09:15:00 | 1291.55 | 2025-02-05 15:15:00 | 1420.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-20 15:00:00 | 1277.75 | 2025-02-21 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-02-21 10:30:00 | 1280.30 | 2025-02-21 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-02-21 11:30:00 | 1277.70 | 2025-02-21 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-02-21 14:30:00 | 1277.10 | 2025-02-21 15:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-02-25 14:00:00 | 1251.65 | 2025-02-27 12:15:00 | 1189.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 14:00:00 | 1251.65 | 2025-02-27 15:15:00 | 1220.05 | STOP_HIT | 0.50 | 2.52% |
| BUY | retest2 | 2025-03-10 12:30:00 | 1283.15 | 2025-03-12 09:15:00 | 1243.30 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2025-03-10 13:15:00 | 1283.20 | 2025-03-12 09:15:00 | 1243.30 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2025-03-11 10:30:00 | 1282.95 | 2025-03-12 09:15:00 | 1243.30 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1254.80 | 2025-03-17 12:15:00 | 1268.70 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-03-13 10:00:00 | 1254.55 | 2025-03-17 12:15:00 | 1268.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-03-13 14:00:00 | 1250.00 | 2025-03-17 12:15:00 | 1268.70 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-03-13 15:00:00 | 1252.70 | 2025-03-17 12:15:00 | 1268.70 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-03-21 12:00:00 | 1349.55 | 2025-03-25 12:15:00 | 1334.80 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-03-25 10:45:00 | 1348.20 | 2025-03-25 12:15:00 | 1334.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-03-25 11:15:00 | 1347.50 | 2025-03-25 12:15:00 | 1334.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-03-25 12:00:00 | 1345.85 | 2025-03-25 12:15:00 | 1334.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1289.30 | 2025-04-09 14:15:00 | 1353.95 | STOP_HIT | 1.00 | -5.01% |
| SELL | retest2 | 2025-04-09 12:15:00 | 1305.60 | 2025-04-09 14:15:00 | 1353.95 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-04-09 13:00:00 | 1306.80 | 2025-04-09 14:15:00 | 1353.95 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2025-04-16 11:15:00 | 1380.10 | 2025-04-24 09:15:00 | 1518.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-16 12:00:00 | 1381.80 | 2025-04-24 09:15:00 | 1519.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 11:15:00 | 1376.60 | 2025-04-24 09:15:00 | 1514.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 12:15:00 | 1380.20 | 2025-04-24 09:15:00 | 1518.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 13:45:00 | 1394.40 | 2025-04-29 14:15:00 | 1464.20 | STOP_HIT | 1.00 | 5.01% |
| BUY | retest2 | 2025-04-22 12:30:00 | 1401.60 | 2025-04-29 14:15:00 | 1464.20 | STOP_HIT | 1.00 | 4.47% |
| BUY | retest2 | 2025-05-06 12:15:00 | 1496.90 | 2025-05-06 14:15:00 | 1464.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-05-09 09:15:00 | 1422.80 | 2025-05-12 09:15:00 | 1474.00 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-05-12 11:30:00 | 1445.20 | 2025-05-14 10:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-05-13 11:00:00 | 1458.80 | 2025-05-14 10:15:00 | 1456.90 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-05-23 09:15:00 | 1563.90 | 2025-06-02 09:15:00 | 1558.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1534.30 | 2025-06-02 09:15:00 | 1558.90 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2025-05-26 11:00:00 | 1520.40 | 2025-06-02 09:15:00 | 1558.90 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2025-06-09 11:15:00 | 1634.30 | 2025-06-12 10:15:00 | 1797.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 12:45:00 | 1634.90 | 2025-06-12 10:15:00 | 1798.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 14:15:00 | 1633.90 | 2025-06-12 10:15:00 | 1797.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-09 09:30:00 | 1727.40 | 2025-07-11 12:15:00 | 1714.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-07-10 10:30:00 | 1724.80 | 2025-07-11 12:15:00 | 1714.20 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-07-11 11:30:00 | 1721.80 | 2025-07-11 12:15:00 | 1714.20 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-07-28 09:15:00 | 1814.90 | 2025-07-28 12:15:00 | 1763.70 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-07-31 13:30:00 | 1823.70 | 2025-08-04 09:15:00 | 1771.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-08-01 12:00:00 | 1831.40 | 2025-08-04 09:15:00 | 1771.00 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2025-08-01 12:45:00 | 1827.50 | 2025-08-04 09:15:00 | 1771.00 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-08-08 12:00:00 | 1714.00 | 2025-08-08 13:15:00 | 1732.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-08 15:15:00 | 1710.00 | 2025-08-18 09:15:00 | 1785.80 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-08-12 09:15:00 | 1683.30 | 2025-08-18 09:15:00 | 1785.80 | STOP_HIT | 1.00 | -6.09% |
| SELL | retest2 | 2025-09-10 14:15:00 | 1699.20 | 2025-09-11 09:15:00 | 1725.10 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-11 13:00:00 | 1697.20 | 2025-09-18 11:15:00 | 1612.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 14:45:00 | 1698.30 | 2025-09-18 11:15:00 | 1613.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 13:00:00 | 1697.20 | 2025-09-18 14:15:00 | 1634.30 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-09-11 14:45:00 | 1698.30 | 2025-09-18 14:15:00 | 1634.30 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1630.10 | 2025-10-01 12:15:00 | 1606.80 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2025-09-25 13:45:00 | 1630.00 | 2025-10-01 12:15:00 | 1606.80 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2025-10-03 12:00:00 | 1607.80 | 2025-10-03 14:15:00 | 1592.20 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-17 09:15:00 | 1579.80 | 2025-10-17 12:15:00 | 1595.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-23 09:15:00 | 1588.30 | 2025-10-23 09:15:00 | 1588.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-10-24 15:15:00 | 1575.00 | 2025-10-27 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1620.40 | 2025-10-31 11:15:00 | 1596.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-30 12:45:00 | 1621.90 | 2025-10-31 11:15:00 | 1596.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1618.00 | 2025-10-31 11:15:00 | 1596.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-04 12:15:00 | 1591.70 | 2025-11-13 13:15:00 | 1512.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:30:00 | 1589.20 | 2025-11-13 13:15:00 | 1509.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 13:00:00 | 1592.20 | 2025-11-13 13:15:00 | 1512.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 14:30:00 | 1591.40 | 2025-11-13 13:15:00 | 1511.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 12:15:00 | 1591.70 | 2025-11-14 12:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2025-11-06 09:30:00 | 1589.20 | 2025-11-14 12:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2025-11-06 13:00:00 | 1592.20 | 2025-11-14 12:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-11-07 14:30:00 | 1591.40 | 2025-11-14 12:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-11-10 11:00:00 | 1584.20 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1584.30 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2025-11-11 15:15:00 | 1579.00 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-11-12 12:00:00 | 1584.20 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1561.60 | 2025-11-17 13:15:00 | 1560.40 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-11-18 11:30:00 | 1571.80 | 2025-11-26 09:15:00 | 1616.30 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-11-19 12:45:00 | 1586.30 | 2025-11-26 09:15:00 | 1616.30 | STOP_HIT | 1.00 | 1.89% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1611.50 | 2025-12-08 12:15:00 | 1530.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:30:00 | 1605.70 | 2025-12-08 12:15:00 | 1528.55 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1611.50 | 2025-12-09 09:15:00 | 1608.00 | STOP_HIT | 0.50 | 0.22% |
| SELL | retest2 | 2025-11-27 13:30:00 | 1605.70 | 2025-12-09 09:15:00 | 1608.00 | STOP_HIT | 0.50 | -0.14% |
| SELL | retest2 | 2025-11-27 15:00:00 | 1609.00 | 2025-12-09 10:15:00 | 1606.00 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-12-16 10:45:00 | 1574.20 | 2025-12-19 14:15:00 | 1571.40 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-01-01 11:30:00 | 1492.80 | 2026-01-02 10:15:00 | 1520.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1338.10 | 2026-01-30 12:15:00 | 1382.10 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-02-02 15:15:00 | 1395.10 | 2026-02-06 09:15:00 | 1398.10 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2026-02-12 09:45:00 | 1484.20 | 2026-02-13 13:15:00 | 1440.60 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2026-02-23 09:15:00 | 1333.00 | 2026-02-23 14:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-23 10:00:00 | 1335.50 | 2026-02-23 14:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-02-23 11:45:00 | 1334.50 | 2026-02-23 14:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-19 10:30:00 | 1296.10 | 2026-03-25 09:15:00 | 1310.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-03-19 11:00:00 | 1299.40 | 2026-03-25 09:15:00 | 1310.10 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1261.20 | 2026-03-25 09:15:00 | 1310.10 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2026-03-27 15:00:00 | 1341.20 | 2026-03-30 10:15:00 | 1289.90 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2026-04-06 11:30:00 | 1317.20 | 2026-04-15 10:15:00 | 1448.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:45:00 | 1317.90 | 2026-04-15 10:15:00 | 1449.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 14:15:00 | 1334.50 | 2026-04-16 09:15:00 | 1467.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:15:00 | 1325.90 | 2026-04-16 09:15:00 | 1458.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:45:00 | 1330.80 | 2026-04-16 09:15:00 | 1463.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-04-23 11:30:00 | 1352.30 | 2026-04-23 14:15:00 | 1394.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-04-27 14:00:00 | 1348.30 | 2026-04-28 09:15:00 | 1213.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 15:15:00 | 1359.00 | 2026-04-28 09:15:00 | 1223.10 | TARGET_HIT | 1.00 | 10.00% |
