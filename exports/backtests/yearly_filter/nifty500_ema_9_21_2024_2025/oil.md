# Oil India Ltd. (OIL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 453.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 162 |
| ALERT1 | 99 |
| ALERT2 | 98 |
| ALERT2_SKIP | 49 |
| ALERT3 | 270 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 80 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 57
- **Target hits / Stop hits / Partials:** 2 / 80 / 7
- **Avg / median % per leg:** -0.11% / -0.65%
- **Sum % (uncompounded):** -10.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 12 | 33.3% | 1 | 35 | 0 | -0.82% | -29.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 36 | 12 | 33.3% | 1 | 35 | 0 | -0.82% | -29.4% |
| SELL (all) | 53 | 20 | 37.7% | 1 | 45 | 7 | 0.36% | 19.2% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.86% | -1.7% |
| SELL @ 3rd Alert (retest2) | 51 | 19 | 37.3% | 1 | 43 | 7 | 0.41% | 21.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.86% | -1.7% |
| retest2 (combined) | 87 | 31 | 35.6% | 2 | 78 | 7 | -0.10% | -8.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 414.20 | 411.08 | 410.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 420.00 | 413.31 | 412.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 15:15:00 | 424.50 | 425.06 | 421.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 09:15:00 | 427.43 | 425.06 | 421.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 435.90 | 437.75 | 435.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 435.90 | 437.75 | 435.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 434.00 | 437.00 | 435.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 432.53 | 437.00 | 435.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 433.17 | 436.23 | 434.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 432.67 | 436.23 | 434.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 434.90 | 435.97 | 434.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 14:45:00 | 435.77 | 435.66 | 434.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 10:45:00 | 435.97 | 435.55 | 435.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 436.77 | 440.38 | 440.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 436.77 | 440.38 | 440.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 434.67 | 439.23 | 439.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 12:15:00 | 428.30 | 427.95 | 431.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 13:00:00 | 428.30 | 427.95 | 431.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 425.23 | 421.71 | 425.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 424.93 | 421.71 | 425.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 423.50 | 422.07 | 425.18 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 447.03 | 427.34 | 427.06 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 367.93 | 419.55 | 426.17 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 409.50 | 404.28 | 404.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 15:15:00 | 410.37 | 407.76 | 406.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 407.33 | 408.26 | 406.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 407.33 | 408.26 | 406.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 406.80 | 407.97 | 406.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 13:00:00 | 406.80 | 407.97 | 406.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 404.43 | 407.17 | 406.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 15:00:00 | 404.43 | 407.17 | 406.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 405.33 | 406.80 | 406.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 410.00 | 406.80 | 406.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-13 10:15:00 | 451.00 | 441.83 | 432.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 584.60 | 599.15 | 599.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 15:15:00 | 583.00 | 595.92 | 598.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 09:15:00 | 566.20 | 560.50 | 569.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 566.20 | 560.50 | 569.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 566.20 | 560.50 | 569.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:00:00 | 558.15 | 560.03 | 568.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 530.24 | 553.78 | 564.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 555.50 | 551.58 | 559.41 | SL hit (close>ema200) qty=0.50 sl=551.58 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 589.65 | 563.66 | 560.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 10:15:00 | 597.40 | 582.81 | 576.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 580.95 | 586.50 | 580.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 14:15:00 | 580.95 | 586.50 | 580.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 580.95 | 586.50 | 580.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 580.95 | 586.50 | 580.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 584.40 | 586.08 | 580.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 582.05 | 586.08 | 580.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 577.60 | 584.39 | 580.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:45:00 | 578.00 | 584.39 | 580.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 583.10 | 584.13 | 580.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:30:00 | 583.60 | 584.13 | 580.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 581.00 | 583.50 | 580.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:00:00 | 581.00 | 583.50 | 580.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 578.80 | 582.56 | 580.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:45:00 | 578.20 | 582.56 | 580.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 578.95 | 581.84 | 580.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:45:00 | 578.75 | 581.84 | 580.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 576.90 | 580.85 | 580.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:30:00 | 576.95 | 580.85 | 580.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 576.90 | 580.06 | 579.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 596.00 | 580.06 | 579.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 580.55 | 594.52 | 594.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 580.55 | 594.52 | 594.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 573.50 | 590.32 | 592.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 10:15:00 | 585.55 | 582.17 | 586.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 10:15:00 | 585.55 | 582.17 | 586.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 585.55 | 582.17 | 586.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 585.55 | 582.17 | 586.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 590.05 | 583.74 | 587.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 12:00:00 | 590.05 | 583.74 | 587.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 588.10 | 584.61 | 587.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 12:45:00 | 589.20 | 584.61 | 587.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 09:15:00 | 620.70 | 587.40 | 587.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 630.65 | 614.18 | 607.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 10:15:00 | 647.65 | 676.95 | 664.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 10:15:00 | 647.65 | 676.95 | 664.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 647.65 | 676.95 | 664.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 647.65 | 676.95 | 664.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 654.90 | 672.54 | 663.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 12:15:00 | 666.90 | 672.54 | 663.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 15:15:00 | 664.40 | 668.93 | 664.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 13:15:00 | 670.40 | 677.88 | 677.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 670.40 | 677.88 | 677.89 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 678.80 | 677.63 | 677.61 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 11:15:00 | 674.05 | 676.91 | 677.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 12:15:00 | 673.25 | 676.18 | 676.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 679.65 | 675.35 | 676.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 679.65 | 675.35 | 676.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 679.65 | 675.35 | 676.12 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 14:15:00 | 680.60 | 676.87 | 676.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 699.80 | 683.61 | 680.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 708.10 | 711.88 | 703.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 10:00:00 | 708.10 | 711.88 | 703.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 710.20 | 716.17 | 710.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:30:00 | 710.55 | 716.17 | 710.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 711.00 | 715.14 | 710.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:45:00 | 718.80 | 716.23 | 711.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 716.20 | 725.12 | 726.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 09:15:00 | 716.20 | 725.12 | 726.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 12:15:00 | 705.50 | 717.66 | 721.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 637.05 | 632.79 | 645.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 627.20 | 632.79 | 645.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 616.85 | 629.60 | 642.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:00:00 | 610.35 | 619.99 | 632.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:15:00 | 579.83 | 608.38 | 624.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-12 14:15:00 | 580.00 | 578.92 | 592.60 | SL hit (close>ema200) qty=0.50 sl=578.92 alert=retest2 |

### Cycle 15 — BUY (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 13:15:00 | 601.35 | 588.75 | 588.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 10:15:00 | 618.70 | 602.16 | 595.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 15:15:00 | 613.00 | 613.39 | 604.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 09:15:00 | 606.30 | 613.39 | 604.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 602.15 | 611.14 | 604.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 605.65 | 611.14 | 604.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 605.85 | 610.08 | 604.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:45:00 | 599.45 | 610.08 | 604.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 601.15 | 608.30 | 604.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:30:00 | 602.70 | 608.30 | 604.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 602.80 | 607.20 | 604.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:15:00 | 596.00 | 607.20 | 604.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 597.55 | 605.27 | 603.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 14:00:00 | 597.55 | 605.27 | 603.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 593.70 | 601.31 | 602.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 572.40 | 595.53 | 599.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 580.95 | 574.78 | 584.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 15:15:00 | 562.00 | 567.81 | 576.61 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 578.50 | 569.02 | 575.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 578.50 | 569.02 | 575.57 | SL hit (close>ema400) qty=1.00 sl=575.57 alert=retest1 |

### Cycle 17 — BUY (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 09:15:00 | 585.65 | 579.09 | 578.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 593.55 | 586.12 | 582.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 12:15:00 | 585.80 | 589.24 | 585.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 12:15:00 | 585.80 | 589.24 | 585.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 585.80 | 589.24 | 585.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 585.80 | 589.24 | 585.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 581.35 | 587.66 | 585.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:00:00 | 581.35 | 587.66 | 585.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 578.75 | 585.88 | 584.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 578.75 | 585.88 | 584.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 578.55 | 584.41 | 584.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 576.05 | 584.41 | 584.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 573.50 | 582.23 | 583.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 11:15:00 | 562.85 | 576.75 | 580.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 591.35 | 570.36 | 572.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 591.35 | 570.36 | 572.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 591.35 | 570.36 | 572.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 591.35 | 570.36 | 572.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 586.50 | 573.59 | 573.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 576.70 | 573.59 | 573.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 09:15:00 | 576.25 | 574.12 | 573.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 09:15:00 | 576.25 | 574.12 | 573.88 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 568.45 | 575.43 | 576.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 564.05 | 572.13 | 574.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 563.00 | 553.95 | 562.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 563.00 | 553.95 | 562.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 563.00 | 553.95 | 562.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 563.00 | 553.95 | 562.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 557.90 | 554.74 | 561.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 553.00 | 563.11 | 563.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 566.00 | 559.46 | 559.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 566.00 | 559.46 | 559.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 11:15:00 | 567.85 | 562.37 | 560.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 573.50 | 576.31 | 570.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 573.50 | 576.31 | 570.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 572.60 | 575.56 | 571.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 574.65 | 574.85 | 571.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:45:00 | 575.80 | 575.52 | 571.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 564.45 | 580.76 | 579.30 | SL hit (close<static) qty=1.00 sl=570.20 alert=retest2 |

### Cycle 22 — SELL (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 10:15:00 | 561.20 | 576.84 | 577.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 11:15:00 | 557.20 | 572.92 | 575.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 531.20 | 528.46 | 538.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 09:30:00 | 533.20 | 528.46 | 538.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 545.70 | 533.34 | 538.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 545.70 | 533.34 | 538.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 539.05 | 534.49 | 538.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 13:15:00 | 537.25 | 534.49 | 538.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 510.39 | 520.35 | 526.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 517.00 | 511.68 | 517.87 | SL hit (close>ema200) qty=0.50 sl=511.68 alert=retest2 |

### Cycle 23 — BUY (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 10:15:00 | 478.10 | 473.99 | 473.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 492.05 | 479.63 | 476.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 12:15:00 | 519.25 | 519.73 | 508.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 13:00:00 | 519.25 | 519.73 | 508.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 510.50 | 518.44 | 511.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 509.75 | 518.44 | 511.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 508.75 | 516.50 | 511.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:30:00 | 510.15 | 516.50 | 511.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 506.75 | 514.55 | 510.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 506.75 | 514.55 | 510.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 509.00 | 510.09 | 509.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 500.60 | 510.09 | 509.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 502.85 | 508.64 | 508.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 484.75 | 496.75 | 501.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 481.60 | 480.15 | 487.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 481.60 | 480.15 | 487.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 486.00 | 477.02 | 479.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 486.00 | 477.02 | 479.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 485.60 | 478.73 | 480.47 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 487.15 | 481.72 | 481.61 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 12:15:00 | 480.00 | 482.07 | 482.16 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 13:15:00 | 483.30 | 482.32 | 482.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 14:15:00 | 485.50 | 482.96 | 482.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 512.00 | 514.73 | 506.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:00:00 | 512.00 | 514.73 | 506.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 504.30 | 511.69 | 506.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:00:00 | 504.30 | 511.69 | 506.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 12:15:00 | 505.60 | 510.47 | 506.36 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 09:15:00 | 491.80 | 504.31 | 504.56 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 508.70 | 503.96 | 503.79 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 497.55 | 503.44 | 503.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 11:15:00 | 490.60 | 500.04 | 502.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 484.45 | 483.47 | 489.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 09:30:00 | 484.25 | 483.47 | 489.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 490.15 | 485.66 | 488.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:00:00 | 490.15 | 485.66 | 488.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 491.05 | 486.74 | 488.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 496.50 | 486.74 | 488.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 10:15:00 | 492.90 | 490.05 | 489.82 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 471.70 | 486.31 | 488.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 470.80 | 483.21 | 486.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 470.95 | 470.41 | 473.68 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 13:30:00 | 467.85 | 469.20 | 472.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 469.50 | 469.28 | 471.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:00:00 | 468.75 | 469.48 | 470.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 13:45:00 | 468.45 | 469.27 | 470.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 13:15:00 | 462.15 | 458.63 | 460.92 | SL hit (close>ema400) qty=1.00 sl=460.92 alert=retest1 |

### Cycle 33 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 429.90 | 426.42 | 425.95 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 419.45 | 425.36 | 425.63 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 12:15:00 | 432.85 | 426.68 | 425.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 435.50 | 428.45 | 426.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 469.85 | 475.79 | 464.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 10:00:00 | 469.85 | 475.79 | 464.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 462.90 | 473.21 | 464.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 462.90 | 473.21 | 464.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 462.20 | 471.01 | 464.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 461.75 | 471.01 | 464.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 462.50 | 469.31 | 463.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:30:00 | 463.95 | 469.31 | 463.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 458.65 | 467.18 | 463.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 460.20 | 467.18 | 463.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 470.85 | 469.06 | 466.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 468.50 | 469.06 | 466.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 464.70 | 479.84 | 475.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 464.70 | 479.84 | 475.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 461.20 | 476.11 | 474.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 461.20 | 476.11 | 474.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 458.55 | 469.93 | 471.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 455.90 | 463.59 | 466.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 11:15:00 | 463.50 | 461.17 | 464.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 11:15:00 | 463.50 | 461.17 | 464.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 463.50 | 461.17 | 464.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:45:00 | 462.05 | 461.17 | 464.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 12:15:00 | 457.20 | 460.38 | 463.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 13:30:00 | 456.70 | 458.73 | 462.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 465.35 | 458.40 | 460.98 | SL hit (close>static) qty=1.00 sl=464.30 alert=retest2 |

### Cycle 37 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 465.00 | 462.70 | 462.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 09:15:00 | 469.45 | 464.05 | 463.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 13:15:00 | 463.95 | 466.23 | 464.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 13:15:00 | 463.95 | 466.23 | 464.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 463.95 | 466.23 | 464.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:00:00 | 463.95 | 466.23 | 464.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 464.20 | 465.83 | 464.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:30:00 | 464.40 | 465.83 | 464.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 464.00 | 465.46 | 464.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 477.25 | 465.46 | 464.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 462.00 | 474.53 | 475.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 462.00 | 474.53 | 475.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 457.80 | 471.18 | 474.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 403.70 | 401.71 | 411.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:30:00 | 406.00 | 401.71 | 411.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 408.60 | 403.66 | 408.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 413.30 | 403.66 | 408.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 415.60 | 406.04 | 409.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 416.80 | 406.04 | 409.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 413.30 | 407.50 | 409.42 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 417.05 | 410.89 | 410.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 423.50 | 417.53 | 414.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 417.35 | 420.35 | 417.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 417.35 | 420.35 | 417.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 417.35 | 420.35 | 417.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 417.35 | 420.35 | 417.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 409.75 | 418.23 | 416.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 409.75 | 418.23 | 416.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 412.00 | 416.98 | 416.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 412.00 | 416.98 | 416.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 411.30 | 415.85 | 415.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 399.15 | 411.65 | 413.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 403.40 | 399.58 | 405.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 403.40 | 399.58 | 405.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 403.40 | 399.58 | 405.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 403.40 | 399.58 | 405.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 403.60 | 400.81 | 404.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:45:00 | 403.00 | 400.81 | 404.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 402.80 | 401.78 | 404.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:45:00 | 404.05 | 401.78 | 404.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 405.30 | 402.48 | 404.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 405.30 | 402.48 | 404.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 403.95 | 402.78 | 404.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 418.35 | 402.78 | 404.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 425.05 | 407.23 | 406.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 426.20 | 411.03 | 408.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 423.35 | 425.20 | 419.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 423.35 | 425.20 | 419.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 416.85 | 423.31 | 420.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 416.85 | 423.31 | 420.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 421.75 | 423.00 | 420.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 423.50 | 423.00 | 420.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:30:00 | 422.45 | 423.55 | 421.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 409.75 | 420.90 | 420.77 | SL hit (close<static) qty=1.00 sl=411.70 alert=retest2 |

### Cycle 42 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 405.90 | 417.90 | 419.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 398.85 | 414.09 | 417.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 11:15:00 | 407.70 | 406.28 | 410.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 11:45:00 | 407.65 | 406.28 | 410.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 413.00 | 407.68 | 410.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 413.00 | 407.68 | 410.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 411.50 | 408.44 | 410.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:15:00 | 412.75 | 408.44 | 410.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 412.30 | 408.96 | 410.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 409.45 | 408.96 | 410.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 415.90 | 410.35 | 410.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 415.90 | 410.35 | 410.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 418.55 | 411.99 | 411.56 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 410.10 | 412.14 | 412.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 397.40 | 409.10 | 410.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 395.70 | 394.20 | 400.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 12:45:00 | 396.05 | 394.20 | 400.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 401.00 | 396.07 | 399.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 401.00 | 396.07 | 399.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 401.40 | 397.14 | 400.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 387.95 | 397.14 | 400.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 388.85 | 395.48 | 399.03 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 404.65 | 399.23 | 398.84 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 15:15:00 | 397.20 | 398.73 | 398.78 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 400.65 | 399.12 | 398.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 404.65 | 400.22 | 399.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 399.55 | 400.09 | 399.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 399.55 | 400.09 | 399.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 405.60 | 401.19 | 400.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:30:00 | 405.20 | 401.19 | 400.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 398.60 | 402.55 | 401.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 398.60 | 402.55 | 401.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 395.30 | 401.10 | 400.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 395.30 | 401.10 | 400.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 396.10 | 400.10 | 400.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 394.70 | 398.67 | 399.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 384.40 | 383.47 | 389.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 09:45:00 | 383.65 | 383.47 | 389.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 350.20 | 346.73 | 349.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 350.20 | 346.73 | 349.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 352.55 | 347.89 | 350.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 358.75 | 347.89 | 350.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 365.00 | 351.31 | 351.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 365.00 | 351.31 | 351.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 373.60 | 355.77 | 353.41 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 361.90 | 365.79 | 366.19 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 10:15:00 | 369.80 | 366.93 | 366.66 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 364.45 | 366.41 | 366.47 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 368.45 | 366.82 | 366.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 15:15:00 | 370.20 | 367.80 | 367.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 09:15:00 | 366.00 | 367.44 | 367.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 366.00 | 367.44 | 367.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 366.00 | 367.44 | 367.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 366.00 | 367.44 | 367.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 364.40 | 366.83 | 366.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 362.60 | 366.83 | 366.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 366.35 | 366.73 | 366.76 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 12:15:00 | 367.65 | 366.92 | 366.84 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 13:15:00 | 364.90 | 366.51 | 366.66 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 379.35 | 369.21 | 367.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 10:15:00 | 381.00 | 371.57 | 369.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 384.90 | 385.09 | 382.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 384.90 | 385.09 | 382.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 394.05 | 396.99 | 394.38 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 384.45 | 391.75 | 392.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 380.25 | 388.09 | 390.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 389.55 | 387.20 | 389.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 389.55 | 387.20 | 389.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 389.55 | 387.20 | 389.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 389.55 | 387.20 | 389.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 389.85 | 387.73 | 389.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:30:00 | 390.00 | 387.73 | 389.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 388.90 | 387.96 | 389.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 386.80 | 388.53 | 389.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 15:00:00 | 386.60 | 388.14 | 389.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 393.15 | 388.80 | 389.23 | SL hit (close>static) qty=1.00 sl=390.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 391.90 | 387.73 | 387.51 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 384.55 | 387.12 | 387.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 383.30 | 386.36 | 386.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 14:15:00 | 386.15 | 385.97 | 386.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 15:00:00 | 386.15 | 385.97 | 386.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 386.40 | 386.06 | 386.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 371.35 | 386.06 | 386.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 334.22 | 359.36 | 370.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 358.55 | 353.27 | 352.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 359.95 | 354.61 | 353.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 11:15:00 | 401.25 | 401.75 | 395.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 11:30:00 | 400.80 | 401.75 | 395.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 398.30 | 403.14 | 399.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 398.30 | 403.14 | 399.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 401.80 | 402.87 | 399.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:00:00 | 402.40 | 402.78 | 399.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 392.60 | 400.19 | 399.75 | SL hit (close<static) qty=1.00 sl=397.20 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 396.05 | 399.36 | 399.41 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 401.40 | 399.65 | 399.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 403.50 | 400.42 | 399.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 14:15:00 | 403.50 | 406.68 | 404.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 14:15:00 | 403.50 | 406.68 | 404.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 403.50 | 406.68 | 404.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 403.50 | 406.68 | 404.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 404.50 | 406.25 | 404.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 400.30 | 406.25 | 404.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 406.95 | 406.39 | 404.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 10:30:00 | 408.30 | 407.16 | 405.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 12:45:00 | 409.05 | 409.58 | 408.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 13:15:00 | 409.45 | 409.58 | 408.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 399.70 | 406.16 | 406.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 09:15:00 | 399.70 | 406.16 | 406.87 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 408.00 | 403.58 | 403.37 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 400.55 | 403.70 | 403.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 399.65 | 402.89 | 403.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 401.80 | 400.79 | 402.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 11:15:00 | 401.80 | 400.79 | 402.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 401.80 | 400.79 | 402.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:00:00 | 401.80 | 400.79 | 402.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 400.40 | 400.71 | 401.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:45:00 | 401.50 | 400.71 | 401.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 402.65 | 401.10 | 401.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 13:45:00 | 402.30 | 401.10 | 401.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 403.60 | 401.60 | 402.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 403.60 | 401.60 | 402.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 402.60 | 401.80 | 402.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 418.45 | 401.80 | 402.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 414.10 | 404.26 | 403.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 422.50 | 414.43 | 411.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 414.80 | 417.11 | 414.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 414.80 | 417.11 | 414.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 414.80 | 417.11 | 414.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 414.20 | 417.11 | 414.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 416.10 | 416.91 | 414.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 417.75 | 417.13 | 415.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 410.40 | 422.59 | 423.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 410.40 | 422.59 | 423.64 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 427.05 | 422.84 | 422.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 429.25 | 424.12 | 423.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 427.00 | 428.12 | 426.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 427.70 | 428.12 | 426.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 426.25 | 427.75 | 426.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 426.25 | 427.75 | 426.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 429.75 | 428.15 | 426.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 431.30 | 428.15 | 426.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 13:15:00 | 425.05 | 432.79 | 433.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 425.05 | 432.79 | 433.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 10:15:00 | 421.25 | 427.73 | 430.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 421.70 | 419.40 | 422.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 421.70 | 419.40 | 422.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 423.80 | 420.28 | 422.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 423.80 | 420.28 | 422.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 425.00 | 421.22 | 422.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 425.00 | 421.22 | 422.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 426.90 | 423.46 | 423.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 428.20 | 424.40 | 423.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 12:15:00 | 421.95 | 424.01 | 423.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 12:15:00 | 421.95 | 424.01 | 423.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 421.95 | 424.01 | 423.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:00:00 | 421.95 | 424.01 | 423.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 420.70 | 423.35 | 423.43 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 425.55 | 423.51 | 423.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 428.55 | 425.59 | 424.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 15:15:00 | 478.90 | 480.80 | 475.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 09:15:00 | 473.60 | 480.80 | 475.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 476.55 | 479.95 | 475.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 11:30:00 | 484.00 | 480.41 | 476.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 472.00 | 477.90 | 476.89 | SL hit (close<static) qty=1.00 sl=472.70 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 467.90 | 474.56 | 475.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 465.35 | 470.73 | 473.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 12:15:00 | 470.25 | 469.91 | 472.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 470.25 | 469.91 | 472.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 470.25 | 469.91 | 472.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 470.25 | 469.91 | 472.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 472.40 | 470.41 | 472.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:00:00 | 472.40 | 470.41 | 472.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 469.70 | 470.27 | 471.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 464.90 | 470.22 | 471.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 473.15 | 468.67 | 468.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 473.15 | 468.67 | 468.66 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 09:15:00 | 453.60 | 466.78 | 467.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 11:15:00 | 450.40 | 461.15 | 465.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 442.75 | 441.65 | 446.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 442.75 | 441.65 | 446.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 436.90 | 436.66 | 440.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 11:45:00 | 434.95 | 436.20 | 439.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:30:00 | 435.10 | 434.95 | 437.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 434.40 | 434.95 | 437.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 433.15 | 433.65 | 435.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 434.75 | 433.90 | 435.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 434.75 | 433.90 | 435.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 435.05 | 434.13 | 435.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 435.50 | 434.13 | 435.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 435.60 | 434.43 | 435.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:00:00 | 435.60 | 434.43 | 435.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 435.95 | 434.73 | 435.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 435.95 | 434.73 | 435.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 436.60 | 435.11 | 435.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 439.75 | 435.11 | 435.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 448.55 | 437.79 | 436.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 448.55 | 437.79 | 436.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 450.00 | 440.24 | 437.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 447.20 | 448.14 | 443.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:00:00 | 447.20 | 448.14 | 443.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 445.80 | 446.93 | 444.70 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 14:15:00 | 440.80 | 443.44 | 443.77 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 445.45 | 443.49 | 443.43 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 439.25 | 443.39 | 443.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 437.80 | 442.27 | 443.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 438.70 | 434.26 | 436.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 438.70 | 434.26 | 436.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 438.70 | 434.26 | 436.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 436.55 | 434.26 | 436.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 438.55 | 435.12 | 436.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 438.55 | 435.12 | 436.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 444.75 | 438.75 | 438.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 445.90 | 440.18 | 438.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 12:15:00 | 442.50 | 442.64 | 440.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:45:00 | 442.50 | 442.64 | 440.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 444.25 | 443.84 | 442.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 15:00:00 | 446.60 | 445.09 | 443.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 447.95 | 445.37 | 443.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:15:00 | 448.80 | 446.75 | 445.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 15:00:00 | 446.70 | 446.89 | 445.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 446.00 | 446.71 | 445.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 450.15 | 446.71 | 445.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 450.20 | 448.22 | 447.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 449.20 | 451.12 | 451.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 449.20 | 451.12 | 451.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 446.95 | 450.29 | 450.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 432.05 | 431.58 | 436.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 433.00 | 431.58 | 436.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 437.05 | 433.62 | 436.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 437.05 | 433.62 | 436.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 439.00 | 434.70 | 436.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 439.00 | 434.70 | 436.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 446.30 | 438.77 | 438.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 449.05 | 443.27 | 440.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 438.05 | 442.75 | 441.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 438.05 | 442.75 | 441.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 438.05 | 442.75 | 441.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 438.05 | 442.75 | 441.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 444.05 | 443.01 | 441.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 444.65 | 443.13 | 441.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 432.40 | 439.48 | 440.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 432.40 | 439.48 | 440.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 11:15:00 | 429.25 | 433.18 | 436.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 12:15:00 | 427.20 | 427.16 | 430.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 14:15:00 | 433.10 | 428.51 | 430.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 433.10 | 428.51 | 430.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 433.10 | 428.51 | 430.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 431.10 | 429.03 | 430.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 430.20 | 429.03 | 430.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 429.85 | 429.48 | 430.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:15:00 | 431.00 | 430.70 | 431.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:15:00 | 430.70 | 431.10 | 431.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 430.05 | 430.68 | 430.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:45:00 | 431.10 | 430.68 | 430.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 431.05 | 430.75 | 430.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 432.20 | 430.75 | 430.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 427.25 | 430.05 | 430.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 426.70 | 430.05 | 430.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 432.80 | 430.59 | 430.77 | SL hit (close>static) qty=1.00 sl=431.80 alert=retest2 |

### Cycle 85 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 432.65 | 431.00 | 430.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 09:15:00 | 434.50 | 431.70 | 431.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 432.45 | 433.17 | 432.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 14:15:00 | 432.45 | 433.17 | 432.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 432.45 | 433.17 | 432.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 432.45 | 433.17 | 432.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 430.70 | 432.67 | 432.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 431.70 | 432.67 | 432.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 425.50 | 431.24 | 431.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 421.00 | 428.36 | 430.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 425.55 | 425.29 | 427.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:00:00 | 425.55 | 425.29 | 427.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 425.65 | 425.44 | 427.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 428.15 | 425.44 | 427.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 415.60 | 423.02 | 425.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:15:00 | 410.10 | 421.01 | 424.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 12:15:00 | 408.60 | 406.21 | 405.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 12:15:00 | 408.60 | 406.21 | 405.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 412.45 | 408.34 | 407.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 410.35 | 410.45 | 408.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 409.35 | 410.23 | 409.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 409.35 | 410.23 | 409.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 409.05 | 410.23 | 409.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 409.70 | 410.12 | 409.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 409.70 | 410.12 | 409.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 409.00 | 409.90 | 409.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 409.00 | 409.90 | 409.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 410.00 | 409.92 | 409.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 409.00 | 409.92 | 409.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 409.00 | 409.75 | 409.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 409.00 | 409.75 | 409.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 409.10 | 409.62 | 409.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 407.65 | 409.62 | 409.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 405.55 | 408.80 | 408.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 404.55 | 407.76 | 408.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 392.15 | 390.71 | 394.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 392.15 | 390.71 | 394.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 392.15 | 390.71 | 394.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 393.90 | 390.71 | 394.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 394.75 | 391.52 | 394.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 394.75 | 391.52 | 394.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 396.05 | 392.42 | 394.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 397.00 | 392.42 | 394.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 400.50 | 396.22 | 395.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 402.10 | 397.39 | 396.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 402.85 | 403.11 | 400.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 402.85 | 403.11 | 400.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 397.80 | 401.78 | 400.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 397.80 | 401.78 | 400.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 396.55 | 400.73 | 400.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 396.55 | 400.73 | 400.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 396.40 | 399.87 | 400.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 395.15 | 398.92 | 399.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 10:15:00 | 396.95 | 395.83 | 397.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 11:00:00 | 396.95 | 395.83 | 397.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 396.35 | 395.46 | 396.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 396.35 | 395.46 | 396.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 396.30 | 395.63 | 396.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 396.50 | 395.63 | 396.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 396.50 | 395.80 | 396.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 395.75 | 395.80 | 396.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 396.80 | 396.00 | 396.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:45:00 | 394.55 | 395.85 | 396.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 394.60 | 395.85 | 396.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:45:00 | 393.90 | 395.55 | 396.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:30:00 | 394.70 | 395.25 | 396.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 395.40 | 394.55 | 395.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 395.40 | 394.55 | 395.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 394.50 | 394.54 | 395.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 395.55 | 394.54 | 395.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 392.10 | 394.05 | 395.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 390.80 | 393.30 | 394.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 396.95 | 393.29 | 393.77 | SL hit (close>static) qty=1.00 sl=395.25 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 396.80 | 394.57 | 394.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 13:15:00 | 398.10 | 395.92 | 395.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 395.00 | 396.46 | 395.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 395.00 | 396.46 | 395.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 395.00 | 396.46 | 395.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 395.05 | 396.46 | 395.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 394.90 | 396.15 | 395.66 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 394.05 | 395.35 | 395.36 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 399.10 | 395.95 | 395.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 401.45 | 398.93 | 397.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 401.70 | 401.81 | 399.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 401.70 | 401.81 | 399.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 400.00 | 401.27 | 399.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 400.00 | 401.27 | 399.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 399.00 | 400.82 | 399.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 397.80 | 400.82 | 399.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 397.80 | 400.21 | 399.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 397.90 | 400.21 | 399.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 399.20 | 400.01 | 399.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 403.25 | 399.79 | 399.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 412.00 | 414.55 | 414.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 11:15:00 | 412.00 | 414.55 | 414.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 12:15:00 | 410.80 | 413.80 | 414.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 15:15:00 | 413.85 | 412.85 | 413.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 15:15:00 | 413.85 | 412.85 | 413.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 413.85 | 412.85 | 413.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:45:00 | 410.75 | 412.28 | 413.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 414.75 | 412.82 | 413.11 | SL hit (close>static) qty=1.00 sl=414.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 416.00 | 413.46 | 413.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 419.70 | 414.71 | 413.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 13:15:00 | 414.50 | 416.26 | 415.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 13:15:00 | 414.50 | 416.26 | 415.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 414.50 | 416.26 | 415.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:00:00 | 414.50 | 416.26 | 415.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 419.90 | 416.99 | 415.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:30:00 | 414.00 | 416.99 | 415.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 418.05 | 420.00 | 418.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 418.05 | 420.00 | 418.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 419.35 | 419.87 | 418.76 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 415.90 | 418.07 | 418.19 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 418.95 | 418.10 | 418.03 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 417.20 | 417.92 | 417.96 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 418.35 | 418.04 | 418.00 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 415.75 | 417.58 | 417.80 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 420.00 | 416.05 | 416.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 422.95 | 418.06 | 416.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 416.90 | 418.96 | 417.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 416.90 | 418.96 | 417.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 416.90 | 418.96 | 417.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:00:00 | 416.90 | 418.96 | 417.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 419.20 | 419.01 | 417.90 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 411.50 | 416.40 | 417.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 407.60 | 413.36 | 415.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 414.85 | 409.30 | 410.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 414.85 | 409.30 | 410.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 414.85 | 409.30 | 410.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 413.75 | 409.30 | 410.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 413.00 | 410.04 | 410.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:45:00 | 413.00 | 410.04 | 410.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 414.95 | 411.90 | 411.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 13:15:00 | 417.50 | 413.02 | 412.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 416.00 | 420.18 | 418.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 416.00 | 420.18 | 418.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 416.00 | 420.18 | 418.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 416.00 | 420.18 | 418.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 414.40 | 419.02 | 418.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 414.40 | 419.02 | 418.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 413.45 | 417.91 | 417.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 412.30 | 416.79 | 417.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 416.00 | 415.09 | 416.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 416.00 | 415.09 | 416.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 416.00 | 415.09 | 416.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 416.40 | 415.09 | 416.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 416.60 | 415.40 | 416.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 416.60 | 415.40 | 416.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 419.20 | 416.16 | 416.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 419.20 | 416.16 | 416.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 420.35 | 416.99 | 416.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 428.90 | 420.72 | 418.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 431.30 | 431.51 | 427.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 13:15:00 | 432.80 | 431.51 | 427.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 433.55 | 435.91 | 433.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 433.55 | 435.91 | 433.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 433.40 | 435.41 | 433.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 433.40 | 435.41 | 433.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 432.60 | 434.85 | 433.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 432.60 | 434.85 | 433.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 430.65 | 434.01 | 433.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 430.65 | 434.01 | 433.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 431.50 | 432.68 | 432.73 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 15:15:00 | 433.15 | 432.78 | 432.73 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 430.15 | 432.25 | 432.50 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 10:15:00 | 435.25 | 432.85 | 432.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 11:15:00 | 436.70 | 433.62 | 433.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 13:15:00 | 433.85 | 433.86 | 433.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 433.85 | 433.86 | 433.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 433.85 | 433.86 | 433.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 433.85 | 433.86 | 433.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 433.60 | 433.80 | 433.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:45:00 | 432.30 | 433.80 | 433.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 437.30 | 434.53 | 433.75 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 432.40 | 433.51 | 433.58 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 436.25 | 433.75 | 433.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 437.00 | 434.40 | 433.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 434.45 | 439.26 | 437.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 434.45 | 439.26 | 437.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 434.45 | 439.26 | 437.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 434.10 | 439.26 | 437.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 435.50 | 438.51 | 437.43 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 434.25 | 436.87 | 436.89 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 13:15:00 | 438.35 | 436.89 | 436.76 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 15:15:00 | 434.25 | 436.69 | 436.71 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 441.90 | 437.74 | 437.18 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 430.90 | 436.72 | 437.26 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 437.00 | 436.06 | 435.95 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 425.65 | 434.24 | 435.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 424.80 | 428.79 | 431.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 421.30 | 420.73 | 424.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:00:00 | 421.30 | 420.73 | 424.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 423.00 | 421.59 | 423.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:00:00 | 421.75 | 421.81 | 423.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 424.40 | 422.54 | 423.16 | SL hit (close>static) qty=1.00 sl=423.65 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 417.00 | 415.58 | 415.55 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 408.35 | 414.13 | 414.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 10:15:00 | 407.60 | 410.36 | 412.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 410.70 | 409.15 | 410.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 410.70 | 409.15 | 410.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 410.70 | 409.15 | 410.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 410.70 | 409.15 | 410.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 408.95 | 409.11 | 410.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 407.90 | 409.73 | 410.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 405.60 | 403.28 | 403.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 405.60 | 403.28 | 403.22 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 400.75 | 403.25 | 403.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 398.10 | 400.94 | 401.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 400.10 | 399.98 | 401.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:00:00 | 400.10 | 399.98 | 401.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 403.00 | 400.28 | 400.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 400.80 | 400.28 | 400.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 401.55 | 400.54 | 400.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 401.15 | 400.54 | 400.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 401.75 | 400.78 | 400.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 403.30 | 400.78 | 400.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 404.15 | 401.45 | 401.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 406.25 | 402.90 | 401.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 406.05 | 406.11 | 404.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 406.05 | 406.11 | 404.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 412.05 | 407.17 | 405.23 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 402.50 | 407.36 | 407.74 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 408.80 | 406.92 | 406.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 411.40 | 408.62 | 407.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 417.95 | 425.68 | 423.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 417.95 | 425.68 | 423.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 417.95 | 425.68 | 423.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 417.95 | 425.68 | 423.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 422.10 | 424.97 | 423.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 417.20 | 424.97 | 423.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 419.75 | 423.92 | 423.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 419.75 | 423.92 | 423.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 419.05 | 422.27 | 422.62 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 13:15:00 | 423.45 | 422.41 | 422.34 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 419.40 | 422.25 | 422.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 11:15:00 | 415.25 | 420.85 | 421.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 423.65 | 413.86 | 415.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 423.65 | 413.86 | 415.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 423.65 | 413.86 | 415.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 423.65 | 413.86 | 415.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 418.70 | 414.83 | 416.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:15:00 | 418.50 | 414.83 | 416.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:15:00 | 418.15 | 416.71 | 416.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 420.60 | 417.49 | 417.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 14:15:00 | 420.60 | 417.49 | 417.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 11:15:00 | 422.45 | 418.97 | 418.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 446.30 | 453.70 | 445.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 446.30 | 453.70 | 445.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 446.30 | 453.70 | 445.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 446.30 | 453.70 | 445.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 443.95 | 451.75 | 445.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 443.95 | 451.75 | 445.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 443.65 | 450.13 | 445.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:15:00 | 443.00 | 450.13 | 445.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 446.10 | 449.33 | 445.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 13:30:00 | 448.60 | 448.79 | 445.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:30:00 | 448.45 | 448.88 | 445.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 440.65 | 446.09 | 445.17 | SL hit (close<static) qty=1.00 sl=442.50 alert=retest2 |

### Cycle 130 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 436.50 | 443.09 | 443.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 433.10 | 438.40 | 441.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 432.60 | 431.56 | 435.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 432.60 | 431.56 | 435.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 433.25 | 431.90 | 434.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 434.50 | 431.90 | 434.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 433.00 | 431.65 | 434.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:45:00 | 433.70 | 431.65 | 434.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 434.85 | 432.29 | 434.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 442.85 | 432.29 | 434.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 440.00 | 433.83 | 434.79 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 440.40 | 436.08 | 435.71 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 430.60 | 435.14 | 435.48 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 13:15:00 | 436.45 | 435.75 | 435.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 442.20 | 437.04 | 436.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 502.85 | 506.46 | 491.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:45:00 | 502.85 | 506.46 | 491.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 514.20 | 508.51 | 498.87 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 478.00 | 495.24 | 497.06 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 510.00 | 490.10 | 489.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 521.45 | 496.37 | 492.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 504.25 | 504.67 | 499.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 504.25 | 504.67 | 499.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 498.15 | 502.39 | 499.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:00:00 | 498.15 | 502.39 | 499.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 498.55 | 501.62 | 499.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 498.55 | 501.62 | 499.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 498.85 | 501.06 | 499.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 498.85 | 501.06 | 499.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 497.20 | 500.29 | 499.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 493.55 | 500.29 | 499.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 494.85 | 498.72 | 498.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 494.85 | 498.72 | 498.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 495.70 | 498.12 | 498.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 09:15:00 | 488.15 | 495.38 | 497.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 489.00 | 488.98 | 491.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 14:15:00 | 489.00 | 488.98 | 491.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 489.00 | 488.98 | 491.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 489.00 | 488.98 | 491.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 482.15 | 481.27 | 484.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:00:00 | 475.10 | 479.49 | 483.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 471.90 | 477.84 | 481.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:15:00 | 451.35 | 460.44 | 463.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:15:00 | 448.30 | 460.44 | 463.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 464.70 | 456.06 | 458.56 | SL hit (close>ema200) qty=0.50 sl=456.06 alert=retest2 |

### Cycle 137 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 465.75 | 460.06 | 459.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 13:15:00 | 476.10 | 463.27 | 461.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 09:15:00 | 471.20 | 473.12 | 469.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 471.20 | 473.12 | 469.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 471.20 | 473.12 | 469.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 471.55 | 473.12 | 469.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 470.00 | 472.50 | 469.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 469.30 | 472.50 | 469.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 468.95 | 471.79 | 469.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 468.25 | 471.79 | 469.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 469.15 | 471.26 | 469.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:30:00 | 469.35 | 471.26 | 469.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 472.95 | 471.60 | 469.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 473.25 | 471.60 | 469.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 473.35 | 471.95 | 470.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 473.45 | 472.56 | 470.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 468.10 | 471.27 | 470.59 | SL hit (close<static) qty=1.00 sl=468.65 alert=retest2 |

### Cycle 138 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 465.20 | 469.97 | 470.32 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 474.20 | 470.82 | 470.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 10:15:00 | 486.00 | 476.07 | 473.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 476.80 | 481.58 | 478.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 476.80 | 481.58 | 478.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 476.80 | 481.58 | 478.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:00:00 | 476.80 | 481.58 | 478.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 484.85 | 482.24 | 478.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:30:00 | 481.20 | 482.24 | 478.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 485.20 | 482.53 | 479.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:30:00 | 480.80 | 482.53 | 479.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 487.65 | 493.76 | 490.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 487.65 | 493.76 | 490.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 486.85 | 492.38 | 490.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 487.70 | 492.38 | 490.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 15:15:00 | 481.50 | 488.44 | 488.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 476.50 | 484.94 | 487.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 13:15:00 | 483.45 | 483.30 | 485.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 13:45:00 | 483.80 | 483.30 | 485.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 485.00 | 483.64 | 485.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 485.00 | 483.64 | 485.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 484.60 | 483.83 | 485.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 478.90 | 483.83 | 485.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 478.30 | 482.73 | 484.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 10:15:00 | 473.65 | 482.73 | 484.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 15:15:00 | 471.40 | 474.27 | 479.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:15:00 | 472.25 | 472.79 | 476.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:15:00 | 472.35 | 472.99 | 476.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 479.60 | 472.65 | 474.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 479.60 | 472.65 | 474.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 482.25 | 474.57 | 475.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 482.25 | 474.57 | 475.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-11 13:15:00 | 482.10 | 476.07 | 475.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 482.10 | 476.07 | 475.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 09:15:00 | 486.15 | 479.61 | 477.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 12:15:00 | 479.30 | 481.04 | 478.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 12:15:00 | 479.30 | 481.04 | 478.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 479.30 | 481.04 | 478.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:30:00 | 478.35 | 481.04 | 478.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 479.15 | 480.66 | 478.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:00:00 | 479.15 | 480.66 | 478.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 479.00 | 480.33 | 478.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 479.10 | 480.33 | 478.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 479.50 | 480.17 | 479.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:30:00 | 477.80 | 479.50 | 478.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 478.20 | 479.24 | 478.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 481.75 | 479.24 | 478.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 470.55 | 477.19 | 477.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 470.55 | 477.19 | 477.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 470.15 | 475.78 | 477.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 465.90 | 463.62 | 468.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 465.90 | 463.62 | 468.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 467.10 | 464.32 | 468.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:15:00 | 466.25 | 464.32 | 468.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 469.50 | 465.35 | 468.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 469.50 | 465.35 | 468.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 470.50 | 466.38 | 468.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 471.25 | 466.38 | 468.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 470.70 | 467.25 | 468.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:30:00 | 470.90 | 467.25 | 468.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 473.00 | 469.53 | 469.52 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 11:15:00 | 466.50 | 469.15 | 469.38 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 470.70 | 469.52 | 469.49 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 15:15:00 | 468.80 | 469.37 | 469.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 466.60 | 468.82 | 469.17 | Break + close below crossover candle low |

### Cycle 147 — BUY (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 10:15:00 | 480.15 | 471.09 | 470.17 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 469.30 | 473.39 | 473.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 465.25 | 471.76 | 473.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 469.50 | 469.06 | 471.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:30:00 | 468.65 | 469.06 | 471.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 472.00 | 469.64 | 471.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:00:00 | 472.00 | 469.64 | 471.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 475.60 | 470.84 | 471.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 476.35 | 470.84 | 471.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 477.50 | 472.17 | 472.15 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 472.00 | 472.79 | 472.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 465.70 | 471.37 | 472.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 12:15:00 | 473.75 | 470.53 | 471.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 473.75 | 470.53 | 471.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 473.75 | 470.53 | 471.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 473.75 | 470.53 | 471.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 13:15:00 | 487.80 | 473.98 | 472.99 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 13:15:00 | 473.05 | 477.60 | 477.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 465.75 | 474.05 | 475.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 478.65 | 473.19 | 474.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 478.65 | 473.19 | 474.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 478.65 | 473.19 | 474.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 478.65 | 473.19 | 474.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 482.00 | 474.95 | 475.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 482.00 | 474.95 | 475.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 481.45 | 476.41 | 475.94 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 14:15:00 | 472.85 | 475.76 | 475.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 15:15:00 | 471.10 | 474.83 | 475.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 477.40 | 475.34 | 475.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 477.40 | 475.34 | 475.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 477.40 | 475.34 | 475.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 474.50 | 475.34 | 475.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 484.70 | 477.21 | 476.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 13:15:00 | 487.85 | 481.23 | 478.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 09:15:00 | 463.55 | 478.16 | 477.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 463.55 | 478.16 | 477.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 463.55 | 478.16 | 477.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 461.85 | 478.16 | 477.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 10:15:00 | 464.50 | 475.43 | 476.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 13:15:00 | 460.80 | 469.52 | 473.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 12:15:00 | 463.40 | 462.90 | 467.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 463.40 | 462.90 | 467.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 469.05 | 464.13 | 467.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 467.80 | 464.13 | 467.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 470.80 | 465.47 | 468.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 470.80 | 465.47 | 468.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 470.90 | 466.55 | 468.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 468.50 | 466.55 | 468.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 470.60 | 468.09 | 468.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:45:00 | 469.35 | 468.09 | 468.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 472.80 | 469.03 | 469.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 472.80 | 469.03 | 469.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 12:15:00 | 473.70 | 469.97 | 469.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 479.10 | 472.64 | 471.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 09:15:00 | 463.05 | 473.18 | 472.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 463.05 | 473.18 | 472.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 463.05 | 473.18 | 472.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 463.05 | 473.18 | 472.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 10:15:00 | 462.15 | 470.97 | 471.50 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 469.55 | 466.67 | 466.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 471.30 | 467.96 | 466.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 470.20 | 470.48 | 468.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 471.25 | 470.48 | 468.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 467.80 | 469.91 | 468.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 467.80 | 469.91 | 468.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 466.85 | 469.30 | 468.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:45:00 | 466.90 | 469.30 | 468.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 13:15:00 | 464.85 | 467.99 | 468.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 464.40 | 467.27 | 467.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 467.10 | 465.90 | 466.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 467.10 | 465.90 | 466.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 467.10 | 465.90 | 466.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:30:00 | 467.00 | 465.90 | 466.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 469.45 | 466.61 | 466.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 15:00:00 | 469.45 | 466.61 | 466.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 470.00 | 467.29 | 467.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 473.25 | 468.48 | 467.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 11:15:00 | 474.30 | 475.56 | 472.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 12:00:00 | 474.30 | 475.56 | 472.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 475.40 | 475.53 | 472.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 476.10 | 474.75 | 473.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 15:00:00 | 476.00 | 474.75 | 473.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 476.70 | 488.18 | 488.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 09:15:00 | 476.70 | 488.18 | 488.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 13:15:00 | 475.75 | 481.74 | 485.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 476.55 | 475.66 | 480.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:45:00 | 476.00 | 475.66 | 480.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-23 14:45:00 | 435.77 | 2024-05-28 14:15:00 | 436.77 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-05-24 10:45:00 | 435.97 | 2024-05-28 14:15:00 | 436.77 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-06-11 09:15:00 | 410.00 | 2024-06-13 10:15:00 | 451.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-23 11:00:00 | 558.15 | 2024-07-23 12:15:00 | 530.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 11:00:00 | 558.15 | 2024-07-24 09:15:00 | 555.50 | STOP_HIT | 0.50 | 0.47% |
| BUY | retest2 | 2024-08-01 09:15:00 | 596.00 | 2024-08-05 11:15:00 | 580.55 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-08-14 12:15:00 | 666.90 | 2024-08-20 13:15:00 | 670.40 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2024-08-14 15:15:00 | 664.40 | 2024-08-20 13:15:00 | 670.40 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-08-29 12:45:00 | 718.80 | 2024-09-03 09:15:00 | 716.20 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2024-09-10 15:00:00 | 610.35 | 2024-09-11 09:15:00 | 579.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-10 15:00:00 | 610.35 | 2024-09-12 14:15:00 | 580.00 | STOP_HIT | 0.50 | 4.97% |
| SELL | retest1 | 2024-09-20 15:15:00 | 562.00 | 2024-09-23 09:15:00 | 578.50 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-09-30 09:15:00 | 576.70 | 2024-09-30 09:15:00 | 576.25 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-10-07 10:15:00 | 553.00 | 2024-10-08 14:15:00 | 566.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-10-11 09:15:00 | 574.65 | 2024-10-15 09:15:00 | 564.45 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-10-11 09:45:00 | 575.80 | 2024-10-15 09:15:00 | 564.45 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-10-18 13:15:00 | 537.25 | 2024-10-23 09:15:00 | 510.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 13:15:00 | 537.25 | 2024-10-24 09:15:00 | 517.00 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest1 | 2024-12-10 13:30:00 | 467.85 | 2024-12-16 13:15:00 | 462.15 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2024-12-12 11:00:00 | 468.75 | 2024-12-17 13:15:00 | 445.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 13:45:00 | 468.45 | 2024-12-17 14:15:00 | 445.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 11:00:00 | 468.75 | 2024-12-19 12:15:00 | 444.50 | STOP_HIT | 0.50 | 5.17% |
| SELL | retest2 | 2024-12-12 13:45:00 | 468.45 | 2024-12-19 12:15:00 | 444.50 | STOP_HIT | 0.50 | 5.11% |
| SELL | retest2 | 2025-01-13 13:30:00 | 456.70 | 2025-01-14 10:15:00 | 465.35 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-01-16 09:15:00 | 477.25 | 2025-01-22 09:15:00 | 462.00 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-02-07 11:15:00 | 423.50 | 2025-02-10 09:15:00 | 409.75 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2025-02-07 14:30:00 | 422.45 | 2025-02-10 09:15:00 | 409.75 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-03-27 14:15:00 | 386.80 | 2025-03-28 09:15:00 | 393.15 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-03-27 15:00:00 | 386.60 | 2025-03-28 09:15:00 | 393.15 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-03-28 12:30:00 | 385.65 | 2025-04-01 09:15:00 | 391.65 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-03-28 14:45:00 | 384.85 | 2025-04-01 09:15:00 | 391.65 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-04-01 13:15:00 | 385.70 | 2025-04-02 14:15:00 | 391.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-04-01 14:30:00 | 385.35 | 2025-04-02 14:15:00 | 391.90 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-04-01 15:00:00 | 385.20 | 2025-04-02 14:15:00 | 391.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-04-04 09:15:00 | 371.35 | 2025-04-07 09:15:00 | 334.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-24 12:00:00 | 402.40 | 2025-04-25 10:15:00 | 392.60 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-04-30 10:30:00 | 408.30 | 2025-05-05 09:15:00 | 399.70 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-05-02 12:45:00 | 409.05 | 2025-05-05 09:15:00 | 399.70 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-05-02 13:15:00 | 409.45 | 2025-05-05 09:15:00 | 399.70 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-05-15 14:00:00 | 417.75 | 2025-05-22 09:15:00 | 410.40 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-05-27 11:15:00 | 431.30 | 2025-05-30 13:15:00 | 425.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-06-17 11:30:00 | 484.00 | 2025-06-18 10:15:00 | 472.00 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-06-20 09:15:00 | 464.90 | 2025-06-23 13:15:00 | 473.15 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-06-30 11:45:00 | 434.95 | 2025-07-03 09:15:00 | 448.55 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-07-01 09:30:00 | 435.10 | 2025-07-03 09:15:00 | 448.55 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-07-01 10:00:00 | 434.40 | 2025-07-03 09:15:00 | 448.55 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-07-02 09:30:00 | 433.15 | 2025-07-03 09:15:00 | 448.55 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-07-16 15:00:00 | 446.60 | 2025-07-24 11:15:00 | 449.20 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-07-17 09:15:00 | 447.95 | 2025-07-24 11:15:00 | 449.20 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-07-17 13:15:00 | 448.80 | 2025-07-24 11:15:00 | 449.20 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-07-17 15:00:00 | 446.70 | 2025-07-24 11:15:00 | 449.20 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-07-18 09:15:00 | 450.15 | 2025-07-24 11:15:00 | 449.20 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-07-21 10:15:00 | 450.20 | 2025-07-24 11:15:00 | 449.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-07-31 11:30:00 | 444.65 | 2025-08-01 09:15:00 | 432.40 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-08-06 09:30:00 | 430.20 | 2025-08-07 14:15:00 | 432.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-08-06 10:30:00 | 429.85 | 2025-08-07 15:15:00 | 432.65 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-08-06 14:15:00 | 431.00 | 2025-08-07 15:15:00 | 432.65 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-08-07 09:15:00 | 430.70 | 2025-08-07 15:15:00 | 432.65 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-08-07 13:15:00 | 426.70 | 2025-08-07 15:15:00 | 432.65 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-08-13 11:15:00 | 410.10 | 2025-08-20 12:15:00 | 408.60 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-09-08 10:45:00 | 394.55 | 2025-09-11 09:15:00 | 396.95 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-08 11:15:00 | 394.60 | 2025-09-11 11:15:00 | 396.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-09-08 11:45:00 | 393.90 | 2025-09-11 11:15:00 | 396.80 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-08 13:30:00 | 394.70 | 2025-09-11 11:15:00 | 396.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-09-10 14:15:00 | 390.80 | 2025-09-11 11:15:00 | 396.80 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-09-19 09:15:00 | 403.25 | 2025-10-01 11:15:00 | 412.00 | STOP_HIT | 1.00 | 2.17% |
| SELL | retest2 | 2025-10-03 09:45:00 | 410.75 | 2025-10-03 14:15:00 | 414.75 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-26 13:00:00 | 421.75 | 2025-11-26 14:15:00 | 424.40 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-11-27 09:30:00 | 420.55 | 2025-12-02 15:15:00 | 417.00 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-12-08 12:00:00 | 407.90 | 2025-12-11 13:15:00 | 405.60 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2026-01-09 11:15:00 | 418.50 | 2026-01-09 14:15:00 | 420.60 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-01-09 14:15:00 | 418.15 | 2026-01-09 14:15:00 | 420.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-01-16 13:30:00 | 448.60 | 2026-01-19 10:15:00 | 440.65 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2026-01-16 14:30:00 | 448.45 | 2026-01-19 10:15:00 | 440.65 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-02-12 13:00:00 | 475.10 | 2026-02-18 09:15:00 | 451.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 14:30:00 | 471.90 | 2026-02-18 09:15:00 | 448.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 13:00:00 | 475.10 | 2026-02-19 09:15:00 | 464.70 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2026-02-12 14:30:00 | 471.90 | 2026-02-19 09:15:00 | 464.70 | STOP_HIT | 0.50 | 1.53% |
| BUY | retest2 | 2026-02-23 14:15:00 | 473.25 | 2026-02-24 12:15:00 | 468.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-23 15:00:00 | 473.35 | 2026-02-24 12:15:00 | 468.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-24 11:00:00 | 473.45 | 2026-02-24 12:15:00 | 468.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-02-24 15:15:00 | 473.85 | 2026-02-25 12:15:00 | 465.20 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-03-09 10:15:00 | 473.65 | 2026-03-11 13:15:00 | 482.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-09 15:15:00 | 471.40 | 2026-03-11 13:15:00 | 482.10 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-03-10 13:15:00 | 472.25 | 2026-03-11 13:15:00 | 482.10 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-03-10 14:15:00 | 472.35 | 2026-03-11 13:15:00 | 482.10 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-03-13 11:15:00 | 481.75 | 2026-03-13 12:15:00 | 470.55 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-04-27 14:30:00 | 476.10 | 2026-05-04 09:15:00 | 476.70 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2026-04-27 15:00:00 | 476.00 | 2026-05-04 09:15:00 | 476.70 | STOP_HIT | 1.00 | 0.15% |
