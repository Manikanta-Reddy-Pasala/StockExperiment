# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 619.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 222 |
| ALERT1 | 155 |
| ALERT2 | 155 |
| ALERT2_SKIP | 90 |
| ALERT3 | 373 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 146 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 149 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 154 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 117
- **Target hits / Stop hits / Partials:** 0 / 148 / 6
- **Avg / median % per leg:** -0.27% / -0.62%
- **Sum % (uncompounded):** -41.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 13 | 16.9% | 0 | 77 | 0 | -0.38% | -28.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 77 | 13 | 16.9% | 0 | 77 | 0 | -0.38% | -28.9% |
| SELL (all) | 77 | 24 | 31.2% | 0 | 71 | 6 | -0.16% | -12.4% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.08% | -2.2% |
| SELL @ 3rd Alert (retest2) | 75 | 24 | 32.0% | 0 | 69 | 6 | -0.14% | -10.3% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.08% | -2.2% |
| retest2 (combined) | 152 | 37 | 24.3% | 0 | 146 | 6 | -0.26% | -39.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 12:15:00 | 557.00 | 561.12 | 561.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 11:15:00 | 554.95 | 558.66 | 559.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-16 12:15:00 | 560.00 | 558.93 | 559.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 12:15:00 | 560.00 | 558.93 | 559.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 12:15:00 | 560.00 | 558.93 | 559.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-16 13:00:00 | 560.00 | 558.93 | 559.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 13:15:00 | 556.70 | 558.48 | 559.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-16 14:15:00 | 560.90 | 558.48 | 559.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 14:15:00 | 559.50 | 558.68 | 559.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-16 14:45:00 | 559.70 | 558.68 | 559.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 15:15:00 | 559.25 | 558.80 | 559.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 09:15:00 | 555.50 | 558.80 | 559.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 558.45 | 558.73 | 559.47 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 09:15:00 | 565.40 | 559.95 | 559.62 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 11:15:00 | 557.45 | 559.96 | 560.27 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 565.50 | 560.54 | 560.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 11:15:00 | 566.85 | 561.81 | 560.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 09:15:00 | 563.20 | 564.31 | 562.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 563.20 | 564.31 | 562.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 563.20 | 564.31 | 562.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 10:00:00 | 563.20 | 564.31 | 562.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 563.10 | 564.07 | 562.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:00:00 | 563.10 | 564.07 | 562.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 563.30 | 563.80 | 562.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-23 13:30:00 | 564.10 | 563.85 | 562.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 14:15:00 | 562.25 | 563.53 | 562.89 | SL hit (close<static) qty=1.00 sl=562.50 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 09:15:00 | 575.10 | 583.20 | 583.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 571.00 | 575.03 | 577.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 13:15:00 | 577.00 | 574.93 | 576.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 13:15:00 | 577.00 | 574.93 | 576.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 13:15:00 | 577.00 | 574.93 | 576.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 13:45:00 | 576.60 | 574.93 | 576.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 580.20 | 575.99 | 577.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-06 15:00:00 | 580.20 | 575.99 | 577.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 582.65 | 577.32 | 577.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 09:15:00 | 588.85 | 577.32 | 577.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 592.70 | 580.40 | 579.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 14:15:00 | 594.80 | 589.35 | 584.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 09:15:00 | 590.50 | 593.09 | 590.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 09:15:00 | 590.50 | 593.09 | 590.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 590.50 | 593.09 | 590.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:30:00 | 585.00 | 593.09 | 590.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 589.35 | 592.34 | 589.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 10:45:00 | 588.20 | 592.34 | 589.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 583.60 | 590.59 | 589.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 12:00:00 | 583.60 | 590.59 | 589.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 581.75 | 588.82 | 588.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:00:00 | 581.75 | 588.82 | 588.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 578.85 | 586.83 | 587.81 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 12:15:00 | 588.85 | 585.50 | 585.48 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 15:15:00 | 583.80 | 585.37 | 585.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 09:15:00 | 580.15 | 584.32 | 584.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 09:15:00 | 588.05 | 580.75 | 582.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 588.05 | 580.75 | 582.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 588.05 | 580.75 | 582.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 10:00:00 | 588.05 | 580.75 | 582.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 10:15:00 | 595.60 | 583.72 | 583.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 11:15:00 | 609.30 | 588.84 | 585.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 14:15:00 | 643.30 | 647.55 | 635.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-21 14:45:00 | 641.50 | 647.55 | 635.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 639.20 | 644.13 | 639.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 15:00:00 | 639.20 | 644.13 | 639.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 640.00 | 643.31 | 639.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 633.40 | 643.31 | 639.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 628.40 | 640.33 | 638.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:30:00 | 627.25 | 640.33 | 638.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 625.45 | 637.35 | 637.42 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 10:15:00 | 641.30 | 634.63 | 633.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 11:15:00 | 651.65 | 638.03 | 635.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 648.30 | 656.38 | 650.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 648.30 | 656.38 | 650.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 648.30 | 656.38 | 650.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 14:30:00 | 642.80 | 656.38 | 650.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 659.70 | 657.04 | 651.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:15:00 | 652.05 | 657.04 | 651.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 642.20 | 654.07 | 650.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-30 09:45:00 | 640.00 | 654.07 | 650.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 640.10 | 651.28 | 649.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 11:30:00 | 643.80 | 650.22 | 649.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-04 13:15:00 | 645.75 | 649.56 | 649.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 13:15:00 | 645.75 | 649.56 | 649.78 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 09:15:00 | 664.55 | 652.04 | 650.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 12:15:00 | 668.80 | 658.93 | 654.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 12:15:00 | 663.65 | 667.14 | 662.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 13:00:00 | 663.65 | 667.14 | 662.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 13:15:00 | 661.70 | 666.06 | 662.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 13:45:00 | 660.45 | 666.06 | 662.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 661.95 | 665.23 | 662.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 14:30:00 | 661.95 | 665.23 | 662.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 15:15:00 | 660.60 | 664.31 | 661.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:15:00 | 664.10 | 664.31 | 661.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 672.20 | 665.89 | 662.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 11:00:00 | 675.90 | 667.89 | 664.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 14:15:00 | 657.65 | 664.41 | 663.59 | SL hit (close<static) qty=1.00 sl=658.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 11:15:00 | 665.70 | 674.67 | 675.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 12:15:00 | 663.15 | 672.36 | 673.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 12:15:00 | 668.00 | 667.74 | 670.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-17 13:00:00 | 668.00 | 667.74 | 670.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 653.65 | 657.07 | 660.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 14:15:00 | 641.95 | 656.55 | 658.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 09:30:00 | 650.75 | 654.28 | 656.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 09:15:00 | 651.50 | 653.87 | 654.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 11:15:00 | 657.40 | 655.30 | 655.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 11:15:00 | 657.40 | 655.30 | 655.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 13:15:00 | 659.40 | 656.54 | 655.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 10:15:00 | 659.45 | 661.93 | 660.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 10:15:00 | 659.45 | 661.93 | 660.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 659.45 | 661.93 | 660.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:45:00 | 659.20 | 661.93 | 660.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 11:15:00 | 655.10 | 660.57 | 659.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 12:00:00 | 655.10 | 660.57 | 659.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 658.40 | 659.32 | 659.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:15:00 | 646.65 | 659.32 | 659.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 09:15:00 | 650.60 | 657.58 | 658.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 09:15:00 | 642.60 | 648.50 | 652.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 14:15:00 | 642.50 | 641.92 | 644.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-02 15:00:00 | 642.50 | 641.92 | 644.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 637.35 | 635.84 | 639.20 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 644.60 | 639.92 | 639.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 14:15:00 | 649.05 | 644.11 | 642.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 13:15:00 | 644.00 | 646.92 | 644.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 13:15:00 | 644.00 | 646.92 | 644.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 13:15:00 | 644.00 | 646.92 | 644.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:00:00 | 644.00 | 646.92 | 644.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 644.20 | 646.38 | 644.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:30:00 | 643.20 | 646.38 | 644.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 647.20 | 646.54 | 644.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 09:15:00 | 650.10 | 646.54 | 644.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 09:15:00 | 643.40 | 646.15 | 645.80 | SL hit (close<static) qty=1.00 sl=644.20 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 10:15:00 | 640.50 | 645.02 | 645.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 634.15 | 639.25 | 641.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 13:15:00 | 637.00 | 634.18 | 637.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 13:15:00 | 637.00 | 634.18 | 637.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 637.00 | 634.18 | 637.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 13:45:00 | 637.30 | 634.18 | 637.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 632.25 | 633.79 | 636.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 15:15:00 | 631.40 | 633.79 | 636.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 09:15:00 | 645.50 | 629.06 | 627.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 645.50 | 629.06 | 627.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 10:15:00 | 652.40 | 633.73 | 629.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 636.35 | 637.00 | 632.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 14:45:00 | 635.35 | 637.00 | 632.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 634.80 | 636.19 | 632.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:30:00 | 633.55 | 636.19 | 632.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 632.20 | 635.39 | 632.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 11:00:00 | 632.20 | 635.39 | 632.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 630.85 | 634.49 | 632.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 12:00:00 | 630.85 | 634.49 | 632.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 631.75 | 633.94 | 632.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 13:15:00 | 632.30 | 633.94 | 632.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 14:00:00 | 632.10 | 633.57 | 632.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 14:45:00 | 633.40 | 633.42 | 632.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 15:15:00 | 631.00 | 632.70 | 632.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 631.00 | 632.70 | 632.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 629.65 | 631.78 | 632.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 13:15:00 | 628.45 | 628.44 | 629.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 13:15:00 | 628.45 | 628.44 | 629.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 628.45 | 628.44 | 629.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 13:30:00 | 629.00 | 628.44 | 629.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 626.80 | 627.95 | 629.16 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 634.20 | 630.27 | 629.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 636.75 | 632.31 | 630.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 10:15:00 | 636.15 | 636.35 | 634.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 10:15:00 | 636.15 | 636.35 | 634.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 636.15 | 636.35 | 634.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:00:00 | 636.15 | 636.35 | 634.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 643.20 | 642.44 | 639.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:45:00 | 641.70 | 642.44 | 639.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 641.00 | 642.38 | 640.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:30:00 | 639.50 | 641.35 | 639.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 10:15:00 | 636.00 | 640.28 | 639.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 11:00:00 | 636.00 | 640.28 | 639.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 12:15:00 | 637.35 | 639.03 | 639.11 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 14:15:00 | 640.05 | 639.29 | 639.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 09:15:00 | 642.45 | 640.11 | 639.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 11:15:00 | 637.45 | 639.69 | 639.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 11:15:00 | 637.45 | 639.69 | 639.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 11:15:00 | 637.45 | 639.69 | 639.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 11:45:00 | 637.35 | 639.69 | 639.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 640.40 | 639.83 | 639.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:30:00 | 636.95 | 639.83 | 639.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 14:15:00 | 641.65 | 640.74 | 640.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 14:45:00 | 641.00 | 640.74 | 640.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 15:15:00 | 641.45 | 640.88 | 640.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:15:00 | 645.50 | 640.88 | 640.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 650.00 | 642.71 | 641.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 13:30:00 | 651.70 | 647.08 | 644.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-13 09:15:00 | 646.35 | 660.11 | 660.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 09:15:00 | 646.35 | 660.11 | 660.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-14 12:15:00 | 645.70 | 650.10 | 653.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-15 11:15:00 | 646.25 | 646.01 | 649.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-15 12:00:00 | 646.25 | 646.01 | 649.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 660.15 | 648.52 | 649.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 10:00:00 | 660.15 | 648.52 | 649.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 661.75 | 651.16 | 650.50 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 13:15:00 | 648.65 | 652.67 | 652.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 13:15:00 | 645.00 | 648.63 | 650.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 648.55 | 647.75 | 649.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 648.55 | 647.75 | 649.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 648.55 | 647.75 | 649.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 648.00 | 647.75 | 649.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 654.20 | 649.04 | 649.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:45:00 | 655.05 | 649.04 | 649.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 11:15:00 | 653.05 | 649.84 | 650.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 13:15:00 | 649.70 | 650.29 | 650.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 15:15:00 | 648.30 | 645.70 | 645.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 15:15:00 | 648.30 | 645.70 | 645.63 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 09:15:00 | 644.05 | 645.37 | 645.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 11:15:00 | 637.70 | 643.43 | 644.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 14:15:00 | 637.05 | 636.04 | 638.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 15:00:00 | 637.05 | 636.04 | 638.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 639.40 | 636.71 | 638.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 09:15:00 | 632.95 | 636.71 | 638.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 11:45:00 | 634.50 | 634.96 | 637.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 12:15:00 | 634.40 | 629.15 | 628.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 12:15:00 | 634.40 | 629.15 | 628.97 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 625.10 | 628.42 | 628.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 11:15:00 | 622.15 | 627.16 | 628.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 11:15:00 | 621.00 | 619.90 | 623.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 12:00:00 | 621.00 | 619.90 | 623.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 622.35 | 620.39 | 622.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 14:00:00 | 622.35 | 620.39 | 622.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 14:15:00 | 621.20 | 620.55 | 622.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 15:15:00 | 619.35 | 620.55 | 622.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 10:15:00 | 620.40 | 620.47 | 622.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 09:30:00 | 620.85 | 621.15 | 621.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 10:45:00 | 620.70 | 621.31 | 621.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 11:15:00 | 622.25 | 621.50 | 621.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-12 11:45:00 | 623.45 | 621.50 | 621.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-12 12:15:00 | 624.65 | 622.13 | 622.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 12:15:00 | 624.65 | 622.13 | 622.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 09:15:00 | 627.75 | 624.25 | 623.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 13:15:00 | 620.55 | 624.13 | 623.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 13:15:00 | 620.55 | 624.13 | 623.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 620.55 | 624.13 | 623.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:00:00 | 620.55 | 624.13 | 623.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 625.85 | 624.47 | 623.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 09:30:00 | 636.40 | 630.38 | 627.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 09:45:00 | 633.35 | 641.65 | 638.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 10:15:00 | 635.50 | 637.54 | 637.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 10:15:00 | 635.50 | 637.54 | 637.80 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 12:15:00 | 638.40 | 638.05 | 638.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-23 10:15:00 | 644.45 | 639.66 | 638.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 11:15:00 | 638.40 | 639.40 | 638.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 11:15:00 | 638.40 | 639.40 | 638.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 11:15:00 | 638.40 | 639.40 | 638.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 12:00:00 | 638.40 | 639.40 | 638.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 12:15:00 | 635.85 | 638.69 | 638.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 13:00:00 | 635.85 | 638.69 | 638.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 13:15:00 | 634.40 | 637.84 | 638.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 624.90 | 635.25 | 636.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 612.60 | 610.03 | 616.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 09:45:00 | 614.40 | 610.03 | 616.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 604.00 | 608.73 | 612.69 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 12:15:00 | 617.55 | 612.48 | 611.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 13:15:00 | 619.90 | 613.96 | 612.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 09:15:00 | 620.00 | 620.57 | 617.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-02 10:00:00 | 620.00 | 620.57 | 617.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 619.30 | 620.47 | 618.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:00:00 | 619.30 | 620.47 | 618.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 620.90 | 620.56 | 618.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:30:00 | 619.30 | 620.56 | 618.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 618.80 | 620.20 | 618.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 15:00:00 | 618.80 | 620.20 | 618.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 620.45 | 620.25 | 618.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 620.90 | 620.25 | 618.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:45:00 | 621.75 | 620.72 | 619.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 14:00:00 | 621.80 | 622.47 | 621.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 15:00:00 | 622.30 | 622.44 | 621.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 622.75 | 622.50 | 621.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:15:00 | 621.75 | 622.50 | 621.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 617.90 | 621.58 | 621.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-07 09:15:00 | 617.90 | 621.58 | 621.40 | SL hit (close<static) qty=1.00 sl=618.60 alert=retest2 |

### Cycle 37 — SELL (started 2023-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 10:15:00 | 619.30 | 621.12 | 621.21 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 14:15:00 | 622.60 | 621.09 | 621.08 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 09:15:00 | 619.85 | 620.83 | 620.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-08 10:15:00 | 618.25 | 620.31 | 620.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 12:15:00 | 621.05 | 620.17 | 620.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 12:15:00 | 621.05 | 620.17 | 620.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 621.05 | 620.17 | 620.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:45:00 | 620.30 | 620.17 | 620.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 619.95 | 620.13 | 620.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 13:30:00 | 622.05 | 620.13 | 620.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 620.75 | 619.47 | 620.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:00:00 | 620.75 | 619.47 | 620.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 621.10 | 619.80 | 620.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 12:15:00 | 622.70 | 619.80 | 620.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 12:15:00 | 623.65 | 620.57 | 620.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 09:15:00 | 626.65 | 622.15 | 621.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 09:15:00 | 621.55 | 624.89 | 623.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 09:15:00 | 621.55 | 624.89 | 623.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 621.55 | 624.89 | 623.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:45:00 | 622.90 | 624.89 | 623.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 622.70 | 624.45 | 623.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 11:15:00 | 624.00 | 624.45 | 623.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 12:15:00 | 619.05 | 622.81 | 622.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 12:15:00 | 619.05 | 622.81 | 622.87 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 630.65 | 624.01 | 623.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 13:15:00 | 638.30 | 629.98 | 626.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 14:15:00 | 633.50 | 635.59 | 632.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 15:00:00 | 633.50 | 635.59 | 632.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 657.20 | 639.87 | 634.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 10:15:00 | 666.00 | 652.71 | 647.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 12:15:00 | 675.15 | 680.26 | 680.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 12:15:00 | 675.15 | 680.26 | 680.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 09:15:00 | 672.30 | 677.29 | 678.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 11:15:00 | 678.15 | 677.08 | 678.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 11:15:00 | 678.15 | 677.08 | 678.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 678.15 | 677.08 | 678.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-05 12:00:00 | 678.15 | 677.08 | 678.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 672.55 | 676.17 | 677.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-05 13:15:00 | 671.60 | 676.17 | 677.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 11:15:00 | 680.90 | 676.27 | 676.91 | SL hit (close>static) qty=1.00 sl=678.80 alert=retest2 |

### Cycle 44 — BUY (started 2023-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 15:15:00 | 677.95 | 677.31 | 677.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 09:15:00 | 681.50 | 678.15 | 677.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 14:15:00 | 678.10 | 680.42 | 679.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 14:15:00 | 678.10 | 680.42 | 679.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 14:15:00 | 678.10 | 680.42 | 679.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 15:00:00 | 678.10 | 680.42 | 679.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 15:15:00 | 680.50 | 680.44 | 679.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 10:00:00 | 676.40 | 679.63 | 679.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 10:15:00 | 672.40 | 678.18 | 678.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 670.45 | 676.41 | 677.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 15:15:00 | 674.10 | 672.17 | 673.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 15:15:00 | 674.10 | 672.17 | 673.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 674.10 | 672.17 | 673.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:15:00 | 687.30 | 672.17 | 673.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 686.75 | 675.08 | 674.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 13:15:00 | 702.00 | 685.45 | 680.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 10:15:00 | 690.95 | 693.17 | 686.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-13 11:00:00 | 690.95 | 693.17 | 686.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 688.05 | 693.99 | 690.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:30:00 | 682.45 | 693.99 | 690.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 685.50 | 692.29 | 689.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 12:15:00 | 685.20 | 692.29 | 689.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2023-12-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 14:15:00 | 684.65 | 688.39 | 688.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 09:15:00 | 671.80 | 684.69 | 686.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 669.85 | 669.17 | 672.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 669.85 | 669.17 | 672.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 669.85 | 669.17 | 672.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 12:45:00 | 667.10 | 667.81 | 670.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 10:15:00 | 649.40 | 643.97 | 643.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 10:15:00 | 649.40 | 643.97 | 643.85 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 12:15:00 | 645.70 | 646.49 | 646.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 10:15:00 | 643.75 | 645.37 | 645.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 09:15:00 | 642.85 | 641.46 | 643.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 642.85 | 641.46 | 643.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 642.85 | 641.46 | 643.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:30:00 | 642.45 | 641.46 | 643.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 645.40 | 642.48 | 643.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:30:00 | 645.20 | 642.48 | 643.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 12:15:00 | 643.55 | 642.69 | 643.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 12:30:00 | 645.00 | 642.69 | 643.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 13:15:00 | 647.50 | 643.65 | 643.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 13:45:00 | 645.90 | 643.65 | 643.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-01-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 14:15:00 | 647.55 | 644.43 | 644.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 09:15:00 | 649.50 | 646.02 | 645.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 12:15:00 | 646.40 | 646.42 | 645.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-05 13:00:00 | 646.40 | 646.42 | 645.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 644.20 | 645.97 | 645.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 644.20 | 645.97 | 645.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 647.10 | 646.20 | 645.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 650.90 | 646.26 | 645.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 12:15:00 | 644.25 | 645.42 | 645.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 644.25 | 645.42 | 645.43 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 13:15:00 | 647.30 | 645.79 | 645.60 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 09:15:00 | 642.45 | 644.96 | 645.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 14:15:00 | 641.05 | 644.40 | 644.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 09:15:00 | 647.35 | 644.29 | 644.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 647.35 | 644.29 | 644.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 647.35 | 644.29 | 644.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:30:00 | 648.10 | 644.29 | 644.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 10:15:00 | 649.10 | 645.25 | 645.14 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 642.70 | 644.74 | 644.92 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 648.30 | 645.13 | 644.93 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 13:15:00 | 640.85 | 644.71 | 644.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 10:15:00 | 639.05 | 642.98 | 643.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 10:15:00 | 614.70 | 613.58 | 619.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-17 10:30:00 | 614.55 | 613.58 | 619.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 616.45 | 610.68 | 612.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:00:00 | 616.45 | 610.68 | 612.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 614.25 | 611.39 | 613.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:30:00 | 616.00 | 611.39 | 613.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 615.30 | 613.93 | 613.87 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 10:15:00 | 612.30 | 613.70 | 613.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 12:15:00 | 610.70 | 612.85 | 613.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-29 10:15:00 | 581.60 | 581.59 | 586.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-29 11:00:00 | 581.60 | 581.59 | 586.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 573.85 | 578.48 | 582.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 11:00:00 | 569.25 | 574.51 | 576.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 11:15:00 | 586.20 | 576.84 | 577.78 | SL hit (close>static) qty=1.00 sl=583.60 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 12:15:00 | 587.60 | 579.00 | 578.68 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 578.25 | 580.43 | 580.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 10:15:00 | 574.15 | 579.18 | 580.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 577.90 | 572.10 | 575.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 577.90 | 572.10 | 575.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 577.90 | 572.10 | 575.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:00:00 | 577.90 | 572.10 | 575.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 582.05 | 574.09 | 575.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:45:00 | 583.50 | 574.09 | 575.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 12:15:00 | 585.30 | 578.13 | 577.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 13:15:00 | 592.25 | 580.95 | 578.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 12:15:00 | 603.25 | 603.62 | 597.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-08 13:15:00 | 599.90 | 603.62 | 597.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 595.25 | 601.95 | 597.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 13:45:00 | 593.85 | 601.95 | 597.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 591.15 | 599.79 | 596.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 15:00:00 | 591.15 | 599.79 | 596.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 591.50 | 598.13 | 596.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 586.80 | 598.13 | 596.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 587.35 | 594.18 | 594.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 578.00 | 585.88 | 589.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 588.05 | 580.45 | 584.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 588.05 | 580.45 | 584.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 588.05 | 580.45 | 584.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:00:00 | 588.05 | 580.45 | 584.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 587.30 | 581.82 | 584.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:15:00 | 587.70 | 581.82 | 584.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 588.25 | 583.11 | 584.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:30:00 | 590.40 | 583.11 | 584.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 14:15:00 | 589.40 | 586.39 | 585.99 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 10:15:00 | 580.95 | 585.39 | 585.67 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 586.85 | 585.10 | 585.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 12:15:00 | 589.90 | 586.80 | 586.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 588.20 | 588.83 | 587.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 588.20 | 588.83 | 587.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 588.20 | 588.83 | 587.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:45:00 | 587.10 | 588.83 | 587.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 587.00 | 588.46 | 587.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:30:00 | 587.45 | 588.46 | 587.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 586.95 | 588.16 | 587.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:30:00 | 587.20 | 588.16 | 587.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 12:15:00 | 586.30 | 587.79 | 587.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 13:00:00 | 586.30 | 587.79 | 587.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 585.00 | 587.00 | 586.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 15:00:00 | 585.00 | 587.00 | 586.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 15:15:00 | 585.60 | 586.72 | 586.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 09:15:00 | 582.95 | 585.97 | 586.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 11:15:00 | 585.95 | 585.81 | 586.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 11:15:00 | 585.95 | 585.81 | 586.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 585.95 | 585.81 | 586.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:45:00 | 585.00 | 585.81 | 586.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 12:15:00 | 583.90 | 585.43 | 586.09 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 15:15:00 | 589.45 | 586.74 | 586.57 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 11:15:00 | 585.40 | 586.53 | 586.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 12:15:00 | 582.55 | 585.74 | 586.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 15:15:00 | 576.85 | 574.73 | 578.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 09:15:00 | 579.20 | 574.73 | 578.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 579.75 | 575.74 | 578.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:45:00 | 580.55 | 575.74 | 578.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 580.75 | 576.74 | 578.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:00:00 | 580.75 | 576.74 | 578.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 15:15:00 | 580.80 | 579.63 | 579.56 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 09:15:00 | 574.95 | 578.70 | 579.14 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 590.15 | 581.14 | 580.02 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 576.85 | 581.67 | 581.92 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 588.40 | 582.92 | 582.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 12:15:00 | 594.75 | 587.06 | 584.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 11:15:00 | 590.30 | 590.65 | 588.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 11:15:00 | 590.30 | 590.65 | 588.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 590.30 | 590.65 | 588.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:45:00 | 588.60 | 590.65 | 588.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 615.50 | 595.62 | 590.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 13:15:00 | 617.35 | 595.62 | 590.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 09:15:00 | 618.20 | 610.88 | 606.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 11:45:00 | 616.95 | 613.27 | 608.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 13:00:00 | 616.85 | 613.99 | 609.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 628.50 | 622.04 | 617.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-13 12:15:00 | 612.65 | 617.31 | 617.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 12:15:00 | 612.65 | 617.31 | 617.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 15:15:00 | 605.00 | 613.05 | 615.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 615.20 | 612.55 | 614.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 11:15:00 | 615.20 | 612.55 | 614.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 615.20 | 612.55 | 614.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 615.20 | 612.55 | 614.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 615.50 | 613.14 | 614.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:15:00 | 619.80 | 613.14 | 614.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 618.35 | 614.18 | 615.09 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 622.20 | 615.78 | 615.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 11:15:00 | 626.95 | 619.54 | 617.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 09:15:00 | 630.45 | 630.94 | 627.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 630.45 | 630.94 | 627.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 630.45 | 630.94 | 627.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 09:30:00 | 627.00 | 630.94 | 627.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 631.05 | 631.56 | 628.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:45:00 | 629.60 | 631.56 | 628.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 624.40 | 630.20 | 628.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 624.40 | 630.20 | 628.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 625.20 | 629.20 | 628.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 624.70 | 629.20 | 628.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 629.60 | 628.98 | 628.52 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 14:15:00 | 626.55 | 627.92 | 628.08 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 631.60 | 628.51 | 628.31 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 13:15:00 | 625.05 | 628.21 | 628.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-21 14:15:00 | 622.65 | 627.09 | 627.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 11:15:00 | 625.15 | 624.60 | 626.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 11:15:00 | 625.15 | 624.60 | 626.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 625.15 | 624.60 | 626.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 11:45:00 | 626.25 | 624.60 | 626.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 12:15:00 | 628.25 | 625.33 | 626.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 13:00:00 | 628.25 | 625.33 | 626.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 13:15:00 | 626.75 | 625.61 | 626.38 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2024-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 10:15:00 | 630.75 | 626.83 | 626.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 10:15:00 | 635.50 | 629.72 | 628.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 13:15:00 | 629.90 | 630.72 | 629.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 13:15:00 | 629.90 | 630.72 | 629.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 629.90 | 630.72 | 629.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 14:00:00 | 629.90 | 630.72 | 629.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 626.15 | 629.81 | 629.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 626.15 | 629.81 | 629.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 625.95 | 629.04 | 628.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 629.55 | 629.04 | 628.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 09:15:00 | 623.05 | 630.94 | 631.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 09:15:00 | 623.05 | 630.94 | 631.89 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 15:15:00 | 629.50 | 625.47 | 625.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 630.25 | 626.43 | 625.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 14:15:00 | 633.50 | 634.57 | 631.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 15:00:00 | 633.50 | 634.57 | 631.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 626.95 | 632.94 | 631.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:00:00 | 626.95 | 632.94 | 631.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 624.15 | 631.18 | 630.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:00:00 | 624.15 | 631.18 | 630.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 11:15:00 | 622.45 | 629.43 | 630.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 15:15:00 | 621.00 | 624.93 | 627.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 12:15:00 | 615.65 | 615.60 | 619.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 12:45:00 | 615.70 | 615.60 | 619.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 619.45 | 609.87 | 612.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:30:00 | 621.00 | 609.87 | 612.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 617.10 | 612.29 | 613.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 613.40 | 613.29 | 613.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 582.73 | 607.89 | 610.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-19 13:15:00 | 603.80 | 603.52 | 607.56 | SL hit (close>ema200) qty=0.50 sl=603.52 alert=retest2 |

### Cycle 84 — BUY (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 09:15:00 | 559.05 | 550.18 | 549.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 568.20 | 559.04 | 555.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 14:15:00 | 562.05 | 562.59 | 558.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 14:30:00 | 564.60 | 562.59 | 558.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 560.00 | 561.79 | 559.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:30:00 | 558.50 | 561.79 | 559.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 558.70 | 561.17 | 559.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 558.80 | 561.17 | 559.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 556.80 | 560.30 | 558.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:30:00 | 556.70 | 560.30 | 558.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 556.45 | 559.53 | 558.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:45:00 | 556.40 | 559.53 | 558.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 555.95 | 558.20 | 558.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 553.00 | 556.80 | 557.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 558.75 | 556.58 | 557.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 558.75 | 556.58 | 557.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 558.75 | 556.58 | 557.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 558.75 | 556.58 | 557.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 559.95 | 557.26 | 557.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 559.95 | 557.26 | 557.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 558.70 | 557.54 | 557.63 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 566.65 | 559.37 | 558.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 569.10 | 562.39 | 560.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 567.80 | 568.41 | 565.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 567.80 | 568.41 | 565.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 567.80 | 568.41 | 565.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 565.35 | 568.41 | 565.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 565.70 | 567.86 | 565.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 565.70 | 567.86 | 565.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 564.45 | 567.18 | 565.52 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 11:15:00 | 563.20 | 564.76 | 564.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 13:15:00 | 561.45 | 563.86 | 564.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 563.50 | 562.48 | 563.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 11:15:00 | 563.50 | 562.48 | 563.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 563.50 | 562.48 | 563.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 564.35 | 562.48 | 563.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 563.70 | 562.72 | 563.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:45:00 | 564.85 | 562.72 | 563.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 568.50 | 563.88 | 563.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 568.50 | 563.88 | 563.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 565.95 | 564.29 | 564.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 571.00 | 566.48 | 565.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 13:15:00 | 566.15 | 566.88 | 565.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 13:15:00 | 566.15 | 566.88 | 565.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 566.15 | 566.88 | 565.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:00:00 | 566.15 | 566.88 | 565.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 565.10 | 566.53 | 565.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 565.10 | 566.53 | 565.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 565.20 | 566.26 | 565.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 566.70 | 566.26 | 565.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 566.75 | 566.12 | 565.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 14:15:00 | 564.05 | 566.32 | 566.00 | SL hit (close<static) qty=1.00 sl=564.40 alert=retest2 |

### Cycle 89 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 563.00 | 568.64 | 569.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 561.80 | 567.27 | 568.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 15:15:00 | 551.00 | 550.47 | 553.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 09:15:00 | 539.45 | 550.47 | 553.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 550.50 | 550.48 | 552.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:00:00 | 533.95 | 547.17 | 551.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:30:00 | 536.20 | 542.62 | 548.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:15:00 | 535.10 | 539.36 | 546.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:45:00 | 534.85 | 538.40 | 544.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 544.90 | 539.24 | 544.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 544.90 | 539.24 | 544.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 548.75 | 541.14 | 544.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 548.75 | 541.14 | 544.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 550.35 | 542.98 | 545.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 550.35 | 542.98 | 545.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 551.90 | 546.90 | 546.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 551.90 | 546.90 | 546.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 555.15 | 549.19 | 547.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 13:15:00 | 571.35 | 571.42 | 566.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:00:00 | 571.35 | 571.42 | 566.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 588.75 | 576.47 | 572.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:15:00 | 592.30 | 579.43 | 574.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:15:00 | 592.00 | 584.93 | 578.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 596.30 | 587.58 | 580.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 10:15:00 | 590.05 | 595.22 | 595.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 10:15:00 | 590.05 | 595.22 | 595.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 585.00 | 590.31 | 592.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 582.45 | 581.08 | 584.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 10:00:00 | 582.45 | 581.08 | 584.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 584.30 | 581.16 | 583.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:00:00 | 584.30 | 581.16 | 583.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 585.00 | 581.93 | 583.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:00:00 | 585.00 | 581.93 | 583.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 591.55 | 583.85 | 584.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 591.55 | 583.85 | 584.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 589.45 | 584.97 | 584.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 593.20 | 586.62 | 585.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 12:15:00 | 588.50 | 589.05 | 587.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 13:00:00 | 588.50 | 589.05 | 587.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 588.55 | 588.95 | 587.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:45:00 | 587.30 | 588.95 | 587.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 594.30 | 590.09 | 588.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 595.35 | 591.10 | 589.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 10:00:00 | 596.45 | 592.17 | 590.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 14:45:00 | 595.70 | 593.74 | 591.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 10:30:00 | 596.55 | 594.33 | 592.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 587.40 | 595.94 | 594.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 587.40 | 595.94 | 594.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 594.60 | 595.67 | 594.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 11:45:00 | 595.35 | 594.87 | 594.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 589.55 | 593.80 | 593.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 589.55 | 593.80 | 593.88 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 597.30 | 594.13 | 593.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 603.70 | 596.76 | 595.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 15:15:00 | 621.55 | 621.97 | 616.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 09:15:00 | 622.65 | 621.97 | 616.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 622.05 | 621.93 | 617.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 619.15 | 621.93 | 617.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 629.50 | 632.38 | 628.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 628.50 | 632.38 | 628.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 637.55 | 635.22 | 632.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:00:00 | 639.00 | 636.18 | 633.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 647.10 | 638.16 | 635.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 10:30:00 | 639.30 | 638.46 | 635.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 637.25 | 642.02 | 642.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 637.25 | 642.02 | 642.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 635.60 | 640.74 | 641.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 640.00 | 639.63 | 640.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:00:00 | 640.00 | 639.63 | 640.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 637.90 | 639.29 | 640.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:30:00 | 636.25 | 638.88 | 640.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:00:00 | 636.25 | 638.35 | 639.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:45:00 | 632.85 | 636.79 | 638.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 624.20 | 637.63 | 638.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 637.10 | 637.52 | 638.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:45:00 | 641.20 | 637.52 | 638.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 642.90 | 638.60 | 639.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-23 13:15:00 | 642.90 | 638.60 | 639.04 | SL hit (close>static) qty=1.00 sl=640.85 alert=retest2 |

### Cycle 96 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 644.50 | 639.78 | 639.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 663.70 | 644.92 | 641.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 696.20 | 697.71 | 685.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:45:00 | 697.65 | 697.71 | 685.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 686.15 | 695.96 | 690.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 686.15 | 695.96 | 690.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 696.35 | 696.03 | 691.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:15:00 | 699.60 | 696.03 | 691.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 698.95 | 698.36 | 694.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 698.40 | 706.80 | 707.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 698.40 | 706.80 | 707.59 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 15:15:00 | 708.90 | 707.24 | 707.21 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 10:15:00 | 696.90 | 705.23 | 706.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 688.00 | 699.39 | 703.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 692.70 | 692.32 | 697.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:30:00 | 692.45 | 692.32 | 697.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 696.95 | 692.31 | 696.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 696.95 | 692.31 | 696.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 695.85 | 693.02 | 695.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 692.00 | 693.02 | 695.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 691.90 | 692.67 | 695.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 703.60 | 694.86 | 696.06 | SL hit (close>static) qty=1.00 sl=697.15 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 12:15:00 | 709.90 | 697.87 | 697.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 13:15:00 | 711.20 | 700.53 | 698.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 704.15 | 705.27 | 702.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 704.15 | 705.27 | 702.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 701.90 | 704.91 | 702.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 701.90 | 704.91 | 702.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 702.60 | 704.45 | 702.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 699.20 | 704.45 | 702.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 695.45 | 702.65 | 702.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 695.45 | 702.65 | 702.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 703.80 | 703.30 | 702.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:45:00 | 702.60 | 703.30 | 702.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 702.15 | 703.41 | 702.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 702.15 | 703.41 | 702.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 701.00 | 702.93 | 702.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 697.10 | 702.93 | 702.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 693.05 | 700.96 | 701.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 10:15:00 | 691.30 | 699.02 | 700.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 688.55 | 686.90 | 691.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 13:00:00 | 688.55 | 686.90 | 691.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 683.35 | 682.99 | 687.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 680.70 | 686.22 | 687.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 11:00:00 | 681.05 | 684.59 | 686.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 696.95 | 687.44 | 687.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 696.95 | 687.44 | 687.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 701.40 | 690.23 | 688.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 10:15:00 | 726.95 | 727.33 | 723.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 11:00:00 | 726.95 | 727.33 | 723.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 723.90 | 726.55 | 723.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 723.90 | 726.55 | 723.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 725.90 | 726.42 | 724.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 728.80 | 726.59 | 724.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:45:00 | 729.15 | 727.63 | 725.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 743.55 | 750.72 | 751.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 743.55 | 750.72 | 751.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 737.35 | 744.79 | 747.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 747.00 | 744.19 | 746.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 10:15:00 | 747.00 | 744.19 | 746.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 747.00 | 744.19 | 746.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 747.00 | 744.19 | 746.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 745.35 | 744.42 | 746.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 745.75 | 744.42 | 746.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 710.65 | 704.63 | 710.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 710.65 | 704.63 | 710.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 715.35 | 706.77 | 710.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 715.35 | 706.77 | 710.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 711.65 | 707.75 | 710.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 09:30:00 | 708.70 | 707.55 | 710.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 706.05 | 701.56 | 701.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 706.05 | 701.56 | 701.33 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 699.25 | 701.18 | 701.20 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 718.75 | 704.52 | 702.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 721.90 | 713.11 | 708.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 719.10 | 721.03 | 715.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 10:00:00 | 719.10 | 721.03 | 715.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 722.55 | 720.88 | 717.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:15:00 | 726.95 | 718.00 | 717.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:00:00 | 727.05 | 719.81 | 718.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:30:00 | 726.65 | 721.80 | 719.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:30:00 | 727.70 | 722.51 | 719.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 730.75 | 731.90 | 727.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 730.75 | 731.90 | 727.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 725.85 | 730.87 | 727.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 725.85 | 730.87 | 727.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 722.15 | 729.13 | 727.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 722.15 | 729.13 | 727.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 723.35 | 727.97 | 727.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 717.80 | 725.94 | 726.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 717.80 | 725.94 | 726.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 714.00 | 720.55 | 723.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 708.05 | 706.48 | 711.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:45:00 | 709.40 | 706.48 | 711.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 712.95 | 707.46 | 710.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 712.95 | 707.46 | 710.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 709.30 | 707.83 | 710.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:30:00 | 705.60 | 707.16 | 710.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 716.00 | 709.47 | 710.48 | SL hit (close>static) qty=1.00 sl=713.50 alert=retest2 |

### Cycle 108 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 718.70 | 709.12 | 708.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 722.80 | 711.86 | 709.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 717.00 | 717.05 | 713.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 09:45:00 | 718.25 | 717.05 | 713.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 715.40 | 716.38 | 714.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 713.95 | 716.38 | 714.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 720.60 | 720.30 | 717.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 718.65 | 720.30 | 717.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 715.15 | 734.51 | 731.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 715.15 | 734.51 | 731.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 714.05 | 730.42 | 730.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 727.00 | 730.42 | 730.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 723.40 | 729.02 | 729.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 723.40 | 729.02 | 729.42 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 737.00 | 728.89 | 727.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 11:15:00 | 739.20 | 730.96 | 728.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 15:15:00 | 742.45 | 744.26 | 739.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 09:15:00 | 746.85 | 744.26 | 739.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 747.60 | 744.93 | 740.46 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 726.95 | 738.82 | 739.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 723.60 | 734.32 | 736.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 15:15:00 | 710.00 | 709.16 | 715.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 09:15:00 | 710.30 | 709.16 | 715.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 713.50 | 710.03 | 715.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:45:00 | 715.20 | 710.03 | 715.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 720.60 | 712.14 | 715.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 720.60 | 712.14 | 715.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 716.25 | 712.96 | 715.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 718.60 | 712.96 | 715.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 716.45 | 713.66 | 716.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:30:00 | 716.70 | 713.66 | 716.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 715.65 | 714.06 | 715.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:45:00 | 717.60 | 714.06 | 715.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 718.15 | 714.88 | 716.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:45:00 | 720.10 | 714.88 | 716.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 719.70 | 715.84 | 716.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:15:00 | 716.25 | 715.84 | 716.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 10:15:00 | 723.90 | 717.65 | 717.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 10:15:00 | 723.90 | 717.65 | 717.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 11:15:00 | 728.45 | 719.81 | 718.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 731.55 | 733.71 | 728.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:00:00 | 731.55 | 733.71 | 728.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 729.00 | 732.77 | 728.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 729.00 | 732.77 | 728.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 726.50 | 731.51 | 728.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 726.50 | 731.51 | 728.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 725.00 | 730.21 | 728.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 718.10 | 730.21 | 728.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 721.15 | 726.56 | 726.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 711.45 | 719.53 | 722.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 724.15 | 707.40 | 712.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 724.15 | 707.40 | 712.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 724.15 | 707.40 | 712.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 724.15 | 707.40 | 712.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 716.75 | 709.27 | 712.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 715.50 | 709.27 | 712.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 09:30:00 | 711.70 | 710.53 | 712.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 09:15:00 | 717.10 | 712.03 | 711.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 09:15:00 | 717.10 | 712.03 | 711.46 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 707.40 | 710.81 | 711.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 703.25 | 709.30 | 710.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 710.00 | 709.44 | 710.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 710.00 | 709.44 | 710.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 710.00 | 709.44 | 710.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 710.00 | 709.44 | 710.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 707.90 | 709.13 | 710.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 710.85 | 709.13 | 710.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 710.10 | 709.33 | 710.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:45:00 | 707.35 | 708.19 | 709.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:15:00 | 707.75 | 707.93 | 709.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 706.90 | 708.00 | 708.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:15:00 | 671.98 | 683.99 | 688.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:15:00 | 672.36 | 683.99 | 688.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:15:00 | 671.55 | 683.99 | 688.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-21 14:15:00 | 675.35 | 675.30 | 679.92 | SL hit (close>ema200) qty=0.50 sl=675.30 alert=retest2 |

### Cycle 116 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 689.45 | 682.51 | 681.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 690.05 | 684.02 | 682.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 683.25 | 686.66 | 684.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 12:15:00 | 683.25 | 686.66 | 684.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 683.25 | 686.66 | 684.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:00:00 | 683.25 | 686.66 | 684.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 688.35 | 686.99 | 685.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:30:00 | 689.60 | 686.80 | 685.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 10:30:00 | 690.95 | 687.55 | 685.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:00:00 | 689.55 | 687.95 | 686.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 14:15:00 | 682.90 | 686.66 | 686.10 | SL hit (close<static) qty=1.00 sl=683.10 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 681.50 | 685.63 | 685.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 681.00 | 684.00 | 684.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 688.30 | 684.45 | 684.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 688.30 | 684.45 | 684.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 688.30 | 684.45 | 684.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:30:00 | 681.25 | 683.84 | 684.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:45:00 | 669.90 | 678.56 | 682.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 13:15:00 | 647.19 | 671.58 | 678.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 658.00 | 657.85 | 666.09 | SL hit (close>ema200) qty=0.50 sl=657.85 alert=retest2 |

### Cycle 118 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 634.60 | 632.02 | 631.95 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 628.65 | 631.89 | 632.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 627.80 | 631.07 | 631.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 623.40 | 622.38 | 624.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 623.40 | 622.38 | 624.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 624.10 | 622.73 | 624.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 619.65 | 622.73 | 624.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 623.00 | 622.57 | 624.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:00:00 | 623.00 | 622.40 | 623.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:45:00 | 622.65 | 622.39 | 623.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 623.75 | 622.66 | 623.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 617.20 | 622.66 | 623.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:00:00 | 620.85 | 622.24 | 623.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:15:00 | 620.50 | 622.01 | 622.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 11:15:00 | 626.15 | 623.18 | 623.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 626.15 | 623.18 | 623.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 10:15:00 | 628.55 | 624.65 | 623.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 12:15:00 | 624.35 | 624.81 | 624.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 12:15:00 | 624.35 | 624.81 | 624.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 624.35 | 624.81 | 624.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:45:00 | 624.35 | 624.81 | 624.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 626.10 | 625.07 | 624.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 625.40 | 625.07 | 624.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 624.30 | 625.23 | 624.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 624.30 | 625.23 | 624.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 624.35 | 625.05 | 624.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:45:00 | 623.20 | 625.05 | 624.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 624.70 | 624.98 | 624.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 623.55 | 624.98 | 624.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 622.95 | 624.58 | 624.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 624.10 | 624.58 | 624.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 621.65 | 623.99 | 624.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 15:15:00 | 620.80 | 623.35 | 623.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 616.85 | 616.74 | 619.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 13:00:00 | 616.85 | 616.74 | 619.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 618.75 | 617.18 | 618.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 11:00:00 | 618.75 | 617.18 | 618.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 618.25 | 617.39 | 618.46 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 619.95 | 618.36 | 618.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 620.35 | 618.76 | 618.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 616.80 | 618.56 | 618.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 616.80 | 618.56 | 618.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 616.80 | 618.56 | 618.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 616.80 | 618.56 | 618.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 617.35 | 618.32 | 618.35 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 11:15:00 | 622.20 | 619.10 | 618.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 14:15:00 | 623.35 | 620.79 | 619.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 615.30 | 620.20 | 619.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 615.30 | 620.20 | 619.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 615.30 | 620.20 | 619.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 614.60 | 620.20 | 619.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 609.50 | 618.06 | 618.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 605.90 | 613.33 | 616.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 612.20 | 610.40 | 613.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 11:15:00 | 612.20 | 610.40 | 613.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 612.20 | 610.40 | 613.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 612.30 | 610.40 | 613.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 619.45 | 612.21 | 613.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:00:00 | 619.45 | 612.21 | 613.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 618.40 | 613.45 | 614.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:15:00 | 619.20 | 613.45 | 614.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 15:15:00 | 619.95 | 615.76 | 615.24 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 613.15 | 616.07 | 616.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 604.95 | 612.59 | 614.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 12:15:00 | 613.40 | 612.50 | 613.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 12:15:00 | 613.40 | 612.50 | 613.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 613.40 | 612.50 | 613.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 13:15:00 | 612.65 | 612.50 | 613.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 13:15:00 | 611.65 | 612.33 | 613.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 14:15:00 | 610.40 | 612.33 | 613.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 653.10 | 605.86 | 602.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 653.10 | 605.86 | 602.89 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 627.65 | 630.98 | 631.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 624.55 | 629.09 | 630.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 619.45 | 619.12 | 622.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 12:15:00 | 624.60 | 620.53 | 622.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 624.60 | 620.53 | 622.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 623.75 | 620.53 | 622.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 623.15 | 621.05 | 622.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:45:00 | 621.35 | 621.04 | 622.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 620.85 | 620.57 | 621.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 621.05 | 620.67 | 621.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 11:15:00 | 622.55 | 614.09 | 613.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 622.55 | 614.09 | 613.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 627.85 | 617.88 | 615.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 634.60 | 634.75 | 629.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 634.60 | 634.75 | 629.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 599.85 | 631.39 | 630.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 599.85 | 631.39 | 630.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 13:15:00 | 620.50 | 629.21 | 629.87 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 635.60 | 626.20 | 625.31 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 624.50 | 632.28 | 632.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 622.95 | 629.24 | 630.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 627.30 | 621.25 | 625.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 627.30 | 621.25 | 625.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 627.30 | 621.25 | 625.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 627.30 | 621.25 | 625.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 633.05 | 623.61 | 625.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 632.25 | 623.61 | 625.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 629.15 | 627.57 | 627.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 636.20 | 629.61 | 628.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 13:15:00 | 629.30 | 632.51 | 630.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 13:15:00 | 629.30 | 632.51 | 630.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 629.30 | 632.51 | 630.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:00:00 | 629.30 | 632.51 | 630.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 629.50 | 631.90 | 630.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 632.75 | 631.38 | 630.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 09:15:00 | 627.75 | 630.66 | 630.08 | SL hit (close<static) qty=1.00 sl=628.45 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 624.55 | 628.73 | 629.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 623.70 | 627.72 | 628.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 622.70 | 622.47 | 624.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:15:00 | 624.00 | 622.47 | 624.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 624.00 | 622.77 | 624.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 615.90 | 622.77 | 624.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 615.25 | 621.27 | 623.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 612.60 | 619.54 | 622.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 15:15:00 | 623.50 | 621.39 | 621.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 623.50 | 621.39 | 621.24 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 10:15:00 | 619.00 | 620.89 | 621.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 12:15:00 | 617.75 | 620.14 | 620.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 14:15:00 | 620.55 | 620.01 | 620.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 14:15:00 | 620.55 | 620.01 | 620.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 620.55 | 620.01 | 620.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 620.55 | 620.01 | 620.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 620.40 | 620.08 | 620.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 620.90 | 620.08 | 620.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 614.40 | 618.95 | 619.94 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 14:15:00 | 622.95 | 620.20 | 620.14 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 617.70 | 619.96 | 620.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 615.30 | 617.43 | 618.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 13:15:00 | 620.85 | 617.25 | 618.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 13:15:00 | 620.85 | 617.25 | 618.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 620.85 | 617.25 | 618.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 620.85 | 617.25 | 618.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 622.50 | 618.30 | 618.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:15:00 | 620.00 | 618.30 | 618.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 15:15:00 | 620.00 | 618.64 | 618.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 15:15:00 | 620.00 | 618.64 | 618.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 10:15:00 | 622.75 | 619.94 | 619.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 12:15:00 | 620.05 | 620.10 | 619.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 12:15:00 | 620.05 | 620.10 | 619.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 620.05 | 620.10 | 619.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:30:00 | 619.95 | 620.10 | 619.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 618.25 | 619.73 | 619.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:45:00 | 618.15 | 619.73 | 619.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 618.90 | 619.56 | 619.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:15:00 | 615.00 | 619.56 | 619.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 15:15:00 | 615.00 | 618.65 | 618.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 603.95 | 615.71 | 617.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 614.50 | 611.29 | 613.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 614.50 | 611.29 | 613.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 614.50 | 611.29 | 613.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:30:00 | 618.10 | 611.29 | 613.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 611.80 | 611.39 | 613.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:30:00 | 615.10 | 611.39 | 613.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 11:15:00 | 615.85 | 612.28 | 613.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:00:00 | 615.85 | 612.28 | 613.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 616.10 | 613.05 | 614.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:45:00 | 616.05 | 613.05 | 614.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 618.10 | 614.68 | 614.64 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 11:15:00 | 613.25 | 614.61 | 614.69 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 616.20 | 614.93 | 614.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 620.30 | 616.47 | 615.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 15:15:00 | 618.70 | 619.77 | 618.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 09:15:00 | 615.10 | 618.83 | 617.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 615.10 | 618.83 | 617.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:45:00 | 615.80 | 618.83 | 617.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 616.60 | 618.39 | 617.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 13:00:00 | 617.85 | 617.91 | 617.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:15:00 | 617.60 | 617.71 | 617.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 631.55 | 617.75 | 617.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 12:15:00 | 622.80 | 628.83 | 629.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 622.80 | 628.83 | 629.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 621.85 | 626.40 | 628.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 627.40 | 625.68 | 627.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 627.40 | 625.68 | 627.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 627.40 | 625.68 | 627.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 631.00 | 625.68 | 627.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 626.15 | 625.78 | 627.37 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 629.30 | 628.18 | 628.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 632.75 | 629.09 | 628.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 675.35 | 679.20 | 675.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 11:15:00 | 675.35 | 679.20 | 675.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 675.35 | 679.20 | 675.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:00:00 | 675.35 | 679.20 | 675.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 676.25 | 678.61 | 675.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 675.85 | 678.61 | 675.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 678.65 | 678.62 | 675.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 676.65 | 678.62 | 675.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 672.35 | 677.36 | 675.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 672.35 | 677.36 | 675.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 672.90 | 676.47 | 675.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 677.55 | 676.47 | 675.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 673.20 | 675.89 | 675.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 673.20 | 675.89 | 675.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 669.55 | 674.62 | 674.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 669.55 | 674.62 | 674.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 666.55 | 673.00 | 673.78 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 681.20 | 673.93 | 673.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 11:15:00 | 681.95 | 675.53 | 674.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 12:15:00 | 692.50 | 693.47 | 689.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 12:45:00 | 690.30 | 693.47 | 689.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 690.45 | 693.69 | 690.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 12:30:00 | 696.25 | 693.85 | 691.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 14:15:00 | 690.45 | 691.58 | 691.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 14:15:00 | 690.45 | 691.58 | 691.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 675.20 | 687.40 | 689.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 677.15 | 671.51 | 678.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 677.15 | 671.51 | 678.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 677.15 | 671.51 | 678.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 675.10 | 671.51 | 678.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 674.10 | 672.03 | 677.80 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 685.00 | 679.92 | 679.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 691.30 | 683.79 | 681.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 684.75 | 685.21 | 683.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 13:15:00 | 684.75 | 685.21 | 683.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 684.75 | 685.21 | 683.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:00:00 | 684.75 | 685.21 | 683.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 708.75 | 715.53 | 710.93 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 707.25 | 711.66 | 711.81 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 11:15:00 | 715.55 | 711.91 | 711.68 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 708.00 | 710.86 | 711.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 15:15:00 | 706.15 | 709.92 | 710.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 706.70 | 705.02 | 707.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 14:00:00 | 706.70 | 705.02 | 707.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 704.50 | 704.92 | 707.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:45:00 | 706.25 | 704.92 | 707.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 704.35 | 704.45 | 706.64 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 710.55 | 706.33 | 706.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 733.90 | 716.02 | 711.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 731.95 | 737.02 | 728.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 12:00:00 | 731.95 | 737.02 | 728.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 729.60 | 734.43 | 728.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:00:00 | 729.60 | 734.43 | 728.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 727.90 | 733.12 | 728.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:30:00 | 729.25 | 733.12 | 728.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 726.50 | 731.80 | 728.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 733.75 | 731.80 | 728.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 11:45:00 | 728.60 | 734.65 | 733.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 12:15:00 | 721.15 | 731.95 | 732.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 721.15 | 731.95 | 732.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 718.05 | 722.61 | 724.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 716.45 | 715.78 | 719.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 730.50 | 715.78 | 719.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 734.25 | 719.47 | 720.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 735.60 | 719.47 | 720.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 733.65 | 722.31 | 721.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 737.70 | 731.68 | 727.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 735.00 | 735.34 | 731.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 735.00 | 735.34 | 731.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 735.50 | 740.30 | 737.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 735.50 | 740.30 | 737.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 737.25 | 739.69 | 737.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 747.60 | 741.31 | 738.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 13:15:00 | 776.10 | 780.61 | 780.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 776.10 | 780.61 | 780.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 772.55 | 777.74 | 779.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 761.00 | 759.95 | 765.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:45:00 | 763.45 | 759.95 | 765.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 765.00 | 760.93 | 764.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 765.00 | 760.93 | 764.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 762.35 | 761.21 | 764.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:15:00 | 761.05 | 761.21 | 764.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:45:00 | 761.60 | 761.18 | 764.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 760.75 | 761.15 | 762.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:45:00 | 761.10 | 761.34 | 762.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 758.45 | 760.71 | 761.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 751.70 | 757.18 | 759.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 751.90 | 756.27 | 757.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 13:15:00 | 760.15 | 758.20 | 757.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 760.15 | 758.20 | 757.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 762.95 | 759.15 | 758.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 15:15:00 | 763.50 | 764.54 | 762.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 15:15:00 | 763.50 | 764.54 | 762.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 763.50 | 764.54 | 762.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 767.10 | 764.54 | 762.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 766.25 | 764.59 | 762.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 759.85 | 763.36 | 762.33 | SL hit (close<static) qty=1.00 sl=762.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 754.50 | 761.06 | 761.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 751.00 | 756.98 | 759.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 753.45 | 751.68 | 755.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 753.45 | 751.68 | 755.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 757.80 | 752.96 | 755.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 760.90 | 752.96 | 755.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 763.70 | 755.11 | 755.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 763.70 | 755.11 | 755.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 767.90 | 757.67 | 757.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 770.00 | 760.13 | 758.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 769.85 | 771.97 | 768.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 769.85 | 771.97 | 768.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 769.85 | 771.97 | 768.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 769.60 | 771.97 | 768.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 768.40 | 771.25 | 768.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 766.85 | 771.25 | 768.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 768.35 | 770.67 | 768.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 767.75 | 770.67 | 768.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 768.80 | 770.30 | 768.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:15:00 | 767.50 | 770.30 | 768.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 766.80 | 769.60 | 768.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 766.80 | 769.60 | 768.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 765.45 | 768.77 | 767.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 766.70 | 768.77 | 767.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 760.60 | 766.48 | 767.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 756.35 | 764.45 | 766.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 772.10 | 763.86 | 764.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 772.10 | 763.86 | 764.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 772.10 | 763.86 | 764.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 772.10 | 763.86 | 764.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 778.45 | 766.78 | 766.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 779.90 | 769.40 | 767.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 770.30 | 774.15 | 770.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 770.30 | 774.15 | 770.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 770.30 | 774.15 | 770.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:45:00 | 776.90 | 775.25 | 772.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 776.95 | 776.32 | 773.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 780.60 | 775.46 | 773.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 800.30 | 805.21 | 805.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 800.30 | 805.21 | 805.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 795.20 | 802.46 | 804.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 786.55 | 786.45 | 791.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 786.55 | 786.45 | 791.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 786.00 | 784.98 | 788.55 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 13:15:00 | 792.15 | 789.07 | 788.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 795.15 | 791.01 | 789.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 09:15:00 | 790.90 | 790.99 | 789.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 790.90 | 790.99 | 789.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 790.90 | 790.99 | 789.86 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 786.10 | 788.83 | 789.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 15:15:00 | 785.10 | 787.62 | 788.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 763.25 | 762.38 | 768.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 14:00:00 | 763.25 | 762.38 | 768.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 767.50 | 764.38 | 767.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:45:00 | 762.00 | 763.39 | 766.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:30:00 | 760.55 | 760.29 | 763.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 754.90 | 750.20 | 750.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 754.90 | 750.20 | 750.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 760.10 | 753.22 | 751.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 759.70 | 759.82 | 756.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:45:00 | 760.60 | 759.82 | 756.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 759.70 | 761.46 | 759.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 761.30 | 761.46 | 759.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 761.50 | 761.47 | 759.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 763.95 | 761.96 | 760.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 757.40 | 761.05 | 759.80 | SL hit (close<static) qty=1.00 sl=759.05 alert=retest2 |

### Cycle 167 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 753.00 | 758.98 | 759.05 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 12:15:00 | 764.70 | 759.79 | 759.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 10:15:00 | 767.85 | 763.50 | 761.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 761.05 | 763.19 | 761.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 12:15:00 | 761.05 | 763.19 | 761.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 761.05 | 763.19 | 761.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 761.05 | 763.19 | 761.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 758.95 | 762.34 | 761.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 758.95 | 762.34 | 761.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 761.40 | 762.06 | 761.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 760.35 | 762.06 | 761.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 763.10 | 762.27 | 761.62 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 14:15:00 | 756.00 | 760.97 | 761.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 15:15:00 | 755.15 | 759.81 | 760.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 760.45 | 759.94 | 760.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 760.45 | 759.94 | 760.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 760.45 | 759.94 | 760.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 760.45 | 759.94 | 760.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 752.25 | 758.40 | 759.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 749.95 | 756.34 | 758.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:45:00 | 749.20 | 755.52 | 756.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 749.25 | 752.53 | 755.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 13:15:00 | 750.75 | 743.52 | 743.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 13:15:00 | 750.75 | 743.52 | 743.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 09:15:00 | 754.90 | 748.34 | 745.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 755.05 | 757.57 | 754.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:00:00 | 755.05 | 757.57 | 754.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 758.00 | 757.66 | 754.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 12:45:00 | 759.65 | 758.13 | 755.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 789.90 | 792.98 | 793.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 789.90 | 792.98 | 793.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 788.35 | 791.43 | 792.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 779.00 | 778.11 | 782.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 779.00 | 778.11 | 782.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 786.65 | 779.82 | 782.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 786.65 | 779.82 | 782.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 783.25 | 780.51 | 782.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:30:00 | 780.80 | 780.37 | 782.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 12:30:00 | 779.95 | 778.26 | 778.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 782.30 | 779.07 | 778.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 782.30 | 779.07 | 778.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 794.40 | 782.61 | 780.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 783.80 | 784.01 | 781.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 783.80 | 784.01 | 781.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 778.70 | 782.95 | 781.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 778.70 | 782.95 | 781.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 784.35 | 783.23 | 781.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 780.00 | 783.23 | 781.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 783.70 | 783.32 | 781.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 777.50 | 783.32 | 781.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 776.10 | 781.88 | 781.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 775.85 | 781.88 | 781.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 774.70 | 780.44 | 780.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 763.55 | 776.15 | 778.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 15:15:00 | 756.90 | 756.54 | 760.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 757.15 | 756.54 | 760.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 757.80 | 756.21 | 759.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 759.05 | 756.21 | 759.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 758.90 | 756.75 | 759.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 758.90 | 756.75 | 759.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 757.30 | 756.86 | 759.20 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 771.15 | 761.01 | 760.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 774.00 | 763.61 | 761.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 770.95 | 773.40 | 770.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 770.95 | 773.40 | 770.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 770.95 | 773.40 | 770.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 770.95 | 773.40 | 770.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 775.20 | 773.76 | 771.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 777.60 | 773.76 | 771.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:30:00 | 775.90 | 776.36 | 774.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 775.65 | 777.27 | 775.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 773.75 | 774.90 | 775.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 773.75 | 774.90 | 775.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 770.55 | 773.72 | 774.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 775.40 | 770.37 | 771.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 775.40 | 770.37 | 771.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 775.40 | 770.37 | 771.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 776.00 | 770.37 | 771.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 783.65 | 773.03 | 773.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 11:15:00 | 788.30 | 776.08 | 774.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 12:15:00 | 782.65 | 784.50 | 780.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 12:45:00 | 782.95 | 784.50 | 780.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 780.65 | 783.59 | 780.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 780.65 | 783.59 | 780.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 785.00 | 783.87 | 781.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 794.00 | 783.87 | 781.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 11:45:00 | 785.50 | 785.56 | 782.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 12:30:00 | 785.45 | 786.18 | 783.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 780.70 | 785.74 | 784.19 | SL hit (close<static) qty=1.00 sl=780.75 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 774.95 | 782.19 | 782.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 769.05 | 776.33 | 779.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 15:15:00 | 768.00 | 766.36 | 770.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 09:15:00 | 768.15 | 766.36 | 770.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 770.05 | 767.09 | 770.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:45:00 | 771.30 | 767.09 | 770.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 769.50 | 767.58 | 770.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 771.55 | 767.58 | 770.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 769.00 | 767.86 | 770.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:30:00 | 766.55 | 767.69 | 769.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:45:00 | 766.20 | 765.96 | 768.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 765.50 | 760.34 | 760.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 765.50 | 760.34 | 760.06 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 758.05 | 759.68 | 759.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 755.00 | 758.71 | 759.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 14:15:00 | 763.30 | 759.31 | 759.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 14:15:00 | 763.30 | 759.31 | 759.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 763.30 | 759.31 | 759.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 763.30 | 759.31 | 759.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 763.30 | 760.11 | 759.77 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 756.85 | 759.55 | 759.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 755.70 | 758.78 | 759.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 15:15:00 | 750.30 | 749.67 | 753.02 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:15:00 | 743.20 | 749.67 | 753.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 750.80 | 748.09 | 751.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 750.80 | 748.09 | 751.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 752.55 | 748.98 | 751.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-09 12:15:00 | 752.55 | 748.98 | 751.42 | SL hit (close>ema400) qty=1.00 sl=751.42 alert=retest1 |

### Cycle 182 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 757.00 | 752.67 | 752.60 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 746.85 | 751.86 | 752.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 745.25 | 748.81 | 750.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 12:15:00 | 748.85 | 748.82 | 750.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:00:00 | 748.85 | 748.82 | 750.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 757.65 | 746.54 | 747.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 758.35 | 746.54 | 747.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 757.60 | 748.75 | 748.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 759.65 | 753.19 | 750.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 741.20 | 753.42 | 751.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 741.20 | 753.42 | 751.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 741.20 | 753.42 | 751.48 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 13:15:00 | 743.05 | 749.22 | 749.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 740.95 | 743.79 | 745.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 744.60 | 743.48 | 744.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 744.60 | 743.48 | 744.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 744.60 | 743.48 | 744.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 745.00 | 743.48 | 744.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 743.95 | 743.57 | 744.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 750.20 | 743.57 | 744.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 748.70 | 744.60 | 745.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 750.50 | 744.60 | 745.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 745.75 | 744.83 | 745.16 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 746.65 | 745.39 | 745.37 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 743.05 | 744.96 | 745.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 738.30 | 743.19 | 744.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 739.85 | 738.67 | 741.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 739.85 | 738.67 | 741.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 739.85 | 738.67 | 741.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 733.85 | 738.20 | 740.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:30:00 | 734.90 | 737.50 | 739.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 745.10 | 739.12 | 739.74 | SL hit (close>static) qty=1.00 sl=743.55 alert=retest2 |

### Cycle 188 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 744.40 | 740.18 | 740.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 748.25 | 743.49 | 741.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 749.50 | 755.04 | 750.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 749.50 | 755.04 | 750.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 749.50 | 755.04 | 750.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 749.50 | 755.04 | 750.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 750.50 | 754.14 | 750.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 750.00 | 754.14 | 750.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 745.65 | 752.44 | 750.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 746.00 | 752.44 | 750.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 747.05 | 751.36 | 749.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 747.05 | 751.36 | 749.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 743.50 | 747.87 | 748.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 741.60 | 746.62 | 747.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 736.35 | 736.23 | 740.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 15:00:00 | 735.20 | 736.15 | 739.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 741.90 | 737.43 | 739.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 741.90 | 737.43 | 739.50 | SL hit (close>ema400) qty=1.00 sl=739.50 alert=retest1 |

### Cycle 190 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 745.65 | 741.26 | 740.91 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 738.90 | 740.81 | 740.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 735.10 | 739.20 | 740.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 739.20 | 738.91 | 739.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 739.20 | 738.91 | 739.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 739.20 | 738.91 | 739.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 739.20 | 738.91 | 739.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 737.20 | 738.57 | 739.55 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 745.50 | 740.71 | 740.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 13:15:00 | 749.50 | 742.47 | 741.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 750.10 | 751.47 | 747.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 750.10 | 751.47 | 747.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 752.80 | 751.50 | 748.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 750.40 | 751.50 | 748.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 771.50 | 779.41 | 774.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 771.50 | 779.41 | 774.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 775.60 | 778.65 | 774.27 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 10:15:00 | 768.80 | 772.66 | 772.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 11:15:00 | 766.35 | 771.40 | 772.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 770.25 | 770.25 | 771.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 770.25 | 770.25 | 771.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 770.25 | 770.25 | 771.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 770.25 | 770.25 | 771.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 763.75 | 759.20 | 761.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 763.75 | 759.20 | 761.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 761.05 | 759.57 | 761.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:15:00 | 765.00 | 759.57 | 761.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 765.00 | 760.65 | 761.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 765.00 | 761.26 | 761.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 763.30 | 761.67 | 761.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:15:00 | 763.50 | 761.67 | 761.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 11:15:00 | 764.00 | 762.14 | 762.10 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 15:15:00 | 761.00 | 762.27 | 762.44 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 768.80 | 763.49 | 762.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 780.30 | 767.56 | 764.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 779.20 | 781.00 | 774.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:45:00 | 778.85 | 781.00 | 774.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 777.40 | 779.58 | 775.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 777.40 | 779.58 | 775.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 771.40 | 777.49 | 775.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 772.75 | 777.49 | 775.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 769.40 | 775.87 | 775.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 769.90 | 775.87 | 775.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 766.10 | 773.92 | 774.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 764.50 | 772.03 | 773.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 767.55 | 764.69 | 767.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 767.55 | 764.69 | 767.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 767.55 | 764.69 | 767.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 767.55 | 764.69 | 767.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 765.00 | 764.75 | 767.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 760.45 | 764.75 | 767.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 761.15 | 757.10 | 756.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 761.15 | 757.10 | 756.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 767.55 | 759.19 | 757.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 770.20 | 770.51 | 766.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:30:00 | 769.70 | 770.51 | 766.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 764.20 | 769.25 | 765.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 764.20 | 769.25 | 765.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 762.30 | 767.86 | 765.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 756.95 | 767.86 | 765.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 756.45 | 763.48 | 763.83 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 769.85 | 764.29 | 763.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 11:15:00 | 771.70 | 765.77 | 764.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 764.95 | 768.12 | 766.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 764.95 | 768.12 | 766.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 764.95 | 768.12 | 766.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 764.95 | 768.12 | 766.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 768.45 | 768.19 | 766.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:15:00 | 769.75 | 768.37 | 766.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:30:00 | 770.20 | 772.15 | 771.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 14:15:00 | 765.45 | 771.45 | 771.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 765.45 | 771.45 | 771.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 762.80 | 769.72 | 770.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 756.40 | 755.51 | 760.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 756.20 | 755.51 | 760.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 756.25 | 756.23 | 758.84 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 764.75 | 759.71 | 759.70 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 758.35 | 761.82 | 761.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 755.30 | 760.52 | 761.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 749.40 | 748.97 | 753.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 746.15 | 742.95 | 746.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 746.15 | 742.95 | 746.28 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 750.00 | 747.91 | 747.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 15:15:00 | 751.95 | 748.72 | 748.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 748.40 | 749.35 | 748.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 12:15:00 | 748.40 | 749.35 | 748.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 748.40 | 749.35 | 748.64 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 754.25 | 767.13 | 767.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 752.75 | 758.28 | 761.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 761.05 | 754.90 | 758.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 761.05 | 754.90 | 758.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 761.05 | 754.90 | 758.30 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 725.95 | 722.00 | 721.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 728.15 | 723.81 | 722.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 727.05 | 727.19 | 725.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 15:15:00 | 725.55 | 726.86 | 725.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 725.55 | 726.86 | 725.37 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 718.80 | 725.52 | 726.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 714.65 | 723.35 | 725.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 717.85 | 717.28 | 720.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 730.00 | 719.94 | 721.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 730.00 | 719.94 | 721.31 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 724.25 | 722.21 | 722.14 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 14:15:00 | 720.10 | 721.83 | 721.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 719.15 | 721.00 | 721.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 723.90 | 721.17 | 721.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 12:15:00 | 723.90 | 721.17 | 721.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 723.90 | 721.17 | 721.45 | EMA400 retest candle locked (from downside) |

### Cycle 210 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 726.10 | 722.15 | 721.88 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 715.10 | 720.66 | 721.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 710.65 | 716.91 | 718.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 708.75 | 707.64 | 712.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 706.80 | 707.44 | 709.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 706.80 | 707.44 | 709.96 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 706.30 | 701.60 | 701.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 708.45 | 705.25 | 703.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 705.00 | 705.38 | 703.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 13:15:00 | 733.30 | 737.99 | 735.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 733.30 | 737.99 | 735.37 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 15:15:00 | 733.25 | 736.05 | 736.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 725.85 | 734.01 | 735.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 681.90 | 680.72 | 691.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 15:15:00 | 686.20 | 681.81 | 691.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 686.20 | 681.81 | 691.31 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 642.90 | 635.28 | 634.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 645.70 | 641.08 | 638.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 631.25 | 640.02 | 638.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 631.25 | 640.02 | 638.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 631.25 | 640.02 | 638.68 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 633.85 | 637.32 | 637.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 09:15:00 | 628.85 | 634.44 | 636.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 605.20 | 599.21 | 608.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 612.30 | 603.60 | 607.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 612.30 | 603.60 | 607.91 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 612.95 | 609.92 | 609.87 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 603.25 | 609.08 | 609.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 601.15 | 606.90 | 608.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 12:15:00 | 607.45 | 607.01 | 608.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 12:15:00 | 607.45 | 607.01 | 608.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 607.45 | 607.01 | 608.33 | EMA400 retest candle locked (from downside) |

### Cycle 218 — BUY (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 13:15:00 | 580.65 | 576.72 | 576.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 582.05 | 577.79 | 576.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 591.80 | 595.45 | 589.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 591.80 | 595.45 | 589.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 591.80 | 595.45 | 589.41 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 614.00 | 624.00 | 625.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 609.75 | 617.55 | 621.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 614.20 | 611.35 | 615.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 614.20 | 611.35 | 615.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 614.20 | 611.35 | 615.19 | EMA400 retest candle locked (from downside) |

### Cycle 220 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 602.25 | 597.27 | 596.67 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 579.70 | 593.21 | 594.99 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 592.15 | 590.73 | 590.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 600.00 | 594.52 | 592.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 622.00 | 622.72 | 616.52 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-23 13:30:00 | 564.10 | 2023-05-23 14:15:00 | 562.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-05-24 09:45:00 | 565.45 | 2023-06-02 09:15:00 | 575.10 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2023-06-30 11:30:00 | 643.80 | 2023-07-04 13:15:00 | 645.75 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2023-07-07 11:00:00 | 675.90 | 2023-07-07 14:15:00 | 657.65 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2023-07-11 09:15:00 | 682.45 | 2023-07-14 10:15:00 | 670.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2023-07-11 13:45:00 | 673.05 | 2023-07-14 11:15:00 | 665.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-07-12 09:15:00 | 674.00 | 2023-07-14 11:15:00 | 665.70 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-07-13 09:15:00 | 683.15 | 2023-07-14 11:15:00 | 665.70 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2023-07-21 14:15:00 | 641.95 | 2023-07-26 11:15:00 | 657.40 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2023-07-24 09:30:00 | 650.75 | 2023-07-26 11:15:00 | 657.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-07-26 09:15:00 | 651.50 | 2023-07-26 11:15:00 | 657.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-08-09 09:15:00 | 650.10 | 2023-08-10 09:15:00 | 643.40 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-08-14 15:15:00 | 631.40 | 2023-08-22 09:15:00 | 645.50 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2023-08-23 13:15:00 | 632.30 | 2023-08-24 15:15:00 | 631.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2023-08-23 14:00:00 | 632.10 | 2023-08-24 15:15:00 | 631.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2023-08-23 14:45:00 | 633.40 | 2023-08-24 15:15:00 | 631.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-09-07 13:30:00 | 651.70 | 2023-09-13 09:15:00 | 646.35 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-09-22 13:15:00 | 649.70 | 2023-09-27 15:15:00 | 648.30 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2023-10-03 09:15:00 | 632.95 | 2023-10-06 12:15:00 | 634.40 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2023-10-03 11:45:00 | 634.50 | 2023-10-06 12:15:00 | 634.40 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2023-10-10 15:15:00 | 619.35 | 2023-10-12 12:15:00 | 624.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-10-11 10:15:00 | 620.40 | 2023-10-12 12:15:00 | 624.65 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-10-12 09:30:00 | 620.85 | 2023-10-12 12:15:00 | 624.65 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2023-10-12 10:45:00 | 620.70 | 2023-10-12 12:15:00 | 624.65 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-10-17 09:30:00 | 636.40 | 2023-10-20 10:15:00 | 635.50 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-10-19 09:45:00 | 633.35 | 2023-10-20 10:15:00 | 635.50 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2023-11-03 09:15:00 | 620.90 | 2023-11-07 09:15:00 | 617.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-11-03 09:45:00 | 621.75 | 2023-11-07 09:15:00 | 617.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-11-06 14:00:00 | 621.80 | 2023-11-07 09:15:00 | 617.90 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-11-06 15:00:00 | 622.30 | 2023-11-07 09:15:00 | 617.90 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-11-13 11:15:00 | 624.00 | 2023-11-13 12:15:00 | 619.05 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-11-21 10:15:00 | 666.00 | 2023-12-04 12:15:00 | 675.15 | STOP_HIT | 1.00 | 1.37% |
| SELL | retest2 | 2023-12-05 13:15:00 | 671.60 | 2023-12-06 11:15:00 | 680.90 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2023-12-20 12:45:00 | 667.10 | 2023-12-28 10:15:00 | 649.40 | STOP_HIT | 1.00 | 2.65% |
| BUY | retest2 | 2024-01-08 09:15:00 | 650.90 | 2024-01-08 12:15:00 | 644.25 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-02-01 11:00:00 | 569.25 | 2024-02-01 11:15:00 | 586.20 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-03-04 13:15:00 | 617.35 | 2024-03-13 12:15:00 | 612.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-03-07 09:15:00 | 618.20 | 2024-03-13 12:15:00 | 612.65 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-03-07 11:45:00 | 616.95 | 2024-03-13 12:15:00 | 612.65 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-03-07 13:00:00 | 616.85 | 2024-03-13 12:15:00 | 612.65 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-03-28 09:15:00 | 629.55 | 2024-04-03 09:15:00 | 623.05 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-04-18 13:15:00 | 613.40 | 2024-04-19 09:15:00 | 582.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 13:15:00 | 613.40 | 2024-04-19 13:15:00 | 603.80 | STOP_HIT | 0.50 | 1.57% |
| BUY | retest2 | 2024-05-27 09:15:00 | 566.70 | 2024-05-27 14:15:00 | 564.05 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-05-27 10:15:00 | 566.75 | 2024-05-27 14:15:00 | 564.05 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-05-28 09:15:00 | 567.00 | 2024-05-29 13:15:00 | 563.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-05-29 10:30:00 | 566.05 | 2024-05-29 13:15:00 | 563.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-06-04 11:00:00 | 533.95 | 2024-06-05 14:15:00 | 551.90 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-06-04 11:30:00 | 536.20 | 2024-06-05 14:15:00 | 551.90 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-06-04 14:15:00 | 535.10 | 2024-06-05 14:15:00 | 551.90 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-06-04 14:45:00 | 534.85 | 2024-06-05 14:15:00 | 551.90 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-06-13 11:15:00 | 592.30 | 2024-06-20 10:15:00 | 590.05 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-06-13 14:15:00 | 592.00 | 2024-06-20 10:15:00 | 590.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-06-14 09:15:00 | 596.30 | 2024-06-20 10:15:00 | 590.05 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-06-28 09:15:00 | 595.35 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-06-28 10:00:00 | 596.45 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-06-28 14:45:00 | 595.70 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-07-01 10:30:00 | 596.55 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-07-02 11:45:00 | 595.35 | 2024-07-02 12:15:00 | 589.55 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-07-15 12:00:00 | 639.00 | 2024-07-19 13:15:00 | 637.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-07-16 09:15:00 | 647.10 | 2024-07-19 13:15:00 | 637.25 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-07-16 10:30:00 | 639.30 | 2024-07-19 13:15:00 | 637.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-07-22 12:30:00 | 636.25 | 2024-07-23 13:15:00 | 642.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-07-22 14:00:00 | 636.25 | 2024-07-23 13:15:00 | 642.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-07-23 09:45:00 | 632.85 | 2024-07-23 13:15:00 | 642.90 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-07-23 12:15:00 | 624.20 | 2024-07-23 13:15:00 | 642.90 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-07-30 11:15:00 | 699.60 | 2024-08-05 09:15:00 | 698.40 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-07-31 09:45:00 | 698.95 | 2024-08-05 09:15:00 | 698.40 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-08-08 09:15:00 | 692.00 | 2024-08-08 11:15:00 | 703.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-08-08 10:30:00 | 691.90 | 2024-08-08 11:15:00 | 703.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-08-19 09:15:00 | 680.70 | 2024-08-20 09:15:00 | 696.95 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-08-19 11:00:00 | 681.05 | 2024-08-20 09:15:00 | 696.95 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-08-27 09:45:00 | 728.80 | 2024-09-06 10:15:00 | 743.55 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2024-08-27 10:45:00 | 729.15 | 2024-09-06 10:15:00 | 743.55 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2024-09-13 09:30:00 | 708.70 | 2024-09-19 10:15:00 | 706.05 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-09-26 10:15:00 | 726.95 | 2024-09-30 13:15:00 | 717.80 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-26 11:00:00 | 727.05 | 2024-09-30 13:15:00 | 717.80 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-09-26 11:30:00 | 726.65 | 2024-09-30 13:15:00 | 717.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-09-26 12:30:00 | 727.70 | 2024-09-30 13:15:00 | 717.80 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-10-04 13:30:00 | 705.60 | 2024-10-07 09:15:00 | 716.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-10-07 12:15:00 | 707.00 | 2024-10-09 09:15:00 | 718.70 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-10-16 09:15:00 | 727.00 | 2024-10-16 09:15:00 | 723.40 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-10-29 09:15:00 | 716.25 | 2024-10-29 10:15:00 | 723.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-11-05 15:15:00 | 715.50 | 2024-11-08 09:15:00 | 717.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-11-06 09:30:00 | 711.70 | 2024-11-08 09:15:00 | 717.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-11-11 13:45:00 | 707.35 | 2024-11-19 11:15:00 | 671.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:15:00 | 707.75 | 2024-11-19 11:15:00 | 672.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:45:00 | 706.90 | 2024-11-19 11:15:00 | 671.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:45:00 | 707.35 | 2024-11-21 14:15:00 | 675.35 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-11-12 10:15:00 | 707.75 | 2024-11-21 14:15:00 | 675.35 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2024-11-12 11:45:00 | 706.90 | 2024-11-21 14:15:00 | 675.35 | STOP_HIT | 0.50 | 4.46% |
| BUY | retest2 | 2024-11-25 14:30:00 | 689.60 | 2024-11-26 14:15:00 | 682.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-11-26 10:30:00 | 690.95 | 2024-11-26 14:15:00 | 682.90 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-11-26 12:00:00 | 689.55 | 2024-11-26 14:15:00 | 682.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-11-28 10:30:00 | 681.25 | 2024-11-28 13:15:00 | 647.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-28 10:30:00 | 681.25 | 2024-11-29 14:15:00 | 658.00 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2024-11-28 12:45:00 | 669.90 | 2024-12-03 09:15:00 | 636.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-28 12:45:00 | 669.90 | 2024-12-04 09:15:00 | 645.85 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2024-12-20 09:15:00 | 619.65 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-12-20 12:15:00 | 623.00 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-12-20 14:00:00 | 623.00 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-12-20 14:45:00 | 622.65 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-12-23 09:15:00 | 617.20 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-12-23 13:00:00 | 620.85 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-23 15:15:00 | 620.50 | 2024-12-24 11:15:00 | 626.15 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-01-10 14:15:00 | 610.40 | 2025-01-16 09:15:00 | 653.10 | STOP_HIT | 1.00 | -7.00% |
| SELL | retest2 | 2025-01-23 14:45:00 | 621.35 | 2025-01-29 11:15:00 | 622.55 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-01-24 12:30:00 | 620.85 | 2025-01-29 11:15:00 | 622.55 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-01-24 14:00:00 | 621.05 | 2025-01-29 11:15:00 | 622.55 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-02-14 09:15:00 | 632.75 | 2025-02-14 09:15:00 | 627.75 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-02-18 11:00:00 | 612.60 | 2025-02-19 15:15:00 | 623.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-02-25 15:15:00 | 620.00 | 2025-02-25 15:15:00 | 620.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-03-06 13:00:00 | 617.85 | 2025-03-13 12:15:00 | 622.80 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2025-03-06 14:15:00 | 617.60 | 2025-03-13 12:15:00 | 622.80 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-03-07 09:15:00 | 631.55 | 2025-03-13 12:15:00 | 622.80 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-04-03 12:30:00 | 696.25 | 2025-04-04 14:15:00 | 690.45 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-05 09:15:00 | 733.75 | 2025-05-06 12:15:00 | 721.15 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-05-06 11:45:00 | 728.60 | 2025-05-06 12:15:00 | 721.15 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-05-15 13:00:00 | 747.60 | 2025-05-30 13:15:00 | 776.10 | STOP_HIT | 1.00 | 3.81% |
| SELL | retest2 | 2025-06-04 13:15:00 | 761.05 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-06-04 13:45:00 | 761.60 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-06-05 13:30:00 | 760.75 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-06-05 14:45:00 | 761.10 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-06-09 09:15:00 | 751.70 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-10 09:15:00 | 751.90 | 2025-06-10 13:15:00 | 760.15 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-06-12 09:15:00 | 767.10 | 2025-06-12 11:15:00 | 759.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-06-12 10:15:00 | 766.25 | 2025-06-12 11:15:00 | 759.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-06-23 11:45:00 | 776.90 | 2025-07-02 11:15:00 | 800.30 | STOP_HIT | 1.00 | 3.01% |
| BUY | retest2 | 2025-06-23 14:15:00 | 776.95 | 2025-07-02 11:15:00 | 800.30 | STOP_HIT | 1.00 | 3.01% |
| BUY | retest2 | 2025-06-24 09:15:00 | 780.60 | 2025-07-02 11:15:00 | 800.30 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2025-07-15 11:45:00 | 762.00 | 2025-07-22 09:15:00 | 754.90 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-07-16 10:30:00 | 760.55 | 2025-07-22 09:15:00 | 754.90 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-07-24 14:00:00 | 763.95 | 2025-07-24 14:15:00 | 757.40 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-31 10:15:00 | 749.95 | 2025-08-06 13:15:00 | 750.75 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-08-01 09:45:00 | 749.20 | 2025-08-06 13:15:00 | 750.75 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-08-01 11:30:00 | 749.25 | 2025-08-06 13:15:00 | 750.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-08-11 12:45:00 | 759.65 | 2025-08-22 11:15:00 | 789.90 | STOP_HIT | 1.00 | 3.98% |
| SELL | retest2 | 2025-08-28 12:30:00 | 780.80 | 2025-09-01 13:15:00 | 782.30 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-09-01 12:30:00 | 779.95 | 2025-09-01 13:15:00 | 782.30 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-09-12 11:15:00 | 777.60 | 2025-09-16 15:15:00 | 773.75 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-15 10:30:00 | 775.90 | 2025-09-16 15:15:00 | 773.75 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-09-16 09:45:00 | 775.65 | 2025-09-16 15:15:00 | 773.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-09-22 09:15:00 | 794.00 | 2025-09-23 09:15:00 | 780.70 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-09-22 11:45:00 | 785.50 | 2025-09-23 09:15:00 | 780.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-22 12:30:00 | 785.45 | 2025-09-23 09:15:00 | 780.70 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-26 13:30:00 | 766.55 | 2025-10-01 15:15:00 | 765.50 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-29 09:45:00 | 766.20 | 2025-10-01 15:15:00 | 765.50 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest1 | 2025-10-09 09:15:00 | 743.20 | 2025-10-09 12:15:00 | 752.55 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-10-27 12:15:00 | 733.85 | 2025-10-28 10:15:00 | 745.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-27 13:30:00 | 734.90 | 2025-10-28 10:15:00 | 745.10 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest1 | 2025-11-03 15:00:00 | 735.20 | 2025-11-04 09:15:00 | 741.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-02 09:15:00 | 760.45 | 2025-12-05 10:15:00 | 761.15 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-12-11 12:15:00 | 769.75 | 2025-12-16 14:15:00 | 765.45 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-12-15 11:30:00 | 770.20 | 2025-12-16 14:15:00 | 765.45 | STOP_HIT | 1.00 | -0.62% |
