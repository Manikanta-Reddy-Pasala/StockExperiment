# Brainbees Solutions Ltd. (FIRSTCRY)

## Backtest Summary

- **Window:** 2024-08-13 09:15:00 → 2026-05-08 15:15:00 (3000 bars)
- **Last close:** 234.91
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 91 |
| ALERT2 | 86 |
| ALERT2_SKIP | 42 |
| ALERT3 | 217 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 108 |
| PARTIAL | 37 |
| TARGET_HIT | 9 |
| STOP_HIT | 103 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 149 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 83 / 66
- **Target hits / Stop hits / Partials:** 9 / 103 / 37
- **Avg / median % per leg:** 1.63% / 1.44%
- **Sum % (uncompounded):** 243.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 9 | 31.0% | 1 | 26 | 2 | -0.21% | -6.1% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.24% | 13.0% |
| BUY @ 3rd Alert (retest2) | 25 | 5 | 20.0% | 1 | 24 | 0 | -0.76% | -19.0% |
| SELL (all) | 120 | 74 | 61.7% | 8 | 77 | 35 | 2.08% | 249.3% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 1.39% | 4.2% |
| SELL @ 3rd Alert (retest2) | 117 | 73 | 62.4% | 8 | 75 | 34 | 2.10% | 245.2% |
| retest1 (combined) | 7 | 5 | 71.4% | 0 | 4 | 3 | 2.45% | 17.1% |
| retest2 (combined) | 142 | 78 | 54.9% | 9 | 99 | 34 | 1.59% | 226.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 10:15:00 | 658.95 | 668.57 | 669.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 11:15:00 | 642.90 | 663.44 | 666.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 659.45 | 637.57 | 644.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 659.45 | 637.57 | 644.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 659.45 | 637.57 | 644.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:00:00 | 659.45 | 637.57 | 644.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 643.00 | 638.66 | 644.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 11:15:00 | 641.80 | 638.66 | 644.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 12:45:00 | 638.00 | 638.95 | 643.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 12:30:00 | 641.55 | 643.41 | 644.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 14:45:00 | 639.70 | 641.72 | 643.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 634.75 | 640.05 | 642.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 631.40 | 640.05 | 642.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 09:30:00 | 628.55 | 629.21 | 634.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 09:30:00 | 630.50 | 625.16 | 627.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 09:15:00 | 638.40 | 629.49 | 628.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 638.40 | 629.49 | 628.79 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 09:15:00 | 623.20 | 630.74 | 631.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 607.90 | 622.47 | 626.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 09:15:00 | 622.55 | 615.20 | 617.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 622.55 | 615.20 | 617.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 622.55 | 615.20 | 617.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 622.55 | 615.20 | 617.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 625.25 | 617.21 | 618.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 625.25 | 617.21 | 618.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 11:15:00 | 627.30 | 619.23 | 618.88 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 595.00 | 616.51 | 618.15 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 640.95 | 613.70 | 611.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 10:15:00 | 646.20 | 635.36 | 625.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 650.55 | 653.01 | 645.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 650.55 | 653.01 | 645.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 650.55 | 653.01 | 645.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:30:00 | 650.25 | 653.01 | 645.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 641.75 | 650.19 | 646.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:30:00 | 644.05 | 650.19 | 646.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 641.15 | 648.39 | 645.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 641.15 | 648.39 | 645.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 639.25 | 643.43 | 643.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 12:15:00 | 635.40 | 640.96 | 642.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 15:15:00 | 641.95 | 640.00 | 641.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 15:15:00 | 641.95 | 640.00 | 641.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 641.95 | 640.00 | 641.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 643.65 | 640.00 | 641.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 645.10 | 641.02 | 642.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 645.10 | 641.02 | 642.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 640.45 | 640.90 | 641.90 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 650.35 | 642.71 | 642.28 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 14:15:00 | 635.70 | 641.31 | 641.84 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 650.00 | 643.05 | 642.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 661.20 | 654.46 | 651.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 655.00 | 657.61 | 653.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 655.00 | 657.61 | 653.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 655.00 | 657.61 | 653.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:30:00 | 656.00 | 657.61 | 653.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 657.00 | 657.53 | 655.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:45:00 | 657.20 | 657.53 | 655.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 650.50 | 656.68 | 655.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 650.50 | 656.68 | 655.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 655.45 | 656.43 | 655.25 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 10:15:00 | 648.70 | 654.85 | 655.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 11:15:00 | 639.00 | 651.68 | 653.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 13:15:00 | 634.95 | 630.62 | 638.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 14:00:00 | 634.95 | 630.62 | 638.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 643.30 | 633.16 | 639.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:45:00 | 646.55 | 633.16 | 639.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 642.00 | 634.93 | 639.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 634.85 | 634.93 | 639.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:00:00 | 639.75 | 635.89 | 639.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 11:15:00 | 650.00 | 639.42 | 640.39 | SL hit (close>static) qty=1.00 sl=645.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 656.30 | 642.80 | 641.84 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 631.70 | 647.34 | 648.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 12:15:00 | 629.55 | 641.21 | 645.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 641.25 | 635.44 | 640.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 641.25 | 635.44 | 640.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 641.25 | 635.44 | 640.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 641.25 | 635.44 | 640.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 649.00 | 638.15 | 641.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 646.70 | 638.15 | 641.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 649.80 | 640.48 | 642.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 649.15 | 640.48 | 642.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 648.50 | 643.45 | 643.32 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 14:15:00 | 641.85 | 643.13 | 643.19 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 644.60 | 643.24 | 643.22 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 11:15:00 | 634.95 | 641.86 | 642.62 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 649.50 | 643.58 | 643.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 11:15:00 | 676.25 | 650.82 | 646.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 15:15:00 | 680.70 | 681.38 | 670.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:15:00 | 686.40 | 681.38 | 670.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 11:15:00 | 685.85 | 680.80 | 672.13 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 14:15:00 | 720.72 | 693.38 | 681.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 14:15:00 | 720.14 | 693.38 | 681.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 696.30 | 712.10 | 701.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 696.30 | 712.10 | 701.59 | SL hit (close<ema200) qty=0.50 sl=712.10 alert=retest1 |

### Cycle 19 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 689.05 | 698.48 | 699.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 685.55 | 695.89 | 697.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 691.90 | 690.84 | 694.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 14:15:00 | 691.90 | 690.84 | 694.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 691.90 | 690.84 | 694.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 691.90 | 690.84 | 694.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 690.00 | 690.67 | 694.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 689.65 | 690.67 | 694.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 698.05 | 692.15 | 694.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 698.05 | 692.15 | 694.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 692.75 | 692.27 | 694.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 12:45:00 | 689.45 | 691.34 | 693.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 13:45:00 | 688.15 | 690.85 | 693.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:15:00 | 688.60 | 688.70 | 691.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 654.98 | 672.14 | 681.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 654.17 | 672.14 | 681.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 653.74 | 664.34 | 676.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 644.90 | 644.49 | 659.36 | SL hit (close>ema200) qty=0.50 sl=644.49 alert=retest2 |

### Cycle 20 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 692.45 | 635.87 | 629.89 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 627.40 | 641.73 | 643.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 617.85 | 633.37 | 638.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 618.80 | 618.44 | 623.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 15:00:00 | 618.80 | 618.44 | 623.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 622.00 | 618.94 | 623.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:45:00 | 621.30 | 618.94 | 623.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 623.00 | 619.75 | 623.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 623.00 | 619.75 | 623.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 625.65 | 620.93 | 623.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:45:00 | 624.20 | 620.93 | 623.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 627.80 | 622.30 | 623.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 14:15:00 | 623.85 | 623.23 | 623.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 15:15:00 | 623.00 | 623.64 | 624.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 14:15:00 | 592.66 | 607.04 | 614.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 14:15:00 | 591.85 | 607.04 | 614.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-08 09:15:00 | 561.47 | 602.14 | 611.01 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 22 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 554.35 | 534.54 | 533.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 13:15:00 | 579.55 | 546.93 | 539.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 15:15:00 | 539.00 | 545.73 | 540.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 15:15:00 | 539.00 | 545.73 | 540.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 539.00 | 545.73 | 540.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 553.00 | 545.73 | 540.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:45:00 | 550.35 | 545.59 | 540.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 13:00:00 | 559.50 | 546.77 | 542.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 10:00:00 | 551.80 | 550.01 | 545.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 548.75 | 550.81 | 546.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:30:00 | 549.20 | 550.81 | 546.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 559.85 | 557.97 | 552.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-25 11:15:00 | 547.70 | 551.25 | 551.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 11:15:00 | 547.70 | 551.25 | 551.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 13:15:00 | 546.40 | 549.87 | 550.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 559.55 | 550.18 | 550.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 559.55 | 550.18 | 550.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 559.55 | 550.18 | 550.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 559.55 | 550.18 | 550.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 10:15:00 | 562.85 | 552.71 | 551.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 11:15:00 | 573.60 | 556.89 | 553.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 565.45 | 568.24 | 561.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:00:00 | 565.45 | 568.24 | 561.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 569.80 | 568.55 | 562.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 570.80 | 568.55 | 562.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:00:00 | 571.05 | 569.05 | 563.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 11:15:00 | 590.05 | 593.97 | 594.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 11:15:00 | 590.05 | 593.97 | 594.17 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 603.45 | 595.55 | 594.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 14:15:00 | 606.70 | 597.78 | 595.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 13:15:00 | 605.05 | 605.59 | 601.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 13:45:00 | 604.05 | 605.59 | 601.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 590.70 | 602.42 | 601.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 590.70 | 602.42 | 601.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 590.00 | 599.94 | 600.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 13:15:00 | 588.85 | 594.66 | 597.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 15:15:00 | 592.00 | 589.60 | 591.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 15:15:00 | 592.00 | 589.60 | 591.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 592.00 | 589.60 | 591.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 597.00 | 589.60 | 591.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 595.50 | 590.78 | 591.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:15:00 | 579.20 | 590.30 | 591.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:45:00 | 580.00 | 588.59 | 590.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 579.80 | 587.04 | 589.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 10:15:00 | 600.70 | 591.31 | 590.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 600.70 | 591.31 | 590.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 11:15:00 | 605.30 | 594.10 | 592.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 15:15:00 | 603.35 | 605.24 | 600.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 09:15:00 | 596.40 | 605.24 | 600.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 591.25 | 602.44 | 600.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 592.05 | 602.44 | 600.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 597.75 | 601.50 | 599.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 598.20 | 601.50 | 599.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 12:15:00 | 592.05 | 598.60 | 598.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 12:15:00 | 592.05 | 598.60 | 598.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 14:15:00 | 590.20 | 595.77 | 597.37 | Break + close below crossover candle low |

### Cycle 30 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 613.50 | 598.61 | 598.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 615.95 | 606.22 | 602.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 610.60 | 611.63 | 607.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 13:00:00 | 610.60 | 611.63 | 607.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 608.85 | 611.08 | 607.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:45:00 | 608.70 | 611.08 | 607.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 608.30 | 610.52 | 607.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:15:00 | 609.00 | 610.52 | 607.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 609.00 | 610.22 | 607.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 625.10 | 610.22 | 607.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:30:00 | 609.70 | 620.82 | 616.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 12:00:00 | 611.00 | 616.72 | 615.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 604.55 | 612.69 | 613.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 13:15:00 | 604.55 | 612.69 | 613.64 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 09:15:00 | 628.35 | 613.76 | 613.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 11:15:00 | 632.00 | 619.19 | 616.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 15:15:00 | 624.30 | 626.34 | 620.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 09:15:00 | 619.15 | 626.34 | 620.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 613.25 | 623.72 | 620.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 612.05 | 623.72 | 620.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 612.55 | 621.49 | 619.56 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2024-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 12:15:00 | 616.95 | 618.28 | 618.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 11:15:00 | 606.80 | 612.24 | 615.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 12:15:00 | 619.65 | 613.72 | 615.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 12:15:00 | 619.65 | 613.72 | 615.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 619.65 | 613.72 | 615.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:00:00 | 619.65 | 613.72 | 615.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 621.70 | 615.32 | 616.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:30:00 | 625.80 | 615.32 | 616.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 635.00 | 619.25 | 617.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 637.60 | 628.87 | 623.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 09:15:00 | 628.20 | 629.88 | 625.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 10:00:00 | 628.20 | 629.88 | 625.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 627.40 | 629.15 | 625.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 12:30:00 | 636.90 | 631.92 | 627.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:15:00 | 631.25 | 631.28 | 628.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 623.20 | 628.46 | 628.14 | SL hit (close<static) qty=1.00 sl=625.05 alert=retest2 |

### Cycle 35 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 618.15 | 626.39 | 627.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 610.90 | 622.21 | 625.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 639.70 | 621.96 | 623.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 639.70 | 621.96 | 623.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 639.70 | 621.96 | 623.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 639.70 | 621.96 | 623.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 652.90 | 628.15 | 626.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 657.05 | 637.90 | 631.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 15:15:00 | 646.00 | 646.78 | 639.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:15:00 | 639.50 | 646.78 | 639.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 642.65 | 645.95 | 639.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:00:00 | 652.05 | 645.87 | 641.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 10:00:00 | 651.20 | 648.40 | 643.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 10:45:00 | 650.25 | 648.97 | 644.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 13:00:00 | 650.80 | 649.53 | 645.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 647.70 | 649.00 | 646.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 631.30 | 649.00 | 646.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 633.00 | 645.80 | 645.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 633.00 | 645.80 | 645.08 | SL hit (close<static) qty=1.00 sl=638.25 alert=retest2 |

### Cycle 37 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 623.90 | 641.42 | 643.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 622.40 | 637.62 | 641.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 499.80 | 499.78 | 513.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:30:00 | 499.00 | 499.78 | 513.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 496.60 | 493.28 | 498.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 487.80 | 493.28 | 498.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:00:00 | 494.75 | 493.57 | 498.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 12:15:00 | 463.41 | 477.72 | 484.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 12:15:00 | 470.01 | 477.72 | 484.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-22 10:15:00 | 445.28 | 461.69 | 473.31 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 38 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 475.55 | 468.33 | 468.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 14:15:00 | 480.95 | 470.85 | 469.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 12:15:00 | 467.20 | 473.36 | 471.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 12:15:00 | 467.20 | 473.36 | 471.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 467.20 | 473.36 | 471.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:45:00 | 467.40 | 473.36 | 471.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 466.80 | 472.05 | 471.13 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 461.00 | 468.70 | 469.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 458.50 | 464.12 | 467.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 459.85 | 452.35 | 456.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 459.85 | 452.35 | 456.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 459.85 | 452.35 | 456.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 460.60 | 452.35 | 456.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 459.55 | 453.79 | 457.23 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 463.00 | 458.86 | 458.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 468.60 | 461.61 | 460.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 486.30 | 491.22 | 483.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 486.30 | 491.22 | 483.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 486.30 | 491.22 | 483.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:15:00 | 475.55 | 491.22 | 483.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 475.60 | 488.10 | 482.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:30:00 | 477.95 | 488.10 | 482.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 11:15:00 | 474.40 | 485.36 | 481.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:45:00 | 475.05 | 485.36 | 481.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 467.10 | 477.91 | 478.89 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 10:15:00 | 479.60 | 477.50 | 477.50 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 470.60 | 476.32 | 476.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 465.50 | 474.15 | 475.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 09:15:00 | 439.50 | 436.73 | 450.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 439.50 | 436.73 | 450.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 439.50 | 436.73 | 450.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:00:00 | 432.80 | 435.95 | 449.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:45:00 | 430.40 | 434.36 | 447.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 411.16 | 422.40 | 435.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 408.88 | 422.40 | 435.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-11 14:15:00 | 418.50 | 417.65 | 427.97 | SL hit (close>ema200) qty=0.50 sl=417.65 alert=retest2 |

### Cycle 44 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 405.00 | 392.31 | 391.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 416.15 | 402.06 | 396.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 407.00 | 407.26 | 400.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 407.00 | 407.26 | 400.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 403.60 | 406.38 | 402.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 402.80 | 406.38 | 402.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 405.55 | 406.21 | 402.81 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 394.65 | 400.21 | 400.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 393.70 | 397.94 | 399.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 09:15:00 | 384.20 | 381.62 | 386.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 384.20 | 381.62 | 386.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 384.20 | 381.62 | 386.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 373.70 | 379.58 | 383.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 14:30:00 | 374.50 | 375.51 | 378.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 372.80 | 376.20 | 378.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 411.20 | 373.72 | 371.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 411.20 | 373.72 | 371.83 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 379.85 | 389.85 | 390.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 374.65 | 386.81 | 388.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 378.05 | 377.81 | 382.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 11:00:00 | 378.05 | 377.81 | 382.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 375.40 | 375.62 | 379.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 379.35 | 375.62 | 379.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 380.40 | 376.37 | 378.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 380.40 | 376.37 | 378.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 380.95 | 377.29 | 379.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:00:00 | 380.95 | 377.29 | 379.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 380.15 | 377.86 | 379.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 375.90 | 378.99 | 379.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:15:00 | 379.00 | 379.36 | 379.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 13:15:00 | 376.05 | 374.80 | 374.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 13:15:00 | 376.05 | 374.80 | 374.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 15:15:00 | 377.90 | 375.77 | 375.21 | Break + close above crossover candle high |

### Cycle 49 — SELL (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 09:15:00 | 370.30 | 374.68 | 374.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-20 11:15:00 | 369.40 | 372.88 | 373.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 371.65 | 368.93 | 371.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 371.65 | 368.93 | 371.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 371.65 | 368.93 | 371.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:30:00 | 375.10 | 368.93 | 371.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 365.95 | 368.33 | 370.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:30:00 | 367.65 | 368.33 | 370.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 370.20 | 368.85 | 370.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:30:00 | 370.50 | 368.85 | 370.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 370.35 | 369.15 | 370.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:00:00 | 370.35 | 369.15 | 370.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 364.70 | 368.26 | 369.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:30:00 | 369.75 | 368.26 | 369.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 369.10 | 367.92 | 369.49 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 375.65 | 370.34 | 370.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 15:15:00 | 379.05 | 372.08 | 370.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 12:15:00 | 375.30 | 376.43 | 373.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 12:30:00 | 375.40 | 376.43 | 373.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 372.25 | 375.34 | 373.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:00:00 | 372.25 | 375.34 | 373.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 369.95 | 374.26 | 373.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 369.00 | 374.26 | 373.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 365.65 | 372.54 | 372.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 363.35 | 370.70 | 371.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 358.25 | 357.04 | 361.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 14:00:00 | 358.25 | 357.04 | 361.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 394.10 | 364.45 | 364.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 394.10 | 364.45 | 364.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 380.30 | 367.62 | 366.07 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 366.45 | 368.78 | 368.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 364.70 | 367.70 | 368.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 368.90 | 367.78 | 368.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 11:15:00 | 368.90 | 367.78 | 368.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 368.90 | 367.78 | 368.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 368.90 | 367.78 | 368.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 370.75 | 368.37 | 368.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 370.75 | 368.37 | 368.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 368.50 | 368.40 | 368.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 368.50 | 368.40 | 368.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 369.25 | 368.57 | 368.57 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 367.05 | 368.27 | 368.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 355.25 | 365.71 | 367.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 332.15 | 330.66 | 341.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 10:30:00 | 330.60 | 330.66 | 341.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 323.55 | 321.41 | 327.35 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 341.80 | 330.58 | 329.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 13:15:00 | 344.15 | 338.93 | 336.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 359.80 | 362.86 | 356.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 359.80 | 362.86 | 356.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 358.90 | 362.76 | 359.74 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 353.40 | 358.39 | 358.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 338.70 | 352.86 | 355.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 337.10 | 333.69 | 337.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 337.10 | 333.69 | 337.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 337.10 | 333.69 | 337.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:00:00 | 337.10 | 333.69 | 337.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 336.85 | 334.32 | 337.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:15:00 | 339.70 | 334.32 | 337.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 343.65 | 336.19 | 337.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:00:00 | 343.65 | 336.19 | 337.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 345.45 | 338.04 | 338.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 345.45 | 338.04 | 338.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 339.75 | 338.77 | 338.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:45:00 | 341.50 | 338.77 | 338.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 336.15 | 338.03 | 338.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 331.15 | 336.45 | 337.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 331.50 | 335.22 | 336.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:00:00 | 331.20 | 334.41 | 336.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:45:00 | 330.25 | 331.07 | 332.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 330.95 | 331.05 | 332.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:30:00 | 328.90 | 330.26 | 331.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 314.59 | 324.43 | 328.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 314.93 | 324.43 | 328.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 314.64 | 324.43 | 328.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 313.74 | 324.43 | 328.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 10:15:00 | 326.35 | 324.82 | 328.28 | SL hit (close>ema200) qty=0.50 sl=324.82 alert=retest2 |

### Cycle 58 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 346.80 | 325.89 | 325.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 351.15 | 335.86 | 330.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 345.10 | 345.11 | 338.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 345.10 | 345.11 | 338.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 340.25 | 343.60 | 340.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 340.25 | 343.60 | 340.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 340.20 | 342.92 | 340.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 353.35 | 341.28 | 340.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:30:00 | 346.55 | 342.67 | 341.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 347.25 | 351.05 | 351.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 347.25 | 351.05 | 351.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 345.15 | 349.87 | 350.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 349.55 | 348.77 | 349.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 349.55 | 348.77 | 349.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 354.25 | 349.65 | 350.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 354.25 | 349.65 | 350.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 356.15 | 350.95 | 350.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 13:15:00 | 360.20 | 352.80 | 351.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 358.85 | 370.00 | 366.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 358.85 | 370.00 | 366.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 358.85 | 370.00 | 366.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 358.85 | 370.00 | 366.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 355.30 | 367.06 | 365.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 355.30 | 367.06 | 365.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 354.00 | 364.45 | 364.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 352.55 | 362.07 | 363.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 11:15:00 | 358.30 | 357.42 | 359.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 12:00:00 | 358.30 | 357.42 | 359.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 357.85 | 357.51 | 359.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 359.00 | 357.51 | 359.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 342.30 | 348.29 | 351.51 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 375.00 | 354.39 | 351.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 12:15:00 | 388.10 | 361.13 | 354.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 362.10 | 366.82 | 359.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 15:15:00 | 362.10 | 366.82 | 359.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 362.10 | 366.82 | 359.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:45:00 | 386.25 | 371.51 | 365.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 10:15:00 | 424.88 | 401.83 | 387.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 396.45 | 401.10 | 401.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 390.20 | 398.10 | 399.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 391.60 | 389.62 | 393.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 391.60 | 389.62 | 393.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 391.60 | 389.62 | 393.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:45:00 | 383.50 | 386.66 | 390.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 376.50 | 378.93 | 383.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 380.05 | 376.92 | 378.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 09:15:00 | 364.32 | 371.86 | 375.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 357.68 | 369.14 | 373.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 361.05 | 369.14 | 373.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-06-19 15:15:00 | 345.15 | 357.30 | 365.64 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 64 — BUY (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 10:15:00 | 362.80 | 343.94 | 342.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 374.00 | 349.96 | 345.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 383.20 | 385.61 | 374.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 15:15:00 | 383.20 | 385.61 | 374.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 383.20 | 385.61 | 374.10 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 374.00 | 378.15 | 378.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 368.35 | 373.48 | 375.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 371.60 | 367.21 | 369.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 13:15:00 | 371.60 | 367.21 | 369.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 371.60 | 367.21 | 369.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 371.60 | 367.21 | 369.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 371.95 | 368.16 | 369.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 370.15 | 368.16 | 369.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 376.35 | 370.12 | 370.20 | SL hit (close>static) qty=1.00 sl=374.45 alert=retest2 |

### Cycle 66 — BUY (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 10:15:00 | 373.25 | 370.74 | 370.48 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 368.40 | 370.27 | 370.29 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 372.55 | 370.73 | 370.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 373.20 | 371.22 | 370.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 379.10 | 379.66 | 376.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 379.10 | 379.66 | 376.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 377.00 | 378.73 | 377.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 378.75 | 378.73 | 377.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 374.50 | 377.89 | 376.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 374.50 | 377.89 | 376.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 372.90 | 376.89 | 376.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 372.00 | 376.89 | 376.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 372.45 | 376.00 | 376.23 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 14:15:00 | 380.20 | 377.08 | 376.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 382.30 | 378.59 | 377.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 383.10 | 383.64 | 381.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 383.10 | 383.64 | 381.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 379.50 | 382.81 | 381.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 379.85 | 382.81 | 381.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 383.40 | 382.93 | 381.29 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 380.40 | 380.72 | 380.73 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 382.35 | 381.04 | 380.88 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 377.05 | 380.49 | 380.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 376.55 | 378.64 | 379.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 356.80 | 356.41 | 359.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 13:00:00 | 356.80 | 356.41 | 359.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 357.05 | 356.79 | 358.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 351.55 | 355.33 | 357.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:15:00 | 351.55 | 354.39 | 356.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:45:00 | 351.50 | 352.92 | 355.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 14:45:00 | 352.10 | 352.46 | 354.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 349.00 | 351.77 | 353.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 347.60 | 351.77 | 353.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 348.30 | 350.39 | 352.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 357.20 | 347.68 | 348.66 | SL hit (close>static) qty=1.00 sl=356.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 09:15:00 | 355.20 | 350.03 | 349.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 364.10 | 354.76 | 351.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 365.40 | 367.05 | 361.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 14:00:00 | 365.40 | 367.05 | 361.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 361.30 | 365.90 | 361.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:45:00 | 357.00 | 365.90 | 361.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 361.10 | 364.94 | 361.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 364.35 | 364.94 | 361.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 358.60 | 363.67 | 360.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 358.60 | 363.67 | 360.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 358.95 | 362.73 | 360.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:15:00 | 358.30 | 362.73 | 360.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 359.40 | 362.06 | 360.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 360.30 | 362.06 | 360.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 14:15:00 | 356.50 | 359.89 | 359.87 | SL hit (close<static) qty=1.00 sl=357.60 alert=retest2 |

### Cycle 75 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 356.45 | 359.20 | 359.56 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 367.40 | 360.60 | 359.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 368.75 | 366.14 | 364.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 368.90 | 372.53 | 369.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 368.90 | 372.53 | 369.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 368.90 | 372.53 | 369.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 368.90 | 372.53 | 369.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 375.30 | 373.08 | 370.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 368.30 | 373.08 | 370.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 370.25 | 373.79 | 372.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 368.30 | 373.79 | 372.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 370.40 | 373.11 | 371.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 368.85 | 373.11 | 371.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 373.15 | 374.27 | 372.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 373.15 | 374.27 | 372.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 370.00 | 373.42 | 372.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 367.60 | 372.41 | 372.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 367.75 | 371.48 | 371.89 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 373.70 | 371.83 | 371.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 381.85 | 374.00 | 372.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 10:15:00 | 383.25 | 386.83 | 383.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 10:15:00 | 383.25 | 386.83 | 383.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 383.25 | 386.83 | 383.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 383.25 | 386.83 | 383.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 384.80 | 386.42 | 384.02 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 376.75 | 382.23 | 382.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 376.10 | 379.58 | 380.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 361.85 | 360.68 | 367.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:45:00 | 362.85 | 360.68 | 367.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 363.00 | 360.89 | 364.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 359.30 | 360.89 | 364.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 13:30:00 | 359.55 | 360.61 | 363.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 10:30:00 | 359.05 | 359.70 | 361.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:15:00 | 359.50 | 359.70 | 361.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 360.00 | 359.71 | 361.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:30:00 | 362.60 | 359.71 | 361.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 360.15 | 359.80 | 361.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 360.15 | 359.80 | 361.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 359.70 | 359.78 | 361.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 359.70 | 359.78 | 361.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 360.25 | 359.87 | 360.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 375.55 | 364.13 | 362.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 375.55 | 364.13 | 362.61 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 352.35 | 361.05 | 361.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 351.70 | 359.18 | 360.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 373.75 | 361.19 | 361.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 373.75 | 361.19 | 361.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 373.75 | 361.19 | 361.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 378.60 | 361.19 | 361.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 390.00 | 366.96 | 363.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 12:15:00 | 394.70 | 376.51 | 369.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 407.70 | 407.73 | 394.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 403.00 | 407.73 | 394.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 395.65 | 406.18 | 401.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 395.65 | 406.18 | 401.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 393.40 | 403.63 | 400.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 394.60 | 403.63 | 400.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 386.70 | 396.37 | 397.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 383.35 | 389.88 | 393.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 395.00 | 390.14 | 392.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 15:15:00 | 395.00 | 390.14 | 392.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 395.00 | 390.14 | 392.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 398.50 | 390.14 | 392.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 395.05 | 391.12 | 393.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 392.90 | 391.34 | 392.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 392.55 | 391.63 | 392.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 390.55 | 391.83 | 392.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 396.85 | 393.12 | 393.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 396.85 | 393.12 | 393.03 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 390.60 | 393.53 | 393.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 12:15:00 | 388.50 | 391.61 | 392.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 390.45 | 390.00 | 391.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 10:15:00 | 390.45 | 390.00 | 391.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 390.45 | 390.00 | 391.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:00:00 | 389.15 | 389.83 | 391.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 389.10 | 389.68 | 390.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 387.45 | 389.73 | 390.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 388.80 | 389.73 | 390.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 390.25 | 389.83 | 390.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 390.25 | 389.83 | 390.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 390.60 | 389.98 | 390.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:30:00 | 390.60 | 389.98 | 390.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 390.70 | 390.13 | 390.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 390.55 | 390.13 | 390.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 390.00 | 390.10 | 390.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 389.30 | 390.01 | 390.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:00:00 | 388.45 | 389.29 | 389.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 13:30:00 | 389.20 | 384.29 | 385.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 386.45 | 385.35 | 386.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 380.80 | 384.62 | 385.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 380.30 | 384.62 | 385.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:30:00 | 379.35 | 382.60 | 383.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 380.50 | 382.18 | 383.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.69 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.64 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 368.08 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.36 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.83 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.03 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 369.74 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 367.13 | 375.70 | 379.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 361.28 | 368.83 | 374.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 15:15:00 | 361.47 | 368.83 | 374.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 09:15:00 | 360.38 | 368.24 | 373.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 366.85 | 364.61 | 368.73 | SL hit (close>ema200) qty=0.50 sl=364.61 alert=retest2 |

### Cycle 86 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 372.35 | 368.28 | 367.73 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 13:15:00 | 368.00 | 368.94 | 368.97 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 370.05 | 369.16 | 369.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 377.40 | 370.94 | 369.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 376.95 | 376.95 | 373.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 376.80 | 376.95 | 373.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 375.75 | 376.71 | 374.15 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 371.80 | 373.02 | 373.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 369.90 | 372.40 | 372.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 372.10 | 371.93 | 372.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 12:15:00 | 372.10 | 371.93 | 372.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 372.10 | 371.93 | 372.47 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 374.60 | 373.06 | 372.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 376.50 | 373.96 | 373.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 374.30 | 374.37 | 373.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 13:30:00 | 375.30 | 374.37 | 373.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 373.25 | 374.19 | 373.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 368.80 | 374.19 | 373.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 368.35 | 373.02 | 373.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 365.50 | 371.52 | 372.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 362.70 | 362.58 | 365.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:45:00 | 363.15 | 362.58 | 365.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 365.50 | 363.27 | 365.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 365.50 | 363.27 | 365.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 364.55 | 363.53 | 365.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 364.05 | 363.53 | 365.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 364.00 | 363.76 | 365.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 368.45 | 363.76 | 365.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 364.95 | 364.00 | 365.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 364.35 | 364.00 | 365.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:45:00 | 363.25 | 363.92 | 364.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 362.50 | 363.85 | 364.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:45:00 | 364.25 | 363.92 | 364.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 364.00 | 363.93 | 364.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 362.05 | 363.54 | 363.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 346.13 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 345.09 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 344.38 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 346.04 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:15:00 | 343.95 | 351.78 | 355.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 351.85 | 351.24 | 354.26 | SL hit (close>ema200) qty=0.50 sl=351.24 alert=retest2 |

### Cycle 92 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 351.00 | 350.21 | 350.18 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 346.80 | 349.81 | 350.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 345.25 | 348.77 | 349.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 338.40 | 338.28 | 341.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:45:00 | 338.15 | 338.28 | 341.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 337.85 | 338.28 | 340.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 340.45 | 338.28 | 340.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 338.30 | 336.06 | 337.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 340.20 | 336.06 | 337.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 341.65 | 337.18 | 338.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 341.65 | 337.18 | 338.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 340.60 | 338.96 | 338.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 13:15:00 | 344.50 | 342.49 | 340.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 15:15:00 | 342.50 | 342.61 | 341.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:15:00 | 338.70 | 342.61 | 341.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 341.35 | 342.36 | 341.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:30:00 | 340.85 | 342.36 | 341.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 331.00 | 340.09 | 340.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 323.45 | 332.08 | 334.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 322.65 | 320.25 | 323.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:00:00 | 322.65 | 320.25 | 323.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 317.20 | 317.68 | 319.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 315.35 | 317.61 | 319.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 314.85 | 312.55 | 312.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 315.45 | 313.13 | 313.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 315.45 | 313.13 | 313.04 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 310.30 | 313.08 | 313.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 309.00 | 312.26 | 312.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 300.10 | 299.11 | 301.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:00:00 | 300.10 | 299.11 | 301.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 300.55 | 299.40 | 301.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 300.90 | 299.40 | 301.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 300.50 | 299.62 | 301.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 300.50 | 299.62 | 301.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 300.05 | 299.70 | 300.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 301.05 | 299.70 | 300.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 300.20 | 299.80 | 300.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:45:00 | 297.60 | 299.64 | 300.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 10:15:00 | 282.72 | 289.68 | 293.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 290.25 | 284.28 | 288.51 | SL hit (close>ema200) qty=0.50 sl=284.28 alert=retest2 |

### Cycle 98 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 288.00 | 286.83 | 286.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 290.50 | 288.02 | 287.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 288.30 | 288.94 | 288.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 288.30 | 288.94 | 288.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 288.30 | 288.94 | 288.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 288.30 | 288.94 | 288.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 287.35 | 288.62 | 288.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 286.60 | 288.62 | 288.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 287.75 | 288.45 | 288.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 286.90 | 288.45 | 288.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 285.60 | 287.73 | 287.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 15:15:00 | 285.25 | 287.23 | 287.53 | Break + close below crossover candle low |

### Cycle 100 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 291.05 | 287.99 | 287.85 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 287.30 | 288.39 | 288.40 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 290.30 | 288.77 | 288.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 09:15:00 | 293.00 | 289.88 | 289.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 12:15:00 | 291.60 | 291.76 | 290.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 291.60 | 291.76 | 290.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 297.15 | 297.34 | 294.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 297.15 | 297.34 | 294.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 294.60 | 296.58 | 294.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 294.15 | 296.58 | 294.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 291.65 | 295.59 | 294.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 291.65 | 295.59 | 294.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 290.20 | 294.51 | 294.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:00:00 | 290.20 | 294.51 | 294.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 12:15:00 | 291.00 | 293.81 | 293.89 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 299.05 | 294.65 | 294.19 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 285.60 | 295.21 | 296.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 285.00 | 293.17 | 295.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 288.15 | 287.89 | 291.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:30:00 | 288.10 | 287.89 | 291.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 286.35 | 284.54 | 286.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 286.35 | 284.54 | 286.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 287.35 | 285.11 | 286.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 292.20 | 285.11 | 286.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 287.45 | 285.57 | 286.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 289.10 | 285.57 | 286.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 286.75 | 285.81 | 286.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 287.00 | 285.81 | 286.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 287.80 | 286.21 | 286.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 287.85 | 286.21 | 286.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 287.85 | 286.54 | 286.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 287.95 | 286.54 | 286.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 288.00 | 287.01 | 286.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 288.45 | 287.40 | 287.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 293.80 | 296.12 | 293.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 10:00:00 | 293.80 | 296.12 | 293.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 289.65 | 294.83 | 292.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 289.65 | 294.83 | 292.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 290.10 | 293.88 | 292.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:45:00 | 289.10 | 293.88 | 292.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 290.90 | 293.29 | 292.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 289.50 | 293.29 | 292.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 292.95 | 292.86 | 292.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:15:00 | 289.65 | 292.86 | 292.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 289.65 | 292.22 | 292.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 286.00 | 292.22 | 292.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 285.50 | 290.88 | 291.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 283.90 | 286.64 | 288.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 273.20 | 273.12 | 276.93 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 12:30:00 | 271.20 | 272.81 | 276.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 13:45:00 | 270.80 | 272.30 | 275.59 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 257.64 | 268.24 | 272.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 272.00 | 268.99 | 272.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 272.00 | 268.99 | 272.41 | SL hit (close>ema200) qty=0.50 sl=268.99 alert=retest1 |

### Cycle 108 — BUY (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 15:15:00 | 273.20 | 271.72 | 271.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 12:15:00 | 274.40 | 273.14 | 272.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 269.85 | 273.28 | 272.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 269.85 | 273.28 | 272.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 269.85 | 273.28 | 272.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 270.15 | 273.28 | 272.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 270.15 | 272.65 | 272.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 270.15 | 272.65 | 272.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 270.00 | 272.12 | 272.36 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 274.10 | 272.27 | 272.16 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 271.05 | 272.35 | 272.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 260.35 | 269.70 | 271.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 266.45 | 263.45 | 266.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 266.45 | 263.45 | 266.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 266.45 | 263.45 | 266.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 266.45 | 263.45 | 266.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 267.15 | 264.19 | 266.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 267.15 | 264.19 | 266.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 267.80 | 264.91 | 266.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 267.80 | 264.91 | 266.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 269.05 | 265.74 | 266.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 269.05 | 265.74 | 266.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 272.05 | 267.81 | 267.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 274.05 | 269.74 | 268.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 272.55 | 272.96 | 271.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 272.55 | 272.96 | 271.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 272.55 | 272.96 | 271.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 272.20 | 272.96 | 271.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 271.30 | 273.17 | 271.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 271.30 | 273.17 | 271.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 266.90 | 271.92 | 271.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 268.95 | 271.92 | 271.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 266.40 | 270.81 | 270.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 266.40 | 270.81 | 270.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 264.05 | 269.46 | 270.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 264.00 | 268.37 | 269.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 269.30 | 268.55 | 269.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 269.30 | 268.55 | 269.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 269.30 | 268.55 | 269.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 269.20 | 268.55 | 269.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 269.00 | 268.64 | 269.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 265.35 | 269.44 | 269.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 271.00 | 269.75 | 269.75 | SL hit (close>static) qty=1.00 sl=270.00 alert=retest2 |

### Cycle 114 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 270.95 | 269.73 | 269.64 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 268.95 | 269.50 | 269.54 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 270.65 | 269.73 | 269.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 272.05 | 270.70 | 270.15 | Break + close above crossover candle high |

### Cycle 117 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 265.50 | 269.74 | 269.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 261.65 | 265.91 | 267.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 269.80 | 265.90 | 267.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 12:15:00 | 269.80 | 265.90 | 267.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 269.80 | 265.90 | 267.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 269.80 | 265.90 | 267.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 268.85 | 266.49 | 267.27 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 270.65 | 268.23 | 267.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 275.50 | 269.68 | 268.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 280.35 | 280.73 | 276.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 280.35 | 280.73 | 276.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 276.80 | 279.47 | 276.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 276.80 | 279.47 | 276.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 275.55 | 278.69 | 276.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 275.55 | 278.69 | 276.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 274.70 | 277.89 | 276.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 274.40 | 277.89 | 276.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 272.25 | 275.87 | 275.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 261.30 | 269.23 | 272.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 267.00 | 266.96 | 269.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 14:00:00 | 267.00 | 266.96 | 269.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 270.05 | 267.93 | 269.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 247.35 | 267.93 | 269.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:15:00 | 234.98 | 243.99 | 254.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-18 09:15:00 | 222.62 | 231.64 | 241.99 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 120 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 217.15 | 215.85 | 215.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 220.50 | 216.78 | 216.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 12:15:00 | 217.40 | 217.76 | 217.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 217.40 | 217.76 | 217.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 217.40 | 217.76 | 217.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 217.25 | 217.76 | 217.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 218.80 | 217.96 | 217.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 215.60 | 217.96 | 217.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 217.80 | 217.93 | 217.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 215.05 | 217.93 | 217.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 216.75 | 217.69 | 217.25 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 213.10 | 216.54 | 216.79 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 12:15:00 | 217.76 | 216.54 | 216.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 13:15:00 | 217.77 | 216.78 | 216.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 11:15:00 | 217.21 | 217.23 | 216.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 11:15:00 | 217.21 | 217.23 | 216.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 217.21 | 217.23 | 216.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 217.21 | 217.23 | 216.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 216.18 | 217.02 | 216.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 216.18 | 217.02 | 216.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 216.45 | 216.91 | 216.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:15:00 | 216.78 | 216.91 | 216.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 215.25 | 216.52 | 216.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 15:15:00 | 215.25 | 216.52 | 216.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 210.07 | 214.81 | 215.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 214.03 | 211.92 | 213.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 214.03 | 211.92 | 213.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 214.03 | 211.92 | 213.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 218.27 | 211.92 | 213.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 218.23 | 213.18 | 213.77 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 218.46 | 215.01 | 214.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 224.78 | 218.18 | 216.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 221.25 | 222.58 | 220.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 221.25 | 222.58 | 220.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 221.25 | 222.58 | 220.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 219.71 | 222.58 | 220.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 228.42 | 224.45 | 222.28 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 213.53 | 220.85 | 221.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 15:15:00 | 212.90 | 216.15 | 218.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 213.10 | 213.03 | 215.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 14:45:00 | 214.40 | 213.03 | 215.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 214.40 | 213.76 | 214.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:45:00 | 214.58 | 213.76 | 214.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 209.21 | 210.57 | 212.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:15:00 | 214.38 | 210.57 | 212.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 235.50 | 215.56 | 214.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 249.86 | 222.42 | 217.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 232.70 | 239.77 | 230.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 232.70 | 239.77 | 230.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 229.58 | 237.73 | 230.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 229.58 | 237.73 | 230.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 227.70 | 235.73 | 230.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:45:00 | 226.65 | 235.73 | 230.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 224.47 | 226.93 | 227.23 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 229.62 | 227.46 | 227.45 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 14:15:00 | 225.17 | 227.39 | 227.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 213.98 | 223.00 | 225.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 12:15:00 | 220.59 | 219.97 | 222.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:00:00 | 220.59 | 219.97 | 222.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 224.69 | 220.71 | 222.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 224.69 | 220.71 | 222.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 221.51 | 220.87 | 222.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 216.29 | 220.87 | 222.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 09:30:00 | 219.90 | 215.02 | 217.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 220.01 | 218.77 | 218.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 224.75 | 219.65 | 219.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 224.75 | 219.65 | 219.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 225.98 | 223.21 | 221.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 244.00 | 244.87 | 236.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 244.00 | 244.87 | 236.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 240.34 | 244.21 | 240.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 240.34 | 244.21 | 240.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 243.65 | 244.10 | 241.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 248.30 | 243.48 | 241.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 237.85 | 243.49 | 243.10 | SL hit (close<static) qty=1.00 sl=240.20 alert=retest2 |

### Cycle 131 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 248.11 | 251.75 | 252.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 244.50 | 249.25 | 250.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 247.66 | 246.02 | 248.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 247.66 | 246.02 | 248.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 247.66 | 246.02 | 248.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 248.50 | 246.02 | 248.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 247.80 | 246.38 | 248.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 247.93 | 246.38 | 248.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 248.15 | 246.73 | 248.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:15:00 | 251.91 | 246.73 | 248.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 251.49 | 247.68 | 248.45 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 251.72 | 249.23 | 249.07 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 247.01 | 248.83 | 249.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 14:15:00 | 245.39 | 247.82 | 248.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 240.70 | 240.66 | 243.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 14:00:00 | 240.70 | 240.66 | 243.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 239.82 | 239.90 | 242.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 238.72 | 239.89 | 242.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 237.82 | 239.48 | 241.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:00:00 | 239.35 | 236.97 | 237.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 240.00 | 237.57 | 237.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 240.00 | 237.57 | 237.36 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 236.36 | 237.87 | 237.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 235.42 | 237.06 | 237.54 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-08-22 11:15:00 | 641.80 | 2024-08-30 09:15:00 | 638.40 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2024-08-22 12:45:00 | 638.00 | 2024-08-30 09:15:00 | 638.40 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-08-23 12:30:00 | 641.55 | 2024-08-30 09:15:00 | 638.40 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2024-08-23 14:45:00 | 639.70 | 2024-08-30 09:15:00 | 638.40 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-08-26 10:15:00 | 631.40 | 2024-08-30 09:15:00 | 638.40 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-08-27 09:30:00 | 628.55 | 2024-08-30 09:15:00 | 638.40 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-08-29 09:30:00 | 630.50 | 2024-08-30 09:15:00 | 638.40 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-10-01 09:15:00 | 634.85 | 2024-10-01 11:15:00 | 650.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-10-01 10:00:00 | 639.75 | 2024-10-01 11:15:00 | 650.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest1 | 2024-10-14 09:15:00 | 686.40 | 2024-10-14 14:15:00 | 720.72 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-14 11:15:00 | 685.85 | 2024-10-14 14:15:00 | 720.14 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-14 09:15:00 | 686.40 | 2024-10-16 09:15:00 | 696.30 | STOP_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2024-10-14 11:15:00 | 685.85 | 2024-10-16 09:15:00 | 696.30 | STOP_HIT | 0.50 | 1.52% |
| BUY | retest2 | 2024-10-16 12:15:00 | 703.05 | 2024-10-17 10:15:00 | 689.05 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-10-17 09:15:00 | 706.00 | 2024-10-17 10:15:00 | 689.05 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-10-18 12:45:00 | 689.45 | 2024-10-22 09:15:00 | 654.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 13:45:00 | 688.15 | 2024-10-22 09:15:00 | 654.17 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-10-21 11:15:00 | 688.60 | 2024-10-22 10:15:00 | 653.74 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2024-10-18 12:45:00 | 689.45 | 2024-10-23 09:15:00 | 644.90 | STOP_HIT | 0.50 | 6.46% |
| SELL | retest2 | 2024-10-18 13:45:00 | 688.15 | 2024-10-23 09:15:00 | 644.90 | STOP_HIT | 0.50 | 6.28% |
| SELL | retest2 | 2024-10-21 11:15:00 | 688.60 | 2024-10-23 09:15:00 | 644.90 | STOP_HIT | 0.50 | 6.35% |
| SELL | retest2 | 2024-11-06 14:15:00 | 623.85 | 2024-11-07 14:15:00 | 592.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 15:15:00 | 623.00 | 2024-11-07 14:15:00 | 591.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-06 14:15:00 | 623.85 | 2024-11-08 09:15:00 | 561.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-06 15:15:00 | 623.00 | 2024-11-08 09:15:00 | 560.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-19 09:15:00 | 553.00 | 2024-11-25 11:15:00 | 547.70 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-11-19 09:45:00 | 550.35 | 2024-11-25 11:15:00 | 547.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-11-19 13:00:00 | 559.50 | 2024-11-25 11:15:00 | 547.70 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-11-21 10:00:00 | 551.80 | 2024-11-25 11:15:00 | 547.70 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-11-27 11:15:00 | 570.80 | 2024-12-03 11:15:00 | 590.05 | STOP_HIT | 1.00 | 3.37% |
| BUY | retest2 | 2024-11-27 12:00:00 | 571.05 | 2024-12-03 11:15:00 | 590.05 | STOP_HIT | 1.00 | 3.33% |
| SELL | retest2 | 2024-12-10 12:15:00 | 579.20 | 2024-12-11 10:15:00 | 600.70 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2024-12-10 12:45:00 | 580.00 | 2024-12-11 10:15:00 | 600.70 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2024-12-10 14:15:00 | 579.80 | 2024-12-11 10:15:00 | 600.70 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2024-12-13 11:15:00 | 598.20 | 2024-12-13 12:15:00 | 592.05 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-12-18 09:15:00 | 625.10 | 2024-12-19 13:15:00 | 604.55 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2024-12-19 09:30:00 | 609.70 | 2024-12-19 13:15:00 | 604.55 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-12-19 12:00:00 | 611.00 | 2024-12-19 13:15:00 | 604.55 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-12-27 12:30:00 | 636.90 | 2024-12-30 13:15:00 | 623.20 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-12-30 11:15:00 | 631.25 | 2024-12-30 13:15:00 | 623.20 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-01-02 15:00:00 | 652.05 | 2025-01-06 09:15:00 | 633.00 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-01-03 10:00:00 | 651.20 | 2025-01-06 09:15:00 | 633.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-01-03 10:45:00 | 650.25 | 2025-01-06 09:15:00 | 633.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-01-03 13:00:00 | 650.80 | 2025-01-06 09:15:00 | 633.00 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2025-01-17 09:15:00 | 487.80 | 2025-01-21 12:15:00 | 463.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 10:00:00 | 494.75 | 2025-01-21 12:15:00 | 470.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 09:15:00 | 487.80 | 2025-01-22 10:15:00 | 445.28 | TARGET_HIT | 0.50 | 8.72% |
| SELL | retest2 | 2025-01-17 10:00:00 | 494.75 | 2025-01-22 11:15:00 | 439.02 | TARGET_HIT | 0.50 | 11.26% |
| SELL | retest2 | 2025-02-10 11:00:00 | 432.80 | 2025-02-11 09:15:00 | 411.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:45:00 | 430.40 | 2025-02-11 09:15:00 | 408.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:00:00 | 432.80 | 2025-02-11 14:15:00 | 418.50 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-02-10 11:45:00 | 430.40 | 2025-02-11 14:15:00 | 418.50 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2025-03-03 09:15:00 | 373.70 | 2025-03-06 09:15:00 | 411.20 | STOP_HIT | 1.00 | -10.03% |
| SELL | retest2 | 2025-03-03 14:30:00 | 374.50 | 2025-03-06 09:15:00 | 411.20 | STOP_HIT | 1.00 | -9.80% |
| SELL | retest2 | 2025-03-04 09:15:00 | 372.80 | 2025-03-06 09:15:00 | 411.20 | STOP_HIT | 1.00 | -10.30% |
| SELL | retest2 | 2025-03-13 09:15:00 | 375.90 | 2025-03-19 13:15:00 | 376.05 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-03-13 12:15:00 | 379.00 | 2025-03-19 13:15:00 | 376.05 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-05-02 12:15:00 | 331.15 | 2025-05-07 09:15:00 | 314.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 14:15:00 | 331.50 | 2025-05-07 09:15:00 | 314.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 15:00:00 | 331.20 | 2025-05-07 09:15:00 | 314.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:45:00 | 330.25 | 2025-05-07 09:15:00 | 313.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 12:15:00 | 331.15 | 2025-05-07 10:15:00 | 326.35 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2025-05-02 14:15:00 | 331.50 | 2025-05-07 10:15:00 | 326.35 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2025-05-02 15:00:00 | 331.20 | 2025-05-07 10:15:00 | 326.35 | STOP_HIT | 0.50 | 1.46% |
| SELL | retest2 | 2025-05-06 09:45:00 | 330.25 | 2025-05-07 10:15:00 | 326.35 | STOP_HIT | 0.50 | 1.18% |
| SELL | retest2 | 2025-05-06 12:30:00 | 328.90 | 2025-05-12 09:15:00 | 346.80 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2025-05-15 09:15:00 | 353.35 | 2025-05-20 12:15:00 | 347.25 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-05-15 13:30:00 | 346.55 | 2025-05-20 12:15:00 | 347.25 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-06-05 09:45:00 | 386.25 | 2025-06-06 10:15:00 | 424.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-12 13:45:00 | 383.50 | 2025-06-19 09:15:00 | 364.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 376.50 | 2025-06-19 10:15:00 | 357.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 09:45:00 | 380.05 | 2025-06-19 10:15:00 | 361.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 13:45:00 | 383.50 | 2025-06-19 15:15:00 | 345.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 376.50 | 2025-06-20 12:15:00 | 342.05 | TARGET_HIT | 0.50 | 9.15% |
| SELL | retest2 | 2025-06-18 09:45:00 | 380.05 | 2025-06-20 14:15:00 | 338.85 | TARGET_HIT | 0.50 | 10.84% |
| SELL | retest2 | 2025-07-07 15:15:00 | 370.15 | 2025-07-08 09:15:00 | 376.35 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-07-28 12:15:00 | 351.55 | 2025-08-01 10:15:00 | 357.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-07-29 10:15:00 | 351.55 | 2025-08-01 10:15:00 | 357.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-07-29 12:45:00 | 351.50 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-29 14:45:00 | 352.10 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-30 10:15:00 | 347.60 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-07-30 11:30:00 | 348.30 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-08-01 14:15:00 | 348.30 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-08-01 15:15:00 | 347.55 | 2025-08-04 09:15:00 | 355.20 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-08-06 12:15:00 | 360.30 | 2025-08-06 14:15:00 | 356.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-01 09:15:00 | 359.30 | 2025-09-04 09:15:00 | 375.55 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2025-09-01 13:30:00 | 359.55 | 2025-09-04 09:15:00 | 375.55 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2025-09-02 10:30:00 | 359.05 | 2025-09-04 09:15:00 | 375.55 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-09-02 11:15:00 | 359.50 | 2025-09-04 09:15:00 | 375.55 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-09-12 10:45:00 | 392.90 | 2025-09-15 11:15:00 | 396.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-12 11:45:00 | 392.55 | 2025-09-15 11:15:00 | 396.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-09-12 13:30:00 | 390.55 | 2025-09-15 11:15:00 | 396.85 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-18 12:00:00 | 389.15 | 2025-09-26 11:15:00 | 369.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 13:00:00 | 389.10 | 2025-09-26 11:15:00 | 369.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 09:30:00 | 387.45 | 2025-09-26 11:15:00 | 368.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:15:00 | 388.80 | 2025-09-26 11:15:00 | 369.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 389.30 | 2025-09-26 11:15:00 | 369.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:00:00 | 388.45 | 2025-09-26 11:15:00 | 369.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:30:00 | 389.20 | 2025-09-26 11:15:00 | 369.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 15:15:00 | 386.45 | 2025-09-26 11:15:00 | 367.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 380.30 | 2025-09-26 15:15:00 | 361.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:30:00 | 379.35 | 2025-09-26 15:15:00 | 361.47 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-25 14:00:00 | 380.50 | 2025-09-29 09:15:00 | 360.38 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2025-09-18 12:00:00 | 389.15 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.73% |
| SELL | retest2 | 2025-09-18 13:00:00 | 389.10 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.72% |
| SELL | retest2 | 2025-09-19 09:30:00 | 387.45 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2025-09-19 10:15:00 | 388.80 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.65% |
| SELL | retest2 | 2025-09-22 09:15:00 | 389.30 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2025-09-22 11:00:00 | 388.45 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.56% |
| SELL | retest2 | 2025-09-23 13:30:00 | 389.20 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2025-09-23 15:15:00 | 386.45 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2025-09-24 10:15:00 | 380.30 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-09-25 12:30:00 | 379.35 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-09-25 14:00:00 | 380.50 | 2025-09-30 09:15:00 | 366.85 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-10-16 10:15:00 | 364.35 | 2025-10-28 11:15:00 | 346.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:45:00 | 363.25 | 2025-10-28 11:15:00 | 345.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 09:30:00 | 362.50 | 2025-10-28 11:15:00 | 344.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-17 10:45:00 | 364.25 | 2025-10-28 11:15:00 | 346.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 11:15:00 | 362.05 | 2025-10-28 11:15:00 | 343.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 10:15:00 | 364.35 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-10-16 10:45:00 | 363.25 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-10-17 09:30:00 | 362.50 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2025-10-17 10:45:00 | 364.25 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2025-10-23 11:15:00 | 362.05 | 2025-10-28 14:15:00 | 351.85 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2025-11-24 11:15:00 | 315.35 | 2025-11-26 14:15:00 | 315.45 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-11-26 14:15:00 | 314.85 | 2025-11-26 14:15:00 | 315.45 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-12-04 14:45:00 | 297.60 | 2025-12-08 10:15:00 | 282.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 14:45:00 | 297.60 | 2025-12-09 09:15:00 | 290.25 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2025-12-09 14:00:00 | 292.05 | 2025-12-12 10:15:00 | 288.00 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest1 | 2026-01-13 12:30:00 | 271.20 | 2026-01-14 10:15:00 | 257.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-13 12:30:00 | 271.20 | 2026-01-14 11:15:00 | 272.00 | STOP_HIT | 0.50 | -0.29% |
| SELL | retest1 | 2026-01-13 13:45:00 | 270.80 | 2026-01-16 12:15:00 | 272.25 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-14 13:45:00 | 267.60 | 2026-01-16 15:15:00 | 273.20 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-01-16 09:15:00 | 267.55 | 2026-01-16 15:15:00 | 273.20 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-01-16 09:45:00 | 268.00 | 2026-01-16 15:15:00 | 273.20 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-03 09:15:00 | 265.35 | 2026-02-03 09:15:00 | 271.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-02-03 11:00:00 | 266.20 | 2026-02-03 11:15:00 | 270.10 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-03 13:15:00 | 267.85 | 2026-02-03 14:15:00 | 270.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-16 09:15:00 | 247.35 | 2026-02-17 09:15:00 | 234.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 09:15:00 | 247.35 | 2026-02-18 09:15:00 | 222.62 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-05 14:15:00 | 216.78 | 2026-03-05 15:15:00 | 215.25 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-03-30 09:15:00 | 216.29 | 2026-04-02 13:15:00 | 224.75 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2026-04-01 09:30:00 | 219.90 | 2026-04-02 13:15:00 | 224.75 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-04-01 13:30:00 | 220.01 | 2026-04-02 13:15:00 | 224.75 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2026-04-10 09:30:00 | 248.30 | 2026-04-13 09:15:00 | 237.85 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2026-04-13 15:00:00 | 245.19 | 2026-04-23 14:15:00 | 248.11 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2026-05-04 10:45:00 | 238.72 | 2026-05-06 15:15:00 | 240.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-05-04 12:00:00 | 237.82 | 2026-05-06 15:15:00 | 240.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-05-06 15:00:00 | 239.35 | 2026-05-06 15:15:00 | 240.00 | STOP_HIT | 1.00 | -0.27% |
