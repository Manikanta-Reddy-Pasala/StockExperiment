# Gravita India Ltd. (GRAVITA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1760.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 208 |
| ALERT1 | 136 |
| ALERT2 | 133 |
| ALERT2_SKIP | 55 |
| ALERT3 | 353 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 146 |
| PARTIAL | 16 |
| TARGET_HIT | 14 |
| STOP_HIT | 133 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 163 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 71 / 92
- **Target hits / Stop hits / Partials:** 14 / 133 / 16
- **Avg / median % per leg:** 0.73% / -0.34%
- **Sum % (uncompounded):** 119.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 20 | 34.5% | 12 | 46 | 0 | 1.15% | 66.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.04% | -2.1% |
| BUY @ 3rd Alert (retest2) | 56 | 20 | 35.7% | 12 | 44 | 0 | 1.23% | 69.0% |
| SELL (all) | 105 | 51 | 48.6% | 2 | 87 | 16 | 0.50% | 52.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 105 | 51 | 48.6% | 2 | 87 | 16 | 0.50% | 52.2% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.04% | -2.1% |
| retest2 (combined) | 161 | 71 | 44.1% | 14 | 131 | 16 | 0.75% | 121.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 13:15:00 | 565.10 | 568.30 | 568.71 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 15:15:00 | 572.45 | 568.84 | 568.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 09:15:00 | 584.00 | 571.87 | 570.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 15:15:00 | 576.25 | 576.81 | 573.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-18 09:15:00 | 574.10 | 576.81 | 573.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 571.60 | 575.76 | 573.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:00:00 | 571.60 | 575.76 | 573.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 571.25 | 574.86 | 573.45 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 13:15:00 | 566.50 | 572.36 | 572.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 557.80 | 569.17 | 571.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 14:15:00 | 568.65 | 565.55 | 568.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 14:15:00 | 568.65 | 565.55 | 568.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 568.65 | 565.55 | 568.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 568.65 | 565.55 | 568.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 566.10 | 565.66 | 567.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 564.75 | 565.66 | 567.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 569.30 | 566.39 | 568.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:45:00 | 568.90 | 566.39 | 568.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 571.50 | 567.41 | 568.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:45:00 | 573.15 | 567.41 | 568.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 10:15:00 | 568.00 | 565.25 | 566.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:00:00 | 568.00 | 565.25 | 566.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 567.65 | 565.73 | 566.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:30:00 | 568.35 | 565.73 | 566.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 565.00 | 565.58 | 566.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 13:15:00 | 562.70 | 565.58 | 566.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 12:15:00 | 558.05 | 555.23 | 554.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 12:15:00 | 558.05 | 555.23 | 554.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 13:15:00 | 563.15 | 556.81 | 555.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 09:15:00 | 574.30 | 576.77 | 570.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 09:30:00 | 575.75 | 576.77 | 570.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 601.60 | 624.86 | 622.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 15:00:00 | 601.60 | 624.86 | 622.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 15:15:00 | 596.50 | 619.19 | 620.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-07 10:15:00 | 595.95 | 610.89 | 616.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 14:15:00 | 606.95 | 604.05 | 610.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-07 15:00:00 | 606.95 | 604.05 | 610.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 604.75 | 604.50 | 609.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 10:00:00 | 596.30 | 601.57 | 605.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 13:45:00 | 596.85 | 599.86 | 603.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 14:15:00 | 596.00 | 599.86 | 603.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 09:15:00 | 588.00 | 599.16 | 602.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 599.45 | 592.52 | 596.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-14 10:15:00 | 608.90 | 598.78 | 597.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 10:15:00 | 608.90 | 598.78 | 597.63 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 15:15:00 | 596.00 | 600.09 | 600.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-16 15:15:00 | 594.00 | 597.78 | 598.86 | Break + close below crossover candle low |

### Cycle 8 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 614.35 | 601.10 | 600.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 11:15:00 | 619.30 | 606.44 | 602.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 09:15:00 | 616.30 | 624.43 | 618.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 616.30 | 624.43 | 618.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 616.30 | 624.43 | 618.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:00:00 | 616.30 | 624.43 | 618.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 617.65 | 623.07 | 618.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 11:15:00 | 618.65 | 623.07 | 618.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 617.65 | 621.99 | 618.59 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 613.95 | 617.10 | 617.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 607.30 | 615.14 | 616.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 14:15:00 | 615.95 | 614.54 | 615.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 14:15:00 | 615.95 | 614.54 | 615.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 615.95 | 614.54 | 615.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 15:00:00 | 615.95 | 614.54 | 615.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 617.90 | 615.22 | 616.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 614.50 | 615.22 | 616.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 609.95 | 614.16 | 615.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 10:00:00 | 601.90 | 607.05 | 608.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 11:00:00 | 605.10 | 606.66 | 608.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 13:00:00 | 604.20 | 605.92 | 607.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 09:45:00 | 603.80 | 604.42 | 606.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 605.85 | 604.70 | 606.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:30:00 | 605.10 | 604.70 | 606.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 11:15:00 | 606.65 | 605.09 | 606.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 11:45:00 | 606.50 | 605.09 | 606.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 12:15:00 | 606.20 | 605.31 | 606.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 12:30:00 | 606.95 | 605.31 | 606.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-04 13:15:00 | 616.20 | 607.49 | 607.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 13:15:00 | 616.20 | 607.49 | 607.18 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 15:15:00 | 609.70 | 612.87 | 613.00 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 11:15:00 | 617.95 | 613.76 | 613.37 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 14:15:00 | 607.25 | 612.90 | 613.29 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 12:15:00 | 617.05 | 613.44 | 613.28 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 14:15:00 | 609.95 | 612.90 | 613.07 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 617.10 | 613.52 | 613.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 627.95 | 618.64 | 616.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 14:15:00 | 625.85 | 626.62 | 621.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 15:00:00 | 625.85 | 626.62 | 621.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 620.25 | 625.68 | 622.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 10:00:00 | 642.55 | 626.36 | 623.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 15:15:00 | 637.00 | 657.17 | 657.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 15:15:00 | 637.00 | 657.17 | 657.82 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 10:15:00 | 663.00 | 659.06 | 658.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 15:15:00 | 670.00 | 665.11 | 662.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 09:15:00 | 676.75 | 678.67 | 672.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-27 10:00:00 | 676.75 | 678.67 | 672.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 11:15:00 | 674.60 | 677.47 | 673.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 11:45:00 | 673.50 | 677.47 | 673.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 12:15:00 | 676.80 | 677.33 | 673.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 12:30:00 | 675.00 | 677.33 | 673.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 672.00 | 676.21 | 673.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 15:00:00 | 672.00 | 676.21 | 673.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 665.80 | 674.12 | 673.06 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 10:15:00 | 667.45 | 671.57 | 672.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 11:15:00 | 664.30 | 670.12 | 671.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 674.75 | 665.56 | 667.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 674.75 | 665.56 | 667.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 674.75 | 665.56 | 667.99 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 12:15:00 | 674.40 | 669.90 | 669.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 679.85 | 673.20 | 671.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 13:15:00 | 670.95 | 675.59 | 673.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 13:15:00 | 670.95 | 675.59 | 673.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 670.95 | 675.59 | 673.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:00:00 | 670.95 | 675.59 | 673.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 675.20 | 675.51 | 673.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 15:15:00 | 678.80 | 675.51 | 673.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 668.00 | 676.41 | 675.31 | SL hit (close<static) qty=1.00 sl=668.05 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 11:15:00 | 717.00 | 726.62 | 727.91 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 735.25 | 729.79 | 729.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 15:15:00 | 737.30 | 731.29 | 729.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 15:15:00 | 737.05 | 739.06 | 735.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 09:15:00 | 743.75 | 739.06 | 735.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 739.00 | 744.68 | 742.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:00:00 | 739.00 | 744.68 | 742.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 738.75 | 743.50 | 742.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 738.65 | 743.50 | 742.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 741.40 | 742.05 | 742.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 09:15:00 | 750.00 | 742.47 | 742.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 09:15:00 | 780.40 | 783.59 | 783.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 09:15:00 | 780.40 | 783.59 | 783.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 15:15:00 | 769.00 | 778.16 | 780.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 09:15:00 | 779.60 | 778.45 | 780.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-08 10:15:00 | 782.00 | 778.45 | 780.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 785.95 | 779.95 | 781.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 10:45:00 | 785.50 | 779.95 | 781.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 11:15:00 | 787.95 | 781.55 | 781.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 12:00:00 | 787.95 | 781.55 | 781.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 12:15:00 | 784.35 | 782.11 | 782.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 10:15:00 | 804.25 | 788.12 | 785.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 775.20 | 791.17 | 789.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 775.20 | 791.17 | 789.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 775.20 | 791.17 | 789.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 774.75 | 791.17 | 789.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 784.00 | 789.73 | 788.55 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 777.40 | 787.27 | 787.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 772.85 | 784.38 | 786.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 755.90 | 751.93 | 765.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 13:00:00 | 755.90 | 751.93 | 765.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 778.00 | 759.75 | 765.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 781.70 | 759.75 | 765.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 786.90 | 765.18 | 767.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 782.00 | 765.18 | 767.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 773.45 | 769.42 | 768.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 784.95 | 774.06 | 771.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 816.35 | 822.46 | 808.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-20 09:45:00 | 821.45 | 822.46 | 808.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 835.10 | 824.99 | 811.07 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 804.05 | 813.37 | 814.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 09:15:00 | 790.90 | 807.47 | 811.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 796.85 | 796.23 | 802.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 10:15:00 | 801.30 | 796.23 | 802.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 796.30 | 796.25 | 801.74 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 09:15:00 | 832.75 | 808.28 | 805.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 10:15:00 | 852.95 | 817.21 | 809.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 11:15:00 | 907.50 | 913.68 | 891.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-29 11:30:00 | 905.95 | 913.68 | 891.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 883.15 | 905.72 | 896.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:00:00 | 883.15 | 905.72 | 896.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 892.00 | 902.97 | 895.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 10:30:00 | 885.40 | 902.97 | 895.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 11:15:00 | 895.55 | 901.49 | 895.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:30:00 | 892.50 | 901.49 | 895.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 895.90 | 899.12 | 895.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 13:45:00 | 892.90 | 899.12 | 895.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 883.05 | 895.90 | 894.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 15:00:00 | 883.05 | 895.90 | 894.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 15:15:00 | 883.00 | 893.32 | 893.48 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 09:15:00 | 907.45 | 896.15 | 894.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 14:15:00 | 912.85 | 904.60 | 900.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 14:15:00 | 923.05 | 924.75 | 915.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-06 15:00:00 | 923.05 | 924.75 | 915.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 919.85 | 923.77 | 915.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 09:15:00 | 865.10 | 923.77 | 915.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 880.50 | 915.12 | 912.34 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 875.00 | 907.09 | 908.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 11:15:00 | 870.55 | 899.79 | 905.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 903.40 | 891.25 | 897.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 903.40 | 891.25 | 897.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 903.40 | 891.25 | 897.96 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 902.95 | 899.24 | 899.16 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 09:15:00 | 892.20 | 898.25 | 899.02 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 12:15:00 | 925.85 | 904.49 | 901.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 13:15:00 | 947.00 | 912.99 | 905.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 979.05 | 986.17 | 975.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 11:00:00 | 979.05 | 986.17 | 975.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 982.85 | 985.50 | 975.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:30:00 | 984.85 | 985.50 | 975.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 983.05 | 985.01 | 976.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:45:00 | 976.85 | 985.01 | 976.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 981.90 | 983.95 | 978.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:15:00 | 965.00 | 983.95 | 978.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 982.25 | 983.61 | 978.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:45:00 | 980.90 | 983.61 | 978.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 993.80 | 993.15 | 986.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 1000.70 | 993.15 | 986.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 10:30:00 | 1001.00 | 995.81 | 988.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 09:15:00 | 972.80 | 989.81 | 988.66 | SL hit (close<static) qty=1.00 sl=985.65 alert=retest2 |

### Cycle 35 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 972.15 | 985.09 | 986.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 961.05 | 980.29 | 984.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 13:15:00 | 963.10 | 935.40 | 948.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 13:15:00 | 963.10 | 935.40 | 948.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 13:15:00 | 963.10 | 935.40 | 948.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 14:00:00 | 963.10 | 935.40 | 948.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 944.10 | 937.14 | 947.66 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 979.85 | 953.13 | 952.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 11:15:00 | 983.00 | 959.10 | 955.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 14:15:00 | 982.05 | 984.39 | 974.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-30 14:45:00 | 985.00 | 984.39 | 974.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 1044.45 | 1060.75 | 1044.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 10:00:00 | 1069.00 | 1062.40 | 1046.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 10:15:00 | 1111.30 | 1120.54 | 1120.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 10:15:00 | 1111.30 | 1120.54 | 1120.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 12:15:00 | 1102.95 | 1115.53 | 1118.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 15:15:00 | 1114.90 | 1110.05 | 1114.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 15:15:00 | 1114.90 | 1110.05 | 1114.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 1114.90 | 1110.05 | 1114.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 1133.20 | 1110.05 | 1114.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 1122.00 | 1112.44 | 1115.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:15:00 | 1136.00 | 1112.44 | 1115.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 1146.85 | 1119.32 | 1118.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 10:15:00 | 1161.55 | 1127.77 | 1122.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-15 10:15:00 | 1138.10 | 1144.69 | 1136.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-15 10:45:00 | 1138.95 | 1144.69 | 1136.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 11:15:00 | 1141.70 | 1144.09 | 1136.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 12:15:00 | 1146.10 | 1144.09 | 1136.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 09:15:00 | 1125.00 | 1138.06 | 1136.37 | SL hit (close<static) qty=1.00 sl=1136.50 alert=retest2 |

### Cycle 39 — SELL (started 2023-11-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 11:15:00 | 1124.75 | 1133.22 | 1134.33 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 1139.60 | 1135.42 | 1135.05 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 10:15:00 | 1123.15 | 1133.85 | 1134.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 11:15:00 | 1121.55 | 1131.39 | 1133.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 13:15:00 | 1131.35 | 1130.90 | 1133.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-20 14:00:00 | 1131.35 | 1130.90 | 1133.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 1130.00 | 1130.72 | 1132.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 15:15:00 | 1132.00 | 1130.72 | 1132.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 1132.00 | 1130.97 | 1132.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:15:00 | 1139.75 | 1130.97 | 1132.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 1138.25 | 1132.43 | 1133.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:45:00 | 1138.75 | 1132.43 | 1133.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 1132.05 | 1132.35 | 1133.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 11:30:00 | 1128.95 | 1129.78 | 1131.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 12:30:00 | 1127.85 | 1129.77 | 1131.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 13:00:00 | 1129.70 | 1129.77 | 1131.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 14:00:00 | 1125.30 | 1128.87 | 1131.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 14:15:00 | 1117.90 | 1126.68 | 1129.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 09:45:00 | 1112.95 | 1122.74 | 1127.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 12:15:00 | 1111.50 | 1121.19 | 1125.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 12:15:00 | 1072.50 | 1097.27 | 1109.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 12:15:00 | 1071.46 | 1097.27 | 1109.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 12:15:00 | 1073.21 | 1097.27 | 1109.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 15:15:00 | 1069.03 | 1087.05 | 1101.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 1099.95 | 1089.63 | 1101.20 | SL hit (close>ema200) qty=0.50 sl=1089.63 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 15:15:00 | 1128.00 | 1106.82 | 1105.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 1128.50 | 1111.16 | 1107.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 13:15:00 | 1119.95 | 1123.33 | 1118.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 13:15:00 | 1119.95 | 1123.33 | 1118.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 1119.95 | 1123.33 | 1118.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 14:00:00 | 1119.95 | 1123.33 | 1118.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 1119.35 | 1122.54 | 1118.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 14:30:00 | 1119.00 | 1122.54 | 1118.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 1104.70 | 1118.97 | 1117.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:15:00 | 1115.95 | 1118.97 | 1117.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 1110.00 | 1117.18 | 1116.79 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 10:15:00 | 1109.00 | 1115.54 | 1116.09 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 1122.00 | 1116.52 | 1116.42 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 13:15:00 | 1114.35 | 1116.09 | 1116.23 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 1119.50 | 1116.77 | 1116.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 11:15:00 | 1120.45 | 1117.50 | 1116.97 | Break + close above crossover candle high |

### Cycle 47 — SELL (started 2023-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 14:15:00 | 1110.25 | 1116.31 | 1116.59 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 10:15:00 | 1123.75 | 1117.86 | 1117.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 11:15:00 | 1134.70 | 1121.23 | 1118.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 14:15:00 | 1118.95 | 1123.36 | 1120.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 14:15:00 | 1118.95 | 1123.36 | 1120.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 1118.95 | 1123.36 | 1120.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 15:00:00 | 1118.95 | 1123.36 | 1120.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 15:15:00 | 1120.35 | 1122.76 | 1120.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 09:30:00 | 1104.10 | 1119.80 | 1119.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 10:15:00 | 1097.20 | 1115.28 | 1117.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 11:15:00 | 1086.90 | 1109.61 | 1114.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 1074.35 | 1071.00 | 1084.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-07 09:45:00 | 1077.00 | 1071.00 | 1084.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 11:15:00 | 1092.00 | 1076.50 | 1084.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 12:00:00 | 1092.00 | 1076.50 | 1084.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 12:15:00 | 1073.80 | 1075.96 | 1083.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 09:45:00 | 1072.65 | 1073.19 | 1079.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 14:45:00 | 1073.45 | 1069.63 | 1075.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 09:30:00 | 1068.40 | 1068.94 | 1073.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-18 09:15:00 | 1073.80 | 1052.42 | 1049.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2023-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 09:15:00 | 1073.80 | 1052.42 | 1049.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 09:15:00 | 1088.65 | 1069.09 | 1060.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 09:15:00 | 1079.30 | 1084.09 | 1074.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 10:15:00 | 1074.45 | 1082.16 | 1074.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 1074.45 | 1082.16 | 1074.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:00:00 | 1074.45 | 1082.16 | 1074.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 1073.40 | 1080.41 | 1074.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 12:00:00 | 1073.40 | 1080.41 | 1074.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 1069.50 | 1078.23 | 1073.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 12:30:00 | 1069.70 | 1078.23 | 1073.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1040.00 | 1065.65 | 1068.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 1031.05 | 1058.73 | 1065.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1046.10 | 1037.27 | 1046.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 1046.10 | 1037.27 | 1046.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1046.10 | 1037.27 | 1046.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:30:00 | 1038.50 | 1040.80 | 1046.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 10:00:00 | 1038.80 | 1042.81 | 1045.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 10:30:00 | 1032.00 | 1040.34 | 1044.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 14:15:00 | 1077.65 | 1042.07 | 1043.18 | SL hit (close>static) qty=1.00 sl=1054.95 alert=retest2 |

### Cycle 52 — BUY (started 2023-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 15:15:00 | 1092.00 | 1052.06 | 1047.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 15:15:00 | 1116.00 | 1082.47 | 1066.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 1092.00 | 1092.81 | 1079.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 14:45:00 | 1092.95 | 1092.81 | 1079.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 1084.55 | 1089.43 | 1080.51 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 1069.90 | 1080.93 | 1082.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 1063.45 | 1068.46 | 1071.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 13:15:00 | 1060.60 | 1058.66 | 1063.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 13:15:00 | 1060.60 | 1058.66 | 1063.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 1060.60 | 1058.66 | 1063.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 13:30:00 | 1060.00 | 1058.66 | 1063.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 1034.40 | 1040.42 | 1048.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 12:00:00 | 1029.05 | 1036.92 | 1045.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 12:15:00 | 1033.00 | 1024.24 | 1023.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 12:15:00 | 1033.00 | 1024.24 | 1023.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 14:15:00 | 1062.00 | 1034.63 | 1028.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 09:15:00 | 1034.05 | 1040.18 | 1032.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 1034.05 | 1040.18 | 1032.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1034.05 | 1040.18 | 1032.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:00:00 | 1034.05 | 1040.18 | 1032.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 1029.80 | 1038.10 | 1031.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:30:00 | 1031.20 | 1038.10 | 1031.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 1024.60 | 1035.40 | 1031.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 11:30:00 | 1023.80 | 1035.40 | 1031.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 14:15:00 | 1027.00 | 1032.19 | 1030.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 15:00:00 | 1027.00 | 1032.19 | 1030.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 15:15:00 | 1031.05 | 1031.96 | 1030.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:15:00 | 1015.95 | 1031.96 | 1030.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 1001.45 | 1025.86 | 1028.10 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 12:15:00 | 1023.05 | 1021.13 | 1020.87 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 13:15:00 | 1015.35 | 1019.97 | 1020.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 996.80 | 1015.41 | 1018.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 10:15:00 | 932.00 | 912.46 | 944.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-25 11:00:00 | 932.00 | 912.46 | 944.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 937.40 | 917.45 | 943.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:30:00 | 941.80 | 917.45 | 943.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 952.95 | 924.55 | 944.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 13:00:00 | 952.95 | 924.55 | 944.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 947.10 | 929.06 | 944.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 13:45:00 | 951.95 | 929.06 | 944.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 944.65 | 932.18 | 944.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 944.65 | 932.18 | 944.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 939.70 | 933.68 | 944.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 928.85 | 933.68 | 944.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 927.30 | 932.40 | 942.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 11:15:00 | 919.05 | 931.80 | 941.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 10:15:00 | 873.10 | 895.97 | 916.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-31 09:15:00 | 880.00 | 874.96 | 894.81 | SL hit (close>ema200) qty=0.50 sl=874.96 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 922.85 | 902.34 | 902.27 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 09:15:00 | 870.05 | 900.31 | 901.64 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 12:15:00 | 898.10 | 889.90 | 889.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 15:15:00 | 900.00 | 892.51 | 890.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 14:15:00 | 897.30 | 904.54 | 901.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 14:15:00 | 897.30 | 904.54 | 901.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 897.30 | 904.54 | 901.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 15:00:00 | 897.30 | 904.54 | 901.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 897.00 | 903.03 | 901.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 09:15:00 | 902.20 | 903.03 | 901.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 891.95 | 900.81 | 900.27 | SL hit (close<static) qty=1.00 sl=892.25 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 896.00 | 899.85 | 899.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 12:15:00 | 880.45 | 887.23 | 892.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 898.00 | 886.72 | 889.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 10:15:00 | 898.00 | 886.72 | 889.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 898.00 | 886.72 | 889.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:00:00 | 898.00 | 886.72 | 889.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 888.00 | 886.98 | 889.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:30:00 | 897.60 | 886.98 | 889.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 898.80 | 889.34 | 890.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:00:00 | 898.80 | 889.34 | 890.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 13:15:00 | 902.00 | 891.87 | 891.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 934.05 | 907.96 | 901.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 12:15:00 | 929.00 | 936.62 | 925.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 13:00:00 | 929.00 | 936.62 | 925.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 930.50 | 935.40 | 926.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:30:00 | 924.95 | 935.40 | 926.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 922.95 | 932.91 | 926.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 15:00:00 | 922.95 | 932.91 | 926.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 932.00 | 932.73 | 926.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 932.95 | 932.73 | 926.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:45:00 | 936.80 | 933.85 | 927.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 12:15:00 | 916.05 | 927.88 | 926.29 | SL hit (close<static) qty=1.00 sl=918.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 907.00 | 922.01 | 923.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 903.10 | 911.92 | 916.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 924.60 | 911.05 | 914.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 924.60 | 911.05 | 914.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 924.60 | 911.05 | 914.34 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 11:15:00 | 927.50 | 917.05 | 916.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 15:15:00 | 929.00 | 923.33 | 920.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 09:15:00 | 968.45 | 971.86 | 959.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-27 10:00:00 | 968.45 | 971.86 | 959.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 954.00 | 966.06 | 960.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:00:00 | 954.00 | 966.06 | 960.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 965.55 | 965.96 | 960.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:15:00 | 958.00 | 965.96 | 960.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 958.00 | 964.37 | 960.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 09:15:00 | 976.80 | 964.37 | 960.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 947.30 | 958.46 | 958.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 947.30 | 958.46 | 958.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 936.55 | 954.08 | 956.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 15:15:00 | 951.00 | 950.84 | 954.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 09:15:00 | 946.50 | 950.84 | 954.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 944.40 | 949.55 | 953.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 09:15:00 | 928.70 | 943.32 | 946.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:15:00 | 882.26 | 898.23 | 911.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-07 12:15:00 | 879.80 | 873.71 | 887.47 | SL hit (close>ema200) qty=0.50 sl=873.71 alert=retest2 |

### Cycle 66 — BUY (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 09:15:00 | 815.60 | 806.70 | 806.12 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 14:15:00 | 803.00 | 805.80 | 805.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 15:15:00 | 800.15 | 804.67 | 805.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-19 09:15:00 | 806.55 | 805.05 | 805.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 09:15:00 | 806.55 | 805.05 | 805.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 806.55 | 805.05 | 805.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-19 09:30:00 | 806.65 | 805.05 | 805.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 804.50 | 804.94 | 805.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 15:15:00 | 795.00 | 802.20 | 803.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 834.15 | 799.83 | 798.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 834.15 | 799.83 | 798.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 840.60 | 819.80 | 809.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 11:15:00 | 872.30 | 873.51 | 855.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 12:00:00 | 872.30 | 873.51 | 855.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 1091.20 | 1091.12 | 1084.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 15:15:00 | 1085.10 | 1091.12 | 1084.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 1085.10 | 1089.91 | 1084.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 09:15:00 | 1081.40 | 1089.91 | 1084.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 1080.00 | 1087.93 | 1084.26 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 12:15:00 | 1030.00 | 1074.10 | 1078.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 1015.00 | 1027.73 | 1041.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 11:15:00 | 992.85 | 981.72 | 992.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 11:15:00 | 992.85 | 981.72 | 992.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 992.85 | 981.72 | 992.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:45:00 | 997.00 | 981.72 | 992.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 985.00 | 982.38 | 991.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 15:00:00 | 979.25 | 982.92 | 990.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 11:00:00 | 979.00 | 973.24 | 978.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 11:30:00 | 979.30 | 974.46 | 978.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 10:45:00 | 979.70 | 976.71 | 977.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 975.05 | 976.76 | 977.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:15:00 | 979.10 | 976.76 | 977.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 977.20 | 976.85 | 977.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:30:00 | 980.00 | 976.85 | 977.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 972.35 | 975.95 | 977.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 14:30:00 | 978.70 | 975.95 | 977.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 974.65 | 974.98 | 976.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 10:45:00 | 972.50 | 973.71 | 975.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 12:45:00 | 968.80 | 973.30 | 975.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 15:00:00 | 970.45 | 971.94 | 974.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 09:45:00 | 972.85 | 972.14 | 973.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 969.15 | 971.54 | 973.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 968.00 | 971.54 | 973.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 970.00 | 971.23 | 973.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:45:00 | 972.30 | 971.23 | 973.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 972.80 | 971.20 | 972.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 13:30:00 | 971.00 | 971.20 | 972.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 968.00 | 970.56 | 972.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:45:00 | 972.50 | 970.56 | 972.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 975.85 | 971.52 | 972.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-26 12:15:00 | 975.70 | 973.37 | 973.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 12:15:00 | 975.70 | 973.37 | 973.18 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-04-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 13:15:00 | 970.70 | 972.84 | 972.95 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 09:15:00 | 982.95 | 974.68 | 973.76 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 916.75 | 966.89 | 972.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 900.80 | 924.91 | 937.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 923.05 | 915.68 | 926.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 923.05 | 915.68 | 926.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 921.50 | 915.15 | 923.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:45:00 | 922.80 | 915.15 | 923.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 924.85 | 917.09 | 923.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:00:00 | 924.85 | 917.09 | 923.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 916.45 | 916.96 | 923.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 15:15:00 | 909.00 | 916.96 | 923.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:00:00 | 913.40 | 914.97 | 921.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:15:00 | 913.55 | 915.54 | 920.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 14:30:00 | 914.65 | 913.99 | 918.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 912.70 | 912.29 | 916.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 898.85 | 911.12 | 914.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 11:15:00 | 921.20 | 911.30 | 910.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 921.20 | 911.30 | 910.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 945.95 | 920.25 | 914.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 13:15:00 | 941.20 | 943.54 | 932.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:30:00 | 939.15 | 943.54 | 932.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 937.10 | 942.03 | 937.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 937.10 | 942.03 | 937.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 939.55 | 941.53 | 937.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:15:00 | 940.00 | 941.53 | 937.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 940.00 | 941.23 | 937.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 973.00 | 941.23 | 937.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 13:00:00 | 954.00 | 953.52 | 952.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 950.80 | 952.61 | 951.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 09:15:00 | 1049.40 | 1004.65 | 982.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 12:15:00 | 1093.80 | 1111.85 | 1112.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 1087.20 | 1106.92 | 1110.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 1080.50 | 1079.05 | 1090.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 15:00:00 | 1080.50 | 1079.05 | 1090.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1091.55 | 1079.47 | 1088.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1013.15 | 1075.15 | 1082.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 962.49 | 1027.16 | 1057.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 13:15:00 | 1041.55 | 1025.69 | 1051.15 | SL hit (close>ema200) qty=0.50 sl=1025.69 alert=retest2 |

### Cycle 76 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1080.75 | 1043.94 | 1042.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 1105.25 | 1074.79 | 1060.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 1112.05 | 1113.96 | 1097.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 1112.05 | 1113.96 | 1097.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1264.10 | 1285.89 | 1271.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 1264.10 | 1285.89 | 1271.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1281.45 | 1285.00 | 1272.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 13:45:00 | 1292.65 | 1282.56 | 1273.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:00:00 | 1399.95 | 1306.96 | 1287.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 12:15:00 | 1421.92 | 1353.02 | 1315.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 1460.00 | 1488.53 | 1488.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 1446.70 | 1469.19 | 1476.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 1493.95 | 1462.51 | 1469.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 1493.95 | 1462.51 | 1469.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1493.95 | 1462.51 | 1469.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 1493.95 | 1462.51 | 1469.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1475.00 | 1465.00 | 1469.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:30:00 | 1463.50 | 1466.46 | 1469.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 14:00:00 | 1462.95 | 1465.76 | 1468.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 09:15:00 | 1390.33 | 1425.11 | 1440.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 09:15:00 | 1389.80 | 1425.11 | 1440.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 1406.75 | 1397.03 | 1413.20 | SL hit (close>ema200) qty=0.50 sl=1397.03 alert=retest2 |

### Cycle 78 — BUY (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 15:15:00 | 1390.00 | 1377.28 | 1376.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 1423.00 | 1386.53 | 1380.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 1405.40 | 1413.52 | 1404.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 1405.40 | 1413.52 | 1404.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1405.40 | 1413.52 | 1404.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1400.20 | 1413.52 | 1404.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1384.15 | 1407.64 | 1403.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 1384.15 | 1407.64 | 1403.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1404.00 | 1406.92 | 1403.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 1405.60 | 1406.92 | 1403.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1399.95 | 1405.52 | 1402.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 1399.95 | 1405.52 | 1402.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1393.90 | 1403.20 | 1402.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:45:00 | 1394.95 | 1403.20 | 1402.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 14:15:00 | 1378.60 | 1398.28 | 1399.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1328.50 | 1383.65 | 1392.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1442.00 | 1366.54 | 1374.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1442.00 | 1366.54 | 1374.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1442.00 | 1366.54 | 1374.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 1446.00 | 1366.54 | 1374.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 1436.20 | 1380.47 | 1380.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 1497.25 | 1437.09 | 1412.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 10:15:00 | 1687.00 | 1687.08 | 1624.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 11:00:00 | 1687.00 | 1687.08 | 1624.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1663.25 | 1676.00 | 1642.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 1705.00 | 1676.00 | 1642.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 10:15:00 | 1639.05 | 1674.83 | 1674.65 | SL hit (close<static) qty=1.00 sl=1641.50 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 11:15:00 | 1641.80 | 1668.23 | 1671.66 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 14:15:00 | 1672.85 | 1658.14 | 1658.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 09:15:00 | 1682.35 | 1665.20 | 1661.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 10:15:00 | 1652.75 | 1662.71 | 1660.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 1652.75 | 1662.71 | 1660.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 1652.75 | 1662.71 | 1660.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:45:00 | 1653.60 | 1662.71 | 1660.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 1667.05 | 1663.58 | 1661.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:30:00 | 1650.00 | 1663.58 | 1661.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 1665.35 | 1665.72 | 1662.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:45:00 | 1657.30 | 1665.72 | 1662.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 1661.25 | 1664.82 | 1662.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 1574.80 | 1664.82 | 1662.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1635.05 | 1658.87 | 1660.28 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 1789.00 | 1661.89 | 1655.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1848.00 | 1767.45 | 1731.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 1794.75 | 1801.86 | 1759.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 1794.75 | 1801.86 | 1759.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1785.70 | 1798.95 | 1784.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:30:00 | 1849.65 | 1802.39 | 1791.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 13:45:00 | 1835.05 | 1817.66 | 1803.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 14:30:00 | 1838.45 | 1818.10 | 1804.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1864.85 | 1814.47 | 1804.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1861.30 | 1823.84 | 1809.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1832.80 | 1823.84 | 1809.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2024-08-16 09:15:00 | 2034.62 | 1943.82 | 1884.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 2187.00 | 2194.37 | 2195.17 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 15:15:00 | 2202.00 | 2195.90 | 2195.79 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 2142.20 | 2185.16 | 2190.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 2112.85 | 2144.47 | 2162.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 2176.00 | 2082.74 | 2098.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 2176.00 | 2082.74 | 2098.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 2176.00 | 2082.74 | 2098.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 2176.00 | 2082.74 | 2098.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 2188.00 | 2103.79 | 2106.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 2188.00 | 2103.79 | 2106.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 2174.80 | 2117.99 | 2113.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 2243.95 | 2186.93 | 2164.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 2284.65 | 2286.90 | 2235.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:00:00 | 2284.65 | 2286.90 | 2235.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 2220.65 | 2271.31 | 2237.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 2220.65 | 2271.31 | 2237.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 2214.00 | 2259.85 | 2234.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 2245.00 | 2259.85 | 2234.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 2228.50 | 2249.69 | 2234.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:30:00 | 2205.05 | 2249.69 | 2234.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 2245.45 | 2248.84 | 2235.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:15:00 | 2266.00 | 2248.84 | 2235.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 2240.00 | 2286.42 | 2290.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 2240.00 | 2286.42 | 2290.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 2182.00 | 2257.31 | 2276.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 2318.80 | 2231.38 | 2246.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 2318.80 | 2231.38 | 2246.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 2318.80 | 2231.38 | 2246.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:45:00 | 2324.35 | 2231.38 | 2246.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 2302.75 | 2245.65 | 2251.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:15:00 | 2269.30 | 2245.65 | 2251.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 13:15:00 | 2285.75 | 2259.36 | 2257.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 2285.75 | 2259.36 | 2257.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 2298.00 | 2271.68 | 2263.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 2270.65 | 2276.61 | 2267.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 2270.65 | 2276.61 | 2267.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 2270.65 | 2276.61 | 2267.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:00:00 | 2270.65 | 2276.61 | 2267.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 2250.90 | 2271.46 | 2265.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:00:00 | 2250.90 | 2271.46 | 2265.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 2243.00 | 2265.77 | 2263.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:30:00 | 2280.55 | 2265.77 | 2263.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 2228.00 | 2258.22 | 2260.50 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 2293.00 | 2262.20 | 2260.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 11:15:00 | 2296.00 | 2268.96 | 2263.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 12:15:00 | 2529.60 | 2532.63 | 2449.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 13:00:00 | 2529.60 | 2532.63 | 2449.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 2595.60 | 2608.11 | 2567.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 12:15:00 | 2604.70 | 2608.11 | 2567.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 2499.00 | 2567.38 | 2561.66 | SL hit (close<static) qty=1.00 sl=2552.55 alert=retest2 |

### Cycle 93 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 2456.00 | 2545.10 | 2552.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 10:15:00 | 2445.65 | 2478.25 | 2499.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 11:15:00 | 2449.95 | 2427.47 | 2456.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:00:00 | 2449.95 | 2427.47 | 2456.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 2499.80 | 2441.94 | 2460.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:45:00 | 2512.80 | 2441.94 | 2460.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 2495.25 | 2452.60 | 2463.68 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 09:15:00 | 2581.25 | 2491.26 | 2479.67 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 2484.95 | 2527.12 | 2529.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 2462.60 | 2514.22 | 2523.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 14:15:00 | 2467.00 | 2460.19 | 2482.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 15:00:00 | 2467.00 | 2460.19 | 2482.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 2405.00 | 2369.17 | 2410.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:45:00 | 2409.00 | 2369.17 | 2410.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 2407.00 | 2379.44 | 2407.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:45:00 | 2405.25 | 2379.44 | 2407.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 2392.80 | 2382.11 | 2406.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:30:00 | 2355.65 | 2372.48 | 2399.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:00:00 | 2362.00 | 2371.58 | 2392.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 15:15:00 | 2243.90 | 2308.96 | 2348.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 2237.87 | 2305.98 | 2343.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 2359.95 | 2316.77 | 2345.16 | SL hit (close>ema200) qty=0.50 sl=2316.77 alert=retest2 |

### Cycle 96 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 2441.70 | 2363.54 | 2357.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 2463.90 | 2383.61 | 2367.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 2419.55 | 2427.24 | 2403.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:00:00 | 2419.55 | 2427.24 | 2403.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 2470.45 | 2445.15 | 2423.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 2557.90 | 2455.08 | 2438.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:00:00 | 2476.10 | 2495.61 | 2491.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 13:15:00 | 2476.50 | 2486.57 | 2487.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 2476.50 | 2486.57 | 2487.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 14:15:00 | 2460.00 | 2481.25 | 2485.12 | Break + close below crossover candle low |

### Cycle 98 — BUY (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 09:15:00 | 2583.55 | 2499.28 | 2492.49 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 2440.75 | 2488.32 | 2491.42 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 2529.00 | 2495.93 | 2492.33 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 2454.90 | 2488.24 | 2491.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 2425.00 | 2475.59 | 2485.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 2049.45 | 2027.61 | 2079.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 10:30:00 | 2070.00 | 2027.61 | 2079.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1994.95 | 2030.18 | 2060.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:15:00 | 1991.40 | 2030.18 | 2060.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:15:00 | 1988.00 | 2015.21 | 2047.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:00:00 | 1991.70 | 2010.51 | 2042.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 13:15:00 | 2085.00 | 2037.64 | 2031.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 13:15:00 | 2085.00 | 2037.64 | 2031.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 2122.00 | 2054.52 | 2040.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 14:15:00 | 2129.80 | 2138.17 | 2107.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 2129.80 | 2138.17 | 2107.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 2125.00 | 2135.54 | 2109.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 2176.10 | 2135.54 | 2109.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:30:00 | 2140.75 | 2139.33 | 2115.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:15:00 | 2139.30 | 2138.65 | 2121.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 2175.00 | 2133.84 | 2122.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 2275.70 | 2279.00 | 2236.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-12 14:15:00 | 2170.65 | 2234.48 | 2240.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 2170.65 | 2234.48 | 2240.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 2148.05 | 2195.17 | 2214.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 2153.75 | 2142.35 | 2169.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 2153.75 | 2142.35 | 2169.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 2153.75 | 2142.35 | 2169.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 2104.00 | 2134.07 | 2150.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:30:00 | 2101.50 | 2112.04 | 2137.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 11:15:00 | 2101.60 | 2081.59 | 2102.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 2176.50 | 2118.11 | 2110.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 2176.50 | 2118.11 | 2110.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 12:15:00 | 2194.80 | 2167.86 | 2148.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 2171.95 | 2183.09 | 2164.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 10:45:00 | 2173.50 | 2183.09 | 2164.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 2175.70 | 2181.34 | 2171.69 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 2152.55 | 2172.55 | 2172.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 12:15:00 | 2145.90 | 2167.22 | 2170.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 2165.90 | 2158.17 | 2164.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 2165.90 | 2158.17 | 2164.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 2165.90 | 2158.17 | 2164.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 2168.00 | 2158.17 | 2164.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 2149.00 | 2156.34 | 2162.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 12:15:00 | 2145.60 | 2154.83 | 2161.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 2142.65 | 2151.43 | 2157.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 2177.15 | 2137.18 | 2142.39 | SL hit (close>static) qty=1.00 sl=2167.60 alert=retest2 |

### Cycle 106 — BUY (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 12:15:00 | 2160.00 | 2146.12 | 2145.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 13:15:00 | 2170.00 | 2150.90 | 2148.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 2161.25 | 2167.70 | 2157.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 10:00:00 | 2161.25 | 2167.70 | 2157.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 2182.95 | 2170.75 | 2159.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 11:45:00 | 2220.55 | 2192.00 | 2170.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 2210.10 | 2250.53 | 2252.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 2210.10 | 2250.53 | 2252.03 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 2348.00 | 2252.29 | 2241.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 14:15:00 | 2410.50 | 2328.82 | 2295.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 2385.00 | 2398.78 | 2369.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 12:45:00 | 2391.55 | 2398.78 | 2369.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 2354.00 | 2389.82 | 2367.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:45:00 | 2361.45 | 2389.82 | 2367.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 2321.10 | 2376.08 | 2363.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 2321.10 | 2376.08 | 2363.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 2325.95 | 2366.05 | 2360.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 2270.40 | 2366.05 | 2360.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 2288.70 | 2350.58 | 2353.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 09:15:00 | 2228.00 | 2317.24 | 2335.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 10:15:00 | 2260.00 | 2258.69 | 2287.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 10:30:00 | 2262.60 | 2258.69 | 2287.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 2265.60 | 2256.89 | 2268.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 2263.50 | 2256.89 | 2268.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 2259.95 | 2257.51 | 2267.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 2259.95 | 2257.51 | 2267.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 2259.95 | 2257.99 | 2266.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 2251.65 | 2257.99 | 2266.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 2268.50 | 2260.10 | 2266.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 2268.50 | 2260.10 | 2266.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 2260.75 | 2260.23 | 2266.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:45:00 | 2258.30 | 2260.23 | 2266.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 2236.00 | 2255.38 | 2263.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 2228.00 | 2255.38 | 2263.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 12:00:00 | 2229.30 | 2203.22 | 2205.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 2224.00 | 2207.38 | 2206.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 2224.00 | 2207.38 | 2206.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 2247.95 | 2219.27 | 2212.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 2268.00 | 2279.95 | 2258.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 2268.00 | 2279.95 | 2258.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 2263.60 | 2274.60 | 2259.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:45:00 | 2280.80 | 2274.60 | 2259.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 2212.00 | 2262.08 | 2255.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:00:00 | 2212.00 | 2262.08 | 2255.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 2195.65 | 2248.79 | 2250.14 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 2256.00 | 2249.49 | 2248.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 14:15:00 | 2287.80 | 2257.16 | 2252.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 2252.50 | 2263.85 | 2258.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 11:15:00 | 2252.50 | 2263.85 | 2258.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 2252.50 | 2263.85 | 2258.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:00:00 | 2252.50 | 2263.85 | 2258.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 2253.45 | 2261.77 | 2257.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:00:00 | 2253.45 | 2261.77 | 2257.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 2264.95 | 2262.41 | 2258.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:30:00 | 2256.10 | 2262.41 | 2258.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 2251.15 | 2260.16 | 2257.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 2251.15 | 2260.16 | 2257.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 2245.15 | 2257.15 | 2256.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 2244.35 | 2257.15 | 2256.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 2253.90 | 2257.59 | 2256.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 2253.90 | 2257.59 | 2256.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 2258.95 | 2257.86 | 2257.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:15:00 | 2252.80 | 2257.86 | 2257.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 12:15:00 | 2251.35 | 2256.56 | 2256.58 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 13:15:00 | 2258.00 | 2256.85 | 2256.71 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 2241.55 | 2254.76 | 2255.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 2195.20 | 2242.85 | 2250.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 13:15:00 | 2126.95 | 2091.23 | 2130.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 14:00:00 | 2126.95 | 2091.23 | 2130.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 2114.15 | 2095.81 | 2128.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:30:00 | 2107.00 | 2109.49 | 2129.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 11:15:00 | 2166.25 | 2127.32 | 2134.73 | SL hit (close>static) qty=1.00 sl=2159.95 alert=retest2 |

### Cycle 116 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 2168.80 | 2139.09 | 2136.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 12:15:00 | 2187.85 | 2155.85 | 2145.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 2141.55 | 2166.44 | 2155.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 2141.55 | 2166.44 | 2155.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 2141.55 | 2166.44 | 2155.29 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 2126.30 | 2148.89 | 2149.22 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 2201.00 | 2152.10 | 2149.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 15:15:00 | 2224.00 | 2195.90 | 2174.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 13:15:00 | 2206.85 | 2207.90 | 2189.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 13:45:00 | 2208.70 | 2207.90 | 2189.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 2182.60 | 2202.84 | 2189.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 2182.60 | 2202.84 | 2189.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 2197.95 | 2201.86 | 2189.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 2136.20 | 2201.86 | 2189.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 2152.10 | 2191.91 | 2186.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 2130.00 | 2191.91 | 2186.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 2147.85 | 2183.10 | 2183.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:15:00 | 2138.00 | 2183.10 | 2183.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 2130.00 | 2172.48 | 2178.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 12:15:00 | 2081.90 | 2146.33 | 2160.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 14:15:00 | 1938.30 | 1898.13 | 1959.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 15:00:00 | 1938.30 | 1898.13 | 1959.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 2001.80 | 1920.28 | 1958.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 2001.80 | 1920.28 | 1958.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 2009.60 | 1938.15 | 1963.28 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 2062.90 | 1990.84 | 1983.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 2096.25 | 2011.92 | 1993.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 2039.00 | 2041.36 | 2016.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 2039.00 | 2041.36 | 2016.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 2005.95 | 2034.28 | 2015.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 2005.95 | 2034.28 | 2015.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1965.05 | 2020.43 | 2010.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 1965.05 | 2020.43 | 2010.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 15:15:00 | 1985.00 | 2004.81 | 2004.83 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 2048.00 | 2012.69 | 2008.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 2050.55 | 2020.26 | 2012.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 2017.80 | 2019.77 | 2012.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 2017.80 | 2019.77 | 2012.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 2054.05 | 2026.63 | 2016.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:30:00 | 2038.15 | 2026.63 | 2016.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2037.75 | 2040.89 | 2027.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 2031.10 | 2040.89 | 2027.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 2022.20 | 2037.15 | 2026.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 2022.20 | 2037.15 | 2026.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 2067.70 | 2043.26 | 2030.52 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1943.40 | 2020.01 | 2022.92 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 2023.30 | 1994.22 | 1993.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 2070.25 | 2009.43 | 2000.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 15:15:00 | 2026.30 | 2027.10 | 2014.67 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:15:00 | 2050.35 | 2027.10 | 2014.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 14:15:00 | 2042.00 | 2041.58 | 2028.26 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2024.80 | 2040.54 | 2031.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 2024.80 | 2040.54 | 2031.31 | SL hit (close<ema400) qty=1.00 sl=2031.31 alert=retest1 |

### Cycle 125 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 2014.85 | 2025.55 | 2026.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 15:15:00 | 2004.40 | 2019.63 | 2023.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 1841.90 | 1838.66 | 1875.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 1785.00 | 1838.66 | 1875.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1705.00 | 1657.11 | 1678.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1705.00 | 1657.11 | 1678.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1713.15 | 1668.31 | 1681.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:45:00 | 1712.80 | 1668.31 | 1681.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 1730.70 | 1691.76 | 1690.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 12:15:00 | 1766.75 | 1740.47 | 1728.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 15:15:00 | 1728.05 | 1742.25 | 1732.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 15:15:00 | 1728.05 | 1742.25 | 1732.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 1728.05 | 1742.25 | 1732.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 1752.65 | 1742.25 | 1732.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1768.10 | 1747.42 | 1736.09 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 1701.95 | 1730.14 | 1731.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 1658.35 | 1711.46 | 1722.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 1538.85 | 1537.13 | 1596.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 1538.85 | 1537.13 | 1596.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1605.80 | 1549.21 | 1586.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 1611.95 | 1549.21 | 1586.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1618.30 | 1563.03 | 1589.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:45:00 | 1616.60 | 1563.03 | 1589.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 1604.05 | 1580.25 | 1591.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 1604.05 | 1580.25 | 1591.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 1603.90 | 1584.98 | 1592.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 1597.00 | 1584.98 | 1592.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1637.00 | 1597.31 | 1597.32 | SL hit (close>static) qty=1.00 sl=1611.90 alert=retest2 |

### Cycle 128 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1599.75 | 1597.80 | 1597.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1712.00 | 1636.31 | 1617.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1696.30 | 1699.94 | 1669.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 1696.30 | 1699.94 | 1669.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1690.35 | 1713.55 | 1699.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1678.00 | 1713.55 | 1699.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1669.70 | 1704.78 | 1696.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 1670.00 | 1704.78 | 1696.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 1660.50 | 1687.40 | 1690.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 14:15:00 | 1653.05 | 1676.47 | 1684.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1635.70 | 1614.61 | 1629.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1635.70 | 1614.61 | 1629.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1635.70 | 1614.61 | 1629.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 1634.75 | 1614.61 | 1629.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1627.55 | 1617.20 | 1629.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:15:00 | 1597.90 | 1617.56 | 1627.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 1598.70 | 1614.84 | 1625.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 1660.00 | 1627.37 | 1625.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 1660.00 | 1627.37 | 1625.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1687.40 | 1647.53 | 1635.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 12:15:00 | 1787.40 | 1788.45 | 1760.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 12:45:00 | 1786.15 | 1788.45 | 1760.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1756.75 | 1779.03 | 1764.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1756.75 | 1779.03 | 1764.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1753.15 | 1773.85 | 1763.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 1753.15 | 1773.85 | 1763.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1780.05 | 1765.85 | 1761.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:30:00 | 1769.25 | 1765.85 | 1761.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1745.00 | 1764.02 | 1761.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 1744.30 | 1764.02 | 1761.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 1731.20 | 1757.46 | 1759.07 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 15:15:00 | 1777.75 | 1761.62 | 1759.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 09:15:00 | 1801.25 | 1769.55 | 1763.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 1835.35 | 1853.05 | 1823.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 13:00:00 | 1835.35 | 1853.05 | 1823.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1863.05 | 1855.05 | 1827.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:45:00 | 1829.20 | 1855.05 | 1827.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 1835.00 | 1851.04 | 1827.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 1835.00 | 1851.04 | 1827.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 1821.00 | 1845.03 | 1827.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 1826.55 | 1845.03 | 1827.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1817.55 | 1839.53 | 1826.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:15:00 | 1800.65 | 1839.53 | 1826.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1781.00 | 1827.83 | 1822.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 1781.00 | 1827.83 | 1822.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1788.65 | 1819.99 | 1819.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 1788.15 | 1819.99 | 1819.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 12:15:00 | 1787.50 | 1813.49 | 1816.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 13:15:00 | 1780.00 | 1806.79 | 1812.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 1785.70 | 1785.63 | 1798.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:30:00 | 1782.10 | 1785.63 | 1798.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1763.85 | 1774.19 | 1787.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 1746.00 | 1778.67 | 1784.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1571.40 | 1699.34 | 1736.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 1748.40 | 1650.37 | 1641.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 1795.80 | 1679.46 | 1655.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1907.50 | 1916.36 | 1872.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1907.50 | 1916.36 | 1872.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 1977.30 | 1973.33 | 1956.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:45:00 | 1959.50 | 1973.33 | 1956.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1939.20 | 1965.56 | 1957.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1939.20 | 1965.56 | 1957.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1936.10 | 1959.67 | 1955.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1929.10 | 1959.67 | 1955.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 1957.70 | 1958.29 | 1955.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:00:00 | 1989.80 | 1965.10 | 1959.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 1949.00 | 1959.15 | 1958.24 | SL hit (close<static) qty=1.00 sl=1951.70 alert=retest2 |

### Cycle 135 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 1945.00 | 1955.67 | 1956.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 1870.40 | 1938.62 | 1948.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1806.30 | 1784.12 | 1814.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 1806.30 | 1784.12 | 1814.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1806.30 | 1784.12 | 1814.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 1806.30 | 1784.12 | 1814.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1785.90 | 1784.48 | 1811.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 1800.50 | 1784.48 | 1811.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 1818.00 | 1792.41 | 1808.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:45:00 | 1813.40 | 1792.41 | 1808.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 1810.20 | 1795.97 | 1808.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:45:00 | 1816.00 | 1795.97 | 1808.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1812.00 | 1799.18 | 1809.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 1972.90 | 1799.18 | 1809.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1961.40 | 1831.62 | 1822.90 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 1842.00 | 1873.31 | 1873.45 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 1948.50 | 1886.42 | 1879.03 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 1842.00 | 1882.22 | 1883.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 1818.00 | 1869.38 | 1877.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 1816.50 | 1814.16 | 1837.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 1901.90 | 1814.16 | 1837.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1898.00 | 1830.93 | 1843.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 1892.20 | 1830.93 | 1843.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1888.00 | 1852.31 | 1851.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1901.10 | 1874.38 | 1862.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1880.40 | 1884.45 | 1873.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 1880.40 | 1884.45 | 1873.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1882.80 | 1884.12 | 1874.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 1873.00 | 1884.12 | 1874.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1913.80 | 1915.43 | 1903.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 1913.80 | 1915.43 | 1903.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2077.20 | 2084.10 | 2036.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 2058.00 | 2084.10 | 2036.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2058.70 | 2084.00 | 2056.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 2058.70 | 2084.00 | 2056.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 2051.00 | 2077.40 | 2055.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 2081.00 | 2077.40 | 2055.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2092.60 | 2080.44 | 2059.22 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 2045.00 | 2058.61 | 2058.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 1978.80 | 2040.74 | 2050.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 1956.50 | 1955.08 | 1982.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 14:30:00 | 1953.00 | 1955.08 | 1982.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1926.90 | 1923.62 | 1942.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 1932.30 | 1923.62 | 1942.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1895.00 | 1915.38 | 1932.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:15:00 | 1886.30 | 1910.59 | 1928.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 11:45:00 | 1885.50 | 1906.15 | 1924.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:30:00 | 1885.30 | 1899.70 | 1918.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 14:15:00 | 1886.20 | 1899.70 | 1918.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1858.40 | 1846.30 | 1863.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 1858.40 | 1846.30 | 1863.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1886.40 | 1857.03 | 1864.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:45:00 | 1862.20 | 1857.83 | 1863.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:45:00 | 1859.70 | 1858.60 | 1863.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1857.50 | 1856.32 | 1860.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 1868.50 | 1852.11 | 1851.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 1868.50 | 1852.11 | 1851.10 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 10:15:00 | 1850.00 | 1857.00 | 1857.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 11:15:00 | 1844.10 | 1854.42 | 1856.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1685.20 | 1683.43 | 1703.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1685.20 | 1683.43 | 1703.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1685.20 | 1683.43 | 1703.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1691.90 | 1683.43 | 1703.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1667.90 | 1664.52 | 1679.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1669.50 | 1664.52 | 1679.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1682.00 | 1668.65 | 1676.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1665.20 | 1668.65 | 1676.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1687.60 | 1674.53 | 1678.29 | SL hit (close>static) qty=1.00 sl=1685.40 alert=retest2 |

### Cycle 144 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 1685.00 | 1680.90 | 1680.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 15:15:00 | 1687.90 | 1682.96 | 1681.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 1859.00 | 1860.34 | 1831.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 1859.00 | 1860.34 | 1831.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1857.20 | 1859.67 | 1847.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 1859.40 | 1853.71 | 1847.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 1872.30 | 1856.37 | 1849.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:45:00 | 1859.40 | 1857.16 | 1851.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 11:00:00 | 1859.90 | 1857.59 | 1853.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1854.80 | 1856.56 | 1853.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 1853.60 | 1856.56 | 1853.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1845.10 | 1854.26 | 1853.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 1851.00 | 1854.26 | 1853.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1845.00 | 1852.41 | 1852.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1851.30 | 1852.41 | 1852.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:00:00 | 1851.90 | 1853.94 | 1853.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 1849.40 | 1852.72 | 1852.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 1849.40 | 1852.72 | 1852.76 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1855.70 | 1853.32 | 1853.03 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 1844.00 | 1851.14 | 1852.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 1833.90 | 1847.69 | 1850.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1810.60 | 1809.29 | 1822.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:30:00 | 1812.00 | 1809.29 | 1822.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1780.00 | 1796.31 | 1808.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 1777.10 | 1796.31 | 1808.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 1773.50 | 1791.81 | 1805.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 1768.00 | 1779.72 | 1793.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 1729.10 | 1720.83 | 1720.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1729.10 | 1720.83 | 1720.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 09:15:00 | 1809.00 | 1741.15 | 1730.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 1841.10 | 1850.46 | 1817.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 12:15:00 | 1816.00 | 1836.59 | 1818.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 1816.00 | 1836.59 | 1818.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 1816.00 | 1836.59 | 1818.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1807.00 | 1830.67 | 1817.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:30:00 | 1811.40 | 1830.67 | 1817.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1817.00 | 1827.94 | 1817.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 1817.00 | 1827.94 | 1817.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1823.70 | 1827.09 | 1817.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 1832.00 | 1827.09 | 1817.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 1834.50 | 1861.50 | 1863.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1834.50 | 1861.50 | 1863.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1803.00 | 1849.80 | 1858.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 1780.40 | 1778.87 | 1792.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 13:15:00 | 1782.70 | 1779.64 | 1791.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1782.70 | 1779.64 | 1791.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 1782.70 | 1779.64 | 1791.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1788.20 | 1781.35 | 1791.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 1788.20 | 1781.35 | 1791.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1788.30 | 1782.74 | 1791.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 1802.00 | 1782.74 | 1791.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1796.90 | 1785.57 | 1791.83 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 1837.50 | 1796.26 | 1794.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 1849.00 | 1806.81 | 1799.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 10:15:00 | 1799.10 | 1805.49 | 1799.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 10:15:00 | 1799.10 | 1805.49 | 1799.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1799.10 | 1805.49 | 1799.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 1795.90 | 1805.49 | 1799.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1798.80 | 1804.15 | 1799.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:45:00 | 1800.50 | 1804.15 | 1799.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1799.30 | 1803.18 | 1799.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:30:00 | 1798.10 | 1803.18 | 1799.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 1795.50 | 1801.65 | 1799.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:45:00 | 1794.30 | 1801.65 | 1799.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 1772.80 | 1795.88 | 1796.96 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 1800.00 | 1796.88 | 1796.55 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 1786.20 | 1795.30 | 1795.97 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 1806.10 | 1793.77 | 1792.82 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 1786.50 | 1791.59 | 1791.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 1779.50 | 1789.17 | 1790.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 12:15:00 | 1789.40 | 1786.92 | 1789.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 12:15:00 | 1789.40 | 1786.92 | 1789.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1789.40 | 1786.92 | 1789.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:30:00 | 1788.80 | 1786.92 | 1789.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1793.80 | 1788.30 | 1789.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 1793.80 | 1788.30 | 1789.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1784.20 | 1787.48 | 1789.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 15:15:00 | 1780.30 | 1787.48 | 1789.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1750.80 | 1738.51 | 1737.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 1750.80 | 1738.51 | 1737.55 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1703.00 | 1735.39 | 1737.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1679.30 | 1703.34 | 1719.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1654.00 | 1639.90 | 1659.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1654.00 | 1639.90 | 1659.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1654.00 | 1639.90 | 1659.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:15:00 | 1659.70 | 1639.90 | 1659.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1656.90 | 1643.30 | 1659.70 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1698.50 | 1668.48 | 1665.31 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1658.10 | 1680.07 | 1681.09 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 1697.50 | 1682.73 | 1681.38 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 1677.90 | 1686.65 | 1686.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 1659.40 | 1667.40 | 1673.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 1665.90 | 1661.88 | 1667.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 1680.80 | 1661.88 | 1667.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1675.30 | 1664.56 | 1668.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 1683.50 | 1664.56 | 1668.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1675.00 | 1666.65 | 1668.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 1669.00 | 1668.24 | 1669.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:30:00 | 1671.50 | 1664.24 | 1665.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 1669.80 | 1665.35 | 1666.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:00:00 | 1670.40 | 1665.93 | 1666.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1663.80 | 1665.51 | 1665.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 1657.10 | 1665.51 | 1665.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 1666.10 | 1663.98 | 1663.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 13:15:00 | 1666.10 | 1663.98 | 1663.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1688.60 | 1669.84 | 1666.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 1674.00 | 1675.69 | 1670.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 1674.00 | 1675.69 | 1670.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1665.60 | 1674.81 | 1671.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1665.60 | 1674.81 | 1671.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1670.00 | 1673.85 | 1671.70 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 1662.60 | 1669.64 | 1670.04 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 1683.00 | 1671.62 | 1670.38 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1655.90 | 1669.04 | 1670.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 1651.10 | 1665.45 | 1668.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1663.00 | 1662.20 | 1665.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1663.00 | 1662.20 | 1665.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1663.00 | 1662.20 | 1665.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1663.00 | 1662.20 | 1665.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1654.10 | 1660.58 | 1664.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 1653.30 | 1656.30 | 1661.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1570.63 | 1606.98 | 1628.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1584.00 | 1575.28 | 1597.88 | SL hit (close>ema200) qty=0.50 sl=1575.28 alert=retest2 |

### Cycle 166 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1580.00 | 1558.31 | 1557.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 1585.90 | 1563.83 | 1560.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1613.60 | 1625.14 | 1606.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:00:00 | 1613.60 | 1625.14 | 1606.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1604.70 | 1621.05 | 1606.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 1605.10 | 1621.05 | 1606.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1604.40 | 1617.72 | 1606.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:30:00 | 1602.80 | 1617.72 | 1606.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 1614.10 | 1617.00 | 1606.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 13:15:00 | 1619.80 | 1617.00 | 1606.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:15:00 | 1619.70 | 1616.07 | 1608.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1599.00 | 1612.79 | 1608.76 | SL hit (close<static) qty=1.00 sl=1601.50 alert=retest2 |

### Cycle 167 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1590.80 | 1606.04 | 1606.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 1586.80 | 1599.16 | 1602.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 14:15:00 | 1600.00 | 1589.29 | 1594.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 1600.00 | 1589.29 | 1594.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1600.00 | 1589.29 | 1594.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1600.00 | 1589.29 | 1594.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1605.00 | 1592.43 | 1595.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 1614.60 | 1592.43 | 1595.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1594.00 | 1593.21 | 1595.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 1592.90 | 1593.21 | 1595.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1574.00 | 1589.37 | 1593.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:45:00 | 1573.20 | 1584.48 | 1590.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 1572.30 | 1579.56 | 1585.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 1601.50 | 1568.70 | 1571.32 | SL hit (close>static) qty=1.00 sl=1594.30 alert=retest2 |

### Cycle 168 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1595.70 | 1574.10 | 1573.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 1628.40 | 1587.65 | 1579.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1593.20 | 1600.76 | 1589.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1593.20 | 1600.76 | 1589.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1681.00 | 1615.09 | 1598.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:15:00 | 1698.00 | 1629.09 | 1606.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 1698.00 | 1642.87 | 1614.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1630.00 | 1649.34 | 1651.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1630.00 | 1649.34 | 1651.60 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 1668.40 | 1651.36 | 1649.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 1677.80 | 1656.65 | 1652.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 1688.00 | 1691.77 | 1673.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 12:45:00 | 1686.00 | 1691.77 | 1673.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1667.60 | 1686.65 | 1674.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:45:00 | 1663.30 | 1686.65 | 1674.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1670.00 | 1683.32 | 1673.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1728.00 | 1683.32 | 1673.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 1702.60 | 1735.82 | 1737.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 1702.60 | 1735.82 | 1737.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 1700.90 | 1728.84 | 1733.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1702.80 | 1701.63 | 1714.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 1700.70 | 1701.63 | 1714.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1711.80 | 1703.67 | 1714.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1691.10 | 1705.13 | 1713.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1746.50 | 1711.31 | 1715.07 | SL hit (close>static) qty=1.00 sl=1727.70 alert=retest2 |

### Cycle 172 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1746.50 | 1718.34 | 1717.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1767.30 | 1738.63 | 1731.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 1747.00 | 1749.45 | 1740.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 14:00:00 | 1747.00 | 1749.45 | 1740.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1739.10 | 1747.38 | 1739.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 1739.10 | 1747.38 | 1739.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 1736.30 | 1745.16 | 1739.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1752.10 | 1745.16 | 1739.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1731.00 | 1741.76 | 1738.98 | SL hit (close<static) qty=1.00 sl=1733.70 alert=retest2 |

### Cycle 173 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 1728.90 | 1737.04 | 1737.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 1711.50 | 1727.69 | 1732.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 09:15:00 | 1729.80 | 1728.11 | 1732.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1729.80 | 1728.11 | 1732.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1729.80 | 1728.11 | 1732.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:30:00 | 1731.40 | 1728.11 | 1732.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1736.00 | 1729.69 | 1732.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 1737.30 | 1729.69 | 1732.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1752.00 | 1734.15 | 1734.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 1757.10 | 1734.15 | 1734.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 1749.00 | 1737.12 | 1735.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 1758.00 | 1750.77 | 1744.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1728.60 | 1746.34 | 1743.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1728.60 | 1746.34 | 1743.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1728.60 | 1746.34 | 1743.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1728.60 | 1746.34 | 1743.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1742.70 | 1745.61 | 1743.18 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 1736.20 | 1740.77 | 1741.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 1725.30 | 1737.07 | 1739.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 1714.80 | 1711.70 | 1719.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:30:00 | 1715.00 | 1711.70 | 1719.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1707.00 | 1705.25 | 1711.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 1706.50 | 1705.25 | 1711.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1705.90 | 1705.38 | 1711.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1690.40 | 1704.55 | 1709.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1715.60 | 1706.76 | 1710.20 | SL hit (close>static) qty=1.00 sl=1714.90 alert=retest2 |

### Cycle 176 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 1738.90 | 1713.12 | 1712.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 13:15:00 | 1783.90 | 1727.27 | 1718.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 1845.00 | 1847.86 | 1818.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:30:00 | 1846.80 | 1847.15 | 1821.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1825.80 | 1839.48 | 1827.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1825.80 | 1839.48 | 1827.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1824.90 | 1836.56 | 1826.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 1824.00 | 1836.56 | 1826.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1820.40 | 1832.39 | 1826.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 1834.70 | 1831.97 | 1826.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 1832.20 | 1832.02 | 1827.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 1792.90 | 1820.51 | 1823.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 1792.90 | 1820.51 | 1823.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 1782.30 | 1812.87 | 1819.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1795.20 | 1791.26 | 1802.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1795.20 | 1791.26 | 1802.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1795.20 | 1791.26 | 1802.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 1795.00 | 1791.26 | 1802.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1776.80 | 1781.84 | 1792.39 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 1814.10 | 1791.00 | 1790.91 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 1785.00 | 1795.70 | 1797.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1768.30 | 1790.22 | 1794.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1813.50 | 1794.88 | 1796.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 1813.50 | 1794.88 | 1796.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1813.50 | 1794.88 | 1796.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 1813.50 | 1794.88 | 1796.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 1814.10 | 1798.72 | 1797.85 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1796.00 | 1801.16 | 1801.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1791.20 | 1799.17 | 1800.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1794.90 | 1784.54 | 1790.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1794.90 | 1784.54 | 1790.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1794.90 | 1784.54 | 1790.58 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 1819.50 | 1796.63 | 1795.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 1841.20 | 1809.38 | 1801.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1854.60 | 1856.37 | 1838.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:15:00 | 1851.40 | 1856.37 | 1838.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1825.60 | 1850.22 | 1837.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1825.60 | 1850.22 | 1837.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1831.80 | 1846.53 | 1836.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:30:00 | 1834.90 | 1840.55 | 1835.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 1841.30 | 1841.04 | 1836.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:45:00 | 1841.50 | 1838.70 | 1836.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:00:00 | 1837.10 | 1838.70 | 1836.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1838.00 | 1838.56 | 1837.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 1835.60 | 1838.56 | 1837.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1835.00 | 1837.85 | 1836.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1818.10 | 1837.85 | 1836.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 1801.30 | 1830.54 | 1833.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1801.30 | 1830.54 | 1833.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 10:15:00 | 1792.50 | 1812.18 | 1820.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 1808.90 | 1806.23 | 1814.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 1808.90 | 1806.23 | 1814.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1811.00 | 1807.19 | 1814.32 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1846.90 | 1821.64 | 1819.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 13:15:00 | 1854.60 | 1844.08 | 1835.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 1843.30 | 1843.93 | 1836.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 1843.30 | 1843.93 | 1836.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1839.60 | 1842.45 | 1836.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 1839.60 | 1842.45 | 1836.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1840.60 | 1842.08 | 1837.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 1841.40 | 1842.08 | 1837.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1841.50 | 1841.96 | 1837.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 1834.10 | 1841.96 | 1837.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1863.00 | 1870.19 | 1861.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1872.90 | 1870.19 | 1861.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1843.70 | 1864.89 | 1859.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:00:00 | 1843.70 | 1864.89 | 1859.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 1823.60 | 1856.64 | 1856.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 1823.60 | 1856.64 | 1856.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1851.10 | 1856.40 | 1856.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1838.70 | 1851.84 | 1854.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 1861.00 | 1853.67 | 1854.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 1861.00 | 1853.67 | 1854.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1861.00 | 1853.67 | 1854.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 1861.00 | 1853.67 | 1854.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 1850.10 | 1852.96 | 1854.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:15:00 | 1863.10 | 1852.96 | 1854.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 12:15:00 | 1874.60 | 1857.28 | 1856.33 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 11:15:00 | 1850.40 | 1856.93 | 1857.27 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 1870.00 | 1857.95 | 1857.42 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1846.00 | 1855.56 | 1856.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 11:15:00 | 1842.30 | 1851.76 | 1854.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 1851.80 | 1849.42 | 1852.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 14:15:00 | 1851.80 | 1849.42 | 1852.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1851.80 | 1849.42 | 1852.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1851.80 | 1849.42 | 1852.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1865.00 | 1852.54 | 1853.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1851.00 | 1852.54 | 1853.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1856.10 | 1853.25 | 1853.87 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1860.10 | 1854.62 | 1854.43 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 1843.90 | 1854.67 | 1854.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 10:15:00 | 1840.00 | 1851.74 | 1853.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 1644.00 | 1632.86 | 1656.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 1634.10 | 1632.86 | 1656.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1588.00 | 1583.95 | 1598.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:45:00 | 1574.30 | 1580.68 | 1595.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:45:00 | 1571.50 | 1554.05 | 1562.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:00:00 | 1574.00 | 1536.58 | 1546.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 1495.58 | 1515.87 | 1530.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 1492.92 | 1515.87 | 1530.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 12:15:00 | 1495.30 | 1515.87 | 1530.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1505.20 | 1495.08 | 1514.04 | SL hit (close>ema200) qty=0.50 sl=1495.08 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 1557.60 | 1526.86 | 1524.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 1564.90 | 1542.52 | 1533.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1602.00 | 1606.04 | 1586.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:45:00 | 1592.20 | 1606.04 | 1586.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1605.60 | 1607.38 | 1590.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:45:00 | 1589.60 | 1607.38 | 1590.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1555.00 | 1596.91 | 1587.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1559.30 | 1596.91 | 1587.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1558.00 | 1589.13 | 1584.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 1566.90 | 1589.13 | 1584.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1571.00 | 1580.98 | 1581.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1533.50 | 1571.49 | 1576.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1626.40 | 1572.26 | 1573.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 1626.40 | 1572.26 | 1573.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1626.40 | 1572.26 | 1573.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1626.40 | 1572.26 | 1573.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1621.70 | 1582.15 | 1578.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 1674.90 | 1607.88 | 1590.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 1660.00 | 1661.77 | 1649.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 1635.00 | 1661.77 | 1649.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1634.20 | 1656.25 | 1648.48 | EMA400 retest candle locked (from upside) |

### Cycle 195 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 1635.10 | 1644.30 | 1644.60 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 1648.40 | 1645.12 | 1644.94 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 1635.40 | 1643.18 | 1644.07 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1658.30 | 1646.20 | 1645.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1719.20 | 1675.42 | 1661.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1681.90 | 1686.93 | 1672.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1681.90 | 1686.93 | 1672.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1678.10 | 1689.18 | 1677.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1674.00 | 1689.18 | 1677.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1690.00 | 1689.34 | 1678.86 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 1672.40 | 1679.48 | 1679.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 1669.70 | 1677.52 | 1678.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 1670.00 | 1666.75 | 1671.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 13:15:00 | 1670.00 | 1666.75 | 1671.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 1670.00 | 1666.75 | 1671.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:45:00 | 1669.00 | 1666.75 | 1671.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1651.80 | 1663.76 | 1669.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:45:00 | 1662.40 | 1663.76 | 1669.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1671.60 | 1654.18 | 1660.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1671.60 | 1654.18 | 1660.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1666.00 | 1656.54 | 1660.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1646.40 | 1656.54 | 1660.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1642.00 | 1599.91 | 1595.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1642.00 | 1599.91 | 1595.54 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1581.50 | 1603.71 | 1604.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 1515.50 | 1529.20 | 1546.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1535.70 | 1530.50 | 1545.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 1535.70 | 1530.50 | 1545.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1544.00 | 1535.20 | 1545.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 1545.00 | 1535.20 | 1545.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1526.00 | 1533.36 | 1543.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:30:00 | 1546.00 | 1533.36 | 1543.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1542.00 | 1535.98 | 1542.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 1537.00 | 1536.98 | 1542.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 1460.15 | 1491.26 | 1499.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-16 13:15:00 | 1383.30 | 1413.55 | 1443.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 202 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1450.00 | 1431.38 | 1430.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1454.10 | 1435.92 | 1433.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1426.70 | 1436.32 | 1434.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1426.70 | 1436.32 | 1434.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1426.70 | 1436.32 | 1434.16 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1414.10 | 1431.88 | 1432.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1410.90 | 1423.15 | 1427.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1342.00 | 1334.70 | 1364.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:45:00 | 1347.80 | 1334.70 | 1364.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1363.10 | 1341.77 | 1362.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 1361.30 | 1341.77 | 1362.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1398.50 | 1353.12 | 1366.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1392.30 | 1353.12 | 1366.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1398.70 | 1362.23 | 1369.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1398.70 | 1362.23 | 1369.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1446.40 | 1387.06 | 1379.36 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1367.60 | 1401.95 | 1402.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 1352.30 | 1379.87 | 1391.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 15:15:00 | 1305.00 | 1301.01 | 1320.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 09:15:00 | 1322.10 | 1301.01 | 1320.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1312.70 | 1303.35 | 1320.05 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1370.40 | 1331.85 | 1330.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1378.70 | 1346.86 | 1337.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1523.20 | 1539.42 | 1510.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1523.20 | 1539.42 | 1510.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1523.20 | 1539.42 | 1510.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1530.90 | 1539.42 | 1510.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 09:15:00 | 1683.99 | 1652.26 | 1632.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 1633.60 | 1648.78 | 1648.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 1629.00 | 1641.57 | 1645.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1587.90 | 1577.91 | 1593.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1587.90 | 1577.91 | 1593.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1587.90 | 1577.91 | 1593.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 1596.20 | 1577.91 | 1593.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1610.90 | 1585.89 | 1593.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 1610.90 | 1585.89 | 1593.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1604.80 | 1589.67 | 1594.27 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1614.90 | 1598.61 | 1597.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1622.00 | 1604.26 | 1600.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1605.40 | 1636.75 | 1627.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 1605.40 | 1636.75 | 1627.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1605.40 | 1636.75 | 1627.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1605.40 | 1636.75 | 1627.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1606.20 | 1630.64 | 1625.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 1602.30 | 1630.64 | 1625.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1624.40 | 1627.26 | 1624.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 1624.40 | 1627.26 | 1624.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1626.90 | 1627.19 | 1625.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1690.30 | 1627.19 | 1625.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-23 13:15:00 | 562.70 | 2023-05-26 12:15:00 | 558.05 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2023-06-09 10:00:00 | 596.30 | 2023-06-14 10:15:00 | 608.90 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2023-06-09 13:45:00 | 596.85 | 2023-06-14 10:15:00 | 608.90 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2023-06-09 14:15:00 | 596.00 | 2023-06-14 10:15:00 | 608.90 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2023-06-12 09:15:00 | 588.00 | 2023-06-14 10:15:00 | 608.90 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2023-07-03 10:00:00 | 601.90 | 2023-07-04 13:15:00 | 616.20 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2023-07-03 11:00:00 | 605.10 | 2023-07-04 13:15:00 | 616.20 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2023-07-03 13:00:00 | 604.20 | 2023-07-04 13:15:00 | 616.20 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2023-07-04 09:45:00 | 603.80 | 2023-07-04 13:15:00 | 616.20 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2023-07-17 10:00:00 | 642.55 | 2023-07-24 15:15:00 | 637.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-08-01 15:15:00 | 678.80 | 2023-08-02 13:15:00 | 668.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-08-02 15:00:00 | 677.60 | 2023-08-14 14:15:00 | 745.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-03 10:00:00 | 679.75 | 2023-08-14 14:15:00 | 747.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-28 09:15:00 | 750.00 | 2023-09-07 09:15:00 | 780.40 | STOP_HIT | 1.00 | 4.05% |
| BUY | retest2 | 2023-10-20 09:15:00 | 1000.70 | 2023-10-23 09:15:00 | 972.80 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2023-10-20 10:30:00 | 1001.00 | 2023-10-23 09:15:00 | 972.80 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2023-11-03 10:00:00 | 1069.00 | 2023-11-10 10:15:00 | 1111.30 | STOP_HIT | 1.00 | 3.96% |
| BUY | retest2 | 2023-11-15 12:15:00 | 1146.10 | 2023-11-16 09:15:00 | 1125.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2023-11-21 11:30:00 | 1128.95 | 2023-11-23 12:15:00 | 1072.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-21 12:30:00 | 1127.85 | 2023-11-23 12:15:00 | 1071.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-21 13:00:00 | 1129.70 | 2023-11-23 12:15:00 | 1073.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-21 14:00:00 | 1125.30 | 2023-11-23 15:15:00 | 1069.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-21 11:30:00 | 1128.95 | 2023-11-24 09:15:00 | 1099.95 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2023-11-21 12:30:00 | 1127.85 | 2023-11-24 09:15:00 | 1099.95 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2023-11-21 13:00:00 | 1129.70 | 2023-11-24 09:15:00 | 1099.95 | STOP_HIT | 0.50 | 2.63% |
| SELL | retest2 | 2023-11-21 14:00:00 | 1125.30 | 2023-11-24 09:15:00 | 1099.95 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2023-11-22 09:45:00 | 1112.95 | 2023-11-24 15:15:00 | 1128.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-11-22 12:15:00 | 1111.50 | 2023-11-24 15:15:00 | 1128.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-12-08 09:45:00 | 1072.65 | 2023-12-18 09:15:00 | 1073.80 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2023-12-08 14:45:00 | 1073.45 | 2023-12-18 09:15:00 | 1073.80 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2023-12-11 09:30:00 | 1068.40 | 2023-12-18 09:15:00 | 1073.80 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2023-12-22 12:30:00 | 1038.50 | 2023-12-26 14:15:00 | 1077.65 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2023-12-26 10:00:00 | 1038.80 | 2023-12-26 14:15:00 | 1077.65 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2023-12-26 10:30:00 | 1032.00 | 2023-12-26 14:15:00 | 1077.65 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2024-01-11 12:00:00 | 1029.05 | 2024-01-16 12:15:00 | 1033.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-01-29 11:15:00 | 919.05 | 2024-01-30 10:15:00 | 873.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-29 11:15:00 | 919.05 | 2024-01-31 09:15:00 | 880.00 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2024-01-31 13:30:00 | 919.15 | 2024-01-31 14:15:00 | 922.85 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-02-09 09:15:00 | 902.20 | 2024-02-09 09:15:00 | 891.95 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-02-19 09:15:00 | 932.95 | 2024-02-19 12:15:00 | 916.05 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-02-19 09:45:00 | 936.80 | 2024-02-19 12:15:00 | 916.05 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-02-28 09:15:00 | 976.80 | 2024-02-28 11:15:00 | 947.30 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2024-03-04 09:15:00 | 928.70 | 2024-03-06 09:15:00 | 882.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 09:15:00 | 928.70 | 2024-03-07 12:15:00 | 879.80 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest2 | 2024-03-19 15:15:00 | 795.00 | 2024-03-21 09:15:00 | 834.15 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2024-04-18 15:00:00 | 979.25 | 2024-04-26 12:15:00 | 975.70 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2024-04-22 11:00:00 | 979.00 | 2024-04-26 12:15:00 | 975.70 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-04-22 11:30:00 | 979.30 | 2024-04-26 12:15:00 | 975.70 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2024-04-23 10:45:00 | 979.70 | 2024-04-26 12:15:00 | 975.70 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2024-04-24 10:45:00 | 972.50 | 2024-04-26 12:15:00 | 975.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-04-24 12:45:00 | 968.80 | 2024-04-26 12:15:00 | 975.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-04-24 15:00:00 | 970.45 | 2024-04-26 12:15:00 | 975.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-04-25 09:45:00 | 972.85 | 2024-04-26 12:15:00 | 975.70 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-05-08 15:15:00 | 909.00 | 2024-05-14 11:15:00 | 921.20 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-05-09 10:00:00 | 913.40 | 2024-05-14 11:15:00 | 921.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-05-09 12:15:00 | 913.55 | 2024-05-14 11:15:00 | 921.20 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-05-09 14:30:00 | 914.65 | 2024-05-14 11:15:00 | 921.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-05-13 09:15:00 | 898.85 | 2024-05-14 11:15:00 | 921.20 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-05-17 09:15:00 | 973.00 | 2024-05-23 09:15:00 | 1049.40 | TARGET_HIT | 1.00 | 7.85% |
| BUY | retest2 | 2024-05-21 13:00:00 | 954.00 | 2024-05-23 09:15:00 | 1045.88 | TARGET_HIT | 1.00 | 9.63% |
| BUY | retest2 | 2024-05-22 09:45:00 | 950.80 | 2024-05-24 11:15:00 | 1070.30 | TARGET_HIT | 1.00 | 12.57% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1013.15 | 2024-06-04 11:15:00 | 962.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1013.15 | 2024-06-04 13:15:00 | 1041.55 | STOP_HIT | 0.50 | -2.80% |
| BUY | retest2 | 2024-06-18 13:45:00 | 1292.65 | 2024-06-19 12:15:00 | 1421.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-19 10:00:00 | 1399.95 | 2024-06-21 09:15:00 | 1539.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-03 12:30:00 | 1463.50 | 2024-07-08 09:15:00 | 1390.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-03 14:00:00 | 1462.95 | 2024-07-08 09:15:00 | 1389.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-03 12:30:00 | 1463.50 | 2024-07-09 10:15:00 | 1406.75 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2024-07-03 14:00:00 | 1462.95 | 2024-07-09 10:15:00 | 1406.75 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2024-07-26 09:15:00 | 1705.00 | 2024-07-30 10:15:00 | 1639.05 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2024-08-13 09:30:00 | 1849.65 | 2024-08-16 09:15:00 | 2034.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-13 13:45:00 | 1835.05 | 2024-08-16 09:15:00 | 2018.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-13 14:30:00 | 1838.45 | 2024-08-16 09:15:00 | 2022.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1864.85 | 2024-08-16 09:15:00 | 2051.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 12:30:00 | 2182.00 | 2024-08-22 14:15:00 | 2187.00 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-08-21 09:15:00 | 2182.00 | 2024-08-22 14:15:00 | 2187.00 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-08-22 12:15:00 | 2194.45 | 2024-08-22 14:15:00 | 2187.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-09-03 12:15:00 | 2266.00 | 2024-09-06 14:15:00 | 2240.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-09-10 11:15:00 | 2269.30 | 2024-09-10 13:15:00 | 2285.75 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-09-18 12:15:00 | 2604.70 | 2024-09-19 09:15:00 | 2499.00 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2024-10-04 13:30:00 | 2355.65 | 2024-10-07 15:15:00 | 2243.90 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2024-10-07 10:00:00 | 2362.00 | 2024-10-08 09:15:00 | 2237.87 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2024-10-04 13:30:00 | 2355.65 | 2024-10-08 10:15:00 | 2359.95 | STOP_HIT | 0.50 | -0.18% |
| SELL | retest2 | 2024-10-07 10:00:00 | 2362.00 | 2024-10-08 10:15:00 | 2359.95 | STOP_HIT | 0.50 | 0.09% |
| BUY | retest2 | 2024-10-14 09:15:00 | 2557.90 | 2024-10-16 13:15:00 | 2476.50 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2024-10-16 12:00:00 | 2476.10 | 2024-10-16 13:15:00 | 2476.50 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-10-29 10:15:00 | 1991.40 | 2024-10-31 13:15:00 | 2085.00 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2024-10-29 12:15:00 | 1988.00 | 2024-10-31 13:15:00 | 2085.00 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2024-10-29 13:00:00 | 1991.70 | 2024-10-31 13:15:00 | 2085.00 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2024-11-05 09:15:00 | 2176.10 | 2024-11-12 14:15:00 | 2170.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-11-05 10:30:00 | 2140.75 | 2024-11-12 14:15:00 | 2170.65 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2024-11-05 14:15:00 | 2139.30 | 2024-11-12 14:15:00 | 2170.65 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2024-11-06 09:15:00 | 2175.00 | 2024-11-12 14:15:00 | 2170.65 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-11-19 14:45:00 | 2104.00 | 2024-11-25 10:15:00 | 2176.50 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2024-11-21 09:30:00 | 2101.50 | 2024-11-25 10:15:00 | 2176.50 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2024-11-22 11:15:00 | 2101.60 | 2024-11-25 10:15:00 | 2176.50 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2024-12-03 12:15:00 | 2145.60 | 2024-12-05 10:15:00 | 2177.15 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-12-04 09:15:00 | 2142.65 | 2024-12-05 10:15:00 | 2177.15 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-12-06 11:45:00 | 2220.55 | 2024-12-13 09:15:00 | 2210.10 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-12-30 12:15:00 | 2228.00 | 2025-01-02 12:15:00 | 2224.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-01-02 12:00:00 | 2229.30 | 2025-01-02 12:15:00 | 2224.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-01-15 09:30:00 | 2107.00 | 2025-01-15 11:15:00 | 2166.25 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-01-15 13:45:00 | 2109.90 | 2025-01-16 09:15:00 | 2169.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest1 | 2025-02-06 09:15:00 | 2050.35 | 2025-02-07 09:15:00 | 2024.80 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest1 | 2025-02-06 14:15:00 | 2042.00 | 2025-02-07 09:15:00 | 2024.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-03-04 15:15:00 | 1597.00 | 2025-03-05 09:15:00 | 1637.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-03-17 13:15:00 | 1597.90 | 2025-03-18 13:15:00 | 1660.00 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2025-03-17 14:15:00 | 1598.70 | 2025-03-18 13:15:00 | 1660.00 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-04-04 09:15:00 | 1746.00 | 2025-04-07 09:15:00 | 1571.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-24 10:00:00 | 1989.80 | 2025-04-24 13:15:00 | 1949.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-05-29 11:15:00 | 1886.30 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-05-29 11:45:00 | 1885.50 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-05-29 13:30:00 | 1885.30 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2025-05-29 14:15:00 | 1886.20 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-06-03 10:45:00 | 1862.20 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-03 11:45:00 | 1859.70 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-06-04 09:15:00 | 1857.50 | 2025-06-06 12:15:00 | 1868.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1665.20 | 2025-06-23 10:15:00 | 1687.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-02 11:15:00 | 1859.40 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-02 11:45:00 | 1872.30 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-02 14:45:00 | 1859.40 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-03 11:00:00 | 1859.90 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1851.30 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-07-04 12:00:00 | 1851.90 | 2025-07-04 13:15:00 | 1849.40 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-07-10 10:15:00 | 1777.10 | 2025-07-23 09:15:00 | 1729.10 | STOP_HIT | 1.00 | 2.70% |
| SELL | retest2 | 2025-07-10 10:45:00 | 1773.50 | 2025-07-23 09:15:00 | 1729.10 | STOP_HIT | 1.00 | 2.50% |
| SELL | retest2 | 2025-07-11 09:15:00 | 1768.00 | 2025-07-23 09:15:00 | 1729.10 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-07-29 09:15:00 | 1832.00 | 2025-08-01 11:15:00 | 1834.50 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-08-14 15:15:00 | 1780.30 | 2025-08-25 10:15:00 | 1750.80 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-09-12 12:15:00 | 1669.00 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-09-15 10:30:00 | 1671.50 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-09-15 12:00:00 | 1669.80 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-09-16 10:00:00 | 1670.40 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-09-16 11:15:00 | 1657.10 | 2025-09-17 13:15:00 | 1666.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-24 14:30:00 | 1653.30 | 2025-09-26 09:15:00 | 1570.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:30:00 | 1653.30 | 2025-09-29 09:15:00 | 1584.00 | STOP_HIT | 0.50 | 4.19% |
| BUY | retest2 | 2025-10-13 13:15:00 | 1619.80 | 2025-10-14 10:15:00 | 1599.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-13 15:15:00 | 1619.70 | 2025-10-14 10:15:00 | 1599.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-16 13:45:00 | 1573.20 | 2025-10-21 13:15:00 | 1601.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-10-17 11:45:00 | 1572.30 | 2025-10-21 13:15:00 | 1601.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-10-24 11:15:00 | 1698.00 | 2025-10-29 09:15:00 | 1630.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-10-24 12:00:00 | 1698.00 | 2025-10-29 09:15:00 | 1630.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1728.00 | 2025-11-06 12:15:00 | 1702.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1691.10 | 2025-11-10 10:15:00 | 1746.50 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-11-13 09:15:00 | 1752.10 | 2025-11-13 10:15:00 | 1731.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1690.40 | 2025-11-24 09:15:00 | 1715.60 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1698.50 | 2025-11-24 12:15:00 | 1738.90 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-11-28 11:45:00 | 1834.70 | 2025-12-01 09:15:00 | 1792.90 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-11-28 13:00:00 | 1832.20 | 2025-12-01 09:15:00 | 1792.90 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-16 13:30:00 | 1834.90 | 2025-12-18 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-12-16 14:45:00 | 1841.30 | 2025-12-18 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-12-17 11:45:00 | 1841.50 | 2025-12-18 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-12-17 14:00:00 | 1837.10 | 2025-12-18 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-01-20 10:45:00 | 1574.30 | 2026-01-27 12:15:00 | 1495.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 09:45:00 | 1571.50 | 2026-01-27 12:15:00 | 1492.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:00:00 | 1574.00 | 2026-01-27 12:15:00 | 1495.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 10:45:00 | 1574.30 | 2026-01-28 09:15:00 | 1505.20 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2026-01-22 09:45:00 | 1571.50 | 2026-01-28 09:15:00 | 1505.20 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2026-01-23 10:00:00 | 1574.00 | 2026-01-28 09:15:00 | 1505.20 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1646.40 | 2026-02-26 09:15:00 | 1642.00 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1537.00 | 2026-03-13 09:15:00 | 1460.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1537.00 | 2026-03-16 13:15:00 | 1383.30 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1530.90 | 2026-04-20 09:15:00 | 1683.99 | TARGET_HIT | 1.00 | 10.00% |
