# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 890.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 214 |
| ALERT1 | 138 |
| ALERT2 | 136 |
| ALERT2_SKIP | 72 |
| ALERT3 | 356 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 193 |
| PARTIAL | 25 |
| TARGET_HIT | 15 |
| STOP_HIT | 182 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 221 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 87 / 134
- **Target hits / Stop hits / Partials:** 15 / 181 / 25
- **Avg / median % per leg:** 0.75% / -0.71%
- **Sum % (uncompounded):** 164.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 85 | 21 | 24.7% | 8 | 77 | 0 | -0.00% | -0.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 85 | 21 | 24.7% | 8 | 77 | 0 | -0.00% | -0.1% |
| SELL (all) | 136 | 66 | 48.5% | 7 | 104 | 25 | 1.21% | 165.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.50% | -4.5% |
| SELL @ 3rd Alert (retest2) | 133 | 66 | 49.6% | 7 | 101 | 25 | 1.27% | 169.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.50% | -4.5% |
| retest2 (combined) | 218 | 87 | 39.9% | 15 | 178 | 25 | 0.78% | 169.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 13:15:00 | 636.50 | 637.73 | 637.81 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 15:15:00 | 639.00 | 638.06 | 637.95 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 10:15:00 | 635.85 | 637.54 | 637.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 09:15:00 | 624.15 | 633.13 | 635.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 15:15:00 | 602.90 | 602.55 | 609.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-29 09:15:00 | 610.00 | 602.55 | 609.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 611.45 | 604.33 | 609.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:30:00 | 610.65 | 604.33 | 609.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 608.40 | 605.14 | 609.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 13:15:00 | 605.60 | 606.08 | 608.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 14:15:00 | 605.55 | 606.47 | 608.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-29 15:00:00 | 606.45 | 606.46 | 608.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 09:15:00 | 605.45 | 608.40 | 608.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 606.75 | 608.07 | 608.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 10:15:00 | 612.50 | 608.07 | 608.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-05-31 10:15:00 | 614.40 | 609.33 | 609.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 10:15:00 | 614.40 | 609.33 | 609.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 11:15:00 | 616.35 | 610.74 | 609.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 14:15:00 | 601.50 | 610.08 | 609.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 601.50 | 610.08 | 609.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 601.50 | 610.08 | 609.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 601.50 | 610.08 | 609.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 15:15:00 | 599.05 | 607.87 | 608.80 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 615.75 | 609.45 | 609.44 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 609.10 | 612.55 | 612.60 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 14:15:00 | 615.50 | 613.05 | 612.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 10:15:00 | 616.55 | 614.35 | 613.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 13:15:00 | 612.00 | 614.23 | 613.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 13:15:00 | 612.00 | 614.23 | 613.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 13:15:00 | 612.00 | 614.23 | 613.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 14:00:00 | 612.00 | 614.23 | 613.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 14:15:00 | 616.00 | 614.58 | 613.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 14:45:00 | 611.65 | 614.58 | 613.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 616.25 | 614.92 | 614.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:15:00 | 612.10 | 614.92 | 614.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 09:15:00 | 615.20 | 614.97 | 614.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 09:30:00 | 612.00 | 614.97 | 614.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 10:15:00 | 614.30 | 614.84 | 614.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 10:45:00 | 615.00 | 614.84 | 614.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 615.90 | 615.05 | 614.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 12:15:00 | 616.70 | 615.05 | 614.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 14:15:00 | 618.10 | 615.17 | 614.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 15:15:00 | 621.30 | 623.86 | 623.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 15:15:00 | 621.30 | 623.86 | 623.92 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 13:15:00 | 625.10 | 624.04 | 623.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 14:15:00 | 627.00 | 624.63 | 624.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 09:15:00 | 622.45 | 625.08 | 624.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 09:15:00 | 622.45 | 625.08 | 624.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 622.45 | 625.08 | 624.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 10:00:00 | 622.45 | 625.08 | 624.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 10:15:00 | 620.45 | 624.15 | 624.16 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 13:15:00 | 636.50 | 626.13 | 624.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 13:15:00 | 640.90 | 636.78 | 634.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 13:15:00 | 641.40 | 642.22 | 638.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 14:00:00 | 641.40 | 642.22 | 638.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 637.60 | 641.30 | 638.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:45:00 | 637.25 | 641.30 | 638.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 634.00 | 639.84 | 638.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 628.85 | 639.84 | 638.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 628.10 | 635.65 | 636.65 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 09:15:00 | 643.00 | 636.20 | 636.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 12:15:00 | 659.45 | 644.61 | 641.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 679.45 | 684.84 | 676.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 09:15:00 | 679.45 | 684.84 | 676.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 679.45 | 684.84 | 676.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 679.45 | 684.84 | 676.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 680.35 | 683.05 | 678.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:45:00 | 678.85 | 683.05 | 678.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 680.80 | 682.34 | 679.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:15:00 | 674.55 | 682.34 | 679.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 677.95 | 681.46 | 679.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:30:00 | 678.50 | 681.46 | 679.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 11:15:00 | 678.45 | 680.86 | 679.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 12:15:00 | 678.20 | 680.86 | 679.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 12:15:00 | 678.25 | 680.34 | 679.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 13:00:00 | 678.25 | 680.34 | 679.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 696.35 | 699.07 | 695.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:30:00 | 694.95 | 699.07 | 695.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 697.10 | 698.67 | 695.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 10:30:00 | 696.55 | 698.67 | 695.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 691.75 | 697.29 | 695.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 12:00:00 | 691.75 | 697.29 | 695.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 12:15:00 | 693.25 | 696.48 | 695.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 13:15:00 | 694.50 | 696.48 | 695.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 15:00:00 | 694.15 | 695.34 | 694.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 09:15:00 | 686.65 | 693.23 | 693.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 09:15:00 | 686.65 | 693.23 | 693.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 13:15:00 | 681.55 | 687.90 | 690.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 09:15:00 | 695.00 | 686.51 | 689.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 09:15:00 | 695.00 | 686.51 | 689.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 695.00 | 686.51 | 689.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:45:00 | 694.60 | 686.51 | 689.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 696.00 | 688.41 | 689.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:45:00 | 697.75 | 688.41 | 689.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 12:15:00 | 695.55 | 691.07 | 690.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 14:15:00 | 698.00 | 693.21 | 691.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 12:15:00 | 694.50 | 695.30 | 693.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 12:45:00 | 694.55 | 695.30 | 693.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 685.75 | 693.39 | 693.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 685.75 | 693.39 | 693.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 689.50 | 692.61 | 692.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 09:15:00 | 652.00 | 679.90 | 684.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 10:15:00 | 655.70 | 653.97 | 665.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-19 11:00:00 | 655.70 | 653.97 | 665.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 628.90 | 626.24 | 628.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:45:00 | 628.45 | 626.24 | 628.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 625.80 | 626.15 | 628.67 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 12:15:00 | 632.60 | 629.62 | 629.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 15:15:00 | 633.50 | 631.27 | 630.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-27 14:15:00 | 631.55 | 633.33 | 632.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 14:15:00 | 631.55 | 633.33 | 632.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 631.55 | 633.33 | 632.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:45:00 | 632.50 | 633.33 | 632.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 632.20 | 633.11 | 632.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:15:00 | 626.30 | 633.11 | 632.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 626.00 | 631.69 | 631.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:00:00 | 626.00 | 631.69 | 631.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 10:15:00 | 627.20 | 630.79 | 631.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 11:15:00 | 624.15 | 629.46 | 630.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 11:15:00 | 630.90 | 626.72 | 628.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 11:15:00 | 630.90 | 626.72 | 628.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 630.90 | 626.72 | 628.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 12:00:00 | 630.90 | 626.72 | 628.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 630.80 | 627.54 | 628.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 13:15:00 | 633.35 | 627.54 | 628.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 13:15:00 | 635.00 | 629.03 | 628.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 10:15:00 | 640.85 | 633.46 | 631.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 636.70 | 637.85 | 634.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 636.70 | 637.85 | 634.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 636.70 | 637.85 | 634.90 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 628.35 | 634.06 | 634.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 11:15:00 | 625.00 | 631.54 | 633.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 630.15 | 628.08 | 630.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 630.15 | 628.08 | 630.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 630.15 | 628.08 | 630.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 10:00:00 | 630.15 | 628.08 | 630.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 630.10 | 628.49 | 630.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 11:00:00 | 630.10 | 628.49 | 630.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 630.90 | 628.97 | 630.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 12:00:00 | 630.90 | 628.97 | 630.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 630.00 | 629.18 | 630.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:30:00 | 628.15 | 628.86 | 630.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:15:00 | 629.20 | 628.40 | 629.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 14:15:00 | 628.45 | 629.46 | 629.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 10:15:00 | 620.90 | 618.02 | 617.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 620.90 | 618.02 | 617.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 13:15:00 | 624.35 | 620.28 | 619.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 09:15:00 | 625.25 | 626.22 | 623.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 625.25 | 626.22 | 623.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 625.25 | 626.22 | 623.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:30:00 | 627.70 | 626.22 | 623.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 624.55 | 625.89 | 623.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:45:00 | 625.00 | 625.89 | 623.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 625.00 | 625.71 | 623.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 12:15:00 | 626.65 | 625.71 | 623.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 14:00:00 | 626.10 | 625.66 | 624.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 15:00:00 | 625.90 | 627.94 | 627.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 15:15:00 | 624.45 | 627.24 | 627.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 624.45 | 627.24 | 627.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 623.40 | 626.04 | 626.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 12:15:00 | 627.85 | 626.05 | 626.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 12:15:00 | 627.85 | 626.05 | 626.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 12:15:00 | 627.85 | 626.05 | 626.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:45:00 | 626.85 | 626.05 | 626.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 627.25 | 626.29 | 626.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 13:30:00 | 629.35 | 626.29 | 626.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 624.85 | 626.00 | 626.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 14:30:00 | 627.20 | 626.00 | 626.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 628.85 | 626.18 | 626.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:45:00 | 628.95 | 626.18 | 626.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 628.05 | 626.55 | 626.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:00:00 | 628.05 | 626.55 | 626.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 628.15 | 626.87 | 626.78 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 14:15:00 | 625.70 | 626.67 | 626.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 13:15:00 | 624.20 | 625.74 | 626.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 14:15:00 | 625.80 | 625.75 | 626.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 15:00:00 | 625.80 | 625.75 | 626.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 626.95 | 625.99 | 626.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 09:15:00 | 627.35 | 625.99 | 626.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 627.95 | 626.38 | 626.40 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 10:15:00 | 627.90 | 626.69 | 626.54 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 624.50 | 626.45 | 626.54 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 14:15:00 | 628.60 | 626.67 | 626.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 15:15:00 | 628.95 | 627.12 | 626.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 11:15:00 | 627.00 | 627.33 | 626.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 11:15:00 | 627.00 | 627.33 | 626.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 627.00 | 627.33 | 626.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:45:00 | 627.00 | 627.33 | 626.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 627.00 | 627.26 | 626.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:45:00 | 626.85 | 627.26 | 626.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 626.65 | 627.14 | 626.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:45:00 | 626.90 | 627.14 | 626.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 14:15:00 | 625.60 | 626.83 | 626.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 10:15:00 | 624.60 | 626.14 | 626.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 13:15:00 | 627.95 | 626.16 | 626.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 13:15:00 | 627.95 | 626.16 | 626.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 627.95 | 626.16 | 626.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 13:30:00 | 628.25 | 626.16 | 626.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 625.10 | 625.95 | 626.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-04 15:15:00 | 624.00 | 625.95 | 626.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 09:15:00 | 648.20 | 630.09 | 628.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 648.20 | 630.09 | 628.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 10:15:00 | 650.95 | 634.26 | 630.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 09:15:00 | 656.35 | 658.95 | 654.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 656.35 | 658.95 | 654.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 656.35 | 658.95 | 654.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 11:45:00 | 670.65 | 660.78 | 655.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 13:00:00 | 664.00 | 660.60 | 658.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 13:45:00 | 663.75 | 660.92 | 658.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 14:45:00 | 664.00 | 661.21 | 658.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 15:15:00 | 659.50 | 660.87 | 658.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-12 09:15:00 | 651.55 | 659.00 | 658.22 | SL hit (close<static) qty=1.00 sl=654.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 651.70 | 657.54 | 657.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 644.50 | 653.07 | 655.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 650.90 | 649.03 | 651.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 650.90 | 649.03 | 651.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 650.90 | 649.03 | 651.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 655.50 | 649.03 | 651.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 651.45 | 649.74 | 651.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 11:30:00 | 651.45 | 649.74 | 651.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 650.40 | 649.87 | 651.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 13:30:00 | 647.00 | 649.55 | 651.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 14:15:00 | 647.40 | 649.55 | 651.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 15:15:00 | 647.30 | 649.39 | 650.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 10:15:00 | 653.35 | 650.17 | 650.82 | SL hit (close>static) qty=1.00 sl=652.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 12:15:00 | 651.65 | 650.92 | 650.82 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 15:15:00 | 650.00 | 650.65 | 650.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 09:15:00 | 645.60 | 649.64 | 650.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 12:15:00 | 636.35 | 635.96 | 638.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-25 12:45:00 | 637.50 | 635.96 | 638.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 644.20 | 637.79 | 638.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:45:00 | 645.80 | 637.79 | 638.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 642.05 | 638.64 | 638.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:30:00 | 643.20 | 638.64 | 638.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 641.65 | 639.24 | 638.98 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 13:15:00 | 638.80 | 639.42 | 639.47 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 642.90 | 639.99 | 639.71 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 636.95 | 639.58 | 639.70 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 642.15 | 640.15 | 639.94 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 13:15:00 | 639.25 | 639.81 | 639.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 14:15:00 | 637.70 | 639.39 | 639.67 | Break + close below crossover candle low |

### Cycle 40 — BUY (started 2023-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 09:15:00 | 646.70 | 640.60 | 640.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 11:15:00 | 649.25 | 643.46 | 641.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 14:15:00 | 645.25 | 645.69 | 643.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-03 14:45:00 | 645.05 | 645.69 | 643.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 15:15:00 | 642.90 | 645.13 | 643.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:15:00 | 652.90 | 645.13 | 643.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 652.05 | 646.52 | 644.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 10:15:00 | 656.15 | 646.52 | 644.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 13:15:00 | 642.00 | 645.14 | 645.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 13:15:00 | 642.00 | 645.14 | 645.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 14:15:00 | 640.25 | 644.17 | 645.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 644.45 | 643.44 | 644.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 09:15:00 | 644.45 | 643.44 | 644.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 644.45 | 643.44 | 644.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 09:30:00 | 645.45 | 643.44 | 644.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 642.65 | 643.28 | 644.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 11:15:00 | 641.90 | 643.28 | 644.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 14:15:00 | 644.95 | 643.55 | 644.11 | SL hit (close>static) qty=1.00 sl=644.90 alert=retest2 |

### Cycle 42 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 642.90 | 639.44 | 639.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 645.25 | 641.90 | 640.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 15:15:00 | 646.70 | 646.96 | 644.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 09:15:00 | 650.10 | 646.96 | 644.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 640.00 | 646.08 | 645.46 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 11:15:00 | 640.75 | 644.20 | 644.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 12:15:00 | 639.85 | 643.33 | 644.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 643.60 | 641.72 | 643.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 643.60 | 641.72 | 643.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 643.60 | 641.72 | 643.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:15:00 | 643.35 | 641.72 | 643.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 645.25 | 642.43 | 643.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:45:00 | 645.70 | 642.43 | 643.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 643.50 | 642.64 | 643.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 12:15:00 | 642.85 | 642.64 | 643.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-17 12:15:00 | 647.30 | 643.57 | 643.61 | SL hit (close>static) qty=1.00 sl=646.40 alert=retest2 |

### Cycle 44 — BUY (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 14:15:00 | 643.95 | 643.64 | 643.63 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 15:15:00 | 643.00 | 643.51 | 643.57 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 09:15:00 | 646.35 | 644.08 | 643.82 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 640.15 | 643.29 | 643.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 638.00 | 641.35 | 642.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 15:15:00 | 608.00 | 607.42 | 613.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 09:15:00 | 618.00 | 607.42 | 613.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 620.95 | 610.13 | 613.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 620.95 | 610.13 | 613.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 618.25 | 611.75 | 614.28 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 621.65 | 616.54 | 616.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 09:15:00 | 624.55 | 619.38 | 617.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 09:15:00 | 625.25 | 627.51 | 623.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 09:30:00 | 627.65 | 627.51 | 623.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 620.45 | 626.09 | 623.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:00:00 | 620.45 | 626.09 | 623.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 621.00 | 625.08 | 623.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:30:00 | 620.05 | 625.08 | 623.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 621.70 | 624.40 | 622.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 12:45:00 | 620.90 | 624.40 | 622.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 13:15:00 | 618.35 | 623.19 | 622.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 14:00:00 | 618.35 | 623.19 | 622.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 14:15:00 | 613.95 | 621.34 | 621.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 15:15:00 | 611.00 | 619.27 | 620.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-01 09:15:00 | 623.10 | 620.04 | 620.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 623.10 | 620.04 | 620.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 623.10 | 620.04 | 620.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 10:30:00 | 620.05 | 620.23 | 620.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 11:30:00 | 619.25 | 619.99 | 620.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 09:30:00 | 619.35 | 619.00 | 619.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 11:15:00 | 621.50 | 619.78 | 619.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 11:15:00 | 621.50 | 619.78 | 619.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 624.10 | 621.38 | 620.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 09:15:00 | 632.80 | 634.64 | 631.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-09 10:15:00 | 632.00 | 634.64 | 631.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 631.05 | 633.93 | 631.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:45:00 | 631.15 | 633.93 | 631.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 629.60 | 633.06 | 631.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:45:00 | 629.40 | 633.06 | 631.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 632.50 | 632.95 | 631.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 12:30:00 | 630.85 | 632.95 | 631.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 631.95 | 632.75 | 631.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 13:45:00 | 631.50 | 632.75 | 631.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 630.35 | 632.27 | 631.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 630.35 | 632.27 | 631.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 631.05 | 632.03 | 631.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 630.00 | 632.03 | 631.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 626.85 | 630.99 | 631.09 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 638.75 | 630.82 | 630.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 13:15:00 | 640.40 | 635.86 | 633.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 14:15:00 | 642.40 | 642.81 | 640.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 15:00:00 | 642.40 | 642.81 | 640.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 642.95 | 642.84 | 640.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:15:00 | 643.20 | 642.84 | 640.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 14:15:00 | 639.25 | 643.15 | 642.03 | SL hit (close<static) qty=1.00 sl=640.55 alert=retest2 |

### Cycle 53 — SELL (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 15:15:00 | 639.00 | 641.36 | 641.62 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 09:15:00 | 644.05 | 641.90 | 641.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 10:15:00 | 648.45 | 643.21 | 642.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 14:15:00 | 649.85 | 651.81 | 649.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 14:15:00 | 649.85 | 651.81 | 649.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 649.85 | 651.81 | 649.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 15:15:00 | 650.45 | 651.81 | 649.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 15:15:00 | 650.45 | 651.54 | 649.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:15:00 | 651.40 | 651.54 | 649.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 650.45 | 651.32 | 649.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 10:15:00 | 655.15 | 651.32 | 649.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 11:00:00 | 654.10 | 651.87 | 649.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-01 09:15:00 | 720.67 | 710.03 | 699.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 753.00 | 757.66 | 758.01 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 10:15:00 | 762.35 | 758.62 | 758.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 10:15:00 | 765.25 | 761.16 | 759.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 759.55 | 760.85 | 759.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 12:15:00 | 759.55 | 760.85 | 759.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 12:15:00 | 759.55 | 760.85 | 759.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 12:45:00 | 759.50 | 760.85 | 759.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 13:15:00 | 757.10 | 760.10 | 759.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 13:45:00 | 757.25 | 760.10 | 759.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2023-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 14:15:00 | 755.15 | 759.11 | 759.29 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 09:15:00 | 769.55 | 760.86 | 760.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 13:15:00 | 778.80 | 766.64 | 763.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 10:15:00 | 770.00 | 770.54 | 766.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-14 11:00:00 | 770.00 | 770.54 | 766.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 767.00 | 770.50 | 768.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 11:00:00 | 767.00 | 770.50 | 768.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 768.40 | 770.08 | 768.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 11:30:00 | 768.05 | 770.08 | 768.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 767.70 | 769.60 | 768.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 12:45:00 | 768.70 | 769.60 | 768.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 13:15:00 | 765.00 | 768.68 | 768.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:45:00 | 764.45 | 768.68 | 768.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 767.85 | 768.52 | 768.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 14:45:00 | 764.60 | 768.52 | 768.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 15:15:00 | 763.60 | 767.53 | 767.67 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2023-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 11:15:00 | 781.50 | 769.83 | 768.62 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 767.50 | 776.77 | 776.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 750.15 | 771.45 | 774.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 760.95 | 759.73 | 766.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 12:00:00 | 760.95 | 759.73 | 766.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 763.90 | 760.85 | 765.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:45:00 | 759.50 | 762.57 | 764.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 774.55 | 764.88 | 764.98 | SL hit (close>static) qty=1.00 sl=765.25 alert=retest2 |

### Cycle 62 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 773.30 | 766.56 | 765.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 11:15:00 | 776.90 | 768.63 | 766.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 14:15:00 | 769.20 | 770.20 | 768.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 15:00:00 | 769.20 | 770.20 | 768.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 772.00 | 770.56 | 768.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 09:15:00 | 775.15 | 770.56 | 768.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 10:15:00 | 775.25 | 770.65 | 768.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-03 13:15:00 | 813.25 | 814.19 | 814.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 13:15:00 | 813.25 | 814.19 | 814.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 14:15:00 | 811.10 | 813.57 | 813.97 | Break + close below crossover candle low |

### Cycle 64 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 819.90 | 814.11 | 814.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 09:15:00 | 827.75 | 819.35 | 817.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 09:15:00 | 826.45 | 828.78 | 824.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 826.45 | 828.78 | 824.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 826.45 | 828.78 | 824.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:45:00 | 827.30 | 828.78 | 824.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 826.95 | 828.41 | 824.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 826.95 | 828.41 | 824.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 824.40 | 827.61 | 824.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 825.70 | 827.61 | 824.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 826.45 | 827.38 | 824.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 832.75 | 826.56 | 824.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-09 13:15:00 | 821.30 | 826.03 | 825.48 | SL hit (close<static) qty=1.00 sl=824.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 14:15:00 | 814.00 | 823.63 | 824.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 810.35 | 819.89 | 822.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 10:15:00 | 814.40 | 814.18 | 817.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-11 11:00:00 | 814.40 | 814.18 | 817.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 815.00 | 805.74 | 808.28 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 11:15:00 | 819.30 | 810.40 | 810.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 820.80 | 815.16 | 812.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 810.85 | 814.50 | 812.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 11:15:00 | 810.85 | 814.50 | 812.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 810.85 | 814.50 | 812.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 11:45:00 | 812.80 | 814.50 | 812.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 806.05 | 812.81 | 812.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 806.05 | 812.81 | 812.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 13:15:00 | 801.75 | 810.60 | 811.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 14:15:00 | 796.65 | 803.05 | 806.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 14:15:00 | 811.00 | 799.23 | 801.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 14:15:00 | 811.00 | 799.23 | 801.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 14:15:00 | 811.00 | 799.23 | 801.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 15:00:00 | 811.00 | 799.23 | 801.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 810.00 | 801.39 | 802.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:15:00 | 812.75 | 801.39 | 802.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 10:15:00 | 807.05 | 803.96 | 803.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 816.90 | 808.60 | 806.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 809.20 | 809.74 | 807.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 10:00:00 | 809.20 | 809.74 | 807.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 811.00 | 809.96 | 808.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 13:30:00 | 810.15 | 809.96 | 808.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 809.00 | 809.77 | 808.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 809.00 | 809.77 | 808.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 810.90 | 810.00 | 808.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 09:15:00 | 813.00 | 810.00 | 808.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 796.70 | 806.65 | 807.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 796.70 | 806.65 | 807.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 792.85 | 802.48 | 805.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 799.80 | 797.51 | 801.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 10:15:00 | 799.80 | 797.51 | 801.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 799.80 | 797.51 | 801.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:30:00 | 800.65 | 797.51 | 801.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 803.85 | 798.64 | 801.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:00:00 | 803.85 | 798.64 | 801.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 800.00 | 798.91 | 801.17 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 814.20 | 802.28 | 802.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 818.05 | 811.04 | 807.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 13:15:00 | 881.25 | 881.77 | 867.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 13:30:00 | 882.50 | 881.77 | 867.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 891.70 | 896.52 | 888.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 15:00:00 | 891.70 | 896.52 | 888.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 892.60 | 895.73 | 888.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 09:15:00 | 903.50 | 895.73 | 888.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-06 14:15:00 | 886.25 | 895.01 | 891.82 | SL hit (close<static) qty=1.00 sl=887.05 alert=retest2 |

### Cycle 71 — SELL (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 13:15:00 | 886.50 | 890.73 | 890.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 15:15:00 | 884.00 | 888.92 | 889.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 09:15:00 | 890.85 | 889.31 | 890.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 890.85 | 889.31 | 890.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 890.85 | 889.31 | 890.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:45:00 | 890.65 | 889.31 | 890.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 885.30 | 888.51 | 889.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 14:45:00 | 878.35 | 886.96 | 888.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-09 09:30:00 | 869.85 | 882.40 | 886.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 09:15:00 | 834.43 | 849.32 | 861.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 09:15:00 | 826.36 | 849.32 | 861.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-13 13:15:00 | 848.65 | 844.11 | 854.32 | SL hit (close>ema200) qty=0.50 sl=844.11 alert=retest2 |

### Cycle 72 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 865.25 | 853.76 | 853.08 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 12:15:00 | 857.00 | 860.70 | 860.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 14:15:00 | 855.55 | 859.14 | 860.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 870.50 | 858.23 | 858.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 870.50 | 858.23 | 858.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 870.50 | 858.23 | 858.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 11:15:00 | 855.00 | 858.17 | 858.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-21 12:00:00 | 845.35 | 855.60 | 857.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 11:30:00 | 855.05 | 844.77 | 846.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 14:15:00 | 856.15 | 848.51 | 847.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 14:15:00 | 856.15 | 848.51 | 847.67 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 843.30 | 847.08 | 847.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 842.20 | 845.74 | 846.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 841.40 | 839.00 | 841.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 841.40 | 839.00 | 841.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 841.40 | 839.00 | 841.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:30:00 | 841.10 | 839.00 | 841.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 833.20 | 837.84 | 841.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 09:15:00 | 827.05 | 835.21 | 838.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 10:00:00 | 825.90 | 833.35 | 837.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 842.20 | 835.39 | 835.82 | SL hit (close>static) qty=1.00 sl=842.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 842.00 | 836.71 | 836.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 846.00 | 840.10 | 838.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 14:15:00 | 850.00 | 851.26 | 848.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 15:00:00 | 850.00 | 851.26 | 848.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 850.00 | 851.01 | 848.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 853.40 | 851.01 | 848.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 11:15:00 | 844.75 | 849.22 | 848.25 | SL hit (close<static) qty=1.00 sl=845.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 12:15:00 | 848.75 | 865.50 | 866.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 13:15:00 | 844.90 | 861.38 | 864.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 806.00 | 802.76 | 823.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 806.00 | 802.76 | 823.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 792.40 | 792.79 | 803.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 10:30:00 | 786.15 | 791.74 | 800.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 12:15:00 | 789.00 | 791.85 | 799.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 13:15:00 | 788.70 | 791.65 | 798.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 14:15:00 | 789.95 | 791.59 | 797.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:15:00 | 746.84 | 769.18 | 781.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:15:00 | 749.55 | 769.18 | 781.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:15:00 | 749.26 | 769.18 | 781.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 10:15:00 | 750.45 | 769.18 | 781.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-21 11:15:00 | 758.30 | 756.11 | 765.95 | SL hit (close>ema200) qty=0.50 sl=756.11 alert=retest2 |

### Cycle 78 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 780.50 | 768.43 | 767.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 11:15:00 | 782.25 | 776.35 | 772.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 12:15:00 | 773.00 | 775.68 | 772.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 12:15:00 | 773.00 | 775.68 | 772.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 12:15:00 | 773.00 | 775.68 | 772.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 13:00:00 | 773.00 | 775.68 | 772.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 769.00 | 774.34 | 772.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:00:00 | 769.00 | 774.34 | 772.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 773.75 | 774.22 | 772.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:45:00 | 766.00 | 774.22 | 772.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 781.90 | 775.76 | 773.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 782.85 | 775.76 | 773.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 10:45:00 | 782.20 | 778.07 | 774.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 15:15:00 | 767.80 | 774.74 | 774.40 | SL hit (close<static) qty=1.00 sl=769.20 alert=retest2 |

### Cycle 79 — SELL (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 10:15:00 | 769.20 | 773.19 | 773.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 11:15:00 | 766.50 | 771.86 | 773.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 791.85 | 771.24 | 771.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 791.85 | 771.24 | 771.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 791.85 | 771.24 | 771.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:00:00 | 791.85 | 771.24 | 771.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 789.95 | 774.98 | 773.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 12:15:00 | 793.95 | 781.57 | 776.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 13:15:00 | 808.70 | 809.02 | 805.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 14:00:00 | 808.70 | 809.02 | 805.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 875.00 | 881.03 | 874.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:00:00 | 875.00 | 881.03 | 874.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 873.35 | 879.49 | 874.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:15:00 | 871.75 | 879.49 | 874.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 869.95 | 877.58 | 873.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 13:45:00 | 870.50 | 877.58 | 873.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 879.00 | 881.11 | 876.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 11:15:00 | 885.40 | 881.28 | 877.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 12:15:00 | 887.20 | 881.59 | 877.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 09:45:00 | 888.40 | 884.92 | 880.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-16 14:15:00 | 973.94 | 929.27 | 906.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 1097.30 | 1101.24 | 1101.47 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 13:15:00 | 1113.70 | 1103.20 | 1102.25 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 15:15:00 | 1097.00 | 1101.30 | 1101.51 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 11:15:00 | 1110.65 | 1102.87 | 1102.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 09:15:00 | 1119.10 | 1109.86 | 1106.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 09:15:00 | 1103.10 | 1116.53 | 1112.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 09:15:00 | 1103.10 | 1116.53 | 1112.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 1103.10 | 1116.53 | 1112.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 1103.10 | 1116.53 | 1112.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1116.35 | 1116.49 | 1112.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 12:45:00 | 1125.40 | 1116.95 | 1113.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 13:30:00 | 1126.05 | 1118.56 | 1114.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 15:00:00 | 1123.00 | 1119.45 | 1115.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 09:45:00 | 1156.55 | 1123.85 | 1118.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 1114.20 | 1122.29 | 1118.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:30:00 | 1113.80 | 1122.29 | 1118.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1094.95 | 1116.83 | 1116.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-07 12:15:00 | 1094.95 | 1116.83 | 1116.27 | SL hit (close<static) qty=1.00 sl=1099.70 alert=retest2 |

### Cycle 85 — SELL (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 14:15:00 | 1106.70 | 1115.32 | 1115.72 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 09:15:00 | 1150.00 | 1120.61 | 1117.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 10:15:00 | 1153.05 | 1127.10 | 1121.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 14:15:00 | 1132.20 | 1132.41 | 1126.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-08 15:00:00 | 1132.20 | 1132.41 | 1126.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 1113.85 | 1128.60 | 1125.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:00:00 | 1113.85 | 1128.60 | 1125.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 10:15:00 | 1099.85 | 1122.85 | 1123.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 1092.00 | 1108.76 | 1115.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 13:15:00 | 1074.00 | 1066.36 | 1081.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 13:45:00 | 1073.90 | 1066.36 | 1081.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1058.00 | 1064.55 | 1076.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:30:00 | 1073.85 | 1064.55 | 1076.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 1070.50 | 1067.16 | 1075.76 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1105.25 | 1084.08 | 1081.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 15:15:00 | 1123.00 | 1107.09 | 1095.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 1155.55 | 1159.45 | 1149.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 1155.55 | 1159.45 | 1149.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1153.00 | 1158.16 | 1149.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 1131.30 | 1155.53 | 1149.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1147.85 | 1153.99 | 1149.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:30:00 | 1144.65 | 1153.99 | 1149.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 1152.70 | 1153.73 | 1149.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 1144.40 | 1153.73 | 1149.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 1144.20 | 1151.83 | 1149.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:30:00 | 1142.00 | 1151.83 | 1149.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 1137.55 | 1148.97 | 1148.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 1140.10 | 1148.97 | 1148.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 1142.60 | 1146.66 | 1147.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 09:15:00 | 1121.00 | 1141.53 | 1144.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 1163.40 | 1132.08 | 1135.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 1163.40 | 1132.08 | 1135.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1163.40 | 1132.08 | 1135.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 1163.40 | 1132.08 | 1135.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 1186.00 | 1142.86 | 1140.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 11:15:00 | 1208.00 | 1155.89 | 1146.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 1217.75 | 1219.51 | 1195.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 15:00:00 | 1217.75 | 1219.51 | 1195.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1244.70 | 1241.61 | 1223.61 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 1205.00 | 1217.63 | 1218.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 1199.75 | 1211.27 | 1215.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 1200.00 | 1198.73 | 1205.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 1200.00 | 1198.73 | 1205.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1200.00 | 1198.73 | 1205.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 1200.00 | 1198.73 | 1205.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1201.95 | 1193.95 | 1201.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:45:00 | 1193.05 | 1193.68 | 1200.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:45:00 | 1194.60 | 1194.66 | 1199.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 15:15:00 | 1194.00 | 1195.81 | 1199.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1133.40 | 1185.54 | 1194.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1134.87 | 1185.54 | 1194.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1134.30 | 1185.54 | 1194.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 10:15:00 | 1073.74 | 1165.95 | 1184.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 92 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 1211.90 | 1161.81 | 1161.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1296.25 | 1209.80 | 1185.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1402.20 | 1406.80 | 1358.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 1398.90 | 1406.80 | 1358.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1381.85 | 1401.04 | 1364.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 1379.00 | 1401.04 | 1364.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 1361.30 | 1384.35 | 1364.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:45:00 | 1353.35 | 1384.35 | 1364.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 1368.00 | 1381.08 | 1365.26 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 1344.55 | 1359.91 | 1361.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 11:15:00 | 1333.95 | 1343.79 | 1350.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 1365.00 | 1345.18 | 1347.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 1365.00 | 1345.18 | 1347.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1365.00 | 1345.18 | 1347.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 1365.20 | 1345.18 | 1347.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 1389.95 | 1354.13 | 1351.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 1421.10 | 1407.01 | 1396.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 1406.40 | 1408.24 | 1399.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 1406.40 | 1408.24 | 1399.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1394.00 | 1405.92 | 1401.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1385.95 | 1405.92 | 1401.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1397.00 | 1404.13 | 1400.96 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 1380.45 | 1396.56 | 1398.18 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1595.75 | 1433.75 | 1414.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 1674.80 | 1597.18 | 1521.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 09:15:00 | 1618.70 | 1651.64 | 1594.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 1608.00 | 1627.84 | 1600.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1608.00 | 1627.84 | 1600.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:45:00 | 1606.95 | 1627.84 | 1600.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 1608.00 | 1621.71 | 1602.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 1599.00 | 1621.71 | 1602.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1604.45 | 1618.25 | 1602.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 11:45:00 | 1690.00 | 1631.91 | 1611.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 1676.95 | 1687.31 | 1688.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 1676.95 | 1687.31 | 1688.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 10:15:00 | 1664.00 | 1682.65 | 1685.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 13:15:00 | 1660.05 | 1652.49 | 1663.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 14:00:00 | 1660.05 | 1652.49 | 1663.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 1685.10 | 1659.02 | 1665.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 1685.10 | 1659.02 | 1665.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1684.80 | 1664.17 | 1667.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 1667.10 | 1664.17 | 1667.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1623.80 | 1650.11 | 1657.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:00:00 | 1616.80 | 1630.09 | 1643.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 1593.20 | 1627.88 | 1641.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 14:15:00 | 1535.96 | 1554.69 | 1570.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 1513.54 | 1548.24 | 1564.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 09:15:00 | 1538.70 | 1537.52 | 1549.90 | SL hit (close>ema200) qty=0.50 sl=1537.52 alert=retest2 |

### Cycle 98 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 1601.95 | 1557.81 | 1554.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 1607.70 | 1581.35 | 1572.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 11:15:00 | 1642.50 | 1643.76 | 1617.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 12:00:00 | 1642.50 | 1643.76 | 1617.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1630.40 | 1640.29 | 1625.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:45:00 | 1627.80 | 1640.29 | 1625.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1678.60 | 1647.95 | 1630.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:00:00 | 1678.60 | 1647.95 | 1630.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1648.75 | 1653.34 | 1640.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 1645.00 | 1653.34 | 1640.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1632.65 | 1649.20 | 1639.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:00:00 | 1632.65 | 1649.20 | 1639.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1626.25 | 1644.61 | 1638.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 1626.25 | 1644.61 | 1638.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1628.35 | 1635.89 | 1635.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1628.35 | 1635.89 | 1635.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 1626.00 | 1633.91 | 1634.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 1617.55 | 1630.65 | 1633.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 1612.05 | 1611.63 | 1620.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 11:00:00 | 1612.05 | 1611.63 | 1620.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 1617.00 | 1612.47 | 1618.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:00:00 | 1617.00 | 1612.47 | 1618.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 1613.65 | 1612.70 | 1618.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 1539.15 | 1612.76 | 1617.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 15:15:00 | 1462.19 | 1494.72 | 1529.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-07 12:15:00 | 1495.70 | 1490.26 | 1515.55 | SL hit (close>ema200) qty=0.50 sl=1490.26 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 09:15:00 | 1576.70 | 1521.06 | 1517.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 1607.90 | 1538.43 | 1525.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 1580.90 | 1597.46 | 1573.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 13:00:00 | 1580.90 | 1597.46 | 1573.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1570.10 | 1591.98 | 1573.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1570.10 | 1591.98 | 1573.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1565.50 | 1586.69 | 1572.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 1565.50 | 1586.69 | 1572.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1569.95 | 1583.34 | 1572.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1538.05 | 1583.34 | 1572.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1542.00 | 1575.07 | 1569.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1532.40 | 1575.07 | 1569.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1534.85 | 1567.03 | 1566.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 1534.85 | 1567.03 | 1566.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 1541.00 | 1561.82 | 1564.08 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 1565.20 | 1560.31 | 1560.10 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 10:15:00 | 1557.85 | 1559.82 | 1559.90 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 1563.75 | 1560.61 | 1560.25 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 1556.85 | 1559.56 | 1559.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 1540.60 | 1555.64 | 1557.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 15:15:00 | 1545.00 | 1544.79 | 1550.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 09:15:00 | 1560.10 | 1544.79 | 1550.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1557.05 | 1547.24 | 1550.75 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 1568.05 | 1554.78 | 1553.20 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 1535.50 | 1550.92 | 1552.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 1526.50 | 1537.68 | 1544.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1542.85 | 1528.11 | 1534.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1542.85 | 1528.11 | 1534.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1542.85 | 1528.11 | 1534.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 1552.00 | 1528.11 | 1534.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1540.20 | 1530.53 | 1534.66 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 11:15:00 | 1590.40 | 1542.50 | 1539.73 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 1537.00 | 1546.95 | 1547.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 1527.85 | 1543.13 | 1545.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 12:15:00 | 1508.80 | 1506.81 | 1512.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 12:15:00 | 1508.80 | 1506.81 | 1512.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 1508.80 | 1506.81 | 1512.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:45:00 | 1510.45 | 1506.81 | 1512.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1508.05 | 1507.58 | 1511.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 1506.80 | 1507.58 | 1511.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:30:00 | 1505.60 | 1506.76 | 1510.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:15:00 | 1431.46 | 1485.83 | 1498.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 09:15:00 | 1430.32 | 1485.83 | 1498.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 14:15:00 | 1412.00 | 1408.59 | 1428.26 | SL hit (close>ema200) qty=0.50 sl=1408.59 alert=retest2 |

### Cycle 110 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 1413.50 | 1412.96 | 1412.93 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 1411.90 | 1412.75 | 1412.84 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 1420.65 | 1414.37 | 1413.56 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 1400.65 | 1411.24 | 1412.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 14:15:00 | 1399.60 | 1405.30 | 1407.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 1377.00 | 1374.90 | 1385.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 1367.65 | 1374.90 | 1385.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1378.70 | 1375.66 | 1384.57 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1399.10 | 1386.88 | 1386.66 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 13:15:00 | 1374.85 | 1385.30 | 1386.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 12:15:00 | 1372.75 | 1379.13 | 1382.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 1368.20 | 1365.99 | 1371.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 15:00:00 | 1368.20 | 1365.99 | 1371.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1349.80 | 1362.59 | 1369.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 11:15:00 | 1345.15 | 1359.98 | 1367.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 1344.05 | 1340.15 | 1351.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 12:15:00 | 1391.85 | 1353.68 | 1355.88 | SL hit (close>static) qty=1.00 sl=1369.70 alert=retest2 |

### Cycle 116 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 1380.55 | 1359.06 | 1358.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 15:15:00 | 1397.95 | 1371.79 | 1364.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 1366.15 | 1370.66 | 1364.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 1366.15 | 1370.66 | 1364.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1366.15 | 1370.66 | 1364.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 1362.90 | 1370.66 | 1364.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1382.90 | 1373.11 | 1366.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:30:00 | 1392.10 | 1380.16 | 1370.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 1359.05 | 1391.43 | 1391.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 1359.05 | 1391.43 | 1391.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 1350.85 | 1377.62 | 1385.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 1375.10 | 1371.46 | 1379.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 1375.10 | 1371.46 | 1379.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1375.10 | 1371.46 | 1379.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:45:00 | 1378.40 | 1371.46 | 1379.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1384.95 | 1374.16 | 1380.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 1386.90 | 1374.16 | 1380.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 1388.60 | 1377.05 | 1381.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:30:00 | 1388.95 | 1377.05 | 1381.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1383.00 | 1378.05 | 1380.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:45:00 | 1386.25 | 1378.05 | 1380.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1383.55 | 1379.15 | 1381.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:15:00 | 1373.00 | 1379.15 | 1381.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1373.00 | 1377.92 | 1380.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 1362.75 | 1377.92 | 1380.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1337.90 | 1369.92 | 1376.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 1331.10 | 1369.92 | 1376.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:30:00 | 1327.00 | 1332.98 | 1350.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 1324.10 | 1332.98 | 1350.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 11:15:00 | 1382.30 | 1351.81 | 1348.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1382.30 | 1351.81 | 1348.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 14:15:00 | 1401.15 | 1384.93 | 1372.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 1393.00 | 1400.79 | 1390.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 10:00:00 | 1393.00 | 1400.79 | 1390.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1390.35 | 1398.70 | 1390.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:00:00 | 1390.35 | 1398.70 | 1390.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 1392.00 | 1397.36 | 1390.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:15:00 | 1390.00 | 1397.36 | 1390.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 1392.90 | 1396.47 | 1390.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:45:00 | 1391.55 | 1396.47 | 1390.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1390.00 | 1395.17 | 1390.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:30:00 | 1391.15 | 1395.17 | 1390.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 1391.35 | 1394.41 | 1390.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1399.40 | 1393.94 | 1390.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 1386.75 | 1392.50 | 1390.54 | SL hit (close<static) qty=1.00 sl=1388.50 alert=retest2 |

### Cycle 119 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 1372.90 | 1390.05 | 1391.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1347.75 | 1369.93 | 1378.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 1289.75 | 1283.27 | 1305.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 1289.75 | 1283.27 | 1305.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1289.60 | 1253.71 | 1263.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 1289.60 | 1253.71 | 1263.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1307.20 | 1264.40 | 1267.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 1307.20 | 1264.40 | 1267.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 1334.60 | 1278.44 | 1273.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 10:15:00 | 1366.05 | 1350.92 | 1334.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1356.85 | 1378.99 | 1361.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1356.85 | 1378.99 | 1361.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1356.85 | 1378.99 | 1361.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1356.85 | 1378.99 | 1361.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1353.20 | 1373.83 | 1360.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1353.20 | 1373.83 | 1360.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1366.30 | 1372.33 | 1360.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 1373.40 | 1371.40 | 1362.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 1326.80 | 1363.29 | 1361.10 | SL hit (close<static) qty=1.00 sl=1348.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 1323.00 | 1355.23 | 1357.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 1307.50 | 1316.58 | 1324.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 1315.60 | 1314.19 | 1322.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:00:00 | 1315.60 | 1314.19 | 1322.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1282.00 | 1292.74 | 1303.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:15:00 | 1280.35 | 1290.54 | 1301.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 1271.05 | 1252.95 | 1252.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1271.05 | 1252.95 | 1252.90 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 1248.00 | 1253.11 | 1253.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 1235.95 | 1249.68 | 1251.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1235.05 | 1235.00 | 1241.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 1235.05 | 1235.00 | 1241.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1245.00 | 1235.11 | 1238.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 1267.50 | 1235.11 | 1238.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 1250.05 | 1241.31 | 1240.70 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 1232.20 | 1239.64 | 1240.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 11:15:00 | 1227.05 | 1237.12 | 1239.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 1239.40 | 1234.89 | 1236.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 1239.40 | 1234.89 | 1236.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1239.40 | 1234.89 | 1236.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:45:00 | 1252.80 | 1234.89 | 1236.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 10:15:00 | 1252.60 | 1238.43 | 1238.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 13:15:00 | 1263.00 | 1247.88 | 1243.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 11:15:00 | 1265.00 | 1265.28 | 1259.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:45:00 | 1263.50 | 1265.28 | 1259.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1272.75 | 1272.43 | 1265.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 1295.00 | 1281.33 | 1273.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 10:15:00 | 1312.60 | 1324.28 | 1325.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 1312.60 | 1324.28 | 1325.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 14:15:00 | 1309.30 | 1318.64 | 1322.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 12:15:00 | 1318.75 | 1313.81 | 1317.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 12:15:00 | 1318.75 | 1313.81 | 1317.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 1318.75 | 1313.81 | 1317.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 12:45:00 | 1319.05 | 1313.81 | 1317.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 1318.95 | 1314.84 | 1318.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:45:00 | 1318.00 | 1314.84 | 1318.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1316.00 | 1315.07 | 1317.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 1316.00 | 1315.07 | 1317.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 1313.80 | 1314.82 | 1317.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 1308.55 | 1314.82 | 1317.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 1243.12 | 1258.43 | 1270.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-23 12:15:00 | 1177.69 | 1197.19 | 1215.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 10:15:00 | 1234.70 | 1204.44 | 1202.66 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 1181.40 | 1206.99 | 1209.05 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 1213.50 | 1204.39 | 1203.48 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 1193.50 | 1202.04 | 1202.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 1189.45 | 1194.69 | 1198.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 1125.95 | 1120.38 | 1137.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 09:45:00 | 1127.90 | 1120.38 | 1137.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1076.35 | 1080.85 | 1097.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 1086.90 | 1080.85 | 1097.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1053.45 | 1055.47 | 1070.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 1052.00 | 1055.47 | 1070.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 10:00:00 | 1051.50 | 1055.63 | 1064.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:15:00 | 1051.00 | 1057.14 | 1063.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:15:00 | 1052.00 | 1054.91 | 1061.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1074.60 | 1058.39 | 1061.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 1074.60 | 1058.39 | 1061.67 | SL hit (close>static) qty=1.00 sl=1071.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1073.30 | 1063.92 | 1063.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 1077.45 | 1069.00 | 1066.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 1068.80 | 1069.60 | 1067.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 1068.80 | 1069.60 | 1067.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1068.80 | 1069.60 | 1067.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:15:00 | 1077.15 | 1068.75 | 1067.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 10:15:00 | 1065.15 | 1085.95 | 1087.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 1065.15 | 1085.95 | 1087.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1060.95 | 1080.95 | 1085.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1081.70 | 1074.17 | 1079.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1081.70 | 1074.17 | 1079.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1081.70 | 1074.17 | 1079.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1081.70 | 1074.17 | 1079.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1091.75 | 1077.68 | 1080.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 1089.30 | 1077.68 | 1080.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 1093.75 | 1082.85 | 1082.38 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 15:15:00 | 1075.60 | 1081.21 | 1081.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 10:15:00 | 1069.30 | 1079.45 | 1080.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1033.35 | 1021.96 | 1038.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 1033.35 | 1021.96 | 1038.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 1041.90 | 1025.95 | 1039.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:45:00 | 1035.70 | 1025.95 | 1039.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1034.70 | 1027.70 | 1038.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 1029.05 | 1028.41 | 1038.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 1063.50 | 1036.16 | 1039.91 | SL hit (close>static) qty=1.00 sl=1043.55 alert=retest2 |

### Cycle 136 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 1087.00 | 1044.05 | 1039.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 1090.85 | 1078.46 | 1071.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 1081.40 | 1083.76 | 1076.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 1081.40 | 1083.76 | 1076.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1079.00 | 1082.93 | 1079.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 1072.25 | 1082.93 | 1079.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1075.00 | 1081.35 | 1078.81 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1067.70 | 1076.08 | 1076.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1035.55 | 1066.29 | 1072.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1005.30 | 1002.82 | 1018.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 1004.85 | 1002.82 | 1018.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 965.80 | 962.33 | 970.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 957.00 | 961.26 | 968.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:00:00 | 954.25 | 952.76 | 960.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 974.30 | 958.29 | 960.06 | SL hit (close>static) qty=1.00 sl=971.55 alert=retest2 |

### Cycle 138 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 998.50 | 968.04 | 964.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 1025.70 | 979.57 | 969.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 1020.00 | 1024.67 | 1004.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 1020.00 | 1024.67 | 1004.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1008.00 | 1017.44 | 1009.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 13:00:00 | 1017.20 | 1014.34 | 1009.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 1025.15 | 1011.03 | 1009.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 09:45:00 | 1018.00 | 1033.13 | 1025.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 12:30:00 | 1017.80 | 1024.49 | 1023.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 13:15:00 | 1002.85 | 1020.16 | 1021.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 13:15:00 | 1002.85 | 1020.16 | 1021.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 14:15:00 | 998.85 | 1015.90 | 1019.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 964.15 | 955.80 | 972.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 964.15 | 955.80 | 972.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 964.15 | 955.80 | 972.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 964.15 | 955.80 | 972.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 969.75 | 958.59 | 971.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 969.75 | 958.59 | 971.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 968.55 | 960.58 | 971.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:15:00 | 961.85 | 962.47 | 971.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:30:00 | 962.90 | 965.21 | 971.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 962.20 | 965.21 | 971.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 997.85 | 971.25 | 972.90 | SL hit (close>static) qty=1.00 sl=972.60 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 988.20 | 974.64 | 974.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 1003.95 | 987.58 | 981.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 1000.00 | 1002.18 | 994.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 1004.80 | 1002.18 | 994.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1008.20 | 1003.39 | 995.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:15:00 | 1013.30 | 1005.08 | 996.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:00:00 | 1019.40 | 1005.58 | 1000.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 1015.80 | 1006.67 | 1001.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 12:45:00 | 1017.55 | 1009.79 | 1003.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1003.95 | 1008.93 | 1004.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 1003.70 | 1008.93 | 1004.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 994.00 | 1005.94 | 1003.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 984.90 | 1001.73 | 1001.73 | SL hit (close<static) qty=1.00 sl=992.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 990.25 | 999.44 | 1000.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 976.50 | 986.10 | 991.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 987.80 | 982.58 | 987.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 987.80 | 982.58 | 987.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 987.80 | 982.58 | 987.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 987.80 | 982.58 | 987.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 978.80 | 981.82 | 986.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:45:00 | 975.00 | 978.80 | 984.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 974.95 | 972.26 | 978.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 973.75 | 973.57 | 977.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 991.60 | 976.21 | 977.57 | SL hit (close>static) qty=1.00 sl=990.45 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 993.65 | 979.70 | 979.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 999.00 | 983.56 | 980.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1034.15 | 1036.20 | 1017.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 1034.15 | 1036.20 | 1017.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1045.60 | 1040.16 | 1028.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 1053.65 | 1040.16 | 1028.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 1040.00 | 1058.28 | 1058.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 1040.00 | 1058.28 | 1058.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 1034.80 | 1053.59 | 1056.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 13:15:00 | 1015.40 | 1014.08 | 1023.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 13:15:00 | 1015.40 | 1014.08 | 1023.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 1015.40 | 1014.08 | 1023.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 1019.30 | 1014.08 | 1023.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1020.35 | 1015.61 | 1020.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 1020.35 | 1015.61 | 1020.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1022.45 | 1016.98 | 1020.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 1022.45 | 1016.98 | 1020.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1027.60 | 1019.10 | 1021.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 1024.00 | 1019.10 | 1021.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1030.45 | 1021.37 | 1022.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1030.45 | 1021.37 | 1022.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 986.15 | 1009.91 | 1015.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 980.75 | 1009.91 | 1015.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 13:00:00 | 983.55 | 998.12 | 1008.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 936.25 | 994.44 | 1003.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 882.68 | 981.13 | 996.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 981.50 | 972.37 | 971.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 998.10 | 979.58 | 975.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 994.90 | 995.65 | 990.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 994.90 | 995.65 | 990.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 1019.60 | 1026.08 | 1021.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:30:00 | 1020.20 | 1026.08 | 1021.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 1017.70 | 1024.41 | 1021.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 1017.70 | 1024.41 | 1021.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 1008.20 | 1017.48 | 1018.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 09:15:00 | 1000.90 | 1009.59 | 1013.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 15:15:00 | 988.50 | 988.40 | 996.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 09:15:00 | 978.90 | 988.40 | 996.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 10:15:00 | 983.70 | 988.92 | 995.63 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 14:15:00 | 982.00 | 987.20 | 992.58 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 996.30 | 988.48 | 991.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 996.30 | 988.48 | 991.76 | SL hit (close>ema400) qty=1.00 sl=991.76 alert=retest1 |

### Cycle 146 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 987.10 | 964.19 | 961.53 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 944.45 | 959.10 | 960.11 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 987.75 | 961.46 | 959.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 996.00 | 968.37 | 963.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 1009.50 | 1011.98 | 1000.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 1008.60 | 1011.98 | 1000.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1035.00 | 1037.22 | 1031.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 1034.20 | 1037.22 | 1031.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1023.15 | 1033.91 | 1030.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1024.40 | 1033.91 | 1030.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1025.00 | 1032.13 | 1030.17 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1021.75 | 1028.73 | 1028.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1013.00 | 1025.58 | 1027.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 1015.20 | 1014.40 | 1019.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 1015.20 | 1014.40 | 1019.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1016.95 | 1014.68 | 1018.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:45:00 | 1008.85 | 1013.43 | 1016.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 1027.05 | 1017.92 | 1017.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1027.05 | 1017.92 | 1017.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1048.95 | 1026.97 | 1022.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1039.25 | 1042.18 | 1034.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 1039.25 | 1042.18 | 1034.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1039.00 | 1043.83 | 1037.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:30:00 | 1037.30 | 1043.83 | 1037.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1047.00 | 1044.47 | 1038.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1049.65 | 1045.88 | 1040.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 1056.50 | 1048.41 | 1042.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 1037.90 | 1052.43 | 1053.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 1037.90 | 1052.43 | 1053.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 1032.80 | 1045.84 | 1050.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 1009.20 | 1008.48 | 1017.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:30:00 | 1010.80 | 1008.48 | 1017.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1001.50 | 1001.67 | 1006.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1005.40 | 1001.67 | 1006.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1005.60 | 1001.85 | 1004.02 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 1013.00 | 1005.23 | 1005.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 10:15:00 | 1022.70 | 1016.03 | 1013.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1011.90 | 1015.68 | 1013.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1003.00 | 1013.14 | 1012.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1003.00 | 1013.14 | 1012.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1007.50 | 1012.01 | 1012.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 998.30 | 1006.90 | 1009.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 993.00 | 1004.17 | 1006.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 1000.50 | 1002.29 | 1005.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:45:00 | 1001.00 | 1002.15 | 1005.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 1001.20 | 1002.15 | 1005.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1004.60 | 1002.31 | 1004.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1004.60 | 1002.31 | 1004.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1004.40 | 1002.73 | 1004.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 998.00 | 1002.73 | 1004.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 977.20 | 973.51 | 973.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 977.20 | 973.51 | 973.37 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 970.20 | 973.31 | 973.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 961.20 | 970.89 | 972.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 974.90 | 970.14 | 971.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 967.20 | 969.55 | 970.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 966.00 | 969.55 | 970.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:45:00 | 966.40 | 969.40 | 970.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 15:00:00 | 963.40 | 968.20 | 970.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 965.80 | 966.45 | 968.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 960.60 | 962.47 | 965.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 959.90 | 961.98 | 964.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 959.05 | 958.90 | 961.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 958.65 | 958.90 | 961.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:30:00 | 960.00 | 959.76 | 961.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 975.70 | 962.84 | 962.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 975.70 | 962.84 | 962.46 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 960.20 | 963.74 | 964.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 957.05 | 962.40 | 963.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 966.50 | 961.46 | 962.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 964.00 | 961.96 | 962.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 973.85 | 961.96 | 962.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 973.05 | 964.18 | 963.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 977.65 | 971.37 | 967.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 977.85 | 981.14 | 976.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 977.85 | 981.14 | 976.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 981.30 | 981.17 | 977.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 977.25 | 981.17 | 977.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 975.80 | 980.01 | 977.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:15:00 | 972.30 | 980.01 | 977.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 978.45 | 979.70 | 977.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 15:00:00 | 978.80 | 979.52 | 977.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:45:00 | 980.35 | 979.75 | 978.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 982.00 | 982.93 | 983.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 982.00 | 982.93 | 983.02 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 987.60 | 983.86 | 983.44 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 978.00 | 983.10 | 983.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 977.45 | 980.73 | 982.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 985.50 | 980.20 | 981.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 990.10 | 982.18 | 982.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 1008.00 | 987.35 | 984.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 11:15:00 | 998.00 | 999.65 | 993.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:30:00 | 997.30 | 999.65 | 993.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1000.50 | 999.06 | 994.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 1002.20 | 999.91 | 996.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 995.10 | 1002.86 | 1003.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 995.10 | 1002.86 | 1003.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 993.45 | 999.84 | 1001.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 981.75 | 980.68 | 987.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 981.75 | 980.68 | 987.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 983.55 | 981.26 | 987.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 978.70 | 984.88 | 987.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 14:15:00 | 929.76 | 939.42 | 948.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 934.15 | 931.13 | 939.34 | SL hit (close>ema200) qty=0.50 sl=931.13 alert=retest2 |

### Cycle 164 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 949.00 | 939.25 | 938.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 950.70 | 941.54 | 939.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 959.20 | 954.95 | 951.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 958.95 | 955.75 | 952.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 947.65 | 951.69 | 951.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 947.65 | 951.69 | 951.70 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 957.65 | 952.89 | 952.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 960.10 | 954.33 | 952.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 955.80 | 956.46 | 954.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 955.80 | 956.46 | 954.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 955.00 | 956.17 | 954.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 961.45 | 956.17 | 954.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 976.50 | 985.84 | 986.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 976.50 | 985.84 | 986.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 971.75 | 980.95 | 983.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 970.85 | 970.55 | 975.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:45:00 | 972.10 | 970.55 | 975.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 982.20 | 972.88 | 976.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 982.20 | 972.88 | 976.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 988.20 | 975.94 | 977.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 987.00 | 975.94 | 977.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 990.45 | 980.29 | 978.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 1010.95 | 989.23 | 983.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1017.05 | 1018.27 | 1006.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 1017.05 | 1018.27 | 1006.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1009.50 | 1016.59 | 1008.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 1020.00 | 1015.03 | 1009.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:30:00 | 1020.80 | 1017.03 | 1011.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1000.50 | 1012.43 | 1012.23 | SL hit (close<static) qty=1.00 sl=1006.20 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1009.00 | 1011.74 | 1011.94 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1021.50 | 1013.69 | 1012.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 1036.00 | 1024.38 | 1019.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 1026.75 | 1028.37 | 1023.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 1026.75 | 1028.37 | 1023.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1023.15 | 1026.81 | 1024.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1023.15 | 1026.81 | 1024.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1022.50 | 1025.95 | 1024.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1028.05 | 1025.95 | 1024.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 1034.00 | 1036.40 | 1036.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 1034.00 | 1036.40 | 1036.65 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1048.00 | 1038.72 | 1037.68 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 1034.40 | 1037.34 | 1037.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 15:15:00 | 1029.00 | 1034.61 | 1036.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1035.00 | 1032.17 | 1034.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1032.80 | 1032.30 | 1034.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 1035.05 | 1032.30 | 1034.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1029.70 | 1031.65 | 1033.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 1027.35 | 1031.93 | 1033.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 1021.10 | 1029.76 | 1032.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 975.98 | 985.62 | 995.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 970.04 | 985.62 | 995.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 15:15:00 | 979.00 | 978.60 | 986.69 | SL hit (close>ema200) qty=0.50 sl=978.60 alert=retest2 |

### Cycle 174 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 991.05 | 987.50 | 987.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 994.20 | 989.63 | 988.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 993.00 | 995.64 | 993.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 993.20 | 995.15 | 993.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 996.15 | 997.49 | 994.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 994.20 | 1000.67 | 1001.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 994.20 | 1000.67 | 1001.29 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 1005.95 | 1001.06 | 1000.65 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 993.30 | 1000.35 | 1000.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 992.15 | 998.71 | 999.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 998.00 | 995.93 | 997.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 998.15 | 996.37 | 997.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 1000.50 | 996.37 | 997.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 991.95 | 995.49 | 997.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 991.25 | 993.71 | 996.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1000.60 | 997.26 | 996.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1000.60 | 997.26 | 996.88 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 994.60 | 996.42 | 996.61 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 997.60 | 996.48 | 996.40 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 993.80 | 995.94 | 996.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 991.40 | 995.03 | 995.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 994.95 | 994.56 | 995.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 995.95 | 994.84 | 995.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 995.95 | 994.84 | 995.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 995.00 | 994.87 | 995.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 995.00 | 994.87 | 995.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 993.20 | 994.54 | 995.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 992.55 | 994.54 | 995.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 988.60 | 993.35 | 994.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 986.60 | 993.35 | 994.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:15:00 | 987.05 | 991.14 | 993.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:45:00 | 987.15 | 989.74 | 992.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 996.85 | 993.79 | 993.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 996.85 | 993.79 | 993.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 15:15:00 | 1000.00 | 996.11 | 994.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:45:00 | 997.95 | 995.74 | 994.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:15:00 | 997.20 | 995.74 | 994.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 997.50 | 996.09 | 995.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1000.05 | 996.31 | 995.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1000.60 | 997.16 | 995.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1009.00 | 999.32 | 997.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:30:00 | 1005.90 | 1001.48 | 999.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1007.35 | 1002.70 | 1000.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:00:00 | 1008.65 | 1003.89 | 1001.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1009.80 | 1012.39 | 1009.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1009.80 | 1012.39 | 1009.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1002.45 | 1010.40 | 1008.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1002.45 | 1010.40 | 1008.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1003.55 | 1009.03 | 1008.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 1001.15 | 1009.03 | 1008.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1002.80 | 1009.22 | 1008.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 1002.80 | 1009.22 | 1008.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1006.00 | 1008.57 | 1008.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1004.60 | 1008.57 | 1008.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1007.20 | 1008.38 | 1008.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1007.60 | 1008.38 | 1008.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 1003.00 | 1007.31 | 1007.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 1003.00 | 1007.31 | 1007.89 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 1012.00 | 1008.42 | 1008.17 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 1001.20 | 1006.98 | 1007.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 1000.40 | 1005.66 | 1006.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 997.80 | 991.89 | 997.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 992.20 | 991.95 | 996.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 989.90 | 991.95 | 996.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 986.20 | 990.80 | 995.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 965.40 | 960.41 | 960.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 965.40 | 960.41 | 960.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 968.30 | 963.12 | 961.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 956.80 | 963.43 | 962.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 955.80 | 961.91 | 961.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 954.80 | 961.91 | 961.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 953.50 | 960.23 | 960.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 952.40 | 957.48 | 959.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 954.90 | 950.04 | 953.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 956.70 | 951.37 | 954.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 944.00 | 951.37 | 954.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:00:00 | 952.30 | 949.81 | 949.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 952.00 | 950.25 | 950.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 952.00 | 950.25 | 950.15 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 943.30 | 949.09 | 949.65 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 952.00 | 949.14 | 948.76 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 946.95 | 949.90 | 950.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 10:15:00 | 945.00 | 947.50 | 948.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 946.00 | 941.71 | 943.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 945.95 | 942.56 | 943.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 940.65 | 942.56 | 943.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 934.60 | 931.22 | 934.96 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 942.80 | 936.67 | 936.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 950.40 | 940.03 | 938.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 934.00 | 938.30 | 938.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 15:15:00 | 930.05 | 935.78 | 937.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 935.15 | 931.00 | 933.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 935.35 | 931.87 | 933.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 935.40 | 931.87 | 933.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 926.75 | 932.14 | 933.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:15:00 | 926.00 | 932.14 | 933.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:45:00 | 926.00 | 930.98 | 932.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 926.00 | 930.98 | 932.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 920.45 | 925.32 | 927.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 925.40 | 921.63 | 923.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 925.40 | 921.63 | 923.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 925.50 | 922.41 | 924.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 927.00 | 922.41 | 924.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 924.30 | 923.53 | 924.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 923.05 | 923.53 | 924.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:00:00 | 922.00 | 923.22 | 924.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 930.80 | 925.44 | 924.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 930.80 | 925.44 | 924.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 934.40 | 929.55 | 927.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 930.80 | 930.89 | 928.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 930.80 | 930.89 | 928.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 931.50 | 931.39 | 929.55 | EMA400 retest candle locked (from upside) |

### Cycle 195 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 925.75 | 928.81 | 928.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 924.40 | 927.93 | 928.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 902.80 | 900.77 | 907.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 906.65 | 901.94 | 907.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 906.65 | 901.94 | 907.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 906.65 | 901.94 | 907.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 909.00 | 904.05 | 907.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 909.00 | 904.05 | 907.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 909.20 | 905.08 | 907.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 909.20 | 905.08 | 907.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 909.30 | 905.93 | 907.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 909.30 | 905.93 | 907.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 906.55 | 906.68 | 907.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 906.55 | 906.68 | 907.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 906.10 | 906.56 | 907.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:45:00 | 907.15 | 906.56 | 907.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 928.00 | 910.85 | 909.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 932.75 | 921.13 | 915.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 926.85 | 932.69 | 926.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 926.85 | 932.69 | 926.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 924.30 | 931.01 | 926.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 924.30 | 931.01 | 926.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 930.80 | 930.97 | 926.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 924.20 | 930.97 | 926.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 927.00 | 930.17 | 926.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 925.90 | 930.17 | 926.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 925.30 | 929.20 | 926.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 926.15 | 929.20 | 926.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 921.65 | 927.69 | 925.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 921.65 | 927.69 | 925.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 924.75 | 927.10 | 925.87 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 919.50 | 924.83 | 925.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 915.95 | 921.77 | 923.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 919.95 | 919.19 | 921.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:45:00 | 919.75 | 919.19 | 921.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 921.95 | 919.74 | 921.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 920.80 | 919.74 | 921.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 926.80 | 921.15 | 921.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:45:00 | 927.15 | 921.15 | 921.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 921.00 | 921.12 | 921.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 919.65 | 921.12 | 921.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 919.25 | 920.50 | 921.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:45:00 | 919.25 | 918.59 | 920.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.67 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.29 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.29 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 879.35 | 876.06 | 882.45 | SL hit (close>ema200) qty=0.50 sl=876.06 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 844.60 | 835.19 | 834.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 851.45 | 840.41 | 837.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 837.90 | 841.28 | 838.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 837.65 | 840.55 | 838.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 841.10 | 840.55 | 838.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 828.65 | 838.17 | 837.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 828.65 | 838.17 | 837.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 829.00 | 836.34 | 836.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 826.95 | 834.46 | 835.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 847.65 | 830.79 | 829.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 857.65 | 840.23 | 835.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 877.05 | 866.07 | 863.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 867.90 | 889.84 | 892.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 867.90 | 889.84 | 892.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 864.85 | 884.84 | 889.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 843.50 | 839.84 | 849.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 843.50 | 839.84 | 849.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 839.80 | 841.84 | 846.01 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 854.95 | 848.65 | 847.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 11:15:00 | 857.25 | 850.95 | 849.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 861.00 | 854.01 | 851.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 856.45 | 858.98 | 856.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:00:00 | 856.55 | 858.29 | 857.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:45:00 | 856.95 | 858.08 | 857.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 859.00 | 858.27 | 857.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 849.20 | 855.80 | 856.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 849.20 | 855.80 | 856.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 844.75 | 852.13 | 853.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 853.15 | 851.16 | 852.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 855.85 | 852.10 | 852.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 855.85 | 852.10 | 852.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 842.80 | 850.24 | 852.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 858.50 | 850.24 | 852.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 807.80 | 806.48 | 810.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:30:00 | 811.00 | 806.48 | 810.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 790.00 | 788.78 | 793.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 800.45 | 788.78 | 793.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 802.00 | 791.42 | 794.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 802.50 | 791.42 | 794.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 799.35 | 793.01 | 794.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:30:00 | 799.00 | 793.01 | 794.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 793.35 | 794.23 | 795.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 795.75 | 794.23 | 795.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 792.95 | 793.98 | 794.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 793.70 | 793.98 | 794.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 802.15 | 794.31 | 794.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 802.15 | 794.31 | 794.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 794.25 | 794.30 | 794.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:45:00 | 790.90 | 793.49 | 794.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:30:00 | 784.70 | 790.19 | 792.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 785.25 | 778.50 | 778.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 785.25 | 778.50 | 778.41 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 775.60 | 779.20 | 779.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 773.25 | 778.01 | 779.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 743.75 | 734.47 | 742.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 740.60 | 735.69 | 742.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 733.25 | 735.55 | 741.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 734.15 | 735.55 | 741.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 696.59 | 703.88 | 717.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 697.44 | 703.88 | 717.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 708.70 | 688.98 | 700.90 | SL hit (close>ema200) qty=0.50 sl=688.98 alert=retest2 |

### Cycle 206 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 712.05 | 706.29 | 706.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 717.20 | 708.47 | 707.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 702.75 | 705.65 | 706.00 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 710.10 | 706.54 | 706.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 716.85 | 708.60 | 707.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 710.00 | 715.62 | 711.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 707.75 | 714.04 | 711.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 707.75 | 714.04 | 711.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 717.75 | 717.47 | 714.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 717.05 | 717.47 | 714.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 740.00 | 743.42 | 738.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 748.00 | 743.42 | 738.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:00:00 | 742.55 | 745.32 | 742.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 735.00 | 742.80 | 741.49 | SL hit (close<static) qty=1.00 sl=738.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 879.30 | 886.75 | 887.01 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 889.00 | 883.48 | 883.36 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 881.55 | 883.02 | 883.17 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 887.90 | 883.57 | 883.26 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 874.00 | 881.45 | 882.39 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 890.25 | 882.73 | 882.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 901.00 | 886.39 | 884.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 894.25 | 894.64 | 889.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 894.25 | 894.64 | 889.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 895.05 | 895.60 | 891.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 895.05 | 895.60 | 891.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 888.30 | 894.14 | 891.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 887.10 | 894.14 | 891.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 886.85 | 892.68 | 890.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 886.75 | 892.68 | 890.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 890.85 | 891.15 | 890.35 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-29 13:15:00 | 605.60 | 2023-05-31 10:15:00 | 614.40 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2023-05-29 14:15:00 | 605.55 | 2023-05-31 10:15:00 | 614.40 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-05-29 15:00:00 | 606.45 | 2023-05-31 10:15:00 | 614.40 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-05-31 09:15:00 | 605.45 | 2023-05-31 10:15:00 | 614.40 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2023-06-06 12:15:00 | 616.70 | 2023-06-13 15:15:00 | 621.30 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2023-06-06 14:15:00 | 618.10 | 2023-06-13 15:15:00 | 621.30 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2023-07-10 13:15:00 | 694.50 | 2023-07-11 09:15:00 | 686.65 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2023-07-10 15:00:00 | 694.15 | 2023-07-11 09:15:00 | 686.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-08-04 13:30:00 | 628.15 | 2023-08-17 10:15:00 | 620.90 | STOP_HIT | 1.00 | 1.15% |
| SELL | retest2 | 2023-08-07 10:15:00 | 629.20 | 2023-08-17 10:15:00 | 620.90 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2023-08-07 14:15:00 | 628.45 | 2023-08-17 10:15:00 | 620.90 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2023-08-21 12:15:00 | 626.65 | 2023-08-24 15:15:00 | 624.45 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-08-21 14:00:00 | 626.10 | 2023-08-24 15:15:00 | 624.45 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2023-08-24 15:00:00 | 625.90 | 2023-08-24 15:15:00 | 624.45 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2023-09-04 15:15:00 | 624.00 | 2023-09-05 09:15:00 | 648.20 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2023-09-08 11:45:00 | 670.65 | 2023-09-12 09:15:00 | 651.55 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2023-09-11 13:00:00 | 664.00 | 2023-09-12 09:15:00 | 651.55 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2023-09-11 13:45:00 | 663.75 | 2023-09-12 09:15:00 | 651.55 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2023-09-11 14:45:00 | 664.00 | 2023-09-12 09:15:00 | 651.55 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2023-09-14 13:30:00 | 647.00 | 2023-09-15 10:15:00 | 653.35 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-09-14 14:15:00 | 647.40 | 2023-09-15 10:15:00 | 653.35 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-09-14 15:15:00 | 647.30 | 2023-09-15 10:15:00 | 653.35 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-10-04 10:15:00 | 656.15 | 2023-10-05 13:15:00 | 642.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2023-10-06 11:15:00 | 641.90 | 2023-10-06 14:15:00 | 644.95 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-10-06 14:45:00 | 641.75 | 2023-10-06 15:15:00 | 645.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-10-09 09:15:00 | 632.10 | 2023-10-11 11:15:00 | 642.90 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2023-10-17 12:15:00 | 642.85 | 2023-10-17 12:15:00 | 647.30 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-11-01 10:30:00 | 620.05 | 2023-11-03 11:15:00 | 621.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2023-11-01 11:30:00 | 619.25 | 2023-11-03 11:15:00 | 621.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2023-11-02 09:30:00 | 619.35 | 2023-11-03 11:15:00 | 621.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-11-17 09:15:00 | 643.20 | 2023-11-17 14:15:00 | 639.25 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-11-20 09:15:00 | 645.70 | 2023-11-20 15:15:00 | 639.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-11-20 12:15:00 | 644.65 | 2023-11-20 15:15:00 | 639.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-11-20 12:45:00 | 644.10 | 2023-11-20 15:15:00 | 639.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-11-23 10:15:00 | 655.15 | 2023-12-01 09:15:00 | 720.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-23 11:00:00 | 654.10 | 2023-12-01 09:15:00 | 719.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-22 12:45:00 | 759.50 | 2023-12-26 09:15:00 | 774.55 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2023-12-27 09:15:00 | 775.15 | 2024-01-03 13:15:00 | 813.25 | STOP_HIT | 1.00 | 4.92% |
| BUY | retest2 | 2023-12-27 10:15:00 | 775.25 | 2024-01-03 13:15:00 | 813.25 | STOP_HIT | 1.00 | 4.90% |
| BUY | retest2 | 2024-01-09 09:15:00 | 832.75 | 2024-01-09 13:15:00 | 821.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-01-23 09:15:00 | 813.00 | 2024-01-23 11:15:00 | 796.70 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-02-06 09:15:00 | 903.50 | 2024-02-06 14:15:00 | 886.25 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-02-07 09:15:00 | 896.00 | 2024-02-07 13:15:00 | 886.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-02-07 09:45:00 | 896.85 | 2024-02-07 13:15:00 | 886.50 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-02-08 14:45:00 | 878.35 | 2024-02-13 09:15:00 | 834.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-09 09:30:00 | 869.85 | 2024-02-13 09:15:00 | 826.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-08 14:45:00 | 878.35 | 2024-02-13 13:15:00 | 848.65 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2024-02-09 09:30:00 | 869.85 | 2024-02-13 13:15:00 | 848.65 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2024-02-21 11:15:00 | 855.00 | 2024-02-23 14:15:00 | 856.15 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-02-21 12:00:00 | 845.35 | 2024-02-23 14:15:00 | 856.15 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-02-23 11:30:00 | 855.05 | 2024-02-23 14:15:00 | 856.15 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-02-29 09:15:00 | 827.05 | 2024-03-01 09:15:00 | 842.20 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-02-29 10:00:00 | 825.90 | 2024-03-01 09:15:00 | 842.20 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-03-06 09:15:00 | 853.40 | 2024-03-06 11:15:00 | 844.75 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-03-07 09:30:00 | 853.10 | 2024-03-12 12:15:00 | 848.75 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-03-12 12:00:00 | 855.80 | 2024-03-12 12:15:00 | 848.75 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-03-18 10:30:00 | 786.15 | 2024-03-20 10:15:00 | 746.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 12:15:00 | 789.00 | 2024-03-20 10:15:00 | 749.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 13:15:00 | 788.70 | 2024-03-20 10:15:00 | 749.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 14:15:00 | 789.95 | 2024-03-20 10:15:00 | 750.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 10:30:00 | 786.15 | 2024-03-21 11:15:00 | 758.30 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2024-03-18 12:15:00 | 789.00 | 2024-03-21 11:15:00 | 758.30 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2024-03-18 13:15:00 | 788.70 | 2024-03-21 11:15:00 | 758.30 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2024-03-18 14:15:00 | 789.95 | 2024-03-21 11:15:00 | 758.30 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2024-03-21 15:00:00 | 756.95 | 2024-03-22 10:15:00 | 772.75 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-03-27 09:15:00 | 782.85 | 2024-03-27 15:15:00 | 767.80 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-03-27 10:45:00 | 782.20 | 2024-03-27 15:15:00 | 767.80 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-04-15 11:15:00 | 885.40 | 2024-04-16 14:15:00 | 973.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-15 12:15:00 | 887.20 | 2024-04-16 14:15:00 | 975.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-16 09:45:00 | 888.40 | 2024-04-16 14:15:00 | 977.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-06 12:45:00 | 1125.40 | 2024-05-07 12:15:00 | 1094.95 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-05-06 13:30:00 | 1126.05 | 2024-05-07 12:15:00 | 1094.95 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-05-06 15:00:00 | 1123.00 | 2024-05-07 12:15:00 | 1094.95 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-05-07 09:45:00 | 1156.55 | 2024-05-07 12:15:00 | 1094.95 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2024-06-03 10:45:00 | 1193.05 | 2024-06-04 09:15:00 | 1133.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:45:00 | 1194.60 | 2024-06-04 09:15:00 | 1134.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 15:15:00 | 1194.00 | 2024-06-04 09:15:00 | 1134.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:45:00 | 1193.05 | 2024-06-04 10:15:00 | 1073.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 12:45:00 | 1194.60 | 2024-06-04 10:15:00 | 1075.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 15:15:00 | 1194.00 | 2024-06-04 10:15:00 | 1074.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-28 11:45:00 | 1690.00 | 2024-07-09 09:15:00 | 1676.95 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-07-12 15:00:00 | 1616.80 | 2024-07-19 14:15:00 | 1535.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 09:15:00 | 1593.20 | 2024-07-22 09:15:00 | 1513.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 15:00:00 | 1616.80 | 2024-07-23 09:15:00 | 1538.70 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2024-07-15 09:15:00 | 1593.20 | 2024-07-23 09:15:00 | 1538.70 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1539.15 | 2024-08-06 15:15:00 | 1462.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1539.15 | 2024-08-07 12:15:00 | 1495.70 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2024-09-04 09:15:00 | 1506.80 | 2024-09-05 09:15:00 | 1431.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 10:30:00 | 1505.60 | 2024-09-05 09:15:00 | 1430.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 09:15:00 | 1506.80 | 2024-09-09 14:15:00 | 1412.00 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2024-09-04 10:30:00 | 1505.60 | 2024-09-09 14:15:00 | 1412.00 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2024-09-26 11:15:00 | 1345.15 | 2024-09-27 12:15:00 | 1391.85 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-09-27 10:45:00 | 1344.05 | 2024-09-27 12:15:00 | 1391.85 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-09-30 12:30:00 | 1392.10 | 2024-10-03 11:15:00 | 1359.05 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-10-07 10:15:00 | 1331.10 | 2024-10-09 11:15:00 | 1382.30 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2024-10-08 09:30:00 | 1327.00 | 2024-10-09 11:15:00 | 1382.30 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2024-10-08 10:00:00 | 1324.10 | 2024-10-09 11:15:00 | 1382.30 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2024-10-15 09:15:00 | 1399.40 | 2024-10-15 09:15:00 | 1386.75 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-10-15 10:30:00 | 1395.25 | 2024-10-17 09:15:00 | 1372.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-10-15 11:30:00 | 1395.00 | 2024-10-17 09:15:00 | 1372.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-10-15 15:00:00 | 1393.40 | 2024-10-17 09:15:00 | 1372.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-11-04 14:00:00 | 1373.40 | 2024-11-05 09:15:00 | 1326.80 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-11-12 11:15:00 | 1280.35 | 2024-11-19 12:15:00 | 1271.05 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2024-12-02 15:15:00 | 1295.00 | 2024-12-10 10:15:00 | 1312.60 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2024-12-12 09:15:00 | 1308.55 | 2024-12-18 12:15:00 | 1243.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 1308.55 | 2024-12-23 12:15:00 | 1177.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-14 12:15:00 | 1052.00 | 2025-01-16 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-01-15 10:00:00 | 1051.50 | 2025-01-16 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-01-15 13:15:00 | 1051.00 | 2025-01-16 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-01-15 15:15:00 | 1052.00 | 2025-01-16 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-01-17 14:15:00 | 1077.15 | 2025-01-22 10:15:00 | 1065.15 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-28 14:45:00 | 1029.05 | 2025-01-29 09:15:00 | 1063.50 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-01-29 12:30:00 | 1032.85 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-01-29 13:00:00 | 1031.90 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-01-29 13:45:00 | 1032.00 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-01-30 14:15:00 | 1029.35 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -5.60% |
| SELL | retest2 | 2025-01-31 14:00:00 | 1023.25 | 2025-02-01 09:15:00 | 1087.00 | STOP_HIT | 1.00 | -6.23% |
| SELL | retest2 | 2025-02-18 11:00:00 | 957.00 | 2025-02-19 14:15:00 | 974.30 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-02-19 10:00:00 | 954.25 | 2025-02-19 14:15:00 | 974.30 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-02-24 13:00:00 | 1017.20 | 2025-02-27 13:15:00 | 1002.85 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-02-25 09:15:00 | 1025.15 | 2025-02-27 13:15:00 | 1002.85 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-02-27 09:45:00 | 1018.00 | 2025-02-27 13:15:00 | 1002.85 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-02-27 12:30:00 | 1017.80 | 2025-02-27 13:15:00 | 1002.85 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-03-04 13:15:00 | 961.85 | 2025-03-05 09:15:00 | 997.85 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-03-04 14:30:00 | 962.90 | 2025-03-05 09:15:00 | 997.85 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-03-04 15:15:00 | 962.20 | 2025-03-05 09:15:00 | 997.85 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-03-07 11:15:00 | 1013.30 | 2025-03-11 09:15:00 | 984.90 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-03-10 10:00:00 | 1019.40 | 2025-03-11 09:15:00 | 984.90 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-03-10 11:15:00 | 1015.80 | 2025-03-11 09:15:00 | 984.90 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-03-10 12:45:00 | 1017.55 | 2025-03-11 09:15:00 | 984.90 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-03-13 12:45:00 | 975.00 | 2025-03-18 09:15:00 | 991.60 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-03-17 11:00:00 | 974.95 | 2025-03-18 09:15:00 | 991.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-03-17 14:15:00 | 973.75 | 2025-03-18 09:15:00 | 991.60 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-03-21 10:15:00 | 1053.65 | 2025-03-26 14:15:00 | 1040.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-04-04 10:15:00 | 980.75 | 2025-04-07 09:15:00 | 882.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 13:00:00 | 983.55 | 2025-04-07 09:15:00 | 885.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 936.25 | 2025-04-07 09:15:00 | 842.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-11 09:45:00 | 979.95 | 2025-04-11 10:15:00 | 981.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-04-28 09:15:00 | 978.90 | 2025-04-29 09:15:00 | 996.30 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest1 | 2025-04-28 10:15:00 | 983.70 | 2025-04-29 09:15:00 | 996.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest1 | 2025-04-28 14:15:00 | 982.00 | 2025-04-29 09:15:00 | 996.30 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-04-29 14:15:00 | 988.30 | 2025-05-06 14:15:00 | 938.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 978.50 | 2025-05-06 14:15:00 | 929.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 14:15:00 | 988.30 | 2025-05-08 09:15:00 | 962.65 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2025-04-30 09:15:00 | 978.50 | 2025-05-08 09:15:00 | 962.65 | STOP_HIT | 0.50 | 1.62% |
| SELL | retest2 | 2025-05-22 13:45:00 | 1008.85 | 2025-05-23 11:15:00 | 1027.05 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-05-28 10:45:00 | 1049.65 | 2025-05-30 12:15:00 | 1037.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-28 11:30:00 | 1056.50 | 2025-05-30 12:15:00 | 1037.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-06-16 09:15:00 | 993.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-06-16 11:15:00 | 1000.50 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-06-16 11:45:00 | 1001.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2025-06-16 12:15:00 | 1001.20 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-06-17 09:15:00 | 998.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2025-06-30 13:15:00 | 966.00 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-30 13:45:00 | 966.40 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-30 15:00:00 | 963.40 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-01 09:45:00 | 965.80 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-02 14:15:00 | 959.90 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-03 11:30:00 | 959.05 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-07-03 12:00:00 | 958.65 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-03 14:30:00 | 960.00 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-07-11 15:00:00 | 978.80 | 2025-07-16 13:15:00 | 982.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-14 09:45:00 | 980.35 | 2025-07-16 13:15:00 | 982.00 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-07-23 10:30:00 | 1002.20 | 2025-07-25 11:15:00 | 995.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-30 15:15:00 | 978.70 | 2025-08-06 14:15:00 | 929.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 15:15:00 | 978.70 | 2025-08-07 13:15:00 | 934.15 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2025-08-13 13:45:00 | 959.20 | 2025-08-18 09:15:00 | 947.65 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-08-13 15:00:00 | 958.95 | 2025-08-18 09:15:00 | 947.65 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-19 09:15:00 | 961.45 | 2025-08-26 09:15:00 | 976.50 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-09-03 11:45:00 | 1020.00 | 2025-09-04 14:15:00 | 1000.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-09-03 14:30:00 | 1020.80 | 2025-09-04 14:15:00 | 1000.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1028.05 | 2025-09-15 15:15:00 | 1034.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-09-19 13:45:00 | 1027.35 | 2025-09-26 09:15:00 | 975.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1021.10 | 2025-09-26 09:15:00 | 970.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 13:45:00 | 1027.35 | 2025-09-26 15:15:00 | 979.00 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1021.10 | 2025-09-26 15:15:00 | 979.00 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2025-10-06 12:30:00 | 996.15 | 2025-10-09 11:15:00 | 994.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-10-14 10:30:00 | 991.25 | 2025-10-15 10:15:00 | 1000.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-20 10:15:00 | 986.60 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-20 13:15:00 | 987.05 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-20 14:45:00 | 987.15 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-24 12:45:00 | 997.95 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-10-24 13:15:00 | 997.20 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-10-24 14:00:00 | 997.50 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1000.05 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1009.00 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-28 12:30:00 | 1005.90 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1007.35 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-10-29 10:00:00 | 1008.65 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-11-06 14:15:00 | 989.90 | 2025-11-20 10:15:00 | 965.40 | STOP_HIT | 1.00 | 2.47% |
| SELL | retest2 | 2025-11-06 15:00:00 | 986.20 | 2025-11-20 10:15:00 | 965.40 | STOP_HIT | 1.00 | 2.11% |
| SELL | retest2 | 2025-11-25 09:15:00 | 944.00 | 2025-11-27 10:15:00 | 952.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-27 10:00:00 | 952.30 | 2025-11-27 10:15:00 | 952.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-12-16 11:15:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-16 11:45:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-16 12:15:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-18 09:15:00 | 920.45 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-19 11:15:00 | 923.05 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-19 12:00:00 | 922.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-08 11:15:00 | 919.65 | 2026-01-14 10:15:00 | 873.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 919.25 | 2026-01-14 10:15:00 | 873.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:45:00 | 919.25 | 2026-01-14 10:15:00 | 873.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 919.65 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2026-01-08 11:45:00 | 919.25 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2026-01-08 14:45:00 | 919.25 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.34% |
| BUY | retest2 | 2026-02-09 09:15:00 | 877.05 | 2026-02-12 11:15:00 | 867.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-23 09:15:00 | 861.00 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-24 10:15:00 | 856.45 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-24 13:00:00 | 856.55 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-24 13:45:00 | 856.95 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-03-12 14:45:00 | 790.90 | 2026-03-18 09:15:00 | 785.25 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2026-03-13 09:30:00 | 784.70 | 2026-03-18 09:15:00 | 785.25 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2026-03-25 11:30:00 | 733.25 | 2026-03-30 09:15:00 | 696.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 734.15 | 2026-03-30 09:15:00 | 697.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 733.25 | 2026-04-01 09:15:00 | 708.70 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-03-25 12:15:00 | 734.15 | 2026-04-01 09:15:00 | 708.70 | STOP_HIT | 0.50 | 3.47% |
| BUY | retest2 | 2026-04-10 09:15:00 | 748.00 | 2026-04-13 09:15:00 | 735.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-04-10 15:00:00 | 742.55 | 2026-04-13 09:15:00 | 735.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-04-13 10:45:00 | 747.85 | 2026-04-22 09:15:00 | 816.15 | TARGET_HIT | 1.00 | 9.13% |
| BUY | retest2 | 2026-04-13 13:30:00 | 741.95 | 2026-04-22 10:15:00 | 822.64 | TARGET_HIT | 1.00 | 10.87% |
| BUY | retest2 | 2026-04-15 09:15:00 | 755.75 | 2026-04-22 10:15:00 | 831.33 | TARGET_HIT | 1.00 | 10.00% |
