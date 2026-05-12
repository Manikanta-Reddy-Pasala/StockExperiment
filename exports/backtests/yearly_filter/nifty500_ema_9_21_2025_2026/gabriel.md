# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1136.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 82 |
| ALERT1 | 54 |
| ALERT2 | 51 |
| ALERT2_SKIP | 33 |
| ALERT3 | 124 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 44 |
| PARTIAL | 3 |
| TARGET_HIT | 8 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 15 / 32
- **Target hits / Stop hits / Partials:** 6 / 38 / 3
- **Avg / median % per leg:** 0.03% / -1.28%
- **Sum % (uncompounded):** 1.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 7 | 29.2% | 6 | 18 | 0 | 0.87% | 20.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.87% | -11.6% |
| BUY @ 3rd Alert (retest2) | 21 | 7 | 33.3% | 6 | 15 | 0 | 1.54% | 32.4% |
| SELL (all) | 23 | 8 | 34.8% | 0 | 20 | 3 | -0.84% | -19.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 8 | 34.8% | 0 | 20 | 3 | -0.84% | -19.3% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.87% | -11.6% |
| retest2 (combined) | 44 | 15 | 34.1% | 6 | 35 | 3 | 0.30% | 13.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 616.50 | 625.89 | 625.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 13:15:00 | 614.05 | 621.85 | 623.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 634.40 | 621.79 | 623.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 634.40 | 621.79 | 623.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 634.40 | 621.79 | 623.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 634.40 | 621.79 | 623.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 632.55 | 623.94 | 624.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 629.50 | 623.94 | 624.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 629.75 | 625.10 | 624.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 636.00 | 627.28 | 625.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 631.70 | 631.82 | 629.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 12:15:00 | 631.70 | 631.82 | 629.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 631.70 | 631.82 | 629.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 629.10 | 631.82 | 629.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 628.05 | 631.07 | 629.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 628.05 | 631.07 | 629.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 622.10 | 629.27 | 628.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 622.10 | 629.27 | 628.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 622.85 | 627.99 | 628.04 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 679.35 | 638.26 | 632.71 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 624.80 | 639.39 | 640.74 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 651.50 | 642.00 | 641.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 15:15:00 | 655.95 | 646.50 | 643.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 648.05 | 648.05 | 645.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 14:00:00 | 648.05 | 648.05 | 645.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 643.30 | 647.39 | 645.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 641.70 | 647.39 | 645.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 646.30 | 647.17 | 645.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 642.10 | 647.17 | 645.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 647.60 | 647.26 | 646.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:30:00 | 646.40 | 647.26 | 646.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 646.65 | 647.14 | 646.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 647.05 | 647.14 | 646.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 646.40 | 647.00 | 646.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:45:00 | 644.10 | 647.00 | 646.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 647.95 | 647.19 | 646.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 652.00 | 647.19 | 646.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 642.10 | 645.77 | 646.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 642.10 | 645.77 | 646.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 640.30 | 644.68 | 645.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 647.50 | 644.41 | 645.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 647.50 | 644.41 | 645.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 647.50 | 644.41 | 645.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 647.50 | 644.41 | 645.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 651.20 | 645.77 | 645.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 661.00 | 650.41 | 648.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 648.75 | 655.66 | 653.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 648.75 | 655.66 | 653.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 648.75 | 655.66 | 653.27 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 650.20 | 651.94 | 652.10 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 657.55 | 652.78 | 652.44 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 650.10 | 652.08 | 652.32 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 654.95 | 652.65 | 652.56 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 645.75 | 651.54 | 652.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 638.65 | 648.97 | 650.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 636.35 | 634.49 | 638.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 636.35 | 634.49 | 638.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 636.35 | 634.49 | 638.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 636.35 | 634.49 | 638.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 643.30 | 636.25 | 638.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 643.00 | 636.25 | 638.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 644.00 | 637.80 | 639.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:30:00 | 639.10 | 638.21 | 639.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 644.45 | 640.01 | 639.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 14:15:00 | 644.45 | 640.01 | 639.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 646.15 | 642.04 | 640.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 642.30 | 642.53 | 641.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 642.30 | 642.53 | 641.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 636.25 | 641.28 | 641.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 636.60 | 641.28 | 641.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 639.70 | 640.96 | 640.89 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 639.00 | 640.57 | 640.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 635.95 | 639.65 | 640.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 626.80 | 607.06 | 610.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 626.80 | 607.06 | 610.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 626.80 | 607.06 | 610.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 626.80 | 607.06 | 610.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 611.50 | 607.94 | 610.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 608.55 | 608.34 | 610.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 604.95 | 607.48 | 609.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 603.85 | 597.85 | 597.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 603.85 | 597.85 | 597.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 621.45 | 604.63 | 600.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 929.90 | 954.41 | 880.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:45:00 | 953.60 | 942.59 | 908.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 15:15:00 | 955.00 | 942.05 | 920.66 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 917.20 | 936.66 | 923.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 917.20 | 936.66 | 923.62 | SL hit (close<ema400) qty=1.00 sl=923.62 alert=retest1 |

### Cycle 17 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 1043.00 | 1076.75 | 1078.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 1031.00 | 1046.06 | 1058.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1038.85 | 996.17 | 1008.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1038.85 | 996.17 | 1008.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1038.85 | 996.17 | 1008.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1033.00 | 996.17 | 1008.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1030.90 | 1003.12 | 1010.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1018.10 | 1010.35 | 1012.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 967.19 | 985.79 | 991.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 990.70 | 985.70 | 990.02 | SL hit (close>ema200) qty=0.50 sl=985.70 alert=retest2 |

### Cycle 18 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 1022.00 | 992.96 | 992.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 1039.05 | 1018.60 | 1007.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1024.55 | 1030.35 | 1017.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1024.55 | 1030.35 | 1017.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1024.55 | 1030.35 | 1017.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1041.00 | 1030.72 | 1023.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:30:00 | 1043.00 | 1030.35 | 1025.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 999.80 | 1020.90 | 1021.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 999.80 | 1020.90 | 1021.94 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 1024.00 | 1022.80 | 1022.70 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 11:15:00 | 1011.10 | 1020.46 | 1021.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 13:15:00 | 1005.00 | 1016.11 | 1019.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1021.30 | 1017.15 | 1019.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 1021.30 | 1017.15 | 1019.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1021.30 | 1017.15 | 1019.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 1022.40 | 1017.15 | 1019.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1028.80 | 1019.48 | 1020.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1033.10 | 1019.48 | 1020.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1005.60 | 1006.33 | 1011.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 992.50 | 1003.84 | 1007.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 999.00 | 1006.61 | 1007.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 1020.40 | 1009.91 | 1009.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 1020.40 | 1009.91 | 1009.04 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 995.60 | 1006.49 | 1007.81 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1049.70 | 1016.51 | 1012.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 1051.90 | 1023.59 | 1015.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1082.00 | 1089.35 | 1068.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1082.00 | 1089.35 | 1068.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1073.00 | 1083.32 | 1073.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 1073.00 | 1083.32 | 1073.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1075.00 | 1081.66 | 1073.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1146.70 | 1081.66 | 1073.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1141.60 | 1177.82 | 1178.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 1141.60 | 1177.82 | 1178.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 1135.60 | 1162.72 | 1171.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1140.40 | 1128.91 | 1142.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 1140.40 | 1128.91 | 1142.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1140.40 | 1128.91 | 1142.14 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1154.90 | 1143.09 | 1142.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 1180.00 | 1154.16 | 1148.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 09:15:00 | 1220.20 | 1232.58 | 1210.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:45:00 | 1227.00 | 1232.58 | 1210.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1223.00 | 1231.20 | 1219.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1241.90 | 1231.20 | 1219.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 1226.90 | 1236.99 | 1229.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 1211.20 | 1230.38 | 1227.94 | SL hit (close<static) qty=1.00 sl=1219.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 13:15:00 | 1217.00 | 1225.60 | 1226.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 1203.80 | 1221.24 | 1224.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1229.00 | 1219.39 | 1222.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1229.00 | 1219.39 | 1222.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1229.00 | 1219.39 | 1222.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1229.00 | 1219.39 | 1222.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1219.90 | 1219.49 | 1222.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:00:00 | 1215.00 | 1218.60 | 1221.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 1265.90 | 1230.48 | 1226.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 1265.90 | 1230.48 | 1226.46 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 1234.00 | 1243.70 | 1244.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 1229.00 | 1240.76 | 1243.47 | Break + close below crossover candle low |

### Cycle 30 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1284.20 | 1245.43 | 1244.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1296.90 | 1268.40 | 1256.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 1309.50 | 1311.12 | 1294.11 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 13:15:00 | 1321.80 | 1311.41 | 1298.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1271.10 | 1303.33 | 1298.85 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1271.10 | 1303.33 | 1298.85 | SL hit (close<ema400) qty=1.00 sl=1298.85 alert=retest1 |

### Cycle 31 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 1256.60 | 1293.98 | 1295.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 1243.90 | 1283.97 | 1290.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 1256.20 | 1248.64 | 1263.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 13:15:00 | 1256.20 | 1248.64 | 1263.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1256.20 | 1248.64 | 1263.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 1261.00 | 1248.64 | 1263.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1277.40 | 1254.39 | 1264.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1277.40 | 1254.39 | 1264.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1300.10 | 1263.53 | 1267.95 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1300.60 | 1270.94 | 1270.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 1310.00 | 1278.76 | 1274.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 1275.00 | 1285.52 | 1280.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 15:15:00 | 1275.00 | 1285.52 | 1280.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1275.00 | 1285.52 | 1280.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1301.50 | 1285.52 | 1280.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 1290.00 | 1293.65 | 1288.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:30:00 | 1293.90 | 1293.24 | 1288.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1278.60 | 1285.54 | 1286.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 1278.60 | 1285.54 | 1286.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 1269.70 | 1280.90 | 1283.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 1270.00 | 1258.92 | 1268.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 1270.00 | 1258.92 | 1268.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1270.00 | 1258.92 | 1268.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 1270.00 | 1258.92 | 1268.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1281.90 | 1263.52 | 1269.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 1286.20 | 1263.52 | 1269.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 1282.00 | 1267.22 | 1271.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 1278.20 | 1267.22 | 1271.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1235.20 | 1207.82 | 1223.00 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1244.60 | 1230.78 | 1229.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1258.10 | 1236.24 | 1231.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 1270.70 | 1271.22 | 1260.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:30:00 | 1269.70 | 1271.22 | 1260.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1298.00 | 1313.07 | 1306.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1298.00 | 1313.07 | 1306.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1290.60 | 1308.58 | 1305.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:30:00 | 1293.90 | 1308.58 | 1305.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 1284.70 | 1301.15 | 1302.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 10:15:00 | 1276.60 | 1296.24 | 1299.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1240.70 | 1240.44 | 1253.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 1240.70 | 1240.44 | 1253.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1259.80 | 1244.60 | 1252.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 1259.80 | 1244.60 | 1252.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1296.90 | 1255.06 | 1256.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1296.90 | 1255.06 | 1256.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 1288.00 | 1261.65 | 1259.10 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 1233.20 | 1263.51 | 1263.63 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1273.60 | 1261.03 | 1260.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 1296.00 | 1269.78 | 1264.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 1273.90 | 1275.91 | 1268.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 1273.90 | 1275.91 | 1268.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1273.90 | 1275.91 | 1268.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1273.90 | 1275.91 | 1268.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1271.00 | 1274.93 | 1269.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1271.00 | 1274.93 | 1269.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1266.90 | 1273.32 | 1268.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 1266.90 | 1273.32 | 1268.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1266.30 | 1271.92 | 1268.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 1263.30 | 1271.92 | 1268.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1263.60 | 1270.25 | 1268.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 1262.50 | 1270.25 | 1268.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1245.50 | 1265.74 | 1266.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 1245.00 | 1258.40 | 1262.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 1245.60 | 1245.14 | 1251.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:30:00 | 1245.20 | 1245.14 | 1251.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1251.00 | 1246.37 | 1251.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 1249.50 | 1246.37 | 1251.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1260.00 | 1249.09 | 1252.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 1260.00 | 1249.09 | 1252.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1240.00 | 1247.28 | 1250.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 1237.80 | 1247.28 | 1250.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1263.00 | 1248.64 | 1250.43 | SL hit (close>static) qty=1.00 sl=1260.20 alert=retest2 |

### Cycle 40 — BUY (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 12:15:00 | 1261.90 | 1252.93 | 1252.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 1284.30 | 1259.95 | 1255.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1264.00 | 1280.09 | 1272.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1264.00 | 1280.09 | 1272.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1264.00 | 1280.09 | 1272.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1264.00 | 1280.09 | 1272.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1255.50 | 1275.18 | 1270.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 1255.50 | 1275.18 | 1270.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 1257.80 | 1267.85 | 1268.10 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1302.80 | 1273.02 | 1270.18 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 1267.00 | 1279.79 | 1281.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1245.80 | 1267.25 | 1274.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1235.90 | 1234.57 | 1246.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 1238.00 | 1234.57 | 1246.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1247.00 | 1237.05 | 1246.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1247.00 | 1237.05 | 1246.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1236.00 | 1236.84 | 1245.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1228.90 | 1236.84 | 1245.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1228.10 | 1235.09 | 1244.11 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 1266.00 | 1248.79 | 1247.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 1272.20 | 1255.92 | 1251.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 15:15:00 | 1259.00 | 1259.84 | 1254.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:15:00 | 1271.00 | 1259.84 | 1254.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1262.60 | 1260.39 | 1255.47 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 1233.20 | 1253.97 | 1255.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 1230.00 | 1245.92 | 1251.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 1028.00 | 1000.28 | 1050.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:00:00 | 1028.00 | 1000.28 | 1050.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1062.30 | 1012.68 | 1051.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 1061.90 | 1012.68 | 1051.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1064.90 | 1023.13 | 1052.59 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 1105.00 | 1065.58 | 1065.26 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 1052.40 | 1065.99 | 1066.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1026.00 | 1055.91 | 1061.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 1066.70 | 1052.84 | 1058.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 11:15:00 | 1066.70 | 1052.84 | 1058.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1066.70 | 1052.84 | 1058.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 1066.70 | 1052.84 | 1058.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1073.30 | 1056.93 | 1060.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 1091.50 | 1056.93 | 1060.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1046.10 | 1038.76 | 1046.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:30:00 | 1047.90 | 1038.76 | 1046.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1045.40 | 1040.09 | 1046.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:30:00 | 1046.90 | 1040.09 | 1046.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1056.20 | 1043.31 | 1047.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1056.20 | 1043.31 | 1047.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1044.00 | 1043.45 | 1047.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1032.00 | 1043.45 | 1047.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:00:00 | 1031.90 | 1035.13 | 1041.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 1055.50 | 1042.93 | 1042.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1055.50 | 1042.93 | 1042.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1061.60 | 1048.61 | 1045.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 1047.70 | 1050.57 | 1047.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1047.70 | 1050.57 | 1047.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1047.70 | 1050.57 | 1047.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 1050.40 | 1050.57 | 1047.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1050.50 | 1050.55 | 1047.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:15:00 | 1048.60 | 1050.55 | 1047.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1054.60 | 1051.36 | 1048.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 1046.60 | 1051.36 | 1048.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1040.50 | 1049.19 | 1047.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 1040.50 | 1049.19 | 1047.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1045.50 | 1048.45 | 1047.29 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 1034.00 | 1044.08 | 1045.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 1025.30 | 1040.33 | 1043.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 1058.10 | 1036.92 | 1038.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1058.10 | 1036.92 | 1038.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1058.10 | 1036.92 | 1038.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 1059.70 | 1036.92 | 1038.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1051.70 | 1039.88 | 1040.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 1054.30 | 1039.88 | 1040.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1050.20 | 1041.94 | 1041.02 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1025.90 | 1039.90 | 1040.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1002.70 | 1025.53 | 1032.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 988.30 | 984.02 | 998.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 995.00 | 986.21 | 997.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 995.00 | 986.21 | 997.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 996.70 | 986.21 | 997.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 942.90 | 935.21 | 948.38 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 983.80 | 958.41 | 956.37 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 958.00 | 958.71 | 958.73 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 961.60 | 959.17 | 958.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 963.00 | 959.94 | 959.30 | Break + close above crossover candle high |

### Cycle 55 — SELL (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 09:15:00 | 954.50 | 958.85 | 958.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 948.50 | 956.78 | 957.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 974.10 | 954.67 | 955.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 974.10 | 954.67 | 955.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 974.10 | 954.67 | 955.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 974.10 | 954.67 | 955.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 956.00 | 954.93 | 955.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 948.90 | 954.93 | 955.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 963.10 | 956.77 | 956.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 963.10 | 956.77 | 956.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 10:15:00 | 976.90 | 964.15 | 960.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 959.90 | 968.76 | 965.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 959.90 | 968.76 | 965.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 959.90 | 968.76 | 965.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 959.90 | 968.76 | 965.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 956.60 | 966.33 | 964.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 956.60 | 966.33 | 964.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 953.00 | 961.80 | 962.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 945.10 | 956.80 | 959.84 | Break + close below crossover candle low |

### Cycle 58 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 982.50 | 961.94 | 961.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 11:15:00 | 1022.50 | 974.05 | 967.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 11:15:00 | 1039.70 | 1040.31 | 1013.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 1039.70 | 1040.31 | 1013.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1041.30 | 1048.98 | 1041.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 1039.20 | 1048.98 | 1041.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1038.90 | 1046.97 | 1041.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1038.90 | 1046.97 | 1041.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1040.20 | 1045.61 | 1040.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 1051.40 | 1049.69 | 1043.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:45:00 | 1045.20 | 1048.24 | 1045.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1042.40 | 1046.23 | 1045.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 1032.30 | 1043.45 | 1044.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1032.30 | 1043.45 | 1044.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1027.50 | 1037.40 | 1040.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1039.20 | 1037.06 | 1040.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1039.20 | 1037.06 | 1040.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1039.20 | 1037.06 | 1040.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1046.60 | 1037.06 | 1040.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1036.00 | 1036.85 | 1039.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 1041.90 | 1036.85 | 1039.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1007.10 | 1016.48 | 1026.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 998.60 | 1009.17 | 1020.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:00:00 | 993.00 | 1002.64 | 1013.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 996.00 | 1002.35 | 1012.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 998.70 | 1001.62 | 1011.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1008.50 | 1002.90 | 1010.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 1008.50 | 1002.90 | 1010.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1008.90 | 1004.10 | 1009.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 1008.90 | 1004.10 | 1009.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1006.70 | 1004.62 | 1009.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 1050.70 | 1004.62 | 1009.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1037.90 | 1011.27 | 1012.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 1037.90 | 1011.27 | 1012.26 | SL hit (close>static) qty=1.00 sl=1030.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1038.90 | 1016.80 | 1014.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1052.10 | 1035.60 | 1026.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1058.00 | 1061.13 | 1049.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 1058.00 | 1061.13 | 1049.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1048.50 | 1057.67 | 1050.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1047.70 | 1057.67 | 1050.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1043.50 | 1054.84 | 1050.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 1043.50 | 1054.84 | 1050.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 1040.00 | 1046.97 | 1047.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 1032.70 | 1044.12 | 1046.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 958.90 | 950.73 | 967.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 958.90 | 950.73 | 967.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 927.00 | 923.80 | 932.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 914.80 | 924.39 | 930.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 869.06 | 898.87 | 912.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 899.90 | 893.71 | 902.88 | SL hit (close>ema200) qty=0.50 sl=893.71 alert=retest2 |

### Cycle 62 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 920.50 | 908.47 | 907.08 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 900.00 | 906.08 | 906.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 890.70 | 903.01 | 904.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 14:15:00 | 900.90 | 900.50 | 903.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 900.90 | 900.50 | 903.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 900.90 | 900.50 | 903.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 900.90 | 900.50 | 903.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 893.60 | 888.49 | 893.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 897.30 | 888.49 | 893.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 893.30 | 889.45 | 893.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:30:00 | 883.00 | 890.21 | 892.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 896.90 | 891.54 | 893.21 | SL hit (close>static) qty=1.00 sl=894.80 alert=retest2 |

### Cycle 64 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 941.70 | 895.24 | 891.70 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 900.60 | 911.89 | 912.87 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 926.70 | 912.89 | 912.87 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 13:15:00 | 889.40 | 913.08 | 913.90 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 946.80 | 916.27 | 914.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 969.35 | 926.89 | 919.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 969.30 | 976.27 | 960.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 969.30 | 976.27 | 960.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 969.30 | 976.27 | 960.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:30:00 | 991.45 | 971.63 | 965.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 1003.05 | 1006.81 | 1007.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 1003.05 | 1006.81 | 1007.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 996.05 | 1004.66 | 1006.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 11:15:00 | 972.80 | 972.57 | 982.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:45:00 | 974.05 | 972.57 | 982.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 979.00 | 970.89 | 974.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 979.00 | 970.89 | 974.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 994.95 | 975.70 | 976.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 994.95 | 975.70 | 976.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 975.65 | 975.69 | 976.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 969.00 | 974.66 | 975.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 970.00 | 964.90 | 964.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 970.00 | 964.90 | 964.21 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 959.25 | 963.43 | 963.91 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 969.15 | 963.68 | 963.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 983.75 | 967.70 | 965.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 14:15:00 | 1008.75 | 1011.56 | 997.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 15:00:00 | 1008.75 | 1011.56 | 997.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1010.65 | 1010.82 | 999.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:30:00 | 1017.00 | 1011.32 | 1001.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 977.85 | 997.00 | 998.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 977.85 | 997.00 | 998.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 956.50 | 988.90 | 994.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 935.00 | 930.20 | 943.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 935.00 | 930.20 | 943.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 935.00 | 930.20 | 943.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 941.95 | 930.20 | 943.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 946.30 | 934.99 | 943.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 946.60 | 934.99 | 943.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 922.35 | 932.46 | 941.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 919.80 | 929.14 | 937.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 873.81 | 913.63 | 928.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 887.50 | 884.06 | 901.11 | SL hit (close>ema200) qty=0.50 sl=884.06 alert=retest2 |

### Cycle 74 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 867.60 | 854.09 | 853.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 876.60 | 862.51 | 857.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 852.95 | 862.93 | 859.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 852.95 | 862.93 | 859.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 852.95 | 862.93 | 859.38 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 844.25 | 856.33 | 856.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 837.00 | 850.39 | 853.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 849.80 | 846.70 | 851.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 849.80 | 846.70 | 851.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 849.80 | 846.70 | 851.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 850.75 | 846.70 | 851.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 851.10 | 847.58 | 851.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 850.15 | 847.58 | 851.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 857.30 | 849.52 | 851.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 857.30 | 849.52 | 851.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 868.00 | 853.22 | 853.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 870.00 | 856.58 | 854.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 838.65 | 854.86 | 854.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 838.65 | 854.86 | 854.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 838.65 | 854.86 | 854.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 838.65 | 854.86 | 854.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 837.15 | 851.32 | 852.92 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 862.00 | 848.05 | 847.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 908.75 | 863.59 | 855.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 858.00 | 875.47 | 867.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 858.00 | 875.47 | 867.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 858.00 | 875.47 | 867.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 850.55 | 875.47 | 867.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 834.00 | 867.18 | 864.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 834.00 | 867.18 | 864.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 828.65 | 859.47 | 861.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 825.45 | 847.45 | 855.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 825.60 | 819.91 | 833.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 15:00:00 | 825.60 | 819.91 | 833.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 863.40 | 829.90 | 835.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 863.30 | 829.90 | 835.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 865.80 | 840.78 | 839.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 869.75 | 846.57 | 842.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 842.95 | 854.36 | 848.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 842.95 | 854.36 | 848.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 842.95 | 854.36 | 848.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 858.95 | 856.51 | 850.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 944.85 | 916.85 | 902.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 1004.95 | 1015.83 | 1016.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 996.80 | 1012.02 | 1014.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 987.40 | 981.61 | 993.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 987.40 | 981.61 | 993.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 987.40 | 981.61 | 993.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 991.45 | 981.61 | 993.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 987.50 | 982.79 | 992.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 990.60 | 982.79 | 992.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1009.80 | 988.19 | 994.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 1009.80 | 988.19 | 994.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1009.70 | 992.49 | 995.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 1008.80 | 992.49 | 995.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1013.75 | 1000.17 | 998.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1020.35 | 1007.71 | 1002.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1018.00 | 1020.65 | 1014.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1018.00 | 1020.65 | 1014.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1018.00 | 1020.65 | 1014.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1018.00 | 1020.65 | 1014.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1011.25 | 1018.77 | 1014.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1011.25 | 1018.77 | 1014.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1012.05 | 1017.42 | 1014.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1020.30 | 1017.42 | 1014.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1028.20 | 1021.12 | 1017.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:30:00 | 1029.75 | 1022.90 | 1018.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1053.40 | 1023.28 | 1019.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 1029.00 | 1026.49 | 1021.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 13:30:00 | 1031.50 | 1026.49 | 1022.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1062.40 | 1033.71 | 1026.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 1075.20 | 1033.71 | 1026.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 1071.60 | 1067.27 | 1057.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 09:15:00 | 1132.73 | 1118.14 | 1095.03 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-29 09:15:00 | 652.00 | 2025-05-29 13:15:00 | 642.10 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-06-10 12:30:00 | 639.10 | 2025-06-10 14:15:00 | 644.45 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-18 12:15:00 | 608.55 | 2025-06-24 12:15:00 | 603.85 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-06-19 09:45:00 | 604.95 | 2025-06-24 12:15:00 | 603.85 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest1 | 2025-07-04 09:45:00 | 953.60 | 2025-07-07 10:15:00 | 917.20 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest1 | 2025-07-04 15:15:00 | 955.00 | 2025-07-07 10:15:00 | 917.20 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-07-07 12:15:00 | 930.90 | 2025-07-09 09:15:00 | 1023.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-08 09:30:00 | 934.65 | 2025-07-09 09:15:00 | 1028.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1018.10 | 2025-07-29 09:15:00 | 967.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1018.10 | 2025-07-29 12:15:00 | 990.70 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest2 | 2025-08-01 09:15:00 | 1041.00 | 2025-08-01 15:15:00 | 999.80 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-08-01 13:30:00 | 1043.00 | 2025-08-01 15:15:00 | 999.80 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-08-07 13:15:00 | 992.50 | 2025-08-11 09:15:00 | 1020.40 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-08-08 14:45:00 | 999.00 | 2025-08-11 09:15:00 | 1020.40 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1146.70 | 2025-08-25 10:15:00 | 1141.60 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1241.90 | 2025-09-05 10:15:00 | 1211.20 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-04 15:15:00 | 1226.90 | 2025-09-05 10:15:00 | 1211.20 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-05 12:30:00 | 1229.10 | 2025-09-05 13:15:00 | 1217.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-08 12:00:00 | 1215.00 | 2025-09-09 09:15:00 | 1265.90 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest1 | 2025-09-17 13:15:00 | 1321.80 | 2025-09-18 09:15:00 | 1271.10 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-09-23 09:15:00 | 1301.50 | 2025-09-24 13:15:00 | 1278.60 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-09-23 15:15:00 | 1290.00 | 2025-09-24 13:15:00 | 1278.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-24 09:30:00 | 1293.90 | 2025-09-24 13:15:00 | 1278.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-27 15:15:00 | 1237.80 | 2025-10-28 10:15:00 | 1263.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1032.00 | 2025-11-26 11:15:00 | 1055.50 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-11-25 14:00:00 | 1031.90 | 2025-11-26 11:15:00 | 1055.50 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-12-15 09:15:00 | 948.90 | 2025-12-15 10:15:00 | 963.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-12-23 14:45:00 | 1051.40 | 2025-12-26 10:15:00 | 1032.30 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-24 13:45:00 | 1045.20 | 2025-12-26 10:15:00 | 1032.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1042.40 | 2025-12-26 10:15:00 | 1032.30 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-30 13:15:00 | 998.60 | 2026-01-01 09:15:00 | 1037.90 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-12-31 10:00:00 | 993.00 | 2026-01-01 09:15:00 | 1037.90 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2025-12-31 11:15:00 | 996.00 | 2026-01-01 09:15:00 | 1037.90 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-12-31 12:00:00 | 998.70 | 2026-01-01 09:15:00 | 1037.90 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2026-01-20 09:15:00 | 914.80 | 2026-01-21 09:15:00 | 869.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 914.80 | 2026-01-21 15:15:00 | 899.90 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-01-28 13:30:00 | 883.00 | 2026-01-28 14:15:00 | 896.90 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-01-29 09:30:00 | 882.10 | 2026-01-30 09:15:00 | 941.70 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2026-02-10 09:30:00 | 991.45 | 2026-02-13 12:15:00 | 1003.05 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2026-02-19 12:30:00 | 969.00 | 2026-02-23 13:15:00 | 970.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-02-27 11:30:00 | 1017.00 | 2026-03-02 10:15:00 | 977.85 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2026-03-06 14:30:00 | 919.80 | 2026-03-09 09:15:00 | 873.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 919.80 | 2026-03-10 10:15:00 | 887.50 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2026-04-02 11:30:00 | 858.95 | 2026-04-09 09:15:00 | 944.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 13:30:00 | 1029.75 | 2026-05-08 09:15:00 | 1132.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1053.40 | 2026-05-08 09:15:00 | 1131.90 | TARGET_HIT | 1.00 | 7.45% |
| BUY | retest2 | 2026-05-04 12:15:00 | 1029.00 | 2026-05-08 09:15:00 | 1134.65 | TARGET_HIT | 1.00 | 10.27% |
