# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1122.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 240 |
| ALERT1 | 152 |
| ALERT2 | 149 |
| ALERT2_SKIP | 87 |
| ALERT3 | 435 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 194 |
| PARTIAL | 15 |
| TARGET_HIT | 36 |
| STOP_HIT | 161 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 212 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 88 / 124
- **Target hits / Stop hits / Partials:** 36 / 161 / 15
- **Avg / median % per leg:** 1.47% / -0.48%
- **Sum % (uncompounded):** 311.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 104 | 43 | 41.3% | 31 | 73 | 0 | 2.65% | 275.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.29% | -2.6% |
| BUY @ 3rd Alert (retest2) | 102 | 43 | 42.2% | 31 | 71 | 0 | 2.73% | 278.0% |
| SELL (all) | 108 | 45 | 41.7% | 5 | 88 | 15 | 0.34% | 36.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.85% | -0.9% |
| SELL @ 3rd Alert (retest2) | 107 | 45 | 42.1% | 5 | 87 | 15 | 0.35% | 37.3% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.14% | -3.4% |
| retest2 (combined) | 209 | 88 | 42.1% | 36 | 158 | 15 | 1.51% | 315.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 09:15:00 | 602.90 | 596.63 | 596.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 13:15:00 | 606.10 | 600.49 | 598.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 13:15:00 | 596.10 | 601.94 | 600.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 13:15:00 | 596.10 | 601.94 | 600.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 13:15:00 | 596.10 | 601.94 | 600.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 14:00:00 | 596.10 | 601.94 | 600.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 14:15:00 | 593.20 | 600.19 | 599.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 15:00:00 | 593.20 | 600.19 | 599.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 15:15:00 | 593.00 | 598.75 | 599.29 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 13:15:00 | 609.10 | 599.88 | 599.50 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 12:15:00 | 594.90 | 602.94 | 604.00 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 14:15:00 | 603.50 | 600.78 | 600.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 09:15:00 | 611.50 | 603.41 | 601.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 604.05 | 604.67 | 602.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 12:00:00 | 604.05 | 604.67 | 602.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 601.00 | 603.94 | 602.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:45:00 | 598.05 | 603.94 | 602.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 603.65 | 603.88 | 602.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 14:30:00 | 608.00 | 605.47 | 603.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 11:15:00 | 628.60 | 635.96 | 636.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 628.60 | 635.96 | 636.89 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 10:15:00 | 642.40 | 636.73 | 636.45 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 12:15:00 | 638.00 | 639.35 | 639.50 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 09:15:00 | 646.20 | 640.11 | 639.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 11:15:00 | 653.85 | 644.02 | 641.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 15:15:00 | 658.05 | 658.19 | 652.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 09:15:00 | 664.85 | 658.19 | 652.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 656.00 | 660.80 | 656.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-20 13:15:00 | 656.00 | 660.80 | 656.48 | SL hit (close<ema400) qty=1.00 sl=656.48 alert=retest1 |

### Cycle 10 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 651.00 | 655.29 | 655.67 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 09:15:00 | 662.80 | 656.79 | 656.32 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 652.00 | 655.88 | 656.05 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 13:15:00 | 660.10 | 656.72 | 656.41 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 15:15:00 | 655.00 | 656.02 | 656.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 650.30 | 654.88 | 655.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 12:15:00 | 642.50 | 642.23 | 647.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 12:45:00 | 641.75 | 642.23 | 647.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 645.65 | 643.24 | 646.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:30:00 | 645.40 | 643.24 | 646.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 649.00 | 644.40 | 646.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 649.65 | 644.40 | 646.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 649.85 | 645.49 | 647.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 12:15:00 | 646.55 | 646.79 | 647.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 650.05 | 647.64 | 647.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 650.05 | 647.64 | 647.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 11:15:00 | 660.00 | 651.07 | 649.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 12:15:00 | 660.30 | 660.34 | 656.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-30 12:45:00 | 660.65 | 660.34 | 656.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 661.65 | 661.32 | 658.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 14:45:00 | 667.10 | 662.56 | 659.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 15:15:00 | 666.95 | 662.56 | 659.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 11:15:00 | 667.60 | 663.76 | 661.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 12:15:00 | 666.95 | 664.39 | 661.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-07 09:15:00 | 733.81 | 713.59 | 698.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 09:15:00 | 704.50 | 726.92 | 728.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 12:15:00 | 690.00 | 713.65 | 721.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-21 09:15:00 | 620.80 | 618.00 | 631.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-21 10:00:00 | 620.80 | 618.00 | 631.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 627.55 | 623.72 | 629.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 15:00:00 | 627.55 | 623.72 | 629.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 622.25 | 624.07 | 628.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 12:00:00 | 616.65 | 622.22 | 625.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 09:30:00 | 616.60 | 615.25 | 620.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 15:15:00 | 616.15 | 615.91 | 616.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 10:15:00 | 617.35 | 615.52 | 615.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 10:15:00 | 617.35 | 615.52 | 615.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-02 10:15:00 | 620.50 | 616.98 | 616.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 13:15:00 | 616.20 | 617.79 | 616.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 13:15:00 | 616.20 | 617.79 | 616.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 616.20 | 617.79 | 616.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 616.20 | 617.79 | 616.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 617.50 | 617.73 | 616.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:30:00 | 616.65 | 617.73 | 616.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 15:15:00 | 614.50 | 617.08 | 616.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:30:00 | 612.50 | 617.17 | 616.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 618.40 | 617.41 | 616.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 10:30:00 | 616.70 | 617.41 | 616.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 616.85 | 617.30 | 616.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 12:00:00 | 616.85 | 617.30 | 616.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 12:15:00 | 615.95 | 617.03 | 616.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 13:00:00 | 615.95 | 617.03 | 616.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 13:15:00 | 619.45 | 617.51 | 617.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 15:15:00 | 622.50 | 617.90 | 617.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 15:15:00 | 614.60 | 617.99 | 617.98 | SL hit (close<static) qty=1.00 sl=615.60 alert=retest2 |

### Cycle 18 — SELL (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 09:15:00 | 613.20 | 617.04 | 617.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 13:15:00 | 607.60 | 611.50 | 613.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 12:15:00 | 610.45 | 607.80 | 610.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 12:15:00 | 610.45 | 607.80 | 610.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 12:15:00 | 610.45 | 607.80 | 610.32 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 10:15:00 | 614.35 | 611.36 | 611.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 13:15:00 | 616.45 | 613.01 | 612.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 15:15:00 | 611.95 | 612.96 | 612.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 15:15:00 | 611.95 | 612.96 | 612.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 611.95 | 612.96 | 612.28 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 12:15:00 | 607.25 | 611.21 | 611.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 605.25 | 609.53 | 610.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 611.50 | 605.05 | 606.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 611.50 | 605.05 | 606.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 611.50 | 605.05 | 606.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:45:00 | 612.00 | 605.05 | 606.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 614.50 | 606.94 | 607.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 11:00:00 | 614.50 | 606.94 | 607.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 12:15:00 | 611.60 | 608.36 | 608.05 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 13:15:00 | 605.70 | 607.83 | 607.83 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 09:15:00 | 613.30 | 608.67 | 608.19 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 15:15:00 | 605.85 | 610.03 | 610.38 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 12:15:00 | 612.00 | 610.65 | 610.57 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 13:15:00 | 607.70 | 610.06 | 610.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 10:15:00 | 606.80 | 609.07 | 609.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 11:15:00 | 611.05 | 609.46 | 609.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 11:15:00 | 611.05 | 609.46 | 609.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 11:15:00 | 611.05 | 609.46 | 609.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:00:00 | 611.05 | 609.46 | 609.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 12:15:00 | 612.25 | 610.02 | 610.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:45:00 | 612.25 | 610.02 | 610.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 13:15:00 | 611.45 | 610.31 | 610.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 14:15:00 | 612.35 | 610.71 | 610.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 09:15:00 | 609.10 | 610.64 | 610.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 609.10 | 610.64 | 610.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 609.10 | 610.64 | 610.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:00:00 | 609.10 | 610.64 | 610.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 10:15:00 | 608.40 | 610.19 | 610.25 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 14:15:00 | 610.95 | 610.34 | 610.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 12:15:00 | 612.55 | 611.12 | 610.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 14:15:00 | 610.40 | 611.16 | 610.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 14:15:00 | 610.40 | 611.16 | 610.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 610.40 | 611.16 | 610.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 15:00:00 | 610.40 | 611.16 | 610.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 610.00 | 610.93 | 610.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 609.95 | 610.93 | 610.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 608.30 | 610.40 | 610.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 607.00 | 609.72 | 610.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 600.15 | 597.82 | 601.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 600.15 | 597.82 | 601.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 600.15 | 597.82 | 601.57 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 15:15:00 | 612.00 | 604.12 | 603.26 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 15:15:00 | 603.20 | 605.37 | 605.50 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 09:15:00 | 608.90 | 606.08 | 605.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 13:15:00 | 612.00 | 608.01 | 606.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 666.35 | 675.63 | 667.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 666.35 | 675.63 | 667.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 666.35 | 675.63 | 667.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 653.05 | 675.63 | 667.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 664.50 | 673.41 | 667.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 659.40 | 673.41 | 667.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 11:15:00 | 665.50 | 671.83 | 666.91 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 09:15:00 | 651.95 | 663.45 | 664.26 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 13:15:00 | 668.00 | 664.79 | 664.56 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 15:15:00 | 658.35 | 663.94 | 664.24 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 671.80 | 665.51 | 664.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 15:15:00 | 682.00 | 673.01 | 670.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 10:15:00 | 672.05 | 673.03 | 670.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 11:00:00 | 672.05 | 673.03 | 670.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 11:15:00 | 667.50 | 671.93 | 670.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 11:45:00 | 667.70 | 671.93 | 670.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 659.65 | 669.47 | 669.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:00:00 | 659.65 | 669.47 | 669.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 13:15:00 | 658.85 | 667.35 | 668.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 14:15:00 | 653.95 | 664.67 | 667.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 09:15:00 | 621.05 | 620.88 | 627.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-25 09:30:00 | 621.90 | 620.88 | 627.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 629.25 | 622.55 | 627.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:30:00 | 629.80 | 622.55 | 627.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 11:15:00 | 635.10 | 625.06 | 628.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 12:00:00 | 635.10 | 625.06 | 628.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 14:15:00 | 635.30 | 630.00 | 629.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 15:15:00 | 636.90 | 631.38 | 630.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 15:15:00 | 640.25 | 640.81 | 636.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 12:15:00 | 640.60 | 641.27 | 638.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 12:15:00 | 640.60 | 641.27 | 638.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 12:30:00 | 637.90 | 641.27 | 638.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 14:15:00 | 633.60 | 639.64 | 638.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 15:00:00 | 633.60 | 639.64 | 638.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 15:15:00 | 629.70 | 637.65 | 637.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 09:15:00 | 633.85 | 637.65 | 637.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 09:15:00 | 628.65 | 635.85 | 636.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 09:15:00 | 628.65 | 635.85 | 636.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 10:15:00 | 624.50 | 633.58 | 635.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 14:15:00 | 635.65 | 632.46 | 634.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 635.65 | 632.46 | 634.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 635.65 | 632.46 | 634.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:30:00 | 636.85 | 632.46 | 634.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 630.45 | 632.05 | 633.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:15:00 | 633.60 | 632.05 | 633.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 635.50 | 632.74 | 633.95 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2023-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 12:15:00 | 638.00 | 635.32 | 634.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 14:15:00 | 646.25 | 638.42 | 636.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 652.15 | 654.35 | 648.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-04 12:00:00 | 652.15 | 654.35 | 648.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 646.80 | 652.84 | 648.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:00:00 | 646.80 | 652.84 | 648.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 647.00 | 651.67 | 648.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 14:00:00 | 647.00 | 651.67 | 648.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 14:15:00 | 644.95 | 650.33 | 648.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 14:45:00 | 645.00 | 650.33 | 648.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 10:15:00 | 643.70 | 647.89 | 647.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 11:00:00 | 643.70 | 647.89 | 647.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 11:15:00 | 646.00 | 647.51 | 647.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 14:30:00 | 656.55 | 649.73 | 648.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 11:15:00 | 642.00 | 648.78 | 648.61 | SL hit (close<static) qty=1.00 sl=642.65 alert=retest2 |

### Cycle 42 — SELL (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 12:15:00 | 646.45 | 648.32 | 648.41 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 09:15:00 | 653.15 | 648.78 | 648.54 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 637.55 | 647.97 | 648.36 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 651.00 | 648.63 | 648.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 14:15:00 | 657.90 | 651.97 | 650.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 10:15:00 | 651.65 | 653.58 | 651.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 10:15:00 | 651.65 | 653.58 | 651.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 651.65 | 653.58 | 651.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 10:45:00 | 651.95 | 653.58 | 651.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 651.80 | 653.22 | 651.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 12:15:00 | 650.80 | 653.22 | 651.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 651.35 | 652.85 | 651.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:15:00 | 648.70 | 652.85 | 651.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 655.00 | 653.28 | 651.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 14:30:00 | 658.65 | 653.62 | 652.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 12:15:00 | 644.20 | 650.74 | 651.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 12:15:00 | 644.20 | 650.74 | 651.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 09:15:00 | 643.40 | 647.60 | 649.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 11:15:00 | 648.00 | 647.34 | 649.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-13 11:45:00 | 647.85 | 647.34 | 649.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 650.45 | 647.96 | 649.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 12:45:00 | 650.20 | 647.96 | 649.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 655.30 | 649.43 | 649.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 13:30:00 | 655.00 | 649.43 | 649.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 14:15:00 | 656.20 | 650.79 | 650.30 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 11:15:00 | 644.20 | 649.56 | 650.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 13:15:00 | 641.05 | 644.45 | 646.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 14:15:00 | 655.45 | 641.91 | 643.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 655.45 | 641.91 | 643.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 655.45 | 641.91 | 643.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 655.45 | 641.91 | 643.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 15:15:00 | 655.45 | 644.62 | 644.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 09:15:00 | 667.00 | 649.10 | 646.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 15:15:00 | 656.05 | 658.46 | 653.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 15:15:00 | 656.05 | 658.46 | 653.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 656.05 | 658.46 | 653.29 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 641.30 | 649.74 | 650.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 14:15:00 | 629.35 | 644.51 | 647.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 14:15:00 | 633.90 | 623.86 | 629.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 14:15:00 | 633.90 | 623.86 | 629.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 633.90 | 623.86 | 629.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 15:00:00 | 633.90 | 623.86 | 629.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 15:15:00 | 629.00 | 624.88 | 629.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:15:00 | 611.05 | 624.88 | 629.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 11:15:00 | 615.75 | 611.26 | 611.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 11:15:00 | 615.75 | 611.26 | 611.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 13:15:00 | 616.40 | 612.48 | 611.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 15:15:00 | 612.90 | 613.20 | 612.29 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 09:30:00 | 618.40 | 613.51 | 612.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 616.75 | 614.87 | 613.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-03 14:15:00 | 610.70 | 614.04 | 613.26 | SL hit (close<ema400) qty=1.00 sl=613.26 alert=retest1 |

### Cycle 52 — SELL (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 12:15:00 | 619.00 | 625.32 | 625.51 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 09:15:00 | 629.40 | 625.37 | 625.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 11:15:00 | 631.65 | 627.02 | 626.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 13:15:00 | 626.55 | 628.49 | 627.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 13:15:00 | 626.55 | 628.49 | 627.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 626.55 | 628.49 | 627.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 14:00:00 | 626.55 | 628.49 | 627.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 626.05 | 628.00 | 626.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 15:15:00 | 627.05 | 628.00 | 626.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 631.60 | 629.40 | 627.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:30:00 | 630.60 | 629.40 | 627.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 627.00 | 628.92 | 627.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:30:00 | 627.90 | 628.92 | 627.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 11:15:00 | 629.05 | 628.94 | 627.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 11:30:00 | 627.55 | 628.94 | 627.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 628.45 | 630.59 | 629.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 15:00:00 | 635.80 | 632.85 | 631.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 09:15:00 | 639.55 | 633.08 | 631.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:45:00 | 637.80 | 638.38 | 635.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 10:30:00 | 636.00 | 637.70 | 635.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 11:15:00 | 630.30 | 636.22 | 635.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:00:00 | 630.30 | 636.22 | 635.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 12:15:00 | 629.45 | 634.87 | 634.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:30:00 | 630.15 | 634.87 | 634.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-17 13:15:00 | 627.00 | 633.29 | 633.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 13:15:00 | 627.00 | 633.29 | 633.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 09:15:00 | 626.05 | 630.39 | 632.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 623.70 | 622.71 | 626.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-21 10:15:00 | 625.55 | 622.71 | 626.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 625.00 | 623.16 | 626.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 11:00:00 | 625.00 | 623.16 | 626.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 11:15:00 | 626.05 | 623.74 | 626.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 12:00:00 | 626.05 | 623.74 | 626.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 12:15:00 | 627.00 | 624.39 | 626.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 13:00:00 | 627.00 | 624.39 | 626.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 13:15:00 | 627.00 | 624.91 | 626.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 13:45:00 | 627.00 | 624.91 | 626.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 631.30 | 627.09 | 627.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:15:00 | 630.05 | 627.09 | 627.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 09:15:00 | 630.00 | 627.67 | 627.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 13:15:00 | 637.85 | 630.88 | 629.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 09:15:00 | 636.80 | 638.08 | 635.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 636.80 | 638.08 | 635.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 636.80 | 638.08 | 635.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:30:00 | 635.60 | 638.08 | 635.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 639.40 | 638.34 | 635.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:30:00 | 635.60 | 638.34 | 635.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 629.95 | 636.67 | 634.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:00:00 | 629.95 | 636.67 | 634.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 633.60 | 636.05 | 634.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:30:00 | 630.90 | 636.05 | 634.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 634.15 | 635.67 | 634.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:45:00 | 634.30 | 635.67 | 634.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 632.05 | 634.95 | 634.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 14:45:00 | 632.40 | 634.95 | 634.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 15:15:00 | 629.00 | 633.76 | 634.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 624.30 | 631.87 | 633.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 13:15:00 | 631.50 | 630.41 | 631.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 13:15:00 | 631.50 | 630.41 | 631.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 631.50 | 630.41 | 631.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 13:30:00 | 630.00 | 630.41 | 631.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 635.10 | 631.35 | 632.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:45:00 | 635.15 | 631.35 | 632.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 631.75 | 631.43 | 632.11 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 643.75 | 633.89 | 633.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 10:15:00 | 650.05 | 637.12 | 634.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 10:15:00 | 644.70 | 646.50 | 641.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-30 10:30:00 | 645.35 | 646.50 | 641.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 651.25 | 656.69 | 653.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 10:00:00 | 651.25 | 656.69 | 653.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 648.00 | 654.95 | 652.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 10:30:00 | 649.35 | 654.95 | 652.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 12:15:00 | 655.00 | 654.49 | 652.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 12:30:00 | 655.40 | 654.49 | 652.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 659.50 | 660.05 | 657.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 12:45:00 | 659.85 | 660.05 | 657.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 13:15:00 | 657.70 | 659.58 | 657.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 13:45:00 | 657.15 | 659.58 | 657.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 14:15:00 | 658.40 | 659.34 | 657.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 15:15:00 | 659.95 | 659.34 | 657.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 15:15:00 | 659.95 | 659.47 | 657.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 09:45:00 | 660.60 | 660.67 | 658.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 09:30:00 | 660.40 | 664.94 | 662.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-07 10:15:00 | 662.50 | 664.94 | 662.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 15:15:00 | 653.60 | 660.62 | 661.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2023-12-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 15:15:00 | 653.60 | 660.62 | 661.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 648.20 | 656.27 | 658.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 15:15:00 | 649.95 | 649.87 | 653.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-12 09:15:00 | 647.95 | 649.87 | 653.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 652.95 | 645.85 | 648.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 09:30:00 | 643.05 | 645.51 | 648.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 10:15:00 | 641.80 | 645.51 | 648.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 12:30:00 | 643.10 | 644.50 | 647.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 14:15:00 | 654.45 | 647.38 | 647.95 | SL hit (close>static) qty=1.00 sl=652.95 alert=retest2 |

### Cycle 59 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 655.55 | 649.65 | 648.87 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 13:15:00 | 643.35 | 647.88 | 648.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-14 14:15:00 | 638.00 | 645.91 | 647.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 633.60 | 630.77 | 633.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 633.60 | 630.77 | 633.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 633.60 | 630.77 | 633.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 11:45:00 | 629.25 | 630.73 | 633.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 12:15:00 | 629.25 | 630.73 | 633.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-21 10:15:00 | 637.30 | 629.07 | 630.88 | SL hit (close>static) qty=1.00 sl=634.95 alert=retest2 |

### Cycle 61 — BUY (started 2023-12-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 13:15:00 | 634.70 | 631.91 | 631.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 637.25 | 633.76 | 632.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-22 12:15:00 | 629.20 | 633.90 | 633.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 12:15:00 | 629.20 | 633.90 | 633.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 629.20 | 633.90 | 633.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 12:45:00 | 628.65 | 633.90 | 633.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 628.45 | 632.81 | 632.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:45:00 | 626.90 | 632.81 | 632.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 14:15:00 | 631.55 | 632.56 | 632.67 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2023-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 11:15:00 | 641.75 | 634.30 | 633.36 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 10:15:00 | 633.15 | 634.89 | 635.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 11:15:00 | 626.10 | 633.13 | 634.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 09:15:00 | 630.70 | 629.38 | 631.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 630.70 | 629.38 | 631.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 630.70 | 629.38 | 631.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 09:30:00 | 630.40 | 629.38 | 631.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 643.85 | 632.28 | 632.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:45:00 | 642.95 | 632.28 | 632.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2023-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 11:15:00 | 648.50 | 635.52 | 634.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 12:15:00 | 649.20 | 638.26 | 635.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 09:15:00 | 635.60 | 640.09 | 637.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 635.60 | 640.09 | 637.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 635.60 | 640.09 | 637.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:00:00 | 635.60 | 640.09 | 637.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 635.55 | 639.18 | 637.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:30:00 | 635.05 | 639.18 | 637.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2024-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 14:15:00 | 633.35 | 635.87 | 636.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 09:15:00 | 631.75 | 634.59 | 635.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 15:15:00 | 630.50 | 630.31 | 632.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-03 09:15:00 | 630.95 | 630.31 | 632.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 627.55 | 629.76 | 632.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-03 10:30:00 | 625.65 | 629.00 | 631.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 09:15:00 | 634.25 | 629.91 | 629.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2024-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 09:15:00 | 634.25 | 629.91 | 629.42 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 624.65 | 629.12 | 629.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 11:15:00 | 621.30 | 627.55 | 628.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-08 13:15:00 | 631.75 | 628.11 | 628.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 13:15:00 | 631.75 | 628.11 | 628.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 631.75 | 628.11 | 628.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-08 14:00:00 | 631.75 | 628.11 | 628.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 630.85 | 628.66 | 629.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 15:15:00 | 627.30 | 628.66 | 629.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-09 10:15:00 | 633.55 | 629.63 | 629.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 10:15:00 | 633.55 | 629.63 | 629.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 12:15:00 | 636.00 | 631.45 | 630.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 631.50 | 633.37 | 631.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 631.50 | 633.37 | 631.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 631.50 | 633.37 | 631.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:30:00 | 633.00 | 633.37 | 631.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 632.10 | 633.11 | 631.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:45:00 | 633.15 | 633.11 | 631.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 631.55 | 632.80 | 631.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 11:30:00 | 631.40 | 632.80 | 631.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 12:15:00 | 631.05 | 632.45 | 631.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 12:45:00 | 631.00 | 632.45 | 631.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 13:15:00 | 639.25 | 633.81 | 632.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 14:15:00 | 640.45 | 633.81 | 632.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 09:15:00 | 641.10 | 635.46 | 633.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 14:30:00 | 642.00 | 639.86 | 636.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 13:30:00 | 640.95 | 639.15 | 637.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 635.40 | 639.81 | 638.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:45:00 | 636.20 | 639.81 | 638.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 635.15 | 638.87 | 638.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 12:00:00 | 635.15 | 638.87 | 638.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-15 13:15:00 | 635.40 | 637.62 | 637.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 13:15:00 | 635.40 | 637.62 | 637.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 15:15:00 | 635.00 | 636.94 | 637.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 13:15:00 | 636.85 | 635.72 | 636.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 13:15:00 | 636.85 | 635.72 | 636.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 636.85 | 635.72 | 636.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 14:00:00 | 636.85 | 635.72 | 636.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 636.25 | 635.82 | 636.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 09:15:00 | 628.00 | 635.56 | 636.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 10:15:00 | 636.40 | 632.06 | 632.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 10:15:00 | 636.40 | 632.06 | 632.02 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 10:15:00 | 629.35 | 632.21 | 632.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 15:15:00 | 627.00 | 629.50 | 630.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 09:15:00 | 630.85 | 629.77 | 630.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 630.85 | 629.77 | 630.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 630.85 | 629.77 | 630.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:45:00 | 634.15 | 629.77 | 630.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 635.70 | 630.95 | 631.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 11:00:00 | 635.70 | 630.95 | 631.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 11:15:00 | 634.05 | 631.57 | 631.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 642.60 | 636.46 | 634.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-24 14:15:00 | 635.05 | 636.18 | 634.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-24 15:00:00 | 635.05 | 636.18 | 634.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 636.20 | 637.78 | 635.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 12:45:00 | 634.00 | 637.78 | 635.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 640.45 | 638.32 | 636.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 13:30:00 | 635.85 | 638.32 | 636.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 671.95 | 674.13 | 667.89 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 12:15:00 | 654.60 | 666.12 | 667.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 14:15:00 | 648.75 | 660.90 | 664.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 649.15 | 641.29 | 648.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 649.15 | 641.29 | 648.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 649.15 | 641.29 | 648.99 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 12:15:00 | 663.35 | 652.32 | 651.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 14:15:00 | 668.80 | 657.24 | 653.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 11:15:00 | 655.15 | 659.58 | 656.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 11:15:00 | 655.15 | 659.58 | 656.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 11:15:00 | 655.15 | 659.58 | 656.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 12:00:00 | 655.15 | 659.58 | 656.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 12:15:00 | 656.65 | 658.99 | 656.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 12:30:00 | 655.25 | 658.99 | 656.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 13:15:00 | 656.70 | 658.54 | 656.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-08 15:00:00 | 665.90 | 660.01 | 657.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-09 09:15:00 | 650.90 | 658.97 | 657.22 | SL hit (close<static) qty=1.00 sl=655.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 643.60 | 655.90 | 655.98 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 15:15:00 | 660.25 | 655.78 | 655.54 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 649.00 | 654.42 | 654.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 10:15:00 | 643.80 | 652.30 | 653.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 646.95 | 646.21 | 649.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 11:00:00 | 646.95 | 646.21 | 649.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 642.40 | 640.10 | 643.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:45:00 | 642.00 | 640.10 | 643.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 643.75 | 640.83 | 643.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:45:00 | 646.85 | 640.83 | 643.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 645.65 | 641.80 | 643.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:45:00 | 647.95 | 641.80 | 643.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 14:15:00 | 656.40 | 644.72 | 644.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 15:00:00 | 656.40 | 644.72 | 644.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 658.00 | 647.37 | 646.12 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 13:15:00 | 643.75 | 649.32 | 649.60 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 10:15:00 | 651.80 | 649.64 | 649.61 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 11:15:00 | 649.00 | 649.51 | 649.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 12:15:00 | 643.00 | 648.21 | 648.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 12:15:00 | 643.55 | 643.30 | 645.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 12:15:00 | 643.55 | 643.30 | 645.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 12:15:00 | 643.55 | 643.30 | 645.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 12:45:00 | 644.65 | 643.30 | 645.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 651.00 | 644.84 | 645.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 651.00 | 644.84 | 645.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 647.40 | 645.35 | 646.08 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 09:15:00 | 653.00 | 647.62 | 647.04 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 644.00 | 646.80 | 647.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 639.10 | 645.26 | 646.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 12:15:00 | 640.55 | 637.84 | 640.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 12:15:00 | 640.55 | 637.84 | 640.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 12:15:00 | 640.55 | 637.84 | 640.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 13:00:00 | 640.55 | 637.84 | 640.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 13:15:00 | 642.85 | 638.84 | 640.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 14:00:00 | 642.85 | 638.84 | 640.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 636.95 | 638.47 | 640.48 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 13:15:00 | 642.00 | 640.21 | 640.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 14:15:00 | 648.75 | 641.92 | 640.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 10:15:00 | 645.00 | 647.61 | 644.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 10:15:00 | 645.00 | 647.61 | 644.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 645.00 | 647.61 | 644.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 11:00:00 | 645.00 | 647.61 | 644.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 638.90 | 645.87 | 643.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:00:00 | 638.90 | 645.87 | 643.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 631.80 | 643.05 | 642.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 13:00:00 | 631.80 | 643.05 | 642.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 13:15:00 | 633.00 | 641.04 | 641.72 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 14:15:00 | 647.05 | 642.24 | 642.20 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 15:15:00 | 638.40 | 641.48 | 641.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 09:15:00 | 629.50 | 639.08 | 640.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 630.00 | 629.75 | 632.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 630.00 | 629.75 | 632.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 630.00 | 629.75 | 632.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-02 11:45:00 | 628.00 | 629.64 | 632.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 09:15:00 | 627.75 | 629.91 | 632.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 09:45:00 | 628.00 | 629.45 | 631.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 10:15:00 | 626.80 | 627.72 | 629.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 623.00 | 624.78 | 627.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 620.25 | 624.78 | 627.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 596.60 | 603.85 | 608.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 596.36 | 603.85 | 608.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 596.60 | 603.85 | 608.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 11:15:00 | 595.46 | 602.07 | 607.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 12:15:00 | 589.24 | 600.11 | 606.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-12 13:15:00 | 606.35 | 601.36 | 606.09 | SL hit (close>ema200) qty=0.50 sl=601.36 alert=retest2 |

### Cycle 89 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 581.20 | 575.15 | 574.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 586.65 | 577.45 | 575.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 593.70 | 598.68 | 591.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 593.70 | 598.68 | 591.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 593.70 | 598.68 | 591.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 10:00:00 | 593.70 | 598.68 | 591.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 595.00 | 597.20 | 592.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 13:30:00 | 593.00 | 597.20 | 592.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 591.50 | 596.06 | 592.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:30:00 | 587.85 | 596.06 | 592.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 588.00 | 594.45 | 592.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:15:00 | 591.50 | 594.45 | 592.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 589.70 | 592.63 | 591.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 13:00:00 | 590.00 | 591.66 | 591.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 13:15:00 | 589.30 | 591.19 | 591.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 13:15:00 | 589.30 | 591.19 | 591.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 11:15:00 | 586.00 | 588.80 | 590.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 589.75 | 588.10 | 589.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 589.75 | 588.10 | 589.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 589.75 | 588.10 | 589.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:00:00 | 589.75 | 588.10 | 589.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 589.80 | 588.44 | 589.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:30:00 | 588.50 | 588.44 | 589.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 11:15:00 | 589.95 | 588.74 | 589.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 11:30:00 | 589.35 | 588.74 | 589.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 12:15:00 | 589.30 | 588.85 | 589.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 13:00:00 | 589.30 | 588.85 | 589.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 13:15:00 | 589.00 | 588.88 | 589.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 13:30:00 | 589.00 | 588.88 | 589.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-02 09:15:00 | 594.40 | 589.72 | 589.50 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 14:15:00 | 588.85 | 589.36 | 589.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-02 15:15:00 | 588.00 | 589.09 | 589.27 | Break + close below crossover candle low |

### Cycle 93 — BUY (started 2024-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 09:15:00 | 593.55 | 589.98 | 589.66 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 15:15:00 | 590.00 | 590.84 | 590.87 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 09:15:00 | 593.95 | 591.46 | 591.15 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 11:15:00 | 588.00 | 590.67 | 590.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 12:15:00 | 586.00 | 589.74 | 590.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 14:15:00 | 573.00 | 572.95 | 577.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-09 15:00:00 | 573.00 | 572.95 | 577.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 580.30 | 574.42 | 578.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:15:00 | 585.50 | 574.42 | 578.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 583.70 | 576.28 | 578.62 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 13:15:00 | 585.60 | 580.57 | 580.15 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 576.85 | 581.90 | 582.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-22 12:15:00 | 566.00 | 572.54 | 573.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 11:15:00 | 574.70 | 570.77 | 571.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 11:15:00 | 574.70 | 570.77 | 571.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 574.70 | 570.77 | 571.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 12:00:00 | 574.70 | 570.77 | 571.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 577.70 | 572.15 | 572.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 12:45:00 | 581.15 | 572.15 | 572.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 13:15:00 | 579.60 | 573.64 | 573.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 14:15:00 | 585.90 | 576.09 | 574.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 12:15:00 | 581.80 | 581.85 | 578.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 13:00:00 | 581.80 | 581.85 | 578.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 13:15:00 | 581.85 | 581.85 | 578.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 13:30:00 | 578.40 | 581.85 | 578.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 580.10 | 581.76 | 579.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 09:45:00 | 583.80 | 581.90 | 579.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 10:15:00 | 576.60 | 580.84 | 579.18 | SL hit (close<static) qty=1.00 sl=578.15 alert=retest2 |

### Cycle 100 — SELL (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 14:15:00 | 582.40 | 586.53 | 586.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 10:15:00 | 579.30 | 584.51 | 585.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 14:15:00 | 586.50 | 581.19 | 583.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 14:15:00 | 586.50 | 581.19 | 583.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 586.50 | 581.19 | 583.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:00:00 | 586.50 | 581.19 | 583.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 583.10 | 581.57 | 583.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 582.60 | 581.57 | 583.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 579.40 | 581.14 | 582.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:00:00 | 575.05 | 579.09 | 580.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 11:30:00 | 575.85 | 578.12 | 579.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:30:00 | 575.60 | 577.40 | 579.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 15:15:00 | 575.60 | 576.99 | 578.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 575.60 | 576.71 | 578.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:15:00 | 576.90 | 576.71 | 578.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 578.95 | 577.16 | 578.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 575.00 | 576.88 | 577.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 09:45:00 | 573.80 | 568.77 | 571.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 09:15:00 | 570.35 | 566.35 | 565.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 09:15:00 | 570.35 | 566.35 | 565.83 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 558.45 | 566.47 | 566.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 556.15 | 560.84 | 563.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 10:15:00 | 558.35 | 558.22 | 561.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 11:15:00 | 560.80 | 558.74 | 561.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 560.80 | 558.74 | 561.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:00:00 | 560.80 | 558.74 | 561.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 560.65 | 559.12 | 561.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 560.65 | 559.12 | 561.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 562.95 | 559.89 | 561.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 562.95 | 559.89 | 561.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 565.75 | 561.06 | 561.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 565.75 | 561.06 | 561.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 564.20 | 562.46 | 562.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 570.60 | 565.45 | 564.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 13:15:00 | 591.00 | 591.24 | 585.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 13:45:00 | 591.25 | 591.24 | 585.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 585.05 | 589.79 | 585.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 585.05 | 589.79 | 585.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 590.35 | 589.90 | 586.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:30:00 | 587.55 | 589.90 | 586.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 590.05 | 590.29 | 587.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:00:00 | 590.05 | 590.29 | 587.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 585.95 | 589.42 | 587.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 585.95 | 589.42 | 587.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 585.10 | 588.56 | 587.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 12:00:00 | 587.75 | 587.28 | 586.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 13:00:00 | 589.30 | 587.68 | 586.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 15:15:00 | 585.00 | 586.46 | 586.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 585.00 | 586.46 | 586.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 13:15:00 | 575.65 | 581.30 | 583.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 579.95 | 578.63 | 581.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 579.95 | 578.63 | 581.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 579.95 | 578.63 | 581.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 574.65 | 577.10 | 580.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:45:00 | 575.30 | 576.96 | 580.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:30:00 | 575.05 | 576.55 | 579.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:00:00 | 574.90 | 576.55 | 579.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 577.40 | 576.72 | 579.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:45:00 | 583.75 | 576.72 | 579.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 575.05 | 576.39 | 578.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 565.00 | 576.39 | 578.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 577.65 | 576.64 | 578.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:15:00 | 580.15 | 576.64 | 578.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 571.35 | 575.58 | 578.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:30:00 | 581.60 | 575.58 | 578.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 565.05 | 573.47 | 576.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:30:00 | 575.00 | 573.47 | 576.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 580.00 | 573.16 | 575.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 582.00 | 574.93 | 576.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 581.00 | 576.14 | 576.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:45:00 | 582.00 | 576.14 | 576.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 575.10 | 576.32 | 576.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:30:00 | 576.95 | 576.32 | 576.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 578.70 | 576.80 | 576.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 578.70 | 576.80 | 576.76 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-06 09:15:00 | 571.00 | 575.83 | 576.34 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 613.00 | 582.12 | 578.54 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 603.55 | 609.02 | 609.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 600.95 | 604.76 | 606.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 15:15:00 | 591.70 | 591.62 | 594.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 09:15:00 | 591.70 | 591.62 | 594.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 589.85 | 591.27 | 594.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 587.10 | 590.37 | 593.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:45:00 | 588.15 | 589.56 | 592.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 595.30 | 591.51 | 591.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 595.30 | 591.51 | 591.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 600.25 | 594.72 | 593.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 589.55 | 594.43 | 593.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 589.55 | 594.43 | 593.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 589.55 | 594.43 | 593.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 589.55 | 594.43 | 593.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 594.80 | 594.50 | 593.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:45:00 | 599.60 | 595.08 | 594.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 15:15:00 | 588.00 | 593.09 | 593.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 15:15:00 | 588.00 | 593.09 | 593.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 11:15:00 | 587.60 | 591.10 | 592.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 11:15:00 | 592.40 | 590.16 | 591.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 11:15:00 | 592.40 | 590.16 | 591.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 592.40 | 590.16 | 591.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:00:00 | 592.40 | 590.16 | 591.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 589.30 | 589.99 | 590.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 15:15:00 | 586.40 | 589.38 | 590.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 595.85 | 590.20 | 590.28 | SL hit (close>static) qty=1.00 sl=592.75 alert=retest2 |

### Cycle 111 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 595.90 | 591.34 | 590.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 12:15:00 | 599.70 | 593.64 | 591.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 592.60 | 596.04 | 593.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 592.60 | 596.04 | 593.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 592.60 | 596.04 | 593.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 592.60 | 596.04 | 593.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 589.00 | 594.63 | 593.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 588.85 | 594.63 | 593.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 592.05 | 594.11 | 593.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 13:30:00 | 595.35 | 593.69 | 593.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:15:00 | 599.30 | 593.69 | 593.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 15:15:00 | 597.00 | 593.88 | 593.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 598.30 | 601.73 | 602.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 598.30 | 601.73 | 602.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 13:15:00 | 593.55 | 598.57 | 600.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 11:15:00 | 604.30 | 598.35 | 599.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 11:15:00 | 604.30 | 598.35 | 599.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 604.30 | 598.35 | 599.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:00:00 | 604.30 | 598.35 | 599.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 605.45 | 599.77 | 599.92 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 13:15:00 | 605.45 | 600.91 | 600.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 14:15:00 | 607.00 | 602.13 | 601.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 600.45 | 602.16 | 601.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 600.45 | 602.16 | 601.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 600.45 | 602.16 | 601.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 600.45 | 602.16 | 601.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 600.45 | 601.81 | 601.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:45:00 | 598.05 | 601.81 | 601.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 600.40 | 601.53 | 601.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 599.80 | 601.53 | 601.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 601.95 | 601.62 | 601.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:30:00 | 600.70 | 601.62 | 601.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 598.20 | 600.93 | 600.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 598.20 | 600.93 | 600.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 589.30 | 598.61 | 599.86 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 601.85 | 597.21 | 596.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 605.05 | 600.56 | 598.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 13:15:00 | 601.30 | 601.44 | 599.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-24 14:00:00 | 601.30 | 601.44 | 599.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 616.00 | 604.69 | 601.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 618.90 | 610.47 | 606.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 13:30:00 | 617.05 | 615.22 | 610.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 14:45:00 | 617.10 | 615.77 | 611.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:30:00 | 618.90 | 617.95 | 613.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 611.90 | 616.74 | 613.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:45:00 | 613.55 | 616.74 | 613.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 613.50 | 616.09 | 613.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 617.20 | 615.40 | 613.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:15:00 | 616.05 | 616.00 | 614.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:45:00 | 616.10 | 615.90 | 614.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 622.40 | 615.67 | 614.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 633.25 | 619.19 | 616.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:15:00 | 643.00 | 619.19 | 616.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-31 14:15:00 | 680.79 | 646.27 | 632.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 629.25 | 646.49 | 647.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 627.55 | 640.44 | 644.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 645.90 | 637.04 | 640.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 645.90 | 637.04 | 640.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 645.90 | 637.04 | 640.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 645.90 | 637.04 | 640.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 642.10 | 638.05 | 641.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:15:00 | 644.95 | 638.05 | 641.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 644.65 | 639.37 | 641.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 641.20 | 640.46 | 641.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 640.40 | 640.46 | 641.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 649.40 | 640.24 | 640.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 649.40 | 640.24 | 640.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 651.00 | 642.39 | 641.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 09:15:00 | 659.35 | 659.50 | 653.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 10:00:00 | 659.35 | 659.50 | 653.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 659.00 | 661.49 | 657.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 652.20 | 661.49 | 657.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 661.85 | 661.56 | 657.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:30:00 | 664.65 | 661.91 | 658.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:45:00 | 665.80 | 660.57 | 658.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 13:45:00 | 663.10 | 661.93 | 660.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 10:30:00 | 665.40 | 660.58 | 659.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 663.20 | 661.10 | 660.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:00:00 | 663.20 | 661.10 | 660.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 660.35 | 661.73 | 660.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:00:00 | 660.35 | 661.73 | 660.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 662.70 | 661.93 | 660.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:30:00 | 660.25 | 661.93 | 660.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 709.10 | 712.50 | 709.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 709.10 | 712.50 | 709.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 706.60 | 711.32 | 708.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 706.60 | 711.32 | 708.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 708.55 | 710.77 | 708.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:30:00 | 705.55 | 710.77 | 708.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 709.95 | 710.60 | 708.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 15:00:00 | 714.50 | 711.38 | 709.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:30:00 | 712.85 | 710.36 | 709.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 712.15 | 709.77 | 709.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 11:15:00 | 707.05 | 709.22 | 709.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 11:15:00 | 707.05 | 709.22 | 709.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 12:15:00 | 705.10 | 708.40 | 708.94 | Break + close below crossover candle low |

### Cycle 119 — BUY (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 13:15:00 | 723.00 | 711.32 | 710.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 10:15:00 | 726.45 | 719.37 | 716.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 14:15:00 | 719.15 | 721.90 | 718.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 14:15:00 | 719.15 | 721.90 | 718.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 719.15 | 721.90 | 718.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 719.15 | 721.90 | 718.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 733.00 | 724.12 | 720.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 09:45:00 | 739.20 | 727.14 | 723.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-04 13:15:00 | 813.12 | 760.67 | 741.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 759.70 | 767.28 | 768.08 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 765.00 | 760.03 | 759.94 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 09:15:00 | 758.30 | 759.68 | 759.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 10:15:00 | 754.20 | 758.59 | 759.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 724.40 | 720.34 | 726.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 13:15:00 | 724.40 | 720.34 | 726.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 724.40 | 720.34 | 726.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:30:00 | 723.75 | 720.34 | 726.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 730.70 | 722.41 | 726.58 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 09:15:00 | 749.65 | 729.36 | 729.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 14:15:00 | 754.50 | 744.01 | 739.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 15:15:00 | 752.15 | 754.69 | 748.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 09:15:00 | 749.30 | 754.69 | 748.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 742.20 | 752.19 | 748.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 742.20 | 752.19 | 748.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 740.55 | 749.87 | 747.40 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 13:15:00 | 738.70 | 744.98 | 745.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 14:15:00 | 736.50 | 743.28 | 744.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 703.40 | 702.30 | 707.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 15:00:00 | 703.40 | 702.30 | 707.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 687.75 | 684.47 | 689.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 687.75 | 684.47 | 689.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 667.00 | 661.81 | 665.32 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 670.00 | 667.33 | 667.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 10:15:00 | 675.40 | 670.58 | 668.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 674.00 | 675.07 | 672.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 09:15:00 | 670.00 | 675.07 | 672.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 670.75 | 674.20 | 672.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:15:00 | 669.10 | 674.20 | 672.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 667.10 | 672.78 | 671.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 667.50 | 672.78 | 671.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 676.15 | 672.84 | 671.83 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 659.35 | 669.91 | 670.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 657.15 | 667.36 | 669.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 12:15:00 | 667.00 | 666.52 | 668.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 13:00:00 | 667.00 | 666.52 | 668.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 661.40 | 664.28 | 666.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:30:00 | 663.50 | 664.28 | 666.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 675.30 | 659.08 | 660.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 675.30 | 659.08 | 660.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 669.00 | 661.06 | 661.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 663.00 | 661.06 | 661.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 629.85 | 639.37 | 645.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-22 14:15:00 | 631.60 | 630.30 | 638.04 | SL hit (close>ema200) qty=0.50 sl=630.30 alert=retest2 |

### Cycle 127 — BUY (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 15:15:00 | 641.10 | 635.00 | 634.74 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 627.75 | 633.55 | 634.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 624.35 | 631.71 | 633.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 630.00 | 629.32 | 631.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 14:15:00 | 630.00 | 629.32 | 631.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 630.00 | 629.32 | 631.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:00:00 | 630.00 | 629.32 | 631.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 627.85 | 629.03 | 631.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 626.40 | 629.03 | 631.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 625.60 | 628.34 | 630.62 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 638.00 | 631.97 | 631.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 14:15:00 | 638.90 | 634.02 | 632.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 631.30 | 634.11 | 633.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 631.30 | 634.11 | 633.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 631.30 | 634.11 | 633.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:30:00 | 631.95 | 634.11 | 633.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 632.00 | 633.69 | 633.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:45:00 | 634.95 | 633.69 | 633.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 635.85 | 634.12 | 633.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 09:45:00 | 642.00 | 634.98 | 633.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 10:15:00 | 641.90 | 634.98 | 633.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-06 09:15:00 | 706.20 | 680.84 | 672.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 689.55 | 697.13 | 697.59 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 14:15:00 | 713.50 | 698.00 | 697.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 15:15:00 | 718.00 | 702.00 | 699.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 12:15:00 | 700.90 | 705.05 | 701.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 12:15:00 | 700.90 | 705.05 | 701.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 700.90 | 705.05 | 701.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:45:00 | 700.20 | 705.05 | 701.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 698.30 | 703.70 | 701.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 698.90 | 703.70 | 701.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 688.90 | 700.74 | 700.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 688.90 | 700.74 | 700.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 690.20 | 698.63 | 699.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 681.90 | 695.28 | 697.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 10:15:00 | 700.05 | 696.24 | 698.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 10:15:00 | 700.05 | 696.24 | 698.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 700.05 | 696.24 | 698.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:45:00 | 708.10 | 696.24 | 698.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 700.00 | 696.99 | 698.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:45:00 | 695.80 | 698.25 | 698.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 15:15:00 | 701.95 | 698.99 | 698.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-13 15:15:00 | 701.95 | 698.99 | 698.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 09:15:00 | 705.90 | 700.37 | 699.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-14 10:15:00 | 699.80 | 700.26 | 699.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-14 11:00:00 | 699.80 | 700.26 | 699.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 701.40 | 700.49 | 699.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:45:00 | 696.75 | 700.49 | 699.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 700.45 | 700.48 | 699.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:45:00 | 695.00 | 700.48 | 699.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 13:15:00 | 693.95 | 699.17 | 699.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 15:15:00 | 692.05 | 697.06 | 698.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 697.25 | 697.10 | 698.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 697.25 | 697.10 | 698.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 697.25 | 697.10 | 698.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:00:00 | 697.25 | 697.10 | 698.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 696.75 | 696.42 | 697.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:00:00 | 696.75 | 696.42 | 697.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 697.20 | 696.58 | 697.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 697.20 | 696.58 | 697.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 694.25 | 696.11 | 697.20 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 707.80 | 698.27 | 697.98 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 695.30 | 699.52 | 699.72 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 14:15:00 | 710.55 | 701.21 | 700.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 741.15 | 710.64 | 704.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 784.00 | 784.29 | 770.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:30:00 | 784.15 | 784.29 | 770.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 777.00 | 783.72 | 774.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 773.65 | 783.72 | 774.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 782.00 | 783.38 | 775.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:30:00 | 783.00 | 783.24 | 776.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:15:00 | 784.70 | 783.24 | 776.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:00:00 | 785.50 | 783.69 | 776.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 10:15:00 | 786.90 | 796.40 | 797.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 10:15:00 | 786.90 | 796.40 | 797.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 11:15:00 | 785.35 | 794.19 | 796.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 13:15:00 | 783.80 | 783.09 | 788.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 13:45:00 | 782.25 | 783.09 | 788.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 785.90 | 783.65 | 787.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:30:00 | 781.40 | 783.65 | 787.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 790.00 | 784.92 | 788.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 792.95 | 784.92 | 788.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 789.20 | 785.78 | 788.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:30:00 | 785.10 | 786.45 | 787.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:00:00 | 784.15 | 786.45 | 787.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 14:15:00 | 811.85 | 791.91 | 790.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 811.85 | 791.91 | 790.23 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 785.80 | 792.38 | 792.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 780.95 | 790.09 | 791.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 15:15:00 | 769.35 | 769.19 | 773.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-17 09:15:00 | 769.50 | 769.19 | 773.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 767.00 | 768.75 | 772.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 764.80 | 768.75 | 772.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:15:00 | 764.40 | 761.82 | 765.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 15:15:00 | 763.95 | 763.20 | 765.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 10:00:00 | 765.20 | 763.72 | 765.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 769.60 | 764.90 | 765.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 10:30:00 | 772.95 | 764.90 | 765.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 766.80 | 765.28 | 765.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:15:00 | 767.75 | 765.28 | 765.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 766.40 | 765.50 | 765.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 14:45:00 | 763.75 | 765.19 | 765.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 769.15 | 766.41 | 766.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 09:15:00 | 769.15 | 766.41 | 766.23 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 762.10 | 765.81 | 766.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 751.30 | 762.91 | 764.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 739.00 | 731.87 | 738.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 14:15:00 | 739.00 | 731.87 | 738.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 739.00 | 731.87 | 738.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 739.00 | 731.87 | 738.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 734.95 | 732.49 | 738.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:15:00 | 731.50 | 732.42 | 737.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 730.30 | 731.74 | 736.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 731.50 | 732.93 | 735.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 743.45 | 735.03 | 736.25 | SL hit (close>static) qty=1.00 sl=739.55 alert=retest2 |

### Cycle 143 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 742.50 | 737.82 | 737.39 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 726.20 | 735.46 | 736.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 11:15:00 | 725.25 | 732.37 | 734.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 14:15:00 | 740.00 | 731.58 | 733.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 14:15:00 | 740.00 | 731.58 | 733.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 740.00 | 731.58 | 733.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 748.90 | 731.58 | 733.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 741.00 | 733.46 | 734.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 737.90 | 733.46 | 734.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 732.70 | 733.48 | 734.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 15:15:00 | 727.00 | 732.67 | 733.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 729.60 | 727.60 | 729.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 09:45:00 | 729.25 | 727.96 | 729.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 12:00:00 | 728.75 | 728.60 | 729.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 728.65 | 728.61 | 729.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 728.65 | 728.61 | 729.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 726.40 | 727.97 | 729.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 728.00 | 727.97 | 729.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 720.00 | 726.22 | 728.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:15:00 | 713.45 | 724.12 | 726.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:15:00 | 693.12 | 702.66 | 709.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 10:15:00 | 692.79 | 702.66 | 709.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:15:00 | 690.65 | 699.61 | 707.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 11:15:00 | 692.31 | 699.61 | 707.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:15:00 | 677.78 | 689.40 | 697.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-10 09:15:00 | 654.30 | 676.51 | 687.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 145 — BUY (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 09:15:00 | 645.70 | 636.21 | 635.13 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 14:15:00 | 632.00 | 635.41 | 635.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 628.05 | 633.55 | 634.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 600.25 | 600.01 | 609.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 600.25 | 600.01 | 609.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 607.50 | 602.66 | 606.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 607.50 | 602.66 | 606.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 605.65 | 603.26 | 606.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 605.65 | 603.26 | 606.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 606.00 | 603.81 | 606.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:30:00 | 606.90 | 603.81 | 606.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 616.50 | 606.35 | 607.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 616.50 | 606.35 | 607.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 627.90 | 610.66 | 609.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 633.00 | 621.98 | 617.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 15:15:00 | 678.05 | 678.16 | 668.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:15:00 | 656.00 | 678.16 | 668.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 658.00 | 674.13 | 667.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 652.15 | 674.13 | 667.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 654.45 | 670.19 | 666.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:15:00 | 654.00 | 670.19 | 666.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 647.80 | 663.67 | 664.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 637.15 | 653.39 | 657.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 639.15 | 635.15 | 645.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 639.15 | 635.15 | 645.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 612.80 | 631.36 | 641.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 586.80 | 600.29 | 607.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 584.85 | 595.27 | 604.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:00:00 | 587.20 | 588.76 | 597.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 584.15 | 589.04 | 597.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 589.50 | 586.69 | 592.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 593.15 | 586.69 | 592.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 588.00 | 586.95 | 592.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:15:00 | 588.00 | 586.95 | 592.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 588.00 | 587.16 | 591.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:45:00 | 598.75 | 588.77 | 592.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 598.15 | 590.64 | 592.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:45:00 | 598.20 | 590.64 | 592.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 594.30 | 593.18 | 593.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:30:00 | 594.50 | 593.18 | 593.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 593.25 | 593.20 | 593.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 600.00 | 593.20 | 593.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 602.05 | 594.97 | 594.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 602.05 | 594.97 | 594.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 10:15:00 | 606.25 | 597.22 | 595.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 12:15:00 | 598.85 | 599.20 | 596.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-19 13:00:00 | 598.85 | 599.20 | 596.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 599.85 | 600.20 | 598.00 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 587.75 | 597.78 | 598.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 584.55 | 590.67 | 593.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 11:15:00 | 591.45 | 590.83 | 593.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 11:15:00 | 591.45 | 590.83 | 593.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 591.45 | 590.83 | 593.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:00:00 | 591.45 | 590.83 | 593.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 594.00 | 591.46 | 593.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 13:45:00 | 591.40 | 591.37 | 593.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:30:00 | 589.20 | 591.13 | 593.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:30:00 | 588.60 | 591.28 | 592.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 14:15:00 | 585.25 | 579.69 | 579.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 585.25 | 579.69 | 579.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 596.60 | 583.32 | 581.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 12:15:00 | 583.55 | 584.36 | 582.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 12:15:00 | 583.55 | 584.36 | 582.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 583.55 | 584.36 | 582.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 13:00:00 | 583.55 | 584.36 | 582.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 582.85 | 584.01 | 582.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 15:00:00 | 582.85 | 584.01 | 582.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 583.05 | 583.82 | 582.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 09:15:00 | 589.40 | 583.82 | 582.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 10:15:00 | 586.50 | 584.27 | 582.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 11:15:00 | 580.80 | 583.44 | 582.75 | SL hit (close<static) qty=1.00 sl=581.95 alert=retest2 |

### Cycle 152 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 577.00 | 586.55 | 587.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 13:15:00 | 573.00 | 578.67 | 582.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 14:15:00 | 572.85 | 567.70 | 570.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 14:15:00 | 572.85 | 567.70 | 570.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 572.85 | 567.70 | 570.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 572.85 | 567.70 | 570.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 568.20 | 567.80 | 570.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 566.00 | 567.80 | 570.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 15:00:00 | 565.65 | 558.23 | 558.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 15:15:00 | 564.80 | 559.55 | 559.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 15:15:00 | 564.80 | 559.55 | 559.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 574.70 | 562.58 | 560.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 12:15:00 | 583.75 | 583.95 | 575.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 12:30:00 | 584.70 | 583.95 | 575.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 573.80 | 580.73 | 577.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 573.80 | 580.73 | 577.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 573.25 | 579.23 | 577.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 575.65 | 579.23 | 577.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 577.00 | 578.41 | 577.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:30:00 | 575.95 | 578.41 | 577.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 577.60 | 578.25 | 577.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 573.55 | 578.25 | 577.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 574.00 | 577.40 | 576.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 571.20 | 577.40 | 576.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 566.50 | 575.22 | 575.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 560.10 | 570.07 | 573.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 563.50 | 560.81 | 566.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 11:00:00 | 563.50 | 560.81 | 566.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 567.15 | 562.17 | 565.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 567.15 | 562.17 | 565.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 562.35 | 562.21 | 565.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 565.15 | 562.21 | 565.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 559.30 | 560.89 | 564.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:15:00 | 553.30 | 558.94 | 561.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 573.20 | 563.96 | 562.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 573.20 | 563.96 | 562.94 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 555.55 | 566.39 | 566.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 550.60 | 559.03 | 562.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 13:15:00 | 555.15 | 551.89 | 556.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 13:15:00 | 555.15 | 551.89 | 556.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 13:15:00 | 555.15 | 551.89 | 556.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 13:45:00 | 556.35 | 551.89 | 556.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 557.00 | 552.92 | 556.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 14:30:00 | 562.50 | 552.92 | 556.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 559.85 | 554.30 | 556.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 564.65 | 554.30 | 556.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 568.45 | 557.13 | 557.63 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 569.30 | 559.57 | 558.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 572.70 | 562.19 | 559.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 557.90 | 563.31 | 561.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 557.90 | 563.31 | 561.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 557.90 | 563.31 | 561.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:30:00 | 559.45 | 563.31 | 561.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 562.10 | 563.07 | 561.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 11:45:00 | 564.05 | 562.87 | 561.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:45:00 | 564.45 | 563.10 | 561.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-11 11:15:00 | 620.46 | 587.57 | 575.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 616.70 | 629.27 | 630.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 613.15 | 621.42 | 625.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 617.40 | 616.28 | 619.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 617.40 | 616.28 | 619.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 617.40 | 616.28 | 619.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:45:00 | 614.75 | 616.14 | 619.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 614.00 | 616.56 | 618.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 09:15:00 | 584.01 | 594.89 | 604.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 09:15:00 | 583.30 | 594.89 | 604.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-02 11:15:00 | 595.00 | 594.86 | 602.73 | SL hit (close>ema200) qty=0.50 sl=594.86 alert=retest2 |

### Cycle 159 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 680.35 | 611.35 | 605.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 14:15:00 | 694.25 | 657.60 | 633.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 734.90 | 736.94 | 707.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 13:45:00 | 737.25 | 736.94 | 707.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 722.05 | 730.39 | 711.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:15:00 | 733.50 | 727.93 | 712.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 741.65 | 725.13 | 716.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 10:00:00 | 730.90 | 729.71 | 724.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 10:30:00 | 734.70 | 730.18 | 725.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 724.25 | 729.05 | 725.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 724.25 | 729.05 | 725.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 720.00 | 727.24 | 725.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 720.00 | 727.24 | 725.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 727.75 | 727.81 | 726.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 13:00:00 | 727.75 | 727.81 | 726.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 727.90 | 727.83 | 726.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 736.85 | 728.17 | 726.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:00:00 | 729.95 | 729.92 | 727.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:45:00 | 730.15 | 729.07 | 728.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 14:15:00 | 724.20 | 728.10 | 727.70 | SL hit (close<static) qty=1.00 sl=725.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 813.00 | 820.35 | 820.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 807.05 | 815.55 | 818.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 823.55 | 815.86 | 817.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 15:15:00 | 823.55 | 815.86 | 817.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 823.55 | 815.86 | 817.77 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 841.00 | 820.89 | 819.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 892.75 | 847.25 | 834.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 885.00 | 885.97 | 874.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:15:00 | 892.65 | 885.97 | 874.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 880.15 | 884.80 | 875.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 881.85 | 884.80 | 875.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 886.65 | 895.01 | 890.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 890.65 | 895.01 | 890.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 885.00 | 893.01 | 890.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 884.85 | 893.01 | 890.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 874.15 | 887.43 | 888.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 11:15:00 | 857.80 | 870.76 | 876.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 11:15:00 | 860.95 | 859.81 | 866.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 12:00:00 | 860.95 | 859.81 | 866.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 834.25 | 829.90 | 838.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:30:00 | 835.75 | 829.90 | 838.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 819.50 | 828.16 | 836.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 13:30:00 | 808.00 | 820.17 | 829.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 812.00 | 815.25 | 823.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 798.00 | 794.81 | 795.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 804.95 | 796.84 | 796.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 804.95 | 796.84 | 796.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 810.35 | 799.54 | 797.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 842.45 | 844.48 | 832.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:30:00 | 841.20 | 844.48 | 832.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 840.05 | 844.21 | 838.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:45:00 | 835.55 | 844.21 | 838.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 838.90 | 843.15 | 838.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 840.70 | 843.15 | 838.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 834.50 | 841.42 | 838.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:30:00 | 849.00 | 843.00 | 839.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 835.55 | 843.96 | 844.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 835.55 | 843.96 | 844.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 832.35 | 841.64 | 843.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 14:15:00 | 840.00 | 838.76 | 841.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 15:00:00 | 840.00 | 838.76 | 841.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 831.50 | 837.31 | 840.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 820.30 | 837.31 | 840.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 854.10 | 833.58 | 836.10 | SL hit (close>static) qty=1.00 sl=841.55 alert=retest2 |

### Cycle 165 — BUY (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 15:15:00 | 865.00 | 839.87 | 838.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 876.05 | 854.72 | 846.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 15:15:00 | 880.10 | 881.78 | 870.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:15:00 | 883.05 | 881.78 | 870.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 887.75 | 890.75 | 886.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 898.90 | 890.64 | 888.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 898.05 | 893.26 | 889.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 15:15:00 | 880.00 | 887.26 | 888.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 880.00 | 887.26 | 888.12 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 900.10 | 889.83 | 889.21 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 883.85 | 888.69 | 889.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 10:15:00 | 880.40 | 887.16 | 888.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 884.75 | 877.19 | 881.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 884.75 | 877.19 | 881.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 884.75 | 877.19 | 881.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:45:00 | 883.70 | 877.19 | 881.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 885.90 | 878.93 | 882.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:30:00 | 888.70 | 878.93 | 882.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 886.10 | 881.27 | 882.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 886.10 | 881.27 | 882.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 886.80 | 882.38 | 883.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 886.80 | 882.38 | 883.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 888.50 | 883.60 | 883.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 895.85 | 886.28 | 884.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 874.80 | 885.84 | 885.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 14:15:00 | 874.80 | 885.84 | 885.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 874.80 | 885.84 | 885.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 874.80 | 885.84 | 885.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 878.30 | 884.34 | 884.83 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 888.00 | 885.50 | 885.30 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 874.90 | 884.02 | 885.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 872.05 | 877.70 | 881.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 872.80 | 870.34 | 874.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 887.15 | 870.34 | 874.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 843.90 | 865.05 | 871.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 13:30:00 | 837.55 | 847.89 | 853.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 838.00 | 844.14 | 850.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:30:00 | 835.75 | 838.47 | 843.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:30:00 | 835.90 | 839.15 | 843.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 843.25 | 839.97 | 843.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:30:00 | 843.70 | 839.97 | 843.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 842.90 | 840.56 | 843.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:30:00 | 844.00 | 840.56 | 843.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 839.30 | 840.31 | 842.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:30:00 | 842.85 | 840.31 | 842.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 839.85 | 840.21 | 842.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 839.85 | 840.21 | 842.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 840.00 | 840.17 | 842.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 841.00 | 840.17 | 842.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 843.60 | 840.86 | 842.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 843.60 | 840.86 | 842.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 840.15 | 840.72 | 842.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 837.90 | 840.15 | 841.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:30:00 | 836.60 | 839.57 | 841.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 860.00 | 843.46 | 842.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 860.00 | 843.46 | 842.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 869.00 | 848.57 | 845.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 12:15:00 | 849.30 | 849.40 | 846.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 13:00:00 | 849.30 | 849.40 | 846.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 849.35 | 849.39 | 846.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:45:00 | 842.45 | 849.39 | 846.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 848.70 | 849.26 | 847.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 848.70 | 849.26 | 847.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 848.75 | 849.15 | 847.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 858.75 | 849.15 | 847.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 861.60 | 851.64 | 848.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:30:00 | 872.00 | 860.50 | 855.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 859.25 | 894.25 | 896.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 859.25 | 894.25 | 896.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 846.60 | 867.12 | 879.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 861.50 | 858.93 | 870.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 861.50 | 858.93 | 870.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 868.00 | 857.30 | 863.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 868.00 | 857.30 | 863.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 870.00 | 859.84 | 864.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 859.45 | 859.84 | 864.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 857.80 | 859.44 | 863.56 | EMA400 retest candle locked (from downside) |

### Cycle 175 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 864.90 | 863.96 | 863.90 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 857.05 | 862.75 | 863.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 844.15 | 859.03 | 861.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 15:15:00 | 863.00 | 856.42 | 858.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 15:15:00 | 863.00 | 856.42 | 858.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 863.00 | 856.42 | 858.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:30:00 | 852.85 | 856.14 | 858.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:00:00 | 853.55 | 855.62 | 857.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 890.80 | 864.03 | 860.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 890.80 | 864.03 | 860.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 896.30 | 875.37 | 866.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 908.85 | 910.35 | 894.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:30:00 | 907.95 | 910.35 | 894.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 901.60 | 907.62 | 901.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 901.60 | 907.62 | 901.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 899.50 | 906.00 | 901.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 893.55 | 906.00 | 901.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 889.35 | 902.67 | 900.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:00:00 | 889.35 | 902.67 | 900.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 890.00 | 900.13 | 899.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 889.15 | 900.13 | 899.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 889.80 | 898.07 | 898.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 888.00 | 893.33 | 895.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 14:15:00 | 878.05 | 877.42 | 883.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 15:00:00 | 878.05 | 877.42 | 883.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 885.00 | 878.93 | 883.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 868.05 | 878.93 | 883.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 864.05 | 875.96 | 882.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:30:00 | 853.55 | 866.20 | 873.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:15:00 | 852.20 | 862.00 | 868.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 889.05 | 873.10 | 870.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 889.05 | 873.10 | 870.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 900.00 | 878.48 | 873.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 930.80 | 931.76 | 922.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:00:00 | 930.80 | 931.76 | 922.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 923.90 | 929.70 | 922.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 923.90 | 929.70 | 922.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 924.70 | 928.70 | 922.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 922.75 | 928.70 | 922.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 912.80 | 925.52 | 921.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 912.80 | 925.52 | 921.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 909.15 | 922.25 | 920.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 908.90 | 922.25 | 920.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 912.35 | 918.66 | 919.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 11:15:00 | 908.65 | 914.42 | 916.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 914.35 | 901.88 | 905.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 914.35 | 901.88 | 905.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 914.35 | 901.88 | 905.48 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 914.90 | 908.33 | 907.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 918.00 | 911.40 | 909.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 902.05 | 910.11 | 909.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 902.05 | 910.11 | 909.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 902.05 | 910.11 | 909.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 904.60 | 910.11 | 909.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 905.60 | 909.21 | 908.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:00:00 | 908.00 | 908.96 | 908.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 13:00:00 | 908.20 | 908.81 | 908.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 904.55 | 907.96 | 908.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 904.55 | 907.96 | 908.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 902.00 | 906.77 | 907.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 904.60 | 899.86 | 902.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 904.60 | 899.86 | 902.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 904.60 | 899.86 | 902.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 904.60 | 899.86 | 902.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 903.00 | 900.49 | 902.81 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 908.65 | 903.60 | 903.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 921.90 | 907.89 | 905.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 911.70 | 912.54 | 908.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 910.00 | 911.96 | 909.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 910.00 | 911.96 | 909.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:45:00 | 910.20 | 911.96 | 909.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 907.05 | 910.98 | 909.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 907.05 | 910.98 | 909.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 906.40 | 910.06 | 909.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 900.30 | 910.06 | 909.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 900.50 | 908.15 | 908.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 10:15:00 | 874.50 | 901.42 | 905.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 886.45 | 884.50 | 890.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 886.45 | 884.50 | 890.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 886.45 | 884.50 | 890.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:30:00 | 880.85 | 884.50 | 890.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 889.90 | 886.07 | 889.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 889.90 | 886.07 | 889.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 895.05 | 887.87 | 890.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 895.05 | 887.87 | 890.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 894.00 | 889.09 | 890.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:45:00 | 895.20 | 889.09 | 890.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 898.25 | 892.93 | 892.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 11:15:00 | 902.10 | 898.07 | 895.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 15:15:00 | 897.00 | 899.60 | 897.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 15:15:00 | 897.00 | 899.60 | 897.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 897.00 | 899.60 | 897.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 902.40 | 899.60 | 897.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 15:15:00 | 897.15 | 904.20 | 905.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 897.15 | 904.20 | 905.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 888.60 | 901.08 | 903.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 893.25 | 890.79 | 896.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:30:00 | 895.00 | 890.79 | 896.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 862.00 | 853.98 | 858.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 862.00 | 853.98 | 858.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 866.80 | 856.54 | 859.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 864.35 | 856.54 | 859.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 854.50 | 858.07 | 859.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 852.40 | 858.07 | 859.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:45:00 | 853.25 | 856.41 | 858.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 841.45 | 831.75 | 830.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 841.45 | 831.75 | 830.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 849.50 | 838.21 | 834.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 12:15:00 | 840.70 | 840.96 | 837.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:15:00 | 841.45 | 840.96 | 837.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 840.75 | 842.99 | 840.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 840.75 | 842.99 | 840.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 838.70 | 842.14 | 840.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 838.70 | 842.14 | 840.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 834.75 | 840.66 | 839.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 840.40 | 839.33 | 839.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 835.10 | 842.43 | 843.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 835.10 | 842.43 | 843.09 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 851.40 | 844.22 | 843.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 854.00 | 849.09 | 846.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 15:15:00 | 850.10 | 850.20 | 848.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 09:15:00 | 850.85 | 850.20 | 848.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 858.85 | 851.93 | 849.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 861.35 | 851.93 | 849.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:30:00 | 860.85 | 855.34 | 851.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:45:00 | 862.80 | 860.25 | 855.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 861.15 | 857.41 | 856.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 865.90 | 859.11 | 857.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:30:00 | 870.05 | 862.58 | 859.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:15:00 | 871.15 | 862.58 | 859.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:30:00 | 869.40 | 867.14 | 862.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-06 09:15:00 | 947.49 | 906.10 | 886.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1017.95 | 1044.41 | 1046.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 1014.50 | 1022.86 | 1027.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 986.20 | 985.40 | 993.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 986.20 | 985.40 | 993.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1019.05 | 991.59 | 994.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 1010.35 | 991.59 | 994.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 1020.45 | 997.37 | 996.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1022.85 | 1009.51 | 1003.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 998.85 | 1010.73 | 1005.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 998.85 | 1010.73 | 1005.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 998.85 | 1010.73 | 1005.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:00:00 | 998.85 | 1010.73 | 1005.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 997.50 | 1008.09 | 1004.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 997.50 | 1008.09 | 1004.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1001.15 | 1006.70 | 1004.53 | EMA400 retest candle locked (from upside) |

### Cycle 192 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 988.30 | 1001.24 | 1002.31 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 1008.70 | 1003.29 | 1002.78 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 991.30 | 1002.45 | 1003.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 986.80 | 999.32 | 1001.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 981.90 | 981.56 | 988.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:00:00 | 981.90 | 981.56 | 988.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 987.80 | 982.80 | 988.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 987.80 | 982.80 | 988.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 986.90 | 983.62 | 988.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 988.10 | 983.62 | 988.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 996.60 | 986.22 | 989.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 997.80 | 986.22 | 989.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1003.90 | 989.75 | 990.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 1003.90 | 989.75 | 990.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 998.40 | 991.48 | 991.33 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 990.70 | 992.24 | 992.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 976.30 | 989.05 | 990.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 955.00 | 954.61 | 966.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 955.00 | 954.61 | 966.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 955.00 | 954.61 | 966.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 965.20 | 954.61 | 966.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 942.90 | 952.27 | 964.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 940.00 | 949.23 | 962.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 15:15:00 | 939.30 | 942.75 | 955.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 971.10 | 955.89 | 956.70 | SL hit (close>static) qty=1.00 sl=965.60 alert=retest2 |

### Cycle 197 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 975.50 | 959.81 | 958.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 987.70 | 971.03 | 967.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 993.10 | 997.96 | 984.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:45:00 | 993.80 | 997.96 | 984.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 986.70 | 994.45 | 985.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 983.80 | 994.45 | 985.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 988.10 | 993.18 | 985.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:30:00 | 985.50 | 993.18 | 985.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 985.30 | 991.09 | 985.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 985.30 | 991.09 | 985.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 989.00 | 990.68 | 986.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 983.60 | 990.68 | 986.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 982.40 | 989.02 | 985.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 981.40 | 989.02 | 985.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 982.00 | 987.62 | 985.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:45:00 | 981.70 | 987.62 | 985.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 976.30 | 982.72 | 983.58 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 1001.00 | 986.35 | 984.96 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 977.30 | 986.37 | 986.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 11:15:00 | 970.60 | 981.04 | 983.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 982.90 | 975.83 | 979.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 982.90 | 975.83 | 979.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 982.90 | 975.83 | 979.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:45:00 | 983.20 | 975.83 | 979.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 981.20 | 976.90 | 979.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 984.20 | 976.90 | 979.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 982.50 | 979.88 | 980.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:45:00 | 983.90 | 979.88 | 980.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 981.90 | 980.38 | 980.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 985.50 | 980.38 | 980.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 983.90 | 981.09 | 980.88 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 976.80 | 980.23 | 980.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 11:15:00 | 973.90 | 978.96 | 979.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 943.40 | 941.43 | 950.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 943.40 | 941.43 | 950.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 952.10 | 944.85 | 950.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 952.10 | 944.85 | 950.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 959.70 | 947.82 | 951.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 959.70 | 947.82 | 951.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 962.50 | 950.76 | 952.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 956.10 | 952.57 | 952.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 954.90 | 952.57 | 952.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 14:15:00 | 908.29 | 928.46 | 938.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 14:15:00 | 907.15 | 928.46 | 938.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 930.05 | 919.04 | 924.74 | SL hit (close>ema200) qty=0.50 sl=919.04 alert=retest2 |

### Cycle 203 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 935.95 | 927.36 | 927.31 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 926.15 | 927.11 | 927.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 920.90 | 925.87 | 926.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 924.60 | 924.35 | 925.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 924.60 | 924.35 | 925.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 924.60 | 924.35 | 925.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 924.60 | 924.35 | 925.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 918.65 | 923.21 | 925.09 | EMA400 retest candle locked (from downside) |

### Cycle 205 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 934.60 | 927.12 | 926.39 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 915.30 | 926.33 | 926.97 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 934.00 | 926.83 | 926.45 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 919.95 | 925.45 | 925.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 912.20 | 922.80 | 924.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 938.20 | 923.83 | 924.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 14:15:00 | 938.20 | 923.83 | 924.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 938.20 | 923.83 | 924.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 15:00:00 | 938.20 | 923.83 | 924.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — BUY (started 2026-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 15:15:00 | 941.00 | 927.27 | 926.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 10:15:00 | 948.25 | 940.71 | 935.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 940.40 | 940.65 | 935.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 940.40 | 940.65 | 935.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 998.00 | 974.11 | 964.55 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 954.95 | 969.51 | 970.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 950.30 | 965.67 | 968.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 953.00 | 946.07 | 954.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:30:00 | 938.30 | 943.39 | 951.53 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 932.00 | 928.14 | 934.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 932.35 | 928.14 | 934.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 946.30 | 930.92 | 934.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-27 11:15:00 | 946.30 | 930.92 | 934.64 | SL hit (close>ema400) qty=1.00 sl=934.64 alert=retest1 |

### Cycle 211 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 960.05 | 937.35 | 936.72 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 15:15:00 | 931.00 | 939.53 | 939.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 922.45 | 936.12 | 938.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 14:15:00 | 932.75 | 931.20 | 934.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 932.75 | 931.20 | 934.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 932.75 | 931.20 | 934.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 932.75 | 931.20 | 934.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 936.60 | 932.28 | 934.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 930.05 | 932.28 | 934.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 955.70 | 936.96 | 936.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 955.70 | 936.96 | 936.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 12:15:00 | 963.30 | 948.08 | 942.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 951.00 | 958.63 | 951.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 951.00 | 958.63 | 951.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 951.00 | 958.63 | 951.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:30:00 | 951.95 | 958.63 | 951.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 951.35 | 957.17 | 951.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 954.00 | 957.17 | 951.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 953.70 | 956.48 | 951.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 953.70 | 956.48 | 951.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 974.95 | 960.17 | 953.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:30:00 | 978.05 | 965.05 | 957.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:45:00 | 978.45 | 965.07 | 961.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 13:45:00 | 978.05 | 969.55 | 963.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 14:30:00 | 978.25 | 971.64 | 965.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 993.80 | 1002.32 | 991.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 993.80 | 1002.32 | 991.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 998.00 | 1001.45 | 992.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 975.45 | 988.96 | 989.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 975.45 | 988.96 | 989.24 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1012.40 | 992.58 | 990.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 15:15:00 | 1031.90 | 1027.26 | 1017.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1018.55 | 1025.52 | 1017.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 1018.55 | 1025.52 | 1017.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1018.55 | 1025.52 | 1017.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1010.95 | 1025.52 | 1017.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1005.80 | 1021.58 | 1016.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1005.80 | 1021.58 | 1016.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 996.95 | 1016.65 | 1014.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 996.95 | 1016.65 | 1014.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 1002.50 | 1011.16 | 1012.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 993.20 | 1008.38 | 1010.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 996.70 | 994.20 | 1001.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 15:00:00 | 996.70 | 994.20 | 1001.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 1004.85 | 995.20 | 999.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 1004.85 | 995.20 | 999.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1010.95 | 998.35 | 1000.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:30:00 | 999.50 | 999.21 | 1000.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 1037.40 | 1005.69 | 1003.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 1037.40 | 1005.69 | 1003.25 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 14:15:00 | 1006.35 | 1009.52 | 1009.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 10:15:00 | 998.35 | 1005.79 | 1007.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 12:15:00 | 1000.70 | 995.71 | 1000.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 12:15:00 | 1000.70 | 995.71 | 1000.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1000.70 | 995.71 | 1000.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 1002.50 | 995.71 | 1000.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 997.55 | 996.08 | 999.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:00:00 | 991.75 | 995.21 | 999.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:00:00 | 990.50 | 993.44 | 997.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 990.15 | 992.78 | 996.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 12:45:00 | 992.50 | 993.22 | 996.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 994.00 | 993.40 | 995.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 1028.15 | 993.40 | 995.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 1038.95 | 1002.51 | 999.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1038.95 | 1002.51 | 999.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 1040.15 | 1010.04 | 1003.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 10:15:00 | 1044.00 | 1049.89 | 1037.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:00:00 | 1044.00 | 1049.89 | 1037.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 1043.75 | 1048.67 | 1038.47 | EMA400 retest candle locked (from upside) |

### Cycle 220 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 1032.75 | 1036.33 | 1036.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1014.35 | 1032.04 | 1034.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 10:15:00 | 1016.30 | 1012.01 | 1020.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 11:00:00 | 1016.30 | 1012.01 | 1020.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1018.80 | 1012.49 | 1017.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 1016.20 | 1012.49 | 1017.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1019.20 | 1013.83 | 1018.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1002.80 | 1013.83 | 1018.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1021.10 | 1015.29 | 1018.30 | SL hit (close>static) qty=1.00 sl=1019.40 alert=retest2 |

### Cycle 221 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 1022.00 | 1017.72 | 1017.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 1027.90 | 1021.27 | 1019.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 1014.70 | 1021.80 | 1020.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 10:15:00 | 1014.70 | 1021.80 | 1020.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1014.70 | 1021.80 | 1020.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 1014.70 | 1021.80 | 1020.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1014.40 | 1020.32 | 1019.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 1014.40 | 1020.32 | 1019.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 13:15:00 | 1005.10 | 1016.57 | 1018.10 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 1026.30 | 1017.20 | 1016.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 1056.10 | 1024.98 | 1020.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 1038.40 | 1042.12 | 1036.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 15:15:00 | 1038.40 | 1042.12 | 1036.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1038.40 | 1042.12 | 1036.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1032.00 | 1042.12 | 1036.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1043.40 | 1042.37 | 1037.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:15:00 | 1048.00 | 1043.48 | 1038.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1019.90 | 1039.94 | 1041.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 1019.90 | 1039.94 | 1041.15 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 12:15:00 | 1046.70 | 1039.10 | 1038.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 1049.60 | 1043.28 | 1041.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 11:15:00 | 1042.10 | 1043.17 | 1041.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 11:15:00 | 1042.10 | 1043.17 | 1041.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1042.10 | 1043.17 | 1041.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:45:00 | 1038.70 | 1043.17 | 1041.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1034.00 | 1041.34 | 1040.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 1034.00 | 1041.34 | 1040.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 226 — SELL (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 13:15:00 | 1027.00 | 1038.47 | 1039.52 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1043.00 | 1040.02 | 1040.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 1051.00 | 1045.00 | 1042.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 11:15:00 | 1044.60 | 1045.10 | 1043.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 12:00:00 | 1044.60 | 1045.10 | 1043.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 1045.00 | 1045.08 | 1043.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 1045.00 | 1045.08 | 1043.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1043.70 | 1044.80 | 1043.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:15:00 | 1041.50 | 1044.80 | 1043.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1032.60 | 1042.36 | 1042.55 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1091.50 | 1051.01 | 1046.38 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1058.80 | 1070.98 | 1071.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1056.00 | 1064.70 | 1068.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1075.00 | 1053.02 | 1057.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1075.00 | 1053.02 | 1057.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1075.00 | 1053.02 | 1057.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 1081.90 | 1053.02 | 1057.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1072.60 | 1056.94 | 1059.11 | EMA400 retest candle locked (from downside) |

### Cycle 231 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1078.00 | 1063.80 | 1062.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 1080.50 | 1068.87 | 1064.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1040.50 | 1064.24 | 1063.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1040.50 | 1064.24 | 1063.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1040.50 | 1064.24 | 1063.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 1040.50 | 1064.24 | 1063.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1040.50 | 1059.49 | 1061.33 | EMA200 below EMA400 |

### Cycle 233 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 1098.90 | 1066.93 | 1063.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 1102.50 | 1081.59 | 1072.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 12:15:00 | 1127.80 | 1139.44 | 1124.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 12:45:00 | 1127.30 | 1139.44 | 1124.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 1116.10 | 1134.77 | 1123.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:00:00 | 1116.10 | 1134.77 | 1123.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 1124.30 | 1132.68 | 1123.42 | EMA400 retest candle locked (from upside) |

### Cycle 234 — SELL (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 10:15:00 | 1073.40 | 1110.61 | 1114.99 | EMA200 below EMA400 |

### Cycle 235 — BUY (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 15:15:00 | 1098.70 | 1090.84 | 1090.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 1104.00 | 1093.47 | 1091.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 11:15:00 | 1093.70 | 1094.75 | 1092.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 11:15:00 | 1093.70 | 1094.75 | 1092.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 11:15:00 | 1093.70 | 1094.75 | 1092.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:00:00 | 1093.70 | 1094.75 | 1092.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 1093.90 | 1094.58 | 1092.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 13:00:00 | 1093.90 | 1094.58 | 1092.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 1081.20 | 1091.91 | 1091.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 1081.20 | 1091.91 | 1091.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — SELL (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 14:15:00 | 1076.90 | 1088.90 | 1090.40 | EMA200 below EMA400 |

### Cycle 237 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 1090.00 | 1085.01 | 1084.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 1095.20 | 1087.72 | 1086.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 1087.30 | 1089.08 | 1087.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 13:15:00 | 1087.30 | 1089.08 | 1087.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 1087.30 | 1089.08 | 1087.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 1087.30 | 1089.08 | 1087.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 1087.60 | 1088.78 | 1087.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 1087.60 | 1088.78 | 1087.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1088.00 | 1088.62 | 1087.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1117.40 | 1088.62 | 1087.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 1090.90 | 1104.86 | 1106.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 238 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 1090.90 | 1104.86 | 1106.13 | EMA200 below EMA400 |

### Cycle 239 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1122.90 | 1106.82 | 1104.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1127.80 | 1111.02 | 1106.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1105.10 | 1109.98 | 1107.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 10:15:00 | 1105.10 | 1109.98 | 1107.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1105.10 | 1109.98 | 1107.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 1105.30 | 1109.98 | 1107.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1113.00 | 1110.58 | 1107.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 1121.80 | 1110.96 | 1108.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 1136.60 | 1170.41 | 1174.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 240 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 1136.60 | 1170.41 | 1174.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 12:15:00 | 1127.10 | 1161.75 | 1170.20 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-24 14:30:00 | 608.00 | 2023-06-08 11:15:00 | 628.60 | STOP_HIT | 1.00 | 3.39% |
| BUY | retest1 | 2023-06-20 09:15:00 | 664.85 | 2023-06-20 13:15:00 | 656.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-06-21 09:15:00 | 657.80 | 2023-06-21 15:15:00 | 651.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-06-21 12:00:00 | 658.00 | 2023-06-21 15:15:00 | 651.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-06-27 12:15:00 | 646.55 | 2023-06-28 09:15:00 | 650.05 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-07-03 14:45:00 | 667.10 | 2023-07-07 09:15:00 | 733.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-03 15:15:00 | 666.95 | 2023-07-07 09:15:00 | 733.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-04 11:15:00 | 667.60 | 2023-07-07 09:15:00 | 734.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-04 12:15:00 | 666.95 | 2023-07-07 09:15:00 | 733.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-12 12:00:00 | 723.35 | 2023-07-17 09:15:00 | 704.50 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2023-07-25 12:00:00 | 616.65 | 2023-08-01 10:15:00 | 617.35 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2023-07-26 09:30:00 | 616.60 | 2023-08-01 10:15:00 | 617.35 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2023-07-27 15:15:00 | 616.15 | 2023-08-01 10:15:00 | 617.35 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2023-08-03 15:15:00 | 622.50 | 2023-08-04 15:15:00 | 614.60 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-09-28 09:15:00 | 633.85 | 2023-09-28 09:15:00 | 628.65 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-10-05 14:30:00 | 656.55 | 2023-10-06 11:15:00 | 642.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2023-10-11 14:30:00 | 658.65 | 2023-10-12 12:15:00 | 644.20 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2023-10-26 09:15:00 | 611.05 | 2023-11-02 11:15:00 | 615.75 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest1 | 2023-11-03 09:30:00 | 618.40 | 2023-11-03 14:15:00 | 610.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2023-11-06 10:00:00 | 622.25 | 2023-11-09 12:15:00 | 619.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-11-08 10:15:00 | 618.65 | 2023-11-09 12:15:00 | 619.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2023-11-15 15:00:00 | 635.80 | 2023-11-17 13:15:00 | 627.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-11-16 09:15:00 | 639.55 | 2023-11-17 13:15:00 | 627.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2023-11-17 09:45:00 | 637.80 | 2023-11-17 13:15:00 | 627.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-11-17 10:30:00 | 636.00 | 2023-11-17 13:15:00 | 627.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2023-12-06 09:45:00 | 660.60 | 2023-12-07 15:15:00 | 653.60 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-12-07 09:30:00 | 660.40 | 2023-12-07 15:15:00 | 653.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-12-07 10:15:00 | 662.50 | 2023-12-07 15:15:00 | 653.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-12-13 09:30:00 | 643.05 | 2023-12-13 14:15:00 | 654.45 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2023-12-13 10:15:00 | 641.80 | 2023-12-13 14:15:00 | 654.45 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2023-12-13 12:30:00 | 643.10 | 2023-12-13 14:15:00 | 654.45 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2023-12-20 11:45:00 | 629.25 | 2023-12-21 10:15:00 | 637.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2023-12-20 12:15:00 | 629.25 | 2023-12-21 10:15:00 | 637.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-01-03 10:30:00 | 625.65 | 2024-01-05 09:15:00 | 634.25 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-01-08 15:15:00 | 627.30 | 2024-01-09 10:15:00 | 633.55 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-01-10 14:15:00 | 640.45 | 2024-01-15 13:15:00 | 635.40 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-01-11 09:15:00 | 641.10 | 2024-01-15 13:15:00 | 635.40 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-01-11 14:30:00 | 642.00 | 2024-01-15 13:15:00 | 635.40 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-01-12 13:30:00 | 640.95 | 2024-01-15 13:15:00 | 635.40 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-01-17 09:15:00 | 628.00 | 2024-01-19 10:15:00 | 636.40 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-02-08 15:00:00 | 665.90 | 2024-02-09 09:15:00 | 650.90 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-03-02 11:45:00 | 628.00 | 2024-03-12 10:15:00 | 596.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 09:15:00 | 627.75 | 2024-03-12 10:15:00 | 596.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 09:45:00 | 628.00 | 2024-03-12 10:15:00 | 596.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 10:15:00 | 626.80 | 2024-03-12 11:15:00 | 595.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 09:15:00 | 620.25 | 2024-03-12 12:15:00 | 589.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-02 11:45:00 | 628.00 | 2024-03-12 13:15:00 | 606.35 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2024-03-04 09:15:00 | 627.75 | 2024-03-12 13:15:00 | 606.35 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2024-03-04 09:45:00 | 628.00 | 2024-03-12 13:15:00 | 606.35 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2024-03-05 10:15:00 | 626.80 | 2024-03-12 13:15:00 | 606.35 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2024-03-06 09:15:00 | 620.25 | 2024-03-12 13:15:00 | 606.35 | STOP_HIT | 0.50 | 2.24% |
| BUY | retest2 | 2024-03-27 13:00:00 | 590.00 | 2024-03-27 13:15:00 | 589.30 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-04-25 09:45:00 | 583.80 | 2024-04-25 10:15:00 | 576.60 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-04-25 14:00:00 | 584.00 | 2024-05-02 14:15:00 | 582.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-04-25 14:45:00 | 584.00 | 2024-05-02 14:15:00 | 582.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-04-26 11:15:00 | 586.75 | 2024-05-02 14:15:00 | 582.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-04-29 09:15:00 | 583.85 | 2024-05-02 14:15:00 | 582.40 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-04-29 11:00:00 | 584.35 | 2024-05-02 14:15:00 | 582.40 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-04-29 14:00:00 | 584.15 | 2024-05-02 14:15:00 | 582.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-04-29 14:45:00 | 585.20 | 2024-05-02 14:15:00 | 582.40 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-05-02 14:00:00 | 586.80 | 2024-05-02 14:15:00 | 582.40 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-05-09 10:00:00 | 575.05 | 2024-05-17 09:15:00 | 570.35 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2024-05-09 11:30:00 | 575.85 | 2024-05-17 09:15:00 | 570.35 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2024-05-09 12:30:00 | 575.60 | 2024-05-17 09:15:00 | 570.35 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2024-05-09 15:15:00 | 575.60 | 2024-05-17 09:15:00 | 570.35 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2024-05-13 09:15:00 | 575.00 | 2024-05-17 09:15:00 | 570.35 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2024-05-14 09:45:00 | 573.80 | 2024-05-17 09:15:00 | 570.35 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-05-30 12:00:00 | 587.75 | 2024-05-30 15:15:00 | 585.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-05-30 13:00:00 | 589.30 | 2024-05-30 15:15:00 | 585.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-06-03 10:30:00 | 574.65 | 2024-06-05 14:15:00 | 578.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-06-03 12:45:00 | 575.30 | 2024-06-05 14:15:00 | 578.70 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-06-03 13:30:00 | 575.05 | 2024-06-05 14:15:00 | 578.70 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-06-03 14:00:00 | 574.90 | 2024-06-05 14:15:00 | 578.70 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-06-27 10:30:00 | 587.10 | 2024-07-01 09:15:00 | 595.30 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-06-27 12:45:00 | 588.15 | 2024-07-01 09:15:00 | 595.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-07-03 10:45:00 | 599.60 | 2024-07-03 15:15:00 | 588.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-07-08 15:15:00 | 586.40 | 2024-07-09 09:15:00 | 595.85 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-07-10 13:30:00 | 595.35 | 2024-07-16 09:15:00 | 598.30 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-07-10 14:15:00 | 599.30 | 2024-07-16 09:15:00 | 598.30 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-07-10 15:15:00 | 597.00 | 2024-07-16 09:15:00 | 598.30 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2024-07-26 09:15:00 | 618.90 | 2024-07-31 14:15:00 | 680.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 13:30:00 | 617.05 | 2024-07-31 14:15:00 | 678.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 14:45:00 | 617.10 | 2024-07-31 14:15:00 | 678.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-29 11:30:00 | 618.90 | 2024-07-31 14:15:00 | 680.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-30 09:15:00 | 617.20 | 2024-07-31 14:15:00 | 678.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-30 14:15:00 | 616.05 | 2024-07-31 14:15:00 | 677.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-30 14:45:00 | 616.10 | 2024-07-31 14:15:00 | 677.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-31 09:15:00 | 622.40 | 2024-07-31 14:15:00 | 684.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-31 10:15:00 | 643.00 | 2024-08-05 10:15:00 | 629.25 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-08-05 09:30:00 | 639.30 | 2024-08-05 10:15:00 | 629.25 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-08-06 13:30:00 | 641.20 | 2024-08-07 13:15:00 | 649.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-08-06 14:00:00 | 640.40 | 2024-08-07 13:15:00 | 649.40 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-08-12 10:30:00 | 664.65 | 2024-08-27 11:15:00 | 707.05 | STOP_HIT | 1.00 | 6.38% |
| BUY | retest2 | 2024-08-13 10:45:00 | 665.80 | 2024-08-27 11:15:00 | 707.05 | STOP_HIT | 1.00 | 6.20% |
| BUY | retest2 | 2024-08-13 13:45:00 | 663.10 | 2024-08-27 11:15:00 | 707.05 | STOP_HIT | 1.00 | 6.63% |
| BUY | retest2 | 2024-08-14 10:30:00 | 665.40 | 2024-08-27 11:15:00 | 707.05 | STOP_HIT | 1.00 | 6.26% |
| BUY | retest2 | 2024-08-23 15:00:00 | 714.50 | 2024-08-27 11:15:00 | 707.05 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-08-26 13:30:00 | 712.85 | 2024-08-27 11:15:00 | 707.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-08-27 09:45:00 | 712.15 | 2024-08-27 11:15:00 | 707.05 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-09-04 09:45:00 | 739.20 | 2024-09-04 13:15:00 | 813.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 663.00 | 2024-10-22 09:15:00 | 629.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 663.00 | 2024-10-22 14:15:00 | 631.60 | STOP_HIT | 0.50 | 4.74% |
| BUY | retest2 | 2024-10-30 09:45:00 | 642.00 | 2024-11-06 09:15:00 | 706.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-30 10:15:00 | 641.90 | 2024-11-06 09:15:00 | 706.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-13 14:45:00 | 695.80 | 2024-11-13 15:15:00 | 701.95 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-11-29 11:30:00 | 783.00 | 2024-12-06 10:15:00 | 786.90 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-11-29 12:15:00 | 784.70 | 2024-12-06 10:15:00 | 786.90 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2024-11-29 13:00:00 | 785.50 | 2024-12-06 10:15:00 | 786.90 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2024-12-10 12:30:00 | 785.10 | 2024-12-10 14:15:00 | 811.85 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-12-10 13:00:00 | 784.15 | 2024-12-10 14:15:00 | 811.85 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2024-12-17 10:15:00 | 764.80 | 2024-12-20 09:15:00 | 769.15 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-12-18 14:15:00 | 764.40 | 2024-12-20 09:15:00 | 769.15 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-12-18 15:15:00 | 763.95 | 2024-12-20 09:15:00 | 769.15 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-12-19 10:00:00 | 765.20 | 2024-12-20 09:15:00 | 769.15 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-12-19 14:45:00 | 763.75 | 2024-12-20 09:15:00 | 769.15 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-12-27 10:15:00 | 731.50 | 2024-12-30 09:15:00 | 743.45 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-12-27 13:15:00 | 730.30 | 2024-12-30 09:15:00 | 743.45 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-12-30 09:15:00 | 731.50 | 2024-12-30 09:15:00 | 743.45 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-01-01 15:15:00 | 727.00 | 2025-01-08 10:15:00 | 693.12 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2025-01-03 09:15:00 | 729.60 | 2025-01-08 10:15:00 | 692.79 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-01-03 09:45:00 | 729.25 | 2025-01-08 11:15:00 | 690.65 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2025-01-03 12:00:00 | 728.75 | 2025-01-08 11:15:00 | 692.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 11:15:00 | 713.45 | 2025-01-09 11:15:00 | 677.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-01 15:15:00 | 727.00 | 2025-01-10 09:15:00 | 654.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 09:15:00 | 729.60 | 2025-01-10 09:15:00 | 656.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 09:45:00 | 729.25 | 2025-01-10 09:15:00 | 656.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 12:00:00 | 728.75 | 2025-01-10 09:15:00 | 655.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 11:15:00 | 713.45 | 2025-01-10 14:15:00 | 642.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 586.80 | 2025-02-19 09:15:00 | 602.05 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-02-14 10:45:00 | 584.85 | 2025-02-19 09:15:00 | 602.05 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-02-14 15:00:00 | 587.20 | 2025-02-19 09:15:00 | 602.05 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-02-17 09:15:00 | 584.15 | 2025-02-19 09:15:00 | 602.05 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-02-25 13:45:00 | 591.40 | 2025-03-04 14:15:00 | 585.25 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-02-25 14:30:00 | 589.20 | 2025-03-04 14:15:00 | 585.25 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-02-27 09:30:00 | 588.60 | 2025-03-04 14:15:00 | 585.25 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-03-06 09:15:00 | 589.40 | 2025-03-06 11:15:00 | 580.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-03-06 10:15:00 | 586.50 | 2025-03-06 11:15:00 | 580.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-03-07 10:15:00 | 588.00 | 2025-03-10 11:15:00 | 580.05 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-03-17 09:15:00 | 566.00 | 2025-03-20 15:15:00 | 564.80 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-03-20 15:00:00 | 565.65 | 2025-03-20 15:15:00 | 564.80 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-03-28 15:15:00 | 553.30 | 2025-04-02 10:15:00 | 573.20 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-04-09 11:45:00 | 564.05 | 2025-04-11 11:15:00 | 620.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-09 12:45:00 | 564.45 | 2025-04-11 11:15:00 | 620.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-29 10:45:00 | 614.75 | 2025-05-02 09:15:00 | 584.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 14:15:00 | 614.00 | 2025-05-02 09:15:00 | 583.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 10:45:00 | 614.75 | 2025-05-02 11:15:00 | 595.00 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2025-04-29 14:15:00 | 614.00 | 2025-05-02 11:15:00 | 595.00 | STOP_HIT | 0.50 | 3.09% |
| BUY | retest2 | 2025-05-09 11:15:00 | 733.50 | 2025-05-15 14:15:00 | 724.20 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-05-12 09:15:00 | 741.65 | 2025-05-15 14:15:00 | 724.20 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-05-13 10:00:00 | 730.90 | 2025-05-15 14:15:00 | 724.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-05-13 10:30:00 | 734.70 | 2025-05-20 11:15:00 | 803.99 | TARGET_HIT | 1.00 | 9.43% |
| BUY | retest2 | 2025-05-14 15:15:00 | 736.85 | 2025-05-20 13:15:00 | 806.85 | TARGET_HIT | 1.00 | 9.50% |
| BUY | retest2 | 2025-05-15 10:00:00 | 729.95 | 2025-05-20 13:15:00 | 815.82 | TARGET_HIT | 1.00 | 11.76% |
| BUY | retest2 | 2025-05-15 13:45:00 | 730.15 | 2025-05-20 13:15:00 | 808.17 | TARGET_HIT | 1.00 | 10.69% |
| BUY | retest2 | 2025-05-16 09:15:00 | 753.50 | 2025-05-20 15:15:00 | 828.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-20 09:15:00 | 792.15 | 2025-05-20 15:15:00 | 843.10 | TARGET_HIT | 1.00 | 6.43% |
| BUY | retest2 | 2025-05-20 10:00:00 | 766.45 | 2025-05-27 12:15:00 | 871.37 | TARGET_HIT | 1.00 | 13.69% |
| SELL | retest2 | 2025-06-16 13:30:00 | 808.00 | 2025-06-23 11:15:00 | 804.95 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-06-17 10:30:00 | 812.00 | 2025-06-23 11:15:00 | 804.95 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-06-23 10:30:00 | 798.00 | 2025-06-23 11:15:00 | 804.95 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-27 09:30:00 | 849.00 | 2025-07-01 10:15:00 | 835.55 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-07-02 09:15:00 | 820.30 | 2025-07-02 14:15:00 | 854.10 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2025-07-10 09:15:00 | 898.90 | 2025-07-10 15:15:00 | 880.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-07-10 10:45:00 | 898.05 | 2025-07-10 15:15:00 | 880.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-24 13:30:00 | 837.55 | 2025-07-29 14:15:00 | 860.00 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-07-25 09:30:00 | 838.00 | 2025-07-29 14:15:00 | 860.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-07-28 09:30:00 | 835.75 | 2025-07-29 14:15:00 | 860.00 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-28 10:30:00 | 835.90 | 2025-07-29 14:15:00 | 860.00 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-07-29 12:00:00 | 837.90 | 2025-07-29 14:15:00 | 860.00 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-29 12:30:00 | 836.60 | 2025-07-29 14:15:00 | 860.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-08-01 12:30:00 | 872.00 | 2025-08-06 10:15:00 | 859.25 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-08-14 09:30:00 | 852.85 | 2025-08-18 09:15:00 | 890.80 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2025-08-14 11:00:00 | 853.55 | 2025-08-18 09:15:00 | 890.80 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2025-08-28 09:30:00 | 853.55 | 2025-09-01 09:15:00 | 889.05 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2025-08-28 15:15:00 | 852.20 | 2025-09-01 09:15:00 | 889.05 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2025-09-11 12:00:00 | 908.00 | 2025-09-11 13:15:00 | 904.55 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-09-11 13:00:00 | 908.20 | 2025-09-11 13:15:00 | 904.55 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-09-24 09:15:00 | 902.40 | 2025-09-25 15:15:00 | 897.15 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-10-06 09:15:00 | 852.40 | 2025-10-15 10:15:00 | 841.45 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2025-10-06 09:45:00 | 853.25 | 2025-10-15 10:15:00 | 841.45 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2025-10-20 09:15:00 | 840.40 | 2025-10-24 14:15:00 | 835.10 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-29 10:15:00 | 861.35 | 2025-11-06 09:15:00 | 947.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-29 12:30:00 | 860.85 | 2025-11-06 09:15:00 | 946.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 09:45:00 | 862.80 | 2025-11-06 09:15:00 | 949.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-31 09:15:00 | 861.15 | 2025-11-06 09:15:00 | 947.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-03 09:30:00 | 870.05 | 2025-11-06 09:15:00 | 957.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-03 10:15:00 | 871.15 | 2025-11-06 09:15:00 | 958.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-03 11:30:00 | 869.40 | 2025-11-06 09:15:00 | 956.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-08 11:30:00 | 940.00 | 2025-12-09 13:15:00 | 971.10 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-12-08 15:15:00 | 939.30 | 2025-12-09 13:15:00 | 971.10 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-12-30 09:45:00 | 956.10 | 2026-01-01 14:15:00 | 908.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 10:45:00 | 954.90 | 2026-01-01 14:15:00 | 907.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 09:45:00 | 956.10 | 2026-01-05 11:15:00 | 930.05 | STOP_HIT | 0.50 | 2.72% |
| SELL | retest2 | 2025-12-30 10:45:00 | 954.90 | 2026-01-05 11:15:00 | 930.05 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest1 | 2026-01-22 10:30:00 | 938.30 | 2026-01-27 11:15:00 | 946.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-01-27 13:15:00 | 933.30 | 2026-01-27 14:15:00 | 960.05 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-01-30 09:15:00 | 930.05 | 2026-01-30 09:15:00 | 955.70 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2026-02-02 09:30:00 | 978.05 | 2026-02-06 10:15:00 | 975.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-02-03 11:45:00 | 978.45 | 2026-02-06 10:15:00 | 975.45 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-03 13:45:00 | 978.05 | 2026-02-06 10:15:00 | 975.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-02-03 14:30:00 | 978.25 | 2026-02-06 10:15:00 | 975.45 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-02-13 14:30:00 | 999.50 | 2026-02-16 09:15:00 | 1037.40 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2026-02-19 15:00:00 | 991.75 | 2026-02-23 09:15:00 | 1038.95 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2026-02-20 10:00:00 | 990.50 | 2026-02-23 09:15:00 | 1038.95 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-02-20 11:00:00 | 990.15 | 2026-02-23 09:15:00 | 1038.95 | STOP_HIT | 1.00 | -4.93% |
| SELL | retest2 | 2026-02-20 12:45:00 | 992.50 | 2026-02-23 09:15:00 | 1038.95 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1002.80 | 2026-03-04 09:15:00 | 1021.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-03-04 11:30:00 | 1013.00 | 2026-03-05 09:15:00 | 1021.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-04 13:15:00 | 1012.90 | 2026-03-05 09:15:00 | 1021.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-03-12 11:15:00 | 1048.00 | 2026-03-13 12:15:00 | 1019.90 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2026-04-21 09:15:00 | 1117.40 | 2026-04-23 14:15:00 | 1090.90 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-04-29 09:30:00 | 1121.80 | 2026-05-08 11:15:00 | 1136.60 | STOP_HIT | 1.00 | 1.32% |
