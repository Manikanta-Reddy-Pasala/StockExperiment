# Aegis Logistics Ltd. (AEGISLOG)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 725.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 143 |
| ALERT1 | 104 |
| ALERT2 | 103 |
| ALERT2_SKIP | 56 |
| ALERT3 | 273 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 120 |
| PARTIAL | 11 |
| TARGET_HIT | 14 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 98
- **Target hits / Stop hits / Partials:** 14 / 107 / 11
- **Avg / median % per leg:** 0.31% / -0.96%
- **Sum % (uncompounded):** 40.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 12 | 25.5% | 9 | 38 | 0 | 0.96% | 45.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 47 | 12 | 25.5% | 9 | 38 | 0 | 0.96% | 45.1% |
| SELL (all) | 85 | 22 | 25.9% | 5 | 69 | 11 | -0.05% | -4.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.11% | -0.1% |
| SELL @ 3rd Alert (retest2) | 84 | 22 | 26.2% | 5 | 68 | 11 | -0.05% | -4.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.11% | -0.1% |
| retest2 (combined) | 131 | 34 | 26.0% | 14 | 106 | 11 | 0.31% | 40.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 599.50 | 597.06 | 596.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 602.70 | 599.39 | 598.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 599.95 | 600.04 | 599.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 13:15:00 | 599.95 | 600.04 | 599.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 599.95 | 600.04 | 599.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 600.15 | 600.04 | 599.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 597.35 | 599.50 | 598.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 597.55 | 599.50 | 598.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 597.00 | 599.00 | 598.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 600.00 | 599.00 | 598.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 598.55 | 599.50 | 599.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:00:00 | 598.55 | 599.50 | 599.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 599.50 | 599.50 | 599.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:00:00 | 599.50 | 599.50 | 599.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 605.90 | 600.78 | 599.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-18 09:15:00 | 623.00 | 600.86 | 599.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-24 14:15:00 | 685.30 | 656.93 | 643.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 702.70 | 720.11 | 721.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 684.70 | 709.98 | 716.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 703.85 | 702.07 | 710.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 12:15:00 | 703.85 | 702.07 | 710.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 703.85 | 702.07 | 710.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:45:00 | 705.05 | 702.07 | 710.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 706.30 | 703.50 | 709.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 713.30 | 703.50 | 709.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 717.00 | 706.20 | 710.11 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 14:15:00 | 721.85 | 712.70 | 711.99 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 672.90 | 706.06 | 709.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 638.55 | 692.56 | 702.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 12:15:00 | 701.15 | 694.28 | 702.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 13:00:00 | 701.15 | 694.28 | 702.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 699.70 | 695.36 | 702.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:45:00 | 699.65 | 695.36 | 702.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 699.50 | 696.19 | 702.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 699.50 | 696.19 | 702.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 693.90 | 695.73 | 701.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 696.45 | 695.73 | 701.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 713.00 | 699.18 | 702.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 713.00 | 699.18 | 702.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 723.05 | 703.96 | 704.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 723.05 | 703.96 | 704.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 706.60 | 704.49 | 704.55 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 713.30 | 706.25 | 705.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 743.80 | 713.43 | 708.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 777.00 | 799.14 | 772.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 14:15:00 | 777.00 | 799.14 | 772.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 14:15:00 | 777.00 | 799.14 | 772.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 15:00:00 | 777.00 | 799.14 | 772.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 778.60 | 795.03 | 773.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:30:00 | 766.00 | 787.39 | 771.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 755.25 | 780.97 | 770.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:00:00 | 755.25 | 780.97 | 770.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 15:15:00 | 757.00 | 764.31 | 764.63 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 10:15:00 | 771.30 | 765.85 | 765.28 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 13:15:00 | 763.00 | 767.32 | 767.55 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 772.55 | 768.47 | 768.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 11:15:00 | 815.05 | 778.31 | 772.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 13:15:00 | 817.95 | 822.40 | 807.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 14:00:00 | 817.95 | 822.40 | 807.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 816.05 | 826.39 | 819.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 816.60 | 826.39 | 819.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 813.35 | 823.78 | 818.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 814.95 | 823.78 | 818.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 798.35 | 818.56 | 817.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:45:00 | 800.80 | 818.56 | 817.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 794.25 | 813.70 | 815.33 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 833.95 | 818.80 | 817.00 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 805.75 | 816.97 | 817.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 775.00 | 806.58 | 812.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 807.70 | 806.80 | 812.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 10:15:00 | 807.70 | 806.80 | 812.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 807.70 | 806.80 | 812.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 807.70 | 806.80 | 812.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 808.75 | 806.28 | 810.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:45:00 | 809.70 | 806.28 | 810.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 808.65 | 806.36 | 809.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 837.85 | 806.36 | 809.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 835.35 | 812.16 | 812.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 852.20 | 832.38 | 824.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 13:15:00 | 832.45 | 835.22 | 828.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 13:15:00 | 832.45 | 835.22 | 828.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 832.45 | 835.22 | 828.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:30:00 | 851.05 | 839.01 | 831.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 848.00 | 860.20 | 860.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 848.00 | 860.20 | 860.86 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 863.60 | 860.62 | 860.43 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 13:15:00 | 858.00 | 860.09 | 860.21 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 863.05 | 860.68 | 860.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 15:15:00 | 870.00 | 862.55 | 861.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 15:15:00 | 930.40 | 933.73 | 921.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 09:15:00 | 921.95 | 933.73 | 921.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 915.40 | 930.06 | 921.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 915.40 | 930.06 | 921.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 913.30 | 926.71 | 920.39 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 14:15:00 | 909.95 | 916.42 | 916.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 09:15:00 | 887.50 | 909.43 | 913.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 12:15:00 | 853.00 | 851.22 | 867.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 13:00:00 | 853.00 | 851.22 | 867.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 874.55 | 855.89 | 868.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 13:45:00 | 873.90 | 855.89 | 868.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 858.85 | 856.48 | 867.17 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 889.25 | 874.05 | 872.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 14:15:00 | 899.80 | 884.24 | 878.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 885.90 | 887.84 | 881.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 885.90 | 887.84 | 881.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 890.00 | 888.64 | 883.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 890.30 | 888.64 | 883.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 880.00 | 886.93 | 883.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 869.00 | 886.93 | 883.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 848.50 | 879.25 | 880.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 837.75 | 870.95 | 876.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 846.85 | 846.08 | 855.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 15:00:00 | 846.85 | 846.08 | 855.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 861.75 | 841.73 | 846.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 861.75 | 841.73 | 846.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 864.15 | 846.21 | 848.31 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 867.10 | 850.39 | 850.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 874.00 | 855.11 | 852.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 874.80 | 880.01 | 870.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 15:15:00 | 874.80 | 880.01 | 870.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 874.80 | 880.01 | 870.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:00:00 | 897.40 | 883.49 | 873.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 861.90 | 880.37 | 882.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 09:15:00 | 861.90 | 880.37 | 882.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 829.25 | 853.70 | 867.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 743.70 | 740.57 | 759.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 743.70 | 740.57 | 759.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 743.70 | 740.57 | 759.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 734.15 | 740.57 | 759.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 10:15:00 | 751.50 | 739.58 | 739.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 751.50 | 739.58 | 739.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 11:15:00 | 754.60 | 750.78 | 749.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 13:15:00 | 751.45 | 751.69 | 749.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-14 14:00:00 | 751.45 | 751.69 | 749.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 751.85 | 751.72 | 750.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:30:00 | 751.85 | 751.72 | 750.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 751.85 | 751.75 | 750.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 757.45 | 751.62 | 750.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 15:15:00 | 761.00 | 752.03 | 751.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-21 09:15:00 | 833.20 | 799.34 | 783.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 784.05 | 797.60 | 798.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 780.10 | 794.10 | 796.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 772.00 | 765.73 | 774.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 772.00 | 765.73 | 774.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 772.00 | 765.73 | 774.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 15:00:00 | 758.00 | 764.31 | 770.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:45:00 | 756.55 | 753.66 | 757.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:30:00 | 756.95 | 754.63 | 758.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 752.55 | 754.90 | 757.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 09:15:00 | 805.90 | 764.28 | 761.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 805.90 | 764.28 | 761.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 827.00 | 799.74 | 783.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 12:15:00 | 826.10 | 846.02 | 840.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 12:15:00 | 826.10 | 846.02 | 840.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 826.10 | 846.02 | 840.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 826.10 | 846.02 | 840.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 822.80 | 841.38 | 838.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:00:00 | 822.80 | 841.38 | 838.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 818.10 | 836.72 | 836.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 10:15:00 | 813.25 | 827.56 | 832.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 829.00 | 823.91 | 828.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 15:15:00 | 829.00 | 823.91 | 828.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 829.00 | 823.91 | 828.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 843.90 | 823.91 | 828.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 832.45 | 825.62 | 828.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 836.95 | 825.62 | 828.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 838.65 | 830.66 | 830.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 848.00 | 836.67 | 833.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 10:15:00 | 836.00 | 839.37 | 835.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 836.00 | 839.37 | 835.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 836.00 | 839.37 | 835.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:00:00 | 836.00 | 839.37 | 835.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 840.70 | 839.64 | 836.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:30:00 | 835.65 | 839.64 | 836.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 835.20 | 843.40 | 839.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 835.20 | 843.40 | 839.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 836.30 | 841.98 | 839.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:30:00 | 839.95 | 841.98 | 839.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 836.00 | 840.78 | 839.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:15:00 | 834.75 | 840.78 | 839.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 829.65 | 838.55 | 838.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 829.65 | 838.55 | 838.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 13:15:00 | 826.35 | 836.11 | 837.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 09:15:00 | 824.35 | 834.39 | 836.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 13:15:00 | 809.00 | 806.29 | 815.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-16 13:30:00 | 808.80 | 806.29 | 815.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 767.75 | 768.65 | 777.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 759.75 | 768.65 | 777.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 12:45:00 | 760.90 | 763.33 | 763.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 15:00:00 | 755.10 | 761.51 | 762.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 721.76 | 737.57 | 743.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 722.85 | 737.57 | 743.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 717.35 | 737.57 | 743.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-30 10:15:00 | 750.20 | 740.10 | 744.41 | SL hit (close>ema200) qty=0.50 sl=740.10 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 12:15:00 | 759.65 | 747.08 | 747.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 13:15:00 | 761.90 | 750.04 | 748.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 15:15:00 | 758.40 | 762.55 | 758.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 15:15:00 | 758.40 | 762.55 | 758.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 758.40 | 762.55 | 758.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 742.55 | 762.55 | 758.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 743.00 | 758.64 | 756.70 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 739.10 | 754.73 | 755.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 735.70 | 748.51 | 752.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 688.90 | 687.52 | 705.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 688.90 | 687.52 | 705.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 691.30 | 686.91 | 697.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:30:00 | 684.10 | 690.37 | 694.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 12:00:00 | 684.35 | 690.37 | 694.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 12:30:00 | 684.75 | 689.26 | 693.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 13:00:00 | 684.80 | 689.26 | 693.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 710.10 | 684.97 | 686.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 710.10 | 684.97 | 686.40 | SL hit (close>static) qty=1.00 sl=697.75 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 709.70 | 689.92 | 688.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 11:15:00 | 718.90 | 695.72 | 691.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 711.80 | 712.18 | 702.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 09:45:00 | 711.05 | 712.18 | 702.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 706.70 | 710.28 | 703.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 708.20 | 710.28 | 703.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 712.90 | 723.44 | 721.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:30:00 | 714.35 | 723.44 | 721.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 726.85 | 724.12 | 721.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 11:15:00 | 740.35 | 724.12 | 721.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 13:00:00 | 734.80 | 738.61 | 733.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 15:15:00 | 737.00 | 736.01 | 733.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-01 17:15:00 | 814.39 | 798.27 | 786.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 800.30 | 803.96 | 804.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 792.70 | 799.74 | 801.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 772.40 | 768.93 | 776.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 12:15:00 | 773.90 | 771.87 | 776.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 773.90 | 771.87 | 776.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 768.75 | 773.50 | 776.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 10:00:00 | 771.95 | 773.19 | 775.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 10:45:00 | 769.80 | 771.33 | 774.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:30:00 | 771.55 | 769.11 | 772.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 767.00 | 768.69 | 771.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:15:00 | 780.40 | 768.69 | 771.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 809.40 | 776.83 | 775.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 809.40 | 776.83 | 775.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 09:15:00 | 847.65 | 806.27 | 792.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 13:15:00 | 842.40 | 844.66 | 828.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 14:00:00 | 842.40 | 844.66 | 828.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 794.20 | 832.27 | 826.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 794.20 | 832.27 | 826.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 786.70 | 823.16 | 823.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:00:00 | 786.70 | 823.16 | 823.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 795.00 | 817.53 | 820.48 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 827.05 | 813.16 | 811.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 12:15:00 | 841.00 | 818.73 | 814.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 828.95 | 831.72 | 823.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:45:00 | 825.70 | 831.72 | 823.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 821.90 | 829.19 | 823.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:00:00 | 821.90 | 829.19 | 823.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 12:15:00 | 818.55 | 827.06 | 822.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:45:00 | 819.00 | 827.06 | 822.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 821.10 | 825.87 | 822.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:30:00 | 821.70 | 825.87 | 822.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 823.95 | 825.48 | 822.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:30:00 | 820.50 | 825.48 | 822.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 821.20 | 824.63 | 822.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 823.45 | 824.63 | 822.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 817.55 | 823.21 | 822.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 817.55 | 823.21 | 822.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 814.60 | 821.49 | 821.55 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 836.05 | 821.84 | 821.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 11:15:00 | 845.65 | 829.61 | 824.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 846.20 | 879.91 | 864.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 846.20 | 879.91 | 864.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 846.20 | 879.91 | 864.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:15:00 | 838.40 | 879.91 | 864.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 838.45 | 871.62 | 861.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:45:00 | 836.00 | 871.62 | 861.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 842.00 | 856.38 | 856.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 15:15:00 | 832.70 | 849.08 | 853.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 840.00 | 839.25 | 844.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 840.00 | 839.25 | 844.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 840.00 | 839.25 | 844.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 840.00 | 839.25 | 844.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 847.00 | 840.80 | 844.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:45:00 | 847.60 | 840.80 | 844.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 846.00 | 841.84 | 844.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 843.85 | 841.84 | 844.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 856.20 | 844.71 | 845.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 854.95 | 844.71 | 845.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 855.50 | 846.87 | 846.66 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 841.10 | 846.68 | 846.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 15:15:00 | 837.35 | 841.88 | 843.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 12:15:00 | 787.00 | 786.07 | 799.50 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 09:15:00 | 782.65 | 786.03 | 796.11 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 783.55 | 766.75 | 772.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-13 11:15:00 | 783.55 | 766.75 | 772.64 | SL hit (close>ema400) qty=1.00 sl=772.64 alert=retest1 |

### Cycle 41 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 789.15 | 776.34 | 775.00 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 12:15:00 | 774.80 | 781.26 | 781.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 13:15:00 | 759.75 | 776.96 | 779.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 807.00 | 777.52 | 778.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 807.00 | 777.52 | 778.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 807.00 | 777.52 | 778.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 807.00 | 777.52 | 778.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 813.50 | 784.71 | 781.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 15:15:00 | 840.00 | 808.51 | 795.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 800.85 | 806.98 | 796.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:00:00 | 800.85 | 806.98 | 796.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 797.85 | 805.15 | 796.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 13:45:00 | 805.15 | 803.25 | 797.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 10:15:00 | 811.00 | 801.46 | 797.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 15:15:00 | 810.00 | 816.88 | 817.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 15:15:00 | 810.00 | 816.88 | 817.69 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 09:15:00 | 846.55 | 822.81 | 820.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 10:15:00 | 915.85 | 841.42 | 829.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 14:15:00 | 839.45 | 860.04 | 843.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 14:15:00 | 839.45 | 860.04 | 843.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 839.45 | 860.04 | 843.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 839.45 | 860.04 | 843.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 850.00 | 858.03 | 844.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 895.00 | 858.03 | 844.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-07 15:15:00 | 984.50 | 921.34 | 885.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 14:15:00 | 887.75 | 915.53 | 916.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 867.45 | 902.78 | 910.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 10:15:00 | 859.40 | 853.42 | 874.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-13 10:45:00 | 856.00 | 853.42 | 874.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 800.80 | 813.27 | 831.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 10:00:00 | 793.65 | 806.31 | 815.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 793.70 | 801.63 | 810.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 14:00:00 | 794.25 | 800.15 | 809.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 15:00:00 | 793.75 | 798.87 | 807.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 753.97 | 769.45 | 784.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 754.01 | 769.45 | 784.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 754.54 | 769.45 | 784.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 754.06 | 769.45 | 784.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-23 09:15:00 | 714.28 | 734.44 | 755.71 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 679.10 | 662.88 | 662.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 688.75 | 670.66 | 665.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 697.35 | 699.25 | 691.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 697.35 | 699.25 | 691.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 697.35 | 699.25 | 691.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 695.60 | 699.25 | 691.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 693.75 | 698.15 | 691.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 693.95 | 698.15 | 691.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 683.50 | 695.22 | 691.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:30:00 | 681.40 | 695.22 | 691.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 685.40 | 693.26 | 690.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 671.95 | 693.26 | 690.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 662.55 | 685.60 | 687.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 659.05 | 676.52 | 682.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 13:15:00 | 674.45 | 674.17 | 680.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 14:00:00 | 674.45 | 674.17 | 680.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 707.15 | 681.96 | 682.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 707.15 | 681.96 | 682.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 710.10 | 687.59 | 685.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 11:15:00 | 715.05 | 693.08 | 687.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 791.55 | 793.38 | 766.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:45:00 | 792.60 | 793.38 | 766.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 811.40 | 814.08 | 799.68 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 10:15:00 | 755.10 | 789.04 | 793.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 751.10 | 776.95 | 786.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 767.95 | 762.96 | 775.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 767.95 | 762.96 | 775.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 769.60 | 764.29 | 774.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 776.85 | 764.29 | 774.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 766.25 | 764.68 | 773.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 769.15 | 764.68 | 773.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 758.85 | 763.51 | 772.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 766.90 | 763.51 | 772.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 796.00 | 770.01 | 774.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 796.00 | 770.01 | 774.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 796.70 | 775.35 | 776.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 776.75 | 775.35 | 776.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 15:15:00 | 737.91 | 753.82 | 764.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 09:15:00 | 699.08 | 741.86 | 758.10 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 739.80 | 694.05 | 693.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 763.75 | 713.95 | 703.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 14:15:00 | 726.00 | 730.55 | 720.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 15:00:00 | 726.00 | 730.55 | 720.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 756.50 | 746.72 | 736.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 13:30:00 | 767.55 | 756.10 | 751.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 09:15:00 | 727.50 | 751.92 | 752.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 727.50 | 751.92 | 752.63 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 761.00 | 752.89 | 752.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 774.50 | 757.22 | 754.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 09:15:00 | 768.55 | 773.89 | 768.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 09:15:00 | 768.55 | 773.89 | 768.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 768.55 | 773.89 | 768.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 10:00:00 | 768.55 | 773.89 | 768.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 763.60 | 771.83 | 768.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 10:45:00 | 761.30 | 771.83 | 768.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 770.85 | 771.63 | 768.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 13:00:00 | 776.85 | 772.68 | 769.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 09:15:00 | 758.30 | 770.49 | 769.50 | SL hit (close<static) qty=1.00 sl=758.80 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 10:15:00 | 760.25 | 768.45 | 768.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 751.00 | 764.96 | 767.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 13:15:00 | 751.35 | 745.96 | 753.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:00:00 | 751.35 | 745.96 | 753.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 750.00 | 746.77 | 753.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 750.05 | 746.77 | 753.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 745.95 | 746.61 | 752.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 09:15:00 | 735.15 | 746.61 | 752.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 15:15:00 | 736.00 | 734.89 | 734.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 15:15:00 | 736.00 | 734.89 | 734.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 12:15:00 | 742.55 | 736.47 | 735.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 09:15:00 | 741.25 | 742.02 | 738.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 741.25 | 742.02 | 738.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 741.25 | 742.02 | 738.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 741.25 | 742.02 | 738.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 752.85 | 746.22 | 742.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 10:15:00 | 757.65 | 746.22 | 742.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 782.65 | 788.97 | 789.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 782.65 | 788.97 | 789.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 775.60 | 782.38 | 784.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 782.40 | 782.38 | 784.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 11:00:00 | 782.40 | 782.38 | 784.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 780.40 | 781.98 | 784.31 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 798.55 | 788.12 | 786.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 925.00 | 815.50 | 799.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 11:15:00 | 831.40 | 833.24 | 815.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 12:15:00 | 825.90 | 833.24 | 815.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 804.50 | 826.38 | 816.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 804.50 | 826.38 | 816.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 798.95 | 820.89 | 814.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 793.00 | 820.89 | 814.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 788.00 | 811.48 | 811.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 788.00 | 811.48 | 811.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 783.55 | 805.90 | 808.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 771.40 | 788.41 | 794.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 13:15:00 | 772.25 | 770.51 | 779.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 13:45:00 | 774.40 | 770.51 | 779.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 788.70 | 774.15 | 780.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 788.70 | 774.15 | 780.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 791.00 | 777.52 | 781.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 798.45 | 777.52 | 781.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 792.35 | 780.48 | 782.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:00:00 | 784.05 | 781.20 | 782.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:30:00 | 778.75 | 780.93 | 782.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 786.00 | 770.34 | 769.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 786.00 | 770.34 | 769.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 11:15:00 | 806.00 | 777.47 | 772.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 788.95 | 792.81 | 783.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 14:15:00 | 787.05 | 791.09 | 786.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 787.05 | 791.09 | 786.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 787.05 | 791.09 | 786.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 785.90 | 790.05 | 786.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 790.10 | 790.05 | 786.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 795.80 | 791.20 | 787.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:45:00 | 801.30 | 793.69 | 788.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 800.75 | 797.46 | 792.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 788.75 | 795.02 | 795.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 788.75 | 795.02 | 795.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 14:15:00 | 779.80 | 789.28 | 792.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 11:15:00 | 785.75 | 785.33 | 788.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 11:15:00 | 785.75 | 785.33 | 788.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 785.75 | 785.33 | 788.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:30:00 | 786.60 | 785.33 | 788.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 755.50 | 775.26 | 782.50 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 780.25 | 777.47 | 777.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 795.50 | 781.58 | 779.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 789.00 | 790.13 | 787.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 15:00:00 | 789.00 | 790.13 | 787.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 776.80 | 787.46 | 786.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 767.05 | 787.46 | 786.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 775.00 | 784.97 | 785.32 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 785.20 | 783.27 | 783.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 792.00 | 785.29 | 784.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 784.25 | 785.83 | 784.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 784.25 | 785.83 | 784.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 784.25 | 785.83 | 784.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 808.15 | 788.83 | 787.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 773.25 | 798.59 | 797.86 | SL hit (close<static) qty=1.00 sl=782.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 778.70 | 794.61 | 796.12 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 818.50 | 799.26 | 797.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 10:15:00 | 832.00 | 820.60 | 816.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 09:15:00 | 909.65 | 918.90 | 896.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 909.65 | 918.90 | 896.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 905.25 | 916.17 | 897.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 901.00 | 916.17 | 897.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 897.00 | 912.34 | 897.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 897.00 | 912.34 | 897.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 899.50 | 909.77 | 897.50 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 881.85 | 891.00 | 892.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 872.80 | 887.36 | 890.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 829.20 | 823.35 | 840.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 829.20 | 823.35 | 840.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 800.30 | 793.00 | 801.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 806.20 | 793.00 | 801.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 795.85 | 793.57 | 800.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 792.40 | 794.81 | 798.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 790.35 | 796.06 | 798.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 13:45:00 | 789.60 | 794.79 | 797.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 792.40 | 792.31 | 795.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 788.65 | 788.30 | 792.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:45:00 | 790.00 | 788.30 | 792.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 790.75 | 788.97 | 791.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 790.60 | 788.97 | 791.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 797.95 | 790.76 | 792.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 797.95 | 790.76 | 792.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 794.15 | 791.44 | 792.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 794.60 | 791.44 | 792.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-04 13:15:00 | 798.80 | 793.54 | 793.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 798.80 | 793.54 | 793.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 806.90 | 800.78 | 797.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 801.40 | 801.88 | 799.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:30:00 | 800.00 | 801.88 | 799.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 800.50 | 801.06 | 799.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 10:30:00 | 804.30 | 800.85 | 799.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:45:00 | 809.25 | 801.80 | 800.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:45:00 | 803.70 | 801.90 | 800.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:30:00 | 802.55 | 802.87 | 801.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 800.85 | 802.46 | 801.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 801.40 | 802.46 | 801.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 801.00 | 802.17 | 801.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 819.40 | 801.18 | 801.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 803.60 | 807.26 | 804.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 803.20 | 805.47 | 804.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:45:00 | 804.80 | 805.17 | 804.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 804.55 | 805.05 | 804.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 803.40 | 805.05 | 804.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 803.15 | 804.67 | 804.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 803.15 | 804.67 | 804.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 801.60 | 804.05 | 804.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 800.20 | 804.05 | 804.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 800.15 | 803.27 | 803.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 800.15 | 803.27 | 803.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 795.00 | 800.97 | 802.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 782.95 | 781.62 | 787.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 09:30:00 | 783.00 | 781.62 | 787.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 786.00 | 782.49 | 787.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 782.10 | 783.31 | 786.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:30:00 | 781.10 | 782.58 | 785.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 790.80 | 780.77 | 782.12 | SL hit (close>static) qty=1.00 sl=790.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 11:15:00 | 792.75 | 783.17 | 783.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 15:15:00 | 806.00 | 791.91 | 787.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 782.55 | 795.03 | 791.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 782.55 | 795.03 | 791.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 782.55 | 795.03 | 791.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 782.55 | 795.03 | 791.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 782.65 | 792.55 | 790.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 778.05 | 792.55 | 790.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 786.45 | 790.45 | 790.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 786.05 | 790.45 | 790.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 787.20 | 789.80 | 790.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 14:15:00 | 785.45 | 788.36 | 789.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 14:15:00 | 787.85 | 786.72 | 787.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 14:15:00 | 787.85 | 786.72 | 787.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 787.85 | 786.72 | 787.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 787.85 | 786.72 | 787.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 789.95 | 787.36 | 787.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 796.10 | 787.36 | 787.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 794.25 | 788.74 | 788.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 804.85 | 796.17 | 792.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 818.90 | 824.03 | 813.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 12:15:00 | 818.90 | 824.03 | 813.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 818.90 | 824.03 | 813.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 818.90 | 824.03 | 813.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 802.75 | 819.77 | 812.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:00:00 | 802.75 | 819.77 | 812.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 783.75 | 812.57 | 810.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 783.75 | 812.57 | 810.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 784.00 | 806.85 | 807.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 738.55 | 772.91 | 786.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 751.65 | 749.12 | 759.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 751.65 | 749.12 | 759.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 753.55 | 751.57 | 756.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 753.90 | 751.57 | 756.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 753.45 | 751.81 | 754.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:45:00 | 752.50 | 751.81 | 754.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 754.00 | 752.25 | 754.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 758.50 | 752.25 | 754.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 760.40 | 753.88 | 754.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 753.20 | 754.16 | 754.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 759.00 | 755.48 | 755.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 759.00 | 755.48 | 755.32 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 751.40 | 754.96 | 755.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 13:15:00 | 747.00 | 751.57 | 753.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 14:15:00 | 747.30 | 746.74 | 749.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 14:15:00 | 747.30 | 746.74 | 749.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 747.30 | 746.74 | 749.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 747.30 | 746.74 | 749.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 747.15 | 747.18 | 749.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:45:00 | 742.00 | 746.20 | 748.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:30:00 | 742.10 | 745.18 | 747.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 749.25 | 741.60 | 741.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 749.25 | 741.60 | 741.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 751.95 | 746.08 | 743.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 746.30 | 748.45 | 746.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 746.30 | 748.45 | 746.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 746.30 | 748.45 | 746.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 746.30 | 748.45 | 746.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 746.95 | 748.15 | 746.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 745.15 | 748.15 | 746.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 745.50 | 747.62 | 746.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 751.30 | 747.62 | 746.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 741.35 | 748.25 | 748.09 | SL hit (close<static) qty=1.00 sl=744.10 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 739.55 | 746.51 | 747.31 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 755.00 | 746.93 | 746.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 15:15:00 | 758.00 | 750.65 | 748.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 753.00 | 754.45 | 751.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 14:15:00 | 753.00 | 754.45 | 751.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 753.00 | 754.45 | 751.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 752.95 | 754.45 | 751.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 751.00 | 753.76 | 751.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 751.30 | 753.76 | 751.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 754.55 | 753.92 | 751.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 757.10 | 753.92 | 751.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:00:00 | 756.80 | 754.50 | 752.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:15:00 | 756.60 | 754.73 | 752.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 749.30 | 753.50 | 752.84 | SL hit (close<static) qty=1.00 sl=749.45 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 747.90 | 751.72 | 752.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 743.15 | 749.21 | 750.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 737.70 | 735.89 | 740.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 737.70 | 735.89 | 740.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 727.85 | 728.34 | 732.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:00:00 | 721.95 | 726.99 | 731.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 722.75 | 725.25 | 729.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:30:00 | 724.05 | 724.84 | 728.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:30:00 | 724.80 | 724.82 | 728.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 714.25 | 713.07 | 717.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 711.00 | 713.07 | 717.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 737.00 | 718.10 | 719.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 737.00 | 718.10 | 719.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 727.60 | 720.00 | 720.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 736.80 | 720.00 | 720.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 734.45 | 722.89 | 721.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 734.45 | 722.89 | 721.42 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 718.25 | 722.78 | 723.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 711.60 | 717.58 | 720.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 723.05 | 712.21 | 715.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 723.05 | 712.21 | 715.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 723.05 | 712.21 | 715.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 723.05 | 712.21 | 715.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 722.00 | 714.17 | 715.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 726.65 | 716.66 | 716.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 724.95 | 718.32 | 717.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 11:15:00 | 727.85 | 720.23 | 718.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 716.20 | 719.97 | 718.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 13:15:00 | 716.20 | 719.97 | 718.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 716.20 | 719.97 | 718.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 716.20 | 719.97 | 718.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 712.00 | 718.37 | 718.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:15:00 | 713.50 | 718.37 | 718.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 713.50 | 717.40 | 717.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 708.10 | 715.54 | 716.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 709.90 | 709.88 | 712.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 709.90 | 709.88 | 712.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 709.90 | 709.88 | 712.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:15:00 | 712.20 | 709.88 | 712.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 710.70 | 710.04 | 712.46 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 720.70 | 714.28 | 713.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 725.00 | 718.14 | 715.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 718.10 | 722.32 | 719.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 718.10 | 722.32 | 719.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 718.10 | 722.32 | 719.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 718.10 | 722.32 | 719.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 706.35 | 719.13 | 718.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 707.00 | 719.13 | 718.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 706.70 | 716.64 | 717.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 702.60 | 713.84 | 716.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 700.75 | 699.16 | 703.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 12:00:00 | 700.75 | 699.16 | 703.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 705.45 | 699.62 | 701.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:45:00 | 705.45 | 699.62 | 701.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 709.85 | 701.67 | 702.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:15:00 | 713.65 | 701.67 | 702.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 732.75 | 707.88 | 705.41 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 726.25 | 728.16 | 728.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 717.50 | 726.03 | 727.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 687.40 | 681.16 | 691.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 687.40 | 681.16 | 691.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 687.40 | 681.16 | 691.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 691.30 | 681.16 | 691.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 686.95 | 683.79 | 691.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 690.15 | 683.79 | 691.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 690.30 | 686.01 | 690.91 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 697.10 | 693.60 | 693.20 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 689.80 | 692.75 | 692.88 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 697.05 | 693.10 | 692.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 12:15:00 | 700.65 | 695.35 | 694.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 699.90 | 700.03 | 697.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:30:00 | 699.50 | 700.03 | 697.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 698.00 | 699.62 | 697.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:30:00 | 698.60 | 699.62 | 697.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 700.00 | 699.70 | 698.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 702.80 | 699.70 | 698.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 701.90 | 705.01 | 705.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 701.90 | 705.01 | 705.19 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 719.00 | 706.48 | 705.16 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 703.05 | 706.94 | 706.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 702.00 | 705.95 | 706.49 | Break + close below crossover candle low |

### Cycle 93 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 741.65 | 711.97 | 708.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 11:15:00 | 765.95 | 727.29 | 716.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 794.25 | 794.49 | 777.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 13:30:00 | 795.30 | 794.49 | 777.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 784.30 | 791.09 | 784.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 784.30 | 791.09 | 784.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 784.00 | 789.67 | 784.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 792.25 | 789.67 | 784.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 785.15 | 788.70 | 785.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 785.15 | 788.70 | 785.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 784.70 | 787.90 | 785.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:15:00 | 785.40 | 787.90 | 785.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 784.15 | 787.15 | 785.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 783.50 | 787.15 | 785.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 791.40 | 788.00 | 785.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 796.00 | 788.00 | 785.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 802.45 | 791.84 | 788.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 777.35 | 786.99 | 786.86 | SL hit (close<static) qty=1.00 sl=783.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 774.25 | 784.44 | 785.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 766.40 | 775.11 | 779.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 11:15:00 | 774.95 | 769.44 | 773.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 774.95 | 769.44 | 773.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 774.95 | 769.44 | 773.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:45:00 | 775.60 | 769.44 | 773.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 778.75 | 771.30 | 774.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:30:00 | 780.00 | 771.30 | 774.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 770.15 | 772.21 | 773.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 767.60 | 772.21 | 773.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 764.95 | 770.75 | 773.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:45:00 | 761.00 | 768.85 | 772.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 756.75 | 758.55 | 763.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 761.15 | 750.06 | 756.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 758.45 | 754.20 | 757.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 757.80 | 754.92 | 757.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 762.35 | 754.92 | 757.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 760.65 | 756.07 | 757.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 760.65 | 756.07 | 757.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 766.95 | 758.24 | 758.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 764.65 | 758.24 | 758.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 765.00 | 759.59 | 759.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 765.00 | 759.59 | 759.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 779.00 | 763.48 | 760.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 850.95 | 852.98 | 822.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 14:15:00 | 829.15 | 841.23 | 827.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 829.15 | 841.23 | 827.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:45:00 | 830.90 | 841.23 | 827.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 828.30 | 838.65 | 827.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 833.00 | 838.65 | 827.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 828.10 | 836.54 | 827.99 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 822.00 | 824.42 | 824.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 815.00 | 821.61 | 823.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 812.65 | 810.83 | 815.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 812.65 | 810.83 | 815.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 812.65 | 810.83 | 815.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 812.80 | 810.83 | 815.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 815.60 | 811.78 | 815.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 816.00 | 811.78 | 815.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 806.90 | 810.81 | 814.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:30:00 | 811.00 | 810.81 | 814.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 814.15 | 810.46 | 813.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 813.05 | 810.46 | 813.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 816.55 | 811.68 | 813.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 816.55 | 811.68 | 813.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 816.85 | 812.71 | 813.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 816.85 | 812.71 | 813.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 815.15 | 813.20 | 813.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 815.15 | 813.20 | 813.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 813.55 | 813.45 | 813.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:45:00 | 813.00 | 813.45 | 813.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 812.00 | 813.16 | 813.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 811.30 | 813.16 | 813.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 814.45 | 813.42 | 813.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 815.75 | 813.42 | 813.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 815.50 | 813.84 | 814.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 817.00 | 813.84 | 814.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 815.00 | 814.07 | 814.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 816.65 | 814.07 | 814.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 807.90 | 812.84 | 813.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:15:00 | 805.50 | 812.84 | 813.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 823.35 | 814.04 | 813.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 823.35 | 814.04 | 813.68 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 808.85 | 813.05 | 813.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 805.70 | 811.58 | 812.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 14:15:00 | 814.50 | 805.32 | 807.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 814.50 | 805.32 | 807.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 814.50 | 805.32 | 807.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 814.50 | 805.32 | 807.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 814.95 | 807.25 | 808.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 809.50 | 808.49 | 808.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 812.85 | 809.36 | 809.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 812.85 | 809.36 | 809.24 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 807.50 | 808.99 | 809.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 13:15:00 | 802.45 | 807.32 | 808.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 10:15:00 | 807.00 | 806.26 | 807.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 807.00 | 806.26 | 807.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 807.00 | 806.26 | 807.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 807.00 | 806.26 | 807.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 806.55 | 805.55 | 806.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 806.55 | 805.55 | 806.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 802.30 | 804.90 | 806.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 806.00 | 804.29 | 805.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 807.85 | 803.67 | 804.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 809.15 | 803.67 | 804.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 800.70 | 803.07 | 804.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:30:00 | 804.25 | 803.07 | 804.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 804.95 | 803.45 | 804.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 804.05 | 803.66 | 804.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 804.50 | 803.83 | 804.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 804.50 | 803.83 | 804.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 802.60 | 803.58 | 804.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 804.00 | 803.58 | 804.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 802.15 | 803.30 | 804.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:45:00 | 798.50 | 801.59 | 803.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 798.25 | 800.92 | 802.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:00:00 | 798.00 | 800.03 | 801.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 806.20 | 799.41 | 798.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 806.20 | 799.41 | 798.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 809.50 | 804.07 | 801.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 805.00 | 805.62 | 803.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 805.00 | 805.62 | 803.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 801.85 | 804.72 | 803.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 801.85 | 804.72 | 803.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 803.40 | 804.45 | 803.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 794.25 | 804.45 | 803.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 791.00 | 801.76 | 801.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 13:15:00 | 786.65 | 794.54 | 798.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 769.15 | 768.70 | 776.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 769.15 | 768.70 | 776.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 774.35 | 763.82 | 771.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:15:00 | 778.05 | 763.82 | 771.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 771.25 | 765.31 | 771.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:30:00 | 768.80 | 767.56 | 771.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 768.90 | 767.56 | 771.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 768.10 | 767.87 | 771.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 767.85 | 767.73 | 770.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 769.85 | 768.15 | 770.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 771.60 | 768.15 | 770.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 755.60 | 763.01 | 766.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 752.80 | 763.01 | 766.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:00:00 | 755.50 | 761.51 | 765.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 769.55 | 764.87 | 765.13 | SL hit (close>static) qty=1.00 sl=769.05 alert=retest2 |

### Cycle 103 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 772.00 | 766.29 | 765.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 780.70 | 772.46 | 769.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 796.80 | 798.23 | 788.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 796.80 | 798.23 | 788.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 796.80 | 798.23 | 788.70 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 788.45 | 793.58 | 793.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 786.15 | 788.86 | 790.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 770.80 | 769.08 | 775.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:15:00 | 770.00 | 769.08 | 775.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 769.60 | 766.63 | 770.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 759.40 | 766.55 | 768.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 15:15:00 | 761.75 | 766.55 | 768.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:00:00 | 761.50 | 761.05 | 764.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:00:00 | 761.55 | 758.76 | 760.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 770.90 | 763.64 | 762.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 770.90 | 763.64 | 762.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 774.50 | 771.72 | 768.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 770.60 | 771.85 | 769.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 770.60 | 771.85 | 769.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 768.75 | 771.23 | 769.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:45:00 | 768.75 | 771.23 | 769.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 772.00 | 771.39 | 769.64 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 766.35 | 768.45 | 768.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 763.00 | 767.36 | 768.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 741.45 | 737.02 | 744.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 741.45 | 737.02 | 744.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 748.35 | 739.90 | 744.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 748.35 | 739.90 | 744.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 752.60 | 742.44 | 745.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 753.65 | 742.44 | 745.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 753.80 | 746.66 | 746.60 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 743.40 | 746.33 | 746.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 742.45 | 745.56 | 746.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 753.25 | 741.14 | 741.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 753.25 | 741.14 | 741.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 753.25 | 741.14 | 741.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 753.25 | 741.14 | 741.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 754.10 | 743.73 | 742.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 759.55 | 746.90 | 744.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 748.95 | 749.33 | 746.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 748.95 | 749.33 | 746.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 748.95 | 749.33 | 746.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 744.20 | 749.33 | 746.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 750.15 | 749.53 | 747.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:45:00 | 748.40 | 749.53 | 747.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 742.70 | 748.20 | 747.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 742.70 | 748.20 | 747.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 743.00 | 747.16 | 747.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 742.85 | 747.16 | 747.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 741.45 | 746.02 | 746.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 737.70 | 743.63 | 745.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 734.65 | 732.11 | 735.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 734.65 | 732.11 | 735.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 734.65 | 732.11 | 735.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 735.20 | 732.11 | 735.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 733.85 | 732.46 | 735.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 736.35 | 732.46 | 735.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 733.85 | 732.74 | 734.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 729.15 | 732.74 | 734.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:15:00 | 731.40 | 732.62 | 734.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 735.80 | 733.26 | 734.58 | SL hit (close>static) qty=1.00 sl=735.50 alert=retest2 |

### Cycle 111 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 737.80 | 735.02 | 735.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 739.85 | 736.43 | 735.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 736.00 | 736.52 | 735.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 736.00 | 736.52 | 735.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 735.55 | 736.33 | 735.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 735.30 | 736.33 | 735.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 733.50 | 735.76 | 735.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 733.05 | 735.76 | 735.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 732.55 | 735.12 | 735.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 730.55 | 734.00 | 734.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 711.50 | 708.71 | 713.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 712.55 | 709.48 | 713.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 712.55 | 709.48 | 713.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 712.55 | 709.48 | 713.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 717.95 | 711.55 | 713.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 717.95 | 711.55 | 713.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 716.40 | 712.52 | 714.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 714.25 | 714.31 | 714.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 718.30 | 714.56 | 714.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 718.30 | 714.56 | 714.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 719.50 | 716.13 | 715.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 734.60 | 736.40 | 728.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 734.60 | 736.40 | 728.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 725.50 | 732.85 | 728.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:45:00 | 725.15 | 732.85 | 728.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 732.00 | 732.68 | 728.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 725.40 | 732.68 | 728.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 745.60 | 735.26 | 730.40 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 738.00 | 742.38 | 742.42 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 743.50 | 742.26 | 742.19 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 14:15:00 | 740.80 | 742.18 | 742.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 09:15:00 | 725.20 | 738.43 | 740.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 665.00 | 663.51 | 675.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 663.30 | 663.51 | 675.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 658.40 | 653.37 | 657.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 660.15 | 653.37 | 657.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 661.70 | 655.04 | 658.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:45:00 | 663.30 | 655.04 | 658.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 659.80 | 655.99 | 658.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:15:00 | 666.00 | 655.99 | 658.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 662.95 | 657.38 | 658.87 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 670.55 | 661.94 | 660.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 695.65 | 668.68 | 663.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 15:15:00 | 695.05 | 697.15 | 689.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 09:15:00 | 699.20 | 697.15 | 689.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 702.35 | 698.19 | 690.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 705.75 | 699.42 | 691.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 705.20 | 699.42 | 691.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:45:00 | 706.35 | 700.73 | 693.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 706.80 | 711.52 | 705.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 704.85 | 710.19 | 705.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 704.85 | 710.19 | 705.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 698.95 | 703.40 | 703.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 698.95 | 703.40 | 703.56 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 708.85 | 704.14 | 703.77 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 693.80 | 702.59 | 703.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 10:15:00 | 688.15 | 699.70 | 702.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 704.20 | 699.92 | 701.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 12:15:00 | 704.20 | 699.92 | 701.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 704.20 | 699.92 | 701.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 704.20 | 699.92 | 701.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 703.95 | 700.73 | 702.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:15:00 | 698.90 | 700.39 | 701.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 709.35 | 694.72 | 694.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 709.35 | 694.72 | 694.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 720.15 | 699.81 | 696.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 708.30 | 715.03 | 710.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 708.30 | 715.03 | 710.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 708.30 | 715.03 | 710.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 708.30 | 715.03 | 710.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 707.10 | 713.44 | 710.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 707.10 | 713.44 | 710.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 709.45 | 712.64 | 710.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:45:00 | 714.00 | 712.45 | 710.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:00:00 | 713.40 | 714.17 | 711.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:00:00 | 712.75 | 713.88 | 712.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 13:15:00 | 704.75 | 711.14 | 711.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 704.75 | 711.14 | 711.18 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 712.85 | 711.48 | 711.33 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 700.00 | 709.11 | 710.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 09:15:00 | 694.50 | 701.30 | 703.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 697.05 | 696.15 | 699.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 697.05 | 696.15 | 699.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 693.00 | 695.81 | 699.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 691.30 | 695.81 | 699.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 11:00:00 | 689.25 | 694.50 | 698.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:30:00 | 691.80 | 692.87 | 695.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:45:00 | 691.60 | 686.57 | 690.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 695.00 | 688.25 | 690.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 695.00 | 688.25 | 690.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 692.40 | 689.08 | 690.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 13:15:00 | 688.80 | 689.08 | 690.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 700.25 | 692.41 | 691.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 700.25 | 692.41 | 691.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 707.00 | 699.40 | 696.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 13:15:00 | 698.60 | 705.88 | 701.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 13:15:00 | 698.60 | 705.88 | 701.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 698.60 | 705.88 | 701.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:00:00 | 698.60 | 705.88 | 701.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 699.70 | 704.65 | 701.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 716.00 | 703.72 | 700.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 14:15:00 | 699.35 | 706.92 | 707.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 699.35 | 706.92 | 707.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 691.45 | 702.32 | 705.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 659.50 | 658.31 | 670.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:45:00 | 660.00 | 658.31 | 670.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 676.80 | 664.16 | 667.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 679.80 | 664.16 | 667.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 677.00 | 666.73 | 668.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 677.00 | 666.73 | 668.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 674.90 | 669.95 | 669.66 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 663.60 | 668.91 | 669.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 656.00 | 665.70 | 667.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 655.00 | 653.48 | 658.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 655.00 | 653.48 | 658.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 655.00 | 653.48 | 658.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:00:00 | 647.90 | 652.69 | 657.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:00:00 | 647.65 | 651.68 | 656.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:15:00 | 615.50 | 634.75 | 644.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 14:15:00 | 615.27 | 631.30 | 642.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 629.10 | 623.08 | 634.25 | SL hit (close>ema200) qty=0.50 sl=623.08 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 646.70 | 617.79 | 616.56 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 13:15:00 | 614.65 | 617.02 | 617.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 14:15:00 | 609.25 | 615.46 | 616.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 11:15:00 | 612.35 | 611.21 | 613.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 11:15:00 | 612.35 | 611.21 | 613.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 612.35 | 611.21 | 613.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 612.35 | 611.21 | 613.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 629.20 | 614.81 | 615.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 629.20 | 614.81 | 615.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 13:15:00 | 625.70 | 616.98 | 616.18 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 607.75 | 615.14 | 615.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 598.60 | 608.35 | 610.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 598.45 | 594.30 | 600.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 598.45 | 594.30 | 600.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 598.45 | 594.30 | 600.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:45:00 | 590.50 | 593.94 | 599.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 591.80 | 593.94 | 599.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 615.50 | 600.66 | 600.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 615.50 | 600.66 | 600.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 618.85 | 604.30 | 602.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 608.00 | 609.78 | 606.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 609.15 | 609.78 | 606.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 608.85 | 609.59 | 606.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:00:00 | 613.90 | 610.72 | 607.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:45:00 | 613.40 | 611.23 | 608.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 613.90 | 610.95 | 608.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 601.50 | 609.43 | 608.96 | SL hit (close<static) qty=1.00 sl=603.60 alert=retest2 |

### Cycle 134 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 604.50 | 608.45 | 608.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 596.20 | 606.00 | 607.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 612.45 | 605.53 | 606.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 612.45 | 605.53 | 606.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 612.45 | 605.53 | 606.86 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 613.10 | 607.69 | 607.58 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 604.35 | 607.51 | 607.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 590.45 | 603.15 | 605.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 10:15:00 | 614.85 | 597.18 | 599.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 10:15:00 | 614.85 | 597.18 | 599.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 614.85 | 597.18 | 599.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 614.85 | 597.18 | 599.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 609.90 | 599.72 | 600.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 12:45:00 | 603.20 | 599.74 | 600.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 614.25 | 598.74 | 597.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 614.25 | 598.74 | 597.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 639.30 | 613.63 | 607.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 613.70 | 624.21 | 617.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 613.70 | 624.21 | 617.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 613.70 | 624.21 | 617.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 616.70 | 624.21 | 617.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 617.20 | 622.81 | 617.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 619.20 | 621.02 | 617.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 619.90 | 620.90 | 617.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 663.95 | 619.97 | 618.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 681.12 | 669.05 | 655.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 707.15 | 714.31 | 714.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 703.85 | 709.93 | 712.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 718.40 | 710.83 | 712.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 718.40 | 710.83 | 712.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 718.40 | 710.83 | 712.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:00:00 | 718.40 | 710.83 | 712.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 712.80 | 711.23 | 712.41 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 12:15:00 | 718.00 | 713.49 | 713.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 720.15 | 715.63 | 714.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 14:15:00 | 722.70 | 722.89 | 719.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 15:00:00 | 722.70 | 722.89 | 719.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 724.10 | 722.95 | 719.80 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 711.75 | 717.98 | 718.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 696.30 | 708.50 | 712.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 703.90 | 702.76 | 707.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 703.70 | 702.76 | 707.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 714.15 | 705.04 | 707.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 714.15 | 705.04 | 707.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 716.00 | 707.23 | 708.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 713.95 | 707.23 | 708.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 715.40 | 710.24 | 709.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 719.00 | 713.44 | 711.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 708.05 | 712.37 | 711.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 708.05 | 712.37 | 711.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 708.05 | 712.37 | 711.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 708.05 | 712.37 | 711.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 709.20 | 711.73 | 710.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:45:00 | 711.85 | 710.89 | 710.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 708.80 | 710.48 | 710.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 708.80 | 710.48 | 710.48 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 723.30 | 712.68 | 711.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 726.85 | 716.96 | 713.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 732.15 | 736.29 | 730.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 732.15 | 736.29 | 730.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 732.15 | 736.29 | 730.08 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-18 09:15:00 | 623.00 | 2024-05-24 14:15:00 | 685.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-27 09:30:00 | 851.05 | 2024-07-02 12:15:00 | 848.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-07-26 10:00:00 | 897.40 | 2024-07-31 09:15:00 | 861.90 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2024-08-06 10:15:00 | 734.15 | 2024-08-09 10:15:00 | 751.50 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-08-16 09:30:00 | 757.45 | 2024-08-21 09:15:00 | 833.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-16 15:15:00 | 761.00 | 2024-08-26 09:15:00 | 784.05 | STOP_HIT | 1.00 | 3.03% |
| SELL | retest2 | 2024-08-28 15:00:00 | 758.00 | 2024-09-02 09:15:00 | 805.90 | STOP_HIT | 1.00 | -6.32% |
| SELL | retest2 | 2024-08-30 11:45:00 | 756.55 | 2024-09-02 09:15:00 | 805.90 | STOP_HIT | 1.00 | -6.52% |
| SELL | retest2 | 2024-08-30 12:30:00 | 756.95 | 2024-09-02 09:15:00 | 805.90 | STOP_HIT | 1.00 | -6.47% |
| SELL | retest2 | 2024-08-30 15:00:00 | 752.55 | 2024-09-02 09:15:00 | 805.90 | STOP_HIT | 1.00 | -7.09% |
| SELL | retest2 | 2024-09-20 09:15:00 | 759.75 | 2024-09-30 09:15:00 | 721.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-24 12:45:00 | 760.90 | 2024-09-30 09:15:00 | 722.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-24 15:00:00 | 755.10 | 2024-09-30 09:15:00 | 717.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 09:15:00 | 759.75 | 2024-09-30 10:15:00 | 750.20 | STOP_HIT | 0.50 | 1.26% |
| SELL | retest2 | 2024-09-24 12:45:00 | 760.90 | 2024-09-30 10:15:00 | 750.20 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2024-09-24 15:00:00 | 755.10 | 2024-09-30 10:15:00 | 750.20 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2024-10-11 11:30:00 | 684.10 | 2024-10-15 09:15:00 | 710.10 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2024-10-11 12:00:00 | 684.35 | 2024-10-15 09:15:00 | 710.10 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2024-10-11 12:30:00 | 684.75 | 2024-10-15 09:15:00 | 710.10 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2024-10-11 13:00:00 | 684.80 | 2024-10-15 09:15:00 | 710.10 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2024-10-21 11:15:00 | 740.35 | 2024-11-01 17:15:00 | 814.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-22 13:00:00 | 734.80 | 2024-11-01 17:15:00 | 808.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-22 15:15:00 | 737.00 | 2024-11-01 17:15:00 | 810.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-13 09:15:00 | 768.75 | 2024-11-14 09:15:00 | 809.40 | STOP_HIT | 1.00 | -5.29% |
| SELL | retest2 | 2024-11-13 10:00:00 | 771.95 | 2024-11-14 09:15:00 | 809.40 | STOP_HIT | 1.00 | -4.85% |
| SELL | retest2 | 2024-11-13 10:45:00 | 769.80 | 2024-11-14 09:15:00 | 809.40 | STOP_HIT | 1.00 | -5.14% |
| SELL | retest2 | 2024-11-13 14:30:00 | 771.55 | 2024-11-14 09:15:00 | 809.40 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest1 | 2024-12-11 09:15:00 | 782.65 | 2024-12-13 11:15:00 | 783.55 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-12-23 13:45:00 | 805.15 | 2025-01-03 15:15:00 | 810.00 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-12-24 10:15:00 | 811.00 | 2025-01-03 15:15:00 | 810.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-01-07 09:15:00 | 895.00 | 2025-01-07 15:15:00 | 984.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-20 10:00:00 | 793.65 | 2025-01-22 09:15:00 | 753.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 12:30:00 | 793.70 | 2025-01-22 09:15:00 | 754.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 14:00:00 | 794.25 | 2025-01-22 09:15:00 | 754.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 15:00:00 | 793.75 | 2025-01-22 09:15:00 | 754.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 10:00:00 | 793.65 | 2025-01-23 09:15:00 | 714.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 12:30:00 | 793.70 | 2025-01-23 09:15:00 | 714.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 14:00:00 | 794.25 | 2025-01-23 09:15:00 | 714.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 15:00:00 | 793.75 | 2025-01-23 09:15:00 | 714.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-29 12:30:00 | 665.00 | 2025-01-29 14:15:00 | 679.10 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-02-13 09:15:00 | 776.75 | 2025-02-13 15:15:00 | 737.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 776.75 | 2025-02-14 09:15:00 | 699.08 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-27 13:30:00 | 767.55 | 2025-03-03 09:15:00 | 727.50 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest2 | 2025-03-06 13:00:00 | 776.85 | 2025-03-07 09:15:00 | 758.30 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-03-11 09:15:00 | 735.15 | 2025-03-13 15:15:00 | 736.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-03-19 10:15:00 | 757.65 | 2025-03-25 14:15:00 | 782.65 | STOP_HIT | 1.00 | 3.30% |
| SELL | retest2 | 2025-04-08 11:00:00 | 784.05 | 2025-04-16 10:15:00 | 786.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-04-08 11:30:00 | 778.75 | 2025-04-16 10:15:00 | 786.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-04-21 10:45:00 | 801.30 | 2025-04-23 10:15:00 | 788.75 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-04-22 09:15:00 | 800.75 | 2025-04-23 10:15:00 | 788.75 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-05-08 09:15:00 | 808.15 | 2025-05-09 09:15:00 | 773.25 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2025-06-02 09:15:00 | 792.40 | 2025-06-04 13:15:00 | 798.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-02 12:45:00 | 790.35 | 2025-06-04 13:15:00 | 798.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-06-02 13:45:00 | 789.60 | 2025-06-04 13:15:00 | 798.80 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-06-03 09:30:00 | 792.40 | 2025-06-04 13:15:00 | 798.80 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-09 10:30:00 | 804.30 | 2025-06-12 13:15:00 | 800.15 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-06-09 12:45:00 | 809.25 | 2025-06-12 13:15:00 | 800.15 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-09 14:45:00 | 803.70 | 2025-06-12 13:15:00 | 800.15 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-10 09:30:00 | 802.55 | 2025-06-12 13:15:00 | 800.15 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-06-11 09:15:00 | 819.40 | 2025-06-12 13:15:00 | 800.15 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-06-11 14:00:00 | 803.60 | 2025-06-12 13:15:00 | 800.15 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-12 09:15:00 | 803.20 | 2025-06-12 13:15:00 | 800.15 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-06-12 09:45:00 | 804.80 | 2025-06-12 13:15:00 | 800.15 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-06-17 15:15:00 | 782.10 | 2025-06-19 10:15:00 | 790.80 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-06-18 09:30:00 | 781.10 | 2025-06-19 10:15:00 | 790.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-07-07 11:45:00 | 753.20 | 2025-07-07 14:15:00 | 759.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-07-11 10:45:00 | 742.00 | 2025-07-15 11:15:00 | 749.25 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-11 11:30:00 | 742.10 | 2025-07-15 11:15:00 | 749.25 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-07-17 09:15:00 | 751.30 | 2025-07-18 10:15:00 | 741.35 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-23 10:15:00 | 757.10 | 2025-07-24 09:15:00 | 749.30 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-07-23 11:00:00 | 756.80 | 2025-07-24 09:15:00 | 749.30 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-23 12:15:00 | 756.60 | 2025-07-24 09:15:00 | 749.30 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-30 12:00:00 | 721.95 | 2025-08-04 13:15:00 | 734.45 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-07-30 15:00:00 | 722.75 | 2025-08-04 13:15:00 | 734.45 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-07-31 09:30:00 | 724.05 | 2025-08-04 13:15:00 | 734.45 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-31 10:30:00 | 724.80 | 2025-08-04 13:15:00 | 734.45 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-05 09:15:00 | 702.80 | 2025-09-09 14:15:00 | 701.90 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-09-19 15:15:00 | 796.00 | 2025-09-22 13:15:00 | 777.35 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-09-22 09:30:00 | 802.45 | 2025-09-22 13:15:00 | 777.35 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-09-26 10:45:00 | 761.00 | 2025-09-30 15:15:00 | 765.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-09-29 11:30:00 | 756.75 | 2025-09-30 15:15:00 | 765.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-30 09:45:00 | 761.15 | 2025-09-30 15:15:00 | 765.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-09-30 11:30:00 | 758.45 | 2025-09-30 15:15:00 | 765.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-10-13 13:15:00 | 805.50 | 2025-10-14 09:15:00 | 823.35 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-10-16 09:45:00 | 809.50 | 2025-10-16 10:15:00 | 812.85 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-23 13:45:00 | 798.50 | 2025-10-27 12:15:00 | 806.20 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-23 15:00:00 | 798.25 | 2025-10-27 12:15:00 | 806.20 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-10-24 10:00:00 | 798.00 | 2025-10-27 12:15:00 | 806.20 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-03 12:30:00 | 768.80 | 2025-11-10 10:15:00 | 769.55 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-11-03 13:00:00 | 768.90 | 2025-11-10 10:15:00 | 769.55 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-11-03 14:45:00 | 768.10 | 2025-11-10 11:15:00 | 772.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-04 10:30:00 | 767.85 | 2025-11-10 11:15:00 | 772.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-06 10:15:00 | 752.80 | 2025-11-10 11:15:00 | 772.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-11-06 11:00:00 | 755.50 | 2025-11-10 11:15:00 | 772.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-11-26 14:45:00 | 759.40 | 2025-12-01 10:15:00 | 770.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-11-26 15:15:00 | 761.75 | 2025-12-01 10:15:00 | 770.90 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-27 14:00:00 | 761.50 | 2025-12-01 10:15:00 | 770.90 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-11-28 15:00:00 | 761.55 | 2025-12-01 10:15:00 | 770.90 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-12-19 12:15:00 | 729.15 | 2025-12-19 14:15:00 | 735.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-19 14:15:00 | 731.40 | 2025-12-19 14:15:00 | 735.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-12-22 11:15:00 | 731.00 | 2025-12-22 13:15:00 | 737.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-01 10:00:00 | 714.25 | 2026-01-02 10:15:00 | 718.30 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-30 10:30:00 | 705.75 | 2026-02-02 10:15:00 | 698.95 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-01-30 11:15:00 | 705.20 | 2026-02-02 10:15:00 | 698.95 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-01-30 11:45:00 | 706.35 | 2026-02-02 10:15:00 | 698.95 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-01 12:30:00 | 706.80 | 2026-02-02 10:15:00 | 698.95 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-02-04 15:15:00 | 698.90 | 2026-02-09 10:15:00 | 709.35 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-11 13:45:00 | 714.00 | 2026-02-12 13:15:00 | 704.75 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-02-12 10:00:00 | 713.40 | 2026-02-12 13:15:00 | 704.75 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-02-12 11:00:00 | 712.75 | 2026-02-12 13:15:00 | 704.75 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-18 10:15:00 | 691.30 | 2026-02-23 09:15:00 | 700.25 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-02-18 11:00:00 | 689.25 | 2026-02-23 09:15:00 | 700.25 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-02-19 10:30:00 | 691.80 | 2026-02-23 09:15:00 | 700.25 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-02-20 10:45:00 | 691.60 | 2026-02-23 09:15:00 | 700.25 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-02-20 13:15:00 | 688.80 | 2026-02-23 09:15:00 | 700.25 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-02-25 09:15:00 | 716.00 | 2026-02-26 14:15:00 | 699.35 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-03-10 12:00:00 | 647.90 | 2026-03-11 13:15:00 | 615.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 13:00:00 | 647.65 | 2026-03-11 14:15:00 | 615.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 12:00:00 | 647.90 | 2026-03-12 11:15:00 | 629.10 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2026-03-10 13:00:00 | 647.65 | 2026-03-12 11:15:00 | 629.10 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-03-12 14:30:00 | 638.35 | 2026-03-13 13:15:00 | 606.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 638.35 | 2026-03-16 12:15:00 | 605.10 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2026-03-24 10:45:00 | 590.50 | 2026-03-25 09:15:00 | 615.50 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2026-03-24 11:15:00 | 591.80 | 2026-03-25 09:15:00 | 615.50 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2026-03-27 12:00:00 | 613.90 | 2026-03-30 12:15:00 | 601.50 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-03-27 12:45:00 | 613.40 | 2026-03-30 12:15:00 | 601.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-03-30 09:15:00 | 613.90 | 2026-03-30 12:15:00 | 601.50 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-04-06 12:45:00 | 603.20 | 2026-04-08 10:15:00 | 614.25 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-04-13 12:30:00 | 619.20 | 2026-04-17 09:15:00 | 681.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:45:00 | 619.90 | 2026-04-17 09:15:00 | 681.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 663.95 | 2026-04-17 09:15:00 | 730.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-05 12:45:00 | 711.85 | 2026-05-05 13:15:00 | 708.80 | STOP_HIT | 1.00 | -0.43% |
