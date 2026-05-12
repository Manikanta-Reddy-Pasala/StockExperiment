# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 954.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 207 |
| ALERT1 | 135 |
| ALERT2 | 132 |
| ALERT2_SKIP | 62 |
| ALERT3 | 361 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 193 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 191 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 201 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 72 / 129
- **Target hits / Stop hits / Partials:** 6 / 190 / 5
- **Avg / median % per leg:** -0.17% / -0.77%
- **Sum % (uncompounded):** -33.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 87 | 25 | 28.7% | 6 | 81 | 0 | -0.03% | -2.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.76% | -5.3% |
| BUY @ 3rd Alert (retest2) | 84 | 25 | 29.8% | 6 | 78 | 0 | 0.03% | 2.5% |
| SELL (all) | 114 | 47 | 41.2% | 0 | 109 | 5 | -0.27% | -31.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 114 | 47 | 41.2% | 0 | 109 | 5 | -0.27% | -31.2% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.76% | -5.3% |
| retest2 (combined) | 198 | 72 | 36.4% | 6 | 187 | 5 | -0.14% | -28.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 14:15:00 | 670.00 | 672.70 | 672.93 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 09:15:00 | 680.38 | 674.11 | 673.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 11:15:00 | 684.00 | 680.91 | 679.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 677.51 | 681.08 | 679.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 14:15:00 | 677.51 | 681.08 | 679.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 677.51 | 681.08 | 679.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 15:00:00 | 677.51 | 681.08 | 679.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 677.30 | 680.32 | 679.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:15:00 | 677.34 | 680.32 | 679.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 679.85 | 681.18 | 680.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 12:45:00 | 679.68 | 681.18 | 680.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 679.61 | 680.87 | 680.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 13:45:00 | 678.90 | 680.87 | 680.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 14:15:00 | 678.71 | 680.44 | 680.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 14:45:00 | 678.24 | 680.44 | 680.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 15:15:00 | 678.50 | 680.05 | 679.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:15:00 | 675.45 | 680.05 | 679.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 09:15:00 | 678.60 | 679.76 | 679.85 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 683.64 | 679.73 | 679.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 09:15:00 | 687.97 | 681.98 | 680.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 10:15:00 | 697.81 | 697.86 | 694.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 11:00:00 | 697.81 | 697.86 | 694.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 11:15:00 | 693.90 | 697.07 | 694.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:45:00 | 694.80 | 697.07 | 694.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 696.02 | 696.86 | 694.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:15:00 | 697.90 | 696.60 | 694.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 09:30:00 | 698.86 | 697.42 | 695.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 11:15:00 | 701.64 | 706.46 | 706.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 701.64 | 706.46 | 706.57 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 09:15:00 | 710.35 | 705.14 | 705.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 714.79 | 709.74 | 707.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 09:15:00 | 710.98 | 714.28 | 711.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 710.98 | 714.28 | 711.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 710.98 | 714.28 | 711.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:00:00 | 710.98 | 714.28 | 711.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 10:15:00 | 710.50 | 713.52 | 711.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:30:00 | 711.95 | 713.52 | 711.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 11:15:00 | 708.90 | 712.60 | 711.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 12:00:00 | 708.90 | 712.60 | 711.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 14:15:00 | 709.86 | 711.57 | 711.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 14:30:00 | 709.39 | 711.57 | 711.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 15:15:00 | 709.40 | 711.14 | 711.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 09:15:00 | 706.80 | 711.14 | 711.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 09:15:00 | 709.61 | 710.83 | 710.90 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 15:15:00 | 712.30 | 710.90 | 710.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 09:15:00 | 718.00 | 712.32 | 711.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 724.52 | 729.75 | 724.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-20 10:00:00 | 724.52 | 729.75 | 724.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 722.50 | 728.30 | 723.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 11:00:00 | 722.50 | 728.30 | 723.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 722.50 | 727.14 | 723.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 13:15:00 | 723.59 | 726.18 | 723.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 14:00:00 | 723.33 | 725.61 | 723.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 14:30:00 | 723.11 | 725.49 | 723.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 10:15:00 | 718.00 | 723.44 | 723.24 | SL hit (close<static) qty=1.00 sl=720.31 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 11:15:00 | 721.70 | 723.09 | 723.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 09:15:00 | 711.61 | 719.83 | 721.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 14:15:00 | 700.09 | 699.04 | 703.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 15:00:00 | 700.09 | 699.04 | 703.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 698.30 | 699.05 | 702.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 10:30:00 | 696.67 | 698.72 | 702.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 11:30:00 | 696.68 | 698.36 | 701.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 710.69 | 702.00 | 702.21 | SL hit (close>static) qty=1.00 sl=703.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 711.48 | 703.90 | 703.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 14:15:00 | 716.44 | 711.28 | 708.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 09:15:00 | 778.63 | 780.90 | 768.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 09:45:00 | 777.60 | 780.90 | 768.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 769.23 | 775.31 | 771.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:00:00 | 769.23 | 775.31 | 771.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 766.67 | 773.58 | 770.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:00:00 | 766.67 | 773.58 | 770.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 764.16 | 769.05 | 769.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 14:15:00 | 761.83 | 767.61 | 768.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 15:15:00 | 756.20 | 754.79 | 759.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-11 09:15:00 | 766.85 | 754.79 | 759.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 761.20 | 756.07 | 759.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 11:00:00 | 756.56 | 756.17 | 759.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 14:15:00 | 751.37 | 748.04 | 747.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 14:15:00 | 751.37 | 748.04 | 747.72 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 738.70 | 746.63 | 747.26 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 750.93 | 747.06 | 746.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 12:15:00 | 755.95 | 748.84 | 747.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 09:15:00 | 751.93 | 752.78 | 750.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-20 09:45:00 | 750.84 | 752.78 | 750.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 755.67 | 756.76 | 753.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 10:00:00 | 755.67 | 756.76 | 753.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 10:15:00 | 755.84 | 756.58 | 754.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 10:45:00 | 755.89 | 756.58 | 754.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 754.72 | 756.21 | 754.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 12:30:00 | 756.66 | 757.08 | 754.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 10:45:00 | 757.38 | 758.54 | 757.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 14:15:00 | 743.80 | 759.38 | 759.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 14:15:00 | 743.80 | 759.38 | 759.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 14:15:00 | 728.67 | 743.64 | 750.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 14:15:00 | 737.50 | 733.85 | 740.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-28 15:00:00 | 737.50 | 733.85 | 740.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 712.35 | 713.09 | 718.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 10:30:00 | 711.00 | 713.00 | 717.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:30:00 | 710.75 | 711.90 | 716.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 09:45:00 | 710.45 | 712.28 | 715.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 09:45:00 | 711.05 | 711.80 | 713.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 710.62 | 711.56 | 713.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:45:00 | 713.31 | 711.56 | 713.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 13:15:00 | 714.79 | 712.23 | 713.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:00:00 | 714.79 | 712.23 | 713.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 717.40 | 713.26 | 713.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 15:00:00 | 717.40 | 713.26 | 713.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-08 15:15:00 | 720.30 | 714.67 | 714.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 15:15:00 | 720.30 | 714.67 | 714.09 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 711.25 | 713.59 | 713.67 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 15:15:00 | 717.20 | 713.48 | 713.16 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 11:15:00 | 711.30 | 712.73 | 712.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 706.15 | 711.28 | 712.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 13:15:00 | 705.21 | 704.04 | 707.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-14 14:00:00 | 705.21 | 704.04 | 707.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 697.10 | 697.73 | 701.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 12:45:00 | 693.80 | 696.29 | 699.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 14:15:00 | 693.85 | 696.01 | 699.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 15:00:00 | 692.26 | 695.26 | 698.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 10:15:00 | 702.19 | 691.30 | 693.07 | SL hit (close>static) qty=1.00 sl=702.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 12:15:00 | 704.00 | 695.75 | 694.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 13:15:00 | 704.98 | 697.59 | 695.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 707.20 | 707.38 | 703.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 15:00:00 | 707.20 | 707.38 | 703.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 701.72 | 705.92 | 703.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:45:00 | 702.54 | 705.92 | 703.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 704.17 | 705.57 | 703.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 11:15:00 | 705.40 | 705.57 | 703.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 11:15:00 | 719.60 | 724.24 | 724.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 719.60 | 724.24 | 724.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 717.08 | 721.55 | 722.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 11:15:00 | 726.30 | 721.19 | 722.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 11:15:00 | 726.30 | 721.19 | 722.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 726.30 | 721.19 | 722.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:00:00 | 726.30 | 721.19 | 722.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 727.17 | 722.39 | 722.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:30:00 | 727.70 | 722.39 | 722.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 13:15:00 | 729.07 | 723.72 | 723.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 731.81 | 725.34 | 723.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 13:15:00 | 729.00 | 729.45 | 727.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-04 14:00:00 | 729.00 | 729.45 | 727.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 727.10 | 728.98 | 727.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 15:15:00 | 727.83 | 728.98 | 727.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 727.83 | 728.75 | 727.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:15:00 | 730.71 | 728.75 | 727.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 10:15:00 | 734.40 | 737.23 | 737.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 734.40 | 737.23 | 737.28 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-12 11:15:00 | 738.57 | 737.50 | 737.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-12 12:15:00 | 742.89 | 738.58 | 737.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 09:15:00 | 744.68 | 744.73 | 742.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-14 10:00:00 | 744.68 | 744.73 | 742.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 749.15 | 750.28 | 747.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:15:00 | 750.81 | 750.28 | 747.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 749.10 | 750.04 | 748.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 11:45:00 | 753.60 | 750.65 | 748.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 12:30:00 | 753.79 | 751.24 | 749.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 14:00:00 | 754.80 | 751.95 | 749.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 09:15:00 | 755.52 | 755.78 | 753.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 747.56 | 754.14 | 753.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 10:00:00 | 747.56 | 754.14 | 753.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 747.95 | 752.90 | 752.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:00:00 | 747.95 | 752.90 | 752.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-09-21 11:15:00 | 747.70 | 751.86 | 752.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-09-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 11:15:00 | 747.70 | 751.86 | 752.24 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 754.80 | 751.87 | 751.62 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 14:15:00 | 747.00 | 750.90 | 751.22 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 09:15:00 | 776.30 | 755.43 | 753.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 10:15:00 | 780.58 | 760.46 | 755.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 09:15:00 | 779.22 | 783.10 | 775.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-27 09:30:00 | 780.89 | 783.10 | 775.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 778.80 | 782.24 | 776.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:30:00 | 776.65 | 782.24 | 776.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 780.12 | 781.94 | 778.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:45:00 | 777.45 | 781.94 | 778.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 776.61 | 780.88 | 778.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 776.61 | 780.88 | 778.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 780.05 | 780.71 | 778.51 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 773.50 | 777.07 | 777.33 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 783.60 | 778.38 | 777.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 11:15:00 | 784.53 | 781.68 | 780.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 789.09 | 790.58 | 786.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-04 12:00:00 | 789.09 | 790.58 | 786.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 785.18 | 789.50 | 786.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:30:00 | 787.50 | 789.50 | 786.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 13:15:00 | 784.84 | 788.57 | 786.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 14:00:00 | 784.84 | 788.57 | 786.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 14:15:00 | 785.48 | 787.95 | 786.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 15:00:00 | 785.48 | 787.95 | 786.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 15:15:00 | 786.20 | 787.60 | 786.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 786.83 | 787.60 | 786.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 10:45:00 | 788.70 | 788.39 | 786.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 14:00:00 | 787.44 | 788.46 | 787.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 14:15:00 | 784.67 | 787.70 | 787.05 | SL hit (close<static) qty=1.00 sl=785.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 803.34 | 806.39 | 806.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 14:15:00 | 801.50 | 805.41 | 806.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 12:15:00 | 804.10 | 803.63 | 804.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 12:15:00 | 804.10 | 803.63 | 804.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 804.10 | 803.63 | 804.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 13:15:00 | 804.17 | 803.63 | 804.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 806.06 | 804.12 | 804.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:00:00 | 806.06 | 804.12 | 804.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 805.52 | 804.40 | 804.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:30:00 | 806.13 | 804.40 | 804.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 804.95 | 804.51 | 804.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 09:15:00 | 802.09 | 804.51 | 804.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 09:15:00 | 806.30 | 804.87 | 805.11 | SL hit (close>static) qty=1.00 sl=805.84 alert=retest2 |

### Cycle 32 — BUY (started 2023-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 09:15:00 | 811.25 | 804.79 | 804.63 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 791.65 | 803.25 | 804.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 786.91 | 795.57 | 800.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 12:15:00 | 788.89 | 787.35 | 793.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 13:00:00 | 788.89 | 787.35 | 793.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 783.03 | 779.73 | 784.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:30:00 | 786.09 | 779.73 | 784.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 784.89 | 780.76 | 784.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 11:00:00 | 784.89 | 780.76 | 784.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 11:15:00 | 781.00 | 780.81 | 784.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 13:00:00 | 777.50 | 780.15 | 783.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 15:00:00 | 778.90 | 780.32 | 782.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 10:00:00 | 778.98 | 779.60 | 782.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 738.62 | 748.66 | 754.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 739.95 | 748.66 | 754.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 740.03 | 748.66 | 754.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-30 10:15:00 | 749.13 | 748.75 | 754.27 | SL hit (close>ema200) qty=0.50 sl=748.75 alert=retest2 |

### Cycle 34 — BUY (started 2023-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 12:15:00 | 752.96 | 747.21 | 747.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 13:15:00 | 755.67 | 748.90 | 747.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 745.82 | 751.59 | 750.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 11:15:00 | 745.82 | 751.59 | 750.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 745.82 | 751.59 | 750.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:00:00 | 745.82 | 751.59 | 750.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 746.34 | 750.54 | 749.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:30:00 | 746.33 | 750.54 | 749.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 750.40 | 750.49 | 749.82 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 09:15:00 | 743.35 | 748.97 | 749.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 10:15:00 | 740.89 | 745.22 | 746.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 741.34 | 740.99 | 743.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 10:15:00 | 742.61 | 741.31 | 743.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 742.61 | 741.31 | 743.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:30:00 | 742.71 | 741.31 | 743.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 742.71 | 741.59 | 743.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 12:00:00 | 742.71 | 741.59 | 743.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 743.71 | 742.01 | 743.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:00:00 | 743.71 | 742.01 | 743.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 744.90 | 742.59 | 743.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:30:00 | 744.80 | 742.59 | 743.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 744.30 | 742.93 | 743.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 15:15:00 | 745.03 | 742.93 | 743.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 745.03 | 743.35 | 743.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:15:00 | 746.20 | 743.35 | 743.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 741.20 | 738.35 | 740.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 13:30:00 | 738.80 | 736.65 | 739.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 09:15:00 | 701.86 | 730.07 | 735.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-16 12:15:00 | 730.45 | 729.88 | 733.72 | SL hit (close>ema200) qty=0.50 sl=729.88 alert=retest2 |

### Cycle 36 — BUY (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 15:15:00 | 713.00 | 709.07 | 708.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 714.31 | 710.12 | 709.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 710.22 | 711.83 | 710.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 710.22 | 711.83 | 710.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 710.22 | 711.83 | 710.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:00:00 | 710.22 | 711.83 | 710.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 713.09 | 712.08 | 711.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 10:30:00 | 711.90 | 712.08 | 711.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 711.69 | 712.01 | 711.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:00:00 | 711.69 | 712.01 | 711.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 710.64 | 711.73 | 711.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:30:00 | 709.70 | 711.73 | 711.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 13:15:00 | 709.31 | 711.25 | 710.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 14:00:00 | 709.31 | 711.25 | 710.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 14:15:00 | 712.51 | 711.50 | 711.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 15:15:00 | 713.00 | 711.50 | 711.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 11:15:00 | 729.28 | 735.99 | 736.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 729.28 | 735.99 | 736.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 724.70 | 730.19 | 731.60 | Break + close below crossover candle low |

### Cycle 38 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 747.82 | 730.43 | 730.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 749.14 | 736.90 | 733.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 755.07 | 755.79 | 750.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 10:00:00 | 755.07 | 755.79 | 750.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 751.68 | 760.00 | 757.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 751.68 | 760.00 | 757.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 748.51 | 757.70 | 756.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 748.51 | 757.70 | 756.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 748.10 | 755.78 | 755.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 09:15:00 | 744.69 | 753.56 | 754.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 09:15:00 | 725.05 | 723.28 | 730.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 09:15:00 | 725.05 | 723.28 | 730.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 725.05 | 723.28 | 730.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:30:00 | 729.99 | 723.28 | 730.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 725.62 | 723.66 | 727.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-28 10:45:00 | 724.24 | 724.07 | 727.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-28 15:15:00 | 722.40 | 725.29 | 726.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 12:15:00 | 730.70 | 726.97 | 726.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 12:15:00 | 730.70 | 726.97 | 726.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 13:15:00 | 731.07 | 727.79 | 727.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 10:15:00 | 729.44 | 729.60 | 728.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 10:15:00 | 729.44 | 729.60 | 728.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 729.44 | 729.60 | 728.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 11:00:00 | 729.44 | 729.60 | 728.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 730.50 | 729.78 | 728.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 11:30:00 | 728.88 | 729.78 | 728.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 728.70 | 729.91 | 729.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:30:00 | 733.56 | 730.77 | 729.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 15:15:00 | 767.50 | 769.91 | 770.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 15:15:00 | 767.50 | 769.91 | 770.18 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 774.63 | 770.85 | 770.59 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 14:15:00 | 766.74 | 770.85 | 770.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 11:15:00 | 764.30 | 767.84 | 769.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 09:15:00 | 756.10 | 754.04 | 759.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 756.10 | 754.04 | 759.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 756.10 | 754.04 | 759.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:30:00 | 757.04 | 754.04 | 759.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 751.18 | 753.47 | 758.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:30:00 | 758.76 | 753.47 | 758.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 733.00 | 731.05 | 737.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:30:00 | 737.43 | 731.05 | 737.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 733.53 | 731.36 | 734.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 12:30:00 | 731.69 | 732.46 | 734.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 13:30:00 | 732.00 | 732.19 | 733.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 13:15:00 | 719.44 | 713.90 | 713.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 13:15:00 | 719.44 | 713.90 | 713.16 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 09:15:00 | 687.81 | 710.32 | 711.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 682.25 | 693.08 | 701.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 11:15:00 | 690.36 | 690.28 | 697.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-31 11:30:00 | 690.22 | 690.28 | 697.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 685.04 | 680.65 | 685.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:00:00 | 685.04 | 680.65 | 685.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 11:15:00 | 688.62 | 682.24 | 685.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 12:15:00 | 684.80 | 682.24 | 685.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 12:45:00 | 683.58 | 682.21 | 685.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 09:15:00 | 683.51 | 683.76 | 685.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 14:15:00 | 664.59 | 661.30 | 661.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 664.59 | 661.30 | 661.29 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 12:15:00 | 660.30 | 661.20 | 661.32 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 14:15:00 | 662.02 | 661.50 | 661.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 09:15:00 | 663.30 | 661.94 | 661.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 11:15:00 | 662.00 | 662.05 | 661.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 11:15:00 | 662.00 | 662.05 | 661.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 662.00 | 662.05 | 661.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 12:15:00 | 661.55 | 662.05 | 661.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 661.20 | 661.88 | 661.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 670.20 | 661.65 | 661.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 09:15:00 | 661.60 | 669.51 | 670.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 661.60 | 669.51 | 670.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 10:15:00 | 658.94 | 667.40 | 669.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 666.86 | 664.17 | 666.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 666.86 | 664.17 | 666.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 666.86 | 664.17 | 666.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 666.86 | 664.17 | 666.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 669.80 | 665.30 | 667.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:30:00 | 670.53 | 666.09 | 667.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 668.16 | 666.51 | 667.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 12:00:00 | 664.68 | 666.14 | 667.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 14:15:00 | 669.60 | 667.23 | 667.46 | SL hit (close>static) qty=1.00 sl=669.50 alert=retest2 |

### Cycle 50 — BUY (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 15:15:00 | 669.20 | 667.62 | 667.62 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 665.27 | 667.23 | 667.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 12:15:00 | 663.99 | 666.35 | 666.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 15:15:00 | 651.17 | 650.74 | 653.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 15:15:00 | 651.17 | 650.74 | 653.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 651.17 | 650.74 | 653.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 652.61 | 650.74 | 653.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 655.49 | 651.69 | 653.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:00:00 | 655.49 | 651.69 | 653.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 654.27 | 652.20 | 653.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 11:15:00 | 656.16 | 652.20 | 653.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 657.22 | 653.21 | 654.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 11:45:00 | 657.04 | 653.21 | 654.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 12:15:00 | 657.99 | 654.16 | 654.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:45:00 | 658.71 | 654.16 | 654.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 14:15:00 | 656.64 | 655.18 | 655.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 659.15 | 656.36 | 655.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 12:15:00 | 658.11 | 658.47 | 657.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 12:15:00 | 658.11 | 658.47 | 657.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 658.11 | 658.47 | 657.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 12:45:00 | 657.20 | 658.47 | 657.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 658.33 | 659.18 | 658.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:30:00 | 657.61 | 659.18 | 658.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 655.90 | 658.52 | 657.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:30:00 | 654.93 | 658.52 | 657.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 12:15:00 | 641.98 | 655.22 | 656.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 13:15:00 | 631.09 | 650.39 | 654.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 641.55 | 633.26 | 639.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 641.55 | 633.26 | 639.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 641.55 | 633.26 | 639.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:30:00 | 644.40 | 633.26 | 639.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 642.09 | 635.02 | 639.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:30:00 | 641.66 | 635.02 | 639.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 11:15:00 | 642.00 | 636.42 | 639.73 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 10:15:00 | 642.29 | 641.02 | 641.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 12:15:00 | 647.22 | 642.79 | 641.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 14:15:00 | 643.50 | 643.61 | 642.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 15:00:00 | 643.50 | 643.61 | 642.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 643.80 | 643.65 | 642.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:15:00 | 638.55 | 643.65 | 642.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 645.31 | 643.98 | 642.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:30:00 | 644.40 | 643.98 | 642.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 641.76 | 643.54 | 642.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:45:00 | 639.92 | 643.54 | 642.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 644.28 | 643.69 | 642.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:45:00 | 641.66 | 643.69 | 642.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 647.30 | 644.41 | 643.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-12 13:15:00 | 647.64 | 644.41 | 643.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-13 10:00:00 | 650.80 | 645.83 | 644.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-13 11:15:00 | 642.43 | 645.38 | 644.37 | SL hit (close<static) qty=1.00 sl=642.57 alert=retest2 |

### Cycle 55 — SELL (started 2024-03-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 12:15:00 | 640.15 | 644.30 | 644.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 14:15:00 | 638.09 | 642.51 | 643.63 | Break + close below crossover candle low |

### Cycle 56 — BUY (started 2024-03-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 09:15:00 | 654.20 | 644.13 | 644.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 11:15:00 | 658.00 | 654.03 | 650.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 13:15:00 | 724.80 | 725.16 | 715.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-01 14:00:00 | 724.80 | 725.16 | 715.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 715.26 | 723.05 | 716.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:00:00 | 715.26 | 723.05 | 716.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 721.27 | 722.69 | 717.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:30:00 | 717.11 | 722.69 | 717.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 13:15:00 | 720.48 | 721.59 | 717.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 15:15:00 | 722.30 | 721.49 | 718.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-05 11:15:00 | 716.76 | 724.14 | 724.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-04-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 11:15:00 | 716.76 | 724.14 | 724.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 713.64 | 717.89 | 719.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 14:15:00 | 718.47 | 717.68 | 719.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 14:15:00 | 718.47 | 717.68 | 719.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 718.47 | 717.68 | 719.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:45:00 | 719.04 | 717.68 | 719.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 718.90 | 717.92 | 719.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:15:00 | 723.56 | 717.92 | 719.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 720.37 | 718.41 | 719.16 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 14:15:00 | 722.03 | 719.93 | 719.66 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 714.92 | 720.10 | 720.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 707.80 | 715.34 | 717.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 13:15:00 | 702.85 | 702.74 | 709.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 14:00:00 | 702.85 | 702.74 | 709.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 706.35 | 694.38 | 697.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:00:00 | 706.35 | 694.38 | 697.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 712.34 | 697.97 | 698.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:30:00 | 712.27 | 697.97 | 698.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 15:15:00 | 711.00 | 700.57 | 699.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 723.00 | 705.06 | 701.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 725.82 | 726.10 | 719.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 15:00:00 | 725.82 | 726.10 | 719.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 721.56 | 729.26 | 725.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 721.56 | 729.26 | 725.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 722.25 | 727.86 | 725.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:30:00 | 727.90 | 727.21 | 725.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 12:45:00 | 725.42 | 727.20 | 725.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 680.84 | 719.36 | 722.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 09:15:00 | 680.84 | 719.36 | 722.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 10:15:00 | 675.80 | 710.65 | 718.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 13:15:00 | 681.80 | 680.28 | 692.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 14:00:00 | 681.80 | 680.28 | 692.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 688.50 | 683.58 | 690.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:45:00 | 688.00 | 683.58 | 690.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 689.46 | 684.76 | 689.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:00:00 | 689.46 | 684.76 | 689.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 688.33 | 685.47 | 689.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:30:00 | 690.00 | 685.47 | 689.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 692.50 | 686.88 | 690.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 13:45:00 | 692.93 | 686.88 | 690.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 691.01 | 687.70 | 690.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:45:00 | 696.82 | 687.70 | 690.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 691.76 | 688.93 | 690.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:15:00 | 690.53 | 688.93 | 690.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:45:00 | 689.91 | 688.72 | 690.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 09:15:00 | 725.20 | 696.07 | 692.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 725.20 | 696.07 | 692.83 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 687.72 | 693.78 | 694.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 14:15:00 | 686.20 | 691.40 | 693.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 12:15:00 | 690.32 | 687.84 | 690.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 12:15:00 | 690.32 | 687.84 | 690.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 690.32 | 687.84 | 690.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:30:00 | 691.53 | 687.84 | 690.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 687.16 | 687.70 | 690.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 14:15:00 | 684.89 | 687.70 | 690.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 684.31 | 687.84 | 689.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 10:15:00 | 684.71 | 687.71 | 689.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 15:15:00 | 673.10 | 669.16 | 668.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 673.10 | 669.16 | 668.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 676.56 | 671.06 | 669.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 672.76 | 673.24 | 671.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 15:00:00 | 672.76 | 673.24 | 671.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 676.57 | 674.25 | 672.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:15:00 | 678.20 | 674.25 | 672.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 677.55 | 675.85 | 674.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 12:30:00 | 681.00 | 677.04 | 675.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 13:15:00 | 683.78 | 687.26 | 687.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 683.78 | 687.26 | 687.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 681.12 | 686.03 | 686.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 672.79 | 670.07 | 675.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:45:00 | 673.90 | 670.07 | 675.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 675.20 | 671.10 | 675.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:30:00 | 671.25 | 671.73 | 675.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 13:15:00 | 672.17 | 671.73 | 675.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:45:00 | 671.98 | 671.78 | 674.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 690.61 | 675.78 | 676.09 | SL hit (close>static) qty=1.00 sl=677.20 alert=retest2 |

### Cycle 66 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 695.55 | 679.74 | 677.86 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 653.66 | 675.58 | 678.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 650.76 | 666.68 | 673.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 668.28 | 664.76 | 670.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 668.28 | 664.76 | 670.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 672.80 | 666.37 | 670.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 672.80 | 666.37 | 670.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 677.90 | 668.68 | 671.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:45:00 | 677.96 | 668.68 | 671.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 683.07 | 673.30 | 673.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 696.94 | 679.68 | 676.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 13:15:00 | 712.77 | 713.28 | 705.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:15:00 | 710.15 | 713.28 | 705.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 712.30 | 711.75 | 706.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:30:00 | 713.30 | 712.23 | 708.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 725.38 | 729.39 | 729.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 725.38 | 729.39 | 729.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 13:15:00 | 722.60 | 728.03 | 728.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 12:15:00 | 721.37 | 721.12 | 724.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 13:00:00 | 721.37 | 721.12 | 724.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 719.29 | 720.75 | 724.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:30:00 | 718.31 | 719.34 | 722.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 717.78 | 710.89 | 710.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 717.78 | 710.89 | 710.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 720.00 | 713.94 | 712.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 711.10 | 714.63 | 713.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 711.10 | 714.63 | 713.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 711.10 | 714.63 | 713.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 711.10 | 714.63 | 713.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 710.23 | 713.75 | 712.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 710.23 | 713.75 | 712.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 716.88 | 714.38 | 713.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 711.21 | 714.38 | 713.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 718.16 | 715.55 | 713.95 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 711.41 | 713.76 | 713.94 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 717.86 | 714.38 | 714.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 724.20 | 716.35 | 715.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 714.21 | 718.93 | 716.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 714.21 | 718.93 | 716.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 714.21 | 718.93 | 716.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 714.21 | 718.93 | 716.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 718.40 | 718.82 | 716.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:45:00 | 721.69 | 718.28 | 717.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 723.18 | 717.55 | 716.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:45:00 | 721.61 | 718.26 | 717.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 720.43 | 722.11 | 720.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 716.70 | 721.03 | 719.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:00:00 | 716.70 | 721.03 | 719.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 713.90 | 719.60 | 719.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-04 13:15:00 | 713.30 | 718.34 | 718.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 13:15:00 | 713.30 | 718.34 | 718.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 15:15:00 | 710.90 | 715.71 | 717.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 714.30 | 713.20 | 715.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 15:00:00 | 714.30 | 713.20 | 715.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 708.40 | 712.25 | 714.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 14:00:00 | 703.51 | 706.52 | 708.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 14:45:00 | 704.40 | 706.47 | 708.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 15:15:00 | 704.50 | 706.47 | 708.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 10:15:00 | 704.30 | 701.30 | 701.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 704.88 | 702.01 | 702.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:15:00 | 703.28 | 702.01 | 702.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 11:15:00 | 705.50 | 702.71 | 702.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 705.50 | 702.71 | 702.36 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 700.72 | 703.49 | 703.80 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 13:15:00 | 709.15 | 704.97 | 704.44 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 700.19 | 704.60 | 704.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 696.26 | 702.16 | 703.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 10:15:00 | 663.20 | 663.13 | 671.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 10:45:00 | 663.17 | 663.13 | 671.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 673.75 | 666.25 | 669.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:30:00 | 673.53 | 666.25 | 669.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 676.02 | 668.21 | 669.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 676.02 | 668.21 | 669.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 679.00 | 671.81 | 671.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 679.95 | 675.35 | 673.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 681.50 | 682.93 | 680.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 681.50 | 682.93 | 680.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 682.59 | 682.57 | 680.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 682.80 | 682.57 | 680.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 680.04 | 682.16 | 681.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 680.04 | 682.16 | 681.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 680.95 | 681.92 | 681.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 683.42 | 681.92 | 681.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 09:15:00 | 678.87 | 681.31 | 680.85 | SL hit (close<static) qty=1.00 sl=679.30 alert=retest2 |

### Cycle 79 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 677.98 | 680.58 | 680.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 677.32 | 679.60 | 680.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 663.00 | 662.58 | 667.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 663.00 | 662.58 | 667.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 663.00 | 662.58 | 667.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 660.21 | 662.58 | 667.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 659.47 | 660.96 | 665.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:00:00 | 662.00 | 659.20 | 663.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:30:00 | 662.20 | 661.15 | 663.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 663.74 | 661.67 | 663.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 663.74 | 661.67 | 663.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 664.01 | 662.14 | 663.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:15:00 | 663.40 | 662.14 | 663.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:00:00 | 662.06 | 662.32 | 663.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 662.00 | 662.32 | 663.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 662.74 | 662.82 | 663.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 660.56 | 662.37 | 662.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:45:00 | 660.53 | 662.37 | 662.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 661.31 | 660.84 | 662.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 659.62 | 660.84 | 662.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:00:00 | 659.95 | 660.89 | 661.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 659.60 | 661.18 | 661.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 14:45:00 | 660.01 | 661.39 | 661.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 660.00 | 661.11 | 661.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 658.00 | 661.11 | 661.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 658.29 | 660.55 | 661.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:45:00 | 656.20 | 659.02 | 660.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 14:15:00 | 659.45 | 652.50 | 652.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 659.45 | 652.50 | 652.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 660.99 | 655.18 | 653.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 671.72 | 673.79 | 671.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 671.72 | 673.79 | 671.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 671.72 | 673.79 | 671.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 671.50 | 673.79 | 671.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 673.11 | 673.65 | 671.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:15:00 | 673.80 | 673.65 | 671.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 677.80 | 673.86 | 672.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-02 10:15:00 | 741.18 | 723.32 | 711.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 725.00 | 727.10 | 727.36 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 10:15:00 | 731.80 | 727.64 | 727.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 11:15:00 | 735.27 | 729.16 | 728.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 09:15:00 | 728.86 | 730.02 | 729.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 728.86 | 730.02 | 729.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 728.86 | 730.02 | 729.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 728.86 | 730.02 | 729.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 732.43 | 730.50 | 729.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 12:30:00 | 734.34 | 731.55 | 730.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 14:45:00 | 734.10 | 732.26 | 730.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 724.18 | 730.96 | 730.32 | SL hit (close<static) qty=1.00 sl=726.28 alert=retest2 |

### Cycle 83 — SELL (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 10:15:00 | 723.40 | 729.44 | 729.69 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 741.10 | 730.08 | 729.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 743.72 | 736.82 | 733.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 736.23 | 750.66 | 745.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 736.23 | 750.66 | 745.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 736.23 | 750.66 | 745.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 736.23 | 750.66 | 745.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 736.40 | 747.81 | 744.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:45:00 | 739.28 | 747.81 | 744.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 737.80 | 745.81 | 743.72 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 735.56 | 742.11 | 742.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 734.20 | 739.48 | 741.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 09:15:00 | 741.49 | 739.88 | 741.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 741.49 | 739.88 | 741.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 741.49 | 739.88 | 741.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 741.49 | 739.88 | 741.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 738.50 | 739.61 | 740.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 14:45:00 | 735.30 | 738.85 | 740.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 752.70 | 741.18 | 740.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 09:15:00 | 752.70 | 741.18 | 740.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 10:15:00 | 755.00 | 743.95 | 742.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 758.72 | 759.24 | 752.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 757.30 | 759.00 | 755.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 757.30 | 759.00 | 755.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 759.72 | 759.00 | 755.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 756.69 | 759.00 | 756.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 753.50 | 759.00 | 756.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 759.20 | 759.04 | 756.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 15:15:00 | 759.80 | 759.04 | 756.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 13:45:00 | 760.00 | 757.73 | 756.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 14:15:00 | 760.10 | 757.73 | 756.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 752.67 | 757.07 | 756.84 | SL hit (close<static) qty=1.00 sl=752.71 alert=retest2 |

### Cycle 87 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 753.79 | 756.42 | 756.57 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 10:15:00 | 759.81 | 756.26 | 756.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 14:15:00 | 763.50 | 759.19 | 757.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 13:15:00 | 774.33 | 775.52 | 772.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 13:30:00 | 773.98 | 775.52 | 772.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 770.22 | 774.46 | 772.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 770.22 | 774.46 | 772.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 769.79 | 773.53 | 771.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 767.22 | 773.53 | 771.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 778.36 | 774.49 | 772.56 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 763.60 | 771.33 | 772.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 759.01 | 768.87 | 770.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 737.02 | 725.41 | 729.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 737.02 | 725.41 | 729.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 737.02 | 725.41 | 729.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 737.02 | 725.41 | 729.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 740.54 | 728.43 | 730.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 740.54 | 728.43 | 730.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 736.00 | 731.43 | 731.32 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 728.90 | 730.93 | 731.10 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 735.82 | 731.90 | 731.51 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 09:15:00 | 727.30 | 731.39 | 731.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 722.35 | 726.00 | 728.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 691.74 | 690.61 | 696.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 691.74 | 690.61 | 696.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 682.00 | 688.68 | 693.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:45:00 | 678.75 | 686.50 | 691.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:30:00 | 679.42 | 679.38 | 685.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:00:00 | 679.44 | 679.39 | 684.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 700.77 | 678.78 | 681.36 | SL hit (close>static) qty=1.00 sl=695.08 alert=retest2 |

### Cycle 94 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 10:15:00 | 705.88 | 684.20 | 683.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 09:15:00 | 708.52 | 698.98 | 692.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 696.55 | 701.02 | 697.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 696.55 | 701.02 | 697.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 696.55 | 701.02 | 697.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 696.55 | 701.02 | 697.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 699.40 | 700.70 | 697.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 693.20 | 700.70 | 697.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 696.56 | 699.97 | 697.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 13:00:00 | 696.56 | 699.97 | 697.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 693.71 | 698.72 | 697.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:00:00 | 693.71 | 698.72 | 697.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 690.50 | 695.75 | 696.08 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 701.75 | 696.72 | 696.45 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 12:15:00 | 691.28 | 695.68 | 696.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 684.30 | 691.77 | 693.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 12:15:00 | 696.43 | 691.98 | 693.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 12:15:00 | 696.43 | 691.98 | 693.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 696.43 | 691.98 | 693.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:45:00 | 696.73 | 691.98 | 693.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 703.78 | 694.34 | 694.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 703.78 | 694.34 | 694.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 703.77 | 696.23 | 695.21 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 692.55 | 695.45 | 695.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 691.09 | 693.98 | 694.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 693.08 | 692.03 | 693.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 693.08 | 692.03 | 693.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 693.08 | 692.03 | 693.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 693.08 | 692.03 | 693.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 691.11 | 691.85 | 693.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:30:00 | 693.08 | 691.85 | 693.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 682.86 | 690.05 | 692.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:15:00 | 680.50 | 690.05 | 692.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 09:30:00 | 681.32 | 683.76 | 687.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 10:30:00 | 678.91 | 683.07 | 686.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 13:15:00 | 697.52 | 683.66 | 685.73 | SL hit (close>static) qty=1.00 sl=697.50 alert=retest2 |

### Cycle 100 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 692.16 | 687.51 | 687.18 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 686.42 | 690.33 | 690.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 681.76 | 686.78 | 688.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 660.41 | 658.06 | 662.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 660.41 | 658.06 | 662.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 660.41 | 658.06 | 662.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:00:00 | 657.94 | 658.03 | 661.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 657.71 | 658.69 | 661.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:30:00 | 657.75 | 659.48 | 660.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:00:00 | 657.47 | 659.48 | 660.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 659.38 | 659.46 | 660.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:45:00 | 660.25 | 659.46 | 660.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 658.00 | 659.17 | 660.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 655.10 | 659.17 | 660.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 650.36 | 657.41 | 659.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 11:30:00 | 648.50 | 654.21 | 657.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:00:00 | 648.16 | 654.21 | 657.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 14:00:00 | 648.50 | 652.12 | 655.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 12:15:00 | 662.47 | 655.05 | 655.44 | SL hit (close>static) qty=1.00 sl=661.40 alert=retest2 |

### Cycle 102 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 665.57 | 657.16 | 656.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 667.63 | 659.25 | 657.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 13:15:00 | 669.09 | 669.38 | 664.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:45:00 | 668.40 | 669.38 | 664.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 669.15 | 668.99 | 665.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:45:00 | 670.87 | 666.12 | 665.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:30:00 | 669.49 | 666.89 | 665.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 669.21 | 667.12 | 666.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 10:00:00 | 671.40 | 669.70 | 667.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 663.52 | 668.47 | 667.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 664.50 | 668.47 | 667.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 662.58 | 667.29 | 667.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-28 11:15:00 | 662.58 | 667.29 | 667.07 | SL hit (close<static) qty=1.00 sl=663.10 alert=retest2 |

### Cycle 103 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 660.00 | 665.83 | 666.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 656.89 | 664.04 | 665.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 659.20 | 658.89 | 662.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:45:00 | 660.20 | 658.89 | 662.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 650.98 | 656.31 | 659.41 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 664.70 | 660.88 | 660.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 666.38 | 661.98 | 660.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 12:15:00 | 664.17 | 664.22 | 662.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 13:00:00 | 664.17 | 664.22 | 662.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 670.91 | 671.63 | 668.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 670.17 | 671.63 | 668.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 686.21 | 688.18 | 685.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 686.07 | 688.18 | 685.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 692.36 | 689.02 | 686.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:45:00 | 690.20 | 689.02 | 686.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 705.74 | 709.94 | 704.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:15:00 | 702.70 | 709.94 | 704.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 702.54 | 708.46 | 704.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 703.14 | 708.46 | 704.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 711.88 | 709.14 | 705.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:15:00 | 714.50 | 709.14 | 705.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:45:00 | 714.97 | 710.08 | 706.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:15:00 | 716.10 | 710.08 | 706.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:45:00 | 718.40 | 719.70 | 715.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 713.50 | 718.31 | 715.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 713.50 | 718.31 | 715.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 713.20 | 717.29 | 715.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:30:00 | 717.41 | 715.07 | 714.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 710.90 | 714.24 | 714.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 710.90 | 714.24 | 714.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 12:15:00 | 708.24 | 712.47 | 713.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 09:15:00 | 693.03 | 688.78 | 694.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 09:15:00 | 693.03 | 688.78 | 694.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 693.03 | 688.78 | 694.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 691.97 | 688.78 | 694.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 695.32 | 690.09 | 694.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 695.32 | 690.09 | 694.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 690.59 | 690.19 | 694.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:00:00 | 689.16 | 689.98 | 693.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 694.81 | 687.03 | 686.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 694.81 | 687.03 | 686.72 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 681.24 | 688.26 | 688.69 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 689.50 | 687.00 | 686.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 692.70 | 688.97 | 687.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 736.78 | 737.74 | 725.28 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 09:15:00 | 755.44 | 737.74 | 725.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 12:45:00 | 743.50 | 741.49 | 731.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-06 13:15:00 | 742.71 | 741.49 | 731.44 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 736.99 | 738.70 | 732.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 740.39 | 738.70 | 732.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:15:00 | 739.04 | 738.26 | 732.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 15:15:00 | 734.00 | 736.42 | 734.27 | SL hit (close<ema400) qty=1.00 sl=734.27 alert=retest1 |

### Cycle 109 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 730.14 | 733.04 | 733.07 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 14:15:00 | 735.05 | 733.32 | 733.16 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 09:15:00 | 730.00 | 732.75 | 732.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 726.50 | 731.50 | 732.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 15:15:00 | 729.17 | 728.90 | 730.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 09:15:00 | 726.83 | 728.90 | 730.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 729.16 | 728.95 | 730.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:15:00 | 729.79 | 728.95 | 730.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 732.97 | 729.75 | 730.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 11:00:00 | 732.97 | 729.75 | 730.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 731.00 | 730.00 | 730.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:45:00 | 728.70 | 729.65 | 730.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 14:00:00 | 727.37 | 729.19 | 730.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 736.50 | 726.92 | 726.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 736.50 | 726.92 | 726.58 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 10:15:00 | 718.63 | 726.09 | 726.93 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 726.77 | 724.47 | 724.32 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 720.71 | 723.77 | 724.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 718.40 | 722.15 | 723.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 723.31 | 721.30 | 722.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 723.31 | 721.30 | 722.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 723.31 | 721.30 | 722.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 722.46 | 721.30 | 722.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 736.39 | 724.32 | 723.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 746.13 | 728.68 | 725.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 734.98 | 736.94 | 732.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:30:00 | 734.93 | 736.94 | 732.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 736.73 | 737.57 | 733.75 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 730.89 | 732.01 | 732.11 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 741.46 | 733.70 | 732.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 744.35 | 737.72 | 735.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 736.83 | 742.00 | 739.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 736.83 | 742.00 | 739.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 736.83 | 742.00 | 739.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 736.83 | 742.00 | 739.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 746.08 | 742.82 | 739.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:30:00 | 747.70 | 744.05 | 740.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 13:00:00 | 747.91 | 744.83 | 741.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 12:15:00 | 735.04 | 740.20 | 740.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2025-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 12:15:00 | 735.04 | 740.20 | 740.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 726.77 | 737.51 | 739.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 740.20 | 735.48 | 737.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 09:15:00 | 740.20 | 735.48 | 737.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 740.20 | 735.48 | 737.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:30:00 | 741.44 | 735.48 | 737.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 742.55 | 736.89 | 738.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 742.55 | 736.89 | 738.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 751.99 | 739.91 | 739.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 760.90 | 744.11 | 741.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 11:15:00 | 786.75 | 788.84 | 779.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 12:00:00 | 786.75 | 788.84 | 779.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 784.58 | 788.00 | 783.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 781.99 | 788.00 | 783.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 788.49 | 788.10 | 784.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 788.49 | 788.10 | 784.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 852.01 | 851.23 | 845.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:45:00 | 841.47 | 851.23 | 845.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 846.31 | 850.25 | 845.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:00:00 | 846.31 | 850.25 | 845.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 851.50 | 850.50 | 846.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:30:00 | 852.16 | 849.90 | 846.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 835.65 | 846.46 | 845.56 | SL hit (close<static) qty=1.00 sl=845.73 alert=retest2 |

### Cycle 121 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 834.40 | 844.05 | 844.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 826.65 | 840.57 | 842.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 821.72 | 818.56 | 825.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 821.72 | 818.56 | 825.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 823.59 | 819.56 | 825.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 821.77 | 819.56 | 825.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 820.42 | 819.73 | 824.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 825.86 | 819.73 | 824.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 840.44 | 824.20 | 825.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 840.44 | 824.20 | 825.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 844.25 | 828.21 | 827.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 846.63 | 842.92 | 840.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 14:15:00 | 844.48 | 846.02 | 843.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 14:15:00 | 844.48 | 846.02 | 843.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 844.48 | 846.02 | 843.49 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 11:15:00 | 838.51 | 842.27 | 842.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 829.40 | 838.44 | 840.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 12:15:00 | 837.75 | 837.67 | 839.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 12:15:00 | 837.75 | 837.67 | 839.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 837.75 | 837.67 | 839.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:30:00 | 838.48 | 837.67 | 839.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 840.43 | 838.18 | 839.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:00:00 | 840.43 | 838.18 | 839.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 838.70 | 838.28 | 839.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 837.00 | 838.28 | 839.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 12:00:00 | 836.73 | 837.56 | 838.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 09:15:00 | 846.61 | 838.50 | 838.61 | SL hit (close>static) qty=1.00 sl=841.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 10:15:00 | 849.85 | 840.77 | 839.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 868.80 | 850.42 | 845.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 858.60 | 863.33 | 856.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 858.60 | 863.33 | 856.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 858.60 | 863.33 | 856.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 858.60 | 863.33 | 856.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 853.39 | 860.92 | 856.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:00:00 | 853.39 | 860.92 | 856.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 855.40 | 859.81 | 856.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:30:00 | 853.20 | 859.81 | 856.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 854.60 | 857.62 | 856.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 849.20 | 857.62 | 856.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-03-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 10:15:00 | 851.00 | 854.75 | 855.01 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 12:15:00 | 862.80 | 856.32 | 855.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 15:15:00 | 864.70 | 859.37 | 857.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 09:15:00 | 855.75 | 858.65 | 857.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 855.75 | 858.65 | 857.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 855.75 | 858.65 | 857.19 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 09:15:00 | 834.75 | 852.86 | 854.97 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 09:15:00 | 851.40 | 843.39 | 843.19 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 834.63 | 843.21 | 843.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 833.00 | 841.16 | 842.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 834.82 | 834.38 | 838.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 13:30:00 | 832.62 | 834.38 | 838.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 831.50 | 833.78 | 836.95 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 845.80 | 838.46 | 838.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 15:15:00 | 852.32 | 842.72 | 840.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 12:15:00 | 840.59 | 845.03 | 842.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 12:15:00 | 840.59 | 845.03 | 842.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 840.59 | 845.03 | 842.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:45:00 | 840.25 | 845.03 | 842.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 843.00 | 844.62 | 842.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 858.32 | 843.65 | 842.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 887.00 | 896.21 | 896.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 887.00 | 896.21 | 896.90 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 901.80 | 896.82 | 896.53 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 09:15:00 | 887.00 | 895.92 | 896.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 872.00 | 889.09 | 892.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 09:15:00 | 866.56 | 861.12 | 866.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 866.56 | 861.12 | 866.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 866.56 | 861.12 | 866.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:30:00 | 863.90 | 861.12 | 866.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 876.18 | 864.13 | 867.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 11:00:00 | 876.18 | 864.13 | 867.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 873.79 | 866.06 | 868.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 13:45:00 | 871.34 | 868.69 | 869.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 14:45:00 | 871.50 | 869.38 | 869.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 15:15:00 | 873.20 | 870.14 | 869.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 15:15:00 | 873.20 | 870.14 | 869.74 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 837.96 | 863.71 | 866.85 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 880.00 | 865.53 | 863.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 885.23 | 869.47 | 865.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 875.36 | 875.64 | 870.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-09 09:30:00 | 883.78 | 875.64 | 870.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 865.25 | 873.56 | 869.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:00:00 | 865.25 | 873.56 | 869.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 864.96 | 871.84 | 869.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 870.25 | 871.84 | 869.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 14:30:00 | 869.19 | 871.61 | 869.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-24 09:15:00 | 957.28 | 928.57 | 925.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 906.65 | 924.76 | 925.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 861.35 | 898.50 | 905.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 12:15:00 | 875.45 | 870.28 | 880.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:00:00 | 875.45 | 870.28 | 880.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 886.30 | 873.48 | 881.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:30:00 | 884.90 | 873.48 | 881.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 887.00 | 876.19 | 881.92 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 899.30 | 885.38 | 885.05 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 878.60 | 886.35 | 886.51 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 890.00 | 885.72 | 885.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 894.80 | 887.53 | 886.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 11:15:00 | 894.45 | 894.67 | 890.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 11:45:00 | 894.40 | 894.67 | 890.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 892.00 | 894.14 | 890.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:45:00 | 893.45 | 894.14 | 890.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 890.00 | 893.31 | 890.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 889.85 | 893.31 | 890.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 884.70 | 891.59 | 890.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 884.70 | 891.59 | 890.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 878.00 | 888.87 | 889.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 873.55 | 885.81 | 887.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 894.00 | 875.16 | 879.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 894.00 | 875.16 | 879.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 894.00 | 875.16 | 879.32 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 897.60 | 882.66 | 882.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 901.55 | 888.87 | 885.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 903.00 | 903.75 | 896.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 903.00 | 903.75 | 896.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 912.90 | 907.59 | 904.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:30:00 | 905.00 | 907.59 | 904.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 921.50 | 922.65 | 918.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 919.80 | 922.65 | 918.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 918.65 | 921.81 | 918.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 918.70 | 921.81 | 918.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 915.55 | 920.56 | 918.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 915.55 | 920.56 | 918.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 910.90 | 918.63 | 917.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 910.90 | 918.63 | 917.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 908.65 | 916.63 | 917.01 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 925.50 | 917.15 | 917.00 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 914.30 | 916.87 | 917.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 912.65 | 916.03 | 916.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 15:15:00 | 915.05 | 914.86 | 915.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 09:15:00 | 917.65 | 914.86 | 915.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 921.05 | 916.10 | 916.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 920.75 | 916.10 | 916.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 924.30 | 917.74 | 917.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 926.40 | 920.33 | 918.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 926.20 | 926.89 | 924.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 920.10 | 926.89 | 924.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 917.25 | 924.96 | 923.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 917.25 | 924.96 | 923.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 923.85 | 924.74 | 923.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 928.00 | 924.74 | 923.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 916.80 | 922.19 | 922.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 916.80 | 922.19 | 922.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 915.50 | 919.42 | 921.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 923.75 | 920.29 | 921.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 923.75 | 920.29 | 921.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 923.75 | 920.29 | 921.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 923.75 | 920.29 | 921.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 924.65 | 921.16 | 921.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:15:00 | 925.30 | 921.16 | 921.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 927.65 | 922.46 | 922.20 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 917.95 | 922.43 | 922.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 916.40 | 920.86 | 921.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 920.95 | 920.88 | 921.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 920.95 | 920.88 | 921.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 920.95 | 920.88 | 921.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 920.95 | 920.88 | 921.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 921.00 | 920.90 | 921.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 918.90 | 920.90 | 921.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 932.00 | 902.67 | 901.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 932.00 | 902.67 | 901.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 938.30 | 909.80 | 904.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 956.65 | 957.04 | 942.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:45:00 | 956.50 | 957.04 | 942.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 953.15 | 952.84 | 946.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 945.20 | 952.84 | 946.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 946.20 | 950.95 | 946.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 946.20 | 950.95 | 946.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 945.30 | 949.82 | 946.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 945.65 | 949.82 | 946.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 942.00 | 948.26 | 946.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 942.00 | 948.26 | 946.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 945.15 | 946.95 | 946.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 945.15 | 946.95 | 946.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 943.80 | 946.32 | 945.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 943.80 | 946.32 | 945.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 945.40 | 946.14 | 945.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 943.00 | 946.14 | 945.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 937.05 | 944.32 | 945.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 926.45 | 938.62 | 942.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 932.05 | 931.69 | 937.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 933.15 | 931.69 | 937.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 934.00 | 932.20 | 936.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 939.50 | 932.20 | 936.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 931.50 | 932.06 | 935.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:00:00 | 923.00 | 932.59 | 935.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 921.00 | 930.17 | 933.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 923.50 | 926.05 | 929.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 924.50 | 925.84 | 929.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 904.00 | 901.91 | 906.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 904.00 | 901.91 | 906.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 912.00 | 903.93 | 907.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 909.50 | 903.93 | 907.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 910.00 | 905.14 | 907.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 912.00 | 905.14 | 907.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 915.50 | 907.21 | 908.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 915.50 | 907.21 | 908.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-23 15:15:00 | 913.00 | 909.78 | 909.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 913.00 | 909.78 | 909.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 924.50 | 912.72 | 910.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 920.00 | 921.01 | 916.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 920.00 | 921.01 | 916.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 916.00 | 920.00 | 916.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 916.00 | 920.00 | 916.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 919.00 | 919.80 | 916.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 922.00 | 919.80 | 916.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 921.50 | 921.26 | 918.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 933.50 | 937.16 | 937.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 933.50 | 937.16 | 937.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 926.60 | 934.64 | 936.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 933.55 | 918.96 | 922.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 933.55 | 918.96 | 922.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 933.55 | 918.96 | 922.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 937.25 | 918.96 | 922.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 932.50 | 921.67 | 923.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 927.00 | 922.29 | 923.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 924.80 | 923.90 | 923.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 924.80 | 923.90 | 923.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 927.60 | 924.92 | 924.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 941.45 | 944.18 | 939.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 941.45 | 944.18 | 939.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 941.45 | 944.18 | 939.40 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 932.40 | 937.21 | 937.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 919.75 | 933.72 | 935.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 923.15 | 922.40 | 927.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 923.15 | 922.40 | 927.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 926.95 | 922.99 | 926.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:45:00 | 927.75 | 922.99 | 926.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 928.65 | 924.12 | 926.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 928.65 | 924.12 | 926.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 929.00 | 925.10 | 926.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 921.25 | 925.10 | 926.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 935.50 | 924.51 | 924.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 935.50 | 924.51 | 924.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 13:15:00 | 936.80 | 927.44 | 925.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 942.50 | 944.20 | 938.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:30:00 | 941.30 | 944.20 | 938.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 956.70 | 961.15 | 955.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 956.70 | 961.15 | 955.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 958.85 | 960.69 | 956.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:30:00 | 959.70 | 960.65 | 956.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 960.50 | 960.65 | 956.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 960.35 | 960.37 | 956.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 918.10 | 951.28 | 953.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 918.10 | 951.28 | 953.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 891.55 | 911.35 | 926.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 889.05 | 889.03 | 903.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:45:00 | 891.40 | 889.03 | 903.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 885.00 | 882.10 | 887.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 885.00 | 882.10 | 887.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 889.70 | 883.16 | 886.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 889.70 | 883.16 | 886.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 890.20 | 884.57 | 886.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 890.20 | 884.57 | 886.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 881.45 | 883.94 | 886.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 880.35 | 883.41 | 885.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:00:00 | 873.90 | 881.51 | 884.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:45:00 | 879.10 | 880.29 | 883.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 880.95 | 880.85 | 883.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 885.00 | 881.68 | 883.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 885.00 | 881.68 | 883.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 882.05 | 881.75 | 883.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 888.65 | 884.44 | 884.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 888.65 | 884.44 | 884.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 892.85 | 887.27 | 885.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 882.60 | 887.14 | 886.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 882.60 | 887.14 | 886.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 882.60 | 887.14 | 886.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 882.60 | 887.14 | 886.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 875.75 | 884.86 | 885.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 872.05 | 878.63 | 881.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 877.25 | 875.78 | 879.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 877.25 | 875.78 | 879.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 880.05 | 876.63 | 879.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 880.05 | 876.63 | 879.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 881.90 | 877.69 | 879.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 885.45 | 877.69 | 879.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 882.25 | 878.60 | 879.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 876.40 | 878.61 | 879.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 876.85 | 877.69 | 878.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 873.55 | 877.35 | 878.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 915.20 | 871.06 | 866.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 915.20 | 871.06 | 866.59 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 886.65 | 895.22 | 896.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 881.05 | 891.03 | 893.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 885.05 | 881.28 | 884.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 885.05 | 881.28 | 884.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 885.05 | 881.28 | 884.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 885.05 | 881.28 | 884.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 887.60 | 882.54 | 885.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 887.60 | 882.54 | 885.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 880.95 | 882.22 | 884.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 878.80 | 881.74 | 884.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 876.50 | 880.69 | 883.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 888.60 | 882.00 | 883.66 | SL hit (close>static) qty=1.00 sl=887.95 alert=retest2 |

### Cycle 162 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 889.30 | 885.32 | 884.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 898.30 | 889.25 | 886.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 888.35 | 891.15 | 888.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 888.35 | 891.15 | 888.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 888.35 | 891.15 | 888.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 888.35 | 891.15 | 888.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 890.40 | 891.00 | 888.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 891.70 | 891.03 | 889.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 891.60 | 891.14 | 889.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-12 09:15:00 | 980.87 | 971.97 | 963.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 994.75 | 1001.61 | 1002.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 994.00 | 1000.09 | 1001.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 995.05 | 994.71 | 997.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 14:00:00 | 995.05 | 994.71 | 997.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1000.25 | 995.44 | 997.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 1002.40 | 995.44 | 997.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1006.20 | 997.59 | 997.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1008.50 | 997.59 | 997.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 1000.90 | 998.25 | 998.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 1021.35 | 1006.20 | 1002.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 1027.10 | 1027.26 | 1021.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1027.10 | 1027.26 | 1021.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1027.10 | 1027.26 | 1021.04 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1012.50 | 1019.22 | 1019.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1002.10 | 1015.80 | 1017.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 992.25 | 991.41 | 999.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 992.25 | 991.41 | 999.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 996.00 | 992.55 | 998.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 995.90 | 992.55 | 998.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 999.80 | 994.00 | 998.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 999.80 | 994.00 | 998.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 994.80 | 994.16 | 998.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 989.10 | 996.73 | 998.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1013.95 | 992.42 | 991.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 1013.95 | 992.42 | 991.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 1024.05 | 1009.35 | 1001.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1014.95 | 1015.69 | 1008.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 1014.95 | 1015.69 | 1008.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1017.85 | 1020.92 | 1016.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 1017.85 | 1020.92 | 1016.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1024.55 | 1023.82 | 1020.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 1026.35 | 1024.32 | 1020.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 1026.80 | 1024.48 | 1021.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1026.20 | 1024.57 | 1022.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:45:00 | 1026.65 | 1025.89 | 1023.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1026.70 | 1030.44 | 1027.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 1026.70 | 1030.44 | 1027.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1020.45 | 1028.44 | 1026.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1020.45 | 1028.44 | 1026.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1018.00 | 1026.35 | 1025.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 1018.25 | 1026.35 | 1025.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 1015.60 | 1024.20 | 1024.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1015.60 | 1024.20 | 1024.80 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1041.00 | 1026.51 | 1025.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1046.75 | 1030.56 | 1027.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 1089.05 | 1091.03 | 1085.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:45:00 | 1088.90 | 1091.03 | 1085.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1081.75 | 1088.56 | 1085.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 1081.95 | 1088.56 | 1085.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1079.80 | 1086.81 | 1085.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 1079.40 | 1086.81 | 1085.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 1076.10 | 1083.55 | 1084.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 1072.15 | 1081.02 | 1082.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1075.25 | 1074.56 | 1078.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 1075.25 | 1074.56 | 1078.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1067.30 | 1073.33 | 1077.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:00:00 | 1064.65 | 1071.60 | 1076.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 1064.60 | 1068.94 | 1073.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 1063.90 | 1067.09 | 1072.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 1065.35 | 1056.75 | 1062.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1055.70 | 1056.54 | 1061.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 1055.20 | 1055.32 | 1060.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:00:00 | 1055.30 | 1048.56 | 1049.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 15:15:00 | 1055.00 | 1050.23 | 1050.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 15:15:00 | 1055.00 | 1051.18 | 1050.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 1055.00 | 1051.18 | 1050.68 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1043.00 | 1049.55 | 1049.98 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 1062.50 | 1048.74 | 1048.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 1072.00 | 1053.39 | 1050.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1007.70 | 1063.15 | 1062.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1007.70 | 1063.15 | 1062.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1007.70 | 1063.15 | 1062.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 1011.90 | 1063.15 | 1062.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1002.70 | 1051.06 | 1056.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 1001.10 | 1041.07 | 1051.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 1010.00 | 1009.09 | 1015.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 11:00:00 | 1010.00 | 1009.09 | 1015.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1014.30 | 1010.57 | 1014.93 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 1028.10 | 1018.38 | 1017.55 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1013.30 | 1019.14 | 1019.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1013.10 | 1016.17 | 1017.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1012.70 | 1010.16 | 1013.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1012.70 | 1010.16 | 1013.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1012.70 | 1010.16 | 1013.31 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1029.40 | 1017.66 | 1016.13 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 1009.20 | 1015.85 | 1016.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1003.00 | 1012.26 | 1014.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 999.70 | 993.12 | 998.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 999.70 | 993.12 | 998.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 999.70 | 993.12 | 998.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 999.70 | 993.12 | 998.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1002.60 | 995.02 | 998.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1002.60 | 995.02 | 998.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1011.00 | 998.21 | 999.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 1011.00 | 998.21 | 999.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1011.90 | 1002.71 | 1001.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 1033.60 | 1011.35 | 1006.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 1033.60 | 1034.72 | 1026.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:00:00 | 1033.60 | 1034.72 | 1026.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1030.20 | 1034.66 | 1028.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1030.20 | 1034.66 | 1028.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1030.00 | 1033.73 | 1028.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 1030.70 | 1033.73 | 1028.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1024.10 | 1031.80 | 1028.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1024.10 | 1031.80 | 1028.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1024.00 | 1030.24 | 1027.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 1022.20 | 1030.24 | 1027.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 1019.80 | 1025.84 | 1026.26 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 1032.50 | 1026.83 | 1026.62 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 1020.00 | 1025.47 | 1026.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1018.00 | 1023.02 | 1024.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1025.90 | 1023.59 | 1024.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 1025.90 | 1023.59 | 1024.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1025.90 | 1023.59 | 1024.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1025.90 | 1023.59 | 1024.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1025.30 | 1023.94 | 1024.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1023.10 | 1023.94 | 1024.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1021.90 | 1023.53 | 1024.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 1017.40 | 1023.53 | 1024.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:30:00 | 1016.20 | 1021.65 | 1023.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1026.40 | 1023.75 | 1023.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1026.40 | 1023.75 | 1023.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 1029.50 | 1025.49 | 1024.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1025.80 | 1041.30 | 1036.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1025.80 | 1041.30 | 1036.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1025.80 | 1041.30 | 1036.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1025.80 | 1041.30 | 1036.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1030.40 | 1039.12 | 1035.69 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1026.10 | 1033.89 | 1033.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1017.20 | 1028.51 | 1031.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 1014.20 | 1013.44 | 1018.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:15:00 | 1016.60 | 1013.44 | 1018.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1015.70 | 1013.90 | 1018.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 1019.10 | 1013.90 | 1018.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1013.30 | 1013.78 | 1017.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 1010.70 | 1013.78 | 1017.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1016.40 | 1010.38 | 1014.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 1016.60 | 1010.38 | 1014.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1018.40 | 1011.98 | 1014.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 1018.40 | 1011.98 | 1014.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1017.90 | 1014.45 | 1015.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 1017.90 | 1014.45 | 1015.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1017.90 | 1015.14 | 1015.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1012.70 | 1015.27 | 1015.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 1005.50 | 1003.98 | 1003.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1005.50 | 1003.98 | 1003.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1008.90 | 1004.96 | 1004.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 1008.10 | 1009.25 | 1007.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 1008.10 | 1009.25 | 1007.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1008.10 | 1009.25 | 1007.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 1007.70 | 1009.25 | 1007.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1011.20 | 1009.64 | 1007.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:45:00 | 1012.90 | 1010.51 | 1008.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 15:00:00 | 1012.50 | 1010.91 | 1008.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1022.90 | 1011.13 | 1009.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 1002.70 | 1014.43 | 1013.81 | SL hit (close<static) qty=1.00 sl=1007.40 alert=retest2 |

### Cycle 185 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1005.50 | 1012.64 | 1013.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 1000.30 | 1008.90 | 1011.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 998.70 | 997.15 | 1002.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 998.70 | 997.15 | 1002.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 986.15 | 979.90 | 984.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 987.45 | 979.90 | 984.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 984.65 | 980.85 | 984.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 988.90 | 980.85 | 984.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 990.00 | 984.57 | 985.42 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 990.55 | 986.76 | 986.33 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 982.80 | 985.82 | 986.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 12:15:00 | 978.80 | 983.89 | 985.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 983.00 | 981.44 | 983.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 983.00 | 981.44 | 983.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 983.00 | 981.44 | 983.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:45:00 | 982.95 | 981.44 | 983.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 978.00 | 980.75 | 982.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:45:00 | 977.10 | 980.26 | 982.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 973.95 | 978.67 | 981.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 976.95 | 973.51 | 973.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 956.60 | 952.06 | 951.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 956.60 | 952.06 | 951.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 10:15:00 | 966.80 | 955.01 | 953.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 945.05 | 960.55 | 957.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 945.05 | 960.55 | 957.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 945.05 | 960.55 | 957.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 943.80 | 960.55 | 957.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 944.00 | 957.24 | 956.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 944.00 | 957.24 | 956.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 942.80 | 954.35 | 955.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 933.30 | 947.25 | 951.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 940.00 | 939.31 | 945.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 940.00 | 939.31 | 945.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 943.10 | 940.07 | 945.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:30:00 | 944.75 | 940.07 | 945.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 942.15 | 939.23 | 943.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 935.20 | 939.60 | 941.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 935.60 | 929.07 | 929.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 935.60 | 929.07 | 929.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 15:15:00 | 937.25 | 933.29 | 931.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 929.10 | 932.45 | 931.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 929.10 | 932.45 | 931.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 929.10 | 932.45 | 931.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 928.80 | 932.45 | 931.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 927.90 | 931.54 | 930.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 927.90 | 931.54 | 930.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 936.55 | 933.48 | 932.03 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 924.70 | 931.07 | 931.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 909.80 | 926.81 | 929.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 908.00 | 905.27 | 912.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 959.85 | 905.27 | 912.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 946.65 | 913.54 | 915.85 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 953.10 | 921.45 | 919.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 961.20 | 934.92 | 926.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 957.30 | 961.53 | 951.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 957.30 | 961.53 | 951.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 964.50 | 976.80 | 972.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 964.50 | 976.80 | 972.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 964.70 | 974.38 | 972.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 970.10 | 971.76 | 971.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:00:00 | 968.85 | 971.18 | 971.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 14:15:00 | 965.50 | 970.04 | 970.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 965.50 | 970.04 | 970.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 963.95 | 968.04 | 969.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 970.50 | 968.53 | 969.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 970.50 | 968.53 | 969.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 970.50 | 968.53 | 969.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 972.00 | 968.53 | 969.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 967.90 | 968.41 | 969.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 970.10 | 968.41 | 969.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 969.65 | 968.65 | 969.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 969.65 | 968.65 | 969.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 970.75 | 969.07 | 969.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 14:30:00 | 967.10 | 968.81 | 969.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 967.05 | 968.81 | 969.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 977.95 | 970.36 | 969.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 977.95 | 970.36 | 969.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 10:15:00 | 987.80 | 973.84 | 971.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 1015.10 | 1016.13 | 1003.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 1015.10 | 1016.13 | 1003.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1012.70 | 1013.02 | 1005.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1017.85 | 1011.03 | 1008.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 1016.05 | 1016.00 | 1015.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1016.40 | 1028.78 | 1026.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 1020.20 | 1025.13 | 1025.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 1020.20 | 1025.13 | 1025.32 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1028.05 | 1025.39 | 1025.25 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 1024.20 | 1025.15 | 1025.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 1014.80 | 1023.08 | 1024.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 1022.30 | 1021.75 | 1023.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 1022.30 | 1021.75 | 1023.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1022.35 | 1021.65 | 1023.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 1019.65 | 1021.65 | 1023.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1023.50 | 1022.02 | 1023.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 1023.50 | 1022.02 | 1023.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1023.60 | 1022.34 | 1023.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1015.10 | 1020.11 | 1022.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 964.35 | 975.83 | 990.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 957.15 | 956.16 | 971.23 | SL hit (close>ema200) qty=0.50 sl=956.16 alert=retest2 |

### Cycle 198 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 880.35 | 871.19 | 871.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 882.95 | 873.54 | 872.15 | Break + close above crossover candle high |

### Cycle 199 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 847.50 | 868.33 | 869.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 844.10 | 863.49 | 867.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 815.05 | 812.31 | 826.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:00:00 | 815.05 | 812.31 | 826.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 822.50 | 814.39 | 824.08 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 849.90 | 832.22 | 829.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 15:15:00 | 852.00 | 836.18 | 831.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 854.15 | 868.88 | 855.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 854.15 | 868.88 | 855.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 854.15 | 868.88 | 855.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 854.15 | 868.88 | 855.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 855.65 | 866.23 | 855.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 850.70 | 866.23 | 855.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 850.20 | 863.03 | 855.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 850.20 | 863.03 | 855.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 848.60 | 860.14 | 854.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 847.70 | 860.14 | 854.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 827.00 | 847.73 | 850.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 818.40 | 841.86 | 847.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 832.80 | 820.87 | 832.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 832.80 | 820.87 | 832.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 832.80 | 820.87 | 832.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 825.70 | 827.88 | 832.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 825.10 | 822.25 | 823.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 843.00 | 827.78 | 826.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 843.00 | 827.78 | 826.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 847.30 | 831.68 | 828.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 893.90 | 903.42 | 888.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 893.90 | 903.42 | 888.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 902.40 | 914.88 | 905.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 919.65 | 903.95 | 903.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 908.20 | 910.91 | 910.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 907.10 | 910.15 | 910.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 10:15:00 | 907.10 | 910.15 | 910.51 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 916.20 | 910.67 | 910.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 14:15:00 | 919.30 | 914.03 | 912.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 932.20 | 933.22 | 925.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 10:00:00 | 932.20 | 933.22 | 925.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 919.85 | 932.24 | 929.18 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 911.15 | 924.89 | 926.18 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 926.90 | 921.34 | 921.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 930.40 | 924.75 | 923.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 935.45 | 940.08 | 934.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 935.45 | 940.08 | 934.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 935.45 | 940.08 | 934.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 935.45 | 940.08 | 934.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 939.70 | 940.00 | 934.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 946.00 | 940.00 | 934.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 941.55 | 946.68 | 942.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 953.50 | 962.53 | 963.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 953.50 | 962.53 | 963.41 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-31 14:15:00 | 697.90 | 2023-06-08 11:15:00 | 701.64 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2023-06-01 09:30:00 | 698.86 | 2023-06-08 11:15:00 | 701.64 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2023-06-20 13:15:00 | 723.59 | 2023-06-21 10:15:00 | 718.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-06-20 14:00:00 | 723.33 | 2023-06-21 10:15:00 | 718.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-06-20 14:30:00 | 723.11 | 2023-06-21 10:15:00 | 718.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-06-27 10:30:00 | 696.67 | 2023-06-28 09:15:00 | 710.69 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2023-06-27 11:30:00 | 696.68 | 2023-06-28 09:15:00 | 710.69 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2023-07-11 11:00:00 | 756.56 | 2023-07-17 14:15:00 | 751.37 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2023-07-21 12:30:00 | 756.66 | 2023-07-26 14:15:00 | 743.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2023-07-25 10:45:00 | 757.38 | 2023-07-26 14:15:00 | 743.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2023-08-04 10:30:00 | 711.00 | 2023-08-08 15:15:00 | 720.30 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-08-04 13:30:00 | 710.75 | 2023-08-08 15:15:00 | 720.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-08-07 09:45:00 | 710.45 | 2023-08-08 15:15:00 | 720.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-08-08 09:45:00 | 711.05 | 2023-08-08 15:15:00 | 720.30 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2023-08-17 12:45:00 | 693.80 | 2023-08-21 10:15:00 | 702.19 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2023-08-17 14:15:00 | 693.85 | 2023-08-21 10:15:00 | 702.19 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-08-17 15:00:00 | 692.26 | 2023-08-21 10:15:00 | 702.19 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-08-23 11:15:00 | 705.40 | 2023-08-31 11:15:00 | 719.60 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2023-09-05 09:15:00 | 730.71 | 2023-09-12 10:15:00 | 734.40 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2023-09-18 11:45:00 | 753.60 | 2023-09-21 11:15:00 | 747.70 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-09-18 12:30:00 | 753.79 | 2023-09-21 11:15:00 | 747.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-09-18 14:00:00 | 754.80 | 2023-09-21 11:15:00 | 747.70 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-09-21 09:15:00 | 755.52 | 2023-09-21 11:15:00 | 747.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-10-05 09:15:00 | 786.83 | 2023-10-05 14:15:00 | 784.67 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2023-10-05 10:45:00 | 788.70 | 2023-10-05 14:15:00 | 784.67 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-10-05 14:00:00 | 787.44 | 2023-10-05 14:15:00 | 784.67 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-10-06 09:15:00 | 795.60 | 2023-10-12 13:15:00 | 803.34 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2023-10-16 09:15:00 | 802.09 | 2023-10-16 09:15:00 | 806.30 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-10-16 10:45:00 | 804.00 | 2023-10-17 09:15:00 | 811.25 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-10-16 15:15:00 | 802.32 | 2023-10-17 09:15:00 | 811.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2023-10-23 13:00:00 | 777.50 | 2023-10-30 09:15:00 | 738.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 15:00:00 | 778.90 | 2023-10-30 09:15:00 | 739.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-25 10:00:00 | 778.98 | 2023-10-30 09:15:00 | 740.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 13:00:00 | 777.50 | 2023-10-30 10:15:00 | 749.13 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2023-10-23 15:00:00 | 778.90 | 2023-10-30 10:15:00 | 749.13 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2023-10-25 10:00:00 | 778.98 | 2023-10-30 10:15:00 | 749.13 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2023-11-15 13:30:00 | 738.80 | 2023-11-16 09:15:00 | 701.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-15 13:30:00 | 738.80 | 2023-11-16 12:15:00 | 730.45 | STOP_HIT | 0.50 | 1.13% |
| BUY | retest2 | 2023-11-30 15:15:00 | 713.00 | 2023-12-08 11:15:00 | 729.28 | STOP_HIT | 1.00 | 2.28% |
| SELL | retest2 | 2023-12-28 10:45:00 | 724.24 | 2023-12-29 12:15:00 | 730.70 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-12-28 15:15:00 | 722.40 | 2023-12-29 12:15:00 | 730.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-01-02 09:30:00 | 733.56 | 2024-01-10 15:15:00 | 767.50 | STOP_HIT | 1.00 | 4.63% |
| SELL | retest2 | 2024-01-20 12:30:00 | 731.69 | 2024-01-29 13:15:00 | 719.44 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2024-01-20 13:30:00 | 732.00 | 2024-01-29 13:15:00 | 719.44 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2024-02-02 12:15:00 | 684.80 | 2024-02-14 14:15:00 | 664.59 | STOP_HIT | 1.00 | 2.95% |
| SELL | retest2 | 2024-02-02 12:45:00 | 683.58 | 2024-02-14 14:15:00 | 664.59 | STOP_HIT | 1.00 | 2.78% |
| SELL | retest2 | 2024-02-05 09:15:00 | 683.51 | 2024-02-14 14:15:00 | 664.59 | STOP_HIT | 1.00 | 2.77% |
| BUY | retest2 | 2024-02-19 09:15:00 | 670.20 | 2024-02-22 09:15:00 | 661.60 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-02-23 12:00:00 | 664.68 | 2024-02-23 14:15:00 | 669.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-03-12 13:15:00 | 647.64 | 2024-03-13 11:15:00 | 642.43 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-03-13 10:00:00 | 650.80 | 2024-03-13 11:15:00 | 642.43 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-03-13 13:00:00 | 649.43 | 2024-03-14 12:15:00 | 640.15 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-04-02 15:15:00 | 722.30 | 2024-04-05 11:15:00 | 716.76 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-04-25 11:30:00 | 727.90 | 2024-04-26 09:15:00 | 680.84 | STOP_HIT | 1.00 | -6.47% |
| BUY | retest2 | 2024-04-25 12:45:00 | 725.42 | 2024-04-26 09:15:00 | 680.84 | STOP_HIT | 1.00 | -6.15% |
| SELL | retest2 | 2024-05-02 10:15:00 | 690.53 | 2024-05-03 09:15:00 | 725.20 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2024-05-02 10:45:00 | 689.91 | 2024-05-03 09:15:00 | 725.20 | STOP_HIT | 1.00 | -5.12% |
| SELL | retest2 | 2024-05-07 14:15:00 | 684.89 | 2024-05-16 15:15:00 | 673.10 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2024-05-08 09:15:00 | 684.31 | 2024-05-16 15:15:00 | 673.10 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2024-05-08 10:15:00 | 684.71 | 2024-05-16 15:15:00 | 673.10 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2024-05-21 10:15:00 | 678.20 | 2024-05-29 13:15:00 | 683.78 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2024-05-23 11:15:00 | 677.55 | 2024-05-29 13:15:00 | 683.78 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2024-05-23 12:30:00 | 681.00 | 2024-05-29 13:15:00 | 683.78 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2024-05-31 12:30:00 | 671.25 | 2024-06-03 09:15:00 | 690.61 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-05-31 13:15:00 | 672.17 | 2024-06-03 09:15:00 | 690.61 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-05-31 14:45:00 | 671.98 | 2024-06-03 09:15:00 | 690.61 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-06-11 12:30:00 | 713.30 | 2024-06-19 12:15:00 | 725.38 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2024-06-21 09:30:00 | 718.31 | 2024-06-26 13:15:00 | 717.78 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-07-02 13:45:00 | 721.69 | 2024-07-04 13:15:00 | 713.30 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-07-03 09:15:00 | 723.18 | 2024-07-04 13:15:00 | 713.30 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-07-03 10:45:00 | 721.61 | 2024-07-04 13:15:00 | 713.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-07-04 11:15:00 | 720.43 | 2024-07-04 13:15:00 | 713.30 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-10 14:00:00 | 703.51 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-07-10 14:45:00 | 704.40 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-07-10 15:15:00 | 704.50 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-07-15 10:15:00 | 704.30 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-07-15 11:15:00 | 703.28 | 2024-07-15 11:15:00 | 705.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-08-01 09:15:00 | 683.42 | 2024-08-01 09:15:00 | 678.87 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-08-01 11:00:00 | 682.60 | 2024-08-01 12:15:00 | 677.98 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-08-06 10:15:00 | 660.21 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2024-08-06 13:30:00 | 659.47 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-08-07 10:00:00 | 662.00 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-08-07 12:30:00 | 662.20 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-08-07 15:15:00 | 663.40 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2024-08-08 10:00:00 | 662.06 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-08-08 10:30:00 | 662.00 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-08-08 13:00:00 | 662.74 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2024-08-09 10:15:00 | 659.62 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-08-09 12:00:00 | 659.95 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-08-12 09:15:00 | 659.60 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-08-12 14:45:00 | 660.01 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-08-13 12:45:00 | 656.20 | 2024-08-16 14:15:00 | 659.45 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-08-23 11:15:00 | 673.80 | 2024-09-02 10:15:00 | 741.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 09:15:00 | 677.80 | 2024-09-03 09:15:00 | 745.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-09 12:30:00 | 734.34 | 2024-09-10 09:15:00 | 724.18 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-09-09 14:45:00 | 734.10 | 2024-09-10 09:15:00 | 724.18 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-09-17 14:45:00 | 735.30 | 2024-09-18 09:15:00 | 752.70 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-09-20 15:15:00 | 759.80 | 2024-09-24 09:15:00 | 752.67 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-09-23 13:45:00 | 760.00 | 2024-09-24 09:15:00 | 752.67 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-09-23 14:15:00 | 760.10 | 2024-09-24 09:15:00 | 752.67 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-10-21 10:45:00 | 678.75 | 2024-10-23 09:15:00 | 700.77 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-10-22 09:30:00 | 679.42 | 2024-10-23 09:15:00 | 700.77 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-10-22 11:00:00 | 679.44 | 2024-10-23 09:15:00 | 700.77 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-11-04 10:15:00 | 680.50 | 2024-11-05 13:15:00 | 697.52 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-11-05 09:30:00 | 681.32 | 2024-11-05 13:15:00 | 697.52 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-11-05 10:30:00 | 678.91 | 2024-11-05 13:15:00 | 697.52 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-11-18 11:00:00 | 657.94 | 2024-11-22 12:15:00 | 662.47 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-11-18 14:00:00 | 657.71 | 2024-11-22 12:15:00 | 662.47 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-11-19 13:30:00 | 657.75 | 2024-11-22 12:15:00 | 662.47 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-11-19 14:00:00 | 657.47 | 2024-11-22 13:15:00 | 665.57 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-11-21 11:30:00 | 648.50 | 2024-11-22 13:15:00 | 665.57 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-11-21 12:00:00 | 648.16 | 2024-11-22 13:15:00 | 665.57 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-11-21 14:00:00 | 648.50 | 2024-11-22 13:15:00 | 665.57 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-11-27 09:45:00 | 670.87 | 2024-11-28 11:15:00 | 662.58 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-11-27 10:30:00 | 669.49 | 2024-11-28 11:15:00 | 662.58 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-11-27 12:15:00 | 669.21 | 2024-11-28 11:15:00 | 662.58 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-11-28 10:00:00 | 671.40 | 2024-11-28 11:15:00 | 662.58 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-12-13 12:15:00 | 714.50 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-13 12:45:00 | 714.97 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-12-13 13:15:00 | 716.10 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-12-17 10:45:00 | 718.40 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-18 09:30:00 | 717.41 | 2024-12-18 10:15:00 | 710.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-12-23 13:00:00 | 689.16 | 2024-12-27 10:15:00 | 694.81 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest1 | 2025-01-06 09:15:00 | 755.44 | 2025-01-07 15:15:00 | 734.00 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2025-01-06 12:45:00 | 743.50 | 2025-01-07 15:15:00 | 734.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest1 | 2025-01-06 13:15:00 | 742.71 | 2025-01-07 15:15:00 | 734.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-01-07 09:15:00 | 740.39 | 2025-01-08 09:15:00 | 729.20 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-01-07 10:15:00 | 739.04 | 2025-01-08 09:15:00 | 729.20 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-01-10 12:45:00 | 728.70 | 2025-01-14 11:15:00 | 736.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-01-10 14:00:00 | 727.37 | 2025-01-14 11:15:00 | 736.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-01-24 11:30:00 | 747.70 | 2025-01-27 12:15:00 | 735.04 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-01-24 13:00:00 | 747.91 | 2025-01-27 12:15:00 | 735.04 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-02-07 13:30:00 | 852.16 | 2025-02-10 09:15:00 | 835.65 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-02-24 09:15:00 | 837.00 | 2025-02-25 09:15:00 | 846.61 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-02-24 12:00:00 | 836.73 | 2025-02-25 09:15:00 | 846.61 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-03-17 09:15:00 | 858.32 | 2025-03-26 14:15:00 | 887.00 | STOP_HIT | 1.00 | 3.34% |
| SELL | retest2 | 2025-04-04 13:45:00 | 871.34 | 2025-04-04 15:15:00 | 873.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-04-04 14:45:00 | 871.50 | 2025-04-04 15:15:00 | 873.20 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-04-09 12:15:00 | 870.25 | 2025-04-24 09:15:00 | 957.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-09 14:30:00 | 869.19 | 2025-04-24 09:15:00 | 956.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-27 11:15:00 | 928.00 | 2025-05-27 13:15:00 | 916.80 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-05-30 09:15:00 | 918.90 | 2025-06-06 10:15:00 | 932.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-06-17 11:00:00 | 923.00 | 2025-06-23 15:15:00 | 913.00 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2025-06-17 11:45:00 | 921.00 | 2025-06-23 15:15:00 | 913.00 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-06-18 10:45:00 | 923.50 | 2025-06-23 15:15:00 | 913.00 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2025-06-18 12:15:00 | 924.50 | 2025-06-23 15:15:00 | 913.00 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2025-06-25 09:15:00 | 922.00 | 2025-07-01 10:15:00 | 933.50 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-06-25 12:45:00 | 921.50 | 2025-07-01 10:15:00 | 933.50 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-07-04 11:30:00 | 927.00 | 2025-07-07 13:15:00 | 924.80 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-07-16 09:15:00 | 921.25 | 2025-07-18 09:15:00 | 935.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-07-24 13:30:00 | 959.70 | 2025-07-25 09:15:00 | 918.10 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-07-24 14:00:00 | 960.50 | 2025-07-25 09:15:00 | 918.10 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-07-24 14:30:00 | 960.35 | 2025-07-25 09:15:00 | 918.10 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-08-01 14:15:00 | 880.35 | 2025-08-05 09:15:00 | 888.65 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-08-01 15:00:00 | 873.90 | 2025-08-05 09:15:00 | 888.65 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-08-04 09:45:00 | 879.10 | 2025-08-05 09:15:00 | 888.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-08-04 10:45:00 | 880.95 | 2025-08-05 09:15:00 | 888.65 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-08-08 15:00:00 | 876.40 | 2025-08-18 09:15:00 | 915.20 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-08-11 11:30:00 | 876.85 | 2025-08-18 09:15:00 | 915.20 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-08-12 09:15:00 | 873.55 | 2025-08-18 09:15:00 | 915.20 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-08-29 13:45:00 | 878.80 | 2025-09-01 09:15:00 | 888.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-08-29 15:00:00 | 876.50 | 2025-09-01 09:15:00 | 888.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-03 11:45:00 | 891.70 | 2025-09-12 09:15:00 | 980.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-03 13:00:00 | 891.60 | 2025-09-12 09:15:00 | 980.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 989.10 | 2025-10-06 09:15:00 | 1013.95 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-10-10 11:00:00 | 1026.35 | 2025-10-14 12:15:00 | 1015.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-10 12:30:00 | 1026.80 | 2025-10-14 12:15:00 | 1015.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-13 09:15:00 | 1026.20 | 2025-10-14 12:15:00 | 1015.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-13 10:45:00 | 1026.65 | 2025-10-14 12:15:00 | 1015.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-29 11:00:00 | 1064.65 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-10-29 13:15:00 | 1064.60 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-10-29 14:30:00 | 1063.90 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-10-31 10:00:00 | 1065.35 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-10-31 11:45:00 | 1055.20 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-11-04 14:00:00 | 1055.30 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-11-04 15:15:00 | 1055.00 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-03 10:15:00 | 1017.40 | 2025-12-04 12:15:00 | 1026.40 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-03 11:30:00 | 1016.20 | 2025-12-04 12:15:00 | 1026.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1012.70 | 2025-12-19 13:15:00 | 1005.50 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-12-23 13:45:00 | 1012.90 | 2025-12-26 09:15:00 | 1002.70 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-23 15:00:00 | 1012.50 | 2025-12-26 09:15:00 | 1002.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1022.90 | 2025-12-26 09:15:00 | 1002.70 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-01-06 11:45:00 | 977.10 | 2026-01-19 09:15:00 | 956.60 | STOP_HIT | 1.00 | 2.10% |
| SELL | retest2 | 2026-01-06 14:00:00 | 973.95 | 2026-01-19 09:15:00 | 956.60 | STOP_HIT | 1.00 | 1.78% |
| SELL | retest2 | 2026-01-09 10:45:00 | 976.95 | 2026-01-19 09:15:00 | 956.60 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2026-01-23 11:30:00 | 935.20 | 2026-01-28 15:15:00 | 935.60 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-02-10 12:45:00 | 970.10 | 2026-02-10 14:15:00 | 965.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-02-10 14:00:00 | 968.85 | 2026-02-10 14:15:00 | 965.50 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-02-11 14:30:00 | 967.10 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-11 15:15:00 | 967.05 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-02-18 09:15:00 | 1017.85 | 2026-02-24 12:15:00 | 1020.20 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2026-02-20 09:30:00 | 1016.05 | 2026-02-24 12:15:00 | 1020.20 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2026-02-24 09:45:00 | 1016.40 | 2026-02-24 12:15:00 | 1020.20 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1015.10 | 2026-03-04 09:15:00 | 964.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1015.10 | 2026-03-05 09:15:00 | 957.15 | STOP_HIT | 0.50 | 5.71% |
| SELL | retest2 | 2026-04-01 13:30:00 | 825.70 | 2026-04-06 12:15:00 | 843.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-04-06 11:00:00 | 825.10 | 2026-04-06 12:15:00 | 843.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-04-15 09:15:00 | 919.65 | 2026-04-17 10:15:00 | 907.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-04-17 09:45:00 | 908.20 | 2026-04-17 10:15:00 | 907.10 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-05-04 09:15:00 | 946.00 | 2026-05-08 12:15:00 | 953.50 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2026-05-05 10:15:00 | 941.55 | 2026-05-08 12:15:00 | 953.50 | STOP_HIT | 1.00 | 1.27% |
