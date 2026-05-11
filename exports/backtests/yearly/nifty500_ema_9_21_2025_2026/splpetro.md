# Supreme Petrochem Ltd. (SPLPETRO)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 738.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 57 |
| ALERT1 | 44 |
| ALERT2 | 44 |
| ALERT2_SKIP | 20 |
| ALERT3 | 126 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 67 |
| PARTIAL | 2 |
| TARGET_HIT | 6 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 60
- **Target hits / Stop hits / Partials:** 5 / 62 / 2
- **Avg / median % per leg:** -0.36% / -1.38%
- **Sum % (uncompounded):** -25.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 5 | 13.5% | 4 | 33 | 0 | -0.23% | -8.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 37 | 5 | 13.5% | 4 | 33 | 0 | -0.23% | -8.4% |
| SELL (all) | 32 | 4 | 12.5% | 1 | 29 | 2 | -0.53% | -16.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 4 | 12.5% | 1 | 29 | 2 | -0.53% | -16.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 69 | 9 | 13.0% | 5 | 62 | 2 | -0.36% | -25.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 692.55 | 696.39 | 696.58 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 701.50 | 696.98 | 696.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 715.55 | 702.69 | 699.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 706.60 | 713.30 | 710.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 706.60 | 713.30 | 710.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 706.60 | 713.30 | 710.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 706.60 | 713.30 | 710.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 709.25 | 712.49 | 709.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:00:00 | 713.45 | 710.32 | 709.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 710.85 | 709.91 | 709.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:45:00 | 711.00 | 710.11 | 709.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 713.60 | 710.81 | 709.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 12:15:00 | 701.05 | 708.85 | 709.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 12:15:00 | 701.05 | 708.85 | 709.13 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 719.05 | 709.48 | 709.15 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 711.00 | 712.11 | 712.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 12:15:00 | 705.20 | 709.60 | 710.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 12:15:00 | 715.75 | 708.99 | 709.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 12:15:00 | 715.75 | 708.99 | 709.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 715.75 | 708.99 | 709.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 715.10 | 708.99 | 709.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 713.75 | 709.94 | 709.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:45:00 | 714.60 | 709.94 | 709.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 715.60 | 711.07 | 710.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 721.40 | 713.77 | 711.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 724.35 | 724.60 | 719.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:45:00 | 723.10 | 724.60 | 719.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 715.75 | 723.39 | 720.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:00:00 | 715.75 | 723.39 | 720.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 715.20 | 721.76 | 720.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:15:00 | 712.65 | 721.76 | 720.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 721.40 | 720.24 | 719.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:15:00 | 724.00 | 720.24 | 719.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 718.30 | 719.97 | 719.64 | SL hit (close<static) qty=1.00 sl=718.85 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 13:15:00 | 717.05 | 719.39 | 719.41 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 724.95 | 719.42 | 719.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 730.55 | 723.19 | 721.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 11:15:00 | 728.20 | 729.75 | 726.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 12:00:00 | 728.20 | 729.75 | 726.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 726.45 | 728.90 | 726.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 726.45 | 728.90 | 726.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 727.10 | 728.54 | 726.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:15:00 | 733.50 | 728.54 | 726.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 733.50 | 729.53 | 727.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 740.65 | 732.17 | 728.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 768.50 | 756.72 | 750.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 775.00 | 760.47 | 757.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 776.90 | 784.93 | 785.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 776.90 | 784.93 | 785.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 775.75 | 783.09 | 785.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 796.25 | 784.57 | 785.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 796.25 | 784.57 | 785.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 796.25 | 784.57 | 785.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 796.25 | 784.57 | 785.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 795.80 | 786.82 | 786.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 802.75 | 791.67 | 788.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 871.45 | 872.16 | 851.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:00:00 | 871.45 | 872.16 | 851.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 870.85 | 877.45 | 874.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 866.15 | 877.45 | 874.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 864.70 | 874.90 | 873.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 864.70 | 874.90 | 873.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 863.50 | 872.62 | 872.56 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 863.65 | 870.83 | 871.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 859.55 | 866.51 | 869.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 15:15:00 | 843.75 | 841.94 | 850.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 847.30 | 843.01 | 850.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 847.30 | 843.01 | 850.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 847.30 | 843.01 | 850.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 812.85 | 807.68 | 811.29 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 825.00 | 814.83 | 813.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 827.55 | 817.38 | 815.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 813.90 | 817.90 | 815.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 813.90 | 817.90 | 815.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 813.90 | 817.90 | 815.74 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 810.40 | 814.84 | 815.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 809.05 | 813.68 | 814.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 808.25 | 808.12 | 811.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 808.25 | 808.12 | 811.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 810.55 | 808.92 | 810.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 809.45 | 808.92 | 810.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 814.20 | 809.98 | 811.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:30:00 | 815.80 | 809.98 | 811.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 816.75 | 811.33 | 811.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 816.90 | 811.33 | 811.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 817.00 | 812.47 | 812.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 821.80 | 814.97 | 813.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 815.35 | 816.29 | 814.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 815.35 | 816.29 | 814.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 815.35 | 816.29 | 814.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 815.35 | 816.29 | 814.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 816.30 | 816.29 | 814.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 816.30 | 816.29 | 814.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 816.25 | 816.28 | 814.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 814.00 | 816.28 | 814.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 816.60 | 816.30 | 815.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 815.00 | 816.30 | 815.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 820.00 | 817.04 | 815.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:00:00 | 820.05 | 817.64 | 815.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 821.95 | 818.81 | 816.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 821.25 | 820.67 | 819.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:15:00 | 821.50 | 820.67 | 819.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 819.65 | 820.40 | 819.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 826.15 | 819.98 | 819.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 815.35 | 818.95 | 818.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 815.35 | 818.95 | 818.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 806.70 | 814.06 | 816.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 14:15:00 | 810.05 | 808.63 | 811.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 15:00:00 | 810.05 | 808.63 | 811.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 817.75 | 810.67 | 811.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 824.95 | 810.67 | 811.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 819.40 | 812.42 | 812.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 819.10 | 812.42 | 812.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 816.60 | 813.25 | 813.00 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 809.55 | 813.47 | 813.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 806.55 | 810.31 | 811.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 805.00 | 803.87 | 807.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 11:00:00 | 805.00 | 803.87 | 807.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 797.55 | 802.10 | 804.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 792.15 | 798.99 | 802.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:30:00 | 790.00 | 792.92 | 796.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 788.00 | 794.12 | 796.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 808.05 | 795.13 | 796.20 | SL hit (close>static) qty=1.00 sl=806.95 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 817.00 | 799.51 | 798.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 11:15:00 | 821.05 | 803.82 | 800.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 12:15:00 | 817.35 | 823.67 | 814.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 817.35 | 823.67 | 814.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 817.35 | 823.67 | 814.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:45:00 | 814.55 | 823.67 | 814.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 807.60 | 820.45 | 814.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 809.15 | 820.45 | 814.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 815.00 | 819.36 | 814.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:45:00 | 822.55 | 819.33 | 815.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:30:00 | 822.10 | 819.27 | 815.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 805.05 | 815.11 | 815.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 805.05 | 815.11 | 815.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 12:15:00 | 802.75 | 811.25 | 813.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 752.75 | 751.55 | 764.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 12:00:00 | 752.75 | 751.55 | 764.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 741.00 | 737.18 | 742.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 741.00 | 737.18 | 742.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 752.05 | 740.16 | 743.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 756.00 | 740.16 | 743.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 758.00 | 743.73 | 744.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 747.30 | 743.73 | 744.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 751.00 | 738.58 | 740.63 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 775.00 | 748.10 | 744.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 800.80 | 758.64 | 749.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 828.30 | 829.09 | 812.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:30:00 | 828.25 | 829.09 | 812.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 822.30 | 827.12 | 820.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 819.15 | 827.12 | 820.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 814.55 | 824.61 | 820.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 814.65 | 824.61 | 820.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 811.85 | 822.06 | 819.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 811.85 | 822.06 | 819.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 797.00 | 815.42 | 816.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 788.00 | 807.21 | 812.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 765.25 | 759.89 | 769.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 765.25 | 759.89 | 769.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 762.65 | 762.77 | 768.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:45:00 | 762.90 | 762.77 | 768.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 765.00 | 763.22 | 767.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 769.30 | 763.22 | 767.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 771.90 | 764.95 | 768.31 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 776.50 | 770.09 | 769.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 778.10 | 771.69 | 770.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 776.10 | 779.96 | 776.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 776.10 | 779.96 | 776.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 776.10 | 779.96 | 776.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 774.05 | 779.96 | 776.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 772.70 | 778.51 | 776.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 773.35 | 778.51 | 776.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 778.10 | 778.42 | 776.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 779.05 | 778.42 | 776.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 773.40 | 775.24 | 775.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 773.40 | 775.24 | 775.33 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 780.35 | 776.26 | 775.79 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 775.95 | 780.38 | 780.86 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 784.15 | 780.91 | 780.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 786.05 | 782.61 | 781.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 780.65 | 782.51 | 781.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 780.65 | 782.51 | 781.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 780.65 | 782.51 | 781.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 780.65 | 782.51 | 781.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 780.40 | 782.09 | 781.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 14:45:00 | 791.20 | 783.82 | 782.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-26 09:15:00 | 870.32 | 860.98 | 857.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 840.55 | 872.68 | 876.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 831.05 | 864.35 | 872.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 09:15:00 | 835.65 | 827.77 | 839.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 835.65 | 827.77 | 839.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 835.65 | 827.77 | 839.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:45:00 | 840.40 | 827.77 | 839.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 841.50 | 830.45 | 838.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:45:00 | 842.65 | 830.45 | 838.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 845.15 | 833.39 | 838.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:30:00 | 843.85 | 833.39 | 838.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 846.35 | 839.42 | 840.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 842.65 | 839.42 | 840.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 840.40 | 839.62 | 840.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 837.90 | 839.62 | 840.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 09:15:00 | 796.00 | 810.70 | 820.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 793.35 | 793.24 | 802.06 | SL hit (close>ema200) qty=0.50 sl=793.24 alert=retest2 |

### Cycle 28 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 805.05 | 780.07 | 777.26 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 790.20 | 792.80 | 792.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 769.50 | 787.51 | 790.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 755.75 | 754.09 | 761.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 755.75 | 754.09 | 761.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 663.15 | 659.68 | 664.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 663.40 | 659.68 | 664.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 666.60 | 661.76 | 664.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 666.95 | 661.76 | 664.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 671.00 | 663.60 | 665.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 671.15 | 663.60 | 665.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 651.35 | 654.35 | 657.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:30:00 | 651.00 | 653.63 | 657.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 644.00 | 651.71 | 655.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 649.30 | 650.68 | 654.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:00:00 | 650.25 | 651.56 | 654.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 653.40 | 652.32 | 654.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:00:00 | 653.40 | 652.32 | 654.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 655.90 | 653.03 | 654.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 655.90 | 653.03 | 654.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 652.10 | 652.85 | 654.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 649.25 | 652.85 | 654.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 12:15:00 | 651.00 | 636.42 | 635.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 12:15:00 | 651.00 | 636.42 | 635.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 654.05 | 639.95 | 637.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 10:15:00 | 663.05 | 664.70 | 659.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 11:15:00 | 662.40 | 664.24 | 660.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 662.40 | 664.24 | 660.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:45:00 | 661.25 | 664.24 | 660.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 663.70 | 664.60 | 661.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:45:00 | 663.90 | 664.60 | 661.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 657.80 | 663.24 | 660.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 660.40 | 663.24 | 660.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 661.00 | 662.79 | 660.96 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 646.80 | 658.11 | 659.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 645.00 | 652.40 | 656.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 11:15:00 | 631.20 | 629.99 | 637.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 11:45:00 | 631.00 | 629.99 | 637.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 638.95 | 631.78 | 637.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 638.95 | 631.78 | 637.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 637.05 | 632.83 | 637.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 15:15:00 | 636.00 | 634.20 | 637.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 635.90 | 634.96 | 637.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 636.25 | 634.82 | 636.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 640.70 | 634.96 | 636.11 | SL hit (close>static) qty=1.00 sl=638.95 alert=retest2 |

### Cycle 32 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 646.10 | 632.76 | 630.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 679.45 | 642.10 | 635.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 658.00 | 663.49 | 652.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 15:00:00 | 658.00 | 663.49 | 652.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 658.30 | 661.12 | 653.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 660.05 | 661.12 | 653.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 660.00 | 660.71 | 653.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:45:00 | 663.90 | 661.17 | 654.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 660.20 | 662.20 | 657.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 658.15 | 661.39 | 657.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 658.15 | 661.39 | 657.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 656.40 | 660.39 | 657.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 657.05 | 660.39 | 657.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 656.90 | 659.69 | 657.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 657.65 | 659.69 | 657.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 657.00 | 659.15 | 657.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 656.25 | 659.15 | 657.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 657.30 | 658.78 | 657.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 656.00 | 658.78 | 657.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 655.40 | 658.11 | 657.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 655.40 | 658.11 | 657.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 650.00 | 656.49 | 656.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 15:15:00 | 650.00 | 656.49 | 656.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 649.85 | 652.32 | 654.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 13:15:00 | 641.65 | 640.61 | 644.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 14:00:00 | 641.65 | 640.61 | 644.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 641.05 | 640.68 | 643.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 637.60 | 640.68 | 643.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 648.30 | 634.85 | 635.58 | SL hit (close>static) qty=1.00 sl=644.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 646.20 | 637.12 | 636.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 649.15 | 643.43 | 640.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 643.25 | 645.65 | 642.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 14:00:00 | 643.25 | 645.65 | 642.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 644.30 | 645.38 | 642.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 644.30 | 645.38 | 642.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 640.50 | 644.41 | 642.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 656.20 | 644.41 | 642.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 639.65 | 644.81 | 645.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 639.65 | 644.81 | 645.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 635.55 | 640.09 | 641.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 640.35 | 639.13 | 640.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 640.35 | 639.13 | 640.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 640.35 | 639.13 | 640.56 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 644.55 | 641.51 | 641.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 649.80 | 643.76 | 642.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 645.10 | 645.21 | 643.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 09:15:00 | 644.60 | 645.21 | 643.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 644.40 | 645.05 | 643.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 645.65 | 645.05 | 643.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 644.25 | 644.89 | 643.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:30:00 | 648.00 | 645.13 | 644.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 648.45 | 646.07 | 644.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 647.70 | 646.42 | 645.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 642.55 | 645.07 | 644.92 | SL hit (close<static) qty=1.00 sl=643.60 alert=retest2 |

### Cycle 37 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 642.95 | 644.64 | 644.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 14:15:00 | 642.50 | 643.66 | 644.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 15:15:00 | 646.00 | 644.13 | 644.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 15:15:00 | 646.00 | 644.13 | 644.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 646.00 | 644.13 | 644.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 635.20 | 644.13 | 644.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 14:15:00 | 603.44 | 629.21 | 636.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 15:15:00 | 571.68 | 580.99 | 591.74 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 38 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 523.75 | 511.98 | 511.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 527.60 | 519.60 | 515.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 517.20 | 521.74 | 517.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 517.20 | 521.74 | 517.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 517.20 | 521.74 | 517.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 517.55 | 521.74 | 517.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 512.20 | 519.83 | 517.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 512.25 | 519.83 | 517.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 521.60 | 520.19 | 517.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:00:00 | 523.55 | 520.86 | 518.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 547.00 | 530.05 | 523.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-30 15:15:00 | 575.90 | 554.85 | 540.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 590.60 | 609.09 | 610.34 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 630.90 | 608.16 | 607.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 635.00 | 613.52 | 609.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 642.30 | 644.40 | 638.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 642.30 | 644.40 | 638.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 642.30 | 644.40 | 638.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 632.70 | 644.40 | 638.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 627.50 | 640.34 | 639.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 628.20 | 640.34 | 639.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 634.10 | 639.10 | 638.66 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 629.55 | 637.19 | 637.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 620.75 | 629.89 | 633.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 625.10 | 623.71 | 628.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 625.10 | 623.71 | 628.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 633.05 | 625.58 | 628.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:45:00 | 635.85 | 625.58 | 628.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 632.20 | 626.90 | 628.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:15:00 | 630.70 | 626.90 | 628.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:00:00 | 630.45 | 628.57 | 629.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 630.40 | 629.36 | 629.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 632.90 | 630.06 | 629.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 632.90 | 630.06 | 629.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 638.85 | 631.82 | 630.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 633.10 | 636.72 | 634.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 633.10 | 636.72 | 634.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 633.10 | 636.72 | 634.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 633.10 | 636.72 | 634.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 658.50 | 641.08 | 636.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:30:00 | 632.65 | 641.08 | 636.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 646.95 | 648.29 | 642.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 672.00 | 654.32 | 646.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 658.00 | 657.41 | 651.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 12:30:00 | 654.60 | 656.02 | 652.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 661.00 | 652.47 | 651.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 661.10 | 654.19 | 652.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 646.80 | 651.35 | 651.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 646.80 | 651.35 | 651.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 641.00 | 649.28 | 650.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 648.45 | 648.27 | 650.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 648.45 | 648.27 | 650.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 652.55 | 649.16 | 650.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 644.40 | 649.38 | 650.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 13:15:00 | 651.60 | 650.60 | 650.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 651.60 | 650.60 | 650.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 656.60 | 651.80 | 651.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 694.55 | 695.36 | 679.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 12:00:00 | 694.55 | 695.36 | 679.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 682.80 | 692.84 | 679.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:00:00 | 682.80 | 692.84 | 679.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 684.65 | 691.21 | 680.11 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 651.60 | 671.79 | 674.09 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 678.85 | 671.88 | 671.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 692.25 | 681.40 | 676.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 669.95 | 681.07 | 677.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 669.95 | 681.07 | 677.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 669.95 | 681.07 | 677.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:15:00 | 670.00 | 681.07 | 677.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 670.00 | 678.86 | 676.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 656.50 | 678.86 | 676.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 661.00 | 672.85 | 674.42 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 693.90 | 676.61 | 674.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 695.65 | 684.28 | 679.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 693.50 | 697.43 | 691.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 693.50 | 697.43 | 691.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 693.50 | 697.43 | 691.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 693.50 | 697.43 | 691.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 698.00 | 697.54 | 692.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 681.50 | 697.54 | 692.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 693.40 | 696.72 | 692.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 702.10 | 696.72 | 692.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 695.30 | 696.13 | 692.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 687.40 | 691.09 | 691.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 687.40 | 691.09 | 691.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 670.10 | 683.45 | 687.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 664.45 | 664.13 | 672.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 667.95 | 664.13 | 672.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 664.40 | 663.48 | 668.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 667.05 | 663.48 | 668.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 654.55 | 662.32 | 667.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:30:00 | 658.70 | 662.32 | 667.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 667.00 | 662.72 | 666.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 667.50 | 662.72 | 666.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 669.55 | 664.09 | 667.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 669.55 | 664.09 | 667.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 666.55 | 664.58 | 666.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 671.85 | 664.58 | 666.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 668.45 | 665.36 | 667.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 668.45 | 665.36 | 667.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 668.45 | 665.97 | 667.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:30:00 | 670.45 | 665.97 | 667.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 673.00 | 667.49 | 667.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 662.30 | 667.49 | 667.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 10:15:00 | 664.15 | 666.95 | 667.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 664.00 | 663.09 | 663.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:00:00 | 663.85 | 663.25 | 663.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 663.65 | 663.33 | 663.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 664.70 | 663.33 | 663.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 656.05 | 661.87 | 663.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 653.90 | 661.87 | 663.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:00:00 | 652.50 | 645.15 | 651.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 676.00 | 652.73 | 653.89 | SL hit (close>static) qty=1.00 sl=674.00 alert=retest2 |

### Cycle 50 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 691.60 | 660.51 | 657.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 714.90 | 682.29 | 674.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 13:15:00 | 740.45 | 741.24 | 726.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-01 13:45:00 | 740.45 | 741.24 | 726.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 721.00 | 736.19 | 727.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 731.25 | 730.84 | 727.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 15:00:00 | 734.95 | 731.66 | 727.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-06 09:15:00 | 804.38 | 740.10 | 732.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 726.55 | 736.08 | 737.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 14:15:00 | 723.90 | 732.66 | 735.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 738.80 | 731.15 | 734.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 738.80 | 731.15 | 734.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 738.80 | 731.15 | 734.04 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 745.90 | 736.54 | 736.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 758.65 | 742.65 | 739.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 12:15:00 | 746.50 | 747.30 | 743.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 13:00:00 | 746.50 | 747.30 | 743.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 746.00 | 746.39 | 743.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:15:00 | 743.00 | 746.39 | 743.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 743.00 | 745.71 | 743.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 744.65 | 745.71 | 743.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 748.25 | 746.22 | 743.75 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 734.95 | 741.56 | 742.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 730.60 | 739.37 | 741.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 737.30 | 737.14 | 739.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 737.30 | 737.14 | 739.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 737.30 | 737.14 | 739.74 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 765.20 | 745.77 | 743.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 15:15:00 | 774.00 | 760.38 | 751.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 11:15:00 | 770.00 | 770.22 | 764.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:00:00 | 770.00 | 770.22 | 764.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 760.05 | 770.19 | 766.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 757.15 | 770.19 | 766.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 758.30 | 767.81 | 766.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:45:00 | 756.75 | 767.81 | 766.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 768.60 | 769.49 | 767.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 765.90 | 769.49 | 767.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 768.15 | 769.22 | 767.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 768.60 | 769.22 | 767.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 774.45 | 770.27 | 768.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 780.25 | 772.78 | 770.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:00:00 | 778.20 | 772.78 | 770.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:15:00 | 784.85 | 771.64 | 770.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 765.10 | 803.86 | 802.98 | SL hit (close<static) qty=1.00 sl=768.15 alert=retest2 |

### Cycle 55 — SELL (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 10:15:00 | 766.85 | 796.46 | 799.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 757.85 | 768.67 | 775.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 764.10 | 762.38 | 768.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:30:00 | 762.20 | 762.38 | 768.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 757.15 | 736.71 | 741.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 757.15 | 736.71 | 741.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 745.00 | 738.37 | 741.46 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 750.95 | 744.73 | 743.94 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 15:15:00 | 738.00 | 743.26 | 743.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 737.60 | 741.35 | 742.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 13:15:00 | 740.00 | 739.27 | 741.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 738.30 | 739.27 | 741.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 739.85 | 739.38 | 741.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:45:00 | 741.00 | 739.38 | 741.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 738.40 | 739.19 | 740.83 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-22 15:00:00 | 713.45 | 2025-05-23 12:15:00 | 701.05 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-05-23 09:15:00 | 710.85 | 2025-05-23 12:15:00 | 701.05 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-05-23 10:45:00 | 711.00 | 2025-05-23 12:15:00 | 701.05 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-05-23 12:00:00 | 713.60 | 2025-05-23 12:15:00 | 701.05 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-06-04 11:15:00 | 724.00 | 2025-06-04 12:15:00 | 718.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-17 10:15:00 | 775.00 | 2025-06-20 11:15:00 | 776.90 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-07-17 10:00:00 | 820.05 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-17 10:30:00 | 821.95 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-18 11:30:00 | 821.25 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-18 12:15:00 | 821.50 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-18 14:30:00 | 826.15 | 2025-07-21 09:15:00 | 815.35 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-07-30 09:15:00 | 792.15 | 2025-08-01 09:15:00 | 808.05 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-31 09:30:00 | 790.00 | 2025-08-01 09:15:00 | 808.05 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-07-31 14:30:00 | 788.00 | 2025-08-01 09:15:00 | 808.05 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-08-01 09:30:00 | 792.05 | 2025-08-01 10:15:00 | 817.00 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2025-08-05 09:45:00 | 822.55 | 2025-08-06 10:15:00 | 805.05 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-08-05 10:30:00 | 822.10 | 2025-08-06 10:15:00 | 805.05 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-03 10:15:00 | 779.05 | 2025-09-03 15:15:00 | 773.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-09-11 14:45:00 | 791.20 | 2025-09-26 09:15:00 | 870.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-06 11:15:00 | 837.90 | 2025-10-08 09:15:00 | 796.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-06 11:15:00 | 837.90 | 2025-10-09 13:15:00 | 793.35 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2025-11-14 13:30:00 | 651.00 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-11-14 14:45:00 | 644.00 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-11-17 09:30:00 | 649.30 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-11-17 12:00:00 | 650.25 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-18 09:15:00 | 649.25 | 2025-11-21 12:15:00 | 651.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-12-02 15:15:00 | 636.00 | 2025-12-04 09:15:00 | 640.70 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-03 10:15:00 | 635.90 | 2025-12-04 09:15:00 | 640.70 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-03 12:30:00 | 636.25 | 2025-12-04 09:15:00 | 640.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-12-04 10:45:00 | 636.35 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-04 12:15:00 | 633.05 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-12-04 13:30:00 | 634.55 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-05 09:15:00 | 630.80 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-12-09 12:00:00 | 633.65 | 2025-12-09 13:15:00 | 646.10 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-12-11 10:15:00 | 660.05 | 2025-12-12 15:15:00 | 650.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-12-11 11:15:00 | 660.00 | 2025-12-12 15:15:00 | 650.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-12-11 11:45:00 | 663.90 | 2025-12-12 15:15:00 | 650.00 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-12-12 09:15:00 | 660.20 | 2025-12-12 15:15:00 | 650.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-12-18 09:15:00 | 637.60 | 2025-12-22 10:15:00 | 648.30 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-12-24 09:15:00 | 656.20 | 2025-12-26 11:15:00 | 639.65 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-01-02 10:30:00 | 648.00 | 2026-01-05 10:15:00 | 642.55 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-02 13:15:00 | 648.45 | 2026-01-05 10:15:00 | 642.55 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-02 14:30:00 | 647.70 | 2026-01-05 10:15:00 | 642.55 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-06 09:15:00 | 635.20 | 2026-01-06 14:15:00 | 603.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 635.20 | 2026-01-09 15:15:00 | 571.68 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-29 13:00:00 | 523.55 | 2026-01-30 15:15:00 | 575.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 09:30:00 | 547.00 | 2026-02-03 09:15:00 | 601.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-17 12:15:00 | 630.70 | 2026-02-18 09:15:00 | 632.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-02-17 15:00:00 | 630.45 | 2026-02-18 09:15:00 | 632.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-02-18 09:15:00 | 630.40 | 2026-02-18 09:15:00 | 632.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-02-20 11:30:00 | 672.00 | 2026-02-25 11:15:00 | 646.80 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2026-02-23 10:45:00 | 658.00 | 2026-02-25 11:15:00 | 646.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-02-23 12:30:00 | 654.60 | 2026-02-25 11:15:00 | 646.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-24 10:15:00 | 661.00 | 2026-02-25 11:15:00 | 646.80 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-02-26 11:30:00 | 644.40 | 2026-02-26 13:15:00 | 651.60 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-03-12 10:15:00 | 702.10 | 2026-03-13 09:15:00 | 687.40 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-03-12 11:45:00 | 695.30 | 2026-03-13 09:15:00 | 687.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-03-19 09:15:00 | 662.30 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-03-19 10:15:00 | 664.15 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-20 12:15:00 | 664.00 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-20 13:00:00 | 663.85 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-03-20 15:15:00 | 653.90 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2026-03-24 10:00:00 | 652.50 | 2026-03-24 11:15:00 | 676.00 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2026-04-02 13:45:00 | 731.25 | 2026-04-06 09:15:00 | 804.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 15:00:00 | 734.95 | 2026-04-07 12:15:00 | 726.55 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-04-07 10:30:00 | 737.15 | 2026-04-07 12:15:00 | 726.55 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-04-22 09:30:00 | 780.25 | 2026-04-27 09:15:00 | 765.10 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-04-22 10:00:00 | 778.20 | 2026-04-27 09:15:00 | 765.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-04-22 13:15:00 | 784.85 | 2026-04-27 09:15:00 | 765.10 | STOP_HIT | 1.00 | -2.52% |
