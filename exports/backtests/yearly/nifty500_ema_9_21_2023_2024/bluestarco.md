# Blue Star Ltd. (BLUESTARCO)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1691.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 219 |
| ALERT1 | 146 |
| ALERT2 | 144 |
| ALERT2_SKIP | 88 |
| ALERT3 | 399 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 191 |
| PARTIAL | 27 |
| TARGET_HIT | 26 |
| STOP_HIT | 164 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 217 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 83 / 134
- **Target hits / Stop hits / Partials:** 26 / 164 / 27
- **Avg / median % per leg:** 1.25% / -0.55%
- **Sum % (uncompounded):** 271.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 25 | 32.5% | 15 | 62 | 0 | 1.25% | 96.2% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.11% | -0.5% |
| BUY @ 3rd Alert (retest2) | 73 | 24 | 32.9% | 15 | 58 | 0 | 1.32% | 96.7% |
| SELL (all) | 140 | 58 | 41.4% | 11 | 102 | 27 | 1.25% | 175.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 140 | 58 | 41.4% | 11 | 102 | 27 | 1.25% | 175.4% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.11% | -0.5% |
| retest2 (combined) | 213 | 82 | 38.5% | 26 | 160 | 27 | 1.28% | 272.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 09:15:00 | 708.05 | 712.05 | 712.44 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 713.88 | 712.21 | 712.16 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 11:15:00 | 710.83 | 711.90 | 712.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 12:15:00 | 709.83 | 711.49 | 711.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-16 14:15:00 | 712.85 | 711.06 | 711.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-16 14:15:00 | 712.85 | 711.06 | 711.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 14:15:00 | 712.85 | 711.06 | 711.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-16 14:30:00 | 712.65 | 711.06 | 711.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 15:15:00 | 711.95 | 711.24 | 711.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 09:15:00 | 713.75 | 711.24 | 711.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 706.85 | 710.36 | 711.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 11:00:00 | 704.00 | 709.09 | 710.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 13:30:00 | 704.75 | 706.82 | 708.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 15:00:00 | 705.23 | 706.50 | 708.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-18 10:15:00 | 720.85 | 710.13 | 709.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 10:15:00 | 720.85 | 710.13 | 709.80 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 11:15:00 | 702.45 | 709.63 | 710.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 12:15:00 | 701.23 | 707.95 | 709.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 11:15:00 | 708.50 | 705.24 | 707.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 11:15:00 | 708.50 | 705.24 | 707.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 708.50 | 705.24 | 707.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:30:00 | 706.40 | 705.24 | 707.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 12:15:00 | 709.38 | 706.07 | 707.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 12:30:00 | 709.28 | 706.07 | 707.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 14:15:00 | 705.43 | 706.11 | 707.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 12:30:00 | 703.58 | 706.07 | 706.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 09:15:00 | 717.53 | 704.38 | 705.34 | SL hit (close>static) qty=1.00 sl=708.18 alert=retest2 |

### Cycle 6 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 717.43 | 706.99 | 706.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 09:15:00 | 725.70 | 715.82 | 711.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 09:15:00 | 725.00 | 728.36 | 724.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 09:45:00 | 724.53 | 728.36 | 724.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 11:15:00 | 732.65 | 728.68 | 725.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 13:30:00 | 735.00 | 730.72 | 726.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 10:15:00 | 734.15 | 732.61 | 728.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 11:15:00 | 725.85 | 728.15 | 728.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 11:15:00 | 725.85 | 728.15 | 728.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 14:15:00 | 720.55 | 726.12 | 727.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 14:15:00 | 722.75 | 721.78 | 723.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-01 15:00:00 | 722.75 | 721.78 | 723.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 725.00 | 722.42 | 723.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:15:00 | 723.00 | 722.42 | 723.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 720.00 | 721.94 | 723.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 10:45:00 | 718.50 | 721.55 | 723.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 11:15:00 | 718.75 | 721.55 | 723.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 12:30:00 | 718.88 | 720.86 | 722.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 13:00:00 | 719.00 | 720.86 | 722.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 717.33 | 720.04 | 721.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-06 11:15:00 | 722.43 | 721.59 | 721.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 11:15:00 | 722.43 | 721.59 | 721.56 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 12:15:00 | 721.00 | 721.47 | 721.51 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 13:15:00 | 726.00 | 722.38 | 721.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 09:15:00 | 729.23 | 724.52 | 723.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 09:15:00 | 730.33 | 730.85 | 727.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 09:30:00 | 729.15 | 730.85 | 727.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 725.35 | 729.45 | 727.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:30:00 | 724.48 | 729.45 | 727.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 724.45 | 728.45 | 727.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 13:00:00 | 724.45 | 728.45 | 727.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 729.78 | 730.25 | 728.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:45:00 | 730.53 | 730.25 | 728.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 727.88 | 729.78 | 728.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 12:45:00 | 729.13 | 729.78 | 728.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 729.88 | 729.80 | 728.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-12 09:15:00 | 732.15 | 729.31 | 728.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 09:15:00 | 726.53 | 728.76 | 728.54 | SL hit (close<static) qty=1.00 sl=727.03 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 11:15:00 | 775.85 | 788.73 | 789.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 14:15:00 | 769.65 | 775.76 | 779.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 11:15:00 | 779.95 | 771.48 | 775.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 11:15:00 | 779.95 | 771.48 | 775.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 11:15:00 | 779.95 | 771.48 | 775.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 12:00:00 | 779.95 | 771.48 | 775.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 12:15:00 | 776.25 | 772.43 | 775.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 13:30:00 | 773.35 | 773.14 | 775.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 14:45:00 | 772.70 | 772.10 | 775.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 09:30:00 | 772.50 | 771.13 | 774.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 13:00:00 | 772.85 | 772.87 | 774.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 769.30 | 772.16 | 773.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 15:15:00 | 767.45 | 771.67 | 773.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 11:15:00 | 767.70 | 770.73 | 772.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-04 12:00:00 | 767.00 | 769.98 | 771.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-04 13:15:00 | 775.00 | 771.07 | 772.13 | SL hit (close>static) qty=1.00 sl=773.95 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 09:15:00 | 783.65 | 774.87 | 773.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 13:15:00 | 790.05 | 782.07 | 777.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 12:15:00 | 785.10 | 785.65 | 781.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 12:30:00 | 785.10 | 785.65 | 781.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 798.00 | 789.58 | 785.03 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 11:15:00 | 785.00 | 788.08 | 788.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 13:15:00 | 784.00 | 786.77 | 787.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 10:15:00 | 786.90 | 786.26 | 786.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 10:15:00 | 786.90 | 786.26 | 786.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 786.90 | 786.26 | 786.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 10:30:00 | 785.70 | 786.26 | 786.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 787.70 | 786.55 | 787.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 12:00:00 | 787.70 | 786.55 | 787.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 12:15:00 | 790.30 | 787.30 | 787.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 13:15:00 | 794.15 | 788.67 | 787.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-14 10:15:00 | 791.30 | 792.02 | 790.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-14 11:00:00 | 791.30 | 792.02 | 790.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 792.05 | 792.02 | 790.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 15:15:00 | 794.10 | 791.88 | 790.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 10:30:00 | 794.90 | 792.99 | 791.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 12:15:00 | 795.75 | 792.99 | 791.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 14:30:00 | 794.25 | 794.01 | 792.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 792.00 | 793.60 | 792.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 788.70 | 793.71 | 792.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 787.55 | 792.48 | 792.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-18 10:15:00 | 787.55 | 792.48 | 792.14 | SL hit (close<static) qty=1.00 sl=790.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 785.45 | 791.07 | 791.53 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 14:15:00 | 794.90 | 790.80 | 790.68 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 780.00 | 789.31 | 790.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 13:15:00 | 774.00 | 782.09 | 784.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 781.90 | 779.98 | 782.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 781.90 | 779.98 | 782.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 781.90 | 779.98 | 782.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 10:00:00 | 781.90 | 779.98 | 782.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 785.75 | 781.14 | 783.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:00:00 | 785.75 | 781.14 | 783.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 785.15 | 781.94 | 783.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:45:00 | 786.65 | 781.94 | 783.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 12:15:00 | 784.55 | 782.46 | 783.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 13:00:00 | 784.55 | 782.46 | 783.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 13:15:00 | 776.15 | 781.20 | 782.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 13:45:00 | 783.95 | 781.20 | 782.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 780.30 | 781.02 | 782.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 14:30:00 | 781.10 | 781.02 | 782.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 771.40 | 778.93 | 781.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 10:45:00 | 770.55 | 777.05 | 780.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 15:00:00 | 767.00 | 773.11 | 777.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 10:15:00 | 768.50 | 771.57 | 775.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-31 13:15:00 | 777.95 | 773.66 | 773.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 13:15:00 | 777.95 | 773.66 | 773.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 785.25 | 775.98 | 774.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 12:15:00 | 786.50 | 787.46 | 781.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 13:00:00 | 786.50 | 787.46 | 781.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 784.95 | 786.65 | 782.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:30:00 | 782.65 | 786.65 | 782.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 15:15:00 | 784.00 | 786.12 | 782.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:15:00 | 783.10 | 786.12 | 782.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 783.75 | 785.65 | 782.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:30:00 | 783.05 | 785.65 | 782.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 784.40 | 785.40 | 782.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 10:45:00 | 778.50 | 785.40 | 782.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 785.10 | 785.34 | 782.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 12:15:00 | 787.50 | 785.34 | 782.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 778.60 | 783.49 | 782.45 | SL hit (close<static) qty=1.00 sl=781.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 15:15:00 | 775.00 | 780.43 | 781.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 12:15:00 | 770.55 | 777.57 | 779.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 11:15:00 | 742.15 | 737.25 | 743.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 11:15:00 | 742.15 | 737.25 | 743.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 11:15:00 | 742.15 | 737.25 | 743.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 11:30:00 | 739.05 | 737.25 | 743.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 12:15:00 | 745.05 | 738.81 | 743.18 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 10:15:00 | 751.85 | 745.67 | 745.19 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 13:15:00 | 740.15 | 746.81 | 746.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 737.00 | 743.60 | 745.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 11:15:00 | 747.00 | 743.26 | 744.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 11:15:00 | 747.00 | 743.26 | 744.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 747.00 | 743.26 | 744.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-14 12:00:00 | 747.00 | 743.26 | 744.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 744.10 | 743.43 | 744.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 15:15:00 | 739.90 | 743.62 | 744.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 10:00:00 | 742.25 | 742.49 | 743.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-17 11:00:00 | 742.00 | 742.39 | 742.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 12:15:00 | 748.45 | 744.08 | 743.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 12:15:00 | 748.45 | 744.08 | 743.65 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 12:15:00 | 742.90 | 744.51 | 744.55 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 13:15:00 | 745.15 | 744.63 | 744.60 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 09:15:00 | 739.20 | 743.65 | 744.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 12:15:00 | 736.95 | 740.70 | 742.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 09:15:00 | 743.80 | 739.92 | 741.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 743.80 | 739.92 | 741.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 743.80 | 739.92 | 741.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:30:00 | 742.30 | 739.92 | 741.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 740.00 | 739.94 | 741.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 14:00:00 | 736.15 | 738.92 | 740.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 09:45:00 | 732.90 | 738.68 | 740.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 10:15:00 | 729.95 | 724.80 | 724.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 729.95 | 724.80 | 724.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 14:15:00 | 734.35 | 728.76 | 726.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 15:15:00 | 734.10 | 738.98 | 736.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 15:15:00 | 734.10 | 738.98 | 736.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 734.10 | 738.98 | 736.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:15:00 | 736.90 | 738.98 | 736.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 732.00 | 737.58 | 736.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-01 10:15:00 | 729.00 | 737.58 | 736.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 734.10 | 736.88 | 735.96 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 12:15:00 | 731.85 | 734.87 | 735.15 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 13:15:00 | 737.40 | 735.38 | 735.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 14:15:00 | 745.05 | 737.31 | 736.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 13:15:00 | 741.15 | 741.78 | 739.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-04 14:00:00 | 741.15 | 741.78 | 739.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 743.05 | 742.04 | 739.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:45:00 | 740.15 | 742.04 | 739.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 739.00 | 741.43 | 739.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:15:00 | 745.60 | 741.43 | 739.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 11:45:00 | 744.10 | 742.25 | 740.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 09:15:00 | 745.30 | 741.46 | 740.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 14:15:00 | 782.05 | 790.97 | 790.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 782.05 | 790.97 | 790.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 777.25 | 787.91 | 789.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 790.00 | 788.33 | 789.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 10:15:00 | 790.00 | 788.33 | 789.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 790.00 | 788.33 | 789.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 11:00:00 | 790.00 | 788.33 | 789.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 795.25 | 789.71 | 790.12 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 12:15:00 | 794.65 | 790.70 | 790.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 14:15:00 | 796.65 | 792.29 | 791.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 11:15:00 | 794.15 | 794.84 | 793.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 11:15:00 | 794.15 | 794.84 | 793.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 794.15 | 794.84 | 793.03 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-09-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 14:15:00 | 786.40 | 791.60 | 791.88 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 796.00 | 792.38 | 792.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 10:15:00 | 823.85 | 801.14 | 796.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 14:15:00 | 800.00 | 803.53 | 799.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 14:15:00 | 800.00 | 803.53 | 799.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 800.00 | 803.53 | 799.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 15:15:00 | 800.00 | 803.53 | 799.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 800.00 | 802.82 | 799.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 09:15:00 | 874.90 | 802.82 | 799.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 14:15:00 | 887.75 | 896.22 | 896.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 14:15:00 | 887.75 | 896.22 | 896.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 14:15:00 | 875.80 | 886.22 | 890.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-03 12:15:00 | 890.90 | 884.17 | 887.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 12:15:00 | 890.90 | 884.17 | 887.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 12:15:00 | 890.90 | 884.17 | 887.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 13:00:00 | 890.90 | 884.17 | 887.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 887.00 | 884.74 | 887.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 15:15:00 | 882.00 | 885.36 | 887.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 12:30:00 | 881.30 | 884.32 | 886.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 14:15:00 | 882.85 | 884.14 | 885.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 882.55 | 884.77 | 885.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 10:15:00 | 887.75 | 885.36 | 885.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 11:00:00 | 887.75 | 885.36 | 885.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 11:15:00 | 880.70 | 884.43 | 885.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 13:15:00 | 875.15 | 884.55 | 885.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 09:30:00 | 878.00 | 880.36 | 882.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 13:15:00 | 874.15 | 879.44 | 881.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 10:15:00 | 889.00 | 880.03 | 880.78 | SL hit (close>static) qty=1.00 sl=888.50 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 11:15:00 | 890.50 | 882.12 | 881.66 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 15:15:00 | 865.00 | 878.51 | 880.34 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 15:15:00 | 885.80 | 874.75 | 874.41 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 12:15:00 | 873.45 | 876.81 | 876.81 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 09:15:00 | 885.65 | 877.66 | 877.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 09:15:00 | 915.65 | 891.48 | 884.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 13:15:00 | 933.75 | 938.52 | 930.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-20 14:00:00 | 933.75 | 938.52 | 930.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 917.55 | 934.33 | 929.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 15:00:00 | 917.55 | 934.33 | 929.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 922.00 | 931.86 | 928.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:15:00 | 910.95 | 931.86 | 928.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 10:15:00 | 917.10 | 925.78 | 926.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 905.25 | 920.05 | 923.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 14:15:00 | 890.05 | 881.84 | 895.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 15:00:00 | 890.05 | 881.84 | 895.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 15:15:00 | 891.00 | 883.67 | 895.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:15:00 | 873.00 | 883.67 | 895.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 927.00 | 881.38 | 877.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 927.00 | 881.38 | 877.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 11:15:00 | 957.00 | 934.88 | 922.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 958.10 | 962.78 | 953.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 12:00:00 | 958.10 | 962.78 | 953.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 962.50 | 959.93 | 955.76 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 15:15:00 | 943.95 | 952.83 | 953.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 10:15:00 | 931.05 | 947.38 | 951.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 931.00 | 930.34 | 938.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 931.00 | 930.34 | 938.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 931.00 | 930.34 | 938.96 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-11-12 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-12 18:15:00 | 961.00 | 944.79 | 942.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 10:15:00 | 980.85 | 970.83 | 962.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 14:15:00 | 969.60 | 975.21 | 967.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 14:15:00 | 969.60 | 975.21 | 967.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 14:15:00 | 969.60 | 975.21 | 967.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-16 15:00:00 | 969.60 | 975.21 | 967.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 976.00 | 975.37 | 968.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 09:15:00 | 977.10 | 975.37 | 968.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 982.70 | 976.83 | 969.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 10:00:00 | 989.60 | 976.65 | 972.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:15:00 | 990.95 | 979.87 | 976.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 13:15:00 | 988.90 | 985.03 | 980.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 14:30:00 | 988.80 | 985.51 | 981.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 989.25 | 986.22 | 982.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-22 12:15:00 | 968.90 | 980.38 | 980.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 968.90 | 980.38 | 980.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 13:15:00 | 963.15 | 976.94 | 979.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 10:15:00 | 979.20 | 976.26 | 977.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 10:15:00 | 979.20 | 976.26 | 977.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 979.20 | 976.26 | 977.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 11:15:00 | 981.15 | 976.26 | 977.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 11:15:00 | 977.15 | 976.44 | 977.85 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 996.20 | 980.41 | 979.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 13:15:00 | 1009.10 | 993.25 | 986.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 13:15:00 | 1003.75 | 1004.15 | 996.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 14:00:00 | 1003.75 | 1004.15 | 996.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 999.90 | 1002.72 | 997.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 09:15:00 | 1006.55 | 1002.72 | 997.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 13:15:00 | 987.25 | 993.98 | 994.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 13:15:00 | 987.25 | 993.98 | 994.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 14:15:00 | 987.00 | 992.59 | 993.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 11:15:00 | 991.55 | 988.71 | 991.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 11:15:00 | 991.55 | 988.71 | 991.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 11:15:00 | 991.55 | 988.71 | 991.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:00:00 | 991.55 | 988.71 | 991.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 987.80 | 988.53 | 990.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:30:00 | 988.80 | 988.53 | 990.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-11-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 13:15:00 | 1015.00 | 993.82 | 993.05 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 12:15:00 | 990.00 | 992.64 | 992.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-04 09:15:00 | 981.40 | 989.28 | 991.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 12:15:00 | 995.00 | 988.92 | 990.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 12:15:00 | 995.00 | 988.92 | 990.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 12:15:00 | 995.00 | 988.92 | 990.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-04 13:00:00 | 995.00 | 988.92 | 990.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 13:15:00 | 991.80 | 989.50 | 990.53 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 09:15:00 | 1019.50 | 996.24 | 993.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 12:15:00 | 1024.70 | 1008.51 | 1000.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 09:15:00 | 1015.60 | 1015.66 | 1006.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-06 09:30:00 | 1019.20 | 1015.66 | 1006.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 1001.30 | 1011.83 | 1007.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:00:00 | 1001.30 | 1011.83 | 1007.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 13:15:00 | 1004.00 | 1010.26 | 1006.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 13:30:00 | 1001.50 | 1010.26 | 1006.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 1005.55 | 1009.32 | 1006.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 14:45:00 | 1004.10 | 1009.32 | 1006.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 15:15:00 | 1004.65 | 1008.39 | 1006.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:15:00 | 1001.05 | 1008.39 | 1006.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 1009.40 | 1008.59 | 1006.85 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 1001.85 | 1007.19 | 1007.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 10:15:00 | 999.90 | 1004.62 | 1005.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 1003.90 | 1000.43 | 1002.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 1003.90 | 1000.43 | 1002.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 1003.90 | 1000.43 | 1002.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:45:00 | 1006.60 | 1000.43 | 1002.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 1012.95 | 1002.94 | 1003.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 11:00:00 | 1012.95 | 1002.94 | 1003.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 11:15:00 | 1006.00 | 1003.55 | 1003.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 12:15:00 | 1004.95 | 1003.55 | 1003.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 09:15:00 | 1002.80 | 1002.17 | 1002.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 14:15:00 | 954.70 | 989.76 | 994.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 10:15:00 | 952.66 | 966.10 | 976.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-12-21 09:15:00 | 904.46 | 938.62 | 951.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 50 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 950.00 | 946.01 | 945.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 11:15:00 | 952.55 | 949.43 | 948.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 946.20 | 949.86 | 948.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 14:15:00 | 946.20 | 949.86 | 948.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 946.20 | 949.86 | 948.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 15:00:00 | 946.20 | 949.86 | 948.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 949.00 | 949.69 | 948.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 09:15:00 | 949.95 | 949.69 | 948.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2023-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 09:15:00 | 940.40 | 947.83 | 948.19 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 15:15:00 | 952.50 | 947.39 | 946.74 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 10:15:00 | 943.45 | 946.13 | 946.25 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 949.90 | 946.88 | 946.58 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 14:15:00 | 941.95 | 946.19 | 946.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-04 09:15:00 | 941.00 | 944.64 | 945.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 15:15:00 | 942.00 | 938.22 | 941.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 15:15:00 | 942.00 | 938.22 | 941.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 942.00 | 938.22 | 941.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:15:00 | 943.90 | 938.22 | 941.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 947.35 | 940.05 | 941.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 14:15:00 | 935.20 | 940.40 | 941.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 09:15:00 | 949.60 | 943.25 | 942.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 09:15:00 | 949.60 | 943.25 | 942.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-08 15:15:00 | 955.00 | 949.48 | 946.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 961.25 | 966.35 | 959.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 961.25 | 966.35 | 959.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 961.25 | 966.35 | 959.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:00:00 | 961.25 | 966.35 | 959.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 957.75 | 964.63 | 959.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:45:00 | 958.75 | 964.63 | 959.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 964.60 | 964.63 | 959.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 14:45:00 | 970.45 | 965.17 | 961.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-15 13:15:00 | 1067.50 | 1041.21 | 1018.36 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 1067.70 | 1076.16 | 1076.41 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 10:15:00 | 1099.35 | 1076.17 | 1075.92 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 1050.95 | 1074.45 | 1076.84 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 13:15:00 | 1080.95 | 1071.91 | 1071.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-30 09:15:00 | 1090.05 | 1077.64 | 1074.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 12:15:00 | 1080.00 | 1082.90 | 1078.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 12:15:00 | 1080.00 | 1082.90 | 1078.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 1080.00 | 1082.90 | 1078.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 13:00:00 | 1080.00 | 1082.90 | 1078.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 1080.45 | 1082.41 | 1078.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 13:45:00 | 1085.20 | 1082.41 | 1078.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 1062.05 | 1078.33 | 1076.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:45:00 | 1071.70 | 1078.33 | 1076.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 1088.00 | 1080.27 | 1077.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 14:00:00 | 1127.95 | 1102.73 | 1090.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-06 14:15:00 | 1161.35 | 1170.47 | 1170.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 14:15:00 | 1161.35 | 1170.47 | 1170.60 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 09:15:00 | 1180.95 | 1170.92 | 1170.10 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 12:15:00 | 1163.60 | 1169.94 | 1169.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 14:15:00 | 1152.10 | 1164.62 | 1167.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 09:15:00 | 1162.20 | 1162.12 | 1165.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 1162.20 | 1162.12 | 1165.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1162.20 | 1162.12 | 1165.69 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 13:15:00 | 1187.00 | 1164.98 | 1162.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 14:15:00 | 1202.15 | 1172.41 | 1165.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 10:15:00 | 1176.75 | 1179.88 | 1171.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-14 11:00:00 | 1176.75 | 1179.88 | 1171.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 1279.85 | 1286.57 | 1275.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 1279.85 | 1286.57 | 1275.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 1270.30 | 1283.31 | 1274.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 1269.00 | 1283.31 | 1274.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 1273.55 | 1281.36 | 1274.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:30:00 | 1272.90 | 1279.54 | 1274.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 1261.10 | 1275.85 | 1273.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 1261.10 | 1275.85 | 1273.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 1267.10 | 1274.10 | 1272.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:30:00 | 1270.00 | 1274.10 | 1272.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 1290.00 | 1277.60 | 1274.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:30:00 | 1279.45 | 1277.60 | 1274.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 1265.10 | 1274.99 | 1273.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 09:15:00 | 1297.20 | 1274.99 | 1273.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-26 14:15:00 | 1254.85 | 1286.28 | 1286.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 14:15:00 | 1254.85 | 1286.28 | 1286.34 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 11:15:00 | 1292.00 | 1286.61 | 1286.02 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 1284.50 | 1286.63 | 1286.86 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 13:15:00 | 1297.45 | 1288.79 | 1287.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-28 14:15:00 | 1300.00 | 1291.04 | 1288.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 15:15:00 | 1285.00 | 1289.83 | 1288.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 15:15:00 | 1285.00 | 1289.83 | 1288.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 1285.00 | 1289.83 | 1288.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 09:15:00 | 1282.80 | 1289.83 | 1288.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 1285.95 | 1289.05 | 1288.33 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 10:15:00 | 1279.45 | 1287.13 | 1287.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 13:15:00 | 1269.15 | 1279.91 | 1283.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 1283.15 | 1278.01 | 1281.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 1283.15 | 1278.01 | 1281.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 1283.15 | 1278.01 | 1281.66 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 1308.00 | 1286.82 | 1284.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 14:15:00 | 1309.50 | 1294.74 | 1289.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 11:15:00 | 1307.25 | 1307.76 | 1300.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 11:15:00 | 1307.25 | 1307.76 | 1300.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 1307.25 | 1307.76 | 1300.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:45:00 | 1298.00 | 1307.76 | 1300.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 1316.80 | 1313.43 | 1306.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 10:45:00 | 1330.00 | 1316.78 | 1308.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 14:15:00 | 1289.10 | 1310.43 | 1308.35 | SL hit (close<static) qty=1.00 sl=1305.05 alert=retest2 |

### Cycle 71 — SELL (started 2024-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 15:15:00 | 1290.00 | 1306.34 | 1306.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1278.15 | 1300.70 | 1304.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 11:15:00 | 1288.20 | 1276.28 | 1285.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 11:15:00 | 1288.20 | 1276.28 | 1285.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 11:15:00 | 1288.20 | 1276.28 | 1285.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:45:00 | 1286.95 | 1276.28 | 1285.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 12:15:00 | 1293.95 | 1279.81 | 1286.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 13:00:00 | 1293.95 | 1279.81 | 1286.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 1331.20 | 1290.09 | 1290.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:00:00 | 1331.20 | 1290.09 | 1290.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 14:15:00 | 1326.60 | 1297.39 | 1293.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 09:15:00 | 1354.05 | 1312.04 | 1301.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 14:15:00 | 1324.45 | 1326.44 | 1313.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 15:00:00 | 1324.45 | 1326.44 | 1313.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 1327.15 | 1326.19 | 1315.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:30:00 | 1327.30 | 1326.19 | 1315.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 1319.60 | 1324.88 | 1316.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:45:00 | 1323.50 | 1324.88 | 1316.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 1310.40 | 1321.98 | 1315.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:00:00 | 1310.40 | 1321.98 | 1315.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 1310.50 | 1319.68 | 1315.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 13:00:00 | 1310.50 | 1319.68 | 1315.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 15:15:00 | 1302.00 | 1312.36 | 1312.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 1289.15 | 1307.72 | 1310.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 15:15:00 | 1284.00 | 1274.34 | 1288.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 09:15:00 | 1258.55 | 1274.34 | 1288.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 1265.40 | 1272.55 | 1286.72 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 1287.00 | 1279.50 | 1278.75 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 1265.25 | 1277.32 | 1278.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 1240.70 | 1267.45 | 1272.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 1252.80 | 1242.74 | 1254.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 1252.80 | 1242.74 | 1254.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1252.80 | 1242.74 | 1254.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:45:00 | 1252.55 | 1242.74 | 1254.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 1251.40 | 1244.48 | 1254.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:45:00 | 1258.20 | 1244.48 | 1254.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 11:15:00 | 1250.15 | 1245.61 | 1253.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 09:15:00 | 1242.00 | 1246.89 | 1251.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 12:00:00 | 1244.55 | 1248.22 | 1251.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 15:15:00 | 1237.90 | 1247.79 | 1250.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 09:30:00 | 1242.20 | 1244.88 | 1248.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 1246.20 | 1245.15 | 1248.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 10:45:00 | 1243.65 | 1245.15 | 1248.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 1254.65 | 1247.05 | 1248.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:30:00 | 1252.75 | 1247.05 | 1248.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-26 12:15:00 | 1269.50 | 1251.54 | 1250.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 12:15:00 | 1269.50 | 1251.54 | 1250.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 09:15:00 | 1284.20 | 1263.05 | 1256.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 09:15:00 | 1272.00 | 1282.45 | 1272.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 1272.00 | 1282.45 | 1272.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1272.00 | 1282.45 | 1272.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:15:00 | 1265.00 | 1282.45 | 1272.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 1269.00 | 1279.76 | 1272.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 1288.05 | 1271.65 | 1270.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 11:45:00 | 1278.05 | 1275.69 | 1272.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 12:30:00 | 1281.80 | 1276.22 | 1273.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 15:00:00 | 1278.30 | 1277.24 | 1274.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 1278.00 | 1277.39 | 1274.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 09:15:00 | 1308.25 | 1277.39 | 1274.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-08 09:15:00 | 1416.86 | 1369.04 | 1353.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 1379.50 | 1400.96 | 1401.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 10:15:00 | 1368.70 | 1389.18 | 1395.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1388.70 | 1378.07 | 1385.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1388.70 | 1378.07 | 1385.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1388.70 | 1378.07 | 1385.68 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 15:15:00 | 1392.00 | 1388.88 | 1388.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 1415.00 | 1394.11 | 1391.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 14:15:00 | 1398.45 | 1405.16 | 1398.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 14:15:00 | 1398.45 | 1405.16 | 1398.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 1398.45 | 1405.16 | 1398.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 15:00:00 | 1398.45 | 1405.16 | 1398.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 1412.00 | 1406.53 | 1400.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 1391.25 | 1406.53 | 1400.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1392.30 | 1403.68 | 1399.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 11:45:00 | 1402.40 | 1401.66 | 1399.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 10:15:00 | 1436.95 | 1472.00 | 1475.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 10:15:00 | 1436.95 | 1472.00 | 1475.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 1428.05 | 1463.21 | 1471.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 1450.70 | 1450.51 | 1459.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 1450.70 | 1450.51 | 1459.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1450.70 | 1450.51 | 1459.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 1450.70 | 1450.51 | 1459.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 1456.70 | 1451.75 | 1459.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 12:00:00 | 1456.70 | 1451.75 | 1459.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 12:15:00 | 1458.90 | 1453.18 | 1459.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 14:00:00 | 1451.70 | 1452.88 | 1458.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 09:15:00 | 1449.55 | 1452.14 | 1457.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 1472.10 | 1456.13 | 1458.74 | SL hit (close>static) qty=1.00 sl=1459.95 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 11:15:00 | 1464.15 | 1460.67 | 1460.54 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 1457.15 | 1459.97 | 1460.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 13:15:00 | 1449.40 | 1457.86 | 1459.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 12:15:00 | 1400.00 | 1399.61 | 1414.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 13:00:00 | 1400.00 | 1399.61 | 1414.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 1413.40 | 1402.37 | 1414.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:30:00 | 1410.05 | 1402.37 | 1414.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 1414.95 | 1404.89 | 1414.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 1414.95 | 1404.89 | 1414.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 1413.55 | 1406.62 | 1414.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 1418.05 | 1406.62 | 1414.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 1413.90 | 1408.08 | 1414.25 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 1437.00 | 1418.08 | 1417.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 1461.80 | 1429.73 | 1423.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 14:15:00 | 1484.45 | 1494.22 | 1474.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 15:00:00 | 1484.45 | 1494.22 | 1474.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1560.00 | 1577.97 | 1559.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 1557.80 | 1577.97 | 1559.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 1557.30 | 1573.84 | 1559.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:30:00 | 1556.00 | 1573.84 | 1559.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 1591.10 | 1577.29 | 1562.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:30:00 | 1556.90 | 1577.29 | 1562.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 1578.60 | 1576.80 | 1564.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:45:00 | 1557.10 | 1576.80 | 1564.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1563.60 | 1574.03 | 1565.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1563.95 | 1574.03 | 1565.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1548.05 | 1568.84 | 1563.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 1548.05 | 1568.84 | 1563.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 1554.55 | 1565.98 | 1563.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 1545.50 | 1565.98 | 1563.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 13:15:00 | 1545.40 | 1559.86 | 1560.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 14:15:00 | 1528.25 | 1553.54 | 1557.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 1498.75 | 1498.38 | 1510.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 13:45:00 | 1499.55 | 1498.38 | 1510.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1501.95 | 1495.43 | 1505.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 1514.10 | 1495.43 | 1505.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1498.05 | 1495.95 | 1504.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 1487.05 | 1493.77 | 1503.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 1516.70 | 1500.94 | 1504.30 | SL hit (close>static) qty=1.00 sl=1510.45 alert=retest2 |

### Cycle 84 — BUY (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 09:15:00 | 1528.05 | 1509.89 | 1508.00 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1484.20 | 1509.03 | 1510.15 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 1541.55 | 1510.51 | 1507.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 1612.90 | 1530.99 | 1517.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1568.15 | 1594.12 | 1575.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 1568.15 | 1594.12 | 1575.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1568.15 | 1594.12 | 1575.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 1568.15 | 1594.12 | 1575.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1567.35 | 1588.77 | 1574.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:30:00 | 1585.10 | 1588.63 | 1575.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 11:15:00 | 1534.30 | 1567.79 | 1570.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 11:15:00 | 1534.30 | 1567.79 | 1570.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 12:15:00 | 1525.00 | 1559.24 | 1566.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1559.10 | 1547.23 | 1556.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1559.10 | 1547.23 | 1556.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1559.10 | 1547.23 | 1556.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:00:00 | 1524.40 | 1545.02 | 1553.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:00:00 | 1526.05 | 1541.22 | 1551.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 11:15:00 | 1567.55 | 1550.10 | 1551.42 | SL hit (close>static) qty=1.00 sl=1565.20 alert=retest2 |

### Cycle 88 — BUY (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 12:15:00 | 1566.85 | 1553.45 | 1552.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1600.75 | 1567.10 | 1559.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 1654.65 | 1666.37 | 1642.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:45:00 | 1650.00 | 1666.37 | 1642.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1652.75 | 1663.65 | 1643.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 1673.00 | 1664.43 | 1645.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 15:15:00 | 1700.00 | 1720.84 | 1721.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 1700.00 | 1720.84 | 1721.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 09:15:00 | 1696.00 | 1715.87 | 1719.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 10:15:00 | 1687.25 | 1684.32 | 1698.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 11:00:00 | 1687.25 | 1684.32 | 1698.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1690.30 | 1685.60 | 1695.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:45:00 | 1692.30 | 1685.60 | 1695.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1670.05 | 1683.60 | 1692.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1702.25 | 1683.60 | 1692.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1719.15 | 1690.71 | 1695.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 1719.15 | 1690.71 | 1695.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1715.00 | 1695.57 | 1697.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 12:00:00 | 1708.30 | 1698.12 | 1698.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 12:15:00 | 1709.85 | 1700.46 | 1699.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 1709.85 | 1700.46 | 1699.20 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 1691.10 | 1697.54 | 1698.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 11:15:00 | 1684.50 | 1694.93 | 1696.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 1654.50 | 1643.97 | 1651.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 1654.50 | 1643.97 | 1651.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1654.50 | 1643.97 | 1651.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 1660.40 | 1643.97 | 1651.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1649.05 | 1644.99 | 1651.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:30:00 | 1645.00 | 1644.75 | 1649.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 09:15:00 | 1657.00 | 1615.81 | 1616.29 | SL hit (close>static) qty=1.00 sl=1656.15 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 1702.40 | 1633.13 | 1624.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 11:15:00 | 1749.90 | 1656.48 | 1635.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 1673.40 | 1684.56 | 1661.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 11:00:00 | 1673.40 | 1684.56 | 1661.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1657.40 | 1679.13 | 1661.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 1657.40 | 1679.13 | 1661.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 1653.15 | 1673.93 | 1660.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:00:00 | 1653.15 | 1673.93 | 1660.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 1652.05 | 1669.55 | 1659.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 1652.05 | 1669.55 | 1659.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 1661.50 | 1667.94 | 1660.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:15:00 | 1657.20 | 1667.94 | 1660.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 1657.20 | 1665.80 | 1659.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 1682.00 | 1665.80 | 1659.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1685.50 | 1669.74 | 1662.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 11:45:00 | 1704.00 | 1678.30 | 1667.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-09 15:15:00 | 1874.40 | 1764.43 | 1715.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 1701.80 | 1722.80 | 1724.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 1692.25 | 1710.12 | 1716.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 1708.05 | 1700.28 | 1708.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 1708.05 | 1700.28 | 1708.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1708.05 | 1700.28 | 1708.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 1711.20 | 1700.28 | 1708.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1704.60 | 1701.14 | 1708.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 11:15:00 | 1694.00 | 1701.14 | 1708.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 12:15:00 | 1709.90 | 1700.25 | 1706.67 | SL hit (close>static) qty=1.00 sl=1709.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 15:15:00 | 1730.00 | 1713.41 | 1711.59 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 1688.40 | 1708.41 | 1709.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 1664.75 | 1699.68 | 1705.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 1710.00 | 1701.25 | 1705.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 12:15:00 | 1710.00 | 1701.25 | 1705.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1710.00 | 1701.25 | 1705.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:45:00 | 1709.65 | 1701.25 | 1705.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1706.60 | 1702.32 | 1705.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 1706.60 | 1702.32 | 1705.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1693.90 | 1700.63 | 1704.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1671.85 | 1700.11 | 1703.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 1588.26 | 1639.68 | 1649.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 13:15:00 | 1646.05 | 1640.95 | 1649.06 | SL hit (close>ema200) qty=0.50 sl=1640.95 alert=retest2 |

### Cycle 96 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 1684.75 | 1656.76 | 1655.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 1689.40 | 1667.00 | 1660.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 14:15:00 | 1658.00 | 1665.20 | 1660.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 14:15:00 | 1658.00 | 1665.20 | 1660.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 1658.00 | 1665.20 | 1660.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 15:00:00 | 1658.00 | 1665.20 | 1660.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 1654.50 | 1663.06 | 1659.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:15:00 | 1676.20 | 1663.06 | 1659.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1654.55 | 1661.36 | 1659.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:30:00 | 1682.95 | 1671.27 | 1665.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 13:15:00 | 1719.00 | 1731.27 | 1732.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 13:15:00 | 1719.00 | 1731.27 | 1732.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 1710.90 | 1725.11 | 1729.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 1596.60 | 1584.36 | 1615.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 13:15:00 | 1596.60 | 1584.36 | 1615.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1596.60 | 1584.36 | 1615.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 1604.30 | 1584.36 | 1615.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1621.95 | 1591.88 | 1615.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 1621.95 | 1591.88 | 1615.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 1624.10 | 1598.33 | 1616.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 1641.35 | 1598.33 | 1616.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1600.00 | 1604.41 | 1614.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 1618.75 | 1604.41 | 1614.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 1618.70 | 1607.26 | 1614.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:00:00 | 1618.70 | 1607.26 | 1614.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1619.70 | 1609.75 | 1615.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 1619.70 | 1609.75 | 1615.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1618.40 | 1611.48 | 1615.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 1619.50 | 1611.48 | 1615.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1603.50 | 1609.89 | 1614.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 1589.90 | 1609.67 | 1612.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 1586.45 | 1603.15 | 1608.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 1702.95 | 1627.41 | 1617.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 1702.95 | 1627.41 | 1617.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 11:15:00 | 1722.75 | 1658.16 | 1633.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 09:15:00 | 1706.55 | 1707.49 | 1685.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-16 10:00:00 | 1706.55 | 1707.49 | 1685.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1713.90 | 1718.00 | 1702.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 10:00:00 | 1721.05 | 1718.61 | 1703.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 11:00:00 | 1722.25 | 1719.34 | 1705.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:45:00 | 1721.80 | 1726.06 | 1715.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 10:30:00 | 1726.00 | 1727.66 | 1717.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1692.10 | 1720.55 | 1715.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 1692.10 | 1720.55 | 1715.11 | SL hit (close<static) qty=1.00 sl=1700.05 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 1690.75 | 1708.29 | 1710.09 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 09:15:00 | 1740.80 | 1709.26 | 1707.52 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 1702.85 | 1719.66 | 1720.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 12:15:00 | 1697.30 | 1712.03 | 1716.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 15:15:00 | 1705.00 | 1702.28 | 1710.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:15:00 | 1704.95 | 1702.28 | 1710.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1720.00 | 1705.82 | 1711.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 1719.15 | 1705.82 | 1711.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1735.90 | 1711.84 | 1713.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 1735.30 | 1711.84 | 1713.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 11:15:00 | 1745.00 | 1718.47 | 1716.41 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 1697.35 | 1721.31 | 1723.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 15:15:00 | 1692.20 | 1708.45 | 1716.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 13:15:00 | 1704.30 | 1701.22 | 1709.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 13:15:00 | 1704.30 | 1701.22 | 1709.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 1704.30 | 1701.22 | 1709.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:45:00 | 1709.05 | 1701.22 | 1709.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1700.65 | 1701.10 | 1708.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 1700.65 | 1701.10 | 1708.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1674.75 | 1671.69 | 1685.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:30:00 | 1676.95 | 1671.69 | 1685.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1684.40 | 1675.16 | 1681.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1684.40 | 1675.16 | 1681.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1680.00 | 1676.13 | 1681.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1724.00 | 1676.13 | 1681.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 1723.30 | 1685.56 | 1685.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 09:15:00 | 1736.80 | 1719.86 | 1710.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 1800.10 | 1800.31 | 1780.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 13:15:00 | 1795.85 | 1800.31 | 1780.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1795.30 | 1799.14 | 1783.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 1773.95 | 1799.14 | 1783.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1775.20 | 1794.35 | 1782.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:30:00 | 1800.65 | 1795.72 | 1784.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-16 09:15:00 | 1980.72 | 1871.13 | 1838.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 1881.20 | 1885.50 | 1885.66 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 1889.25 | 1885.85 | 1885.77 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 1869.05 | 1883.16 | 1884.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 09:15:00 | 1856.60 | 1870.80 | 1877.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1884.15 | 1873.47 | 1877.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 10:15:00 | 1884.15 | 1873.47 | 1877.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1884.15 | 1873.47 | 1877.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 1884.15 | 1873.47 | 1877.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1904.10 | 1879.60 | 1880.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 1915.70 | 1879.60 | 1880.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1882.05 | 1878.77 | 1879.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:00:00 | 1882.05 | 1878.77 | 1879.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 1934.15 | 1889.85 | 1884.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 1966.95 | 1924.00 | 1905.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 11:15:00 | 2122.30 | 2124.90 | 2100.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 12:00:00 | 2122.30 | 2124.90 | 2100.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 2099.20 | 2119.76 | 2100.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:45:00 | 2097.60 | 2119.76 | 2100.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 2089.00 | 2113.61 | 2099.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 2089.00 | 2113.61 | 2099.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 2049.90 | 2100.87 | 2095.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 2049.90 | 2100.87 | 2095.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 2068.00 | 2094.29 | 2092.69 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 09:15:00 | 2071.00 | 2089.63 | 2090.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 11:15:00 | 2050.00 | 2078.57 | 2085.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 14:15:00 | 2075.30 | 2068.60 | 2078.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 14:15:00 | 2075.30 | 2068.60 | 2078.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 2075.30 | 2068.60 | 2078.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 2075.30 | 2068.60 | 2078.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 2100.00 | 2074.88 | 2080.13 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 12:15:00 | 2121.90 | 2087.74 | 2084.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 14:15:00 | 2150.85 | 2101.09 | 2091.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 2116.50 | 2120.56 | 2107.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 13:15:00 | 2116.50 | 2120.56 | 2107.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 2116.50 | 2120.56 | 2107.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 2116.50 | 2120.56 | 2107.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 2052.95 | 2107.04 | 2102.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 2052.95 | 2107.04 | 2102.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 15:15:00 | 2067.60 | 2099.15 | 2099.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 2025.00 | 2084.32 | 2092.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1980.65 | 1934.72 | 1970.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 1980.65 | 1934.72 | 1970.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1980.65 | 1934.72 | 1970.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 1986.65 | 1934.72 | 1970.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1978.00 | 1943.38 | 1970.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:15:00 | 2013.10 | 1943.38 | 1970.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 15:15:00 | 2005.00 | 1984.93 | 1983.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 14:15:00 | 2020.00 | 2001.27 | 1993.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 11:15:00 | 2004.65 | 2010.37 | 2000.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:45:00 | 2002.55 | 2010.37 | 2000.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 2017.20 | 2013.33 | 2004.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 2038.30 | 2015.05 | 2006.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:00:00 | 2024.85 | 2017.01 | 2007.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:00:00 | 2024.00 | 2020.82 | 2013.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 2026.00 | 2022.74 | 2015.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 2047.20 | 2041.32 | 2030.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 2011.90 | 2024.19 | 2025.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 2011.90 | 2024.19 | 2025.24 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 2045.00 | 2025.02 | 2022.51 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 2009.45 | 2022.00 | 2023.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 2005.00 | 2016.98 | 2020.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 15:15:00 | 1888.00 | 1879.44 | 1905.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-25 09:15:00 | 1875.50 | 1879.44 | 1905.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1847.15 | 1872.98 | 1899.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:30:00 | 1835.65 | 1865.90 | 1894.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 1829.40 | 1851.00 | 1873.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 10:00:00 | 1829.80 | 1851.00 | 1873.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:15:00 | 1832.00 | 1848.24 | 1865.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 1850.85 | 1833.37 | 1846.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 1850.85 | 1833.37 | 1846.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 1850.00 | 1836.70 | 1847.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:00:00 | 1836.05 | 1838.70 | 1846.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:00:00 | 1839.60 | 1823.20 | 1832.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 13:15:00 | 1844.20 | 1837.64 | 1837.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 13:15:00 | 1844.20 | 1837.64 | 1837.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 14:15:00 | 1870.00 | 1844.11 | 1840.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1820.20 | 1863.72 | 1853.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1820.20 | 1863.72 | 1853.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1820.20 | 1863.72 | 1853.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1820.20 | 1863.72 | 1853.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1833.50 | 1857.68 | 1851.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 1838.00 | 1857.68 | 1851.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 12:15:00 | 1823.90 | 1847.38 | 1847.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1823.90 | 1847.38 | 1847.47 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 1855.70 | 1843.74 | 1843.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1887.85 | 1853.56 | 1848.06 | Break + close above crossover candle high |

### Cycle 119 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 1753.75 | 1849.49 | 1852.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 10:15:00 | 1742.60 | 1828.11 | 1842.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 1817.25 | 1797.50 | 1817.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 1817.25 | 1797.50 | 1817.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1817.25 | 1797.50 | 1817.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:00:00 | 1817.25 | 1797.50 | 1817.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1829.90 | 1803.98 | 1818.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:30:00 | 1824.75 | 1803.98 | 1818.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 1815.00 | 1806.18 | 1817.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 12:30:00 | 1809.95 | 1806.05 | 1816.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 13:30:00 | 1806.75 | 1802.74 | 1814.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 11:15:00 | 1814.00 | 1784.31 | 1781.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 11:15:00 | 1814.00 | 1784.31 | 1781.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 09:15:00 | 1843.90 | 1813.98 | 1806.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 1829.90 | 1833.33 | 1823.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 1829.90 | 1833.33 | 1823.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1829.90 | 1833.33 | 1823.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 1826.20 | 1833.33 | 1823.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1832.55 | 1832.97 | 1824.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:45:00 | 1823.20 | 1832.97 | 1824.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 1847.05 | 1847.47 | 1841.24 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 12:15:00 | 1802.20 | 1835.09 | 1836.95 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 14:15:00 | 1853.30 | 1836.23 | 1834.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 15:15:00 | 1854.55 | 1839.89 | 1836.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 1839.90 | 1840.54 | 1837.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 1839.90 | 1840.54 | 1837.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1839.90 | 1840.54 | 1837.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 1836.40 | 1840.54 | 1837.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1838.65 | 1840.16 | 1837.69 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 1831.95 | 1835.56 | 1835.95 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 1856.85 | 1840.15 | 1837.92 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 1818.00 | 1838.63 | 1839.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 12:15:00 | 1814.60 | 1833.82 | 1837.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1840.25 | 1835.09 | 1837.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 14:15:00 | 1840.25 | 1835.09 | 1837.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1840.25 | 1835.09 | 1837.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:30:00 | 1844.00 | 1835.09 | 1837.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1845.25 | 1837.13 | 1837.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 1861.30 | 1837.13 | 1837.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 1872.10 | 1844.12 | 1841.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 1900.00 | 1868.72 | 1856.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 2021.60 | 2040.85 | 1991.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 12:00:00 | 2086.70 | 2051.81 | 2005.49 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 14:15:00 | 2092.00 | 2059.87 | 2017.29 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2107.00 | 2097.88 | 2071.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:30:00 | 2118.00 | 2103.94 | 2076.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 2090.25 | 2111.25 | 2098.42 | SL hit (close<ema400) qty=1.00 sl=2098.42 alert=retest1 |

### Cycle 127 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 2048.70 | 2087.61 | 2089.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 2046.20 | 2079.33 | 2085.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 2060.80 | 2059.87 | 2072.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 2060.80 | 2059.87 | 2072.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 2090.75 | 2066.05 | 2074.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 2090.75 | 2066.05 | 2074.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 2090.00 | 2070.84 | 2075.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:30:00 | 2094.45 | 2070.84 | 2075.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 2136.65 | 2085.71 | 2081.90 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 2070.00 | 2099.23 | 2100.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 2065.75 | 2092.54 | 2097.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 13:15:00 | 2030.90 | 2029.84 | 2049.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:45:00 | 2029.55 | 2029.84 | 2049.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2048.90 | 2033.65 | 2049.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 2048.90 | 2033.65 | 2049.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2033.00 | 2033.52 | 2047.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 2039.95 | 2033.52 | 2047.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2039.80 | 2034.78 | 2047.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 2024.15 | 2035.63 | 2044.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 2033.45 | 2004.87 | 2003.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 2033.45 | 2004.87 | 2003.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 15:15:00 | 2045.00 | 2012.90 | 2007.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 2115.55 | 2120.48 | 2087.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 11:15:00 | 2122.95 | 2117.92 | 2091.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 2122.95 | 2117.92 | 2091.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 2099.95 | 2117.92 | 2091.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 2160.55 | 2138.36 | 2113.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:15:00 | 2168.00 | 2138.36 | 2113.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-01 14:15:00 | 2384.80 | 2209.49 | 2162.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 2231.00 | 2262.14 | 2264.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 2175.00 | 2244.71 | 2256.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 2138.80 | 2122.99 | 2166.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 2138.80 | 2122.99 | 2166.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1881.85 | 1875.64 | 1907.80 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1928.00 | 1917.72 | 1917.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 12:15:00 | 1934.95 | 1921.17 | 1918.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 1924.10 | 1928.08 | 1923.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 1924.10 | 1928.08 | 1923.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 1924.10 | 1928.08 | 1923.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 1921.55 | 1928.08 | 1923.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 1926.95 | 1927.85 | 1923.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 1922.00 | 1927.85 | 1923.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1925.85 | 1927.44 | 1924.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 1924.30 | 1927.44 | 1924.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1921.95 | 1926.34 | 1924.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 1923.95 | 1926.34 | 1924.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1926.65 | 1926.40 | 1924.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 1917.80 | 1926.40 | 1924.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 1899.85 | 1921.09 | 1922.21 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 1934.15 | 1922.29 | 1922.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 1946.95 | 1927.22 | 1924.53 | Break + close above crossover candle high |

### Cycle 135 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1886.50 | 1920.26 | 1922.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1872.85 | 1910.78 | 1917.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1879.85 | 1866.37 | 1882.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1879.85 | 1866.37 | 1882.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1879.85 | 1866.37 | 1882.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1879.85 | 1866.37 | 1882.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 1885.00 | 1870.10 | 1882.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 1850.00 | 1870.10 | 1882.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 1922.70 | 1883.69 | 1886.54 | SL hit (close>static) qty=1.00 sl=1892.90 alert=retest2 |

### Cycle 136 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 1920.45 | 1891.04 | 1889.63 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1878.75 | 1892.67 | 1893.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1859.35 | 1886.01 | 1890.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1818.40 | 1778.90 | 1797.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1818.40 | 1778.90 | 1797.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1818.40 | 1778.90 | 1797.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1818.40 | 1778.90 | 1797.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1833.90 | 1789.90 | 1801.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1833.90 | 1789.90 | 1801.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1829.75 | 1812.06 | 1809.72 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 12:15:00 | 1806.20 | 1810.49 | 1810.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 1800.85 | 1808.56 | 1809.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 1818.15 | 1805.97 | 1807.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 10:15:00 | 1818.15 | 1805.97 | 1807.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1818.15 | 1805.97 | 1807.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 1818.15 | 1805.97 | 1807.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 1820.90 | 1808.96 | 1808.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 1835.90 | 1814.35 | 1811.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 14:15:00 | 1815.55 | 1816.80 | 1813.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 15:00:00 | 1815.55 | 1816.80 | 1813.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 1823.10 | 1818.06 | 1813.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 1866.10 | 1818.06 | 1813.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-01 14:15:00 | 2052.71 | 1921.46 | 1870.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 1971.05 | 1983.63 | 1985.27 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 2002.05 | 1986.24 | 1985.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 11:15:00 | 2029.95 | 1994.98 | 1989.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 1976.55 | 2010.35 | 2001.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 1976.55 | 2010.35 | 2001.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1976.55 | 2010.35 | 2001.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:30:00 | 1985.60 | 2010.35 | 2001.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1969.45 | 2002.17 | 1998.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1969.45 | 2002.17 | 1998.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1972.05 | 1996.15 | 1996.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1888.95 | 1962.69 | 1979.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1940.25 | 1900.12 | 1929.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1940.25 | 1900.12 | 1929.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1940.25 | 1900.12 | 1929.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 1940.25 | 1900.12 | 1929.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1964.90 | 1913.08 | 1932.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 1964.90 | 1913.08 | 1932.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1985.10 | 1927.48 | 1937.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1985.10 | 1927.48 | 1937.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 1997.20 | 1948.72 | 1945.78 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 1897.85 | 1948.03 | 1953.57 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 09:15:00 | 1911.15 | 1904.39 | 1904.29 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1885.25 | 1905.04 | 1905.67 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 11:15:00 | 1925.00 | 1900.26 | 1896.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-28 10:15:00 | 1943.15 | 1921.17 | 1910.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 14:15:00 | 1906.85 | 1925.34 | 1916.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 1906.85 | 1925.34 | 1916.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1906.85 | 1925.34 | 1916.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 1906.85 | 1925.34 | 1916.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1938.95 | 1928.06 | 1918.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 10:15:00 | 1947.80 | 1928.49 | 1919.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-04 09:15:00 | 2142.58 | 2032.05 | 1982.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 2098.65 | 2105.16 | 2105.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 2092.50 | 2102.62 | 2104.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 2121.60 | 2049.52 | 2062.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 2121.60 | 2049.52 | 2062.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2121.60 | 2049.52 | 2062.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 2121.60 | 2049.52 | 2062.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 2097.00 | 2059.01 | 2065.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 2105.25 | 2059.01 | 2065.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 2051.65 | 2056.68 | 2063.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:00:00 | 2051.65 | 2056.68 | 2063.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 2024.20 | 2046.96 | 2056.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 2010.80 | 2029.14 | 2038.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 2099.55 | 2041.76 | 2042.48 | SL hit (close>static) qty=1.00 sl=2066.15 alert=retest2 |

### Cycle 150 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 2127.50 | 2058.91 | 2050.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 2138.95 | 2074.92 | 2058.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 11:15:00 | 2162.50 | 2185.27 | 2153.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 12:00:00 | 2162.50 | 2185.27 | 2153.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2156.55 | 2192.65 | 2179.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2156.55 | 2192.65 | 2179.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2174.00 | 2188.92 | 2178.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 2219.65 | 2188.92 | 2178.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 2191.00 | 2195.07 | 2195.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 2191.00 | 2195.07 | 2195.40 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 12:15:00 | 2229.80 | 2201.52 | 2197.66 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 2192.05 | 2196.08 | 2196.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 2168.45 | 2191.14 | 2194.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 2115.30 | 2103.62 | 2132.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 2115.30 | 2103.62 | 2132.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 2115.30 | 2103.62 | 2132.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:30:00 | 2066.20 | 2083.72 | 2100.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1859.58 | 2016.88 | 2055.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 2013.80 | 1970.47 | 1968.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 2023.70 | 1992.00 | 1979.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2023.60 | 2033.11 | 2014.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 12:15:00 | 2020.00 | 2027.81 | 2016.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 2020.00 | 2027.81 | 2016.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:30:00 | 2017.50 | 2027.81 | 2016.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 2021.70 | 2026.59 | 2016.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:00:00 | 2030.60 | 2018.25 | 2015.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 1949.90 | 2020.26 | 2024.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 1949.90 | 2020.26 | 2024.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 14:15:00 | 1926.10 | 1974.01 | 1999.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1785.30 | 1778.10 | 1841.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 09:30:00 | 1797.00 | 1778.10 | 1841.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1682.10 | 1680.05 | 1700.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 1691.60 | 1680.05 | 1700.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1689.70 | 1679.01 | 1691.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 1697.90 | 1679.01 | 1691.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1702.00 | 1683.61 | 1692.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:30:00 | 1709.20 | 1683.61 | 1692.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1717.40 | 1690.36 | 1695.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:00:00 | 1717.40 | 1690.36 | 1695.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 1739.20 | 1700.13 | 1699.02 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 1688.00 | 1699.49 | 1699.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 1648.90 | 1689.37 | 1694.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 1680.00 | 1674.63 | 1682.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 15:15:00 | 1680.00 | 1674.63 | 1682.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1680.00 | 1674.63 | 1682.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1672.20 | 1674.63 | 1682.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1655.90 | 1670.88 | 1680.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 1623.40 | 1656.90 | 1671.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:45:00 | 1637.70 | 1651.40 | 1667.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1542.23 | 1621.99 | 1648.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1555.82 | 1621.99 | 1648.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1600.00 | 1596.16 | 1620.98 | SL hit (close>ema200) qty=0.50 sl=1596.16 alert=retest2 |

### Cycle 158 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1618.90 | 1575.74 | 1575.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 1630.80 | 1586.75 | 1580.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1602.60 | 1607.84 | 1596.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1602.60 | 1607.84 | 1596.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1602.60 | 1607.84 | 1596.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1601.40 | 1607.84 | 1596.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1612.20 | 1608.71 | 1597.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 1618.40 | 1608.71 | 1597.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:30:00 | 1620.50 | 1609.53 | 1601.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1630.50 | 1610.43 | 1602.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 1580.00 | 1604.74 | 1602.14 | SL hit (close<static) qty=1.00 sl=1596.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 1580.00 | 1599.79 | 1600.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 1568.00 | 1587.08 | 1592.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 1569.80 | 1569.08 | 1578.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 1569.80 | 1569.08 | 1578.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1569.80 | 1569.08 | 1578.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 1569.80 | 1569.08 | 1578.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 1568.10 | 1568.88 | 1577.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:15:00 | 1564.20 | 1568.88 | 1577.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1572.80 | 1557.65 | 1557.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 1572.80 | 1557.65 | 1557.54 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 15:15:00 | 1551.70 | 1558.37 | 1558.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 1548.70 | 1555.26 | 1556.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 1549.60 | 1547.11 | 1551.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 1549.60 | 1547.11 | 1551.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1549.60 | 1547.11 | 1551.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:30:00 | 1556.20 | 1547.11 | 1551.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1566.10 | 1550.91 | 1553.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 1564.50 | 1550.91 | 1553.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1564.90 | 1553.70 | 1554.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:15:00 | 1567.50 | 1553.70 | 1554.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1548.40 | 1551.08 | 1552.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1540.30 | 1551.08 | 1552.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1535.80 | 1548.02 | 1551.18 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 1555.00 | 1549.13 | 1548.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1562.00 | 1551.71 | 1549.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 10:15:00 | 1550.80 | 1551.53 | 1550.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 1550.80 | 1551.53 | 1550.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1550.80 | 1551.53 | 1550.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 1547.10 | 1551.53 | 1550.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1549.60 | 1551.14 | 1550.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 1556.00 | 1552.38 | 1550.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 1628.20 | 1649.48 | 1649.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1628.20 | 1649.48 | 1649.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1606.00 | 1634.29 | 1642.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1602.70 | 1601.48 | 1617.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:30:00 | 1604.00 | 1601.48 | 1617.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1610.50 | 1603.35 | 1613.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 1610.50 | 1603.35 | 1613.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1606.80 | 1604.04 | 1612.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 1616.80 | 1604.04 | 1612.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1616.00 | 1606.43 | 1612.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:00:00 | 1605.10 | 1608.08 | 1611.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1634.70 | 1615.82 | 1614.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1634.70 | 1615.82 | 1614.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 1640.10 | 1623.38 | 1618.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 1629.20 | 1631.48 | 1625.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 12:15:00 | 1629.20 | 1631.48 | 1625.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1629.20 | 1631.48 | 1625.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 1631.40 | 1631.48 | 1625.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1650.30 | 1661.93 | 1652.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 1650.30 | 1661.93 | 1652.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1645.00 | 1658.54 | 1651.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:45:00 | 1646.70 | 1658.54 | 1651.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1620.50 | 1650.93 | 1648.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1620.50 | 1650.93 | 1648.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1632.00 | 1647.15 | 1647.22 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 1682.20 | 1649.94 | 1646.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 12:15:00 | 1688.50 | 1661.82 | 1652.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1830.80 | 1832.37 | 1787.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 1830.80 | 1832.37 | 1787.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1805.50 | 1827.31 | 1807.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 1805.50 | 1827.31 | 1807.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1810.00 | 1823.85 | 1807.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:30:00 | 1804.10 | 1823.85 | 1807.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1805.00 | 1818.24 | 1809.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 1806.10 | 1818.24 | 1809.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1805.10 | 1815.61 | 1809.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:45:00 | 1807.30 | 1815.61 | 1809.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1805.80 | 1813.65 | 1809.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 1805.80 | 1813.65 | 1809.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1811.00 | 1813.12 | 1809.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:45:00 | 1812.20 | 1811.85 | 1809.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 1813.90 | 1811.85 | 1809.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 1814.90 | 1811.50 | 1809.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 1813.00 | 1812.61 | 1810.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1808.70 | 1811.83 | 1810.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:30:00 | 1810.60 | 1811.83 | 1810.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1810.10 | 1811.48 | 1810.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:30:00 | 1808.00 | 1811.48 | 1810.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1810.00 | 1811.18 | 1810.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 1803.50 | 1811.18 | 1810.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 1810.00 | 1810.95 | 1810.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 1811.70 | 1810.95 | 1810.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1800.60 | 1808.88 | 1809.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1800.60 | 1808.88 | 1809.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 1799.10 | 1806.92 | 1808.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 12:15:00 | 1800.40 | 1798.34 | 1801.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 12:15:00 | 1800.40 | 1798.34 | 1801.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 1800.40 | 1798.34 | 1801.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 1801.40 | 1798.34 | 1801.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1800.70 | 1798.81 | 1801.78 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 1803.00 | 1802.84 | 1802.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 1825.50 | 1808.18 | 1805.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1860.20 | 1867.56 | 1846.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:45:00 | 1860.70 | 1867.56 | 1846.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1901.10 | 1872.86 | 1858.91 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1840.10 | 1863.94 | 1864.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 1838.00 | 1851.38 | 1857.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 10:15:00 | 1755.20 | 1753.70 | 1773.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 1756.20 | 1753.70 | 1773.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 1769.80 | 1757.55 | 1770.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 1770.40 | 1757.55 | 1770.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1766.80 | 1759.40 | 1770.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 1765.00 | 1759.40 | 1770.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1782.00 | 1764.82 | 1770.83 | SL hit (close>static) qty=1.00 sl=1770.20 alert=retest2 |

### Cycle 170 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1755.30 | 1747.62 | 1747.10 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1739.30 | 1746.37 | 1746.65 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1757.20 | 1746.39 | 1746.13 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 1737.40 | 1744.59 | 1745.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1725.50 | 1737.45 | 1741.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 1718.70 | 1717.16 | 1728.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:45:00 | 1719.40 | 1717.16 | 1728.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1732.50 | 1719.38 | 1726.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1732.50 | 1719.38 | 1726.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1742.90 | 1724.09 | 1728.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1742.90 | 1724.09 | 1728.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1748.00 | 1730.89 | 1730.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 1773.00 | 1739.31 | 1734.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 1747.40 | 1749.34 | 1741.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 14:00:00 | 1747.40 | 1749.34 | 1741.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1731.60 | 1746.42 | 1742.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1731.60 | 1746.42 | 1742.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1736.80 | 1744.50 | 1741.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 1786.30 | 1746.01 | 1743.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 1760.00 | 1777.89 | 1780.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 1760.00 | 1777.89 | 1780.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 1746.20 | 1765.56 | 1773.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 14:15:00 | 1745.40 | 1745.14 | 1758.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:45:00 | 1748.20 | 1745.14 | 1758.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1761.80 | 1749.25 | 1757.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 1768.00 | 1749.25 | 1757.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1760.80 | 1751.56 | 1758.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:00:00 | 1760.80 | 1751.56 | 1758.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1764.30 | 1754.11 | 1758.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:30:00 | 1760.00 | 1754.11 | 1758.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1765.80 | 1756.45 | 1759.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 1765.80 | 1756.45 | 1759.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1758.30 | 1759.24 | 1760.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 1750.50 | 1759.24 | 1760.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 1753.40 | 1758.21 | 1759.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 1770.00 | 1761.18 | 1760.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 11:15:00 | 1770.00 | 1761.18 | 1760.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 12:15:00 | 1791.90 | 1767.32 | 1763.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 1928.80 | 1930.33 | 1901.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:00:00 | 1928.80 | 1930.33 | 1901.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1906.10 | 1925.20 | 1906.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 1907.50 | 1925.20 | 1906.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1906.10 | 1921.38 | 1906.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1907.00 | 1921.38 | 1906.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1897.00 | 1916.50 | 1905.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:30:00 | 1895.10 | 1916.50 | 1905.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1912.70 | 1915.74 | 1906.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 1917.00 | 1915.49 | 1906.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1922.30 | 1910.56 | 1905.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:00:00 | 1914.50 | 1910.65 | 1907.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 1913.10 | 1915.17 | 1911.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1917.00 | 1918.10 | 1914.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 1913.70 | 1918.10 | 1914.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1904.00 | 1915.28 | 1913.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 1904.00 | 1915.28 | 1913.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1904.90 | 1913.20 | 1912.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1888.60 | 1913.20 | 1912.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1886.10 | 1907.78 | 1910.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1886.10 | 1907.78 | 1910.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1863.40 | 1886.06 | 1897.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 1904.10 | 1883.75 | 1892.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 11:15:00 | 1904.10 | 1883.75 | 1892.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1904.10 | 1883.75 | 1892.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 1904.10 | 1883.75 | 1892.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 1901.40 | 1887.28 | 1892.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:45:00 | 1893.10 | 1888.74 | 1893.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:15:00 | 1888.60 | 1888.74 | 1893.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 1890.40 | 1887.89 | 1890.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 1914.30 | 1893.88 | 1891.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1914.30 | 1893.88 | 1891.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 1920.50 | 1899.21 | 1894.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 1953.50 | 1964.31 | 1951.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:00:00 | 1953.50 | 1964.31 | 1951.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 1947.50 | 1960.95 | 1951.16 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1933.50 | 1946.32 | 1946.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 1910.30 | 1936.27 | 1941.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 1892.90 | 1889.77 | 1902.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:15:00 | 1896.80 | 1889.77 | 1902.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1887.20 | 1889.26 | 1900.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 1876.10 | 1885.47 | 1889.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:30:00 | 1879.00 | 1886.07 | 1887.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 1924.10 | 1895.38 | 1891.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1924.10 | 1895.38 | 1891.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 1931.60 | 1907.68 | 1898.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 1911.80 | 1914.86 | 1905.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 10:00:00 | 1911.80 | 1914.86 | 1905.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1943.70 | 1954.14 | 1941.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1943.70 | 1954.14 | 1941.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1952.10 | 1953.26 | 1943.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:30:00 | 1947.40 | 1953.26 | 1943.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1976.90 | 1962.84 | 1951.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 1993.30 | 1962.84 | 1951.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1943.30 | 1963.28 | 1964.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 1943.30 | 1963.28 | 1964.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1913.00 | 1935.85 | 1945.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 1911.00 | 1898.18 | 1909.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 15:15:00 | 1911.00 | 1898.18 | 1909.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1911.00 | 1898.18 | 1909.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1881.00 | 1898.18 | 1909.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 1889.90 | 1886.90 | 1898.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 15:15:00 | 1898.00 | 1880.67 | 1880.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 1898.00 | 1880.67 | 1880.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 10:15:00 | 1923.40 | 1895.48 | 1888.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1892.60 | 1903.38 | 1897.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1892.60 | 1903.38 | 1897.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1892.60 | 1903.38 | 1897.04 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 1881.00 | 1891.97 | 1893.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 1869.50 | 1887.48 | 1891.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1888.00 | 1884.15 | 1888.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 1888.00 | 1884.15 | 1888.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1890.60 | 1885.44 | 1888.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:45:00 | 1890.40 | 1885.44 | 1888.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1896.00 | 1887.55 | 1889.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1924.80 | 1887.55 | 1889.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 1939.30 | 1897.90 | 1893.69 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1892.90 | 1907.63 | 1909.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 1886.00 | 1903.31 | 1906.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1898.90 | 1892.82 | 1899.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1898.90 | 1892.82 | 1899.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1898.90 | 1892.82 | 1899.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 1897.00 | 1892.82 | 1899.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1898.00 | 1893.86 | 1899.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1897.70 | 1893.86 | 1899.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1896.70 | 1894.43 | 1899.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1896.10 | 1894.43 | 1899.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1899.90 | 1895.52 | 1899.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 1899.90 | 1895.52 | 1899.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1896.60 | 1895.74 | 1899.25 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 1920.00 | 1904.18 | 1902.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 1943.50 | 1912.05 | 1906.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 1961.10 | 1961.84 | 1947.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:45:00 | 1964.00 | 1961.84 | 1947.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1959.00 | 1964.53 | 1956.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1959.00 | 1964.53 | 1956.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1969.10 | 1965.45 | 1958.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 1979.60 | 1967.48 | 1960.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 1975.00 | 1971.00 | 1963.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1962.10 | 1983.82 | 1985.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 1962.10 | 1983.82 | 1985.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 1959.00 | 1978.85 | 1983.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 12:15:00 | 1968.00 | 1958.31 | 1966.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 12:15:00 | 1968.00 | 1958.31 | 1966.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1968.00 | 1958.31 | 1966.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 1965.60 | 1958.31 | 1966.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1965.30 | 1959.71 | 1965.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1949.00 | 1960.73 | 1965.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 14:45:00 | 1958.70 | 1956.44 | 1960.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:45:00 | 1954.70 | 1959.27 | 1960.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 1952.00 | 1924.33 | 1930.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1944.00 | 1928.26 | 1931.82 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1851.55 | 1906.17 | 1921.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1860.76 | 1906.17 | 1921.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1856.96 | 1906.17 | 1921.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 1854.40 | 1906.17 | 1921.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 1925.40 | 1906.17 | 1921.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 12:15:00 | 1829.13 | 1892.14 | 1913.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-07 09:15:00 | 1754.10 | 1822.21 | 1870.44 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1800.00 | 1786.80 | 1786.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 12:15:00 | 1805.00 | 1795.52 | 1791.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 1790.50 | 1795.25 | 1791.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 1790.50 | 1795.25 | 1791.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1790.50 | 1795.25 | 1791.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1790.50 | 1795.25 | 1791.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1787.70 | 1793.74 | 1791.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1795.30 | 1793.74 | 1791.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1785.90 | 1792.17 | 1790.95 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1776.00 | 1788.86 | 1789.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1772.70 | 1784.21 | 1787.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1782.90 | 1779.42 | 1783.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1782.90 | 1779.42 | 1783.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1782.90 | 1779.42 | 1783.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 1790.60 | 1779.42 | 1783.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1780.30 | 1779.60 | 1783.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 1785.90 | 1779.60 | 1783.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1779.00 | 1779.48 | 1783.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 1785.10 | 1779.48 | 1783.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1777.70 | 1779.12 | 1782.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:30:00 | 1783.80 | 1779.12 | 1782.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1786.40 | 1780.90 | 1782.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 1786.40 | 1780.90 | 1782.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1787.10 | 1782.14 | 1783.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1779.00 | 1782.14 | 1783.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:45:00 | 1780.00 | 1777.84 | 1779.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 1796.70 | 1780.58 | 1779.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 1796.70 | 1780.58 | 1779.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 1802.30 | 1788.45 | 1784.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1778.50 | 1790.82 | 1787.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1778.50 | 1790.82 | 1787.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1778.50 | 1790.82 | 1787.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 1778.50 | 1790.82 | 1787.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1771.40 | 1786.93 | 1786.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 1769.00 | 1786.93 | 1786.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1779.70 | 1785.49 | 1785.50 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 1790.00 | 1783.52 | 1782.98 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 1765.30 | 1779.87 | 1781.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 1757.70 | 1775.44 | 1779.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1766.70 | 1761.65 | 1769.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1766.70 | 1761.65 | 1769.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1766.70 | 1761.65 | 1769.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 1763.90 | 1761.65 | 1769.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1771.10 | 1763.54 | 1769.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1774.70 | 1763.54 | 1769.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1768.80 | 1764.59 | 1769.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 1757.50 | 1769.72 | 1770.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 1762.00 | 1764.32 | 1766.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 1765.00 | 1761.40 | 1763.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:45:00 | 1766.10 | 1762.34 | 1763.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1766.00 | 1763.50 | 1764.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1771.80 | 1763.50 | 1764.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1754.70 | 1761.77 | 1763.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1771.10 | 1763.54 | 1763.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 1771.10 | 1763.54 | 1763.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 1776.60 | 1767.35 | 1765.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 1762.30 | 1767.31 | 1765.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 11:15:00 | 1762.30 | 1767.31 | 1765.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1762.30 | 1767.31 | 1765.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:45:00 | 1758.80 | 1767.31 | 1765.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1755.30 | 1764.90 | 1764.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:00:00 | 1755.30 | 1764.90 | 1764.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 1754.90 | 1762.90 | 1763.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 14:15:00 | 1743.60 | 1759.04 | 1762.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1759.50 | 1749.42 | 1755.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 1759.50 | 1749.42 | 1755.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1759.50 | 1749.42 | 1755.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 1759.50 | 1749.42 | 1755.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1755.90 | 1750.71 | 1755.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 1750.00 | 1750.71 | 1755.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1755.40 | 1751.65 | 1755.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 1755.40 | 1751.65 | 1755.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 1752.70 | 1751.86 | 1754.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1739.30 | 1751.86 | 1754.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:30:00 | 1749.80 | 1748.32 | 1751.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:45:00 | 1749.20 | 1749.40 | 1751.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1738.00 | 1750.52 | 1752.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1735.60 | 1747.53 | 1750.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:15:00 | 1722.60 | 1735.14 | 1741.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 15:15:00 | 1710.00 | 1723.66 | 1732.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 10:45:00 | 1723.00 | 1722.02 | 1729.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1741.90 | 1734.61 | 1733.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1741.90 | 1734.61 | 1733.71 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 1726.00 | 1732.89 | 1733.01 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1739.80 | 1732.60 | 1732.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1745.50 | 1735.18 | 1733.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1799.50 | 1800.84 | 1785.14 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1806.70 | 1803.31 | 1787.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 1813.20 | 1803.31 | 1787.69 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1805.00 | 1834.96 | 1827.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 1805.00 | 1834.96 | 1827.12 | SL hit (close<ema400) qty=1.00 sl=1827.12 alert=retest1 |

### Cycle 199 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 1773.00 | 1814.42 | 1818.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 1752.10 | 1769.10 | 1785.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 14:15:00 | 1768.60 | 1764.03 | 1777.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 1768.60 | 1764.03 | 1777.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1779.10 | 1767.55 | 1776.52 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 1791.60 | 1777.16 | 1777.15 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1774.50 | 1776.62 | 1776.91 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 1781.60 | 1777.62 | 1777.34 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1773.00 | 1776.70 | 1776.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1763.30 | 1773.31 | 1775.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1722.10 | 1715.83 | 1731.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:00:00 | 1722.10 | 1715.83 | 1731.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1735.00 | 1719.67 | 1731.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1735.00 | 1719.67 | 1731.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1737.90 | 1723.31 | 1732.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 1729.50 | 1727.28 | 1732.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 1748.70 | 1731.92 | 1734.02 | SL hit (close>static) qty=1.00 sl=1739.90 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1774.90 | 1740.52 | 1737.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 1782.90 | 1753.98 | 1744.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 1817.20 | 1835.26 | 1816.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 1817.20 | 1835.26 | 1816.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1817.20 | 1835.26 | 1816.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 1817.20 | 1835.26 | 1816.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1811.20 | 1830.45 | 1816.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 1811.20 | 1830.45 | 1816.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1825.40 | 1829.44 | 1816.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 1839.10 | 1821.19 | 1817.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1832.40 | 1824.44 | 1822.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 1790.80 | 1815.84 | 1818.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 1790.80 | 1815.84 | 1818.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 13:15:00 | 1787.20 | 1801.68 | 1809.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1795.90 | 1786.06 | 1794.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 1795.90 | 1786.06 | 1794.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1795.90 | 1786.06 | 1794.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 1795.90 | 1786.06 | 1794.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1788.70 | 1786.59 | 1794.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1797.90 | 1786.59 | 1794.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1787.00 | 1786.67 | 1793.63 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 1813.70 | 1799.69 | 1798.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1817.10 | 1806.24 | 1801.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1801.70 | 1807.95 | 1804.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1801.70 | 1807.95 | 1804.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1801.70 | 1807.95 | 1804.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 1797.80 | 1807.95 | 1804.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1805.80 | 1807.52 | 1804.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 1810.00 | 1807.52 | 1804.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1800.00 | 1806.02 | 1804.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1814.30 | 1806.02 | 1804.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1809.90 | 1806.79 | 1804.61 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1793.80 | 1802.39 | 1802.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 1776.80 | 1795.43 | 1799.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1730.00 | 1720.80 | 1740.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1730.00 | 1720.80 | 1740.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1730.00 | 1720.80 | 1740.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1733.20 | 1720.80 | 1740.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1735.20 | 1723.68 | 1739.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1721.20 | 1723.18 | 1738.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:00:00 | 1721.60 | 1722.87 | 1736.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 1722.00 | 1722.44 | 1729.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 1721.80 | 1692.69 | 1692.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 1721.80 | 1692.69 | 1692.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 1746.90 | 1703.54 | 1697.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 1793.90 | 1803.75 | 1775.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 13:30:00 | 1793.20 | 1803.75 | 1775.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1795.50 | 1798.82 | 1780.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:30:00 | 1803.50 | 1796.98 | 1782.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:00:00 | 1800.10 | 1796.98 | 1782.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:45:00 | 1802.90 | 1799.78 | 1785.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-10 09:15:00 | 1983.85 | 1927.09 | 1899.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 1976.20 | 1990.97 | 1992.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1965.00 | 1983.06 | 1988.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1986.00 | 1974.33 | 1978.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1986.00 | 1974.33 | 1978.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1986.00 | 1974.33 | 1978.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 1987.00 | 1974.33 | 1978.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1968.70 | 1973.21 | 1977.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1959.80 | 1973.21 | 1977.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 1963.10 | 1970.12 | 1975.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 1963.10 | 1968.92 | 1974.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 1961.30 | 1967.39 | 1973.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1967.00 | 1957.09 | 1962.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1955.40 | 1957.09 | 1962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1974.80 | 1960.63 | 1964.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1974.80 | 1960.63 | 1964.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1965.40 | 1961.58 | 1964.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 1971.10 | 1961.58 | 1964.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1956.90 | 1960.31 | 1963.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 1957.00 | 1960.31 | 1963.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1959.50 | 1951.69 | 1956.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 1959.50 | 1951.69 | 1956.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1950.20 | 1951.39 | 1956.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:45:00 | 1944.30 | 1952.34 | 1955.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:30:00 | 1945.80 | 1952.85 | 1955.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:00:00 | 1946.60 | 1951.60 | 1954.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 1941.30 | 1947.62 | 1951.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1864.94 | 1941.66 | 1948.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1864.94 | 1941.66 | 1948.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1863.23 | 1941.66 | 1948.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1861.81 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1847.08 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1848.51 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1849.27 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1844.23 | 1905.60 | 1923.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1889.10 | 1871.23 | 1891.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 10:15:00 | 1889.10 | 1871.23 | 1891.27 | SL hit (close>ema200) qty=0.50 sl=1871.23 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 1942.90 | 1904.79 | 1902.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 1962.90 | 1924.03 | 1911.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 1916.00 | 1922.42 | 1912.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 10:45:00 | 1918.60 | 1922.42 | 1912.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1914.10 | 1920.76 | 1912.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 1914.10 | 1920.76 | 1912.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1937.00 | 1924.01 | 1914.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 1943.80 | 1924.01 | 1914.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 10:15:00 | 1897.90 | 1927.08 | 1921.51 | SL hit (close<static) qty=1.00 sl=1908.70 alert=retest2 |

### Cycle 211 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 1866.00 | 1914.87 | 1916.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 1862.80 | 1898.08 | 1908.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 1909.20 | 1893.41 | 1902.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1909.20 | 1893.41 | 1902.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1909.20 | 1893.41 | 1902.95 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1971.90 | 1912.01 | 1908.51 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1880.90 | 1919.28 | 1924.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1835.60 | 1893.73 | 1911.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 1802.90 | 1799.27 | 1827.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 1802.90 | 1799.27 | 1827.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1822.80 | 1807.54 | 1824.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1841.60 | 1807.54 | 1824.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1817.90 | 1809.61 | 1824.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 11:45:00 | 1811.60 | 1810.92 | 1822.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:15:00 | 1815.70 | 1812.45 | 1821.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1771.70 | 1815.62 | 1821.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:15:00 | 1721.02 | 1747.58 | 1774.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 10:15:00 | 1724.91 | 1747.58 | 1774.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 1630.44 | 1708.90 | 1742.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 214 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1745.80 | 1682.62 | 1681.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 1753.20 | 1713.58 | 1697.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1699.10 | 1719.93 | 1705.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1699.10 | 1719.93 | 1705.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1699.10 | 1719.93 | 1705.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 1701.70 | 1719.93 | 1705.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1696.40 | 1715.23 | 1704.42 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 1673.00 | 1695.57 | 1698.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1633.80 | 1683.22 | 1692.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1646.20 | 1634.58 | 1656.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1646.20 | 1634.58 | 1656.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1646.20 | 1634.58 | 1656.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:30:00 | 1633.60 | 1623.53 | 1649.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 13:15:00 | 1551.92 | 1588.93 | 1626.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-04-02 09:15:00 | 1470.24 | 1553.29 | 1599.77 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 216 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1611.70 | 1551.34 | 1551.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 1643.00 | 1612.42 | 1589.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 11:15:00 | 1894.20 | 1895.24 | 1869.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 12:15:00 | 1888.80 | 1895.24 | 1869.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1865.60 | 1886.62 | 1870.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 1865.60 | 1886.62 | 1870.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1867.50 | 1882.80 | 1869.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 1864.60 | 1882.80 | 1869.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1872.60 | 1879.51 | 1870.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:45:00 | 1884.20 | 1879.51 | 1870.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 1870.60 | 1877.73 | 1870.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 1891.80 | 1881.40 | 1874.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 14:30:00 | 1894.00 | 1882.08 | 1875.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 15:15:00 | 1895.00 | 1882.08 | 1875.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1854.40 | 1878.61 | 1874.83 | SL hit (close<static) qty=1.00 sl=1870.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1850.00 | 1870.49 | 1871.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 1830.80 | 1859.29 | 1866.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 1826.90 | 1825.51 | 1842.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 1826.90 | 1825.51 | 1842.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1873.50 | 1832.98 | 1841.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1892.30 | 1832.98 | 1841.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1890.00 | 1844.38 | 1845.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1891.00 | 1844.38 | 1845.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1901.70 | 1855.84 | 1850.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 1917.90 | 1883.44 | 1866.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1891.00 | 1891.38 | 1874.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 1891.00 | 1891.38 | 1874.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1886.00 | 1890.45 | 1880.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 1884.00 | 1890.45 | 1880.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1868.80 | 1886.12 | 1879.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 1868.80 | 1886.12 | 1879.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1863.70 | 1881.64 | 1878.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:30:00 | 1861.10 | 1881.64 | 1878.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 1859.40 | 1872.66 | 1874.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 1841.00 | 1863.40 | 1869.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1803.10 | 1796.35 | 1821.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 1790.60 | 1795.30 | 1816.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1790.60 | 1795.30 | 1816.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 1802.20 | 1795.30 | 1816.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1804.80 | 1800.74 | 1811.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1780.60 | 1799.69 | 1806.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 1780.90 | 1792.85 | 1800.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1785.00 | 1798.34 | 1801.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:15:00 | 1785.20 | 1788.57 | 1795.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1752.40 | 1781.33 | 1791.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:15:00 | 1747.00 | 1776.07 | 1787.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1691.57 | 1756.33 | 1775.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1691.86 | 1756.33 | 1775.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1695.75 | 1756.33 | 1775.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1695.94 | 1756.33 | 1775.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-17 11:00:00 | 704.00 | 2023-05-18 10:15:00 | 720.85 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2023-05-17 13:30:00 | 704.75 | 2023-05-18 10:15:00 | 720.85 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2023-05-17 15:00:00 | 705.23 | 2023-05-18 10:15:00 | 720.85 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2023-05-23 12:30:00 | 703.58 | 2023-05-24 09:15:00 | 717.53 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2023-05-29 13:30:00 | 735.00 | 2023-05-31 11:15:00 | 725.85 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-05-30 10:15:00 | 734.15 | 2023-05-31 11:15:00 | 725.85 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-06-02 10:45:00 | 718.50 | 2023-06-06 11:15:00 | 722.43 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-06-02 11:15:00 | 718.75 | 2023-06-06 11:15:00 | 722.43 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2023-06-02 12:30:00 | 718.88 | 2023-06-06 11:15:00 | 722.43 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-06-02 13:00:00 | 719.00 | 2023-06-06 11:15:00 | 722.43 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-06-12 09:15:00 | 732.15 | 2023-06-12 09:15:00 | 726.53 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-06-12 12:15:00 | 733.65 | 2023-06-21 11:15:00 | 807.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-06-30 13:30:00 | 773.35 | 2023-07-04 13:15:00 | 775.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2023-06-30 14:45:00 | 772.70 | 2023-07-04 13:15:00 | 775.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2023-07-03 09:30:00 | 772.50 | 2023-07-04 13:15:00 | 775.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2023-07-03 13:00:00 | 772.85 | 2023-07-05 09:15:00 | 783.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-07-03 15:15:00 | 767.45 | 2023-07-05 09:15:00 | 783.65 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2023-07-04 11:15:00 | 767.70 | 2023-07-05 09:15:00 | 783.65 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2023-07-04 12:00:00 | 767.00 | 2023-07-05 09:15:00 | 783.65 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2023-07-14 15:15:00 | 794.10 | 2023-07-18 10:15:00 | 787.55 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-07-17 10:30:00 | 794.90 | 2023-07-18 10:15:00 | 787.55 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-07-17 12:15:00 | 795.75 | 2023-07-18 10:15:00 | 787.55 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-07-17 14:30:00 | 794.25 | 2023-07-18 10:15:00 | 787.55 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-07-27 10:45:00 | 770.55 | 2023-07-31 13:15:00 | 777.95 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-07-27 15:00:00 | 767.00 | 2023-07-31 13:15:00 | 777.95 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2023-07-28 10:15:00 | 768.50 | 2023-07-31 13:15:00 | 777.95 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-08-02 12:15:00 | 787.50 | 2023-08-02 13:15:00 | 778.60 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-08-14 15:15:00 | 739.90 | 2023-08-17 12:15:00 | 748.45 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2023-08-17 10:00:00 | 742.25 | 2023-08-17 12:15:00 | 748.45 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-08-17 11:00:00 | 742.00 | 2023-08-17 12:15:00 | 748.45 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-08-23 14:00:00 | 736.15 | 2023-08-29 10:15:00 | 729.95 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2023-08-24 09:45:00 | 732.90 | 2023-08-29 10:15:00 | 729.95 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2023-09-05 09:15:00 | 745.60 | 2023-09-12 14:15:00 | 782.05 | STOP_HIT | 1.00 | 4.89% |
| BUY | retest2 | 2023-09-05 11:45:00 | 744.10 | 2023-09-12 14:15:00 | 782.05 | STOP_HIT | 1.00 | 5.10% |
| BUY | retest2 | 2023-09-06 09:15:00 | 745.30 | 2023-09-12 14:15:00 | 782.05 | STOP_HIT | 1.00 | 4.93% |
| BUY | retest2 | 2023-09-20 09:15:00 | 874.90 | 2023-09-27 14:15:00 | 887.75 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2023-10-03 15:15:00 | 882.00 | 2023-10-09 10:15:00 | 889.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-10-04 12:30:00 | 881.30 | 2023-10-09 10:15:00 | 889.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-10-04 14:15:00 | 882.85 | 2023-10-09 10:15:00 | 889.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2023-10-05 10:15:00 | 882.55 | 2023-10-09 11:15:00 | 890.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-10-05 13:15:00 | 875.15 | 2023-10-09 11:15:00 | 890.50 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2023-10-06 09:30:00 | 878.00 | 2023-10-09 11:15:00 | 890.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-10-06 13:15:00 | 874.15 | 2023-10-09 11:15:00 | 890.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2023-10-26 09:15:00 | 873.00 | 2023-10-31 09:15:00 | 927.00 | STOP_HIT | 1.00 | -6.19% |
| BUY | retest2 | 2023-11-20 10:00:00 | 989.60 | 2023-11-22 12:15:00 | 968.90 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2023-11-21 09:15:00 | 990.95 | 2023-11-22 12:15:00 | 968.90 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2023-11-21 13:15:00 | 988.90 | 2023-11-22 12:15:00 | 968.90 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2023-11-21 14:30:00 | 988.80 | 2023-11-22 12:15:00 | 968.90 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2023-11-29 09:15:00 | 1006.55 | 2023-11-29 13:15:00 | 987.25 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2023-12-12 12:15:00 | 1004.95 | 2023-12-15 14:15:00 | 954.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-14 09:15:00 | 1002.80 | 2023-12-19 10:15:00 | 952.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-12 12:15:00 | 1004.95 | 2023-12-21 09:15:00 | 904.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-12-14 09:15:00 | 1002.80 | 2023-12-21 09:15:00 | 902.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-05 14:15:00 | 935.20 | 2024-01-08 09:15:00 | 949.60 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-01-10 14:45:00 | 970.45 | 2024-01-15 13:15:00 | 1067.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 14:00:00 | 1127.95 | 2024-02-06 14:15:00 | 1161.35 | STOP_HIT | 1.00 | 2.96% |
| BUY | retest2 | 2024-02-23 09:15:00 | 1297.20 | 2024-02-26 14:15:00 | 1254.85 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-03-05 10:45:00 | 1330.00 | 2024-03-05 14:15:00 | 1289.10 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-03-22 09:15:00 | 1242.00 | 2024-03-26 12:15:00 | 1269.50 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-03-22 12:00:00 | 1244.55 | 2024-03-26 12:15:00 | 1269.50 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-03-22 15:15:00 | 1237.90 | 2024-03-26 12:15:00 | 1269.50 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-03-26 09:30:00 | 1242.20 | 2024-03-26 12:15:00 | 1269.50 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-04-01 09:15:00 | 1288.05 | 2024-04-08 09:15:00 | 1416.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-01 11:45:00 | 1278.05 | 2024-04-08 09:15:00 | 1405.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-01 12:30:00 | 1281.80 | 2024-04-08 09:15:00 | 1409.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-01 15:00:00 | 1278.30 | 2024-04-08 09:15:00 | 1406.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-02 09:15:00 | 1308.25 | 2024-04-08 10:15:00 | 1439.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 11:45:00 | 1402.40 | 2024-05-03 10:15:00 | 1436.95 | STOP_HIT | 1.00 | 2.46% |
| SELL | retest2 | 2024-05-06 14:00:00 | 1451.70 | 2024-05-07 09:15:00 | 1472.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-05-07 09:15:00 | 1449.55 | 2024-05-07 09:15:00 | 1472.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-05-28 11:30:00 | 1487.05 | 2024-05-28 14:15:00 | 1516.70 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-06-04 13:30:00 | 1585.10 | 2024-06-05 11:15:00 | 1534.30 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-06-06 13:00:00 | 1524.40 | 2024-06-07 11:15:00 | 1567.55 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-06-06 14:00:00 | 1526.05 | 2024-06-07 11:15:00 | 1567.55 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-06-13 12:15:00 | 1673.00 | 2024-06-19 15:15:00 | 1700.00 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2024-06-24 12:00:00 | 1708.30 | 2024-06-24 12:15:00 | 1709.85 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-07-01 13:30:00 | 1645.00 | 2024-07-05 09:15:00 | 1657.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-07-09 11:45:00 | 1704.00 | 2024-07-09 15:15:00 | 1874.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-16 11:15:00 | 1694.00 | 2024-07-16 12:15:00 | 1709.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1671.85 | 2024-07-23 12:15:00 | 1588.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1671.85 | 2024-07-23 13:15:00 | 1646.05 | STOP_HIT | 0.50 | 1.54% |
| BUY | retest2 | 2024-07-26 10:30:00 | 1682.95 | 2024-08-02 13:15:00 | 1719.00 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2024-08-12 09:15:00 | 1589.90 | 2024-08-13 09:15:00 | 1702.95 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2024-08-12 11:15:00 | 1586.45 | 2024-08-13 09:15:00 | 1702.95 | STOP_HIT | 1.00 | -7.34% |
| BUY | retest2 | 2024-08-19 10:00:00 | 1721.05 | 2024-08-20 11:15:00 | 1692.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-08-19 11:00:00 | 1722.25 | 2024-08-20 11:15:00 | 1692.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-08-20 09:45:00 | 1721.80 | 2024-08-20 11:15:00 | 1692.10 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-08-20 10:30:00 | 1726.00 | 2024-08-20 11:15:00 | 1692.10 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-09-12 09:30:00 | 1800.65 | 2024-09-16 09:15:00 | 1980.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-14 09:15:00 | 2038.30 | 2024-10-16 14:15:00 | 2011.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-10-14 10:00:00 | 2024.85 | 2024-10-16 14:15:00 | 2011.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-10-14 15:00:00 | 2024.00 | 2024-10-16 14:15:00 | 2011.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-10-15 09:30:00 | 2026.00 | 2024-10-16 14:15:00 | 2011.90 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-25 10:30:00 | 1835.65 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-10-28 09:30:00 | 1829.40 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-10-28 10:00:00 | 1829.80 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-28 14:15:00 | 1832.00 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-10-30 10:00:00 | 1836.05 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-10-31 10:00:00 | 1839.60 | 2024-10-31 13:15:00 | 1844.20 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-11-04 11:15:00 | 1838.00 | 2024-11-04 12:15:00 | 1823.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-11-08 12:30:00 | 1809.95 | 2024-11-14 11:15:00 | 1814.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-11-08 13:30:00 | 1806.75 | 2024-11-14 11:15:00 | 1814.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-12-09 12:00:00 | 2086.70 | 2024-12-12 11:15:00 | 2090.25 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest1 | 2024-12-09 14:15:00 | 2092.00 | 2024-12-12 11:15:00 | 2090.25 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-12-11 10:30:00 | 2118.00 | 2024-12-12 14:15:00 | 2048.70 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-12-20 13:00:00 | 2024.15 | 2024-12-26 14:15:00 | 2033.45 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-01-01 10:15:00 | 2168.00 | 2025-01-01 14:15:00 | 2384.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 1850.00 | 2025-01-23 10:15:00 | 1922.70 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2025-02-01 09:15:00 | 1866.10 | 2025-02-01 14:15:00 | 2052.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-03 10:15:00 | 1947.80 | 2025-03-04 09:15:00 | 2142.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-17 15:00:00 | 2010.80 | 2025-03-18 09:15:00 | 2099.55 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-03-24 09:15:00 | 2219.65 | 2025-03-26 10:15:00 | 2191.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-04-04 09:30:00 | 2066.20 | 2025-04-07 09:15:00 | 1859.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 13:00:00 | 2030.60 | 2025-04-23 10:15:00 | 1949.90 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-05-08 13:00:00 | 1623.40 | 2025-05-09 09:15:00 | 1542.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:45:00 | 1637.70 | 2025-05-09 09:15:00 | 1555.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 1623.40 | 2025-05-09 15:15:00 | 1600.00 | STOP_HIT | 0.50 | 1.44% |
| SELL | retest2 | 2025-05-08 13:45:00 | 1637.70 | 2025-05-09 15:15:00 | 1600.00 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2025-05-12 10:00:00 | 1631.20 | 2025-05-16 09:15:00 | 1549.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-12 10:00:00 | 1631.20 | 2025-05-16 12:15:00 | 1569.80 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-05-12 10:30:00 | 1626.80 | 2025-05-19 09:15:00 | 1618.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-05-13 10:15:00 | 1618.00 | 2025-05-19 09:15:00 | 1618.90 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-05-20 11:15:00 | 1618.40 | 2025-05-21 11:15:00 | 1580.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-05-20 14:30:00 | 1620.50 | 2025-05-21 11:15:00 | 1580.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-05-21 09:15:00 | 1630.50 | 2025-05-21 11:15:00 | 1580.00 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-05-23 13:15:00 | 1564.20 | 2025-05-29 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-05 13:30:00 | 1556.00 | 2025-06-18 12:15:00 | 1628.20 | STOP_HIT | 1.00 | 4.64% |
| SELL | retest2 | 2025-06-23 13:00:00 | 1605.10 | 2025-06-24 10:15:00 | 1634.70 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-08 13:45:00 | 1812.20 | 2025-07-10 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-07-08 14:15:00 | 1813.90 | 2025-07-10 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-08 15:15:00 | 1814.90 | 2025-07-10 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-09 11:30:00 | 1813.00 | 2025-07-10 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-24 15:15:00 | 1765.00 | 2025-07-25 09:15:00 | 1782.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-25 12:15:00 | 1765.00 | 2025-07-30 14:15:00 | 1755.30 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-08-06 15:15:00 | 1786.30 | 2025-08-11 12:15:00 | 1760.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-08-13 15:15:00 | 1750.50 | 2025-08-14 11:15:00 | 1770.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-14 09:45:00 | 1753.40 | 2025-08-14 11:15:00 | 1770.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-21 13:30:00 | 1917.00 | 2025-08-26 09:15:00 | 1886.10 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-22 09:15:00 | 1922.30 | 2025-08-26 09:15:00 | 1886.10 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-08-22 13:00:00 | 1914.50 | 2025-08-26 09:15:00 | 1886.10 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-25 10:00:00 | 1913.10 | 2025-08-26 09:15:00 | 1886.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-28 13:45:00 | 1893.10 | 2025-09-01 10:15:00 | 1914.30 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-08-28 14:15:00 | 1888.60 | 2025-09-01 10:15:00 | 1914.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-08-29 12:15:00 | 1890.40 | 2025-09-01 10:15:00 | 1914.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-09-12 10:45:00 | 1876.10 | 2025-09-16 09:15:00 | 1924.10 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-09-15 12:30:00 | 1879.00 | 2025-09-16 09:15:00 | 1924.10 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-09-22 10:15:00 | 1993.30 | 2025-09-24 09:15:00 | 1943.30 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1881.00 | 2025-10-03 15:15:00 | 1898.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-30 14:45:00 | 1889.90 | 2025-10-03 15:15:00 | 1898.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-10-23 12:15:00 | 1979.60 | 2025-10-28 09:15:00 | 1962.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-23 15:15:00 | 1975.00 | 2025-10-28 09:15:00 | 1962.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1949.00 | 2025-11-06 11:15:00 | 1851.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 14:45:00 | 1958.70 | 2025-11-06 11:15:00 | 1860.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:45:00 | 1954.70 | 2025-11-06 11:15:00 | 1856.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:00:00 | 1952.00 | 2025-11-06 11:15:00 | 1854.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 11:30:00 | 1925.40 | 2025-11-06 12:15:00 | 1829.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1949.00 | 2025-11-07 09:15:00 | 1754.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-30 14:45:00 | 1958.70 | 2025-11-07 09:15:00 | 1762.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-31 10:45:00 | 1954.70 | 2025-11-07 09:15:00 | 1759.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 10:00:00 | 1952.00 | 2025-11-07 09:15:00 | 1756.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 11:30:00 | 1925.40 | 2025-11-07 09:15:00 | 1732.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1779.00 | 2025-11-19 14:15:00 | 1796.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-11-18 13:45:00 | 1780.00 | 2025-11-19 14:15:00 | 1796.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-27 09:15:00 | 1757.50 | 2025-12-01 14:15:00 | 1771.10 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-27 14:15:00 | 1762.00 | 2025-12-01 14:15:00 | 1771.10 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-28 13:00:00 | 1765.00 | 2025-12-01 14:15:00 | 1771.10 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-11-28 13:45:00 | 1766.10 | 2025-12-01 14:15:00 | 1771.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-12-04 09:15:00 | 1739.30 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-12-04 13:30:00 | 1749.80 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-12-04 14:45:00 | 1749.20 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1738.00 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-12-08 10:15:00 | 1722.60 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-08 15:15:00 | 1710.00 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-09 10:45:00 | 1723.00 | 2025-12-09 14:15:00 | 1741.90 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest1 | 2025-12-16 10:30:00 | 1806.70 | 2025-12-19 10:15:00 | 1805.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest1 | 2025-12-16 11:00:00 | 1813.20 | 2025-12-19 10:15:00 | 1805.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-12-31 15:15:00 | 1729.50 | 2026-01-01 09:15:00 | 1748.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-01-07 14:30:00 | 1839.10 | 2026-01-09 13:15:00 | 1790.80 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1832.40 | 2026-01-09 13:15:00 | 1790.80 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1721.20 | 2026-01-29 12:15:00 | 1721.80 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-01-22 13:00:00 | 1721.60 | 2026-01-29 12:15:00 | 1721.80 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-01-23 12:00:00 | 1722.00 | 2026-01-29 12:15:00 | 1721.80 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-02-02 11:30:00 | 1803.50 | 2026-02-10 09:15:00 | 1983.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 12:00:00 | 1800.10 | 2026-02-10 09:15:00 | 1980.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 12:45:00 | 1802.90 | 2026-02-10 09:15:00 | 1983.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1959.80 | 2026-03-02 09:15:00 | 1864.94 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2026-02-23 13:15:00 | 1963.10 | 2026-03-02 09:15:00 | 1864.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 14:15:00 | 1963.10 | 2026-03-02 09:15:00 | 1863.23 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2026-02-23 15:00:00 | 1961.30 | 2026-03-04 09:15:00 | 1861.81 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-02-27 09:45:00 | 1944.30 | 2026-03-04 09:15:00 | 1847.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:30:00 | 1945.80 | 2026-03-04 09:15:00 | 1848.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 12:00:00 | 1946.60 | 2026-03-04 09:15:00 | 1849.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1941.30 | 2026-03-04 09:15:00 | 1844.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1959.80 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2026-02-23 13:15:00 | 1963.10 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-02-23 14:15:00 | 1963.10 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-02-23 15:00:00 | 1961.30 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2026-02-27 09:45:00 | 1944.30 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-02-27 10:30:00 | 1945.80 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-02-27 12:00:00 | 1946.60 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2026-02-27 15:00:00 | 1941.30 | 2026-03-05 10:15:00 | 1889.10 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest2 | 2026-03-06 13:15:00 | 1943.80 | 2026-03-09 10:15:00 | 1897.90 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-03-18 11:45:00 | 1811.60 | 2026-03-20 10:15:00 | 1721.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 14:15:00 | 1815.70 | 2026-03-20 10:15:00 | 1724.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 11:45:00 | 1811.60 | 2026-03-23 09:15:00 | 1630.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-18 14:15:00 | 1815.70 | 2026-03-23 09:15:00 | 1634.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1771.70 | 2026-03-23 09:15:00 | 1683.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1771.70 | 2026-03-24 10:15:00 | 1638.60 | STOP_HIT | 0.50 | 7.51% |
| SELL | retest2 | 2026-04-01 10:30:00 | 1633.60 | 2026-04-01 13:15:00 | 1551.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:30:00 | 1633.60 | 2026-04-02 09:15:00 | 1470.24 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-22 14:00:00 | 1891.80 | 2026-04-23 09:15:00 | 1854.40 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-04-22 14:30:00 | 1894.00 | 2026-04-23 09:15:00 | 1854.40 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-04-22 15:15:00 | 1895.00 | 2026-04-23 09:15:00 | 1854.40 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-05-06 09:15:00 | 1780.60 | 2026-05-08 09:15:00 | 1691.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-06 12:30:00 | 1780.90 | 2026-05-08 09:15:00 | 1691.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-07 09:15:00 | 1785.00 | 2026-05-08 09:15:00 | 1695.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-07 12:15:00 | 1785.20 | 2026-05-08 09:15:00 | 1695.94 | PARTIAL | 0.50 | 5.00% |
