# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 1952.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 221 |
| ALERT1 | 145 |
| ALERT2 | 144 |
| ALERT2_SKIP | 71 |
| ALERT3 | 400 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 11 |
| ENTRY2 | 148 |
| PARTIAL | 25 |
| TARGET_HIT | 20 |
| STOP_HIT | 139 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 184 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 72 / 112
- **Target hits / Stop hits / Partials:** 20 / 139 / 25
- **Avg / median % per leg:** 0.95% / -0.59%
- **Sum % (uncompounded):** 175.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 22 | 38.6% | 11 | 46 | 0 | 1.01% | 57.8% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.47% | -8.8% |
| BUY @ 3rd Alert (retest2) | 51 | 22 | 43.1% | 11 | 40 | 0 | 1.31% | 66.6% |
| SELL (all) | 127 | 50 | 39.4% | 9 | 93 | 25 | 0.92% | 117.4% |
| SELL @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 1 | 4 | 2 | 1.87% | 13.1% |
| SELL @ 3rd Alert (retest2) | 120 | 46 | 38.3% | 8 | 89 | 23 | 0.87% | 104.3% |
| retest1 (combined) | 13 | 4 | 30.8% | 1 | 10 | 2 | 0.33% | 4.3% |
| retest2 (combined) | 171 | 68 | 39.8% | 19 | 129 | 23 | 1.00% | 170.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 715.10 | 698.95 | 696.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 722.50 | 705.86 | 700.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 14:15:00 | 710.98 | 712.79 | 706.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-16 15:00:00 | 710.98 | 712.79 | 706.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 09:15:00 | 713.95 | 713.22 | 708.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 09:45:00 | 706.13 | 713.22 | 708.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 709.08 | 712.39 | 708.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 11:00:00 | 709.08 | 712.39 | 708.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 11:15:00 | 705.43 | 711.00 | 707.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 11:45:00 | 706.58 | 711.00 | 707.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 12:15:00 | 702.78 | 709.35 | 707.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-17 13:30:00 | 708.50 | 708.29 | 707.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-17 14:15:00 | 705.63 | 708.29 | 707.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 13:15:00 | 710.88 | 713.87 | 714.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 13:15:00 | 710.88 | 713.87 | 714.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 11:15:00 | 708.18 | 712.00 | 713.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 12:15:00 | 709.08 | 708.63 | 710.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 12:15:00 | 709.08 | 708.63 | 710.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 12:15:00 | 709.08 | 708.63 | 710.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 12:45:00 | 709.43 | 708.63 | 710.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 706.40 | 707.24 | 709.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 711.58 | 707.24 | 709.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 704.43 | 706.68 | 708.71 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 09:15:00 | 715.40 | 709.70 | 709.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 10:15:00 | 720.48 | 716.86 | 714.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 14:15:00 | 718.68 | 718.85 | 716.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 15:00:00 | 718.68 | 718.85 | 716.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 719.50 | 718.98 | 716.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 09:30:00 | 721.50 | 719.18 | 716.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 10:15:00 | 720.70 | 719.18 | 716.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 10:45:00 | 744.70 | 724.11 | 719.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 11:15:00 | 753.38 | 756.53 | 756.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 11:15:00 | 753.38 | 756.53 | 756.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 13:15:00 | 745.50 | 753.63 | 755.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 760.00 | 753.55 | 754.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 760.00 | 753.55 | 754.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 760.00 | 753.55 | 754.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:45:00 | 761.50 | 753.55 | 754.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 757.25 | 754.29 | 755.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:30:00 | 760.80 | 754.29 | 755.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 756.48 | 755.24 | 755.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 12:30:00 | 757.50 | 755.24 | 755.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 14:15:00 | 757.28 | 755.80 | 755.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 15:15:00 | 761.00 | 756.84 | 756.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 10:15:00 | 756.30 | 757.23 | 756.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 10:15:00 | 756.30 | 757.23 | 756.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 756.30 | 757.23 | 756.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:45:00 | 754.95 | 757.23 | 756.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 11:15:00 | 754.63 | 756.71 | 756.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 12:00:00 | 754.63 | 756.71 | 756.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2023-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 12:15:00 | 752.90 | 755.95 | 755.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-13 13:15:00 | 749.95 | 754.75 | 755.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 10:15:00 | 757.73 | 753.48 | 754.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 10:15:00 | 757.73 | 753.48 | 754.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 10:15:00 | 757.73 | 753.48 | 754.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:30:00 | 756.93 | 753.48 | 754.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 11:15:00 | 754.63 | 753.71 | 754.40 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 14:15:00 | 756.65 | 755.06 | 754.93 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 11:15:00 | 751.65 | 754.66 | 754.86 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 09:15:00 | 776.50 | 758.82 | 756.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 785.15 | 774.38 | 767.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 781.90 | 782.93 | 776.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-20 09:30:00 | 779.05 | 782.93 | 776.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 820.35 | 832.62 | 813.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 14:30:00 | 822.50 | 832.62 | 813.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 808.00 | 824.80 | 816.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 808.00 | 824.80 | 816.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 811.50 | 822.14 | 816.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:15:00 | 807.13 | 822.14 | 816.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 810.65 | 819.84 | 815.70 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 796.73 | 810.61 | 812.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 10:15:00 | 790.20 | 802.78 | 807.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 10:15:00 | 793.00 | 792.65 | 798.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-27 11:00:00 | 793.00 | 792.65 | 798.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 801.60 | 791.74 | 795.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:45:00 | 802.58 | 791.74 | 795.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 10:15:00 | 806.20 | 794.63 | 796.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-28 10:45:00 | 804.40 | 794.63 | 796.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 11:15:00 | 810.25 | 797.76 | 797.40 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2023-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 13:15:00 | 797.33 | 803.79 | 804.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 11:15:00 | 791.85 | 798.50 | 801.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 799.05 | 794.48 | 797.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 799.05 | 794.48 | 797.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 799.05 | 794.48 | 797.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:30:00 | 800.00 | 794.48 | 797.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 796.98 | 794.98 | 797.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-05 13:30:00 | 794.83 | 796.51 | 797.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 12:30:00 | 795.68 | 797.25 | 797.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-06 13:00:00 | 795.05 | 797.25 | 797.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 09:30:00 | 795.65 | 794.83 | 796.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 783.53 | 780.87 | 785.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 15:00:00 | 783.53 | 780.87 | 785.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 796.40 | 784.08 | 786.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 796.40 | 784.08 | 786.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-11 10:15:00 | 804.48 | 788.16 | 787.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 804.48 | 788.16 | 787.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 11:15:00 | 812.50 | 793.03 | 789.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 12:15:00 | 804.23 | 805.54 | 799.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 12:15:00 | 804.23 | 805.54 | 799.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 804.23 | 805.54 | 799.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 12:45:00 | 801.38 | 805.54 | 799.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 13:15:00 | 798.83 | 804.20 | 799.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:15:00 | 790.00 | 804.20 | 799.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 785.00 | 800.36 | 798.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 15:00:00 | 785.00 | 800.36 | 798.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 785.00 | 797.29 | 797.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 795.78 | 797.29 | 797.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 10:15:00 | 796.03 | 796.85 | 796.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 10:15:00 | 796.03 | 796.85 | 796.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 13:15:00 | 785.08 | 794.17 | 795.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 792.15 | 791.13 | 793.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 09:15:00 | 792.15 | 791.13 | 793.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 792.15 | 791.13 | 793.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:30:00 | 793.00 | 791.13 | 793.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 791.60 | 791.23 | 793.45 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 14:15:00 | 795.58 | 794.66 | 794.65 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 15:15:00 | 793.10 | 794.35 | 794.51 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 812.90 | 798.06 | 796.18 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 793.35 | 798.83 | 799.17 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 799.85 | 799.13 | 799.13 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 12:15:00 | 798.90 | 799.08 | 799.11 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 801.00 | 799.47 | 799.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 14:15:00 | 805.35 | 800.64 | 799.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 13:15:00 | 802.55 | 804.33 | 802.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 13:15:00 | 802.55 | 804.33 | 802.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 802.55 | 804.33 | 802.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:00:00 | 802.55 | 804.33 | 802.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 803.85 | 804.24 | 802.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 15:00:00 | 803.85 | 804.24 | 802.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 803.50 | 804.09 | 802.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 806.43 | 804.09 | 802.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 811.00 | 805.47 | 803.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 11:00:00 | 820.00 | 808.38 | 804.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-25 09:15:00 | 902.00 | 877.86 | 852.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 09:15:00 | 971.00 | 1002.89 | 1006.61 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 09:15:00 | 1036.70 | 1003.13 | 1002.78 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 993.95 | 1012.70 | 1012.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 979.48 | 999.02 | 1004.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 10:15:00 | 972.65 | 969.65 | 981.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 10:15:00 | 972.65 | 969.65 | 981.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 972.65 | 969.65 | 981.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 10:45:00 | 974.50 | 969.65 | 981.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 972.50 | 966.25 | 973.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:45:00 | 977.03 | 966.25 | 973.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 997.45 | 972.49 | 976.06 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 12:15:00 | 993.73 | 980.34 | 979.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 13:15:00 | 1001.65 | 984.60 | 981.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 12:15:00 | 999.33 | 1001.12 | 993.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-18 13:00:00 | 999.33 | 1001.12 | 993.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1073.20 | 1085.98 | 1071.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:45:00 | 1074.93 | 1085.98 | 1071.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 1064.93 | 1081.77 | 1071.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:30:00 | 1058.47 | 1081.77 | 1071.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 11:15:00 | 1062.55 | 1077.93 | 1070.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:15:00 | 1061.50 | 1077.93 | 1070.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 14:15:00 | 1053.50 | 1066.55 | 1066.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 15:15:00 | 1048.25 | 1062.89 | 1064.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 1069.50 | 1064.21 | 1065.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 1069.50 | 1064.21 | 1065.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1069.50 | 1064.21 | 1065.32 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 1080.53 | 1068.74 | 1067.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 13:15:00 | 1092.00 | 1075.78 | 1070.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 15:15:00 | 1241.50 | 1243.09 | 1222.98 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 09:15:00 | 1259.53 | 1243.09 | 1222.98 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 11:15:00 | 1249.97 | 1245.17 | 1227.48 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 15:00:00 | 1256.47 | 1246.01 | 1233.37 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 11:15:00 | 1251.90 | 1248.41 | 1237.80 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 1246.35 | 1249.62 | 1244.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 1242.65 | 1249.62 | 1244.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 1246.63 | 1249.02 | 1244.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 14:15:00 | 1249.15 | 1248.65 | 1244.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 15:15:00 | 1244.50 | 1247.63 | 1244.88 | SL hit (close<ema400) qty=1.00 sl=1244.88 alert=retest1 |

### Cycle 28 — SELL (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 09:15:00 | 1218.00 | 1241.71 | 1242.44 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 12:15:00 | 1252.50 | 1236.55 | 1236.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 13:15:00 | 1296.58 | 1248.55 | 1241.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 14:15:00 | 1298.50 | 1302.58 | 1280.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-11 15:00:00 | 1298.50 | 1302.58 | 1280.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1227.10 | 1287.31 | 1277.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 1224.10 | 1287.31 | 1277.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 1245.03 | 1278.85 | 1274.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 1243.50 | 1278.85 | 1274.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1237.47 | 1270.58 | 1270.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 1164.63 | 1249.39 | 1261.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 1217.30 | 1197.40 | 1217.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 14:15:00 | 1217.30 | 1197.40 | 1217.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 1217.30 | 1197.40 | 1217.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 1217.30 | 1197.40 | 1217.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 1220.33 | 1201.99 | 1217.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 1202.50 | 1201.99 | 1217.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1193.50 | 1200.29 | 1215.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 11:00:00 | 1189.75 | 1198.18 | 1213.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:30:00 | 1190.00 | 1195.99 | 1209.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 09:30:00 | 1188.28 | 1193.55 | 1203.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 13:30:00 | 1189.25 | 1196.46 | 1202.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 1205.38 | 1198.24 | 1202.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 14:45:00 | 1203.55 | 1198.24 | 1202.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 1199.00 | 1198.39 | 1202.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-18 09:15:00 | 1206.90 | 1198.39 | 1202.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 1191.80 | 1197.08 | 1201.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 10:15:00 | 1189.45 | 1197.08 | 1201.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-18 13:45:00 | 1189.80 | 1192.40 | 1197.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 09:45:00 | 1182.55 | 1179.56 | 1180.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-22 10:15:00 | 1194.47 | 1182.54 | 1181.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 10:15:00 | 1194.47 | 1182.54 | 1181.57 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 15:15:00 | 1172.22 | 1181.70 | 1182.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 09:15:00 | 1150.65 | 1175.49 | 1179.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 09:15:00 | 1159.68 | 1158.84 | 1166.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 10:15:00 | 1171.63 | 1161.40 | 1167.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 1171.63 | 1161.40 | 1167.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 11:00:00 | 1171.63 | 1161.40 | 1167.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 1165.22 | 1162.16 | 1167.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:45:00 | 1160.10 | 1162.46 | 1166.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 13:15:00 | 1162.58 | 1162.46 | 1166.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 10:30:00 | 1159.97 | 1165.54 | 1167.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 13:15:00 | 1161.00 | 1165.02 | 1166.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 1164.80 | 1162.15 | 1164.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:45:00 | 1168.85 | 1162.15 | 1164.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 1179.33 | 1165.58 | 1165.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-28 10:15:00 | 1179.33 | 1165.58 | 1165.75 | SL hit (close>static) qty=1.00 sl=1172.35 alert=retest2 |

### Cycle 33 — BUY (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 11:15:00 | 1176.13 | 1167.69 | 1166.70 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 1159.72 | 1165.52 | 1165.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 15:15:00 | 1157.53 | 1163.92 | 1165.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 1172.78 | 1165.69 | 1165.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 1172.78 | 1165.69 | 1165.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 1172.78 | 1165.69 | 1165.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:45:00 | 1173.65 | 1165.69 | 1165.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 1162.25 | 1165.00 | 1165.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 14:00:00 | 1159.25 | 1162.94 | 1164.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 10:30:00 | 1154.68 | 1160.37 | 1162.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 10:15:00 | 1174.00 | 1163.48 | 1162.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 10:15:00 | 1174.00 | 1163.48 | 1162.59 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 14:15:00 | 1158.50 | 1162.38 | 1162.41 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 15:15:00 | 1163.93 | 1162.69 | 1162.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 09:15:00 | 1169.35 | 1164.02 | 1163.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 11:15:00 | 1162.50 | 1163.79 | 1163.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 11:15:00 | 1162.50 | 1163.79 | 1163.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 11:15:00 | 1162.50 | 1163.79 | 1163.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 11:30:00 | 1162.50 | 1163.79 | 1163.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 12:15:00 | 1161.85 | 1163.41 | 1163.09 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 14:15:00 | 1156.83 | 1161.94 | 1162.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 09:15:00 | 1151.97 | 1159.24 | 1161.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 14:15:00 | 1166.95 | 1156.01 | 1158.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 14:15:00 | 1166.95 | 1156.01 | 1158.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 1166.95 | 1156.01 | 1158.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 15:00:00 | 1166.95 | 1156.01 | 1158.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 1167.50 | 1158.31 | 1159.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 09:15:00 | 1167.68 | 1158.31 | 1159.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 09:15:00 | 1175.90 | 1161.83 | 1160.57 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 1146.53 | 1160.01 | 1160.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 1140.75 | 1153.97 | 1157.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1167.85 | 1156.74 | 1158.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1167.85 | 1156.74 | 1158.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1167.85 | 1156.74 | 1158.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:00:00 | 1167.85 | 1156.74 | 1158.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 1170.50 | 1159.49 | 1159.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 10:15:00 | 1177.00 | 1166.83 | 1163.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 1165.40 | 1168.08 | 1164.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 13:15:00 | 1165.40 | 1168.08 | 1164.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 1165.40 | 1168.08 | 1164.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:45:00 | 1166.15 | 1168.08 | 1164.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 1166.43 | 1167.75 | 1165.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 15:00:00 | 1166.43 | 1167.75 | 1165.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 1165.55 | 1167.31 | 1165.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 1174.97 | 1167.31 | 1165.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 10:00:00 | 1169.38 | 1167.72 | 1165.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 1163.58 | 1166.85 | 1165.85 | SL hit (close<static) qty=1.00 sl=1165.03 alert=retest2 |

### Cycle 42 — SELL (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 09:15:00 | 1152.80 | 1163.45 | 1164.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 15:15:00 | 1145.00 | 1154.03 | 1158.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 1169.90 | 1157.21 | 1159.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 09:15:00 | 1169.90 | 1157.21 | 1159.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 1169.90 | 1157.21 | 1159.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 10:15:00 | 1179.95 | 1157.21 | 1159.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 10:15:00 | 1190.00 | 1163.76 | 1162.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 14:15:00 | 1207.05 | 1183.64 | 1173.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 13:15:00 | 1194.47 | 1196.52 | 1185.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-17 14:00:00 | 1194.47 | 1196.52 | 1185.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 1173.83 | 1191.52 | 1186.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 1173.83 | 1191.52 | 1186.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 1171.97 | 1187.61 | 1185.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:45:00 | 1171.13 | 1187.61 | 1185.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 1165.13 | 1183.11 | 1183.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 1154.00 | 1174.57 | 1179.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 10:15:00 | 1170.95 | 1169.47 | 1175.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 10:15:00 | 1170.95 | 1169.47 | 1175.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 1170.95 | 1169.47 | 1175.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:00:00 | 1170.95 | 1169.47 | 1175.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 1168.80 | 1169.25 | 1173.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:45:00 | 1173.78 | 1169.25 | 1173.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1002.48 | 991.94 | 1019.06 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 12:15:00 | 1030.00 | 1011.60 | 1011.48 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 1005.20 | 1013.02 | 1013.65 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 1027.78 | 1015.97 | 1014.93 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 15:15:00 | 1022.75 | 1023.97 | 1023.98 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 10:15:00 | 1031.50 | 1025.32 | 1024.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 12:15:00 | 1051.28 | 1030.46 | 1027.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 15:15:00 | 1064.50 | 1068.00 | 1054.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 09:15:00 | 1059.95 | 1066.39 | 1055.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 1059.95 | 1066.39 | 1055.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 1056.97 | 1066.39 | 1055.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 1046.50 | 1059.93 | 1056.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 1046.50 | 1059.93 | 1056.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 1047.00 | 1057.34 | 1055.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 09:15:00 | 1082.10 | 1057.34 | 1055.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-21 10:15:00 | 1190.31 | 1157.77 | 1144.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 11:15:00 | 1205.30 | 1227.65 | 1229.71 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 1227.58 | 1222.01 | 1221.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 13:15:00 | 1237.18 | 1225.05 | 1223.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 12:15:00 | 1218.45 | 1228.51 | 1226.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 12:15:00 | 1218.45 | 1228.51 | 1226.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 1218.45 | 1228.51 | 1226.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 1218.45 | 1228.51 | 1226.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 1221.08 | 1227.02 | 1225.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:30:00 | 1213.53 | 1227.02 | 1225.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 1227.15 | 1227.05 | 1226.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 13:00:00 | 1234.90 | 1227.67 | 1226.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:15:00 | 1259.83 | 1226.79 | 1226.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-19 09:15:00 | 1358.39 | 1301.25 | 1292.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1254.47 | 1320.06 | 1320.79 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 13:15:00 | 1342.53 | 1319.07 | 1317.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 14:15:00 | 1355.35 | 1326.32 | 1321.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 1419.18 | 1423.91 | 1402.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:30:00 | 1415.85 | 1423.91 | 1402.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 13:15:00 | 1417.78 | 1424.86 | 1415.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 13:45:00 | 1415.23 | 1424.86 | 1415.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 1415.78 | 1423.04 | 1415.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 15:00:00 | 1415.78 | 1423.04 | 1415.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 1407.00 | 1419.83 | 1414.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 1434.48 | 1419.83 | 1414.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 10:15:00 | 1423.48 | 1418.47 | 1414.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 13:15:00 | 1400.43 | 1414.95 | 1414.19 | SL hit (close<static) qty=1.00 sl=1402.53 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 1397.88 | 1416.85 | 1416.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 1379.98 | 1409.48 | 1413.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 09:15:00 | 1410.95 | 1398.94 | 1404.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 09:15:00 | 1410.95 | 1398.94 | 1404.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 1410.95 | 1398.94 | 1404.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:45:00 | 1409.35 | 1398.94 | 1404.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 1406.33 | 1400.42 | 1404.97 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 1453.08 | 1414.40 | 1410.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 09:15:00 | 1488.65 | 1444.66 | 1428.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 14:15:00 | 1575.45 | 1581.19 | 1553.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 1563.00 | 1578.16 | 1556.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 1563.00 | 1578.16 | 1556.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:45:00 | 1563.50 | 1578.16 | 1556.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 1563.53 | 1575.24 | 1557.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:45:00 | 1561.00 | 1575.24 | 1557.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 12:15:00 | 1559.23 | 1569.86 | 1557.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 13:45:00 | 1564.75 | 1567.79 | 1557.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 14:45:00 | 1569.68 | 1569.62 | 1559.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 09:15:00 | 1572.65 | 1578.22 | 1578.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 09:15:00 | 1572.65 | 1578.22 | 1578.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 12:15:00 | 1555.60 | 1568.33 | 1572.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 09:15:00 | 1573.48 | 1559.90 | 1566.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 1573.48 | 1559.90 | 1566.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 1573.48 | 1559.90 | 1566.16 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-01-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 15:15:00 | 1575.00 | 1569.24 | 1568.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 1591.43 | 1573.68 | 1570.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 15:15:00 | 1587.50 | 1591.24 | 1585.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:15:00 | 1607.00 | 1591.24 | 1585.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1592.50 | 1594.87 | 1588.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:45:00 | 1593.00 | 1594.87 | 1588.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 1539.20 | 1583.74 | 1584.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 58 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 1539.20 | 1583.74 | 1584.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 12:15:00 | 1512.53 | 1569.50 | 1577.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 1568.50 | 1549.17 | 1563.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 1568.50 | 1549.17 | 1563.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 1568.50 | 1549.17 | 1563.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:45:00 | 1576.30 | 1549.17 | 1563.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 1574.48 | 1554.23 | 1564.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:30:00 | 1573.00 | 1554.23 | 1564.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 1562.95 | 1558.66 | 1564.81 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 1588.00 | 1568.83 | 1568.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 09:15:00 | 1603.48 | 1575.76 | 1571.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 10:15:00 | 1588.50 | 1591.20 | 1583.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-29 11:00:00 | 1588.50 | 1591.20 | 1583.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 1577.50 | 1588.46 | 1583.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 11:45:00 | 1573.00 | 1588.46 | 1583.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 12:15:00 | 1574.00 | 1585.57 | 1582.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 13:00:00 | 1574.00 | 1585.57 | 1582.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 15:15:00 | 1589.55 | 1588.30 | 1584.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 09:15:00 | 1647.18 | 1588.30 | 1584.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-31 09:15:00 | 1811.90 | 1714.43 | 1660.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 11:15:00 | 1805.03 | 1841.43 | 1844.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 12:15:00 | 1801.20 | 1833.38 | 1840.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 1849.60 | 1825.66 | 1833.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 1849.60 | 1825.66 | 1833.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 1849.60 | 1825.66 | 1833.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:30:00 | 1861.80 | 1825.66 | 1833.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 1839.70 | 1828.47 | 1834.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 10:30:00 | 1852.75 | 1828.47 | 1834.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 1835.35 | 1829.84 | 1834.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:15:00 | 1840.40 | 1829.84 | 1834.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 1846.50 | 1833.18 | 1835.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 12:45:00 | 1849.95 | 1833.18 | 1835.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 13:15:00 | 1837.73 | 1834.09 | 1835.53 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2024-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 15:15:00 | 1844.50 | 1837.72 | 1837.02 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 1814.58 | 1833.09 | 1834.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 1795.00 | 1825.47 | 1831.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 1460.78 | 1431.76 | 1502.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 10:00:00 | 1460.78 | 1431.76 | 1502.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 1495.53 | 1444.51 | 1502.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:00:00 | 1495.53 | 1444.51 | 1502.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 1493.45 | 1454.30 | 1501.43 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 1541.05 | 1518.02 | 1515.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 14:15:00 | 1552.00 | 1530.23 | 1522.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 09:15:00 | 1520.58 | 1531.62 | 1524.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 1520.58 | 1531.62 | 1524.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 1520.58 | 1531.62 | 1524.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:45:00 | 1532.00 | 1531.62 | 1524.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 1531.28 | 1531.55 | 1525.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:30:00 | 1520.18 | 1531.55 | 1525.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1544.50 | 1565.60 | 1556.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 1544.50 | 1565.60 | 1556.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 1534.85 | 1559.45 | 1554.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:00:00 | 1534.85 | 1559.45 | 1554.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 12:15:00 | 1528.65 | 1549.82 | 1550.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 13:15:00 | 1509.25 | 1541.70 | 1547.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 12:15:00 | 1499.75 | 1490.83 | 1509.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 13:00:00 | 1499.75 | 1490.83 | 1509.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 1505.50 | 1493.77 | 1508.94 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 11:15:00 | 1524.50 | 1515.75 | 1515.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 09:15:00 | 1629.05 | 1551.98 | 1536.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 11:15:00 | 1623.75 | 1637.45 | 1602.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 12:00:00 | 1623.75 | 1637.45 | 1602.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 1617.38 | 1626.89 | 1607.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 09:15:00 | 1603.93 | 1626.89 | 1607.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 1599.13 | 1621.34 | 1607.10 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 14:15:00 | 1561.03 | 1600.54 | 1601.55 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 14:15:00 | 1609.35 | 1598.44 | 1597.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 09:15:00 | 1641.18 | 1607.56 | 1601.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 14:15:00 | 1619.55 | 1624.31 | 1613.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 15:00:00 | 1619.55 | 1624.31 | 1613.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 1581.63 | 1615.05 | 1611.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 1581.63 | 1615.05 | 1611.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 1578.33 | 1607.70 | 1608.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 1555.28 | 1597.22 | 1603.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 1587.83 | 1586.98 | 1597.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 14:00:00 | 1587.83 | 1586.98 | 1597.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 1594.50 | 1588.49 | 1597.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 14:30:00 | 1600.68 | 1588.49 | 1597.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 1601.00 | 1590.99 | 1597.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 1609.20 | 1590.99 | 1597.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 1624.03 | 1597.60 | 1599.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:00:00 | 1624.03 | 1597.60 | 1599.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 1615.38 | 1603.49 | 1602.23 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 09:15:00 | 1588.80 | 1601.68 | 1602.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 1580.43 | 1597.43 | 1600.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 13:15:00 | 1596.33 | 1594.43 | 1597.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 13:15:00 | 1596.33 | 1594.43 | 1597.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 1596.33 | 1594.43 | 1597.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 13:30:00 | 1595.48 | 1594.43 | 1597.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 1602.53 | 1596.05 | 1598.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 14:45:00 | 1608.50 | 1596.05 | 1598.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 1585.00 | 1593.84 | 1597.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 09:15:00 | 1552.00 | 1593.84 | 1597.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:15:00 | 1474.40 | 1517.80 | 1548.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 14:15:00 | 1396.80 | 1461.16 | 1512.50 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 71 — BUY (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 14:15:00 | 1438.00 | 1425.24 | 1423.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 15:15:00 | 1442.50 | 1428.69 | 1425.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 10:15:00 | 1410.65 | 1426.39 | 1425.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 10:15:00 | 1410.65 | 1426.39 | 1425.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1410.65 | 1426.39 | 1425.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:45:00 | 1399.00 | 1426.39 | 1425.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 1424.00 | 1425.91 | 1425.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:15:00 | 1409.50 | 1425.91 | 1425.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 12:15:00 | 1401.50 | 1421.03 | 1422.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 13:15:00 | 1389.35 | 1414.69 | 1419.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 1410.63 | 1405.90 | 1413.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 10:15:00 | 1410.63 | 1405.90 | 1413.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 1410.63 | 1405.90 | 1413.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:00:00 | 1410.63 | 1405.90 | 1413.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 1429.00 | 1410.52 | 1414.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:45:00 | 1435.00 | 1410.52 | 1414.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 1434.05 | 1415.23 | 1416.31 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 13:15:00 | 1436.00 | 1419.38 | 1418.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 1495.53 | 1439.30 | 1427.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 12:15:00 | 1588.98 | 1592.93 | 1571.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 13:00:00 | 1588.98 | 1592.93 | 1571.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 1664.23 | 1637.52 | 1625.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 09:30:00 | 1652.20 | 1637.52 | 1625.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1736.23 | 1734.02 | 1720.50 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 1666.43 | 1714.21 | 1717.29 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 13:15:00 | 1725.05 | 1713.92 | 1713.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-16 14:15:00 | 1736.00 | 1718.33 | 1715.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 09:15:00 | 1714.83 | 1720.06 | 1717.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 1714.83 | 1720.06 | 1717.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 1714.83 | 1720.06 | 1717.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:45:00 | 1714.05 | 1720.06 | 1717.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 1716.35 | 1719.32 | 1717.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 11:30:00 | 1735.70 | 1725.46 | 1720.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 09:15:00 | 1710.45 | 1726.79 | 1723.85 | SL hit (close<static) qty=1.00 sl=1712.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-04-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 10:15:00 | 1698.65 | 1721.16 | 1721.56 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 1733.75 | 1721.40 | 1720.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 1759.95 | 1736.87 | 1729.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 15:15:00 | 1795.00 | 1796.72 | 1774.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 09:15:00 | 1814.50 | 1796.72 | 1774.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 1801.10 | 1805.07 | 1792.58 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 1772.55 | 1789.17 | 1789.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-29 09:15:00 | 1757.00 | 1780.14 | 1784.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 1753.98 | 1750.80 | 1764.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 1753.98 | 1750.80 | 1764.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 1753.98 | 1750.80 | 1764.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:45:00 | 1764.68 | 1750.80 | 1764.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 1754.03 | 1755.40 | 1762.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:30:00 | 1763.25 | 1755.40 | 1762.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 1758.43 | 1756.18 | 1761.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:30:00 | 1761.85 | 1756.18 | 1761.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 1773.70 | 1751.69 | 1755.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 10:00:00 | 1773.70 | 1751.69 | 1755.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 10:15:00 | 1824.60 | 1766.27 | 1761.48 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 10:15:00 | 1747.00 | 1763.94 | 1764.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 14:15:00 | 1733.00 | 1748.76 | 1756.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 1696.20 | 1690.87 | 1714.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 1696.20 | 1690.87 | 1714.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 1706.33 | 1692.64 | 1709.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 1706.33 | 1692.64 | 1709.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 1678.58 | 1689.83 | 1706.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 11:00:00 | 1665.75 | 1684.55 | 1698.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 14:15:00 | 1582.46 | 1648.69 | 1676.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 1693.50 | 1631.11 | 1645.71 | SL hit (close>ema200) qty=0.50 sl=1631.11 alert=retest2 |

### Cycle 81 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1805.40 | 1665.97 | 1660.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 11:15:00 | 1818.35 | 1696.44 | 1674.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 11:15:00 | 1878.50 | 1879.01 | 1828.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 11:45:00 | 1873.48 | 1879.01 | 1828.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 1839.00 | 1880.45 | 1850.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:30:00 | 1839.20 | 1880.45 | 1850.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 1819.98 | 1868.36 | 1847.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:45:00 | 1814.23 | 1868.36 | 1847.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1851.48 | 1852.16 | 1845.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 1906.40 | 1852.16 | 1845.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-21 09:15:00 | 2097.04 | 2036.62 | 1970.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 2231.50 | 2252.38 | 2254.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 2153.75 | 2224.05 | 2239.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 2168.15 | 2156.94 | 2188.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:00:00 | 2168.15 | 2156.94 | 2188.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 2240.00 | 2173.55 | 2192.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 2240.00 | 2173.55 | 2192.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 2224.63 | 2183.77 | 2195.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:15:00 | 2233.73 | 2183.77 | 2195.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 2187.50 | 2195.49 | 2199.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 2256.00 | 2195.49 | 2199.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 2284.60 | 2213.31 | 2207.14 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 2024.03 | 2240.89 | 2241.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1940.00 | 2180.71 | 2214.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1998.78 | 1906.22 | 1986.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1998.78 | 1906.22 | 1986.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1998.78 | 1906.22 | 1986.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 1998.78 | 1906.22 | 1986.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 2057.38 | 1936.45 | 1992.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 2049.45 | 1936.45 | 1992.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 1977.90 | 1952.26 | 1990.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 12:30:00 | 1979.78 | 1952.26 | 1990.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 1987.58 | 1959.32 | 1990.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 2011.50 | 1959.32 | 1990.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 1988.50 | 1965.16 | 1990.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 15:00:00 | 1988.50 | 1965.16 | 1990.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1972.48 | 1969.40 | 1988.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 11:45:00 | 1956.10 | 1965.47 | 1983.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-10 11:30:00 | 1956.38 | 1971.56 | 1978.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-10 14:15:00 | 1942.50 | 1967.50 | 1975.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 10:15:00 | 2013.43 | 1979.19 | 1978.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 10:15:00 | 2013.43 | 1979.19 | 1978.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 11:15:00 | 2026.50 | 1988.66 | 1982.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 2005.00 | 2006.15 | 1995.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:00:00 | 2005.00 | 2006.15 | 1995.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1995.13 | 2003.95 | 1995.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 1995.13 | 2003.95 | 1995.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 2007.80 | 2004.72 | 1996.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:30:00 | 2002.50 | 2004.72 | 1996.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1984.50 | 2000.67 | 1995.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:45:00 | 1994.03 | 2000.67 | 1995.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1967.20 | 1993.98 | 1992.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 1967.20 | 1993.98 | 1992.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 1975.23 | 1990.23 | 1991.07 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 15:15:00 | 2008.50 | 1991.22 | 1989.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 09:15:00 | 2081.50 | 2009.28 | 1997.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 2282.60 | 2322.08 | 2244.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 2282.60 | 2322.08 | 2244.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 2282.60 | 2322.08 | 2244.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 2287.03 | 2322.08 | 2244.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 2257.43 | 2275.91 | 2254.93 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 2231.85 | 2246.37 | 2248.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 2190.00 | 2225.40 | 2237.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 2270.95 | 2221.22 | 2230.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 2270.95 | 2221.22 | 2230.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 2270.95 | 2221.22 | 2230.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 2270.95 | 2221.22 | 2230.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 2309.00 | 2238.78 | 2237.37 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 2248.75 | 2264.12 | 2265.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 2209.30 | 2241.54 | 2251.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 13:15:00 | 2233.93 | 2233.83 | 2242.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 13:30:00 | 2234.40 | 2233.83 | 2242.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 2269.32 | 2240.93 | 2244.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 2269.32 | 2240.93 | 2244.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 2265.00 | 2245.75 | 2246.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 2268.50 | 2245.75 | 2246.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 2268.60 | 2250.32 | 2248.54 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 2241.57 | 2248.62 | 2248.70 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 2313.50 | 2258.46 | 2252.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 2342.98 | 2291.02 | 2273.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 09:15:00 | 2511.50 | 2571.68 | 2518.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 2511.50 | 2571.68 | 2518.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 2511.50 | 2571.68 | 2518.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 2511.50 | 2571.68 | 2518.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 2452.53 | 2547.85 | 2512.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 2452.53 | 2547.85 | 2512.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 2460.50 | 2530.38 | 2507.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 13:30:00 | 2477.50 | 2511.36 | 2502.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 2463.30 | 2494.87 | 2496.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 2463.30 | 2494.87 | 2496.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 2437.23 | 2483.34 | 2491.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 2469.00 | 2456.91 | 2471.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 2469.00 | 2456.91 | 2471.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 2469.00 | 2456.91 | 2471.39 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 2527.25 | 2479.11 | 2478.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 13:15:00 | 2579.35 | 2499.16 | 2487.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 2523.75 | 2532.39 | 2511.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 12:00:00 | 2523.75 | 2532.39 | 2511.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 2519.48 | 2529.80 | 2512.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 2519.40 | 2529.80 | 2512.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 2517.75 | 2528.07 | 2514.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:30:00 | 2525.00 | 2528.07 | 2514.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 2535.00 | 2529.45 | 2516.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 2513.28 | 2529.45 | 2516.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2496.70 | 2522.90 | 2514.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 2496.70 | 2522.90 | 2514.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 2517.60 | 2521.84 | 2515.05 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 14:15:00 | 2503.23 | 2511.96 | 2512.04 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 2551.43 | 2518.74 | 2515.04 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 2490.32 | 2514.48 | 2515.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 15:15:00 | 2475.00 | 2506.58 | 2511.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2369.00 | 2359.08 | 2400.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 2369.00 | 2359.08 | 2400.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2369.00 | 2359.08 | 2400.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 2376.98 | 2359.08 | 2400.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 2371.15 | 2364.11 | 2383.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 2222.28 | 2368.11 | 2381.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 13:15:00 | 2317.50 | 2286.39 | 2283.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 13:15:00 | 2317.50 | 2286.39 | 2283.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 2366.43 | 2312.83 | 2297.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 2329.40 | 2337.47 | 2318.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 2329.40 | 2337.47 | 2318.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 2320.85 | 2332.95 | 2319.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:45:00 | 2321.50 | 2332.95 | 2319.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 2317.78 | 2329.92 | 2319.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 2317.78 | 2329.92 | 2319.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 2322.50 | 2328.43 | 2319.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 2318.50 | 2328.43 | 2319.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 2316.55 | 2326.06 | 2319.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:00:00 | 2316.55 | 2326.06 | 2319.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 2316.95 | 2324.24 | 2319.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:30:00 | 2316.98 | 2324.24 | 2319.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 2272.07 | 2313.80 | 2314.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 10:15:00 | 2265.75 | 2293.57 | 2304.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 15:15:00 | 2282.00 | 2278.69 | 2291.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:15:00 | 2238.05 | 2278.69 | 2291.40 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 2126.15 | 2215.38 | 2247.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-08-06 12:15:00 | 2014.25 | 2072.00 | 2136.31 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 101 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 1940.43 | 1897.57 | 1896.77 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 11:15:00 | 1865.60 | 1899.01 | 1901.02 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 1935.98 | 1904.87 | 1900.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 1972.50 | 1928.15 | 1914.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 15:15:00 | 1945.50 | 1947.59 | 1932.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:15:00 | 1946.98 | 1947.59 | 1932.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1966.25 | 1951.32 | 1935.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:00:00 | 1971.45 | 1955.85 | 1947.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 13:15:00 | 1941.15 | 1946.48 | 1946.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 1941.15 | 1946.48 | 1946.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 1937.05 | 1944.60 | 1946.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 13:15:00 | 1941.50 | 1941.24 | 1943.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-28 14:00:00 | 1941.50 | 1941.24 | 1943.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1947.00 | 1942.39 | 1943.78 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 09:15:00 | 1957.50 | 1946.55 | 1945.50 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 1935.20 | 1944.28 | 1944.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 1926.10 | 1940.64 | 1942.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 1930.00 | 1923.19 | 1931.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 1930.00 | 1923.19 | 1931.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1930.00 | 1923.19 | 1931.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 1930.00 | 1923.19 | 1931.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1918.13 | 1922.18 | 1930.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:15:00 | 1917.00 | 1922.18 | 1930.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:00:00 | 1916.63 | 1921.07 | 1929.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:00:00 | 1916.20 | 1920.09 | 1928.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:45:00 | 1917.43 | 1916.07 | 1925.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 1891.50 | 1907.55 | 1919.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 1887.48 | 1907.55 | 1919.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 1932.53 | 1911.06 | 1913.72 | SL hit (close>static) qty=1.00 sl=1931.53 alert=retest2 |

### Cycle 107 — BUY (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 11:15:00 | 1935.43 | 1915.93 | 1915.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 2032.58 | 1943.64 | 1929.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 15:15:00 | 2025.00 | 2030.83 | 2005.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 09:15:00 | 1978.58 | 2030.83 | 2005.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1972.50 | 2019.17 | 2002.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 1972.50 | 2019.17 | 2002.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 1975.38 | 2010.41 | 2000.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:30:00 | 1988.25 | 2006.28 | 1999.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 1978.25 | 1995.39 | 1995.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 1978.25 | 1995.39 | 1995.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 15:15:00 | 1974.05 | 1991.12 | 1993.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 14:15:00 | 1940.00 | 1937.80 | 1948.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:30:00 | 1939.73 | 1937.80 | 1948.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1952.50 | 1940.74 | 1949.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 1966.80 | 1940.74 | 1949.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1966.05 | 1945.80 | 1950.80 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 1968.50 | 1953.81 | 1953.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1999.28 | 1968.84 | 1961.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 12:15:00 | 1971.00 | 1972.18 | 1965.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 13:00:00 | 1971.00 | 1972.18 | 1965.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 1961.48 | 1971.99 | 1966.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 1961.48 | 1971.99 | 1966.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 1958.00 | 1969.19 | 1965.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 1954.28 | 1969.19 | 1965.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 1953.00 | 1963.56 | 1963.58 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 15:15:00 | 1971.63 | 1963.56 | 1963.32 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 1949.73 | 1960.80 | 1962.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 11:15:00 | 1946.60 | 1955.98 | 1959.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 1833.58 | 1818.04 | 1858.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:15:00 | 1840.00 | 1818.04 | 1858.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1868.43 | 1828.12 | 1859.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 1868.43 | 1828.12 | 1859.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1865.63 | 1835.62 | 1860.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 1884.73 | 1835.62 | 1860.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1865.50 | 1842.89 | 1859.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:00:00 | 1865.50 | 1842.89 | 1859.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1867.00 | 1847.71 | 1859.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 1867.00 | 1847.71 | 1859.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 1867.00 | 1851.57 | 1860.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 1912.65 | 1851.57 | 1860.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 1909.48 | 1872.01 | 1868.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 1915.98 | 1891.04 | 1879.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 1892.40 | 1907.17 | 1898.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 1892.40 | 1907.17 | 1898.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1892.40 | 1907.17 | 1898.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 1892.40 | 1907.17 | 1898.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1888.00 | 1903.33 | 1897.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:15:00 | 1888.00 | 1903.33 | 1897.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 1885.85 | 1892.69 | 1893.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 14:15:00 | 1869.00 | 1887.95 | 1891.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 1887.93 | 1886.03 | 1889.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 1887.93 | 1886.03 | 1889.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1887.93 | 1886.03 | 1889.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 1882.95 | 1886.03 | 1889.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1864.20 | 1881.67 | 1887.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:00:00 | 1855.75 | 1876.48 | 1884.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 1853.35 | 1869.71 | 1875.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1849.05 | 1868.27 | 1874.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 1848.60 | 1849.80 | 1859.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1832.50 | 1828.13 | 1837.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 1798.00 | 1828.13 | 1837.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1819.00 | 1826.31 | 1835.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 1779.53 | 1819.46 | 1828.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1762.96 | 1808.40 | 1822.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1760.68 | 1808.40 | 1822.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1756.60 | 1808.40 | 1822.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1756.17 | 1808.40 | 1822.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 1767.38 | 1768.39 | 1792.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 1790.98 | 1772.91 | 1792.07 | SL hit (close>ema200) qty=0.50 sl=1772.91 alert=retest2 |

### Cycle 115 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1826.83 | 1801.12 | 1798.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1836.00 | 1807.21 | 1802.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 14:15:00 | 1894.53 | 1899.11 | 1869.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 15:00:00 | 1894.53 | 1899.11 | 1869.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1871.33 | 1892.89 | 1872.08 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 1852.00 | 1864.92 | 1866.29 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 09:15:00 | 1943.20 | 1878.27 | 1871.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 10:15:00 | 2038.10 | 1966.27 | 1936.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 2028.05 | 2028.41 | 1999.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 15:00:00 | 2028.05 | 2028.41 | 1999.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 1952.50 | 2011.88 | 1997.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:30:00 | 1966.28 | 2011.88 | 1997.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1946.38 | 1998.78 | 1992.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 1935.80 | 1998.78 | 1992.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 1938.13 | 1986.65 | 1987.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 1928.75 | 1975.07 | 1982.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1951.98 | 1944.77 | 1963.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 1951.98 | 1944.77 | 1963.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1955.03 | 1946.82 | 1962.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:00:00 | 1938.50 | 1944.90 | 1957.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 1937.40 | 1943.33 | 1954.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:00:00 | 1940.05 | 1943.26 | 1952.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:00:00 | 1940.08 | 1942.63 | 1951.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 1937.50 | 1938.25 | 1946.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:15:00 | 1908.00 | 1938.25 | 1946.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1868.58 | 1924.32 | 1939.36 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1843.05 | 1911.79 | 1932.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1843.08 | 1911.79 | 1932.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:30:00 | 1860.20 | 1911.79 | 1932.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:30:00 | 1864.73 | 1903.43 | 1926.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 12:15:00 | 1859.38 | 1903.43 | 1926.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:45:00 | 1860.78 | 1887.70 | 1915.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 14:15:00 | 1841.57 | 1884.83 | 1911.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1904.55 | 1885.12 | 1904.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 1904.55 | 1885.12 | 1904.49 | SL hit (close>ema200) qty=0.50 sl=1885.12 alert=retest2 |

### Cycle 119 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1935.03 | 1903.70 | 1901.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 1957.98 | 1914.56 | 1906.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1987.58 | 2012.78 | 1990.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1987.58 | 2012.78 | 1990.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1987.58 | 2012.78 | 1990.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1987.58 | 2012.78 | 1990.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1979.93 | 2006.21 | 1989.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:30:00 | 1982.00 | 2006.21 | 1989.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1996.50 | 2004.27 | 1989.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:15:00 | 2011.95 | 1997.96 | 1991.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 14:15:00 | 2046.03 | 2083.49 | 2087.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 2046.03 | 2083.49 | 2087.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 2022.75 | 2065.35 | 2078.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 1906.00 | 1879.21 | 1904.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 1906.00 | 1879.21 | 1904.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1906.00 | 1879.21 | 1904.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 1917.05 | 1879.21 | 1904.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1895.50 | 1882.47 | 1903.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:15:00 | 1886.13 | 1884.39 | 1902.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 11:15:00 | 1908.95 | 1882.67 | 1881.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 1908.95 | 1882.67 | 1881.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1995.98 | 1917.23 | 1899.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 2008.05 | 2009.46 | 1975.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 15:00:00 | 2008.05 | 2009.46 | 1975.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2114.38 | 2119.73 | 2101.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 2151.00 | 2115.75 | 2107.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 2189.00 | 2127.45 | 2118.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:15:00 | 2145.03 | 2139.39 | 2134.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 10:15:00 | 2180.73 | 2207.70 | 2208.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 2180.73 | 2207.70 | 2208.18 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 2223.85 | 2208.12 | 2206.51 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 2188.57 | 2215.45 | 2216.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 2184.00 | 2204.93 | 2211.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 2011.00 | 2010.50 | 2028.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:30:00 | 2005.05 | 2010.16 | 2026.43 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 2031.05 | 2014.60 | 2024.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-27 13:15:00 | 2031.05 | 2014.60 | 2024.41 | SL hit (close>ema400) qty=1.00 sl=2024.41 alert=retest1 |

### Cycle 125 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 2063.07 | 2029.17 | 2029.13 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 2011.00 | 2028.93 | 2030.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 1990.40 | 2021.22 | 2026.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 2034.75 | 2018.03 | 2022.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 2034.75 | 2018.03 | 2022.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 2034.75 | 2018.03 | 2022.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 2034.75 | 2018.03 | 2022.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 2035.40 | 2021.51 | 2024.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 2034.53 | 2021.51 | 2024.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 2047.83 | 2028.93 | 2027.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 15:15:00 | 2062.32 | 2049.96 | 2040.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 2081.45 | 2088.80 | 2075.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 13:15:00 | 2081.45 | 2088.80 | 2075.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 2081.45 | 2088.80 | 2075.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 2081.45 | 2088.80 | 2075.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2071.80 | 2085.40 | 2075.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 2071.80 | 2085.40 | 2075.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 2070.00 | 2082.32 | 2074.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 2041.03 | 2082.32 | 2074.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2033.38 | 2072.53 | 2071.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2031.45 | 2072.53 | 2071.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 2009.75 | 2059.98 | 2065.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 1993.63 | 2038.40 | 2054.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 14:15:00 | 1895.00 | 1866.20 | 1895.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 14:15:00 | 1895.00 | 1866.20 | 1895.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 1895.00 | 1866.20 | 1895.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:30:00 | 1893.30 | 1866.20 | 1895.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 1875.00 | 1867.96 | 1893.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1862.48 | 1867.96 | 1893.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:45:00 | 1862.95 | 1863.39 | 1882.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 1769.36 | 1800.21 | 1830.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 1769.80 | 1800.21 | 1830.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-15 12:15:00 | 1776.30 | 1770.41 | 1790.76 | SL hit (close>ema200) qty=0.50 sl=1770.41 alert=retest2 |

### Cycle 129 — BUY (started 2025-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 15:15:00 | 1797.23 | 1787.20 | 1787.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 1817.73 | 1793.31 | 1789.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 1854.78 | 1866.95 | 1843.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 1854.78 | 1866.95 | 1843.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 1857.63 | 1863.83 | 1847.82 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 1803.18 | 1834.48 | 1837.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1783.45 | 1824.27 | 1832.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 1814.60 | 1811.85 | 1822.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:15:00 | 1800.03 | 1811.85 | 1822.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1838.43 | 1817.17 | 1824.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1838.43 | 1817.17 | 1824.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1866.08 | 1826.95 | 1828.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 1864.50 | 1826.95 | 1828.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 1858.00 | 1833.16 | 1830.89 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 1801.65 | 1829.10 | 1830.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1739.98 | 1802.89 | 1816.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1745.55 | 1706.65 | 1733.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1745.55 | 1706.65 | 1733.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1745.55 | 1706.65 | 1733.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1745.55 | 1706.65 | 1733.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1737.90 | 1712.90 | 1734.15 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 1782.83 | 1742.76 | 1741.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 1859.98 | 1782.99 | 1764.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1851.13 | 1899.24 | 1856.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1851.13 | 1899.24 | 1856.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1851.13 | 1899.24 | 1856.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1851.13 | 1899.24 | 1856.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1849.98 | 1889.39 | 1856.25 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 1744.35 | 1828.90 | 1836.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 1735.45 | 1810.21 | 1826.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1488.20 | 1456.73 | 1487.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 1488.20 | 1456.73 | 1487.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1488.20 | 1456.73 | 1487.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 1498.35 | 1456.73 | 1487.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1524.50 | 1470.28 | 1490.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 1524.50 | 1470.28 | 1490.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1502.50 | 1476.73 | 1491.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:00:00 | 1499.65 | 1481.31 | 1492.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 1424.67 | 1465.89 | 1481.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-17 09:15:00 | 1349.69 | 1402.76 | 1435.74 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 135 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1264.40 | 1241.23 | 1241.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1289.20 | 1265.95 | 1255.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1301.58 | 1317.55 | 1300.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1301.58 | 1317.55 | 1300.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1301.58 | 1317.55 | 1300.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1301.58 | 1317.55 | 1300.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1297.58 | 1313.55 | 1300.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 1297.58 | 1313.55 | 1300.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1298.75 | 1310.59 | 1300.30 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1274.50 | 1295.87 | 1296.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1233.97 | 1283.49 | 1290.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 1250.65 | 1242.49 | 1261.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 1250.65 | 1242.49 | 1261.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1250.65 | 1242.49 | 1261.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 1253.45 | 1242.49 | 1261.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1284.30 | 1249.88 | 1255.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 1284.30 | 1249.88 | 1255.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1282.00 | 1256.31 | 1257.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 1279.18 | 1256.31 | 1257.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 11:15:00 | 1281.58 | 1261.36 | 1259.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 11:15:00 | 1281.58 | 1261.36 | 1259.88 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 13:15:00 | 1260.58 | 1263.46 | 1263.50 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1284.40 | 1266.27 | 1264.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1308.00 | 1283.36 | 1274.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1408.95 | 1430.65 | 1407.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 1408.95 | 1430.65 | 1407.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1397.00 | 1423.92 | 1406.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 1397.00 | 1423.92 | 1406.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1394.93 | 1418.12 | 1405.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1383.73 | 1418.12 | 1405.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 1387.33 | 1398.23 | 1399.04 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 1397.75 | 1395.28 | 1395.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 1598.38 | 1435.90 | 1413.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 1577.88 | 1583.47 | 1531.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 10:30:00 | 1580.55 | 1583.47 | 1531.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1548.55 | 1576.43 | 1551.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 1545.90 | 1576.43 | 1551.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1558.75 | 1572.89 | 1551.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 1576.15 | 1561.02 | 1552.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:15:00 | 1581.13 | 1561.81 | 1553.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 14:00:00 | 1568.50 | 1567.70 | 1559.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 1512.50 | 1559.39 | 1558.24 | SL hit (close<static) qty=1.00 sl=1542.98 alert=retest2 |

### Cycle 142 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1502.03 | 1547.92 | 1553.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 1481.48 | 1518.67 | 1537.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1400.80 | 1396.48 | 1447.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1419.48 | 1396.48 | 1447.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1413.43 | 1399.87 | 1443.98 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 1501.13 | 1442.19 | 1439.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1517.98 | 1463.44 | 1449.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1550.80 | 1555.41 | 1538.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1550.80 | 1555.41 | 1538.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1569.90 | 1582.00 | 1573.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1569.90 | 1582.00 | 1573.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1571.45 | 1579.89 | 1573.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1559.70 | 1579.89 | 1573.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1572.55 | 1578.42 | 1573.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:45:00 | 1573.20 | 1578.42 | 1573.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 1574.40 | 1577.62 | 1573.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:15:00 | 1579.90 | 1577.62 | 1573.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 1582.60 | 1579.63 | 1574.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1536.80 | 1574.18 | 1576.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1536.80 | 1574.18 | 1576.37 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1588.45 | 1560.39 | 1559.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 11:15:00 | 1619.10 | 1575.44 | 1566.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 1587.80 | 1592.83 | 1580.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1587.80 | 1592.83 | 1580.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1587.80 | 1592.83 | 1580.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 09:30:00 | 1635.30 | 1614.18 | 1605.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 1567.10 | 1595.29 | 1598.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 1567.10 | 1595.29 | 1598.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 1550.00 | 1586.23 | 1594.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 1567.50 | 1564.97 | 1576.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 1570.00 | 1564.97 | 1576.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1570.30 | 1566.03 | 1575.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 1562.60 | 1572.25 | 1576.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1484.47 | 1539.62 | 1558.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 13:15:00 | 1530.00 | 1529.41 | 1546.39 | SL hit (close>ema200) qty=0.50 sl=1529.41 alert=retest2 |

### Cycle 147 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1592.20 | 1558.14 | 1554.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1642.60 | 1595.98 | 1576.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1677.05 | 1685.71 | 1658.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:30:00 | 1679.65 | 1685.71 | 1658.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1785.85 | 1837.92 | 1802.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1783.90 | 1837.92 | 1802.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1811.05 | 1832.55 | 1803.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 1826.50 | 1805.74 | 1799.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 1815.20 | 1807.23 | 1802.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 1814.65 | 1808.22 | 1803.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 1827.00 | 1813.86 | 1806.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1809.40 | 1812.75 | 1807.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:30:00 | 1804.25 | 1812.75 | 1807.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1805.80 | 1811.36 | 1807.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 1805.30 | 1811.36 | 1807.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1809.60 | 1811.01 | 1807.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:15:00 | 1797.60 | 1811.01 | 1807.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1814.65 | 1811.74 | 1808.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 1799.75 | 1811.74 | 1808.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1814.50 | 1813.21 | 1809.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 1812.20 | 1813.21 | 1809.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1826.80 | 1815.93 | 1811.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 1852.00 | 1829.18 | 1818.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-26 11:15:00 | 2009.15 | 1890.04 | 1852.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 2183.45 | 2192.58 | 2192.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 10:15:00 | 2171.65 | 2182.62 | 2186.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 2155.75 | 2131.57 | 2149.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 2155.75 | 2131.57 | 2149.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2155.75 | 2131.57 | 2149.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 2155.75 | 2131.57 | 2149.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 2180.10 | 2141.28 | 2152.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 2180.10 | 2141.28 | 2152.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 2152.00 | 2143.42 | 2152.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:30:00 | 2177.65 | 2143.42 | 2152.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 2144.85 | 2143.71 | 2151.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2120.60 | 2144.53 | 2149.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 12:15:00 | 2175.00 | 2150.35 | 2151.10 | SL hit (close>static) qty=1.00 sl=2163.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 2178.00 | 2155.88 | 2153.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 2193.90 | 2163.49 | 2157.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 2185.00 | 2190.02 | 2178.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 2177.55 | 2190.02 | 2178.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2215.00 | 2195.02 | 2181.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 2225.50 | 2197.75 | 2184.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 2225.00 | 2204.80 | 2188.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:15:00 | 2224.50 | 2208.44 | 2191.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 2148.45 | 2195.34 | 2195.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 2148.45 | 2195.34 | 2195.60 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 2334.00 | 2205.48 | 2196.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 13:15:00 | 2339.75 | 2251.54 | 2219.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 2248.40 | 2343.17 | 2304.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 2248.40 | 2343.17 | 2304.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2248.40 | 2343.17 | 2304.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 2248.40 | 2343.17 | 2304.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 2250.00 | 2324.54 | 2299.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:15:00 | 2242.90 | 2324.54 | 2299.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 2224.30 | 2275.17 | 2281.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 09:15:00 | 2219.90 | 2255.78 | 2271.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 10:15:00 | 2226.05 | 2225.61 | 2243.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:30:00 | 2228.25 | 2225.61 | 2243.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 2224.50 | 2224.38 | 2235.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 2240.35 | 2224.38 | 2235.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2240.05 | 2227.51 | 2236.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 2253.10 | 2227.51 | 2236.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2232.30 | 2228.47 | 2235.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 2227.65 | 2228.47 | 2235.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:45:00 | 2222.00 | 2228.14 | 2235.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 09:45:00 | 2229.00 | 2223.36 | 2229.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:45:00 | 2228.80 | 2224.22 | 2229.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 2234.65 | 2226.30 | 2230.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 2234.65 | 2226.30 | 2230.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 2229.05 | 2226.85 | 2229.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 2223.25 | 2226.85 | 2229.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 15:15:00 | 2227.05 | 2226.27 | 2229.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 2244.15 | 2229.97 | 2230.27 | SL hit (close>static) qty=1.00 sl=2242.20 alert=retest2 |

### Cycle 153 — BUY (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 15:15:00 | 2226.00 | 2213.44 | 2211.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 2322.15 | 2235.18 | 2221.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 15:15:00 | 2280.50 | 2283.35 | 2266.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 2294.45 | 2283.35 | 2266.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2264.05 | 2279.49 | 2265.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 2264.05 | 2279.49 | 2265.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 2263.80 | 2276.35 | 2265.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 2258.50 | 2276.35 | 2265.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2260.05 | 2273.09 | 2265.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 2260.05 | 2273.09 | 2265.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2262.70 | 2271.02 | 2264.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 2293.90 | 2265.47 | 2263.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 2227.35 | 2292.24 | 2294.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 2227.35 | 2292.24 | 2294.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 2217.50 | 2244.74 | 2266.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 2252.50 | 2241.93 | 2261.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 2252.50 | 2241.93 | 2261.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 2252.50 | 2241.93 | 2261.34 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 2307.00 | 2264.32 | 2263.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 2328.20 | 2277.10 | 2269.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 2311.15 | 2311.90 | 2300.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:45:00 | 2311.40 | 2311.90 | 2300.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2302.70 | 2310.06 | 2300.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 2303.95 | 2310.06 | 2300.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2262.50 | 2300.55 | 2296.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 2262.50 | 2300.55 | 2296.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 2282.00 | 2296.84 | 2295.60 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 2279.50 | 2293.37 | 2294.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 13:15:00 | 2270.10 | 2286.90 | 2290.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 2159.50 | 2153.88 | 2173.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 2159.50 | 2153.88 | 2173.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2159.50 | 2153.88 | 2173.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 2154.90 | 2156.48 | 2167.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 2047.15 | 2095.45 | 2123.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 2019.80 | 2017.49 | 2053.64 | SL hit (close>ema200) qty=0.50 sl=2017.49 alert=retest2 |

### Cycle 157 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 1996.15 | 1981.38 | 1980.59 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 1969.80 | 1982.14 | 1983.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1953.00 | 1972.37 | 1977.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 1962.80 | 1939.62 | 1947.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 1962.80 | 1939.62 | 1947.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1962.80 | 1939.62 | 1947.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:15:00 | 1985.00 | 1939.62 | 1947.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1999.35 | 1951.57 | 1952.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 2015.45 | 1951.57 | 1952.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1982.90 | 1957.84 | 1955.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 2005.75 | 1972.14 | 1966.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2013.05 | 2015.47 | 1995.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 2013.05 | 2015.47 | 1995.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 2042.40 | 2059.83 | 2052.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 2042.40 | 2059.83 | 2052.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 2043.35 | 2056.53 | 2051.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 2058.90 | 2056.53 | 2051.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 2046.50 | 2054.27 | 2052.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 2031.50 | 2049.72 | 2050.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 2031.50 | 2049.72 | 2050.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 2013.40 | 2042.45 | 2046.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1945.55 | 1928.47 | 1947.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1945.55 | 1928.47 | 1947.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1938.25 | 1930.43 | 1947.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1944.60 | 1930.43 | 1947.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1933.15 | 1925.37 | 1937.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1941.65 | 1925.37 | 1937.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1933.85 | 1927.07 | 1937.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 1932.55 | 1927.07 | 1937.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1931.50 | 1927.96 | 1936.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 1934.00 | 1927.96 | 1936.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1936.85 | 1929.73 | 1936.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 1936.00 | 1929.73 | 1936.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1931.80 | 1930.15 | 1936.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:15:00 | 1936.50 | 1930.15 | 1936.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1940.90 | 1932.30 | 1936.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:30:00 | 1941.45 | 1932.30 | 1936.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1966.00 | 1939.04 | 1939.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1992.10 | 1939.04 | 1939.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1993.70 | 1949.97 | 1944.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 2050.90 | 1981.11 | 1960.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 2016.05 | 2034.20 | 2012.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 2016.05 | 2034.20 | 2012.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2010.30 | 2029.42 | 2012.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 2010.00 | 2029.42 | 2012.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 2007.25 | 2024.99 | 2012.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 2007.25 | 2024.99 | 2012.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1994.65 | 2018.92 | 2010.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 1994.65 | 2018.92 | 2010.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1976.00 | 2005.40 | 2005.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 1971.05 | 1998.53 | 2002.36 | Break + close below crossover candle low |

### Cycle 163 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 2042.00 | 2007.23 | 2005.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 10:15:00 | 2062.95 | 2018.37 | 2011.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 13:15:00 | 2046.45 | 2046.57 | 2034.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 14:00:00 | 2046.45 | 2046.57 | 2034.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 2041.95 | 2045.84 | 2036.98 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 2020.50 | 2031.60 | 2032.97 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 2039.15 | 2033.97 | 2033.87 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 2020.05 | 2033.32 | 2034.17 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 2086.70 | 2041.03 | 2037.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 2178.75 | 2079.45 | 2055.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 2176.50 | 2181.68 | 2147.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 13:15:00 | 2212.45 | 2182.85 | 2168.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2180.90 | 2190.80 | 2182.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 2180.90 | 2190.80 | 2182.24 | SL hit (close<ema400) qty=1.00 sl=2182.24 alert=retest1 |

### Cycle 168 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 2173.25 | 2187.94 | 2189.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 12:15:00 | 2149.35 | 2173.61 | 2181.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 2168.00 | 2167.60 | 2174.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:30:00 | 2163.00 | 2167.60 | 2174.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 2162.50 | 2165.85 | 2172.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 2174.10 | 2165.85 | 2172.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2181.35 | 2166.32 | 2170.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 2191.10 | 2166.32 | 2170.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 2212.05 | 2175.46 | 2174.51 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 2135.00 | 2171.94 | 2174.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 2096.35 | 2141.44 | 2157.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 2073.40 | 2066.42 | 2096.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 11:00:00 | 2073.40 | 2066.42 | 2096.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 2100.00 | 2072.76 | 2087.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 2123.60 | 2072.76 | 2087.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 2157.15 | 2089.64 | 2094.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 2157.45 | 2089.64 | 2094.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 2152.60 | 2102.23 | 2099.35 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 2129.40 | 2159.68 | 2162.95 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 2190.60 | 2153.92 | 2153.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 2194.40 | 2172.00 | 2168.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 2183.75 | 2185.85 | 2176.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 2183.75 | 2185.85 | 2176.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2183.75 | 2185.85 | 2176.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 2183.75 | 2185.85 | 2176.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 2159.90 | 2180.66 | 2175.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 2159.90 | 2180.66 | 2175.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 2156.70 | 2175.87 | 2173.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 2156.70 | 2175.87 | 2173.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2180.15 | 2176.73 | 2174.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 2198.15 | 2188.34 | 2180.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 2174.95 | 2208.33 | 2212.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 2174.95 | 2208.33 | 2212.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 13:15:00 | 2166.75 | 2186.75 | 2199.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 2189.95 | 2182.83 | 2194.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 2179.60 | 2182.83 | 2194.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 2205.15 | 2187.29 | 2195.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 2205.15 | 2187.29 | 2195.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 2202.70 | 2190.37 | 2195.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 2208.00 | 2190.37 | 2195.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 2210.00 | 2194.30 | 2197.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 2210.00 | 2194.30 | 2197.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 2210.50 | 2199.73 | 2199.19 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 2194.35 | 2198.65 | 2198.75 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 2226.10 | 2204.17 | 2201.24 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 2183.50 | 2208.36 | 2210.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 2180.90 | 2202.87 | 2207.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 2157.35 | 2154.38 | 2170.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:30:00 | 2157.45 | 2154.38 | 2170.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2161.25 | 2156.66 | 2169.15 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 12:15:00 | 2204.95 | 2179.17 | 2177.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 2233.55 | 2202.45 | 2189.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 2203.30 | 2206.41 | 2195.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 12:15:00 | 2203.30 | 2206.41 | 2195.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 2203.30 | 2206.41 | 2195.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 2198.65 | 2206.41 | 2195.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 2200.00 | 2205.15 | 2196.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 2200.00 | 2205.15 | 2196.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 2203.50 | 2204.82 | 2197.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 2177.20 | 2204.82 | 2197.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 2181.20 | 2200.10 | 2195.83 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 12:15:00 | 2183.60 | 2192.03 | 2192.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 2172.40 | 2188.37 | 2190.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 2002.30 | 2001.88 | 2054.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:45:00 | 1999.50 | 2001.88 | 2054.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2032.10 | 2008.41 | 2040.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 2034.00 | 2008.41 | 2040.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2036.50 | 2014.03 | 2040.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 2035.50 | 2014.03 | 2040.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2005.00 | 2009.17 | 2026.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 2017.70 | 2009.17 | 2026.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 2015.80 | 2011.06 | 2024.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 2015.80 | 2011.06 | 2024.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 2023.00 | 2013.66 | 2021.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 2025.20 | 2013.66 | 2021.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 2035.00 | 2017.93 | 2022.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 2035.00 | 2017.93 | 2022.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 2017.90 | 2017.92 | 2021.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 2012.60 | 2017.92 | 2021.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 2013.90 | 2017.12 | 2021.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 2014.70 | 2020.27 | 2022.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 2013.00 | 2018.82 | 2021.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2001.20 | 2013.08 | 2018.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 1994.10 | 2009.07 | 2015.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 1987.50 | 1993.59 | 2003.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 2029.40 | 2000.75 | 2006.07 | SL hit (close>static) qty=1.00 sl=2026.80 alert=retest2 |

### Cycle 181 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 2016.00 | 2009.49 | 2009.20 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 2004.00 | 2008.39 | 2008.73 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 2010.20 | 2009.01 | 2008.97 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 2008.50 | 2008.91 | 2008.93 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 2010.00 | 2009.13 | 2009.02 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 2007.70 | 2008.84 | 2008.90 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 2010.00 | 2009.07 | 2009.00 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1983.60 | 2003.98 | 2006.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1979.00 | 1989.99 | 1998.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1972.90 | 1961.44 | 1975.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1972.90 | 1961.44 | 1975.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1972.90 | 1961.44 | 1975.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1972.90 | 1961.44 | 1975.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1957.50 | 1960.65 | 1973.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1946.50 | 1964.06 | 1970.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 1849.17 | 1903.72 | 1933.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1850.90 | 1845.98 | 1867.32 | SL hit (close>ema200) qty=0.50 sl=1845.98 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1683.90 | 1679.83 | 1679.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 1701.10 | 1687.34 | 1683.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1714.50 | 1722.71 | 1707.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 1711.20 | 1719.16 | 1708.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1711.20 | 1719.16 | 1708.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 1709.30 | 1719.16 | 1708.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1706.40 | 1716.61 | 1708.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:45:00 | 1707.30 | 1716.61 | 1708.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1705.00 | 1714.29 | 1707.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:15:00 | 1703.00 | 1714.29 | 1707.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1705.10 | 1712.45 | 1707.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 1702.20 | 1712.45 | 1707.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1703.00 | 1710.56 | 1707.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1704.30 | 1710.56 | 1707.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1702.40 | 1708.93 | 1706.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:15:00 | 1692.10 | 1708.93 | 1706.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1697.70 | 1706.68 | 1706.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 1694.70 | 1706.68 | 1706.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 1696.00 | 1704.55 | 1705.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 1688.00 | 1701.24 | 1703.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1683.30 | 1675.47 | 1683.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1683.30 | 1675.47 | 1683.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1683.30 | 1675.47 | 1683.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 1691.90 | 1675.47 | 1683.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1686.00 | 1677.58 | 1683.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 1682.20 | 1677.58 | 1683.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1689.80 | 1680.02 | 1684.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 1689.80 | 1680.02 | 1684.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1681.40 | 1680.30 | 1684.19 | EMA400 retest candle locked (from downside) |

### Cycle 191 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1720.90 | 1692.39 | 1689.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1765.10 | 1712.08 | 1699.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 1875.00 | 1881.35 | 1854.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 1875.00 | 1881.35 | 1854.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1853.40 | 1882.86 | 1869.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1853.40 | 1882.86 | 1869.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1856.70 | 1877.63 | 1868.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 1840.20 | 1877.63 | 1868.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1823.00 | 1859.98 | 1861.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1820.30 | 1852.04 | 1857.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1866.70 | 1847.89 | 1852.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 1866.70 | 1847.89 | 1852.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1866.70 | 1847.89 | 1852.67 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1857.30 | 1855.13 | 1855.13 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1845.20 | 1854.90 | 1855.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 1840.50 | 1849.96 | 1852.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 1849.00 | 1848.40 | 1851.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 15:00:00 | 1849.00 | 1848.40 | 1851.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1852.00 | 1849.12 | 1851.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1863.00 | 1849.12 | 1851.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1859.80 | 1851.25 | 1852.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:15:00 | 1873.20 | 1851.25 | 1852.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1882.00 | 1857.40 | 1854.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 1902.70 | 1872.71 | 1864.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1873.30 | 1883.27 | 1875.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1873.30 | 1883.27 | 1875.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1873.30 | 1883.27 | 1875.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1872.50 | 1883.27 | 1875.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1863.70 | 1879.36 | 1874.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 1861.30 | 1879.36 | 1874.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 1843.50 | 1866.35 | 1869.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 1838.80 | 1854.22 | 1858.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 1785.40 | 1783.97 | 1803.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 1789.00 | 1783.97 | 1803.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1817.00 | 1790.57 | 1805.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 1779.20 | 1790.12 | 1803.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1784.20 | 1788.38 | 1801.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1781.10 | 1776.15 | 1787.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1831.80 | 1789.33 | 1792.04 | SL hit (close>static) qty=1.00 sl=1820.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 1819.00 | 1795.27 | 1794.49 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 1791.00 | 1798.48 | 1798.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1780.40 | 1794.86 | 1797.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 15:15:00 | 1779.00 | 1778.97 | 1786.30 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:15:00 | 1751.20 | 1778.97 | 1786.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1663.64 | 1714.84 | 1743.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1713.60 | 1692.82 | 1717.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1713.60 | 1692.82 | 1717.27 | SL hit (close>ema200) qty=0.50 sl=1692.82 alert=retest1 |

### Cycle 199 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 1741.90 | 1679.53 | 1675.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 1778.40 | 1699.31 | 1684.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 1785.60 | 1787.07 | 1765.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:15:00 | 1783.80 | 1787.07 | 1765.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1788.30 | 1794.01 | 1777.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 1788.30 | 1794.01 | 1777.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1750.20 | 1785.25 | 1774.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1750.20 | 1785.25 | 1774.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1713.40 | 1770.88 | 1769.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1719.40 | 1770.88 | 1769.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1699.10 | 1756.52 | 1762.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1682.70 | 1741.76 | 1755.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 1712.00 | 1706.40 | 1727.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 1712.00 | 1706.40 | 1727.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1734.00 | 1711.92 | 1728.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1734.00 | 1711.92 | 1728.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1744.00 | 1718.33 | 1729.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1755.50 | 1718.33 | 1729.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1758.60 | 1738.59 | 1737.20 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1742.00 | 1746.30 | 1746.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1715.90 | 1739.64 | 1743.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 1745.80 | 1739.79 | 1742.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 1745.80 | 1739.79 | 1742.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1745.80 | 1739.79 | 1742.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 1752.00 | 1739.79 | 1742.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1739.00 | 1739.64 | 1742.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 14:15:00 | 1604.00 | 1739.57 | 1742.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 13:15:00 | 1758.30 | 1713.90 | 1719.74 | SL hit (close>static) qty=1.00 sl=1748.00 alert=retest2 |

### Cycle 203 — BUY (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 15:15:00 | 1759.00 | 1730.36 | 1726.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1775.80 | 1739.45 | 1731.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1749.70 | 1757.66 | 1746.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 1749.70 | 1757.66 | 1746.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1749.70 | 1757.66 | 1746.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1745.40 | 1757.66 | 1746.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1771.00 | 1760.33 | 1749.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:30:00 | 1765.00 | 1760.33 | 1749.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1743.20 | 1760.40 | 1754.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 1743.20 | 1760.40 | 1754.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1751.60 | 1758.64 | 1754.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 1745.00 | 1758.64 | 1754.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1749.00 | 1756.71 | 1753.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 1749.00 | 1756.71 | 1753.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1743.10 | 1752.91 | 1752.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:45:00 | 1738.00 | 1752.91 | 1752.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1739.00 | 1750.13 | 1751.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 1733.00 | 1746.71 | 1749.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 1742.90 | 1740.51 | 1745.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 1742.90 | 1740.51 | 1745.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1744.80 | 1741.37 | 1745.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 1737.50 | 1740.87 | 1744.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 1735.60 | 1739.82 | 1744.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:45:00 | 1736.00 | 1729.64 | 1734.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 1732.30 | 1728.55 | 1733.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1729.60 | 1728.07 | 1732.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:00:00 | 1729.60 | 1728.07 | 1732.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1731.80 | 1728.81 | 1732.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:45:00 | 1734.00 | 1728.81 | 1732.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1737.00 | 1730.45 | 1733.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 1737.00 | 1730.45 | 1733.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1736.50 | 1731.66 | 1733.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1745.00 | 1731.66 | 1733.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 1757.00 | 1736.73 | 1735.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1757.00 | 1736.73 | 1735.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 1762.50 | 1741.88 | 1737.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 1743.30 | 1758.07 | 1751.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 1743.30 | 1758.07 | 1751.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1743.30 | 1758.07 | 1751.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1743.30 | 1758.07 | 1751.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1743.30 | 1755.11 | 1750.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 1747.10 | 1755.11 | 1750.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 13:15:00 | 1736.30 | 1751.35 | 1749.24 | SL hit (close<static) qty=1.00 sl=1738.50 alert=retest2 |

### Cycle 206 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 1719.80 | 1745.04 | 1746.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1716.60 | 1739.35 | 1743.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 13:15:00 | 1696.90 | 1693.62 | 1704.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 14:00:00 | 1696.90 | 1693.62 | 1704.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1691.50 | 1693.21 | 1702.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1713.30 | 1693.21 | 1702.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1722.60 | 1699.09 | 1703.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 1729.80 | 1699.09 | 1703.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1721.00 | 1703.47 | 1705.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 1723.90 | 1703.47 | 1705.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1722.60 | 1707.30 | 1707.03 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 1703.00 | 1709.49 | 1709.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1656.90 | 1697.32 | 1703.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 1688.00 | 1686.42 | 1695.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 14:00:00 | 1688.00 | 1686.42 | 1695.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1669.30 | 1681.23 | 1690.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:15:00 | 1597.70 | 1642.24 | 1662.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 12:45:00 | 1599.00 | 1627.97 | 1652.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 13:15:00 | 1599.10 | 1627.97 | 1652.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 13:45:00 | 1598.40 | 1623.22 | 1647.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1633.60 | 1617.01 | 1632.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 1637.80 | 1617.01 | 1632.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1612.80 | 1616.17 | 1630.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-06 10:15:00 | 1689.60 | 1647.08 | 1641.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 1689.60 | 1647.08 | 1641.50 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1593.80 | 1644.53 | 1644.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 1572.90 | 1604.51 | 1623.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 1599.10 | 1595.18 | 1611.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 1599.10 | 1595.18 | 1611.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 1609.00 | 1599.16 | 1608.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 15:00:00 | 1609.00 | 1599.16 | 1608.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1612.00 | 1601.73 | 1608.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1626.00 | 1601.73 | 1608.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1612.30 | 1603.84 | 1609.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 1603.70 | 1603.67 | 1608.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:00:00 | 1603.00 | 1603.67 | 1608.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 12:15:00 | 1624.60 | 1612.87 | 1612.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 12:15:00 | 1624.60 | 1612.87 | 1612.20 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1578.70 | 1606.19 | 1609.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1553.20 | 1597.67 | 1603.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1514.70 | 1507.01 | 1533.39 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1498.60 | 1506.41 | 1530.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1497.60 | 1505.72 | 1528.20 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1556.30 | 1519.40 | 1526.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1556.30 | 1519.40 | 1526.40 | SL hit (close>ema400) qty=1.00 sl=1526.40 alert=retest1 |

### Cycle 213 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1553.90 | 1534.92 | 1532.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 14:15:00 | 1561.20 | 1542.97 | 1536.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1536.20 | 1544.98 | 1538.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1536.20 | 1544.98 | 1538.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1536.20 | 1544.98 | 1538.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1537.50 | 1544.98 | 1538.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1537.80 | 1543.54 | 1538.87 | EMA400 retest candle locked (from upside) |

### Cycle 214 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1515.10 | 1535.23 | 1536.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1499.90 | 1528.16 | 1532.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 1539.00 | 1527.18 | 1530.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 1539.00 | 1527.18 | 1530.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1539.00 | 1527.18 | 1530.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1539.00 | 1527.18 | 1530.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1519.70 | 1525.68 | 1529.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1513.60 | 1525.68 | 1529.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 1514.10 | 1521.29 | 1526.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1437.92 | 1502.09 | 1516.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1438.39 | 1502.09 | 1516.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 1430.90 | 1430.73 | 1465.31 | SL hit (close>ema200) qty=0.50 sl=1430.73 alert=retest2 |

### Cycle 215 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1510.50 | 1466.51 | 1465.75 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1432.60 | 1471.15 | 1472.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1413.30 | 1441.01 | 1454.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1484.00 | 1412.82 | 1428.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1484.00 | 1412.82 | 1428.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1484.00 | 1412.82 | 1428.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1484.00 | 1412.82 | 1428.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1462.20 | 1422.70 | 1431.49 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1490.00 | 1446.80 | 1441.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1504.70 | 1478.99 | 1464.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 13:15:00 | 1619.20 | 1619.27 | 1594.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 14:00:00 | 1619.20 | 1619.27 | 1594.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1593.80 | 1614.96 | 1598.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 1605.30 | 1615.37 | 1600.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 1765.83 | 1734.50 | 1697.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 14:15:00 | 1790.80 | 1800.29 | 1801.09 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1819.00 | 1803.19 | 1802.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 1838.00 | 1814.97 | 1808.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 14:15:00 | 1818.00 | 1820.74 | 1812.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 15:00:00 | 1818.00 | 1820.74 | 1812.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1822.60 | 1821.47 | 1814.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:00:00 | 1822.60 | 1821.47 | 1814.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1818.60 | 1821.62 | 1815.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 1820.00 | 1821.62 | 1815.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1810.90 | 1819.48 | 1815.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 1813.10 | 1819.48 | 1815.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1809.50 | 1817.48 | 1814.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1809.00 | 1817.48 | 1814.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1842.50 | 1822.49 | 1817.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 1854.30 | 1825.99 | 1819.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1783.10 | 1818.71 | 1820.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1783.10 | 1818.71 | 1820.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 1779.30 | 1810.83 | 1816.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 1804.90 | 1801.01 | 1809.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 1804.90 | 1801.01 | 1809.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1804.90 | 1801.01 | 1809.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1813.80 | 1801.01 | 1809.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1801.40 | 1801.09 | 1808.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1788.00 | 1801.09 | 1808.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1792.20 | 1799.31 | 1807.03 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 1829.80 | 1809.61 | 1809.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 1830.50 | 1813.79 | 1811.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 1950.10 | 1956.57 | 1927.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:45:00 | 1952.00 | 1956.57 | 1927.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-17 13:30:00 | 708.50 | 2023-05-23 13:15:00 | 710.88 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2023-05-17 14:15:00 | 705.63 | 2023-05-23 13:15:00 | 710.88 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2023-06-01 09:30:00 | 721.50 | 2023-06-09 11:15:00 | 753.38 | STOP_HIT | 1.00 | 4.42% |
| BUY | retest2 | 2023-06-01 10:15:00 | 720.70 | 2023-06-09 11:15:00 | 753.38 | STOP_HIT | 1.00 | 4.53% |
| BUY | retest2 | 2023-06-01 10:45:00 | 744.70 | 2023-06-09 11:15:00 | 753.38 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2023-07-05 13:30:00 | 794.83 | 2023-07-11 10:15:00 | 804.48 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2023-07-06 12:30:00 | 795.68 | 2023-07-11 10:15:00 | 804.48 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2023-07-06 13:00:00 | 795.05 | 2023-07-11 10:15:00 | 804.48 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-07-07 09:30:00 | 795.65 | 2023-07-11 10:15:00 | 804.48 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-07-13 09:15:00 | 795.78 | 2023-07-13 10:15:00 | 796.03 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2023-07-21 11:00:00 | 820.00 | 2023-07-25 09:15:00 | 902.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2023-09-04 09:15:00 | 1259.53 | 2023-09-06 15:15:00 | 1244.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest1 | 2023-09-04 11:15:00 | 1249.97 | 2023-09-06 15:15:00 | 1244.50 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-09-04 15:00:00 | 1256.47 | 2023-09-06 15:15:00 | 1244.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2023-09-05 11:15:00 | 1251.90 | 2023-09-06 15:15:00 | 1244.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-09-06 14:15:00 | 1249.15 | 2023-09-07 09:15:00 | 1218.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2023-09-14 11:00:00 | 1189.75 | 2023-09-22 10:15:00 | 1194.47 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2023-09-14 12:30:00 | 1190.00 | 2023-09-22 10:15:00 | 1194.47 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2023-09-15 09:30:00 | 1188.28 | 2023-09-22 10:15:00 | 1194.47 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-09-15 13:30:00 | 1189.25 | 2023-09-22 10:15:00 | 1194.47 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2023-09-18 10:15:00 | 1189.45 | 2023-09-22 10:15:00 | 1194.47 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-09-18 13:45:00 | 1189.80 | 2023-09-22 10:15:00 | 1194.47 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2023-09-22 09:45:00 | 1182.55 | 2023-09-22 10:15:00 | 1194.47 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-09-26 12:45:00 | 1160.10 | 2023-09-28 10:15:00 | 1179.33 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-09-26 13:15:00 | 1162.58 | 2023-09-28 10:15:00 | 1179.33 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-09-27 10:30:00 | 1159.97 | 2023-09-28 10:15:00 | 1179.33 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2023-09-27 13:15:00 | 1161.00 | 2023-09-28 10:15:00 | 1179.33 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2023-09-29 14:00:00 | 1159.25 | 2023-10-04 10:15:00 | 1174.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2023-10-03 10:30:00 | 1154.68 | 2023-10-04 10:15:00 | 1174.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-10-12 09:15:00 | 1174.97 | 2023-10-12 13:15:00 | 1163.58 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-10-12 10:00:00 | 1169.38 | 2023-10-12 13:15:00 | 1163.58 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-11-10 09:15:00 | 1082.10 | 2023-11-21 10:15:00 | 1190.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-11 13:00:00 | 1234.90 | 2023-12-19 09:15:00 | 1358.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-12 09:15:00 | 1259.83 | 2023-12-20 14:15:00 | 1254.47 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-12-29 09:15:00 | 1434.48 | 2023-12-29 13:15:00 | 1400.43 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2023-12-29 10:15:00 | 1423.48 | 2023-12-29 13:15:00 | 1400.43 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-12-29 15:15:00 | 1417.50 | 2024-01-02 09:15:00 | 1397.88 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-01-01 10:45:00 | 1424.25 | 2024-01-02 09:15:00 | 1397.88 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-01-10 13:45:00 | 1564.75 | 2024-01-16 09:15:00 | 1572.65 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-01-10 14:45:00 | 1569.68 | 2024-01-16 09:15:00 | 1572.65 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest1 | 2024-01-23 09:15:00 | 1607.00 | 2024-01-23 11:15:00 | 1539.20 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest2 | 2024-01-30 09:15:00 | 1647.18 | 2024-01-31 09:15:00 | 1811.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-12 09:15:00 | 1552.00 | 2024-03-13 11:15:00 | 1474.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-12 09:15:00 | 1552.00 | 2024-03-13 14:15:00 | 1396.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-04-18 11:30:00 | 1735.70 | 2024-04-19 09:15:00 | 1710.45 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-05-09 11:00:00 | 1665.75 | 2024-05-09 14:15:00 | 1582.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 11:00:00 | 1665.75 | 2024-05-13 09:15:00 | 1693.50 | STOP_HIT | 0.50 | -1.67% |
| BUY | retest2 | 2024-05-17 09:15:00 | 1906.40 | 2024-05-21 09:15:00 | 2097.04 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-07 11:45:00 | 1956.10 | 2024-06-11 10:15:00 | 2013.43 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-06-10 11:30:00 | 1956.38 | 2024-06-11 10:15:00 | 2013.43 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-06-10 14:15:00 | 1942.50 | 2024-06-11 10:15:00 | 2013.43 | STOP_HIT | 1.00 | -3.65% |
| BUY | retest2 | 2024-07-09 13:30:00 | 2477.50 | 2024-07-10 09:15:00 | 2463.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-07-23 12:15:00 | 2222.28 | 2024-07-29 13:15:00 | 2317.50 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest1 | 2024-08-02 09:15:00 | 2238.05 | 2024-08-05 09:15:00 | 2126.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-02 09:15:00 | 2238.05 | 2024-08-06 12:15:00 | 2014.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-08 15:00:00 | 2014.85 | 2024-08-13 14:15:00 | 1922.45 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2024-08-09 09:45:00 | 2022.50 | 2024-08-13 14:15:00 | 1922.13 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2024-08-09 11:15:00 | 2023.63 | 2024-08-13 15:15:00 | 1914.11 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2024-08-09 12:00:00 | 2023.30 | 2024-08-13 15:15:00 | 1921.38 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2024-08-12 12:15:00 | 2017.00 | 2024-08-13 15:15:00 | 1916.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-13 09:30:00 | 2009.98 | 2024-08-13 15:15:00 | 1909.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 15:00:00 | 2014.85 | 2024-08-14 09:15:00 | 1821.27 | TARGET_HIT | 0.50 | 9.61% |
| SELL | retest2 | 2024-08-09 09:45:00 | 2022.50 | 2024-08-14 09:15:00 | 1820.97 | TARGET_HIT | 0.50 | 9.96% |
| SELL | retest2 | 2024-08-09 11:15:00 | 2023.63 | 2024-08-14 10:15:00 | 1813.37 | TARGET_HIT | 0.50 | 10.39% |
| SELL | retest2 | 2024-08-09 12:00:00 | 2023.30 | 2024-08-14 10:15:00 | 1820.25 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2024-08-12 12:15:00 | 2017.00 | 2024-08-14 10:15:00 | 1815.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-13 09:30:00 | 2009.98 | 2024-08-14 10:15:00 | 1808.98 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-26 13:00:00 | 1971.45 | 2024-08-27 13:15:00 | 1941.15 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-08-30 12:15:00 | 1917.00 | 2024-09-03 10:15:00 | 1932.53 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-08-30 13:00:00 | 1916.63 | 2024-09-03 10:15:00 | 1932.53 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-08-30 14:00:00 | 1916.20 | 2024-09-03 10:15:00 | 1932.53 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-08-30 14:45:00 | 1917.43 | 2024-09-03 10:15:00 | 1932.53 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-09-02 10:15:00 | 1887.48 | 2024-09-03 10:15:00 | 1932.53 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-09-06 11:30:00 | 1988.25 | 2024-09-06 14:15:00 | 1978.25 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-09-26 12:00:00 | 1855.75 | 2024-10-07 10:15:00 | 1762.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 15:00:00 | 1853.35 | 2024-10-07 10:15:00 | 1760.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1849.05 | 2024-10-07 10:15:00 | 1756.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 09:30:00 | 1848.60 | 2024-10-07 10:15:00 | 1756.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 12:00:00 | 1855.75 | 2024-10-08 10:15:00 | 1790.98 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2024-09-27 15:00:00 | 1853.35 | 2024-10-08 10:15:00 | 1790.98 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1849.05 | 2024-10-08 10:15:00 | 1790.98 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2024-10-01 09:30:00 | 1848.60 | 2024-10-08 10:15:00 | 1790.98 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2024-10-07 10:15:00 | 1779.53 | 2024-10-09 10:15:00 | 1826.83 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-10-08 10:00:00 | 1767.38 | 2024-10-09 10:15:00 | 1826.83 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-10-08 12:15:00 | 1777.55 | 2024-10-09 10:15:00 | 1826.83 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-10-08 13:15:00 | 1779.70 | 2024-10-09 10:15:00 | 1826.83 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-10-23 14:00:00 | 1938.50 | 2024-10-25 10:15:00 | 1843.05 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2024-10-24 09:15:00 | 1937.40 | 2024-10-25 10:15:00 | 1843.08 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2024-10-24 11:00:00 | 1940.05 | 2024-10-25 14:15:00 | 1841.57 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2024-10-23 14:00:00 | 1938.50 | 2024-10-28 10:15:00 | 1904.55 | STOP_HIT | 0.50 | 1.75% |
| SELL | retest2 | 2024-10-24 09:15:00 | 1937.40 | 2024-10-28 10:15:00 | 1904.55 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2024-10-24 11:00:00 | 1940.05 | 2024-10-28 10:15:00 | 1904.55 | STOP_HIT | 0.50 | 1.83% |
| SELL | retest2 | 2024-10-24 12:00:00 | 1940.08 | 2024-10-30 09:15:00 | 1935.03 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-10-25 10:30:00 | 1860.20 | 2024-10-30 09:15:00 | 1935.03 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2024-10-25 11:30:00 | 1864.73 | 2024-10-30 09:15:00 | 1935.03 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2024-10-25 12:15:00 | 1859.38 | 2024-10-30 09:15:00 | 1935.03 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2024-10-25 13:45:00 | 1860.78 | 2024-10-30 09:15:00 | 1935.03 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2024-10-28 14:00:00 | 1892.38 | 2024-10-30 09:15:00 | 1935.03 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-10-29 10:15:00 | 1890.30 | 2024-10-30 09:15:00 | 1935.03 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-11-05 10:15:00 | 2011.95 | 2024-11-11 14:15:00 | 2046.03 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2024-11-19 12:15:00 | 1886.13 | 2024-11-22 11:15:00 | 1908.95 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-12-03 09:15:00 | 2151.00 | 2024-12-13 10:15:00 | 2180.73 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2024-12-04 09:15:00 | 2189.00 | 2024-12-13 10:15:00 | 2180.73 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-12-05 15:15:00 | 2145.03 | 2024-12-13 10:15:00 | 2180.73 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest1 | 2024-12-27 10:30:00 | 2005.05 | 2024-12-27 13:15:00 | 2031.05 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1862.48 | 2025-01-14 09:15:00 | 1769.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 12:45:00 | 1862.95 | 2025-01-14 09:15:00 | 1769.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1862.48 | 2025-01-15 12:15:00 | 1776.30 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2025-01-10 12:45:00 | 1862.95 | 2025-01-15 12:15:00 | 1776.30 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2025-02-13 13:00:00 | 1499.65 | 2025-02-14 09:15:00 | 1424.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:00:00 | 1499.65 | 2025-02-17 09:15:00 | 1349.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-13 11:15:00 | 1279.18 | 2025-03-13 11:15:00 | 1281.58 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-04-03 09:15:00 | 1576.15 | 2025-04-04 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2025-04-03 10:15:00 | 1581.13 | 2025-04-04 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest2 | 2025-04-03 14:00:00 | 1568.50 | 2025-04-04 09:15:00 | 1512.50 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-04-23 13:15:00 | 1579.90 | 2025-04-25 09:15:00 | 1536.80 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-04-23 14:45:00 | 1582.60 | 2025-04-25 09:15:00 | 1536.80 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-05-06 09:30:00 | 1635.30 | 2025-05-06 14:15:00 | 1567.10 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-05-08 13:15:00 | 1562.60 | 2025-05-09 09:15:00 | 1484.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:15:00 | 1562.60 | 2025-05-09 13:15:00 | 1530.00 | STOP_HIT | 0.50 | 2.09% |
| BUY | retest2 | 2025-05-21 10:00:00 | 1826.50 | 2025-05-26 11:15:00 | 2009.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 12:30:00 | 1815.20 | 2025-05-26 11:15:00 | 1996.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 15:00:00 | 1814.65 | 2025-05-26 11:15:00 | 1996.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 09:30:00 | 1827.00 | 2025-05-26 11:15:00 | 2009.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-23 13:30:00 | 1852.00 | 2025-05-26 11:15:00 | 2037.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2120.60 | 2025-06-16 12:15:00 | 2175.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-06-18 10:30:00 | 2225.50 | 2025-06-19 12:15:00 | 2148.45 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2025-06-18 11:45:00 | 2225.00 | 2025-06-19 12:15:00 | 2148.45 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-06-18 13:15:00 | 2224.50 | 2025-06-19 12:15:00 | 2148.45 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-06-27 11:15:00 | 2227.65 | 2025-07-01 09:15:00 | 2244.15 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-06-27 11:45:00 | 2222.00 | 2025-07-01 09:15:00 | 2244.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-30 09:45:00 | 2229.00 | 2025-07-01 09:15:00 | 2244.15 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-06-30 10:45:00 | 2228.80 | 2025-07-01 09:15:00 | 2244.15 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-30 13:15:00 | 2223.25 | 2025-07-03 13:15:00 | 2228.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-06-30 15:15:00 | 2227.05 | 2025-07-03 13:15:00 | 2228.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-07-01 10:45:00 | 2226.00 | 2025-07-03 13:15:00 | 2228.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-07-01 12:45:00 | 2223.95 | 2025-07-03 13:15:00 | 2228.50 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-07-02 12:00:00 | 2194.65 | 2025-07-03 15:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-02 12:30:00 | 2195.00 | 2025-07-03 15:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-07-02 13:00:00 | 2190.00 | 2025-07-03 15:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-07-03 10:15:00 | 2192.15 | 2025-07-03 15:15:00 | 2226.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-07-09 09:15:00 | 2293.90 | 2025-07-11 09:15:00 | 2227.35 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-24 14:45:00 | 2154.90 | 2025-07-28 09:15:00 | 2047.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 14:45:00 | 2154.90 | 2025-07-29 12:15:00 | 2019.80 | STOP_HIT | 0.50 | 6.27% |
| BUY | retest2 | 2025-08-21 09:15:00 | 2058.90 | 2025-08-21 13:15:00 | 2031.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-08-21 12:30:00 | 2046.50 | 2025-08-21 13:15:00 | 2031.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-09-17 13:15:00 | 2212.45 | 2025-09-18 13:15:00 | 2180.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-22 09:15:00 | 2189.30 | 2025-09-22 15:15:00 | 2157.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-10-15 09:30:00 | 2198.15 | 2025-10-20 09:15:00 | 2174.95 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-11-12 11:15:00 | 2012.60 | 2025-11-14 10:15:00 | 2029.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-11-12 12:00:00 | 2013.90 | 2025-11-14 10:15:00 | 2029.40 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-12 14:15:00 | 2014.70 | 2025-11-14 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-11-12 15:00:00 | 2013.00 | 2025-11-14 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-11-13 10:45:00 | 1994.10 | 2025-11-14 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-11-14 09:30:00 | 1987.50 | 2025-11-14 15:15:00 | 2016.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1946.50 | 2025-11-24 10:15:00 | 1849.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1946.50 | 2025-11-26 09:15:00 | 1850.90 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1779.20 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1784.20 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1781.10 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest1 | 2026-01-20 09:15:00 | 1751.20 | 2026-01-21 10:15:00 | 1663.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-20 09:15:00 | 1751.20 | 2026-01-22 09:15:00 | 1713.60 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1691.90 | 2026-01-28 10:15:00 | 1741.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-22 13:15:00 | 1696.00 | 2026-01-28 10:15:00 | 1741.90 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-02-06 14:15:00 | 1604.00 | 2026-02-09 13:15:00 | 1758.30 | STOP_HIT | 1.00 | -9.62% |
| SELL | retest2 | 2026-02-13 14:15:00 | 1737.50 | 2026-02-18 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-13 15:00:00 | 1735.60 | 2026-02-18 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-02-17 09:45:00 | 1736.00 | 2026-02-18 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-17 10:30:00 | 1732.30 | 2026-02-18 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-02-19 13:15:00 | 1747.10 | 2026-02-19 13:15:00 | 1736.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-03-04 11:15:00 | 1597.70 | 2026-03-06 10:15:00 | 1689.60 | STOP_HIT | 1.00 | -5.75% |
| SELL | retest2 | 2026-03-04 12:45:00 | 1599.00 | 2026-03-06 10:15:00 | 1689.60 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2026-03-04 13:15:00 | 1599.10 | 2026-03-06 10:15:00 | 1689.60 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2026-03-04 13:45:00 | 1598.40 | 2026-03-06 10:15:00 | 1689.60 | STOP_HIT | 1.00 | -5.71% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1603.70 | 2026-03-11 12:15:00 | 1624.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-11 11:00:00 | 1603.00 | 2026-03-11 12:15:00 | 1624.60 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2026-03-17 11:15:00 | 1498.60 | 2026-03-18 09:15:00 | 1556.30 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest1 | 2026-03-17 12:15:00 | 1497.60 | 2026-03-18 09:15:00 | 1556.30 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1513.60 | 2026-03-23 09:15:00 | 1437.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1514.10 | 2026-03-23 09:15:00 | 1438.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1513.60 | 2026-03-24 09:15:00 | 1430.90 | STOP_HIT | 0.50 | 5.46% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1514.10 | 2026-03-24 09:15:00 | 1430.90 | STOP_HIT | 0.50 | 5.50% |
| BUY | retest2 | 2026-04-13 10:30:00 | 1605.30 | 2026-04-17 09:15:00 | 1765.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:15:00 | 1854.30 | 2026-04-30 09:15:00 | 1783.10 | STOP_HIT | 1.00 | -3.84% |
