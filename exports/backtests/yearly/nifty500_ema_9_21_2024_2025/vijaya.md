# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1275.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 158 |
| ALERT1 | 104 |
| ALERT2 | 103 |
| ALERT2_SKIP | 62 |
| ALERT3 | 255 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 115 |
| PARTIAL | 14 |
| TARGET_HIT | 4 |
| STOP_HIT | 115 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 133 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 90
- **Target hits / Stop hits / Partials:** 4 / 115 / 14
- **Avg / median % per leg:** -0.05% / -0.84%
- **Sum % (uncompounded):** -6.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 12 | 19.7% | 3 | 58 | 0 | -0.81% | -49.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.83% | -2.5% |
| BUY @ 3rd Alert (retest2) | 58 | 12 | 20.7% | 3 | 55 | 0 | -0.80% | -46.6% |
| SELL (all) | 72 | 31 | 43.1% | 1 | 57 | 14 | 0.59% | 42.5% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL @ 3rd Alert (retest2) | 71 | 30 | 42.3% | 0 | 57 | 14 | 0.46% | 32.5% |
| retest1 (combined) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.88% | 7.5% |
| retest2 (combined) | 129 | 42 | 32.6% | 3 | 112 | 14 | -0.11% | -14.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 11:15:00 | 771.25 | 780.75 | 781.11 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 797.90 | 781.71 | 780.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 805.00 | 786.37 | 783.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 777.85 | 787.01 | 784.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 777.85 | 787.01 | 784.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 777.85 | 787.01 | 784.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 777.95 | 787.01 | 784.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 781.20 | 785.85 | 783.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 777.00 | 785.85 | 783.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 780.45 | 784.79 | 783.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:00:00 | 780.45 | 784.79 | 783.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 782.00 | 784.23 | 783.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 15:15:00 | 791.00 | 783.99 | 783.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:30:00 | 783.55 | 783.75 | 783.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 11:15:00 | 780.00 | 783.00 | 783.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 11:15:00 | 780.00 | 783.00 | 783.27 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 801.70 | 785.66 | 784.19 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 772.90 | 782.56 | 783.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 13:15:00 | 770.90 | 778.68 | 781.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 11:15:00 | 777.85 | 776.06 | 778.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 11:15:00 | 777.85 | 776.06 | 778.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 777.85 | 776.06 | 778.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 777.85 | 776.06 | 778.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 775.70 | 775.99 | 778.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:30:00 | 776.50 | 775.99 | 778.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 784.15 | 777.78 | 778.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:30:00 | 782.60 | 777.78 | 778.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 783.05 | 778.84 | 778.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:45:00 | 785.25 | 778.84 | 778.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 11:15:00 | 782.10 | 779.49 | 779.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 12:15:00 | 788.50 | 781.29 | 780.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 774.00 | 780.32 | 779.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 14:15:00 | 774.00 | 780.32 | 779.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 774.00 | 780.32 | 779.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 15:00:00 | 774.00 | 780.32 | 779.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 779.90 | 780.24 | 779.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 787.00 | 780.24 | 779.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 789.65 | 803.56 | 805.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 789.65 | 803.56 | 805.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 767.40 | 791.52 | 799.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 12:15:00 | 796.40 | 792.50 | 798.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 12:15:00 | 796.40 | 792.50 | 798.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 796.40 | 792.50 | 798.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:00:00 | 796.40 | 792.50 | 798.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 812.25 | 796.45 | 800.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:30:00 | 820.80 | 796.45 | 800.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 779.65 | 793.09 | 798.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 09:30:00 | 772.55 | 788.91 | 795.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 10:15:00 | 779.05 | 788.91 | 795.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 871.50 | 804.50 | 799.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 871.50 | 804.50 | 799.83 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 11:15:00 | 806.15 | 813.43 | 814.05 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 11:15:00 | 827.80 | 815.87 | 814.58 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 14:15:00 | 804.20 | 814.72 | 815.93 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 10:15:00 | 821.80 | 816.75 | 816.55 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 802.75 | 815.57 | 816.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 785.00 | 809.45 | 813.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 11:15:00 | 793.00 | 790.62 | 798.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 12:00:00 | 793.00 | 790.62 | 798.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 783.05 | 789.11 | 796.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 13:45:00 | 780.00 | 787.26 | 795.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:15:00 | 741.00 | 769.87 | 773.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 771.50 | 770.10 | 772.73 | SL hit (close>ema200) qty=0.50 sl=770.10 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 778.45 | 760.94 | 758.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 09:15:00 | 801.00 | 791.80 | 786.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 12:15:00 | 792.60 | 792.88 | 788.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 13:15:00 | 792.20 | 792.88 | 788.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 787.35 | 791.71 | 788.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 15:00:00 | 787.35 | 791.71 | 788.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 785.60 | 790.49 | 788.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 783.45 | 789.17 | 788.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 778.50 | 787.04 | 787.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 772.70 | 784.17 | 785.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 14:15:00 | 782.80 | 781.69 | 784.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 14:15:00 | 782.80 | 781.69 | 784.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 782.80 | 781.69 | 784.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 782.80 | 781.69 | 784.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 780.00 | 781.35 | 783.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 782.90 | 781.35 | 783.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 785.00 | 782.08 | 783.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 787.15 | 782.08 | 783.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 792.45 | 784.16 | 784.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:30:00 | 796.00 | 784.16 | 784.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 807.45 | 788.82 | 786.70 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 14:15:00 | 795.00 | 800.65 | 800.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 11:15:00 | 794.30 | 799.07 | 800.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 15:15:00 | 768.00 | 766.64 | 773.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 09:15:00 | 766.50 | 766.64 | 773.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 767.55 | 763.15 | 767.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 767.55 | 763.15 | 767.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 765.05 | 763.53 | 767.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 11:45:00 | 762.75 | 763.76 | 766.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:30:00 | 763.20 | 764.15 | 766.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:45:00 | 762.45 | 763.52 | 766.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 771.30 | 765.78 | 766.28 | SL hit (close>static) qty=1.00 sl=768.35 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 771.85 | 766.99 | 766.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 778.75 | 771.14 | 768.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 11:15:00 | 779.20 | 784.95 | 779.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 11:15:00 | 779.20 | 784.95 | 779.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 779.20 | 784.95 | 779.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:00:00 | 779.20 | 784.95 | 779.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 779.95 | 783.95 | 779.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:30:00 | 780.05 | 783.95 | 779.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 780.05 | 783.17 | 779.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 780.05 | 783.17 | 779.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 768.50 | 780.24 | 778.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 768.50 | 780.24 | 778.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 773.00 | 778.79 | 778.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 783.20 | 778.79 | 778.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 784.55 | 788.52 | 789.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 14:15:00 | 784.55 | 788.52 | 789.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 782.30 | 786.89 | 788.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 13:15:00 | 783.20 | 780.22 | 784.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 13:15:00 | 783.20 | 780.22 | 784.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 13:15:00 | 783.20 | 780.22 | 784.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 14:00:00 | 783.20 | 780.22 | 784.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 790.95 | 782.36 | 784.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 14:30:00 | 790.60 | 782.36 | 784.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 800.00 | 785.89 | 786.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 826.05 | 785.89 | 786.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 831.00 | 794.91 | 790.11 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 794.40 | 802.63 | 803.42 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 823.45 | 803.30 | 802.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 14:15:00 | 829.25 | 818.14 | 811.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 12:15:00 | 824.80 | 826.25 | 818.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 857.65 | 832.29 | 823.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 857.65 | 832.29 | 823.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:30:00 | 866.45 | 838.65 | 827.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 15:15:00 | 828.00 | 831.18 | 831.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 15:15:00 | 828.00 | 831.18 | 831.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 14:15:00 | 818.35 | 825.20 | 827.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 10:15:00 | 823.85 | 823.80 | 826.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 10:15:00 | 823.85 | 823.80 | 826.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 823.85 | 823.80 | 826.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:45:00 | 819.00 | 822.08 | 825.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 14:15:00 | 833.35 | 824.33 | 825.78 | SL hit (close>static) qty=1.00 sl=827.10 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 888.10 | 838.95 | 832.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 894.90 | 850.14 | 837.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 14:15:00 | 902.80 | 906.51 | 884.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 15:00:00 | 902.80 | 906.51 | 884.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 897.90 | 905.22 | 887.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:15:00 | 914.90 | 910.10 | 899.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:45:00 | 914.10 | 912.97 | 904.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:00:00 | 915.75 | 919.70 | 912.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 915.10 | 917.02 | 912.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 917.25 | 917.07 | 913.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 12:15:00 | 926.00 | 918.34 | 914.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 14:15:00 | 906.75 | 915.44 | 914.09 | SL hit (close<static) qty=1.00 sl=912.85 alert=retest2 |

### Cycle 25 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 909.25 | 912.77 | 913.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 10:15:00 | 903.00 | 910.82 | 912.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 15:15:00 | 897.70 | 896.20 | 900.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:15:00 | 891.05 | 896.20 | 900.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 897.00 | 896.36 | 900.55 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 915.00 | 901.91 | 901.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 15:15:00 | 917.60 | 906.98 | 904.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 898.65 | 907.27 | 905.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 898.65 | 907.27 | 905.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 898.65 | 907.27 | 905.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:45:00 | 898.45 | 907.27 | 905.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 896.45 | 905.10 | 904.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 896.45 | 905.10 | 904.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 14:15:00 | 896.75 | 902.78 | 903.49 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 902.15 | 901.68 | 901.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 920.35 | 905.62 | 903.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 931.45 | 934.32 | 922.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 10:00:00 | 931.45 | 934.32 | 922.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 927.60 | 933.71 | 924.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 927.60 | 933.71 | 924.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 926.95 | 932.36 | 925.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:15:00 | 925.00 | 932.36 | 925.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 919.45 | 929.77 | 924.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 917.35 | 929.77 | 924.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 918.10 | 927.44 | 924.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 908.30 | 927.44 | 924.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 907.10 | 919.04 | 920.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 906.90 | 916.61 | 919.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 920.65 | 916.38 | 918.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 920.65 | 916.38 | 918.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 920.65 | 916.38 | 918.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 922.35 | 916.38 | 918.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 921.00 | 917.30 | 918.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 919.35 | 917.30 | 918.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 913.00 | 916.44 | 918.40 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 927.55 | 920.19 | 919.86 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 15:15:00 | 918.25 | 919.60 | 919.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 09:15:00 | 911.20 | 917.92 | 918.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 10:15:00 | 921.30 | 918.59 | 919.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 921.30 | 918.59 | 919.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 921.30 | 918.59 | 919.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:45:00 | 926.05 | 918.59 | 919.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 920.15 | 918.91 | 919.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:15:00 | 917.35 | 918.91 | 919.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 947.50 | 922.66 | 920.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 947.50 | 922.66 | 920.43 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 917.20 | 926.35 | 926.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 09:15:00 | 908.45 | 921.65 | 924.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 14:15:00 | 919.85 | 916.89 | 920.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 919.85 | 916.89 | 920.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 918.00 | 917.11 | 920.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 923.20 | 917.11 | 920.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 910.90 | 915.87 | 919.47 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 910.90 | 905.09 | 904.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 10:15:00 | 927.25 | 911.06 | 907.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 913.55 | 921.48 | 915.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 913.55 | 921.48 | 915.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 913.55 | 921.48 | 915.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 913.55 | 921.48 | 915.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 905.85 | 918.35 | 914.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 906.25 | 918.35 | 914.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 909.95 | 913.80 | 913.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 909.95 | 913.80 | 913.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 15:15:00 | 909.95 | 913.03 | 913.10 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 916.75 | 913.78 | 913.43 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 910.25 | 913.33 | 913.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 15:15:00 | 905.55 | 911.52 | 912.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 10:15:00 | 904.30 | 897.23 | 903.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 904.30 | 897.23 | 903.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 904.30 | 897.23 | 903.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 904.30 | 897.23 | 903.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 904.00 | 898.58 | 903.09 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 13:15:00 | 959.95 | 912.49 | 908.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 12:15:00 | 964.00 | 950.71 | 938.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 11:15:00 | 985.20 | 986.92 | 973.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 12:00:00 | 985.20 | 986.92 | 973.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 978.00 | 984.69 | 976.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:15:00 | 950.85 | 984.69 | 976.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 957.25 | 979.20 | 974.72 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 12:15:00 | 956.45 | 970.12 | 971.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 14:15:00 | 948.00 | 962.94 | 967.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 985.45 | 965.03 | 967.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 985.45 | 965.03 | 967.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 985.45 | 965.03 | 967.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 992.75 | 965.03 | 967.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 981.90 | 968.41 | 968.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 994.90 | 968.41 | 968.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 982.20 | 971.16 | 970.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 991.55 | 981.65 | 976.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 988.00 | 988.12 | 981.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:30:00 | 988.60 | 988.12 | 981.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 974.95 | 984.88 | 981.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 973.05 | 984.88 | 981.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 978.00 | 983.50 | 981.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 974.90 | 983.50 | 981.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 986.30 | 983.87 | 981.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:45:00 | 990.00 | 985.44 | 982.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 09:15:00 | 969.70 | 982.81 | 982.23 | SL hit (close<static) qty=1.00 sl=980.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 969.55 | 980.16 | 981.08 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 1001.60 | 984.09 | 981.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 10:15:00 | 1019.95 | 991.26 | 985.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 14:15:00 | 1045.70 | 1046.33 | 1027.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 14:45:00 | 1044.00 | 1046.33 | 1027.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1018.00 | 1039.66 | 1028.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1018.00 | 1039.66 | 1028.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1000.60 | 1031.85 | 1025.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:30:00 | 1008.65 | 1031.85 | 1025.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 998.00 | 1020.21 | 1021.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 991.80 | 1008.22 | 1014.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 15:15:00 | 1008.90 | 999.34 | 1005.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 15:15:00 | 1008.90 | 999.34 | 1005.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 1008.90 | 999.34 | 1005.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 990.95 | 999.34 | 1005.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 941.40 | 968.21 | 983.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 14:15:00 | 937.90 | 936.44 | 946.44 | SL hit (close>ema200) qty=0.50 sl=936.44 alert=retest2 |

### Cycle 44 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 962.90 | 933.50 | 929.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 974.55 | 957.72 | 946.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 969.50 | 983.66 | 969.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 18:15:00 | 969.50 | 983.66 | 969.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 969.50 | 983.66 | 969.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 969.50 | 983.66 | 969.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 989.65 | 984.86 | 971.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 998.80 | 989.09 | 975.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 958.30 | 970.61 | 971.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 958.30 | 970.61 | 971.05 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 980.05 | 971.46 | 971.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 14:15:00 | 1021.00 | 985.57 | 980.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 12:15:00 | 1090.15 | 1092.67 | 1060.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 12:30:00 | 1082.35 | 1092.67 | 1060.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 1070.00 | 1082.73 | 1063.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 1071.85 | 1082.73 | 1063.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1077.50 | 1081.68 | 1065.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 1083.35 | 1080.00 | 1067.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 1035.80 | 1067.68 | 1065.87 | SL hit (close<static) qty=1.00 sl=1057.50 alert=retest2 |

### Cycle 47 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 1029.00 | 1059.94 | 1062.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 1025.55 | 1053.06 | 1059.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1037.20 | 1034.73 | 1046.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 10:15:00 | 1045.05 | 1036.80 | 1046.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1045.05 | 1036.80 | 1046.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1051.25 | 1036.80 | 1046.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1049.75 | 1039.39 | 1046.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:45:00 | 1045.55 | 1039.39 | 1046.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1049.70 | 1041.45 | 1046.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 1049.70 | 1041.45 | 1046.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1040.45 | 1041.25 | 1046.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 1019.30 | 1040.71 | 1045.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1062.90 | 1026.83 | 1032.35 | SL hit (close>static) qty=1.00 sl=1055.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 1066.25 | 1040.62 | 1038.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 1068.65 | 1046.23 | 1040.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 09:15:00 | 1137.05 | 1165.20 | 1130.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 10:00:00 | 1137.05 | 1165.20 | 1130.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 1156.35 | 1162.96 | 1151.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:45:00 | 1149.75 | 1162.96 | 1151.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1168.50 | 1164.07 | 1152.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 1147.35 | 1164.07 | 1152.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1146.60 | 1160.58 | 1152.36 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 1145.95 | 1154.52 | 1155.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 10:15:00 | 1139.30 | 1148.26 | 1151.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 13:15:00 | 1149.50 | 1145.89 | 1149.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 13:15:00 | 1149.50 | 1145.89 | 1149.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 1149.50 | 1145.89 | 1149.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 14:00:00 | 1149.50 | 1145.89 | 1149.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 1160.85 | 1148.88 | 1150.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:00:00 | 1160.85 | 1148.88 | 1150.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 1164.00 | 1151.91 | 1151.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 1165.00 | 1154.52 | 1152.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 1145.80 | 1155.09 | 1153.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 11:15:00 | 1145.80 | 1155.09 | 1153.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1145.80 | 1155.09 | 1153.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 1145.80 | 1155.09 | 1153.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1149.50 | 1153.97 | 1153.10 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 13:15:00 | 1145.60 | 1152.30 | 1152.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 1136.00 | 1149.04 | 1150.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 10:15:00 | 1148.50 | 1147.12 | 1149.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 10:15:00 | 1148.50 | 1147.12 | 1149.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1148.50 | 1147.12 | 1149.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 1148.50 | 1147.12 | 1149.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 1148.10 | 1147.32 | 1149.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 1145.00 | 1147.51 | 1148.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 09:45:00 | 1143.75 | 1147.17 | 1148.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 10:45:00 | 1145.10 | 1146.43 | 1148.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:15:00 | 1087.75 | 1102.25 | 1113.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:15:00 | 1087.84 | 1102.25 | 1113.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:15:00 | 1086.56 | 1099.51 | 1111.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-12 14:15:00 | 1089.70 | 1089.03 | 1101.71 | SL hit (close>ema200) qty=0.50 sl=1089.03 alert=retest2 |

### Cycle 52 — BUY (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 11:15:00 | 1059.15 | 1048.41 | 1047.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 12:15:00 | 1075.55 | 1053.84 | 1050.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 09:15:00 | 1071.40 | 1072.20 | 1061.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 10:00:00 | 1071.40 | 1072.20 | 1061.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1061.90 | 1070.14 | 1061.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:45:00 | 1061.50 | 1070.14 | 1061.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1058.65 | 1067.84 | 1061.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 1064.40 | 1066.46 | 1061.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 14:00:00 | 1075.90 | 1068.35 | 1062.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 1051.05 | 1067.53 | 1063.84 | SL hit (close<static) qty=1.00 sl=1056.30 alert=retest2 |

### Cycle 53 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 1049.85 | 1061.36 | 1061.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 1048.00 | 1057.32 | 1059.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 12:15:00 | 1060.00 | 1055.86 | 1057.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 12:15:00 | 1060.00 | 1055.86 | 1057.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 1060.00 | 1055.86 | 1057.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 1061.75 | 1055.86 | 1057.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 1056.30 | 1055.95 | 1057.63 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 1066.55 | 1059.53 | 1058.78 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 1048.90 | 1058.98 | 1059.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 15:15:00 | 1045.75 | 1051.72 | 1054.93 | Break + close below crossover candle low |

### Cycle 56 — BUY (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 10:15:00 | 1080.90 | 1057.18 | 1056.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 11:15:00 | 1094.95 | 1064.74 | 1060.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 15:15:00 | 1196.25 | 1199.80 | 1174.68 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:15:00 | 1223.90 | 1199.80 | 1174.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1217.35 | 1222.29 | 1204.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 10:30:00 | 1234.45 | 1222.02 | 1205.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:00:00 | 1235.10 | 1225.85 | 1210.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 14:45:00 | 1251.80 | 1227.74 | 1213.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 1261.00 | 1228.19 | 1215.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 1244.40 | 1235.51 | 1221.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:00:00 | 1244.40 | 1235.51 | 1221.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 12:15:00 | 1213.90 | 1229.16 | 1220.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-13 12:15:00 | 1213.90 | 1229.16 | 1220.59 | SL hit (close<ema400) qty=1.00 sl=1220.59 alert=retest1 |

### Cycle 57 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 1180.00 | 1213.00 | 1215.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 10:15:00 | 1164.15 | 1203.23 | 1210.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 1027.90 | 1026.99 | 1047.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:15:00 | 1032.35 | 1026.99 | 1047.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 1037.20 | 1028.83 | 1041.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:00:00 | 1037.20 | 1028.83 | 1041.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 963.30 | 932.67 | 949.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 963.30 | 932.67 | 949.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 960.75 | 938.28 | 950.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 960.75 | 938.28 | 950.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 974.95 | 959.19 | 957.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1014.00 | 972.36 | 964.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 10:15:00 | 1062.10 | 1067.36 | 1046.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 10:15:00 | 1062.10 | 1067.36 | 1046.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 1062.10 | 1067.36 | 1046.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:30:00 | 1059.55 | 1067.36 | 1046.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1123.30 | 1137.44 | 1123.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 1123.30 | 1137.44 | 1123.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 1133.00 | 1136.55 | 1124.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 1129.30 | 1135.10 | 1124.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1131.35 | 1134.35 | 1125.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 1133.95 | 1135.07 | 1126.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:00:00 | 1132.25 | 1134.51 | 1126.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:45:00 | 1134.70 | 1133.54 | 1127.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 1089.95 | 1124.26 | 1124.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1089.95 | 1124.26 | 1124.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1063.40 | 1096.27 | 1108.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 1084.10 | 1083.94 | 1095.57 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:15:00 | 1053.20 | 1083.94 | 1095.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-13 09:15:00 | 947.88 | 1029.15 | 1058.24 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 60 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 931.00 | 921.37 | 921.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 944.75 | 926.83 | 923.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 928.55 | 932.18 | 927.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 12:15:00 | 928.55 | 932.18 | 927.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 928.55 | 932.18 | 927.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:45:00 | 929.95 | 932.18 | 927.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 936.80 | 933.10 | 928.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 945.00 | 929.29 | 927.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 12:15:00 | 925.75 | 956.29 | 955.17 | SL hit (close<static) qty=1.00 sl=928.15 alert=retest2 |

### Cycle 61 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 916.90 | 948.41 | 951.69 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 13:15:00 | 1002.10 | 958.88 | 953.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 14:15:00 | 1050.60 | 977.22 | 962.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 10:15:00 | 955.85 | 985.17 | 970.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 10:15:00 | 955.85 | 985.17 | 970.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 955.85 | 985.17 | 970.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 955.85 | 985.17 | 970.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 971.80 | 982.49 | 971.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:45:00 | 958.00 | 982.49 | 971.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 970.90 | 980.17 | 971.01 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 906.80 | 961.28 | 964.70 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 1020.00 | 956.24 | 948.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 1056.05 | 976.20 | 958.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 15:15:00 | 978.00 | 992.26 | 973.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 15:15:00 | 978.00 | 992.26 | 973.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 978.00 | 992.26 | 973.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 09:15:00 | 1008.65 | 992.26 | 973.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 10:15:00 | 1008.85 | 994.59 | 976.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 11:15:00 | 1007.05 | 996.16 | 978.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 12:15:00 | 1008.70 | 997.49 | 980.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1062.40 | 1060.09 | 1046.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 1019.95 | 1037.87 | 1039.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1019.95 | 1037.87 | 1039.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1005.00 | 1031.30 | 1036.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 1019.50 | 1007.44 | 1014.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 09:15:00 | 1019.50 | 1007.44 | 1014.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1019.50 | 1007.44 | 1014.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:45:00 | 1012.45 | 1007.44 | 1014.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1003.40 | 1006.64 | 1013.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 12:45:00 | 995.00 | 1002.81 | 1010.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1049.60 | 1015.09 | 1013.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1049.60 | 1015.09 | 1013.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1089.00 | 1054.02 | 1036.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 13:15:00 | 1101.00 | 1108.24 | 1082.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 13:30:00 | 1099.30 | 1108.24 | 1082.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1090.60 | 1107.31 | 1089.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 1090.60 | 1107.31 | 1089.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1090.50 | 1103.95 | 1089.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:45:00 | 1089.20 | 1103.95 | 1089.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 1081.05 | 1099.37 | 1088.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 12:00:00 | 1081.05 | 1099.37 | 1088.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1079.85 | 1095.46 | 1087.67 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2025-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 15:15:00 | 1052.00 | 1077.79 | 1080.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 11:15:00 | 1025.95 | 1059.93 | 1071.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 09:15:00 | 1058.35 | 1041.81 | 1055.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 09:15:00 | 1058.35 | 1041.81 | 1055.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1058.35 | 1041.81 | 1055.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:45:00 | 1060.15 | 1041.81 | 1055.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 1058.00 | 1045.04 | 1056.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 11:00:00 | 1058.00 | 1045.04 | 1056.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 11:15:00 | 1062.00 | 1048.44 | 1056.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 11:45:00 | 1062.80 | 1048.44 | 1056.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 1070.05 | 1052.76 | 1057.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:30:00 | 1073.00 | 1052.76 | 1057.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 1052.15 | 1052.01 | 1056.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 1038.85 | 1052.01 | 1056.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1029.80 | 1047.56 | 1053.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:30:00 | 1022.65 | 1041.13 | 1050.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 1020.05 | 1029.23 | 1041.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:45:00 | 1021.55 | 1026.64 | 1035.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 09:15:00 | 971.52 | 999.58 | 1006.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 09:15:00 | 969.05 | 999.58 | 1006.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 09:15:00 | 970.47 | 999.58 | 1006.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 967.80 | 966.80 | 977.17 | SL hit (close>ema200) qty=0.50 sl=966.80 alert=retest2 |

### Cycle 68 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 984.60 | 941.25 | 937.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 13:15:00 | 990.70 | 951.14 | 942.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 13:15:00 | 970.85 | 975.24 | 962.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 15:15:00 | 969.00 | 973.22 | 963.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 969.00 | 973.22 | 963.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:15:00 | 951.60 | 973.22 | 963.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 949.50 | 968.48 | 962.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:30:00 | 946.50 | 968.48 | 962.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 949.70 | 964.72 | 961.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:30:00 | 950.00 | 964.72 | 961.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 956.75 | 962.79 | 961.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 12:45:00 | 956.05 | 962.79 | 961.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 954.80 | 961.19 | 960.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 13:45:00 | 953.75 | 961.19 | 960.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 962.30 | 961.86 | 960.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 980.80 | 961.86 | 960.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 09:15:00 | 1078.88 | 1045.12 | 1019.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 12:15:00 | 1087.40 | 1103.89 | 1104.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 13:15:00 | 1079.80 | 1099.07 | 1102.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1059.00 | 1046.16 | 1064.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 09:30:00 | 1049.30 | 1046.16 | 1064.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1026.60 | 1037.77 | 1050.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:00:00 | 1021.90 | 1030.90 | 1043.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:45:00 | 1019.90 | 1025.12 | 1037.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 10:15:00 | 1023.50 | 1013.07 | 1012.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 10:15:00 | 1023.50 | 1013.07 | 1012.06 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 1007.30 | 1011.27 | 1011.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 1001.30 | 1009.27 | 1010.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 1007.10 | 1003.87 | 1006.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 1007.10 | 1003.87 | 1006.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1007.10 | 1003.87 | 1006.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1007.10 | 1003.87 | 1006.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1006.50 | 1004.40 | 1006.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1017.80 | 1004.40 | 1006.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1012.90 | 1006.10 | 1006.75 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1014.00 | 1007.68 | 1007.41 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 1002.20 | 1006.72 | 1007.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 987.50 | 1000.94 | 1004.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1011.80 | 997.85 | 999.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1011.80 | 997.85 | 999.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1011.80 | 997.85 | 999.90 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1013.70 | 1002.77 | 1001.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1017.10 | 1007.45 | 1004.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-12 14:15:00 | 987.10 | 1003.38 | 1002.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 14:15:00 | 987.10 | 1003.38 | 1002.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 14:15:00 | 987.10 | 1003.38 | 1002.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-12 15:00:00 | 987.10 | 1003.38 | 1002.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 15:15:00 | 990.00 | 1000.70 | 1001.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 978.40 | 991.30 | 996.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 11:15:00 | 905.50 | 903.11 | 923.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-16 12:00:00 | 905.50 | 903.11 | 923.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 924.30 | 908.71 | 918.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:15:00 | 935.00 | 908.71 | 918.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 938.60 | 914.69 | 920.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 938.60 | 914.69 | 920.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 922.20 | 921.83 | 922.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:30:00 | 928.20 | 921.83 | 922.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 921.00 | 921.66 | 922.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 922.40 | 921.66 | 922.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 925.20 | 922.37 | 922.65 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 931.00 | 923.52 | 922.90 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 10:15:00 | 919.00 | 927.50 | 927.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 11:15:00 | 916.20 | 925.24 | 926.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 938.90 | 922.40 | 924.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 938.90 | 922.40 | 924.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 938.90 | 922.40 | 924.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 939.00 | 922.40 | 924.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 933.40 | 924.60 | 925.00 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 936.40 | 926.96 | 926.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 10:15:00 | 950.50 | 935.23 | 930.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 11:15:00 | 948.10 | 948.19 | 941.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 11:30:00 | 947.00 | 948.19 | 941.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 941.90 | 946.38 | 943.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 940.10 | 946.38 | 943.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 951.00 | 947.31 | 944.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 949.20 | 947.31 | 944.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 945.90 | 949.26 | 946.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:15:00 | 940.70 | 949.26 | 946.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 946.70 | 948.75 | 946.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 953.00 | 948.36 | 946.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 958.80 | 950.31 | 948.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 945.00 | 947.16 | 947.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 12:15:00 | 945.00 | 947.16 | 947.21 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 956.00 | 948.93 | 948.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 961.00 | 951.34 | 949.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 956.70 | 959.14 | 955.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 956.70 | 959.14 | 955.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 956.70 | 959.14 | 955.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 955.20 | 959.14 | 955.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 969.10 | 973.12 | 968.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 964.55 | 973.12 | 968.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 965.00 | 971.50 | 968.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 966.20 | 971.50 | 968.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 963.20 | 969.84 | 967.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:30:00 | 964.35 | 969.84 | 967.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 971.85 | 972.64 | 970.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 973.05 | 972.64 | 970.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 974.95 | 973.13 | 970.80 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 965.00 | 968.65 | 969.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 949.40 | 961.34 | 965.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 15:15:00 | 959.00 | 954.43 | 957.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 15:15:00 | 959.00 | 954.43 | 957.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 959.00 | 954.43 | 957.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 951.30 | 954.43 | 957.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 956.35 | 954.81 | 957.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 14:45:00 | 948.60 | 952.57 | 954.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 947.40 | 952.25 | 954.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 10:30:00 | 948.05 | 950.66 | 953.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 961.00 | 954.29 | 954.56 | SL hit (close>static) qty=1.00 sl=960.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 966.15 | 956.66 | 955.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 970.45 | 959.42 | 956.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 953.55 | 958.24 | 956.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 953.55 | 958.24 | 956.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 953.55 | 958.24 | 956.65 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 946.05 | 954.53 | 955.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 944.10 | 952.45 | 954.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 950.40 | 950.22 | 952.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 12:15:00 | 958.45 | 951.68 | 952.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 958.45 | 951.68 | 952.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:00:00 | 958.45 | 951.68 | 952.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 957.60 | 952.86 | 952.97 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 956.35 | 953.56 | 953.28 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 948.00 | 952.68 | 952.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 943.95 | 950.93 | 952.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 11:15:00 | 937.05 | 935.82 | 940.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 12:00:00 | 937.05 | 935.82 | 940.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 931.15 | 933.11 | 937.18 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 945.70 | 937.85 | 937.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 950.50 | 940.38 | 938.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 988.15 | 989.14 | 980.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 988.15 | 989.14 | 980.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 976.95 | 985.27 | 982.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 976.00 | 985.27 | 982.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 982.60 | 984.73 | 982.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 985.70 | 982.36 | 981.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 974.75 | 980.68 | 981.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 974.75 | 980.68 | 981.07 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 1001.30 | 984.89 | 982.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 11:15:00 | 1021.45 | 1004.32 | 996.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 1009.45 | 1013.07 | 1005.17 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 12:45:00 | 1014.05 | 1013.27 | 1006.63 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 14:00:00 | 1015.55 | 1013.73 | 1007.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1012.00 | 1013.14 | 1008.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1010.75 | 1013.14 | 1008.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1006.30 | 1011.78 | 1008.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1006.30 | 1011.78 | 1008.08 | SL hit (close<ema400) qty=1.00 sl=1008.08 alert=retest1 |

### Cycle 89 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 1000.55 | 1005.37 | 1005.81 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 1036.55 | 1011.12 | 1008.26 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 1009.95 | 1014.43 | 1014.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 1002.70 | 1012.08 | 1013.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 1016.70 | 1012.51 | 1013.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 1016.70 | 1012.51 | 1013.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1016.70 | 1012.51 | 1013.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 1016.70 | 1012.51 | 1013.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 1018.05 | 1013.62 | 1013.92 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 1022.40 | 1015.38 | 1014.69 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 1003.50 | 1012.39 | 1013.52 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 1022.00 | 1014.58 | 1014.27 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 11:15:00 | 1010.10 | 1014.10 | 1014.42 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1018.00 | 1014.79 | 1014.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1025.90 | 1017.52 | 1015.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 1034.30 | 1042.72 | 1037.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 1034.30 | 1042.72 | 1037.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1034.30 | 1042.72 | 1037.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 1034.30 | 1042.72 | 1037.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 1031.00 | 1040.38 | 1036.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 1031.00 | 1040.38 | 1036.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 1016.30 | 1033.28 | 1033.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 10:15:00 | 1010.00 | 1028.62 | 1031.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1018.10 | 1018.08 | 1024.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1018.10 | 1018.08 | 1024.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1018.10 | 1018.08 | 1024.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 1022.15 | 1018.08 | 1024.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1016.65 | 1014.88 | 1019.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1009.50 | 1014.88 | 1019.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1016.50 | 1015.20 | 1019.18 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1060.50 | 1024.77 | 1021.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 10:15:00 | 1082.10 | 1036.24 | 1027.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1076.00 | 1080.98 | 1057.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:45:00 | 1076.25 | 1080.98 | 1057.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1062.25 | 1073.26 | 1061.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 1062.25 | 1073.26 | 1061.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1070.00 | 1072.61 | 1061.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 1063.00 | 1072.61 | 1061.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1084.10 | 1075.29 | 1065.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:00:00 | 1116.75 | 1077.31 | 1068.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 1103.65 | 1098.77 | 1097.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1076.55 | 1093.09 | 1095.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 1076.55 | 1093.09 | 1095.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 1070.85 | 1083.64 | 1089.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 1086.60 | 1084.23 | 1089.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 1086.60 | 1084.23 | 1089.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1086.60 | 1084.23 | 1089.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 1081.50 | 1084.23 | 1089.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1086.60 | 1084.71 | 1089.27 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 1095.70 | 1089.76 | 1089.26 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1083.50 | 1088.72 | 1089.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 14:15:00 | 1072.10 | 1081.67 | 1085.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1057.90 | 1056.05 | 1063.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 1057.90 | 1056.05 | 1063.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1060.00 | 1056.84 | 1063.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1053.60 | 1056.84 | 1063.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1053.80 | 1056.23 | 1062.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:45:00 | 1042.00 | 1052.97 | 1058.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:15:00 | 1047.20 | 1049.42 | 1054.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 1058.00 | 1055.43 | 1055.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 15:15:00 | 1058.00 | 1055.43 | 1055.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 1058.10 | 1055.96 | 1055.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 13:15:00 | 1055.80 | 1056.79 | 1056.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 13:15:00 | 1055.80 | 1056.79 | 1056.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1055.80 | 1056.79 | 1056.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1055.80 | 1056.79 | 1056.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1057.60 | 1056.95 | 1056.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 1068.10 | 1056.95 | 1056.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 1055.10 | 1058.37 | 1057.08 | SL hit (close<static) qty=1.00 sl=1055.80 alert=retest2 |

### Cycle 103 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1050.50 | 1055.98 | 1056.39 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 1061.00 | 1054.93 | 1054.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 1065.30 | 1057.00 | 1055.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 1057.00 | 1057.00 | 1055.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 11:15:00 | 1057.00 | 1057.00 | 1055.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 1057.00 | 1057.00 | 1055.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 1057.00 | 1057.00 | 1055.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 1042.40 | 1054.08 | 1054.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 1035.90 | 1044.93 | 1049.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1027.00 | 1023.76 | 1032.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 12:45:00 | 1025.70 | 1023.76 | 1032.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 1024.30 | 1022.52 | 1030.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:45:00 | 1025.30 | 1022.52 | 1030.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1034.20 | 1024.50 | 1029.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 1034.20 | 1024.50 | 1029.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1020.70 | 1023.74 | 1029.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 1019.10 | 1022.79 | 1028.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1013.00 | 1022.18 | 1026.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1027.90 | 1007.75 | 1006.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1027.90 | 1007.75 | 1006.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1033.50 | 1012.90 | 1009.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1021.10 | 1027.94 | 1019.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1021.10 | 1027.94 | 1019.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1021.10 | 1027.94 | 1019.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1049.40 | 1026.14 | 1022.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 1046.70 | 1033.77 | 1025.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 1083.20 | 1092.71 | 1093.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 1083.20 | 1092.71 | 1093.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 1062.70 | 1084.84 | 1089.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 1053.00 | 1052.42 | 1062.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 1053.00 | 1052.42 | 1062.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1050.10 | 1052.05 | 1060.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 1039.30 | 1046.53 | 1048.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 1039.80 | 1045.08 | 1047.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:45:00 | 1039.90 | 1043.77 | 1046.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1062.20 | 1047.76 | 1047.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 1062.20 | 1047.76 | 1047.04 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1045.00 | 1046.83 | 1046.86 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 1051.60 | 1047.31 | 1047.03 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1042.40 | 1046.05 | 1046.49 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 1051.20 | 1045.73 | 1045.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 1062.30 | 1049.04 | 1047.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 11:15:00 | 1048.10 | 1049.81 | 1047.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 11:15:00 | 1048.10 | 1049.81 | 1047.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1048.10 | 1049.81 | 1047.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1048.10 | 1049.81 | 1047.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1049.40 | 1049.72 | 1048.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 1048.50 | 1049.72 | 1048.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1051.40 | 1050.52 | 1048.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1047.70 | 1049.96 | 1048.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1047.20 | 1049.41 | 1048.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 1046.00 | 1049.41 | 1048.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 1039.50 | 1047.43 | 1047.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1020.40 | 1038.49 | 1043.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 1025.50 | 1018.54 | 1024.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 15:15:00 | 1025.50 | 1018.54 | 1024.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1025.50 | 1018.54 | 1024.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 1009.90 | 1014.44 | 1022.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1009.05 | 1009.04 | 1009.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:30:00 | 1011.40 | 1009.29 | 1009.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:00:00 | 1011.30 | 1009.69 | 1010.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1013.50 | 1010.45 | 1010.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 1013.50 | 1010.45 | 1010.37 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 1007.60 | 1010.15 | 1010.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 1002.95 | 1008.47 | 1009.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 1005.80 | 1003.95 | 1006.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 1005.80 | 1003.95 | 1006.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1005.80 | 1003.95 | 1006.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:45:00 | 997.00 | 1001.89 | 1005.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 995.75 | 1000.94 | 1003.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 990.00 | 980.02 | 979.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 990.00 | 980.02 | 979.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 995.25 | 988.02 | 984.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 990.30 | 992.15 | 987.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 990.30 | 992.15 | 987.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 988.00 | 991.32 | 987.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 988.00 | 991.32 | 987.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 987.90 | 990.64 | 987.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:45:00 | 986.50 | 990.64 | 987.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 980.90 | 988.69 | 987.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 980.90 | 988.69 | 987.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 986.50 | 988.25 | 987.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 988.70 | 988.65 | 987.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 989.00 | 988.65 | 987.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 989.00 | 992.41 | 992.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 987.85 | 991.50 | 991.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 987.85 | 991.50 | 991.71 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 998.65 | 991.75 | 991.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 1013.55 | 1002.51 | 999.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1001.25 | 1005.84 | 1002.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1001.25 | 1005.84 | 1002.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1001.25 | 1005.84 | 1002.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 1000.05 | 1005.84 | 1002.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1005.15 | 1005.70 | 1002.76 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 994.35 | 1000.89 | 1001.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 983.30 | 996.73 | 999.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 990.85 | 990.72 | 995.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 990.85 | 990.72 | 995.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1005.00 | 993.08 | 995.01 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1006.00 | 997.11 | 996.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 1012.85 | 1000.26 | 998.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 1017.00 | 1023.89 | 1015.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 15:15:00 | 1017.00 | 1023.89 | 1015.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1017.00 | 1023.89 | 1015.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1001.05 | 1023.89 | 1015.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1013.00 | 1021.71 | 1015.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:30:00 | 1024.55 | 1020.43 | 1017.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1006.30 | 1016.17 | 1017.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 1006.30 | 1016.17 | 1017.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 1000.70 | 1013.08 | 1015.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 1021.50 | 1010.17 | 1013.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 14:15:00 | 1021.50 | 1010.17 | 1013.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1021.50 | 1010.17 | 1013.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 1021.50 | 1010.17 | 1013.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1020.00 | 1012.14 | 1013.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 1006.00 | 1012.14 | 1013.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1013.80 | 1010.48 | 1012.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 1013.80 | 1010.48 | 1012.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1018.65 | 1012.12 | 1012.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 1018.65 | 1012.12 | 1012.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1018.90 | 1013.47 | 1013.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 1019.30 | 1013.47 | 1013.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 1015.30 | 1013.84 | 1013.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 1024.20 | 1017.15 | 1015.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1013.45 | 1019.68 | 1017.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1013.45 | 1019.68 | 1017.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1013.45 | 1019.68 | 1017.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1013.45 | 1019.68 | 1017.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1043.00 | 1024.34 | 1020.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:45:00 | 1058.25 | 1038.62 | 1030.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:30:00 | 1060.45 | 1048.09 | 1037.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 1016.00 | 1038.57 | 1040.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1016.00 | 1038.57 | 1040.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 12:15:00 | 1009.05 | 1021.98 | 1028.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1016.10 | 1013.99 | 1021.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1016.10 | 1013.99 | 1021.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1016.10 | 1013.99 | 1021.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:15:00 | 1003.00 | 1012.32 | 1019.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 982.10 | 1009.90 | 1013.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1034.00 | 1004.63 | 1007.82 | SL hit (close>static) qty=1.00 sl=1028.00 alert=retest2 |

### Cycle 124 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 1012.50 | 1000.60 | 1000.38 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 1000.05 | 1003.60 | 1003.93 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 1012.55 | 1005.47 | 1004.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 1036.75 | 1011.73 | 1007.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 1037.10 | 1047.37 | 1032.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:00:00 | 1037.10 | 1047.37 | 1032.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1021.50 | 1042.20 | 1031.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 1020.45 | 1042.20 | 1031.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1017.95 | 1037.35 | 1030.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 1017.95 | 1037.35 | 1030.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1014.60 | 1025.20 | 1025.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1000.40 | 1015.37 | 1020.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 1015.75 | 1015.45 | 1020.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 15:00:00 | 1015.75 | 1015.45 | 1020.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1026.50 | 1015.90 | 1018.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1026.50 | 1015.90 | 1018.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1020.00 | 1016.72 | 1018.85 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1031.10 | 1020.17 | 1020.08 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 1013.75 | 1019.40 | 1019.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1001.75 | 1014.47 | 1017.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 13:15:00 | 988.00 | 984.02 | 991.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 13:45:00 | 987.45 | 984.02 | 991.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 977.05 | 979.51 | 984.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 961.85 | 979.53 | 982.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 14:15:00 | 986.80 | 978.06 | 979.40 | SL hit (close>static) qty=1.00 sl=986.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 989.00 | 981.02 | 980.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 998.00 | 985.85 | 983.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 990.00 | 990.20 | 986.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 10:15:00 | 990.00 | 990.20 | 986.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 990.00 | 990.20 | 986.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:45:00 | 986.95 | 990.20 | 986.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 1005.00 | 996.96 | 991.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:30:00 | 1007.80 | 1001.69 | 994.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1039.00 | 1045.46 | 1045.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 1039.00 | 1045.46 | 1045.80 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1049.70 | 1046.31 | 1046.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1054.50 | 1047.94 | 1046.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1043.70 | 1053.96 | 1051.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 1043.70 | 1053.96 | 1051.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1043.70 | 1053.96 | 1051.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1043.70 | 1053.96 | 1051.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1054.50 | 1054.07 | 1051.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1057.20 | 1054.07 | 1051.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:45:00 | 1060.60 | 1054.63 | 1052.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:45:00 | 1056.30 | 1055.07 | 1052.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 1056.60 | 1055.05 | 1052.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1050.70 | 1054.38 | 1053.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 1050.00 | 1052.26 | 1052.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 1050.00 | 1052.26 | 1052.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1049.10 | 1051.63 | 1051.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 991.30 | 985.13 | 995.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:00:00 | 991.30 | 985.13 | 995.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 998.50 | 987.80 | 995.89 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 1005.00 | 998.53 | 998.23 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 990.40 | 996.90 | 997.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 954.50 | 974.87 | 982.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 15:15:00 | 975.00 | 965.72 | 972.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 15:15:00 | 975.00 | 965.72 | 972.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 975.00 | 965.72 | 972.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 950.00 | 965.72 | 972.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 951.90 | 962.96 | 970.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:15:00 | 946.10 | 962.96 | 970.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:45:00 | 946.30 | 960.38 | 968.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 985.90 | 968.25 | 970.07 | SL hit (close>static) qty=1.00 sl=975.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 983.30 | 973.12 | 972.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 991.80 | 982.09 | 977.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 992.40 | 993.82 | 987.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 992.40 | 993.82 | 987.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 971.30 | 990.29 | 986.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 971.30 | 990.29 | 986.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 972.80 | 986.79 | 985.54 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 975.00 | 984.43 | 984.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 14:15:00 | 968.80 | 978.26 | 981.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 15:15:00 | 940.00 | 939.59 | 951.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 09:15:00 | 939.10 | 939.59 | 951.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 943.80 | 940.44 | 951.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 950.00 | 940.44 | 951.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 945.80 | 941.51 | 950.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 948.30 | 941.51 | 950.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 946.80 | 942.57 | 950.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 951.00 | 942.57 | 950.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 948.80 | 943.81 | 950.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 948.80 | 943.81 | 950.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 960.10 | 947.07 | 951.10 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 959.55 | 953.03 | 953.00 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 952.10 | 952.84 | 952.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 948.15 | 951.90 | 952.49 | Break + close below crossover candle low |

### Cycle 140 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 964.25 | 954.37 | 953.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 14:15:00 | 981.10 | 959.72 | 956.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 949.05 | 959.23 | 956.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 10:15:00 | 949.05 | 959.23 | 956.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 949.05 | 959.23 | 956.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 949.05 | 959.23 | 956.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 945.30 | 956.45 | 955.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 945.30 | 956.45 | 955.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 948.00 | 954.76 | 955.15 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 967.55 | 957.51 | 956.35 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 10:15:00 | 949.95 | 955.41 | 955.67 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 964.40 | 956.60 | 956.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 966.30 | 958.54 | 957.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 962.00 | 962.13 | 959.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 962.00 | 962.13 | 959.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 961.45 | 961.99 | 959.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:45:00 | 958.30 | 961.99 | 959.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 960.35 | 961.66 | 959.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 960.35 | 961.66 | 959.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 961.65 | 961.66 | 959.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 958.10 | 961.66 | 959.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 980.10 | 965.35 | 961.80 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 963.55 | 965.05 | 965.16 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 980.55 | 967.76 | 966.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 990.00 | 972.21 | 968.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 11:15:00 | 1011.20 | 1013.29 | 1001.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 11:30:00 | 1010.15 | 1013.29 | 1001.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1003.65 | 1011.46 | 1004.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 1001.75 | 1011.46 | 1004.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1004.35 | 1010.03 | 1004.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 1003.15 | 1010.03 | 1004.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1003.60 | 1008.75 | 1004.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:30:00 | 1006.55 | 1008.75 | 1004.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 1000.35 | 1007.07 | 1004.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 1000.60 | 1007.07 | 1004.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1000.10 | 1005.67 | 1003.96 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 991.85 | 1001.37 | 1002.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 987.70 | 998.64 | 1000.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1003.15 | 994.91 | 997.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 1003.15 | 994.91 | 997.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1003.15 | 994.91 | 997.26 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 1013.90 | 999.96 | 999.22 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 1001.65 | 1009.23 | 1009.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 13:15:00 | 1000.15 | 1007.42 | 1008.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 1016.55 | 1009.24 | 1009.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1016.55 | 1009.24 | 1009.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1016.55 | 1009.24 | 1009.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1016.55 | 1009.24 | 1009.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 1015.95 | 1010.59 | 1009.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 1019.05 | 1012.28 | 1010.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 1011.15 | 1013.19 | 1011.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 1011.15 | 1013.19 | 1011.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1011.15 | 1013.19 | 1011.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1011.15 | 1013.19 | 1011.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1005.20 | 1011.59 | 1010.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 1000.15 | 1011.59 | 1010.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1010.00 | 1011.27 | 1010.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 1011.30 | 1010.82 | 1010.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 1010.80 | 1010.67 | 1010.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 12:00:00 | 1011.45 | 1010.82 | 1010.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 12:15:00 | 1005.00 | 1009.66 | 1010.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 1005.00 | 1009.66 | 1010.17 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1014.85 | 1010.17 | 1010.14 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1000.75 | 1010.80 | 1010.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 10:15:00 | 990.55 | 1006.75 | 1009.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 10:15:00 | 993.75 | 993.12 | 999.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:45:00 | 992.65 | 993.12 | 999.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 996.05 | 991.33 | 995.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 989.00 | 991.17 | 994.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:30:00 | 986.90 | 990.27 | 993.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 972.20 | 987.99 | 988.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 939.55 | 956.49 | 968.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 937.55 | 956.49 | 968.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 951.70 | 948.87 | 958.42 | SL hit (close>ema200) qty=0.50 sl=948.87 alert=retest2 |

### Cycle 154 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 957.00 | 949.46 | 948.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 963.00 | 953.93 | 951.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 955.00 | 956.33 | 952.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 945.40 | 956.33 | 952.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 944.30 | 953.92 | 952.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 943.90 | 953.92 | 952.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 952.00 | 953.54 | 952.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 954.40 | 953.39 | 952.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 953.20 | 953.39 | 952.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 954.50 | 952.65 | 952.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 15:15:00 | 945.00 | 951.12 | 951.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 945.00 | 951.12 | 951.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 942.30 | 949.35 | 950.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 934.10 | 927.05 | 933.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 934.10 | 927.05 | 933.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 934.10 | 927.05 | 933.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 934.10 | 927.05 | 933.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 927.00 | 927.04 | 932.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 933.00 | 927.04 | 932.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 944.20 | 930.47 | 933.83 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 942.20 | 936.62 | 936.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 947.60 | 939.88 | 938.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 938.40 | 942.19 | 940.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 14:15:00 | 938.40 | 942.19 | 940.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 938.40 | 942.19 | 940.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:45:00 | 939.20 | 942.19 | 940.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 938.00 | 941.36 | 939.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 919.80 | 941.36 | 939.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 920.90 | 937.26 | 938.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 912.60 | 918.23 | 924.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 919.50 | 918.48 | 923.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 919.50 | 918.48 | 923.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 907.50 | 892.46 | 897.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 904.60 | 892.46 | 897.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 898.00 | 893.57 | 897.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 894.20 | 894.23 | 897.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 896.00 | 894.23 | 897.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:15:00 | 895.10 | 895.29 | 897.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 849.49 | 872.39 | 882.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 851.20 | 872.39 | 882.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 850.35 | 872.39 | 882.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 863.00 | 861.12 | 871.79 | SL hit (close>ema200) qty=0.50 sl=861.12 alert=retest2 |

### Cycle 158 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 885.00 | 876.45 | 875.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 895.85 | 884.55 | 880.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 883.65 | 885.56 | 881.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 883.65 | 885.56 | 881.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 883.65 | 885.56 | 881.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:45:00 | 881.50 | 885.56 | 881.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 881.85 | 884.82 | 881.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 881.85 | 884.82 | 881.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 884.70 | 884.80 | 882.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:15:00 | 885.65 | 884.80 | 882.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:45:00 | 886.25 | 885.18 | 882.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 11:15:00 | 974.22 | 962.71 | 942.50 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-15 09:15:00 | 788.45 | 2024-05-17 11:15:00 | 771.25 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-05-21 15:15:00 | 791.00 | 2024-05-22 11:15:00 | 780.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-05-22 10:30:00 | 783.55 | 2024-05-22 11:15:00 | 780.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-05-28 09:15:00 | 787.00 | 2024-06-04 09:15:00 | 789.65 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-06-05 09:30:00 | 772.55 | 2024-06-05 14:15:00 | 871.50 | STOP_HIT | 1.00 | -12.81% |
| SELL | retest2 | 2024-06-05 10:15:00 | 779.05 | 2024-06-05 14:15:00 | 871.50 | STOP_HIT | 1.00 | -11.87% |
| SELL | retest2 | 2024-06-19 13:45:00 | 780.00 | 2024-06-25 09:15:00 | 741.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-19 13:45:00 | 780.00 | 2024-06-25 12:15:00 | 771.50 | STOP_HIT | 0.50 | 1.09% |
| SELL | retest2 | 2024-07-22 11:45:00 | 762.75 | 2024-07-23 11:15:00 | 771.30 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-07-22 12:30:00 | 763.20 | 2024-07-23 11:15:00 | 771.30 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-07-22 14:45:00 | 762.45 | 2024-07-23 11:15:00 | 771.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-23 12:15:00 | 760.35 | 2024-07-23 12:15:00 | 771.85 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-07-26 09:15:00 | 783.20 | 2024-08-02 14:15:00 | 784.55 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-08-13 10:30:00 | 866.45 | 2024-08-14 15:15:00 | 828.00 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2024-08-19 13:45:00 | 819.00 | 2024-08-19 14:15:00 | 833.35 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-08-23 10:15:00 | 914.90 | 2024-08-27 14:15:00 | 906.75 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-08-23 13:45:00 | 914.10 | 2024-08-28 09:15:00 | 909.25 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-08-26 14:00:00 | 915.75 | 2024-08-28 09:15:00 | 909.25 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-08-27 09:15:00 | 915.10 | 2024-08-28 09:15:00 | 909.25 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-27 12:15:00 | 926.00 | 2024-08-28 09:15:00 | 909.25 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-09-11 12:15:00 | 917.35 | 2024-09-12 09:15:00 | 947.50 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2024-10-11 14:45:00 | 990.00 | 2024-10-14 09:15:00 | 969.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-10-21 09:15:00 | 990.95 | 2024-10-22 10:15:00 | 941.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 990.95 | 2024-10-24 14:15:00 | 937.90 | STOP_HIT | 0.50 | 5.35% |
| BUY | retest2 | 2024-11-04 11:30:00 | 998.80 | 2024-11-05 10:15:00 | 958.30 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-11-12 11:45:00 | 1083.35 | 2024-11-13 09:15:00 | 1035.80 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2024-11-18 09:15:00 | 1019.30 | 2024-11-19 09:15:00 | 1062.90 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2024-12-05 15:00:00 | 1145.00 | 2024-12-12 09:15:00 | 1087.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-06 09:45:00 | 1143.75 | 2024-12-12 09:15:00 | 1087.84 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2024-12-06 10:45:00 | 1145.10 | 2024-12-12 10:15:00 | 1086.56 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2024-12-05 15:00:00 | 1145.00 | 2024-12-12 14:15:00 | 1089.70 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2024-12-06 09:45:00 | 1143.75 | 2024-12-12 14:15:00 | 1089.70 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2024-12-06 10:45:00 | 1145.10 | 2024-12-12 14:15:00 | 1089.70 | STOP_HIT | 0.50 | 4.84% |
| BUY | retest2 | 2024-12-27 13:15:00 | 1064.40 | 2024-12-30 09:15:00 | 1051.05 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-12-27 14:00:00 | 1075.90 | 2024-12-30 09:15:00 | 1051.05 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest1 | 2025-01-09 09:15:00 | 1223.90 | 2025-01-13 12:15:00 | 1213.90 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-10 10:30:00 | 1234.45 | 2025-01-14 09:15:00 | 1180.00 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-01-10 13:00:00 | 1235.10 | 2025-01-14 09:15:00 | 1180.00 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2025-01-10 14:45:00 | 1251.80 | 2025-01-14 09:15:00 | 1180.00 | STOP_HIT | 1.00 | -5.74% |
| BUY | retest2 | 2025-01-13 09:15:00 | 1261.00 | 2025-01-14 09:15:00 | 1180.00 | STOP_HIT | 1.00 | -6.42% |
| BUY | retest2 | 2025-02-07 11:30:00 | 1133.95 | 2025-02-10 09:15:00 | 1089.95 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2025-02-07 13:00:00 | 1132.25 | 2025-02-10 09:15:00 | 1089.95 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-02-07 14:45:00 | 1134.70 | 2025-02-10 09:15:00 | 1089.95 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest1 | 2025-02-12 09:15:00 | 1053.20 | 2025-02-13 09:15:00 | 947.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-21 09:15:00 | 945.00 | 2025-02-24 12:15:00 | 925.75 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-03-05 09:15:00 | 1008.65 | 2025-03-10 15:15:00 | 1019.95 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2025-03-05 10:15:00 | 1008.85 | 2025-03-10 15:15:00 | 1019.95 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-03-05 11:15:00 | 1007.05 | 2025-03-10 15:15:00 | 1019.95 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2025-03-05 12:15:00 | 1008.70 | 2025-03-10 15:15:00 | 1019.95 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2025-03-13 12:45:00 | 995.00 | 2025-03-17 09:15:00 | 1049.60 | STOP_HIT | 1.00 | -5.49% |
| SELL | retest2 | 2025-03-25 10:30:00 | 1022.65 | 2025-04-01 09:15:00 | 971.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 14:15:00 | 1020.05 | 2025-04-01 09:15:00 | 969.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 11:45:00 | 1021.55 | 2025-04-01 09:15:00 | 970.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 10:30:00 | 1022.65 | 2025-04-03 09:15:00 | 967.80 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2025-03-25 14:15:00 | 1020.05 | 2025-04-03 09:15:00 | 967.80 | STOP_HIT | 0.50 | 5.12% |
| SELL | retest2 | 2025-03-26 11:45:00 | 1021.55 | 2025-04-03 09:15:00 | 967.80 | STOP_HIT | 0.50 | 5.26% |
| BUY | retest2 | 2025-04-15 09:15:00 | 980.80 | 2025-04-21 09:15:00 | 1078.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-29 14:00:00 | 1021.90 | 2025-05-06 10:15:00 | 1023.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-04-30 09:45:00 | 1019.90 | 2025-05-06 10:15:00 | 1023.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-05-30 12:30:00 | 953.00 | 2025-06-02 12:15:00 | 945.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-05-30 15:00:00 | 958.80 | 2025-06-02 12:15:00 | 945.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-06-13 14:45:00 | 948.60 | 2025-06-16 13:15:00 | 961.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-16 09:15:00 | 947.40 | 2025-06-16 13:15:00 | 961.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-06-16 10:30:00 | 948.05 | 2025-06-16 13:15:00 | 961.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-02 09:15:00 | 985.70 | 2025-07-02 10:15:00 | 974.75 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest1 | 2025-07-07 12:45:00 | 1014.05 | 2025-07-08 09:15:00 | 1006.30 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2025-07-07 14:00:00 | 1015.55 | 2025-07-08 09:15:00 | 1006.30 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-28 14:00:00 | 1116.75 | 2025-07-31 12:15:00 | 1076.55 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-07-31 09:15:00 | 1103.65 | 2025-07-31 12:15:00 | 1076.55 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-08-11 09:45:00 | 1042.00 | 2025-08-12 15:15:00 | 1058.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-08-11 15:15:00 | 1047.20 | 2025-08-12 15:15:00 | 1058.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-13 15:15:00 | 1068.10 | 2025-08-14 09:15:00 | 1055.10 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-08-14 11:45:00 | 1060.00 | 2025-08-14 12:15:00 | 1055.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-08-14 15:15:00 | 1060.00 | 2025-08-18 09:15:00 | 1050.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-25 11:30:00 | 1019.10 | 2025-09-01 09:15:00 | 1027.90 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1013.00 | 2025-09-01 09:15:00 | 1027.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1049.40 | 2025-09-09 14:15:00 | 1083.20 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-09-03 09:45:00 | 1046.70 | 2025-09-09 14:15:00 | 1083.20 | STOP_HIT | 1.00 | 3.49% |
| SELL | retest2 | 2025-09-18 11:00:00 | 1039.30 | 2025-09-19 14:15:00 | 1062.20 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-09-18 13:00:00 | 1039.80 | 2025-09-19 14:15:00 | 1062.20 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-09-18 13:45:00 | 1039.90 | 2025-09-19 14:15:00 | 1062.20 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-09-30 09:45:00 | 1009.90 | 2025-10-03 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1009.05 | 2025-10-03 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-03 11:30:00 | 1011.40 | 2025-10-03 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-10-03 13:00:00 | 1011.30 | 2025-10-03 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-10-07 11:45:00 | 997.00 | 2025-10-15 11:15:00 | 990.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-10-08 10:15:00 | 995.75 | 2025-10-15 11:15:00 | 990.00 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-10-17 14:30:00 | 988.70 | 2025-10-24 09:15:00 | 987.85 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-10-20 10:45:00 | 989.00 | 2025-10-24 09:15:00 | 987.85 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-10-24 09:15:00 | 989.00 | 2025-10-24 09:15:00 | 987.85 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-11-07 10:30:00 | 1024.55 | 2025-11-10 10:15:00 | 1006.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-11-14 09:45:00 | 1058.25 | 2025-11-18 09:15:00 | 1016.00 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2025-11-14 12:30:00 | 1060.45 | 2025-11-18 09:15:00 | 1016.00 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-11-20 12:15:00 | 1003.00 | 2025-11-24 14:15:00 | 1034.00 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-11-24 09:15:00 | 982.10 | 2025-11-24 14:15:00 | 1034.00 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2025-11-26 09:15:00 | 996.45 | 2025-11-27 14:15:00 | 1011.20 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-11-26 14:15:00 | 1003.65 | 2025-11-27 14:15:00 | 1011.20 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-11-27 10:45:00 | 1000.05 | 2025-11-27 14:15:00 | 1011.20 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-11-27 11:15:00 | 999.80 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-11-27 14:00:00 | 998.95 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-11-28 09:15:00 | 999.45 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-28 10:15:00 | 989.05 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-11-28 13:30:00 | 994.40 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-01 09:15:00 | 993.55 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-12-01 10:45:00 | 994.30 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-18 09:15:00 | 961.85 | 2025-12-18 14:15:00 | 986.80 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-12-23 09:30:00 | 1007.80 | 2026-01-02 09:15:00 | 1039.00 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2026-01-05 11:15:00 | 1057.20 | 2026-01-06 12:15:00 | 1050.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-01-05 11:45:00 | 1060.60 | 2026-01-06 12:15:00 | 1050.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-01-05 12:45:00 | 1056.30 | 2026-01-06 12:15:00 | 1050.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-01-05 13:30:00 | 1056.60 | 2026-01-06 12:15:00 | 1050.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-01-21 10:15:00 | 946.10 | 2026-01-21 14:15:00 | 985.90 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2026-01-21 10:45:00 | 946.30 | 2026-01-21 14:15:00 | 985.90 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-02-19 15:15:00 | 1011.30 | 2026-02-20 12:15:00 | 1005.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-20 11:00:00 | 1010.80 | 2026-02-20 12:15:00 | 1005.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-02-20 12:00:00 | 1011.45 | 2026-02-20 12:15:00 | 1005.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-26 12:15:00 | 989.00 | 2026-03-04 13:15:00 | 939.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:30:00 | 986.90 | 2026-03-04 13:15:00 | 937.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:15:00 | 989.00 | 2026-03-05 12:15:00 | 951.70 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-02-26 13:30:00 | 986.90 | 2026-03-05 12:15:00 | 951.70 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-03-02 09:15:00 | 972.20 | 2026-03-09 09:15:00 | 923.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 972.20 | 2026-03-09 14:15:00 | 945.70 | STOP_HIT | 0.50 | 2.73% |
| BUY | retest2 | 2026-03-12 11:45:00 | 954.40 | 2026-03-12 15:15:00 | 945.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-03-12 12:15:00 | 953.20 | 2026-03-12 15:15:00 | 945.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-03-12 15:00:00 | 954.50 | 2026-03-12 15:15:00 | 945.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 894.20 | 2026-03-30 09:15:00 | 849.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 896.00 | 2026-03-30 09:15:00 | 851.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 895.10 | 2026-03-30 09:15:00 | 850.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 894.20 | 2026-03-30 14:15:00 | 863.00 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2026-03-25 12:15:00 | 896.00 | 2026-03-30 14:15:00 | 863.00 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2026-03-25 14:15:00 | 895.10 | 2026-03-30 14:15:00 | 863.00 | STOP_HIT | 0.50 | 3.59% |
| BUY | retest2 | 2026-04-06 12:15:00 | 885.65 | 2026-04-09 11:15:00 | 974.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 12:45:00 | 886.25 | 2026-04-09 11:15:00 | 974.88 | TARGET_HIT | 1.00 | 10.00% |
