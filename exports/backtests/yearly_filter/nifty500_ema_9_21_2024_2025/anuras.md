# Anupam Rasayan India Ltd. (ANURAS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1369.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 175 |
| ALERT1 | 101 |
| ALERT2 | 102 |
| ALERT2_SKIP | 68 |
| ALERT3 | 307 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 177 |
| PARTIAL | 23 |
| TARGET_HIT | 7 |
| STOP_HIT | 175 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 205 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 73 / 132
- **Target hits / Stop hits / Partials:** 7 / 175 / 23
- **Avg / median % per leg:** 0.43% / -0.76%
- **Sum % (uncompounded):** 87.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 16 | 25.8% | 6 | 56 | 0 | 0.27% | 16.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.20% | -2.4% |
| BUY @ 3rd Alert (retest2) | 60 | 16 | 26.7% | 6 | 54 | 0 | 0.31% | 18.9% |
| SELL (all) | 143 | 57 | 39.9% | 1 | 119 | 23 | 0.50% | 70.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.13% | -3.4% |
| SELL @ 3rd Alert (retest2) | 140 | 57 | 40.7% | 1 | 116 | 23 | 0.53% | 74.3% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.16% | -5.8% |
| retest2 (combined) | 200 | 73 | 36.5% | 7 | 170 | 23 | 0.47% | 93.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 14:15:00 | 794.75 | 789.40 | 789.26 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 09:15:00 | 788.40 | 789.02 | 789.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 10:15:00 | 785.20 | 788.26 | 788.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 13:15:00 | 785.30 | 784.47 | 785.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 13:15:00 | 785.30 | 784.47 | 785.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 785.30 | 784.47 | 785.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 14:00:00 | 785.30 | 784.47 | 785.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 790.20 | 785.61 | 786.22 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 790.00 | 786.71 | 786.63 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 11:15:00 | 785.00 | 786.37 | 786.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-18 12:15:00 | 780.00 | 785.10 | 785.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 14:15:00 | 784.80 | 782.37 | 783.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 14:15:00 | 784.80 | 782.37 | 783.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 784.80 | 782.37 | 783.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 784.80 | 782.37 | 783.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 785.05 | 782.91 | 783.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 785.00 | 782.91 | 783.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 783.40 | 783.30 | 783.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 13:00:00 | 783.15 | 783.44 | 783.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 14:15:00 | 773.45 | 763.48 | 762.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 14:15:00 | 773.45 | 763.48 | 762.13 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 743.30 | 759.97 | 761.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 737.35 | 752.33 | 757.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 755.00 | 743.94 | 749.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 13:15:00 | 755.00 | 743.94 | 749.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 755.00 | 743.94 | 749.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 755.00 | 743.94 | 749.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 775.05 | 750.17 | 751.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 775.05 | 750.17 | 751.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 774.85 | 756.84 | 754.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 14:15:00 | 777.00 | 771.81 | 769.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 13:15:00 | 779.95 | 780.00 | 776.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 14:00:00 | 779.95 | 780.00 | 776.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 781.15 | 786.54 | 784.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 781.40 | 786.54 | 784.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 782.50 | 785.73 | 784.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 780.00 | 785.73 | 784.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 782.50 | 785.09 | 784.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:00:00 | 782.50 | 785.09 | 784.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 782.30 | 784.53 | 783.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:30:00 | 781.45 | 784.53 | 783.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 777.45 | 782.51 | 783.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 772.55 | 779.19 | 780.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 12:15:00 | 773.90 | 773.63 | 776.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 773.25 | 773.18 | 775.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 773.25 | 773.18 | 775.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 773.25 | 773.18 | 775.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 768.30 | 766.75 | 769.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 769.15 | 766.75 | 769.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 771.50 | 767.70 | 769.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 771.50 | 767.70 | 769.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 770.35 | 768.23 | 769.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 772.25 | 768.23 | 769.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 764.10 | 766.05 | 767.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 764.20 | 766.05 | 767.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 768.00 | 766.57 | 767.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:45:00 | 767.95 | 766.57 | 767.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 767.25 | 766.70 | 767.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:30:00 | 764.55 | 766.63 | 767.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 765.00 | 765.20 | 765.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 12:30:00 | 765.60 | 765.02 | 765.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:30:00 | 765.60 | 765.02 | 765.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 765.40 | 765.09 | 765.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 765.40 | 765.09 | 765.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 764.45 | 764.97 | 765.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 763.75 | 764.97 | 765.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 763.00 | 764.57 | 765.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 09:45:00 | 761.00 | 762.94 | 763.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 14:15:00 | 752.15 | 746.68 | 746.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 14:15:00 | 752.15 | 746.68 | 746.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 772.30 | 755.88 | 752.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-22 10:15:00 | 781.75 | 782.66 | 777.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-22 11:00:00 | 781.75 | 782.66 | 777.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 797.30 | 795.23 | 787.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 789.25 | 795.23 | 787.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 792.10 | 794.89 | 790.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 790.85 | 794.89 | 790.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 787.50 | 792.95 | 790.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:30:00 | 789.45 | 792.95 | 790.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 12:15:00 | 792.20 | 792.80 | 790.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:30:00 | 788.75 | 792.80 | 790.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 793.40 | 792.92 | 790.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 13:30:00 | 791.60 | 792.92 | 790.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 791.30 | 792.85 | 790.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:15:00 | 789.85 | 792.85 | 790.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 789.95 | 792.27 | 790.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 10:45:00 | 795.45 | 791.82 | 790.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 13:15:00 | 781.90 | 788.58 | 789.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 781.90 | 788.58 | 789.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 15:15:00 | 778.00 | 785.73 | 787.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 14:15:00 | 785.20 | 784.64 | 786.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 14:15:00 | 785.20 | 784.64 | 786.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 785.20 | 784.64 | 786.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:00:00 | 785.20 | 784.64 | 786.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 785.00 | 783.81 | 785.59 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 805.00 | 787.12 | 785.10 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 785.80 | 789.39 | 789.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 771.65 | 785.84 | 788.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 781.20 | 777.81 | 782.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 14:15:00 | 781.20 | 777.81 | 782.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 781.20 | 777.81 | 782.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 15:00:00 | 781.20 | 777.81 | 782.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 782.90 | 778.82 | 782.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:15:00 | 778.75 | 778.82 | 782.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 782.10 | 779.48 | 782.45 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 783.50 | 781.49 | 781.46 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 779.00 | 781.13 | 781.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 14:15:00 | 778.05 | 780.51 | 781.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 14:15:00 | 779.60 | 777.80 | 779.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 779.60 | 777.80 | 779.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 779.60 | 777.80 | 779.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 779.60 | 777.80 | 779.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 780.00 | 778.24 | 779.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 775.70 | 778.24 | 779.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:15:00 | 736.91 | 772.17 | 773.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-14 09:15:00 | 774.00 | 772.17 | 773.53 | SL hit (close>static) qty=0.50 sl=772.17 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 788.10 | 774.23 | 773.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 803.95 | 789.58 | 782.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 802.35 | 802.79 | 798.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:15:00 | 801.55 | 802.79 | 798.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 792.25 | 801.64 | 801.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:00:00 | 792.25 | 801.64 | 801.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 798.95 | 801.10 | 800.85 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 15:15:00 | 797.50 | 800.38 | 800.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 09:15:00 | 793.65 | 799.04 | 799.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 779.80 | 775.64 | 779.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 779.80 | 775.64 | 779.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 779.80 | 775.64 | 779.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 779.80 | 775.64 | 779.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 782.00 | 776.91 | 779.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:00:00 | 779.00 | 777.97 | 779.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 777.85 | 779.73 | 780.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 09:30:00 | 776.75 | 777.08 | 778.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 778.95 | 777.19 | 777.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 777.95 | 777.34 | 777.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 774.15 | 777.34 | 777.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:45:00 | 776.00 | 773.07 | 773.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 779.25 | 774.30 | 774.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 779.25 | 774.30 | 774.04 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 11:15:00 | 772.55 | 774.07 | 774.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 766.20 | 772.16 | 773.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 771.25 | 769.72 | 771.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 11:15:00 | 771.25 | 769.72 | 771.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 771.25 | 769.72 | 771.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:00:00 | 771.25 | 769.72 | 771.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 763.70 | 768.52 | 770.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:15:00 | 761.45 | 767.51 | 770.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 14:00:00 | 761.75 | 763.97 | 766.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 771.00 | 767.36 | 767.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 771.00 | 767.36 | 767.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 774.70 | 769.46 | 768.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 768.65 | 769.48 | 768.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 13:00:00 | 768.65 | 769.48 | 768.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 768.10 | 769.20 | 768.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 768.10 | 769.20 | 768.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 769.50 | 769.26 | 768.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 768.00 | 769.26 | 768.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 768.00 | 769.01 | 768.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 767.20 | 769.01 | 768.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 766.60 | 768.53 | 768.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 766.60 | 768.53 | 768.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 765.80 | 767.98 | 768.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 13:15:00 | 764.80 | 767.14 | 767.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 14:15:00 | 770.30 | 767.77 | 767.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 770.30 | 767.77 | 767.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 770.30 | 767.77 | 767.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 770.30 | 767.77 | 767.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 769.90 | 768.19 | 768.09 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 766.40 | 768.04 | 768.05 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 15:15:00 | 773.90 | 768.81 | 768.29 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 764.15 | 767.76 | 767.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 763.25 | 765.58 | 766.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 763.00 | 762.55 | 764.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 13:15:00 | 763.00 | 762.55 | 764.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 763.00 | 762.55 | 764.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 13:00:00 | 757.50 | 761.05 | 762.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 13:30:00 | 758.10 | 760.84 | 762.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 15:00:00 | 758.60 | 760.39 | 762.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:30:00 | 757.40 | 759.81 | 761.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 751.90 | 752.48 | 756.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 15:15:00 | 743.00 | 751.76 | 754.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:45:00 | 745.45 | 749.26 | 752.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 741.50 | 736.69 | 736.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 741.50 | 736.69 | 736.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 12:15:00 | 765.00 | 743.28 | 739.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 13:15:00 | 757.00 | 759.33 | 752.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 14:00:00 | 757.00 | 759.33 | 752.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 751.45 | 757.76 | 752.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:00:00 | 751.45 | 757.76 | 752.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 756.00 | 757.41 | 752.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 739.55 | 757.41 | 752.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 749.90 | 755.90 | 752.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 743.60 | 755.90 | 752.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 750.00 | 754.72 | 751.94 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 745.30 | 750.08 | 750.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 732.15 | 745.84 | 748.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 736.00 | 733.64 | 738.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:45:00 | 733.95 | 733.64 | 738.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 734.50 | 733.81 | 738.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 10:45:00 | 733.00 | 737.25 | 738.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:15:00 | 733.55 | 737.25 | 738.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:30:00 | 733.30 | 735.98 | 737.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 13:30:00 | 734.05 | 735.04 | 736.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 735.10 | 735.05 | 735.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 735.10 | 735.05 | 735.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 734.00 | 734.84 | 735.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 735.45 | 734.84 | 735.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 733.00 | 734.47 | 735.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:30:00 | 731.80 | 733.78 | 734.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 13:45:00 | 731.80 | 733.62 | 734.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 14:30:00 | 732.05 | 733.67 | 734.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 731.30 | 733.09 | 734.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 732.00 | 732.87 | 734.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 732.00 | 732.87 | 734.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 729.30 | 731.25 | 732.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 729.30 | 731.25 | 732.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 731.95 | 731.39 | 732.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:00:00 | 725.00 | 729.64 | 731.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 10:15:00 | 725.00 | 728.33 | 729.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 13:45:00 | 725.50 | 724.37 | 725.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 14:45:00 | 725.00 | 724.74 | 725.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 730.00 | 725.79 | 726.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 728.05 | 725.79 | 726.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 727.00 | 726.04 | 726.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 719.70 | 724.30 | 725.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 720.00 | 722.67 | 724.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 696.87 | 712.22 | 717.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 697.35 | 712.22 | 717.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 696.35 | 709.47 | 716.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 15:15:00 | 696.63 | 709.47 | 716.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 695.21 | 707.93 | 714.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 695.21 | 707.93 | 714.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 695.45 | 707.93 | 714.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 694.73 | 707.93 | 714.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 688.75 | 694.87 | 700.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 688.75 | 694.87 | 700.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 689.23 | 694.87 | 700.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 688.75 | 694.87 | 700.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 684.00 | 692.70 | 699.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 11:15:00 | 683.72 | 691.32 | 698.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 693.00 | 690.09 | 695.65 | SL hit (close>ema200) qty=0.50 sl=690.09 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 721.00 | 696.74 | 695.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 15:15:00 | 735.25 | 704.44 | 699.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 728.50 | 728.94 | 722.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:30:00 | 730.60 | 729.45 | 723.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 11:30:00 | 731.00 | 730.06 | 724.06 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 722.00 | 731.71 | 728.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 722.00 | 731.71 | 728.40 | SL hit (close<ema400) qty=1.00 sl=728.40 alert=retest1 |

### Cycle 28 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 731.30 | 735.56 | 735.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 727.70 | 733.07 | 734.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 736.45 | 733.75 | 734.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 14:15:00 | 736.45 | 733.75 | 734.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 736.45 | 733.75 | 734.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 15:00:00 | 736.45 | 733.75 | 734.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 736.00 | 734.20 | 734.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 739.00 | 734.20 | 734.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 735.10 | 734.38 | 734.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:45:00 | 733.15 | 734.23 | 734.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 732.50 | 733.91 | 734.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 15:15:00 | 741.00 | 724.02 | 723.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 741.00 | 724.02 | 723.58 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 718.75 | 722.97 | 723.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 13:15:00 | 715.75 | 719.56 | 721.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 720.10 | 718.71 | 720.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 720.10 | 718.71 | 720.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 720.10 | 718.71 | 720.37 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 728.05 | 721.39 | 721.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 13:15:00 | 732.00 | 724.39 | 722.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 719.60 | 724.79 | 723.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 719.60 | 724.79 | 723.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 719.60 | 724.79 | 723.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:45:00 | 717.90 | 724.79 | 723.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 718.15 | 723.47 | 723.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:00:00 | 718.15 | 723.47 | 723.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 717.55 | 722.28 | 722.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 12:15:00 | 714.75 | 720.78 | 721.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 14:15:00 | 728.40 | 721.66 | 722.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 14:15:00 | 728.40 | 721.66 | 722.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 728.40 | 721.66 | 722.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 15:00:00 | 728.40 | 721.66 | 722.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 717.00 | 720.73 | 721.55 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 724.60 | 722.36 | 722.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 12:15:00 | 731.15 | 724.12 | 722.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 13:15:00 | 732.00 | 732.20 | 728.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:45:00 | 731.75 | 732.20 | 728.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 730.15 | 731.79 | 728.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 730.15 | 731.79 | 728.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 728.00 | 731.03 | 728.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 15:00:00 | 732.95 | 729.27 | 728.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 13:15:00 | 726.65 | 728.06 | 728.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 13:15:00 | 726.65 | 728.06 | 728.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 14:15:00 | 725.10 | 727.47 | 727.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 727.50 | 727.24 | 727.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 727.50 | 727.24 | 727.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 727.50 | 727.24 | 727.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:15:00 | 733.40 | 727.24 | 727.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 730.25 | 727.84 | 727.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 729.95 | 727.84 | 727.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 11:15:00 | 733.65 | 729.00 | 728.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 15:15:00 | 742.00 | 733.68 | 731.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 11:15:00 | 731.40 | 733.32 | 731.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 11:15:00 | 731.40 | 733.32 | 731.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 731.40 | 733.32 | 731.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 731.50 | 733.32 | 731.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 732.00 | 733.05 | 731.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 731.80 | 733.05 | 731.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 732.95 | 733.03 | 731.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:45:00 | 731.60 | 733.03 | 731.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 732.05 | 732.98 | 731.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 730.80 | 732.98 | 731.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 728.50 | 732.09 | 731.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 729.80 | 732.09 | 731.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 727.75 | 731.22 | 731.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 11:15:00 | 726.90 | 730.36 | 730.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 732.00 | 729.81 | 730.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 14:15:00 | 732.00 | 729.81 | 730.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 732.00 | 729.81 | 730.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 732.00 | 729.81 | 730.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 15:15:00 | 739.20 | 731.69 | 731.19 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 12:15:00 | 733.90 | 735.45 | 735.46 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 738.05 | 735.74 | 735.57 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 733.60 | 735.20 | 735.35 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 737.40 | 735.67 | 735.52 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 14:15:00 | 733.25 | 735.19 | 735.31 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 12:15:00 | 737.85 | 735.74 | 735.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 765.50 | 741.85 | 738.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 15:15:00 | 764.60 | 768.29 | 760.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 09:15:00 | 759.85 | 768.29 | 760.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 749.25 | 764.49 | 759.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 749.25 | 764.49 | 759.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 751.25 | 761.84 | 759.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 755.50 | 761.84 | 759.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 13:15:00 | 751.10 | 756.74 | 757.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 751.10 | 756.74 | 757.12 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 14:15:00 | 761.45 | 757.68 | 757.51 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 738.80 | 754.12 | 755.93 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 759.00 | 755.10 | 754.76 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 15:15:00 | 752.50 | 754.88 | 754.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 748.75 | 753.65 | 754.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 14:15:00 | 743.40 | 741.45 | 745.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 14:15:00 | 743.40 | 741.45 | 745.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 743.40 | 741.45 | 745.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 15:00:00 | 743.40 | 741.45 | 745.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 738.35 | 736.80 | 740.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 738.35 | 736.80 | 740.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 739.30 | 737.30 | 740.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 718.10 | 729.60 | 735.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 12:45:00 | 721.95 | 724.93 | 730.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 09:30:00 | 721.35 | 722.66 | 727.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 10:45:00 | 722.25 | 722.53 | 726.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 725.60 | 723.27 | 726.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:30:00 | 722.05 | 723.23 | 725.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 719.60 | 721.78 | 724.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 727.20 | 717.82 | 719.09 | SL hit (close>static) qty=1.00 sl=726.90 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 740.40 | 723.02 | 721.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 12:15:00 | 745.05 | 727.42 | 723.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 717.65 | 727.04 | 724.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 717.65 | 727.04 | 724.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 717.65 | 727.04 | 724.77 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 15:15:00 | 718.00 | 722.78 | 723.39 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 726.95 | 723.89 | 723.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 728.25 | 724.76 | 724.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 726.60 | 727.35 | 726.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 13:15:00 | 726.60 | 727.35 | 726.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 726.60 | 727.35 | 726.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 15:15:00 | 729.00 | 727.38 | 726.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 718.05 | 725.77 | 725.69 | SL hit (close<static) qty=1.00 sl=725.50 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 713.75 | 723.37 | 724.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 712.00 | 721.09 | 723.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 721.70 | 708.43 | 710.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 721.70 | 708.43 | 710.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 721.70 | 708.43 | 710.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:45:00 | 723.15 | 708.43 | 710.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 718.40 | 710.43 | 711.60 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 721.70 | 712.68 | 712.51 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 697.65 | 711.15 | 712.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 679.05 | 695.80 | 703.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 14:15:00 | 689.90 | 685.20 | 694.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-13 15:00:00 | 689.90 | 685.20 | 694.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 15:15:00 | 695.00 | 687.16 | 694.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 09:15:00 | 686.50 | 687.16 | 694.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:15:00 | 684.85 | 681.05 | 682.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 15:15:00 | 693.75 | 684.48 | 684.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 15:15:00 | 693.75 | 684.48 | 684.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 695.10 | 686.61 | 685.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 14:15:00 | 688.50 | 690.09 | 687.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 15:00:00 | 688.50 | 690.09 | 687.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 691.00 | 690.28 | 688.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 09:30:00 | 702.80 | 691.83 | 688.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:15:00 | 694.95 | 692.41 | 689.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 12:00:00 | 695.00 | 692.93 | 690.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:45:00 | 695.00 | 693.59 | 690.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 700.00 | 698.54 | 695.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 687.80 | 698.54 | 695.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 695.35 | 697.90 | 695.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 690.65 | 697.90 | 695.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 692.55 | 696.83 | 695.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 692.55 | 696.83 | 695.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 687.85 | 695.04 | 694.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 687.85 | 695.04 | 694.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-22 12:15:00 | 688.85 | 693.80 | 694.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 12:15:00 | 688.85 | 693.80 | 694.05 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 697.70 | 694.42 | 694.02 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 692.80 | 695.29 | 695.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 690.85 | 694.40 | 695.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 11:15:00 | 658.95 | 658.87 | 666.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 12:00:00 | 658.95 | 658.87 | 666.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 681.70 | 660.17 | 663.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 681.70 | 660.17 | 663.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 682.45 | 664.63 | 665.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:45:00 | 683.15 | 664.63 | 665.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 681.25 | 667.95 | 666.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 689.00 | 679.21 | 673.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 688.20 | 688.97 | 682.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 685.20 | 688.97 | 682.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 685.75 | 690.03 | 686.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 685.75 | 690.03 | 686.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 696.85 | 691.40 | 686.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:15:00 | 699.25 | 691.40 | 686.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 14:15:00 | 716.75 | 729.53 | 730.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 14:15:00 | 716.75 | 729.53 | 730.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 694.00 | 720.64 | 726.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 09:15:00 | 700.85 | 683.93 | 695.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 700.85 | 683.93 | 695.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 700.85 | 683.93 | 695.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:15:00 | 687.80 | 685.78 | 695.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 653.41 | 669.38 | 682.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-18 10:15:00 | 619.02 | 636.29 | 655.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 644.50 | 641.74 | 641.65 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 634.95 | 640.38 | 641.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 630.40 | 638.39 | 640.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 14:15:00 | 635.20 | 635.20 | 637.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 14:45:00 | 633.70 | 635.20 | 637.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 641.00 | 636.36 | 638.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:30:00 | 630.80 | 637.63 | 638.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 12:15:00 | 641.60 | 639.11 | 639.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 12:15:00 | 641.60 | 639.11 | 639.02 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 636.10 | 638.68 | 638.85 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 639.85 | 638.97 | 638.95 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 637.50 | 638.67 | 638.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 636.75 | 638.29 | 638.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 638.35 | 638.30 | 638.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 12:15:00 | 638.35 | 638.30 | 638.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 638.35 | 638.30 | 638.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:00:00 | 638.35 | 638.30 | 638.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 638.10 | 638.26 | 638.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 638.10 | 638.26 | 638.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 635.95 | 637.80 | 638.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:45:00 | 638.05 | 637.80 | 638.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 639.70 | 638.18 | 638.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 633.40 | 638.18 | 638.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 601.73 | 620.89 | 628.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 11:15:00 | 634.35 | 623.57 | 628.04 | SL hit (close>ema200) qty=0.50 sl=623.57 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 652.85 | 630.96 | 629.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 14:15:00 | 701.90 | 645.15 | 635.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 12:15:00 | 784.00 | 784.08 | 771.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-11 13:15:00 | 783.65 | 784.08 | 771.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 783.05 | 783.29 | 779.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 786.90 | 783.28 | 780.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 770.25 | 781.11 | 780.29 | SL hit (close<static) qty=1.00 sl=774.55 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 11:15:00 | 768.75 | 777.67 | 778.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 764.55 | 775.04 | 777.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 15:15:00 | 778.80 | 773.25 | 775.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 15:15:00 | 778.80 | 773.25 | 775.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 778.80 | 773.25 | 775.84 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 786.00 | 776.86 | 776.09 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 14:15:00 | 774.00 | 777.15 | 777.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 09:15:00 | 768.60 | 774.94 | 776.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 12:15:00 | 775.05 | 773.46 | 775.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 12:15:00 | 775.05 | 773.46 | 775.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 775.05 | 773.46 | 775.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:00:00 | 775.05 | 773.46 | 775.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 769.70 | 772.71 | 774.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 11:30:00 | 765.00 | 770.76 | 772.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 12:15:00 | 766.40 | 770.76 | 772.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 765.20 | 769.71 | 771.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:00:00 | 767.00 | 769.17 | 771.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 763.50 | 768.03 | 770.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:30:00 | 761.00 | 766.95 | 769.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 12:30:00 | 761.70 | 765.79 | 768.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 15:00:00 | 761.60 | 764.69 | 767.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 15:15:00 | 778.25 | 755.62 | 759.42 | SL hit (close>static) qty=1.00 sl=776.10 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 15:15:00 | 764.70 | 754.73 | 753.80 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 750.60 | 753.40 | 753.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 749.80 | 752.68 | 753.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 753.10 | 752.76 | 753.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 753.10 | 752.76 | 753.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 753.10 | 752.76 | 753.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 753.10 | 752.76 | 753.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 753.75 | 752.96 | 753.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 753.75 | 752.96 | 753.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 752.20 | 752.81 | 753.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 748.00 | 751.87 | 752.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 749.40 | 751.82 | 752.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 14:15:00 | 761.50 | 752.68 | 752.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 761.50 | 752.68 | 752.43 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 742.00 | 750.61 | 751.68 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 15:15:00 | 763.00 | 752.69 | 751.72 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 709.50 | 744.05 | 747.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 10:15:00 | 696.65 | 734.57 | 743.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 714.00 | 710.45 | 724.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 714.00 | 710.45 | 724.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 714.00 | 710.45 | 724.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:45:00 | 703.95 | 708.31 | 722.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 694.40 | 709.06 | 716.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 15:15:00 | 711.00 | 703.57 | 709.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:45:00 | 710.05 | 706.08 | 709.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 710.10 | 706.88 | 709.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:45:00 | 710.35 | 706.88 | 709.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 709.50 | 707.40 | 709.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:30:00 | 709.50 | 707.40 | 709.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 716.05 | 709.13 | 710.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 13:00:00 | 716.05 | 709.13 | 710.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 707.65 | 708.84 | 710.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 14:15:00 | 698.05 | 708.84 | 710.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 09:30:00 | 698.50 | 704.69 | 707.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 13:15:00 | 718.80 | 705.42 | 706.79 | SL hit (close>static) qty=1.00 sl=718.10 alert=retest2 |

### Cycle 77 — BUY (started 2025-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 14:15:00 | 743.65 | 713.07 | 710.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 10:15:00 | 782.40 | 738.86 | 723.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 14:15:00 | 800.25 | 801.81 | 787.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 15:00:00 | 800.25 | 801.81 | 787.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 803.45 | 805.33 | 796.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 803.45 | 805.33 | 796.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 805.35 | 805.80 | 799.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 803.20 | 805.80 | 799.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 801.25 | 805.42 | 800.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:00:00 | 801.25 | 805.42 | 800.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 813.00 | 806.94 | 801.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 15:00:00 | 813.00 | 806.94 | 801.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 812.05 | 828.71 | 819.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 812.05 | 828.71 | 819.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 836.35 | 830.24 | 821.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 809.90 | 830.24 | 821.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 851.25 | 858.55 | 851.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 851.25 | 858.55 | 851.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 863.55 | 859.55 | 852.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 859.20 | 859.55 | 852.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 860.75 | 859.79 | 853.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 878.75 | 866.58 | 861.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 14:45:00 | 875.05 | 870.60 | 866.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 15:15:00 | 874.60 | 870.60 | 866.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 875.20 | 871.53 | 867.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 869.60 | 871.40 | 868.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 869.10 | 871.40 | 868.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 885.40 | 874.20 | 869.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:00:00 | 893.20 | 883.44 | 877.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 12:15:00 | 881.45 | 889.35 | 890.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 12:15:00 | 881.45 | 889.35 | 890.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 13:15:00 | 879.60 | 887.40 | 889.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 890.00 | 887.92 | 889.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 14:15:00 | 890.00 | 887.92 | 889.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 890.00 | 887.92 | 889.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 890.00 | 887.92 | 889.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 892.55 | 888.85 | 889.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 915.45 | 888.85 | 889.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 911.00 | 893.28 | 891.54 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 15:15:00 | 891.05 | 895.71 | 896.27 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 906.15 | 897.27 | 896.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 910.80 | 899.97 | 898.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 14:15:00 | 948.85 | 949.39 | 932.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 15:00:00 | 948.85 | 949.39 | 932.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 952.00 | 948.80 | 940.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:30:00 | 946.10 | 948.80 | 940.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 942.00 | 948.08 | 942.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 942.00 | 948.08 | 942.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 956.95 | 949.86 | 943.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 933.25 | 949.86 | 943.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 955.60 | 957.53 | 953.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:30:00 | 969.70 | 960.61 | 957.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:00:00 | 970.00 | 960.61 | 957.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 946.00 | 957.60 | 956.79 | SL hit (close<static) qty=1.00 sl=952.80 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 942.00 | 954.48 | 955.44 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 985.75 | 959.22 | 957.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 13:15:00 | 1005.00 | 991.86 | 988.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 10:15:00 | 994.00 | 996.39 | 991.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 11:00:00 | 994.00 | 996.39 | 991.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 994.00 | 995.91 | 992.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 996.00 | 995.91 | 992.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 999.60 | 996.51 | 993.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:30:00 | 1006.50 | 996.51 | 993.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1003.10 | 999.49 | 995.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:15:00 | 1005.80 | 999.49 | 995.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:30:00 | 1007.00 | 1001.47 | 998.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-11 13:15:00 | 1106.38 | 1089.60 | 1074.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1105.00 | 1114.62 | 1115.06 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 13:15:00 | 1120.90 | 1115.87 | 1115.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 1125.00 | 1118.74 | 1117.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 1121.40 | 1122.39 | 1119.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 11:15:00 | 1121.40 | 1122.39 | 1119.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1121.40 | 1122.39 | 1119.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1121.40 | 1122.39 | 1119.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1114.00 | 1120.72 | 1118.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 1114.00 | 1120.72 | 1118.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1127.80 | 1122.13 | 1119.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1139.30 | 1128.47 | 1125.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:15:00 | 1137.00 | 1129.40 | 1126.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 13:45:00 | 1138.50 | 1142.43 | 1140.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 1125.00 | 1136.29 | 1137.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 1125.00 | 1136.29 | 1137.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 14:15:00 | 1119.60 | 1128.18 | 1132.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 15:15:00 | 1138.00 | 1130.14 | 1133.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 15:15:00 | 1138.00 | 1130.14 | 1133.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1138.00 | 1130.14 | 1133.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:15:00 | 1110.10 | 1131.58 | 1132.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 1140.00 | 1133.70 | 1133.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 15:15:00 | 1140.00 | 1133.70 | 1133.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 1142.00 | 1138.98 | 1136.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 1138.70 | 1139.01 | 1137.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 1138.70 | 1139.01 | 1137.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1138.70 | 1139.01 | 1137.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 1148.10 | 1141.74 | 1138.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 12:15:00 | 1137.90 | 1140.09 | 1140.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 1137.90 | 1140.09 | 1140.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 15:15:00 | 1130.10 | 1136.95 | 1138.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 1137.30 | 1136.01 | 1137.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 11:15:00 | 1137.30 | 1136.01 | 1137.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1137.30 | 1136.01 | 1137.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 1137.30 | 1136.01 | 1137.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1145.70 | 1137.95 | 1138.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 1145.70 | 1137.95 | 1138.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 1147.50 | 1139.86 | 1139.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 1151.90 | 1143.90 | 1141.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1141.80 | 1145.03 | 1142.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 11:15:00 | 1141.80 | 1145.03 | 1142.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1141.80 | 1145.03 | 1142.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 1140.50 | 1145.03 | 1142.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1143.40 | 1144.70 | 1142.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:45:00 | 1148.30 | 1144.56 | 1142.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 1137.30 | 1143.11 | 1142.17 | SL hit (close<static) qty=1.00 sl=1141.60 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 1132.00 | 1140.89 | 1141.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 13:15:00 | 1127.60 | 1134.91 | 1137.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 1148.40 | 1135.17 | 1137.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1148.40 | 1135.17 | 1137.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1148.40 | 1135.17 | 1137.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1148.40 | 1135.17 | 1137.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 10:15:00 | 1151.70 | 1138.47 | 1138.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 11:15:00 | 1157.20 | 1142.22 | 1140.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 12:15:00 | 1140.50 | 1141.88 | 1140.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1140.50 | 1141.88 | 1140.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1140.50 | 1141.88 | 1140.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 1140.50 | 1141.88 | 1140.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1141.40 | 1141.78 | 1140.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:15:00 | 1142.20 | 1141.78 | 1140.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1141.20 | 1141.66 | 1140.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:30:00 | 1145.50 | 1141.66 | 1140.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1141.80 | 1141.69 | 1140.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 1147.50 | 1141.69 | 1140.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 1144.40 | 1147.05 | 1146.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 1140.00 | 1145.03 | 1145.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1140.00 | 1145.03 | 1145.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 1138.90 | 1143.80 | 1144.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 1142.70 | 1142.17 | 1143.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 1142.70 | 1142.17 | 1143.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1149.30 | 1142.82 | 1143.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:30:00 | 1141.40 | 1142.69 | 1143.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:00:00 | 1140.90 | 1142.69 | 1143.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 1146.80 | 1144.16 | 1143.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1146.80 | 1144.16 | 1143.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1155.80 | 1146.48 | 1144.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1148.80 | 1154.01 | 1150.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1148.80 | 1154.01 | 1150.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1148.80 | 1154.01 | 1150.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 1149.30 | 1154.01 | 1150.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1140.80 | 1151.37 | 1149.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1140.80 | 1151.37 | 1149.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1147.40 | 1150.57 | 1149.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 1141.70 | 1150.57 | 1149.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1156.60 | 1151.78 | 1149.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:30:00 | 1162.50 | 1153.96 | 1151.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 1146.50 | 1153.34 | 1152.29 | SL hit (close<static) qty=1.00 sl=1146.80 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 1149.00 | 1151.83 | 1151.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 1145.10 | 1149.23 | 1150.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 1149.90 | 1149.24 | 1150.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 1149.90 | 1149.24 | 1150.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 1149.90 | 1149.24 | 1150.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:15:00 | 1148.80 | 1149.24 | 1150.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 1148.80 | 1149.16 | 1150.12 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 11:15:00 | 1156.20 | 1150.81 | 1150.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 15:15:00 | 1160.80 | 1155.07 | 1152.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 13:15:00 | 1156.40 | 1156.56 | 1154.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 13:15:00 | 1156.40 | 1156.56 | 1154.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1156.40 | 1156.56 | 1154.61 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 1150.00 | 1153.28 | 1153.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 1147.20 | 1149.34 | 1151.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 1148.70 | 1147.18 | 1149.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 1148.70 | 1147.18 | 1149.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1148.70 | 1147.18 | 1149.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 1148.70 | 1147.18 | 1149.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1149.00 | 1147.54 | 1149.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1141.80 | 1147.54 | 1149.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1141.50 | 1144.38 | 1146.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 1150.50 | 1139.45 | 1138.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 1150.50 | 1139.45 | 1138.55 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1130.70 | 1140.56 | 1141.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 14:15:00 | 1125.00 | 1131.83 | 1134.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 15:15:00 | 1150.00 | 1130.93 | 1131.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 15:15:00 | 1150.00 | 1130.93 | 1131.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1150.00 | 1130.93 | 1131.70 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 1153.00 | 1134.85 | 1133.18 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 1126.10 | 1131.24 | 1131.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 15:15:00 | 1120.20 | 1128.01 | 1129.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 13:15:00 | 1124.20 | 1119.96 | 1122.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 13:15:00 | 1124.20 | 1119.96 | 1122.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1124.20 | 1119.96 | 1122.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 1124.20 | 1119.96 | 1122.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1136.30 | 1123.23 | 1123.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 1145.00 | 1123.23 | 1123.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1133.00 | 1125.18 | 1124.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 1138.90 | 1130.06 | 1127.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 1148.30 | 1150.37 | 1144.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 14:15:00 | 1148.30 | 1150.37 | 1144.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1148.30 | 1150.37 | 1144.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 1148.30 | 1150.37 | 1144.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1168.00 | 1153.89 | 1146.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 1146.60 | 1153.89 | 1146.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1130.00 | 1149.12 | 1145.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1130.00 | 1149.12 | 1145.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1129.50 | 1145.19 | 1143.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 1130.50 | 1145.19 | 1143.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1130.00 | 1142.15 | 1142.66 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 1144.50 | 1140.19 | 1139.97 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 1135.00 | 1141.42 | 1141.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 1130.00 | 1137.62 | 1139.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 15:15:00 | 1132.30 | 1131.88 | 1135.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:15:00 | 1127.50 | 1131.88 | 1135.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1147.00 | 1134.91 | 1136.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 1146.50 | 1134.91 | 1136.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 1155.90 | 1139.11 | 1138.14 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1131.00 | 1137.20 | 1137.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 1126.00 | 1134.96 | 1136.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 1125.40 | 1123.86 | 1128.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 1125.40 | 1123.86 | 1128.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1125.40 | 1123.86 | 1128.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1125.40 | 1123.86 | 1128.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1125.10 | 1124.11 | 1128.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1111.40 | 1124.11 | 1128.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 1118.10 | 1122.76 | 1126.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:30:00 | 1117.60 | 1121.12 | 1125.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 1136.00 | 1124.10 | 1126.13 | SL hit (close>static) qty=1.00 sl=1129.50 alert=retest2 |

### Cycle 107 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1132.70 | 1126.75 | 1126.71 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 09:15:00 | 1115.00 | 1125.75 | 1126.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 11:15:00 | 1100.50 | 1118.26 | 1122.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 1114.10 | 1110.31 | 1116.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1114.10 | 1110.31 | 1116.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1114.10 | 1110.31 | 1116.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:45:00 | 1099.00 | 1108.59 | 1111.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:15:00 | 1099.10 | 1108.59 | 1111.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:45:00 | 1098.10 | 1106.77 | 1110.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 1092.50 | 1105.62 | 1109.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1100.00 | 1104.49 | 1108.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:45:00 | 1083.90 | 1099.15 | 1103.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:45:00 | 1089.40 | 1098.00 | 1102.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 1087.00 | 1094.08 | 1099.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:00:00 | 1089.30 | 1091.02 | 1096.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1088.80 | 1087.00 | 1091.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 1090.90 | 1087.00 | 1091.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1099.30 | 1088.07 | 1090.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 1099.30 | 1088.07 | 1090.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1099.80 | 1090.41 | 1091.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:30:00 | 1098.00 | 1090.41 | 1091.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 1097.50 | 1092.34 | 1092.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 1097.50 | 1092.34 | 1092.20 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 1088.70 | 1092.87 | 1093.13 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 14:15:00 | 1101.70 | 1093.41 | 1093.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 15:15:00 | 1115.00 | 1097.73 | 1095.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 11:15:00 | 1098.70 | 1099.99 | 1097.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 11:15:00 | 1098.70 | 1099.99 | 1097.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1098.70 | 1099.99 | 1097.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 1098.60 | 1099.99 | 1097.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1099.90 | 1099.97 | 1097.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 1110.00 | 1098.85 | 1097.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 1096.20 | 1100.11 | 1098.21 | SL hit (close<static) qty=1.00 sl=1097.10 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 11:15:00 | 1089.40 | 1096.22 | 1096.66 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 1101.50 | 1096.88 | 1096.81 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 1095.40 | 1096.58 | 1096.68 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1099.10 | 1097.08 | 1096.90 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 1093.50 | 1096.23 | 1096.54 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1115.10 | 1100.11 | 1098.25 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 1084.00 | 1095.98 | 1096.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1082.90 | 1090.08 | 1093.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 15:15:00 | 1081.00 | 1080.76 | 1086.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 1084.10 | 1081.43 | 1086.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1084.10 | 1081.43 | 1086.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 1083.20 | 1081.43 | 1086.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1083.00 | 1081.74 | 1085.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1085.50 | 1081.74 | 1085.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1086.00 | 1082.55 | 1084.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 1086.90 | 1082.55 | 1084.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1079.20 | 1081.88 | 1084.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 1087.60 | 1081.88 | 1084.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1087.00 | 1082.90 | 1084.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 1087.00 | 1082.90 | 1084.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1093.60 | 1085.04 | 1085.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 1097.50 | 1085.04 | 1085.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 1095.50 | 1087.13 | 1086.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 13:15:00 | 1098.00 | 1089.31 | 1087.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 1092.40 | 1095.12 | 1091.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:00:00 | 1092.40 | 1095.12 | 1091.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1090.00 | 1094.10 | 1091.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1090.00 | 1094.10 | 1091.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1095.40 | 1094.36 | 1091.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:30:00 | 1098.20 | 1096.23 | 1092.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 1098.80 | 1100.90 | 1096.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 1099.80 | 1099.22 | 1096.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:30:00 | 1099.30 | 1099.82 | 1097.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1093.00 | 1098.46 | 1097.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 1093.50 | 1098.46 | 1097.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 1087.90 | 1096.34 | 1096.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1087.90 | 1096.34 | 1096.54 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 14:15:00 | 1106.90 | 1098.05 | 1097.21 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 1086.30 | 1095.15 | 1096.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 14:15:00 | 1079.80 | 1090.09 | 1093.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 12:15:00 | 1072.30 | 1068.27 | 1079.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 13:00:00 | 1072.30 | 1068.27 | 1079.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1071.00 | 1069.43 | 1078.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 1074.70 | 1069.43 | 1078.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 1066.10 | 1068.76 | 1076.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 1062.70 | 1072.78 | 1076.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1089.70 | 1071.92 | 1073.31 | SL hit (close>static) qty=1.00 sl=1083.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 1091.80 | 1075.90 | 1074.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 14:15:00 | 1106.80 | 1095.18 | 1088.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 10:15:00 | 1091.30 | 1096.45 | 1091.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 1091.30 | 1096.45 | 1091.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1091.30 | 1096.45 | 1091.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 1091.30 | 1096.45 | 1091.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1086.40 | 1094.44 | 1090.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 1086.20 | 1094.44 | 1090.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1088.10 | 1093.17 | 1090.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 1093.20 | 1093.17 | 1090.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:45:00 | 1093.40 | 1092.11 | 1091.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 1090.30 | 1092.11 | 1091.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 1089.30 | 1094.72 | 1095.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 1089.30 | 1094.72 | 1095.27 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1113.40 | 1098.48 | 1096.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1125.50 | 1103.88 | 1099.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 1124.80 | 1127.01 | 1117.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 14:00:00 | 1124.80 | 1127.01 | 1117.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1127.70 | 1127.15 | 1118.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:30:00 | 1119.00 | 1127.15 | 1118.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1090.00 | 1120.69 | 1117.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 1090.00 | 1120.69 | 1117.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 1072.30 | 1111.01 | 1113.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 13:15:00 | 1053.80 | 1092.86 | 1104.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 1091.60 | 1076.19 | 1088.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 1091.60 | 1076.19 | 1088.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 1091.60 | 1076.19 | 1088.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 1091.60 | 1076.19 | 1088.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 1095.70 | 1080.09 | 1089.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 1097.20 | 1080.09 | 1089.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1080.60 | 1083.20 | 1089.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1099.90 | 1085.86 | 1090.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1084.20 | 1083.23 | 1086.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:30:00 | 1075.70 | 1081.47 | 1085.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 1075.50 | 1079.94 | 1084.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:00:00 | 1073.80 | 1079.94 | 1084.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:45:00 | 1075.70 | 1079.13 | 1083.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1082.90 | 1079.88 | 1083.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1082.90 | 1079.88 | 1083.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1080.00 | 1079.91 | 1083.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 1094.60 | 1082.84 | 1084.15 | SL hit (close>static) qty=1.00 sl=1092.70 alert=retest2 |

### Cycle 127 — BUY (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 15:15:00 | 1095.00 | 1085.28 | 1085.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 1098.50 | 1088.58 | 1086.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 11:15:00 | 1088.50 | 1088.56 | 1086.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 12:00:00 | 1088.50 | 1088.56 | 1086.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1088.10 | 1088.47 | 1086.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 13:30:00 | 1094.50 | 1090.32 | 1087.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 1084.70 | 1091.02 | 1089.70 | SL hit (close<static) qty=1.00 sl=1085.50 alert=retest2 |

### Cycle 128 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1086.50 | 1089.50 | 1089.73 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 13:15:00 | 1097.60 | 1090.85 | 1090.23 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 1083.40 | 1089.68 | 1090.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 1082.10 | 1088.17 | 1089.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 1080.60 | 1076.56 | 1081.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 1080.60 | 1076.56 | 1081.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1080.60 | 1076.56 | 1081.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 1080.60 | 1076.56 | 1081.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1079.00 | 1077.05 | 1080.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 1072.10 | 1077.05 | 1080.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1073.60 | 1076.36 | 1080.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 1070.20 | 1076.36 | 1080.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:00:00 | 1071.20 | 1072.33 | 1077.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1067.60 | 1072.90 | 1076.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:30:00 | 1070.20 | 1069.74 | 1073.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1063.90 | 1068.62 | 1072.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 1063.90 | 1068.62 | 1072.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1067.00 | 1064.38 | 1068.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 1072.40 | 1064.38 | 1068.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1078.60 | 1067.48 | 1069.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1076.00 | 1067.48 | 1069.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1081.30 | 1070.25 | 1070.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1081.30 | 1070.25 | 1070.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-07 15:15:00 | 1094.40 | 1075.08 | 1072.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 1094.40 | 1075.08 | 1072.61 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 1067.20 | 1074.48 | 1074.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 15:15:00 | 1067.00 | 1071.98 | 1073.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 14:15:00 | 1071.80 | 1068.51 | 1070.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 14:15:00 | 1071.80 | 1068.51 | 1070.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1071.80 | 1068.51 | 1070.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 1071.80 | 1068.51 | 1070.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 1068.60 | 1068.52 | 1070.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 1069.00 | 1069.20 | 1070.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1072.50 | 1069.86 | 1070.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 1075.40 | 1069.86 | 1070.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1072.00 | 1070.29 | 1070.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 1068.40 | 1070.83 | 1070.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 13:15:00 | 1067.30 | 1070.83 | 1070.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 1074.80 | 1071.08 | 1071.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 14:15:00 | 1074.80 | 1071.08 | 1071.03 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1066.80 | 1070.98 | 1071.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1060.70 | 1068.01 | 1069.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 1075.00 | 1068.72 | 1069.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 1075.00 | 1068.72 | 1069.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1075.00 | 1068.72 | 1069.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1075.00 | 1068.72 | 1069.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1075.00 | 1069.98 | 1070.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1087.90 | 1069.98 | 1070.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1082.10 | 1072.40 | 1071.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 10:15:00 | 1095.10 | 1076.94 | 1073.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 14:15:00 | 1083.00 | 1084.78 | 1078.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 15:00:00 | 1083.00 | 1084.78 | 1078.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1081.50 | 1084.00 | 1079.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 1110.90 | 1095.12 | 1087.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 1110.10 | 1100.64 | 1091.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 1109.60 | 1102.43 | 1093.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 14:30:00 | 1110.50 | 1104.49 | 1096.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-21 09:15:00 | 1221.99 | 1158.18 | 1129.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 1209.40 | 1217.83 | 1218.91 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1228.30 | 1217.11 | 1216.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 1246.20 | 1224.79 | 1220.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 10:15:00 | 1219.00 | 1225.66 | 1222.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1219.00 | 1225.66 | 1222.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1219.00 | 1225.66 | 1222.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 1220.00 | 1225.66 | 1222.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1224.00 | 1225.33 | 1222.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:30:00 | 1216.00 | 1225.33 | 1222.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1220.60 | 1224.39 | 1222.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 1218.80 | 1224.39 | 1222.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1216.70 | 1222.85 | 1221.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 13:30:00 | 1218.70 | 1222.85 | 1221.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 1208.90 | 1220.06 | 1220.65 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 1240.00 | 1223.45 | 1221.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 14:15:00 | 1259.10 | 1230.58 | 1224.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1252.20 | 1257.15 | 1250.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1252.20 | 1257.15 | 1250.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1252.20 | 1257.15 | 1250.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 1246.20 | 1257.15 | 1250.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1246.90 | 1254.60 | 1250.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 1246.90 | 1254.60 | 1250.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1247.30 | 1253.14 | 1249.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:15:00 | 1241.80 | 1253.14 | 1249.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 1246.10 | 1247.97 | 1248.06 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 10:15:00 | 1260.90 | 1249.76 | 1248.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 11:15:00 | 1270.30 | 1253.87 | 1250.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1296.30 | 1302.20 | 1284.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:45:00 | 1292.60 | 1302.20 | 1284.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1284.90 | 1298.39 | 1286.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 1284.90 | 1298.39 | 1286.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1284.50 | 1295.61 | 1286.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 1273.50 | 1295.61 | 1286.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1286.70 | 1293.83 | 1286.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:15:00 | 1280.50 | 1293.83 | 1286.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1277.50 | 1290.56 | 1285.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 1277.50 | 1290.56 | 1285.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1283.70 | 1289.19 | 1285.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 1283.80 | 1289.19 | 1285.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1280.10 | 1287.37 | 1284.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 1280.10 | 1287.37 | 1284.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1280.70 | 1286.04 | 1284.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:30:00 | 1282.80 | 1286.57 | 1284.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 1319.60 | 1326.52 | 1326.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 1319.60 | 1326.52 | 1326.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 1310.00 | 1321.55 | 1324.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 1320.00 | 1318.10 | 1321.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 14:15:00 | 1320.00 | 1318.10 | 1321.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1320.00 | 1318.10 | 1321.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 1320.00 | 1318.10 | 1321.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1321.60 | 1318.80 | 1321.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 1318.50 | 1318.80 | 1321.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1326.00 | 1320.24 | 1321.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:15:00 | 1323.90 | 1320.24 | 1321.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1319.70 | 1320.13 | 1321.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:30:00 | 1316.20 | 1319.04 | 1320.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1317.50 | 1315.58 | 1318.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 1313.60 | 1314.93 | 1316.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 1320.30 | 1318.15 | 1317.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 1320.30 | 1318.15 | 1317.86 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 1316.00 | 1317.74 | 1317.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 11:15:00 | 1311.00 | 1316.39 | 1317.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 1323.70 | 1317.13 | 1317.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 14:15:00 | 1323.70 | 1317.13 | 1317.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1323.70 | 1317.13 | 1317.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1323.70 | 1317.13 | 1317.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 15:15:00 | 1329.50 | 1319.61 | 1318.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 1338.80 | 1327.19 | 1323.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 1343.70 | 1346.45 | 1336.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 1343.70 | 1346.45 | 1336.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1343.70 | 1346.45 | 1336.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 1335.80 | 1346.45 | 1336.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1343.70 | 1344.44 | 1337.51 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 1324.50 | 1334.64 | 1335.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1317.10 | 1328.00 | 1331.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 1315.30 | 1314.94 | 1321.40 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 09:15:00 | 1301.70 | 1314.94 | 1321.40 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 12:45:00 | 1304.00 | 1309.76 | 1316.74 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 14:30:00 | 1304.10 | 1309.38 | 1315.29 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1318.00 | 1311.10 | 1315.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1318.00 | 1311.10 | 1315.54 | SL hit (close>ema400) qty=1.00 sl=1315.54 alert=retest1 |

### Cycle 147 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 1339.00 | 1322.08 | 1319.92 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 1316.40 | 1322.80 | 1323.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 11:15:00 | 1315.60 | 1321.36 | 1322.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1250.60 | 1239.36 | 1261.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 14:15:00 | 1230.80 | 1225.87 | 1243.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1230.80 | 1225.87 | 1243.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 1230.80 | 1225.87 | 1243.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1242.20 | 1227.47 | 1235.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1215.30 | 1227.47 | 1235.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 1218.70 | 1226.38 | 1233.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 11:45:00 | 1217.70 | 1223.22 | 1231.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 1220.10 | 1218.61 | 1225.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1221.00 | 1216.30 | 1220.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:45:00 | 1213.30 | 1215.94 | 1220.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:30:00 | 1213.80 | 1215.33 | 1219.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:15:00 | 1212.70 | 1215.15 | 1219.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 13:15:00 | 1211.40 | 1215.02 | 1218.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 1217.80 | 1215.88 | 1218.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:30:00 | 1219.30 | 1215.88 | 1218.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1220.00 | 1216.42 | 1218.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 1218.70 | 1216.42 | 1218.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1222.30 | 1217.59 | 1218.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 1229.30 | 1217.59 | 1218.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 1231.30 | 1220.33 | 1219.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 1231.30 | 1220.33 | 1219.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 1236.70 | 1225.89 | 1222.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 1232.20 | 1236.77 | 1230.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 1232.20 | 1236.77 | 1230.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1232.20 | 1236.77 | 1230.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1234.60 | 1236.77 | 1230.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1225.70 | 1234.56 | 1230.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 1225.70 | 1234.56 | 1230.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1225.40 | 1232.73 | 1229.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 1224.10 | 1232.73 | 1229.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 1213.00 | 1225.51 | 1226.92 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1255.80 | 1231.82 | 1228.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 1272.70 | 1243.94 | 1235.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 11:15:00 | 1242.90 | 1248.67 | 1241.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 11:15:00 | 1242.90 | 1248.67 | 1241.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 1242.90 | 1248.67 | 1241.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:00:00 | 1242.90 | 1248.67 | 1241.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1247.90 | 1248.52 | 1242.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:45:00 | 1245.10 | 1248.52 | 1242.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1315.30 | 1321.30 | 1313.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 1317.00 | 1321.30 | 1313.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1313.60 | 1319.76 | 1313.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 1316.20 | 1319.76 | 1313.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 1309.90 | 1317.79 | 1312.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1321.90 | 1317.79 | 1312.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 1335.70 | 1346.11 | 1346.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1335.70 | 1346.11 | 1346.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 1273.20 | 1331.53 | 1340.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 1266.10 | 1264.01 | 1285.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 1266.10 | 1264.01 | 1285.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1278.50 | 1267.71 | 1283.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1281.30 | 1267.71 | 1283.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1278.20 | 1269.81 | 1282.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 1282.90 | 1269.81 | 1282.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 1281.90 | 1272.67 | 1281.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:30:00 | 1283.30 | 1272.67 | 1281.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1288.00 | 1275.74 | 1282.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 1288.00 | 1275.74 | 1282.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1287.30 | 1278.05 | 1282.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 1291.00 | 1278.05 | 1282.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1286.60 | 1281.26 | 1283.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1286.60 | 1281.26 | 1283.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 11:15:00 | 1302.50 | 1285.51 | 1285.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 12:15:00 | 1314.50 | 1291.31 | 1287.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 10:15:00 | 1308.00 | 1308.87 | 1299.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:30:00 | 1308.00 | 1308.87 | 1299.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1298.40 | 1306.04 | 1299.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:00:00 | 1298.40 | 1306.04 | 1299.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1299.00 | 1304.64 | 1299.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1299.00 | 1304.64 | 1299.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1287.20 | 1301.15 | 1298.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 1287.20 | 1301.15 | 1298.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 1278.20 | 1296.56 | 1296.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 1277.20 | 1292.69 | 1294.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1275.00 | 1270.52 | 1277.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 1275.00 | 1270.52 | 1277.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1275.00 | 1270.52 | 1277.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:00:00 | 1268.20 | 1271.74 | 1276.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 1267.20 | 1270.88 | 1275.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 1269.20 | 1270.88 | 1275.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1204.79 | 1239.57 | 1250.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1203.84 | 1239.57 | 1250.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1205.74 | 1239.57 | 1250.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 1226.80 | 1226.33 | 1237.93 | SL hit (close>ema200) qty=0.50 sl=1226.33 alert=retest2 |

### Cycle 155 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 1243.20 | 1225.53 | 1224.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 14:15:00 | 1251.80 | 1233.86 | 1230.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1215.00 | 1231.71 | 1230.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1215.00 | 1231.71 | 1230.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1215.00 | 1231.71 | 1230.52 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 1216.90 | 1228.75 | 1229.28 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 1262.00 | 1231.84 | 1230.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 1289.00 | 1258.66 | 1246.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1296.10 | 1296.12 | 1275.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 1296.10 | 1296.12 | 1275.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1275.80 | 1291.41 | 1277.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1273.20 | 1291.41 | 1277.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1280.60 | 1289.25 | 1277.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:30:00 | 1282.80 | 1286.96 | 1278.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 1285.20 | 1286.96 | 1278.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 1266.50 | 1281.87 | 1277.57 | SL hit (close<static) qty=1.00 sl=1273.70 alert=retest2 |

### Cycle 158 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1254.90 | 1273.17 | 1274.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 1247.70 | 1264.56 | 1269.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1230.80 | 1228.45 | 1242.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 1236.20 | 1228.45 | 1242.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1239.10 | 1232.27 | 1242.03 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1268.00 | 1246.41 | 1244.06 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 15:15:00 | 1235.90 | 1242.82 | 1243.16 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 12:15:00 | 1245.40 | 1243.22 | 1243.11 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1241.70 | 1242.93 | 1243.00 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1266.60 | 1247.37 | 1244.99 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1221.50 | 1246.41 | 1246.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 1210.00 | 1232.31 | 1239.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 1245.00 | 1232.25 | 1238.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 14:15:00 | 1245.00 | 1232.25 | 1238.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 1245.00 | 1232.25 | 1238.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 1245.00 | 1232.25 | 1238.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 1242.60 | 1234.32 | 1238.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 1247.50 | 1234.32 | 1238.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 1243.30 | 1237.17 | 1239.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:30:00 | 1242.30 | 1237.17 | 1239.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1234.50 | 1236.21 | 1238.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1235.60 | 1236.21 | 1238.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1237.00 | 1236.37 | 1238.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1237.00 | 1236.37 | 1238.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1255.00 | 1240.09 | 1239.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1280.40 | 1250.54 | 1244.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 1264.00 | 1267.06 | 1257.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 1260.80 | 1267.06 | 1257.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1245.50 | 1262.75 | 1256.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 1245.50 | 1262.75 | 1256.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1238.00 | 1257.80 | 1254.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1238.00 | 1257.80 | 1254.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1240.30 | 1251.84 | 1252.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 1231.80 | 1247.83 | 1250.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 1240.00 | 1221.73 | 1231.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 1240.00 | 1221.73 | 1231.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 1240.00 | 1221.73 | 1231.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 1253.20 | 1221.73 | 1231.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1252.10 | 1227.81 | 1233.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 1244.90 | 1227.81 | 1233.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1242.50 | 1230.74 | 1234.16 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1269.20 | 1240.49 | 1238.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 13:15:00 | 1274.50 | 1247.29 | 1241.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1244.90 | 1255.51 | 1247.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1244.90 | 1255.51 | 1247.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1244.90 | 1255.51 | 1247.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 1260.20 | 1250.80 | 1247.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 1257.70 | 1254.36 | 1249.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 1264.50 | 1258.12 | 1252.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 1257.70 | 1258.45 | 1254.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 1246.50 | 1256.06 | 1253.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 1246.50 | 1256.06 | 1253.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 1243.10 | 1253.46 | 1252.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:15:00 | 1237.00 | 1253.46 | 1252.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1238.40 | 1250.45 | 1251.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 1238.40 | 1250.45 | 1251.26 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 1270.00 | 1251.31 | 1249.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 12:15:00 | 1275.40 | 1262.31 | 1257.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1273.40 | 1280.64 | 1273.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1273.40 | 1280.64 | 1273.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1273.40 | 1280.64 | 1273.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1280.10 | 1280.64 | 1273.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 1277.80 | 1280.49 | 1276.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1292.70 | 1278.99 | 1276.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 1288.30 | 1284.06 | 1281.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1277.00 | 1282.65 | 1280.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 1273.80 | 1282.65 | 1280.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 1277.60 | 1281.64 | 1280.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 1269.00 | 1279.11 | 1279.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 1269.00 | 1279.11 | 1279.48 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 1290.50 | 1278.41 | 1277.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 1316.40 | 1286.42 | 1281.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 1319.70 | 1324.56 | 1311.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 14:45:00 | 1323.90 | 1324.56 | 1311.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1318.00 | 1323.25 | 1311.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1340.80 | 1326.12 | 1314.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 11:15:00 | 1331.50 | 1340.02 | 1340.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 11:15:00 | 1331.50 | 1340.02 | 1340.34 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1343.30 | 1340.68 | 1340.61 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 13:15:00 | 1337.10 | 1339.96 | 1340.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 15:15:00 | 1330.50 | 1337.26 | 1338.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 10:15:00 | 1313.00 | 1305.75 | 1314.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 10:15:00 | 1313.00 | 1305.75 | 1314.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1313.00 | 1305.75 | 1314.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 1313.00 | 1305.75 | 1314.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1318.10 | 1308.22 | 1314.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 1318.10 | 1308.22 | 1314.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1313.30 | 1309.24 | 1314.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 1308.30 | 1310.03 | 1314.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 14:15:00 | 1348.30 | 1317.68 | 1317.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 1348.30 | 1317.68 | 1317.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 1359.30 | 1349.31 | 1339.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 1357.00 | 1357.48 | 1349.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:45:00 | 1355.70 | 1357.48 | 1349.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1340.40 | 1361.67 | 1356.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 1340.40 | 1361.67 | 1356.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1340.30 | 1357.39 | 1354.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 1334.90 | 1357.39 | 1354.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1361.90 | 1366.10 | 1361.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 1360.80 | 1366.10 | 1361.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1363.40 | 1365.56 | 1361.77 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-22 13:00:00 | 783.15 | 2024-06-03 14:15:00 | 773.45 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-07-02 12:30:00 | 764.55 | 2024-07-12 14:15:00 | 752.15 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2024-07-04 09:15:00 | 765.00 | 2024-07-12 14:15:00 | 752.15 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2024-07-04 12:30:00 | 765.60 | 2024-07-12 14:15:00 | 752.15 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2024-07-04 13:30:00 | 765.60 | 2024-07-12 14:15:00 | 752.15 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2024-07-08 09:45:00 | 761.00 | 2024-07-12 14:15:00 | 752.15 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2024-07-25 10:45:00 | 795.45 | 2024-07-25 13:15:00 | 781.90 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-08-12 09:15:00 | 775.70 | 2024-08-14 09:15:00 | 736.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-12 09:15:00 | 775.70 | 2024-08-14 09:15:00 | 774.00 | STOP_HIT | 0.50 | 0.22% |
| SELL | retest2 | 2024-08-28 13:00:00 | 779.00 | 2024-09-04 10:15:00 | 779.25 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2024-08-29 10:15:00 | 777.85 | 2024-09-04 10:15:00 | 779.25 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-08-30 09:30:00 | 776.75 | 2024-09-04 10:15:00 | 779.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-09-02 09:15:00 | 778.95 | 2024-09-04 10:15:00 | 779.25 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-09-02 10:15:00 | 774.15 | 2024-09-04 10:15:00 | 779.25 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-09-04 09:45:00 | 776.00 | 2024-09-04 10:15:00 | 779.25 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-09-06 14:15:00 | 761.45 | 2024-09-10 14:15:00 | 771.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-09-09 14:00:00 | 761.75 | 2024-09-10 14:15:00 | 771.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-09-18 13:00:00 | 757.50 | 2024-10-01 10:15:00 | 741.50 | STOP_HIT | 1.00 | 2.11% |
| SELL | retest2 | 2024-09-18 13:30:00 | 758.10 | 2024-10-01 10:15:00 | 741.50 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2024-09-18 15:00:00 | 758.60 | 2024-10-01 10:15:00 | 741.50 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2024-09-19 09:30:00 | 757.40 | 2024-10-01 10:15:00 | 741.50 | STOP_HIT | 1.00 | 2.10% |
| SELL | retest2 | 2024-09-20 15:15:00 | 743.00 | 2024-10-01 10:15:00 | 741.50 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-09-23 09:45:00 | 745.45 | 2024-10-01 10:15:00 | 741.50 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2024-10-10 10:45:00 | 733.00 | 2024-10-22 14:15:00 | 696.87 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-10-10 11:15:00 | 733.55 | 2024-10-22 14:15:00 | 697.35 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-10-10 12:30:00 | 733.30 | 2024-10-22 15:15:00 | 696.35 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2024-10-11 13:30:00 | 734.05 | 2024-10-22 15:15:00 | 696.63 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2024-10-14 12:30:00 | 731.80 | 2024-10-23 09:15:00 | 695.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 13:45:00 | 731.80 | 2024-10-23 09:15:00 | 695.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 14:30:00 | 732.05 | 2024-10-23 09:15:00 | 695.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 10:00:00 | 731.30 | 2024-10-23 09:15:00 | 694.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 11:00:00 | 725.00 | 2024-10-25 09:15:00 | 688.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 10:15:00 | 725.00 | 2024-10-25 09:15:00 | 688.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 13:45:00 | 725.50 | 2024-10-25 09:15:00 | 689.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 14:45:00 | 725.00 | 2024-10-25 09:15:00 | 688.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 719.70 | 2024-10-25 10:15:00 | 684.00 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2024-10-22 10:00:00 | 720.00 | 2024-10-25 11:15:00 | 683.72 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2024-10-10 10:45:00 | 733.00 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 5.46% |
| SELL | retest2 | 2024-10-10 11:15:00 | 733.55 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2024-10-10 12:30:00 | 733.30 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 5.50% |
| SELL | retest2 | 2024-10-11 13:30:00 | 734.05 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 5.59% |
| SELL | retest2 | 2024-10-14 12:30:00 | 731.80 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 5.30% |
| SELL | retest2 | 2024-10-14 13:45:00 | 731.80 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 5.30% |
| SELL | retest2 | 2024-10-14 14:30:00 | 732.05 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 5.33% |
| SELL | retest2 | 2024-10-15 10:00:00 | 731.30 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 5.24% |
| SELL | retest2 | 2024-10-16 11:00:00 | 725.00 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2024-10-17 10:15:00 | 725.00 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2024-10-18 13:45:00 | 725.50 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest2 | 2024-10-18 14:45:00 | 725.00 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2024-10-21 11:30:00 | 719.70 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2024-10-22 10:00:00 | 720.00 | 2024-10-25 14:15:00 | 693.00 | STOP_HIT | 0.50 | 3.75% |
| BUY | retest1 | 2024-10-31 10:30:00 | 730.60 | 2024-11-04 09:15:00 | 722.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2024-10-31 11:30:00 | 731.00 | 2024-11-04 09:15:00 | 722.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-11-04 11:15:00 | 729.55 | 2024-11-11 09:15:00 | 731.30 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-11-04 11:45:00 | 729.70 | 2024-11-11 09:15:00 | 731.30 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2024-11-05 09:45:00 | 731.20 | 2024-11-11 09:15:00 | 731.30 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-11-12 10:45:00 | 733.15 | 2024-11-14 15:15:00 | 741.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-11-12 11:45:00 | 732.50 | 2024-11-14 15:15:00 | 741.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-11-26 15:00:00 | 732.95 | 2024-11-27 13:15:00 | 726.65 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-12-12 11:15:00 | 755.50 | 2024-12-12 13:15:00 | 751.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-12-20 15:00:00 | 718.10 | 2024-12-30 09:15:00 | 727.20 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-12-23 12:45:00 | 721.95 | 2024-12-30 09:15:00 | 727.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-12-24 09:30:00 | 721.35 | 2024-12-30 11:15:00 | 740.40 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-12-24 10:45:00 | 722.25 | 2024-12-30 11:15:00 | 740.40 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-12-24 14:30:00 | 722.05 | 2024-12-30 11:15:00 | 740.40 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2024-12-26 09:30:00 | 719.60 | 2024-12-30 11:15:00 | 740.40 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-12-30 10:15:00 | 721.10 | 2024-12-30 11:15:00 | 740.40 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-12-30 11:00:00 | 722.10 | 2024-12-30 11:15:00 | 740.40 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-01-03 15:15:00 | 729.00 | 2025-01-06 09:15:00 | 718.05 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-01-14 09:15:00 | 686.50 | 2025-01-16 15:15:00 | 693.75 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-01-16 14:15:00 | 684.85 | 2025-01-16 15:15:00 | 693.75 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-01-20 09:30:00 | 702.80 | 2025-01-22 12:15:00 | 688.85 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-01-20 11:15:00 | 694.95 | 2025-01-22 12:15:00 | 688.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-01-20 12:00:00 | 695.00 | 2025-01-22 12:15:00 | 688.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-01-20 13:45:00 | 695.00 | 2025-01-22 12:15:00 | 688.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-03 11:15:00 | 699.25 | 2025-02-11 14:15:00 | 716.75 | STOP_HIT | 1.00 | 2.50% |
| SELL | retest2 | 2025-02-14 11:15:00 | 687.80 | 2025-02-17 09:15:00 | 653.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 11:15:00 | 687.80 | 2025-02-18 10:15:00 | 619.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-24 09:30:00 | 630.80 | 2025-02-24 12:15:00 | 641.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-02-27 09:15:00 | 633.40 | 2025-02-28 09:15:00 | 601.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 633.40 | 2025-02-28 11:15:00 | 634.35 | STOP_HIT | 0.50 | -0.15% |
| SELL | retest2 | 2025-02-28 12:00:00 | 634.35 | 2025-03-03 10:15:00 | 602.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 13:45:00 | 634.85 | 2025-03-03 10:15:00 | 603.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 14:30:00 | 634.90 | 2025-03-03 10:15:00 | 603.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 12:00:00 | 634.35 | 2025-03-03 12:15:00 | 644.00 | STOP_HIT | 0.50 | -1.52% |
| SELL | retest2 | 2025-02-28 13:45:00 | 634.85 | 2025-03-03 12:15:00 | 644.00 | STOP_HIT | 0.50 | -1.44% |
| SELL | retest2 | 2025-02-28 14:30:00 | 634.90 | 2025-03-03 12:15:00 | 644.00 | STOP_HIT | 0.50 | -1.43% |
| SELL | retest2 | 2025-03-03 10:45:00 | 608.35 | 2025-03-03 12:15:00 | 644.00 | STOP_HIT | 1.00 | -5.86% |
| BUY | retest2 | 2025-03-13 15:00:00 | 786.90 | 2025-03-17 09:15:00 | 770.25 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-03-24 11:30:00 | 765.00 | 2025-03-26 15:15:00 | 778.25 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-03-24 12:15:00 | 766.40 | 2025-03-26 15:15:00 | 778.25 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-03-25 09:15:00 | 765.20 | 2025-03-26 15:15:00 | 778.25 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-03-25 10:00:00 | 767.00 | 2025-03-26 15:15:00 | 778.25 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-03-25 11:30:00 | 761.00 | 2025-03-26 15:15:00 | 778.25 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-03-25 12:30:00 | 761.70 | 2025-03-26 15:15:00 | 778.25 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-03-25 15:00:00 | 761.60 | 2025-03-26 15:15:00 | 778.25 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-03-27 09:15:00 | 749.40 | 2025-03-28 14:15:00 | 767.75 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-03-27 12:30:00 | 744.65 | 2025-03-28 14:15:00 | 767.75 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-03-27 14:30:00 | 746.00 | 2025-03-28 14:15:00 | 767.75 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-03-27 15:15:00 | 745.00 | 2025-03-28 14:15:00 | 767.75 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-03-28 09:30:00 | 746.05 | 2025-03-28 15:15:00 | 764.70 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-04-02 13:30:00 | 748.00 | 2025-04-03 14:15:00 | 761.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-04-03 09:15:00 | 749.40 | 2025-04-03 14:15:00 | 761.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-04-08 10:45:00 | 703.95 | 2025-04-15 13:15:00 | 718.80 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-04-09 09:15:00 | 694.40 | 2025-04-15 13:15:00 | 718.80 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-04-09 15:15:00 | 711.00 | 2025-04-15 14:15:00 | 743.65 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-04-11 09:45:00 | 710.05 | 2025-04-15 14:15:00 | 743.65 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2025-04-11 14:15:00 | 698.05 | 2025-04-15 14:15:00 | 743.65 | STOP_HIT | 1.00 | -6.53% |
| SELL | retest2 | 2025-04-15 09:30:00 | 698.50 | 2025-04-15 14:15:00 | 743.65 | STOP_HIT | 1.00 | -6.46% |
| BUY | retest2 | 2025-05-05 09:15:00 | 878.75 | 2025-05-09 12:15:00 | 881.45 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-05-05 14:45:00 | 875.05 | 2025-05-09 12:15:00 | 881.45 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-05-05 15:15:00 | 874.60 | 2025-05-09 12:15:00 | 881.45 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-05-06 10:15:00 | 875.20 | 2025-05-09 12:15:00 | 881.45 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-05-07 14:00:00 | 893.20 | 2025-05-09 12:15:00 | 881.45 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-05-23 10:30:00 | 969.70 | 2025-05-23 13:15:00 | 946.00 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-05-23 11:00:00 | 970.00 | 2025-05-23 13:15:00 | 946.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-06-03 10:15:00 | 1005.80 | 2025-06-11 13:15:00 | 1106.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 11:30:00 | 1007.00 | 2025-06-11 13:15:00 | 1107.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 09:15:00 | 1139.30 | 2025-06-26 09:15:00 | 1125.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-06-23 10:15:00 | 1137.00 | 2025-06-26 09:15:00 | 1125.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-25 13:45:00 | 1138.50 | 2025-06-26 09:15:00 | 1125.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-06-27 14:15:00 | 1110.10 | 2025-06-27 15:15:00 | 1140.00 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-07-01 10:45:00 | 1148.10 | 2025-07-02 12:15:00 | 1137.90 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-04 13:45:00 | 1148.30 | 2025-07-04 14:15:00 | 1137.30 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-07-09 09:15:00 | 1147.50 | 2025-07-11 10:15:00 | 1140.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-07-11 09:15:00 | 1144.40 | 2025-07-11 10:15:00 | 1140.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-07-14 11:30:00 | 1141.40 | 2025-07-14 15:15:00 | 1146.80 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-14 12:00:00 | 1140.90 | 2025-07-14 15:15:00 | 1146.80 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-07-16 13:30:00 | 1162.50 | 2025-07-17 12:15:00 | 1146.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-07-25 09:15:00 | 1141.80 | 2025-07-30 10:15:00 | 1150.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-28 09:15:00 | 1141.50 | 2025-07-30 10:15:00 | 1150.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1111.40 | 2025-08-28 14:15:00 | 1136.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-08-28 12:15:00 | 1118.10 | 2025-08-28 14:15:00 | 1136.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-08-28 13:30:00 | 1117.60 | 2025-08-28 14:15:00 | 1136.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-29 09:30:00 | 1117.90 | 2025-08-29 10:15:00 | 1132.30 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-04 13:45:00 | 1099.00 | 2025-09-11 09:15:00 | 1097.50 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-04 14:15:00 | 1099.10 | 2025-09-11 09:15:00 | 1097.50 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-09-04 14:45:00 | 1098.10 | 2025-09-11 09:15:00 | 1097.50 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-09-05 09:15:00 | 1092.50 | 2025-09-11 09:15:00 | 1097.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-08 10:45:00 | 1083.90 | 2025-09-11 09:15:00 | 1097.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-08 11:45:00 | 1089.40 | 2025-09-11 09:15:00 | 1097.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-08 15:00:00 | 1087.00 | 2025-09-11 09:15:00 | 1097.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-09-09 11:00:00 | 1089.30 | 2025-09-11 09:15:00 | 1097.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-15 15:15:00 | 1110.00 | 2025-09-16 09:15:00 | 1096.20 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-09-24 13:30:00 | 1098.20 | 2025-09-26 11:15:00 | 1087.90 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-25 11:15:00 | 1098.80 | 2025-09-26 11:15:00 | 1087.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-25 13:15:00 | 1099.80 | 2025-09-26 11:15:00 | 1087.90 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-26 09:30:00 | 1099.30 | 2025-09-26 11:15:00 | 1087.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-01 15:00:00 | 1062.70 | 2025-10-03 13:15:00 | 1089.70 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-10-09 13:15:00 | 1093.20 | 2025-10-14 13:15:00 | 1089.30 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-10 11:45:00 | 1093.40 | 2025-10-14 13:15:00 | 1089.30 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-10-10 12:15:00 | 1090.30 | 2025-10-14 13:15:00 | 1089.30 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-10-24 09:30:00 | 1075.70 | 2025-10-24 14:15:00 | 1094.60 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-10-24 10:30:00 | 1075.50 | 2025-10-24 14:15:00 | 1094.60 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-24 11:00:00 | 1073.80 | 2025-10-24 14:15:00 | 1094.60 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-10-24 11:45:00 | 1075.70 | 2025-10-24 14:15:00 | 1094.60 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-10-27 13:30:00 | 1094.50 | 2025-10-28 12:15:00 | 1084.70 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-10-28 15:00:00 | 1094.00 | 2025-10-30 09:15:00 | 1086.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-10-29 12:30:00 | 1094.60 | 2025-10-30 09:15:00 | 1086.50 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-29 13:15:00 | 1092.40 | 2025-10-30 09:15:00 | 1086.50 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-04 10:15:00 | 1070.20 | 2025-11-07 15:15:00 | 1094.40 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-11-04 13:00:00 | 1071.20 | 2025-11-07 15:15:00 | 1094.40 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1067.60 | 2025-11-07 15:15:00 | 1094.40 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-11-06 12:30:00 | 1070.20 | 2025-11-07 15:15:00 | 1094.40 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-11-13 12:30:00 | 1068.40 | 2025-11-13 14:15:00 | 1074.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-13 13:15:00 | 1067.30 | 2025-11-13 14:15:00 | 1074.80 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-11-18 15:15:00 | 1110.90 | 2025-11-21 09:15:00 | 1221.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-19 10:00:00 | 1110.10 | 2025-11-21 09:15:00 | 1221.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-19 11:00:00 | 1109.60 | 2025-11-21 09:15:00 | 1220.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-19 14:30:00 | 1110.50 | 2025-11-21 09:15:00 | 1221.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-25 09:15:00 | 1222.10 | 2025-11-28 09:15:00 | 1209.40 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-12-12 09:30:00 | 1282.80 | 2025-12-24 15:15:00 | 1319.60 | STOP_HIT | 1.00 | 2.87% |
| SELL | retest2 | 2025-12-29 12:30:00 | 1316.20 | 2025-12-31 11:15:00 | 1320.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-12-30 10:15:00 | 1317.50 | 2025-12-31 11:15:00 | 1320.30 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-12-31 09:15:00 | 1313.60 | 2025-12-31 11:15:00 | 1320.30 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-01-12 09:15:00 | 1301.70 | 2026-01-12 15:15:00 | 1318.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest1 | 2026-01-12 12:45:00 | 1304.00 | 2026-01-12 15:15:00 | 1318.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest1 | 2026-01-12 14:30:00 | 1304.10 | 2026-01-12 15:15:00 | 1318.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-27 09:15:00 | 1215.30 | 2026-01-30 11:15:00 | 1231.30 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-27 10:15:00 | 1218.70 | 2026-01-30 11:15:00 | 1231.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-27 11:45:00 | 1217.70 | 2026-01-30 11:15:00 | 1231.30 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-28 10:00:00 | 1220.10 | 2026-01-30 11:15:00 | 1231.30 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-29 09:45:00 | 1213.30 | 2026-01-30 11:15:00 | 1231.30 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-29 10:30:00 | 1213.80 | 2026-01-30 11:15:00 | 1231.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-01-29 12:15:00 | 1212.70 | 2026-01-30 11:15:00 | 1231.30 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-29 13:15:00 | 1211.40 | 2026-01-30 11:15:00 | 1231.30 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1321.90 | 2026-02-13 15:15:00 | 1335.70 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2026-02-25 13:00:00 | 1268.20 | 2026-03-02 09:15:00 | 1204.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:45:00 | 1267.20 | 2026-03-02 09:15:00 | 1203.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 1269.20 | 2026-03-02 09:15:00 | 1205.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 13:00:00 | 1268.20 | 2026-03-02 15:15:00 | 1226.80 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2026-02-25 14:45:00 | 1267.20 | 2026-03-02 15:15:00 | 1226.80 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-02-25 15:15:00 | 1269.20 | 2026-03-02 15:15:00 | 1226.80 | STOP_HIT | 0.50 | 3.34% |
| BUY | retest2 | 2026-03-12 12:30:00 | 1282.80 | 2026-03-12 14:15:00 | 1266.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-03-12 13:00:00 | 1285.20 | 2026-03-12 14:15:00 | 1266.50 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-04-02 14:15:00 | 1260.20 | 2026-04-07 09:15:00 | 1238.40 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-04-02 14:45:00 | 1257.70 | 2026-04-07 09:15:00 | 1238.40 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-04-06 11:15:00 | 1264.50 | 2026-04-07 09:15:00 | 1238.40 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-04-06 13:45:00 | 1257.70 | 2026-04-07 09:15:00 | 1238.40 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1280.10 | 2026-04-16 11:15:00 | 1269.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-04-13 14:45:00 | 1277.80 | 2026-04-16 11:15:00 | 1269.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1292.70 | 2026-04-16 11:15:00 | 1269.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-04-16 09:15:00 | 1288.30 | 2026-04-16 11:15:00 | 1269.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-04-22 09:30:00 | 1340.80 | 2026-04-27 11:15:00 | 1331.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-04-30 14:15:00 | 1308.30 | 2026-04-30 14:15:00 | 1348.30 | STOP_HIT | 1.00 | -3.06% |
