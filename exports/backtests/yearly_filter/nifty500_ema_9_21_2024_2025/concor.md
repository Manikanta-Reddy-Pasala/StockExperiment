# Container Corporation of India Ltd. (CONCOR)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 533.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 141 |
| ALERT1 | 93 |
| ALERT2 | 91 |
| ALERT2_SKIP | 39 |
| ALERT3 | 290 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 98 |
| PARTIAL | 11 |
| TARGET_HIT | 3 |
| STOP_HIT | 104 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 116 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 79
- **Target hits / Stop hits / Partials:** 3 / 102 / 11
- **Avg / median % per leg:** 0.43% / -0.78%
- **Sum % (uncompounded):** 49.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 12 | 27.3% | 0 | 43 | 1 | -0.04% | -1.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.64% | 6.6% |
| BUY @ 3rd Alert (retest2) | 40 | 10 | 25.0% | 0 | 40 | 0 | -0.21% | -8.4% |
| SELL (all) | 72 | 25 | 34.7% | 3 | 59 | 10 | 0.71% | 51.3% |
| SELL @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 3 | 1 | 3 | 6.19% | 43.3% |
| SELL @ 3rd Alert (retest2) | 65 | 19 | 29.2% | 0 | 58 | 7 | 0.12% | 8.0% |
| retest1 (combined) | 11 | 8 | 72.7% | 3 | 4 | 4 | 4.53% | 49.9% |
| retest2 (combined) | 105 | 29 | 27.6% | 0 | 98 | 7 | -0.00% | -0.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 812.80 | 806.29 | 805.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 815.16 | 808.06 | 806.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 821.48 | 822.32 | 817.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:15:00 | 828.08 | 822.32 | 817.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 826.64 | 824.59 | 819.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-16 13:15:00 | 819.00 | 823.20 | 820.13 | SL hit (close<ema400) qty=1.00 sl=820.13 alert=retest1 |

### Cycle 2 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 861.04 | 877.00 | 878.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 855.40 | 866.00 | 870.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 861.80 | 852.13 | 858.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 861.80 | 852.13 | 858.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 861.80 | 852.13 | 858.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 861.80 | 852.13 | 858.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 858.72 | 853.45 | 858.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:15:00 | 860.80 | 853.45 | 858.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 860.80 | 854.92 | 858.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 889.08 | 854.92 | 858.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 909.00 | 865.73 | 862.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 10:15:00 | 912.76 | 875.14 | 867.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 901.56 | 908.29 | 890.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 901.56 | 908.29 | 890.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 901.56 | 908.29 | 890.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 880.72 | 908.29 | 890.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 845.48 | 895.73 | 886.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 845.48 | 895.73 | 886.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 751.52 | 866.89 | 873.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 746.24 | 795.71 | 831.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 789.04 | 788.47 | 816.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 789.04 | 788.47 | 816.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 839.88 | 799.61 | 814.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 839.88 | 799.61 | 814.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 847.12 | 809.11 | 817.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 847.12 | 809.11 | 817.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 837.44 | 820.54 | 821.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 841.60 | 820.54 | 821.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 14:15:00 | 842.44 | 824.92 | 823.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 15:15:00 | 844.80 | 828.90 | 825.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 841.32 | 841.89 | 837.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 863.20 | 841.89 | 837.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:15:00 | 906.36 | 877.95 | 861.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 909.16 | 909.41 | 899.15 | SL hit (close<ema200) qty=0.50 sl=909.41 alert=retest1 |

### Cycle 6 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 893.16 | 901.32 | 902.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 888.68 | 898.79 | 901.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 884.04 | 883.47 | 889.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 884.04 | 883.47 | 889.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 884.04 | 883.47 | 889.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 885.68 | 883.47 | 889.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 839.36 | 837.98 | 842.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 841.20 | 837.98 | 842.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 817.20 | 821.60 | 826.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 815.00 | 821.60 | 826.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 11:00:00 | 813.72 | 820.03 | 825.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:30:00 | 815.32 | 818.78 | 824.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 13:45:00 | 813.76 | 817.92 | 823.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 834.00 | 820.99 | 823.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 834.00 | 820.99 | 823.34 | SL hit (close>static) qty=1.00 sl=828.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 12:15:00 | 833.88 | 826.05 | 825.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 10:15:00 | 842.68 | 831.13 | 828.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 840.68 | 843.99 | 837.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 11:00:00 | 840.68 | 843.99 | 837.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 828.24 | 840.84 | 837.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 828.24 | 840.84 | 837.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 831.68 | 839.01 | 836.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:45:00 | 829.32 | 839.01 | 836.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 835.00 | 836.68 | 835.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 826.80 | 836.68 | 835.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 09:15:00 | 826.16 | 834.58 | 835.03 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 840.76 | 832.83 | 831.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 845.88 | 836.71 | 833.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 837.68 | 840.71 | 837.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 11:15:00 | 837.68 | 840.71 | 837.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 837.68 | 840.71 | 837.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 836.20 | 840.71 | 837.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 841.16 | 840.80 | 837.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 840.12 | 840.80 | 837.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 840.44 | 840.73 | 838.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:45:00 | 836.40 | 840.73 | 838.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 845.84 | 841.75 | 838.89 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 838.40 | 840.74 | 840.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 836.00 | 839.79 | 840.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 818.12 | 817.51 | 824.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 818.12 | 817.51 | 824.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 820.20 | 818.05 | 824.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 823.56 | 818.05 | 824.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 821.96 | 818.83 | 824.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 823.52 | 818.83 | 824.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 823.44 | 819.75 | 824.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:15:00 | 824.96 | 819.75 | 824.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 825.68 | 820.94 | 824.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 826.72 | 820.94 | 824.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 829.40 | 822.63 | 824.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 834.08 | 822.63 | 824.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 832.88 | 825.92 | 825.86 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 13:15:00 | 821.64 | 825.77 | 826.15 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 830.44 | 820.97 | 820.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 833.96 | 823.57 | 821.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 832.12 | 833.05 | 829.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 15:00:00 | 832.12 | 833.05 | 829.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 837.04 | 837.74 | 835.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:45:00 | 836.36 | 837.74 | 835.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 836.80 | 837.55 | 835.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:30:00 | 834.56 | 837.55 | 835.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 834.00 | 836.84 | 835.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 834.00 | 836.84 | 835.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 833.28 | 836.13 | 834.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 833.28 | 836.13 | 834.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 830.40 | 834.98 | 834.52 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 829.96 | 833.57 | 833.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 823.48 | 828.89 | 831.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 801.52 | 795.32 | 807.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 801.52 | 795.32 | 807.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 807.12 | 797.99 | 806.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 12:00:00 | 807.12 | 797.99 | 806.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 804.52 | 799.30 | 806.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 13:00:00 | 804.52 | 799.30 | 806.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 795.80 | 794.27 | 801.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 789.36 | 800.36 | 800.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:00:00 | 788.16 | 797.92 | 799.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 14:15:00 | 783.60 | 773.41 | 772.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 783.60 | 773.41 | 772.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 788.24 | 780.83 | 777.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 791.84 | 791.94 | 787.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:45:00 | 789.16 | 791.94 | 787.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 792.00 | 793.48 | 790.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 795.60 | 794.35 | 791.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:00:00 | 794.44 | 794.37 | 791.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 15:00:00 | 795.04 | 794.53 | 792.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 795.36 | 794.28 | 792.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 792.16 | 793.85 | 792.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 792.16 | 793.85 | 792.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 794.40 | 793.96 | 792.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 791.68 | 793.96 | 792.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 792.52 | 793.67 | 792.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 792.52 | 793.67 | 792.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 790.60 | 793.06 | 792.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:45:00 | 789.76 | 793.06 | 792.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-27 13:15:00 | 788.28 | 792.10 | 792.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 13:15:00 | 788.28 | 792.10 | 792.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 15:15:00 | 787.20 | 790.43 | 791.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 773.16 | 771.68 | 777.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 773.16 | 771.68 | 777.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 773.16 | 771.68 | 777.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 775.92 | 771.68 | 777.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 777.20 | 772.79 | 777.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 777.20 | 772.79 | 777.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 777.00 | 773.63 | 777.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:30:00 | 777.96 | 773.63 | 777.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 778.76 | 774.66 | 777.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 12:45:00 | 779.20 | 774.66 | 777.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 775.16 | 774.76 | 777.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 772.44 | 774.29 | 776.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 773.48 | 772.10 | 773.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 779.44 | 774.85 | 774.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 779.44 | 774.85 | 774.52 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 771.64 | 774.45 | 774.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 13:15:00 | 770.52 | 773.22 | 773.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 15:15:00 | 774.00 | 773.07 | 773.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 15:15:00 | 774.00 | 773.07 | 773.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 774.00 | 773.07 | 773.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 776.04 | 773.07 | 773.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 772.96 | 773.05 | 773.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:15:00 | 770.96 | 773.05 | 773.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 15:15:00 | 760.56 | 755.97 | 755.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 760.56 | 755.97 | 755.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 769.08 | 758.59 | 756.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 761.40 | 765.07 | 762.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 11:15:00 | 761.40 | 765.07 | 762.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 761.40 | 765.07 | 762.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 761.40 | 765.07 | 762.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 765.28 | 765.11 | 762.73 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 753.76 | 761.65 | 761.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 751.24 | 756.09 | 758.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 733.28 | 732.97 | 741.14 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 09:15:00 | 707.60 | 732.97 | 741.14 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 719.52 | 713.81 | 717.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-24 12:15:00 | 719.52 | 713.81 | 717.52 | SL hit (close>ema400) qty=1.00 sl=717.52 alert=retest1 |

### Cycle 21 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 720.28 | 714.19 | 714.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 727.16 | 716.78 | 715.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 15:15:00 | 735.20 | 736.76 | 732.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 15:15:00 | 735.20 | 736.76 | 732.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 735.20 | 736.76 | 732.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 730.92 | 736.76 | 732.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 726.40 | 734.69 | 732.09 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 718.36 | 729.96 | 730.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 713.96 | 726.76 | 728.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 706.84 | 699.70 | 706.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 706.84 | 699.70 | 706.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 706.84 | 699.70 | 706.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 706.84 | 699.70 | 706.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 712.28 | 702.22 | 706.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 711.28 | 702.22 | 706.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 710.96 | 705.04 | 707.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 710.96 | 705.04 | 707.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 711.04 | 706.24 | 707.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 712.20 | 706.24 | 707.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 713.60 | 708.72 | 708.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 717.16 | 710.40 | 709.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 716.36 | 716.99 | 713.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 716.36 | 716.99 | 713.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 714.76 | 717.53 | 715.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 714.76 | 717.53 | 715.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 711.04 | 716.23 | 714.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:45:00 | 711.08 | 716.23 | 714.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 711.88 | 715.36 | 714.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:45:00 | 710.36 | 715.36 | 714.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 714.80 | 714.99 | 714.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 716.80 | 714.99 | 714.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 712.76 | 714.55 | 714.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:30:00 | 711.08 | 714.55 | 714.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 714.24 | 714.48 | 714.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:30:00 | 712.80 | 714.48 | 714.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 713.16 | 714.22 | 714.28 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 715.68 | 714.51 | 714.41 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 702.68 | 712.31 | 713.44 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 13:15:00 | 708.60 | 707.78 | 707.78 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 703.04 | 706.86 | 707.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 699.40 | 705.37 | 706.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 697.64 | 695.64 | 699.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:30:00 | 697.36 | 695.64 | 699.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 698.16 | 696.14 | 699.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 697.20 | 696.14 | 699.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 699.20 | 696.76 | 699.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 694.36 | 696.76 | 699.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 696.24 | 696.65 | 698.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:30:00 | 687.64 | 693.32 | 696.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 09:15:00 | 653.26 | 663.96 | 672.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 09:15:00 | 631.64 | 630.80 | 639.17 | SL hit (close>ema200) qty=0.50 sl=630.80 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 665.40 | 643.83 | 641.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 669.48 | 648.96 | 644.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 670.32 | 674.52 | 667.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 670.32 | 674.52 | 667.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 670.32 | 674.52 | 667.69 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 658.92 | 666.29 | 666.78 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 670.60 | 667.31 | 666.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 678.24 | 669.50 | 667.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 680.24 | 682.82 | 678.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 14:00:00 | 680.24 | 682.82 | 678.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 678.68 | 681.99 | 678.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 678.68 | 681.99 | 678.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 675.00 | 680.66 | 678.25 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 667.20 | 676.45 | 676.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 12:15:00 | 664.80 | 674.12 | 675.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 663.88 | 662.47 | 666.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 15:00:00 | 663.88 | 662.47 | 666.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 658.76 | 662.04 | 665.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 655.52 | 659.62 | 663.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 622.74 | 637.01 | 646.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 631.52 | 630.55 | 636.02 | SL hit (close>ema200) qty=0.50 sl=630.55 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 644.00 | 637.21 | 637.00 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 627.36 | 635.78 | 636.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 625.88 | 633.80 | 635.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 622.28 | 619.96 | 625.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 622.28 | 619.96 | 625.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 624.80 | 620.93 | 625.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 624.80 | 620.93 | 625.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 626.88 | 622.12 | 625.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 626.88 | 622.12 | 625.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 629.96 | 623.69 | 625.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 629.96 | 623.69 | 625.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 632.80 | 627.85 | 627.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 647.68 | 631.81 | 629.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 10:15:00 | 643.48 | 643.48 | 638.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:30:00 | 642.88 | 643.48 | 638.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 638.96 | 642.11 | 639.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 638.96 | 642.11 | 639.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 639.64 | 641.62 | 639.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 642.80 | 641.62 | 639.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 666.96 | 680.89 | 681.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 666.96 | 680.89 | 681.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 662.08 | 674.92 | 678.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 662.76 | 660.76 | 666.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 662.76 | 660.76 | 666.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 625.40 | 622.46 | 625.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 625.92 | 622.46 | 625.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 624.08 | 622.79 | 625.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:30:00 | 625.40 | 622.79 | 625.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 624.80 | 621.09 | 622.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 624.80 | 621.09 | 622.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 626.20 | 622.11 | 622.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 626.20 | 622.11 | 622.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 627.80 | 623.25 | 623.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 628.72 | 625.14 | 624.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 11:15:00 | 624.76 | 626.14 | 625.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 11:15:00 | 624.76 | 626.14 | 625.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 624.76 | 626.14 | 625.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 624.76 | 626.14 | 625.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 624.48 | 625.81 | 625.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:30:00 | 622.72 | 625.81 | 625.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 619.48 | 624.54 | 624.65 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 12:15:00 | 628.00 | 624.59 | 624.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 633.28 | 626.33 | 625.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 625.44 | 627.30 | 626.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 09:15:00 | 625.44 | 627.30 | 626.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 625.44 | 627.30 | 626.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 625.44 | 627.30 | 626.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 628.60 | 627.56 | 626.29 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 622.76 | 625.09 | 625.40 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 629.92 | 625.29 | 625.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 632.00 | 626.63 | 625.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 628.04 | 628.19 | 626.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 628.04 | 628.19 | 626.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 628.80 | 628.74 | 627.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 628.80 | 628.74 | 627.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 629.04 | 628.80 | 627.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 628.00 | 628.80 | 627.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 628.04 | 628.65 | 627.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 621.36 | 628.65 | 627.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 621.04 | 627.13 | 627.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 618.28 | 627.13 | 627.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 608.40 | 623.38 | 625.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 607.20 | 613.56 | 619.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 606.12 | 602.85 | 607.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 606.12 | 602.85 | 607.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 606.44 | 603.56 | 607.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 15:15:00 | 604.00 | 603.56 | 607.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 603.92 | 603.95 | 606.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:45:00 | 604.24 | 604.27 | 606.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 604.04 | 604.27 | 606.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 609.28 | 605.27 | 606.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 609.28 | 605.27 | 606.91 | SL hit (close>static) qty=1.00 sl=608.40 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 14:15:00 | 609.32 | 599.93 | 598.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 615.96 | 607.40 | 604.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 13:15:00 | 619.72 | 619.72 | 615.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 13:45:00 | 619.00 | 619.72 | 615.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 616.60 | 618.48 | 615.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 616.60 | 618.48 | 615.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 616.04 | 617.99 | 615.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:45:00 | 616.88 | 617.99 | 615.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 615.88 | 617.57 | 615.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:00:00 | 615.88 | 617.57 | 615.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 615.60 | 617.17 | 615.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:30:00 | 615.20 | 617.17 | 615.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 619.48 | 617.64 | 615.98 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 603.44 | 613.40 | 614.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 595.52 | 608.64 | 612.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 605.48 | 602.33 | 607.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 605.48 | 602.33 | 607.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 610.28 | 604.19 | 607.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 610.28 | 604.19 | 607.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 614.64 | 606.28 | 607.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 615.08 | 606.28 | 607.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 615.68 | 609.50 | 609.11 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 603.24 | 610.77 | 611.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 593.76 | 606.70 | 609.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 599.40 | 594.41 | 599.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 599.40 | 594.41 | 599.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 599.40 | 594.41 | 599.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 599.40 | 594.41 | 599.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 601.96 | 595.92 | 599.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 601.96 | 595.92 | 599.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 595.80 | 595.90 | 599.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 591.76 | 594.94 | 598.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:15:00 | 595.28 | 595.57 | 597.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:00:00 | 594.92 | 595.44 | 597.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 15:00:00 | 594.84 | 595.64 | 597.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 604.64 | 597.26 | 597.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 604.64 | 597.26 | 597.65 | SL hit (close>static) qty=1.00 sl=602.44 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 604.48 | 598.71 | 598.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 608.32 | 601.41 | 599.63 | Break + close above crossover candle high |

### Cycle 48 — SELL (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 09:15:00 | 582.36 | 598.51 | 598.98 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 610.20 | 600.63 | 599.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 13:15:00 | 617.36 | 605.56 | 602.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 610.56 | 614.41 | 608.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 610.56 | 614.41 | 608.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 610.56 | 614.41 | 608.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 611.88 | 614.41 | 608.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 596.76 | 610.88 | 607.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 596.76 | 610.88 | 607.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 601.88 | 609.08 | 607.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 606.48 | 607.76 | 606.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 583.32 | 602.67 | 604.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 583.32 | 602.67 | 604.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 578.84 | 594.69 | 600.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 584.12 | 583.39 | 591.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 593.08 | 586.17 | 588.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 593.08 | 586.17 | 588.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 593.08 | 586.17 | 588.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 595.56 | 588.05 | 589.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 594.56 | 588.05 | 589.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 13:15:00 | 594.40 | 591.11 | 590.73 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 587.12 | 590.02 | 590.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 11:15:00 | 582.64 | 588.54 | 589.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 585.36 | 585.07 | 587.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 585.36 | 585.07 | 587.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 564.56 | 556.07 | 559.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 564.56 | 556.07 | 559.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 565.72 | 558.00 | 560.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 565.72 | 558.00 | 560.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 555.72 | 559.56 | 560.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 552.24 | 558.53 | 560.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:00:00 | 551.00 | 557.02 | 559.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 563.52 | 549.85 | 550.39 | SL hit (close>static) qty=1.00 sl=561.76 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 562.80 | 552.44 | 551.51 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 10:15:00 | 545.24 | 549.96 | 550.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 12:15:00 | 540.80 | 547.18 | 549.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 549.00 | 545.35 | 547.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 549.00 | 545.35 | 547.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 549.00 | 545.35 | 547.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 552.04 | 545.35 | 547.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 551.28 | 546.54 | 547.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 552.92 | 546.54 | 547.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 555.80 | 548.39 | 548.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:45:00 | 556.52 | 548.39 | 548.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 554.20 | 549.55 | 548.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 558.00 | 552.36 | 550.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 12:15:00 | 553.36 | 555.55 | 553.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 12:15:00 | 553.36 | 555.55 | 553.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 553.36 | 555.55 | 553.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 553.36 | 555.55 | 553.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 555.76 | 555.59 | 553.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:30:00 | 553.60 | 555.59 | 553.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 553.60 | 555.14 | 553.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 555.36 | 555.14 | 553.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 547.56 | 553.63 | 552.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 547.56 | 553.63 | 552.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 549.36 | 552.77 | 552.60 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 549.92 | 552.20 | 552.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 547.96 | 550.99 | 551.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 10:15:00 | 543.84 | 543.51 | 546.11 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 11:30:00 | 542.20 | 543.10 | 545.69 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 14:15:00 | 541.32 | 542.78 | 545.08 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 15:00:00 | 540.92 | 542.41 | 544.70 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 515.09 | 526.91 | 534.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 514.25 | 526.91 | 534.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 513.87 | 526.91 | 534.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-03-03 09:15:00 | 487.98 | 502.38 | 516.39 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 57 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 523.64 | 508.99 | 508.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 528.64 | 512.92 | 509.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 528.84 | 529.03 | 523.24 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:15:00 | 536.52 | 529.03 | 523.24 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 526.40 | 530.89 | 526.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 526.40 | 530.89 | 526.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 522.20 | 529.16 | 525.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-07 13:15:00 | 522.20 | 529.16 | 525.94 | SL hit (close<ema400) qty=1.00 sl=525.94 alert=retest1 |

### Cycle 58 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 520.04 | 524.49 | 524.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 518.08 | 523.21 | 524.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 515.96 | 515.54 | 518.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 13:45:00 | 516.44 | 515.54 | 518.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 515.84 | 515.60 | 518.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 517.20 | 515.60 | 518.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 517.12 | 515.83 | 518.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 520.64 | 515.83 | 518.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 513.16 | 515.30 | 517.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 510.60 | 515.30 | 517.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 14:15:00 | 517.96 | 515.34 | 516.77 | SL hit (close>static) qty=1.00 sl=517.76 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 529.96 | 517.99 | 516.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 531.44 | 525.07 | 520.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 539.32 | 539.53 | 532.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 539.32 | 539.53 | 532.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 564.92 | 568.61 | 563.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 563.36 | 568.61 | 563.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 563.20 | 567.52 | 563.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 563.20 | 567.52 | 563.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 564.24 | 566.87 | 563.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 564.24 | 566.87 | 563.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 560.88 | 565.67 | 563.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 560.88 | 565.67 | 563.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 556.36 | 563.81 | 562.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:00:00 | 556.36 | 563.81 | 562.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 550.40 | 561.13 | 561.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 548.00 | 558.50 | 560.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 561.20 | 556.72 | 558.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 561.20 | 556.72 | 558.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 561.20 | 556.72 | 558.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 561.20 | 556.72 | 558.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 558.64 | 557.10 | 558.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:30:00 | 558.24 | 557.01 | 558.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:45:00 | 558.44 | 557.95 | 558.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 12:15:00 | 561.92 | 558.27 | 557.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 12:15:00 | 561.92 | 558.27 | 557.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 12:15:00 | 563.16 | 560.32 | 559.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 555.96 | 566.43 | 564.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 555.96 | 566.43 | 564.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 555.96 | 566.43 | 564.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 555.96 | 566.43 | 564.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 557.84 | 564.71 | 564.07 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 554.72 | 562.71 | 563.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 552.92 | 560.75 | 562.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 528.88 | 527.08 | 538.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 14:15:00 | 538.32 | 532.03 | 536.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 538.32 | 532.03 | 536.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:00:00 | 538.32 | 532.03 | 536.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 539.68 | 533.56 | 537.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 532.56 | 533.56 | 537.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 12:15:00 | 543.20 | 534.14 | 536.01 | SL hit (close>static) qty=1.00 sl=540.24 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 549.00 | 538.58 | 537.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 553.60 | 542.99 | 540.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 548.48 | 548.99 | 544.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 548.48 | 548.99 | 544.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 552.56 | 549.18 | 545.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 15:00:00 | 558.20 | 551.22 | 547.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 15:15:00 | 564.80 | 566.74 | 566.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 564.80 | 566.74 | 566.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 553.12 | 564.02 | 565.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 11:15:00 | 552.00 | 549.52 | 554.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 12:00:00 | 552.00 | 549.52 | 554.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 554.72 | 550.51 | 553.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:15:00 | 549.48 | 551.08 | 552.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:45:00 | 549.48 | 550.98 | 552.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:15:00 | 549.64 | 550.94 | 552.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:45:00 | 549.24 | 550.80 | 551.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 546.24 | 543.93 | 546.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 547.32 | 543.93 | 546.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 552.64 | 545.67 | 546.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 552.64 | 545.67 | 546.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 553.40 | 547.22 | 547.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 552.80 | 548.34 | 547.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 552.80 | 548.34 | 547.80 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 544.36 | 548.28 | 548.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 539.56 | 545.22 | 546.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 542.84 | 542.34 | 544.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 542.84 | 542.34 | 544.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 543.88 | 542.33 | 544.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 543.88 | 542.33 | 544.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 543.28 | 542.52 | 544.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:30:00 | 543.28 | 542.52 | 544.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 544.12 | 542.84 | 544.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 540.84 | 542.84 | 544.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 543.20 | 542.91 | 544.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 537.60 | 542.28 | 543.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 09:30:00 | 536.48 | 527.03 | 531.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 12:15:00 | 546.56 | 535.61 | 534.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 546.56 | 535.61 | 534.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 547.76 | 538.04 | 535.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 552.40 | 552.93 | 548.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 552.92 | 552.93 | 548.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 551.44 | 552.47 | 549.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 550.32 | 552.47 | 549.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 551.24 | 552.30 | 550.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 555.12 | 552.30 | 550.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 574.88 | 582.99 | 583.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 10:15:00 | 574.88 | 582.99 | 583.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 12:15:00 | 569.88 | 578.93 | 581.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 15:15:00 | 578.00 | 577.72 | 580.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 09:15:00 | 583.80 | 577.72 | 580.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 585.24 | 579.22 | 580.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 584.56 | 579.22 | 580.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 586.08 | 580.59 | 581.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:30:00 | 586.04 | 580.59 | 581.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 585.24 | 582.09 | 581.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 589.92 | 583.66 | 582.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 623.20 | 623.38 | 616.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 623.20 | 623.38 | 616.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 620.88 | 622.70 | 620.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 620.88 | 622.70 | 620.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 622.08 | 622.58 | 620.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:30:00 | 621.04 | 622.58 | 620.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 621.08 | 622.28 | 620.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 626.44 | 622.28 | 620.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 15:15:00 | 640.76 | 642.03 | 642.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 640.76 | 642.03 | 642.04 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 644.36 | 642.50 | 642.25 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 638.84 | 641.81 | 642.12 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 644.80 | 642.41 | 642.36 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 10:15:00 | 641.04 | 642.14 | 642.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 11:15:00 | 640.08 | 641.73 | 642.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 611.72 | 611.66 | 618.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 611.72 | 611.66 | 618.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 588.00 | 587.95 | 591.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 587.60 | 587.95 | 591.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 11:45:00 | 587.56 | 588.67 | 591.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:15:00 | 587.72 | 589.18 | 590.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 598.36 | 590.78 | 591.17 | SL hit (close>static) qty=1.00 sl=592.32 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 602.24 | 593.08 | 592.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 604.88 | 600.33 | 597.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 599.16 | 600.72 | 598.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 599.16 | 600.72 | 598.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 599.16 | 600.72 | 598.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 599.72 | 600.72 | 598.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 596.96 | 599.97 | 598.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 596.96 | 599.97 | 598.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 600.80 | 600.13 | 598.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:30:00 | 602.16 | 600.34 | 598.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:15:00 | 601.68 | 600.34 | 598.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 597.32 | 603.35 | 603.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 597.32 | 603.35 | 603.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 595.76 | 598.93 | 601.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 15:15:00 | 596.48 | 596.40 | 598.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 09:15:00 | 596.76 | 596.40 | 598.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 600.44 | 597.20 | 598.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 600.00 | 597.20 | 598.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 599.88 | 597.74 | 598.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 601.52 | 597.74 | 598.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 599.80 | 597.43 | 598.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 603.45 | 597.43 | 598.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 599.95 | 597.93 | 598.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 600.00 | 597.93 | 598.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 600.00 | 598.21 | 598.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 600.00 | 598.21 | 598.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 598.80 | 598.33 | 598.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:00:00 | 598.80 | 598.33 | 598.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 599.95 | 598.66 | 598.55 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 591.10 | 597.20 | 597.91 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 608.00 | 598.29 | 597.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 610.00 | 600.64 | 598.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 619.35 | 622.18 | 615.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 619.35 | 622.18 | 615.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 618.30 | 621.41 | 615.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 618.30 | 621.41 | 615.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 615.95 | 618.89 | 616.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 615.95 | 618.89 | 616.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 614.15 | 617.94 | 615.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 613.70 | 617.94 | 615.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 620.35 | 618.43 | 616.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:45:00 | 623.05 | 618.23 | 617.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 623.85 | 619.42 | 618.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:45:00 | 622.60 | 620.47 | 618.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:30:00 | 621.75 | 621.00 | 619.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 620.00 | 620.70 | 619.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 620.95 | 620.21 | 619.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:15:00 | 621.80 | 620.03 | 619.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 616.95 | 619.78 | 619.70 | SL hit (close<static) qty=1.00 sl=618.25 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 611.35 | 618.09 | 618.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 611.00 | 614.74 | 616.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 614.50 | 614.12 | 615.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:45:00 | 614.00 | 614.12 | 615.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 615.60 | 614.11 | 615.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 612.90 | 614.11 | 615.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 609.90 | 613.26 | 614.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 608.85 | 613.26 | 614.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 608.70 | 608.38 | 609.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:00:00 | 608.35 | 608.64 | 609.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 578.41 | 588.06 | 591.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 578.26 | 588.06 | 591.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 577.93 | 588.06 | 591.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 12:15:00 | 580.10 | 578.88 | 583.55 | SL hit (close>ema200) qty=0.50 sl=578.88 alert=retest2 |

### Cycle 81 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 545.95 | 539.24 | 539.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 549.90 | 541.37 | 540.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 555.80 | 556.06 | 551.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 554.05 | 556.06 | 551.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 553.35 | 555.31 | 553.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 553.35 | 555.31 | 553.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 553.60 | 554.97 | 553.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 553.50 | 554.97 | 553.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 553.45 | 554.67 | 553.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 552.20 | 554.67 | 553.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 553.10 | 554.35 | 553.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 553.15 | 554.35 | 553.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 553.10 | 554.10 | 553.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:30:00 | 552.60 | 554.10 | 553.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 552.45 | 553.77 | 553.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 552.30 | 553.77 | 553.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 550.20 | 553.06 | 552.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 550.20 | 553.06 | 552.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 548.00 | 552.05 | 552.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 546.35 | 549.24 | 550.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 530.70 | 530.34 | 535.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:15:00 | 529.20 | 530.34 | 535.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 535.80 | 530.32 | 532.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 536.60 | 530.32 | 532.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 537.80 | 531.82 | 532.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 538.55 | 531.82 | 532.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 538.10 | 533.90 | 533.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 539.45 | 535.01 | 534.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 547.50 | 547.75 | 544.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 11:15:00 | 543.10 | 546.51 | 544.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 543.10 | 546.51 | 544.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 542.65 | 546.51 | 544.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 544.15 | 546.03 | 544.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 546.35 | 544.37 | 544.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 540.00 | 543.57 | 543.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 540.00 | 543.57 | 543.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 539.00 | 542.66 | 543.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 543.45 | 542.82 | 543.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 543.45 | 542.82 | 543.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 543.45 | 542.82 | 543.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 543.45 | 542.82 | 543.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 542.35 | 542.72 | 543.35 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 552.65 | 544.91 | 544.24 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 540.85 | 544.87 | 545.33 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 552.05 | 545.48 | 544.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 553.40 | 547.07 | 545.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 550.20 | 551.15 | 549.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:45:00 | 550.05 | 551.15 | 549.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 561.90 | 553.10 | 550.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 563.80 | 556.09 | 553.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 563.10 | 561.41 | 557.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:30:00 | 562.80 | 562.62 | 559.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 563.15 | 561.91 | 560.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 560.25 | 561.86 | 560.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 560.25 | 561.86 | 560.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 556.60 | 560.80 | 560.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 556.60 | 560.80 | 560.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 559.00 | 560.44 | 560.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 559.80 | 560.44 | 560.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 561.80 | 560.34 | 560.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 558.55 | 559.81 | 559.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 558.55 | 559.81 | 559.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 557.15 | 558.83 | 559.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 559.00 | 557.02 | 557.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 12:15:00 | 559.00 | 557.02 | 557.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 559.00 | 557.02 | 557.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:45:00 | 558.40 | 557.02 | 557.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 554.65 | 556.55 | 557.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:15:00 | 553.40 | 556.55 | 557.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 525.73 | 531.44 | 536.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 525.95 | 524.14 | 526.42 | SL hit (close>ema200) qty=0.50 sl=524.14 alert=retest2 |

### Cycle 89 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 529.40 | 526.57 | 526.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 530.45 | 527.92 | 527.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 530.00 | 530.57 | 529.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 530.00 | 530.57 | 529.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 529.70 | 530.38 | 529.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 529.70 | 530.38 | 529.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 530.15 | 530.34 | 529.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 532.50 | 530.53 | 529.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:30:00 | 532.00 | 530.92 | 530.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:15:00 | 533.60 | 531.08 | 530.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 532.50 | 531.21 | 530.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 531.70 | 532.09 | 531.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 531.70 | 532.09 | 531.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 531.30 | 531.93 | 531.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 530.10 | 531.93 | 531.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 530.05 | 531.55 | 531.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 530.35 | 531.55 | 531.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 527.30 | 530.70 | 530.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 527.30 | 530.70 | 530.75 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 537.80 | 531.69 | 531.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 544.10 | 538.08 | 535.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 533.15 | 538.56 | 537.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 533.15 | 538.56 | 537.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 533.15 | 538.56 | 537.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 534.70 | 538.56 | 537.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 531.30 | 537.11 | 536.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 531.10 | 537.11 | 536.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 531.05 | 535.90 | 535.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 529.70 | 534.66 | 535.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 529.10 | 528.35 | 530.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:15:00 | 529.60 | 528.35 | 530.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 529.90 | 529.09 | 530.40 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 534.40 | 531.37 | 530.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 535.25 | 532.15 | 531.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 531.75 | 532.66 | 531.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 531.75 | 532.66 | 531.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 531.75 | 532.66 | 531.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 531.65 | 532.66 | 531.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 532.00 | 532.53 | 531.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 533.00 | 532.53 | 531.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 531.00 | 532.22 | 531.78 | SL hit (close<static) qty=1.00 sl=531.20 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 526.75 | 531.13 | 531.32 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 534.70 | 531.44 | 531.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 537.60 | 534.58 | 533.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 534.20 | 535.31 | 534.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 534.20 | 535.31 | 534.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 534.20 | 535.31 | 534.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 534.20 | 535.31 | 534.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 535.60 | 535.36 | 534.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 538.50 | 535.36 | 534.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 542.90 | 546.98 | 547.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 542.90 | 546.98 | 547.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 542.55 | 546.09 | 546.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 523.15 | 522.90 | 529.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 523.15 | 522.90 | 529.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 522.30 | 521.02 | 523.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 522.00 | 521.02 | 523.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 523.35 | 521.48 | 523.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 522.45 | 521.48 | 523.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 524.00 | 521.99 | 523.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 529.90 | 521.99 | 523.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 527.90 | 523.17 | 523.73 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 533.45 | 525.23 | 524.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 537.50 | 527.68 | 525.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 528.45 | 529.98 | 527.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 528.45 | 529.98 | 527.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 528.45 | 529.98 | 527.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 528.20 | 529.98 | 527.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 528.15 | 529.61 | 527.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 528.85 | 529.61 | 527.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 528.45 | 529.38 | 528.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 528.50 | 529.38 | 528.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 526.30 | 528.76 | 527.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 526.00 | 528.76 | 527.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 524.75 | 527.96 | 527.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 524.75 | 527.96 | 527.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 524.20 | 527.21 | 527.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 521.00 | 525.45 | 526.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 11:15:00 | 518.95 | 517.77 | 519.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 11:15:00 | 518.95 | 517.77 | 519.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 518.95 | 517.77 | 519.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 518.95 | 517.77 | 519.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 521.05 | 518.42 | 519.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 521.10 | 518.42 | 519.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 523.00 | 519.34 | 520.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 525.00 | 519.34 | 520.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 15:15:00 | 524.00 | 520.93 | 520.82 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 519.00 | 520.85 | 520.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 517.00 | 519.66 | 520.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 10:15:00 | 518.00 | 516.98 | 518.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 518.00 | 516.98 | 518.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 518.00 | 516.98 | 518.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 518.75 | 516.98 | 518.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 516.50 | 516.88 | 518.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:45:00 | 512.55 | 515.35 | 517.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:45:00 | 515.05 | 514.25 | 515.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 515.15 | 514.49 | 515.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:45:00 | 513.90 | 513.93 | 514.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 512.40 | 512.93 | 514.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 521.00 | 514.39 | 514.48 | SL hit (close>static) qty=1.00 sl=519.35 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 519.55 | 515.43 | 514.94 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 514.20 | 515.47 | 515.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 11:15:00 | 512.25 | 514.38 | 515.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 505.50 | 504.06 | 506.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 12:00:00 | 505.50 | 504.06 | 506.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 508.50 | 505.45 | 506.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 510.05 | 505.45 | 506.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 511.75 | 506.71 | 506.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 511.75 | 506.71 | 506.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 513.05 | 507.98 | 507.34 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 503.35 | 508.37 | 509.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 498.60 | 506.42 | 508.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 503.10 | 502.99 | 505.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 503.10 | 502.99 | 505.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 507.10 | 504.18 | 505.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 507.10 | 504.18 | 505.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 507.55 | 504.85 | 505.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 508.70 | 504.85 | 505.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 505.00 | 505.03 | 505.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 507.25 | 505.03 | 505.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 509.70 | 505.97 | 506.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 512.70 | 505.97 | 506.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 507.55 | 506.28 | 506.30 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 507.30 | 506.49 | 506.39 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 505.20 | 506.26 | 506.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 504.05 | 505.81 | 506.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 10:15:00 | 505.65 | 505.11 | 505.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 10:15:00 | 505.65 | 505.11 | 505.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 505.65 | 505.11 | 505.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 506.20 | 505.11 | 505.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 506.95 | 505.48 | 505.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 506.95 | 505.48 | 505.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 505.80 | 505.54 | 505.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 507.00 | 505.54 | 505.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 504.50 | 505.33 | 505.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 507.05 | 505.33 | 505.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 505.95 | 505.46 | 505.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 505.95 | 505.46 | 505.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 507.10 | 505.78 | 505.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 500.55 | 505.78 | 505.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 501.20 | 504.87 | 505.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 15:15:00 | 498.30 | 501.13 | 502.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 498.25 | 500.16 | 501.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 498.45 | 496.29 | 498.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:45:00 | 498.30 | 496.67 | 498.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 498.10 | 496.96 | 498.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:30:00 | 498.25 | 496.96 | 498.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 498.45 | 497.26 | 498.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 498.45 | 497.26 | 498.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 498.45 | 497.50 | 498.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 500.05 | 497.50 | 498.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 497.85 | 497.57 | 498.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 500.00 | 497.57 | 498.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 498.25 | 497.70 | 498.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 498.05 | 497.70 | 498.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 498.90 | 497.94 | 498.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 498.90 | 497.94 | 498.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 497.30 | 497.81 | 498.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 499.50 | 497.81 | 498.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 499.30 | 498.11 | 498.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 499.30 | 498.11 | 498.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 501.85 | 498.86 | 498.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 501.85 | 498.86 | 498.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 506.00 | 500.72 | 499.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 511.15 | 511.73 | 508.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 511.15 | 511.73 | 508.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 508.00 | 510.18 | 508.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 508.00 | 510.18 | 508.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 508.55 | 509.85 | 508.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 507.50 | 509.85 | 508.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 509.05 | 509.69 | 508.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 511.90 | 509.69 | 508.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 512.25 | 510.20 | 509.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 515.95 | 510.20 | 509.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 521.35 | 528.39 | 528.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 521.35 | 528.39 | 528.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 515.60 | 522.27 | 525.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 513.00 | 512.83 | 516.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:30:00 | 513.05 | 512.83 | 516.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 516.60 | 513.61 | 516.12 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 520.20 | 516.91 | 516.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 522.85 | 519.18 | 517.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 518.15 | 520.10 | 518.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 518.15 | 520.10 | 518.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 518.15 | 520.10 | 518.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 518.15 | 520.10 | 518.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 518.65 | 519.81 | 518.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 519.80 | 519.81 | 518.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 520.00 | 519.85 | 518.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 517.55 | 519.85 | 518.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 519.60 | 519.80 | 519.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 517.80 | 519.80 | 519.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 518.50 | 519.54 | 518.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 518.30 | 519.54 | 518.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 515.90 | 518.81 | 518.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:00:00 | 515.90 | 518.81 | 518.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 514.60 | 517.97 | 518.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 514.00 | 516.21 | 517.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 505.40 | 498.77 | 503.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 505.40 | 498.77 | 503.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 505.40 | 498.77 | 503.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 505.40 | 498.77 | 503.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 503.00 | 499.61 | 503.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 500.25 | 499.85 | 502.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 475.24 | 485.76 | 492.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 484.90 | 482.93 | 488.02 | SL hit (close>ema200) qty=0.50 sl=482.93 alert=retest2 |

### Cycle 111 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 496.40 | 490.52 | 489.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 497.00 | 491.82 | 490.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 499.60 | 505.64 | 501.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 499.60 | 505.64 | 501.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 499.60 | 505.64 | 501.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 506.05 | 505.64 | 501.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 504.30 | 505.37 | 501.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:15:00 | 506.80 | 505.37 | 501.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 507.35 | 505.40 | 502.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 509.55 | 504.85 | 502.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 499.45 | 503.95 | 502.35 | SL hit (close<static) qty=1.00 sl=501.25 alert=retest2 |

### Cycle 112 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 512.40 | 519.46 | 519.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 509.35 | 516.15 | 518.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 512.80 | 512.43 | 515.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 512.80 | 512.43 | 515.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 514.70 | 512.88 | 515.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:30:00 | 514.00 | 512.88 | 515.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 515.05 | 513.46 | 515.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:00:00 | 515.05 | 513.46 | 515.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 513.15 | 513.40 | 514.88 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 522.00 | 516.62 | 516.03 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 513.85 | 515.73 | 515.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 510.75 | 514.74 | 515.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 516.40 | 515.07 | 515.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 516.40 | 515.07 | 515.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 516.40 | 515.07 | 515.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 516.40 | 515.07 | 515.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 512.35 | 514.53 | 515.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 11:30:00 | 510.00 | 512.40 | 513.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:45:00 | 507.20 | 504.53 | 504.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 506.70 | 505.07 | 505.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 506.70 | 505.07 | 505.06 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 504.35 | 504.93 | 504.99 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 506.15 | 505.17 | 505.10 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 15:15:00 | 503.75 | 504.89 | 504.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 09:15:00 | 502.20 | 504.35 | 504.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 505.45 | 504.57 | 504.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 10:15:00 | 505.45 | 504.57 | 504.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 505.45 | 504.57 | 504.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 505.45 | 504.57 | 504.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 506.70 | 505.00 | 504.96 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 504.75 | 505.23 | 505.26 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 14:15:00 | 505.50 | 505.28 | 505.28 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 501.10 | 504.45 | 504.90 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 510.75 | 505.17 | 504.80 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 502.20 | 505.01 | 505.09 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 505.65 | 504.76 | 504.74 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 502.25 | 504.26 | 504.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 499.50 | 503.31 | 504.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 500.00 | 499.82 | 501.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:15:00 | 496.55 | 499.82 | 501.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 500.00 | 497.94 | 499.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 500.00 | 497.94 | 499.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 493.80 | 497.11 | 498.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 501.00 | 497.11 | 498.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 471.95 | 468.28 | 473.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 471.65 | 468.28 | 473.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 480.80 | 470.79 | 474.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 480.80 | 470.79 | 474.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 481.70 | 472.97 | 474.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 482.80 | 472.97 | 474.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 485.80 | 477.74 | 476.90 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 470.40 | 478.15 | 478.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 11:15:00 | 468.50 | 476.22 | 477.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 474.50 | 473.80 | 475.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 15:00:00 | 474.50 | 473.80 | 475.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 473.00 | 473.95 | 475.53 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 478.25 | 475.12 | 475.09 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 469.30 | 474.96 | 475.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 469.00 | 473.77 | 474.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 455.20 | 453.02 | 458.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 455.00 | 453.02 | 458.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 453.95 | 453.19 | 457.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 450.45 | 452.86 | 456.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 460.65 | 455.12 | 456.02 | SL hit (close>static) qty=1.00 sl=458.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 462.00 | 457.38 | 456.95 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 449.20 | 456.50 | 456.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 446.70 | 454.54 | 455.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 446.40 | 446.37 | 450.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:30:00 | 447.55 | 446.37 | 450.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 451.05 | 447.30 | 450.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 451.05 | 447.30 | 450.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 452.05 | 448.25 | 450.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 452.45 | 448.25 | 450.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 448.80 | 448.65 | 450.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 449.60 | 448.65 | 450.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 432.85 | 429.64 | 435.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 432.45 | 429.64 | 435.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 433.80 | 430.47 | 435.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 436.40 | 430.47 | 435.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 439.00 | 432.60 | 435.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 444.85 | 432.60 | 435.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 446.75 | 435.43 | 436.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 446.75 | 435.43 | 436.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 447.85 | 437.92 | 437.32 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 436.60 | 438.88 | 439.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 430.85 | 436.89 | 438.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 443.35 | 432.57 | 434.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 443.35 | 432.57 | 434.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 443.35 | 432.57 | 434.46 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 451.10 | 438.32 | 436.87 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 431.30 | 437.29 | 437.80 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 442.00 | 438.52 | 438.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 447.55 | 440.32 | 438.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 476.95 | 481.34 | 476.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 476.95 | 481.34 | 476.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 476.95 | 481.34 | 476.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 479.00 | 481.34 | 476.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 480.80 | 480.74 | 476.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 502.75 | 506.08 | 506.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 502.75 | 506.08 | 506.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 500.35 | 503.08 | 504.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 503.70 | 503.19 | 504.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 509.30 | 503.19 | 504.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 509.50 | 504.45 | 504.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 512.00 | 504.45 | 504.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 514.95 | 506.55 | 505.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 518.80 | 509.00 | 506.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 513.75 | 513.95 | 511.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:30:00 | 514.80 | 513.95 | 511.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 509.80 | 513.12 | 511.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 510.40 | 513.12 | 511.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 509.65 | 512.42 | 510.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:15:00 | 511.05 | 512.42 | 510.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 510.00 | 511.94 | 510.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 510.00 | 511.94 | 510.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 509.00 | 511.35 | 510.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 510.65 | 511.35 | 510.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 514.55 | 515.40 | 513.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:45:00 | 515.30 | 515.40 | 513.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 513.55 | 514.96 | 513.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 509.45 | 514.96 | 513.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 506.60 | 513.29 | 512.81 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 509.80 | 512.05 | 512.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 13:15:00 | 508.20 | 510.72 | 511.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 515.85 | 511.17 | 511.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 515.85 | 511.17 | 511.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 515.85 | 511.17 | 511.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 515.85 | 511.17 | 511.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 518.50 | 512.64 | 512.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 520.35 | 516.67 | 514.99 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:30:00 | 797.96 | 2024-05-14 10:15:00 | 812.80 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-05-13 12:45:00 | 805.20 | 2024-05-14 10:15:00 | 812.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest1 | 2024-05-16 09:15:00 | 828.08 | 2024-05-16 13:15:00 | 819.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-05-16 14:45:00 | 829.40 | 2024-05-17 09:15:00 | 814.88 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-05-17 10:15:00 | 833.60 | 2024-05-28 11:15:00 | 861.04 | STOP_HIT | 1.00 | 3.29% |
| BUY | retest1 | 2024-06-11 09:15:00 | 863.20 | 2024-06-12 10:15:00 | 906.36 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-11 09:15:00 | 863.20 | 2024-06-14 11:15:00 | 909.16 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2024-07-03 10:15:00 | 815.00 | 2024-07-04 09:15:00 | 834.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-07-03 11:00:00 | 813.72 | 2024-07-04 09:15:00 | 834.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-07-03 12:30:00 | 815.32 | 2024-07-04 09:15:00 | 834.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-07-03 13:45:00 | 813.76 | 2024-07-04 09:15:00 | 834.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-08-09 09:15:00 | 789.36 | 2024-08-16 14:15:00 | 783.60 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-08-09 10:00:00 | 788.16 | 2024-08-16 14:15:00 | 783.60 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-08-26 09:30:00 | 795.60 | 2024-08-27 13:15:00 | 788.28 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-08-26 11:00:00 | 794.44 | 2024-08-27 13:15:00 | 788.28 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-08-26 15:00:00 | 795.04 | 2024-08-27 13:15:00 | 788.28 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-08-27 09:15:00 | 795.36 | 2024-08-27 13:15:00 | 788.28 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-08-30 15:00:00 | 772.44 | 2024-09-03 14:15:00 | 779.44 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-09-03 10:15:00 | 773.48 | 2024-09-03 14:15:00 | 779.44 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-09-05 10:15:00 | 770.96 | 2024-09-12 15:15:00 | 760.56 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest1 | 2024-09-20 09:15:00 | 707.60 | 2024-09-24 12:15:00 | 719.52 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-09-24 14:45:00 | 715.68 | 2024-09-27 09:15:00 | 720.28 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-10-21 12:30:00 | 687.64 | 2024-10-24 09:15:00 | 653.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:30:00 | 687.64 | 2024-10-29 09:15:00 | 631.64 | STOP_HIT | 0.50 | 8.14% |
| SELL | retest2 | 2024-11-12 12:45:00 | 655.52 | 2024-11-14 09:15:00 | 622.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 655.52 | 2024-11-18 12:15:00 | 631.52 | STOP_HIT | 0.50 | 3.66% |
| BUY | retest2 | 2024-11-27 09:15:00 | 642.80 | 2024-12-12 09:15:00 | 666.96 | STOP_HIT | 1.00 | 3.76% |
| SELL | retest2 | 2025-01-08 15:15:00 | 604.00 | 2025-01-09 12:15:00 | 609.28 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-01-09 10:45:00 | 603.92 | 2025-01-09 12:15:00 | 609.28 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-01-09 11:45:00 | 604.24 | 2025-01-09 12:15:00 | 609.28 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-01-09 12:15:00 | 604.04 | 2025-01-09 12:15:00 | 609.28 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-01-09 14:15:00 | 607.04 | 2025-01-14 14:15:00 | 609.32 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-01-09 15:00:00 | 606.40 | 2025-01-14 14:15:00 | 609.32 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-01-14 13:30:00 | 606.36 | 2025-01-14 14:15:00 | 609.32 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-01-28 14:45:00 | 591.76 | 2025-01-30 09:15:00 | 604.64 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-01-29 12:15:00 | 595.28 | 2025-01-30 09:15:00 | 604.64 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-01-29 13:00:00 | 594.92 | 2025-01-30 09:15:00 | 604.64 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-01-29 15:00:00 | 594.84 | 2025-01-30 09:15:00 | 604.64 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-02-01 15:15:00 | 606.48 | 2025-02-03 09:15:00 | 583.32 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-02-14 09:15:00 | 552.24 | 2025-02-17 14:15:00 | 563.52 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-02-14 10:00:00 | 551.00 | 2025-02-17 14:15:00 | 563.52 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest1 | 2025-02-25 11:30:00 | 542.20 | 2025-02-28 09:15:00 | 515.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-25 14:15:00 | 541.32 | 2025-02-28 09:15:00 | 514.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-25 15:00:00 | 540.92 | 2025-02-28 09:15:00 | 513.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-25 11:30:00 | 542.20 | 2025-03-03 09:15:00 | 487.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-02-25 14:15:00 | 541.32 | 2025-03-03 09:15:00 | 487.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-02-25 15:00:00 | 540.92 | 2025-03-03 09:15:00 | 486.83 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-03-07 09:15:00 | 536.52 | 2025-03-07 13:15:00 | 522.20 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-03-12 11:15:00 | 510.60 | 2025-03-12 14:15:00 | 517.96 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-03-13 15:00:00 | 512.40 | 2025-03-18 09:15:00 | 529.96 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-03-17 10:15:00 | 512.28 | 2025-03-18 09:15:00 | 529.96 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-03-28 09:30:00 | 558.24 | 2025-04-01 12:15:00 | 561.92 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-03-28 11:45:00 | 558.44 | 2025-04-01 12:15:00 | 561.92 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-04-09 09:15:00 | 532.56 | 2025-04-09 12:15:00 | 543.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-04-15 15:00:00 | 558.20 | 2025-04-24 15:15:00 | 564.80 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-04-29 14:15:00 | 549.48 | 2025-05-05 11:15:00 | 552.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-04-30 09:45:00 | 549.48 | 2025-05-05 11:15:00 | 552.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-04-30 11:15:00 | 549.64 | 2025-05-05 11:15:00 | 552.80 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-04-30 11:45:00 | 549.24 | 2025-05-05 11:15:00 | 552.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-05-08 13:15:00 | 537.60 | 2025-05-12 12:15:00 | 546.56 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-05-12 09:30:00 | 536.48 | 2025-05-12 12:15:00 | 546.56 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-05-16 09:15:00 | 555.12 | 2025-05-23 10:15:00 | 574.88 | STOP_HIT | 1.00 | 3.56% |
| BUY | retest2 | 2025-06-03 09:15:00 | 626.44 | 2025-06-09 15:15:00 | 640.76 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2025-06-23 09:15:00 | 587.60 | 2025-06-24 09:15:00 | 598.36 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-06-23 11:45:00 | 587.56 | 2025-06-24 09:15:00 | 598.36 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-06-23 15:15:00 | 587.72 | 2025-06-24 09:15:00 | 598.36 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-06-26 12:30:00 | 602.16 | 2025-07-01 11:15:00 | 597.32 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-06-26 13:15:00 | 601.68 | 2025-07-01 11:15:00 | 597.32 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-15 12:45:00 | 623.05 | 2025-07-18 10:15:00 | 616.95 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-16 10:15:00 | 623.85 | 2025-07-18 10:15:00 | 616.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-16 11:45:00 | 622.60 | 2025-07-18 11:15:00 | 611.35 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-07-16 13:30:00 | 621.75 | 2025-07-18 11:15:00 | 611.35 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-17 11:30:00 | 620.95 | 2025-07-18 11:15:00 | 611.35 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-07-17 13:15:00 | 621.80 | 2025-07-18 11:15:00 | 611.35 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-07-22 10:15:00 | 608.85 | 2025-07-31 09:15:00 | 578.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 09:15:00 | 608.70 | 2025-07-31 09:15:00 | 578.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 11:00:00 | 608.35 | 2025-07-31 09:15:00 | 577.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 608.85 | 2025-08-01 12:15:00 | 580.10 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-07-24 09:15:00 | 608.70 | 2025-08-01 12:15:00 | 580.10 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2025-07-24 11:00:00 | 608.35 | 2025-08-01 12:15:00 | 580.10 | STOP_HIT | 0.50 | 4.64% |
| BUY | retest2 | 2025-09-05 09:30:00 | 546.35 | 2025-09-05 11:15:00 | 540.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-09-16 09:15:00 | 563.80 | 2025-09-19 10:15:00 | 558.55 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-16 14:00:00 | 563.10 | 2025-09-19 10:15:00 | 558.55 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-17 11:30:00 | 562.80 | 2025-09-19 10:15:00 | 558.55 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-09-18 09:15:00 | 563.15 | 2025-09-19 10:15:00 | 558.55 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-09-18 14:15:00 | 559.80 | 2025-09-19 10:15:00 | 558.55 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-09-19 09:15:00 | 561.80 | 2025-09-19 10:15:00 | 558.55 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-22 14:15:00 | 553.40 | 2025-09-26 09:15:00 | 525.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:15:00 | 553.40 | 2025-09-30 14:15:00 | 525.95 | STOP_HIT | 0.50 | 4.96% |
| BUY | retest2 | 2025-10-06 15:15:00 | 532.50 | 2025-10-08 14:15:00 | 527.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-07 10:30:00 | 532.00 | 2025-10-08 14:15:00 | 527.30 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-07 14:15:00 | 533.60 | 2025-10-08 14:15:00 | 527.30 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-07 15:15:00 | 532.50 | 2025-10-08 14:15:00 | 527.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-17 11:15:00 | 533.00 | 2025-10-17 11:15:00 | 531.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-10-24 09:15:00 | 538.50 | 2025-11-04 11:15:00 | 542.90 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-11-21 14:45:00 | 512.55 | 2025-11-26 09:15:00 | 521.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-11-25 09:45:00 | 515.05 | 2025-11-26 09:15:00 | 521.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-11-25 10:45:00 | 515.15 | 2025-11-26 09:15:00 | 521.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-25 11:45:00 | 513.90 | 2025-11-26 09:15:00 | 521.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-16 15:15:00 | 498.30 | 2025-12-19 14:15:00 | 501.85 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-17 10:15:00 | 498.25 | 2025-12-19 14:15:00 | 501.85 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-18 12:00:00 | 498.45 | 2025-12-19 14:15:00 | 501.85 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-12-18 12:45:00 | 498.30 | 2025-12-19 14:15:00 | 501.85 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-12-26 10:15:00 | 515.95 | 2026-01-08 11:15:00 | 521.35 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2026-01-22 11:30:00 | 500.25 | 2026-01-27 09:15:00 | 475.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 500.25 | 2026-01-27 14:15:00 | 484.90 | STOP_HIT | 0.50 | 3.07% |
| BUY | retest2 | 2026-02-01 14:15:00 | 506.80 | 2026-02-02 10:15:00 | 499.45 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-02-01 14:45:00 | 507.35 | 2026-02-02 10:15:00 | 499.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-02-02 09:15:00 | 509.55 | 2026-02-02 10:15:00 | 499.45 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-02 15:00:00 | 508.15 | 2026-02-05 15:15:00 | 519.00 | STOP_HIT | 1.00 | 2.14% |
| BUY | retest2 | 2026-02-05 11:45:00 | 522.95 | 2026-02-06 10:15:00 | 512.40 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-12 11:30:00 | 510.00 | 2026-02-17 12:15:00 | 506.70 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2026-02-17 10:45:00 | 507.20 | 2026-02-17 12:15:00 | 506.70 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2026-03-17 12:15:00 | 450.45 | 2026-03-18 10:15:00 | 460.65 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-04-13 10:15:00 | 479.00 | 2026-04-24 09:15:00 | 502.75 | STOP_HIT | 1.00 | 4.96% |
| BUY | retest2 | 2026-04-13 10:45:00 | 480.80 | 2026-04-24 09:15:00 | 502.75 | STOP_HIT | 1.00 | 4.57% |
