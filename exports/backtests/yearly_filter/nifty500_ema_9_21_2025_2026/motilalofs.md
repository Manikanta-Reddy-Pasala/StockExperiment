# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 882.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 44 |
| ALERT2 | 44 |
| ALERT2_SKIP | 19 |
| ALERT3 | 116 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 21 / 28
- **Target hits / Stop hits / Partials:** 2 / 42 / 5
- **Avg / median % per leg:** 0.90% / -0.14%
- **Sum % (uncompounded):** 44.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 9 | 33.3% | 2 | 25 | 0 | 0.44% | 11.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 9 | 33.3% | 2 | 25 | 0 | 0.44% | 11.8% |
| SELL (all) | 22 | 12 | 54.5% | 0 | 17 | 5 | 1.47% | 32.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 12 | 54.5% | 0 | 17 | 5 | 1.47% | 32.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 21 | 42.9% | 2 | 42 | 5 | 0.90% | 44.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 779.35 | 783.73 | 783.76 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 794.75 | 784.99 | 784.27 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 15:15:00 | 782.50 | 784.18 | 784.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 779.50 | 783.25 | 783.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 780.00 | 779.35 | 780.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 780.00 | 779.35 | 780.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 780.00 | 779.35 | 780.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 780.00 | 779.35 | 780.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 784.40 | 780.36 | 781.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:45:00 | 787.05 | 780.36 | 781.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 786.85 | 781.66 | 781.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:45:00 | 788.65 | 781.66 | 781.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 786.10 | 782.55 | 782.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 790.05 | 784.05 | 782.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 801.80 | 805.63 | 798.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 801.80 | 805.63 | 798.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 804.25 | 805.35 | 798.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 812.00 | 805.35 | 798.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:15:00 | 810.00 | 807.01 | 801.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 11:45:00 | 810.15 | 812.75 | 809.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 15:15:00 | 802.00 | 807.87 | 808.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 802.00 | 807.87 | 808.20 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 825.35 | 811.36 | 809.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 842.60 | 829.55 | 822.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 884.50 | 885.86 | 870.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:30:00 | 887.80 | 885.86 | 870.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 871.15 | 880.23 | 871.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:45:00 | 872.85 | 880.23 | 871.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 875.00 | 879.18 | 871.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 15:00:00 | 879.85 | 879.32 | 872.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 868.20 | 878.08 | 873.00 | SL hit (close<static) qty=1.00 sl=870.70 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 850.10 | 866.96 | 868.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 847.95 | 863.15 | 866.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 820.70 | 817.40 | 826.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 820.70 | 817.40 | 826.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 829.10 | 819.67 | 826.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 829.10 | 819.67 | 826.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 821.65 | 820.06 | 825.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 823.90 | 820.06 | 825.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 826.00 | 819.14 | 822.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 826.00 | 819.14 | 822.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 825.05 | 820.32 | 822.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 827.55 | 820.32 | 822.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 819.30 | 820.48 | 822.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 815.65 | 819.60 | 821.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 816.80 | 818.98 | 820.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 815.90 | 818.98 | 820.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:00:00 | 815.95 | 818.37 | 820.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 819.90 | 817.82 | 819.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 819.90 | 817.82 | 819.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 816.00 | 817.46 | 819.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 820.00 | 817.46 | 819.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 830.80 | 820.13 | 820.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 830.80 | 820.13 | 820.16 | SL hit (close>static) qty=1.00 sl=823.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 826.45 | 821.39 | 820.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 848.40 | 831.26 | 826.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 861.00 | 863.91 | 854.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 861.00 | 863.91 | 854.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 859.25 | 861.65 | 856.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:45:00 | 857.65 | 861.65 | 856.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 859.20 | 861.16 | 856.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 868.05 | 861.16 | 856.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 14:15:00 | 855.30 | 863.92 | 860.94 | SL hit (close<static) qty=1.00 sl=855.50 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 851.35 | 863.63 | 865.29 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 902.30 | 866.78 | 865.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 903.95 | 874.22 | 868.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 926.00 | 927.62 | 917.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 10:00:00 | 926.00 | 927.62 | 917.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 917.40 | 923.91 | 917.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 927.50 | 921.43 | 918.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:00:00 | 923.00 | 922.21 | 919.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 923.50 | 922.51 | 919.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:00:00 | 922.60 | 923.17 | 920.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 932.90 | 934.44 | 929.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 928.85 | 934.44 | 929.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 934.10 | 934.37 | 929.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 938.00 | 934.84 | 931.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 922.45 | 929.92 | 929.76 | SL hit (close<static) qty=1.00 sl=924.55 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 922.25 | 928.39 | 929.08 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 934.80 | 930.08 | 929.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 943.70 | 933.16 | 931.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 939.80 | 940.45 | 936.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:30:00 | 938.80 | 940.45 | 936.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 936.60 | 939.68 | 936.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 936.60 | 939.68 | 936.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 936.35 | 939.01 | 936.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:30:00 | 939.20 | 938.52 | 936.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 939.00 | 938.01 | 936.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 12:15:00 | 932.80 | 936.84 | 936.64 | SL hit (close<static) qty=1.00 sl=934.20 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 930.85 | 935.64 | 936.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 927.35 | 932.67 | 934.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 924.00 | 920.34 | 926.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 924.00 | 920.34 | 926.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 931.50 | 922.57 | 926.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 931.50 | 922.57 | 926.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 935.25 | 925.11 | 927.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 935.50 | 925.11 | 927.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 933.15 | 928.52 | 928.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 933.15 | 928.52 | 928.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 932.70 | 929.36 | 928.94 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 923.80 | 927.87 | 928.31 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 935.75 | 929.75 | 929.01 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 918.00 | 928.20 | 929.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 905.35 | 924.03 | 926.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 907.00 | 907.00 | 914.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 899.65 | 907.00 | 914.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 903.65 | 896.56 | 902.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:00:00 | 903.65 | 896.56 | 902.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 911.50 | 899.54 | 903.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 911.50 | 899.54 | 903.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 914.90 | 902.62 | 904.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 914.90 | 902.62 | 904.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 935.90 | 909.27 | 907.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 940.85 | 915.59 | 910.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 913.05 | 924.35 | 919.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 913.05 | 924.35 | 919.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 913.05 | 924.35 | 919.15 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 911.20 | 917.41 | 917.59 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 927.85 | 918.54 | 918.02 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 914.45 | 921.37 | 921.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 908.10 | 916.14 | 918.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 915.65 | 913.44 | 916.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 12:00:00 | 915.65 | 913.44 | 916.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 916.00 | 913.95 | 916.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:15:00 | 917.20 | 913.95 | 916.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 922.00 | 915.56 | 917.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:30:00 | 921.00 | 915.56 | 917.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 925.00 | 917.45 | 917.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 926.05 | 917.45 | 917.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 922.30 | 918.42 | 918.31 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 912.55 | 917.24 | 917.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 908.00 | 915.40 | 916.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 916.65 | 909.53 | 913.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 916.65 | 909.53 | 913.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 916.65 | 909.53 | 913.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 916.65 | 909.53 | 913.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 918.00 | 911.23 | 913.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 916.15 | 911.23 | 913.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 913.00 | 911.44 | 913.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 909.45 | 911.04 | 912.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 907.75 | 907.51 | 909.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 919.50 | 911.21 | 911.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 919.50 | 911.21 | 911.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 933.00 | 919.03 | 916.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 924.70 | 926.11 | 922.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 924.70 | 926.11 | 922.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 929.40 | 926.77 | 922.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:30:00 | 930.00 | 927.75 | 923.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 930.35 | 928.79 | 924.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:30:00 | 930.00 | 929.01 | 925.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 937.50 | 928.24 | 925.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 950.60 | 957.58 | 952.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 948.55 | 957.58 | 952.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 953.25 | 956.71 | 952.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 941.00 | 949.72 | 950.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 941.00 | 949.72 | 950.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 10:15:00 | 936.05 | 945.37 | 947.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 880.05 | 874.78 | 888.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 883.70 | 874.78 | 888.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 888.65 | 879.70 | 887.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 888.65 | 879.70 | 887.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 890.25 | 881.81 | 888.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 888.90 | 881.81 | 888.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 893.70 | 884.19 | 888.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 893.70 | 884.19 | 888.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 895.00 | 886.35 | 889.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 877.30 | 886.35 | 889.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 882.80 | 880.00 | 883.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 881.65 | 880.00 | 883.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 879.65 | 879.93 | 882.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 879.75 | 879.93 | 882.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 881.95 | 878.86 | 881.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 881.95 | 878.86 | 881.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 880.50 | 879.19 | 881.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 887.40 | 879.19 | 881.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 881.60 | 879.67 | 881.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 878.90 | 879.26 | 880.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 873.50 | 874.43 | 876.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 886.05 | 876.95 | 876.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 886.05 | 876.95 | 876.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 889.60 | 879.48 | 877.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 885.00 | 889.06 | 884.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 885.00 | 889.06 | 884.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 885.00 | 889.06 | 884.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 886.00 | 889.06 | 884.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 881.85 | 887.62 | 884.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 881.85 | 887.62 | 884.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 882.00 | 886.49 | 883.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:00:00 | 885.35 | 885.87 | 884.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 895.55 | 885.15 | 884.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 920.50 | 929.00 | 929.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 920.50 | 929.00 | 929.81 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 947.75 | 930.56 | 929.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 958.20 | 936.08 | 932.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 950.75 | 952.93 | 946.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 15:00:00 | 950.75 | 952.93 | 946.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 946.90 | 951.33 | 946.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 948.95 | 951.33 | 946.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 947.85 | 950.64 | 946.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:15:00 | 952.10 | 950.47 | 946.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 944.30 | 953.30 | 951.61 | SL hit (close<static) qty=1.00 sl=945.40 alert=retest2 |

### Cycle 29 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 938.75 | 948.59 | 949.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 930.70 | 943.38 | 947.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 925.45 | 919.31 | 927.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 925.45 | 919.31 | 927.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 925.45 | 919.31 | 927.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 925.45 | 919.31 | 927.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 913.45 | 918.14 | 926.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 912.40 | 916.43 | 922.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 912.55 | 915.66 | 921.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 15:15:00 | 911.00 | 904.38 | 903.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 911.00 | 904.38 | 903.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 917.95 | 907.10 | 905.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 935.00 | 936.82 | 928.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 933.20 | 936.82 | 928.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 932.35 | 935.92 | 928.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 928.00 | 935.92 | 928.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 925.00 | 933.74 | 928.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 925.00 | 933.74 | 928.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 930.55 | 933.10 | 928.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 932.10 | 930.07 | 928.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-21 13:15:00 | 1025.31 | 1018.37 | 1013.09 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1008.55 | 1046.89 | 1050.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 15:15:00 | 1005.10 | 1023.52 | 1037.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 10:15:00 | 1027.35 | 1023.20 | 1034.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:45:00 | 1027.90 | 1023.20 | 1034.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 996.05 | 983.44 | 997.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 997.45 | 983.44 | 997.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1004.20 | 987.59 | 998.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 1004.20 | 987.59 | 998.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1002.50 | 990.57 | 998.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 1001.00 | 1000.54 | 1001.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 1005.55 | 987.72 | 989.83 | SL hit (close>static) qty=1.00 sl=1004.65 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 995.85 | 991.79 | 991.46 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 989.00 | 994.54 | 994.86 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 996.10 | 994.25 | 994.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 1013.95 | 998.46 | 996.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 996.00 | 1003.91 | 1000.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 996.00 | 1003.91 | 1000.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 996.00 | 1003.91 | 1000.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 996.00 | 1003.91 | 1000.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 991.60 | 1001.45 | 999.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 991.60 | 1001.45 | 999.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 999.70 | 999.51 | 998.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 998.55 | 999.51 | 998.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 998.15 | 999.24 | 998.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:30:00 | 996.75 | 999.24 | 998.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 993.85 | 998.16 | 998.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 988.30 | 996.19 | 997.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 991.25 | 990.87 | 993.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 14:00:00 | 991.25 | 990.87 | 993.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 947.45 | 938.92 | 945.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 947.45 | 938.92 | 945.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 945.05 | 940.14 | 945.32 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 962.85 | 948.52 | 948.13 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 926.80 | 946.96 | 948.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 919.40 | 938.25 | 944.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 939.00 | 930.64 | 936.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 11:15:00 | 939.00 | 930.64 | 936.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 939.00 | 930.64 | 936.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:45:00 | 941.40 | 930.64 | 936.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 944.55 | 933.42 | 937.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:30:00 | 941.00 | 933.42 | 937.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 959.40 | 938.61 | 939.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 959.40 | 938.61 | 939.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 960.95 | 943.08 | 941.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 978.40 | 952.39 | 946.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 14:15:00 | 959.05 | 964.24 | 955.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 959.05 | 964.24 | 955.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 955.05 | 962.40 | 955.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 949.50 | 962.40 | 955.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 949.35 | 959.79 | 954.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 949.60 | 959.79 | 954.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 953.85 | 958.60 | 954.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 958.60 | 956.00 | 954.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 936.05 | 951.73 | 952.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 936.05 | 951.73 | 952.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 933.70 | 944.28 | 948.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 849.50 | 845.87 | 864.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 849.50 | 845.87 | 864.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 865.35 | 842.23 | 846.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 865.35 | 842.23 | 846.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 865.15 | 846.81 | 847.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 869.50 | 846.81 | 847.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 868.00 | 851.05 | 849.68 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 844.40 | 851.11 | 851.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 836.30 | 845.11 | 848.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 838.65 | 837.37 | 841.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 838.65 | 837.37 | 841.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 839.00 | 837.76 | 841.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:45:00 | 839.10 | 837.76 | 841.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 855.50 | 839.89 | 840.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 855.50 | 839.89 | 840.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 859.20 | 843.75 | 842.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 12:15:00 | 865.55 | 850.97 | 846.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 855.20 | 857.75 | 851.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 10:00:00 | 855.20 | 857.75 | 851.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 851.05 | 856.41 | 851.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 851.05 | 856.41 | 851.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 852.70 | 855.67 | 851.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:45:00 | 849.35 | 855.67 | 851.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 856.20 | 855.77 | 852.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:15:00 | 854.75 | 855.77 | 852.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 854.80 | 855.58 | 852.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:30:00 | 858.15 | 856.61 | 853.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 861.80 | 856.59 | 853.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 861.30 | 878.04 | 879.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 861.30 | 878.04 | 879.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 857.50 | 863.62 | 869.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 854.35 | 848.88 | 855.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 854.35 | 848.88 | 855.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 854.35 | 848.88 | 855.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 854.35 | 848.88 | 855.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 858.65 | 851.78 | 856.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 858.65 | 851.78 | 856.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 857.50 | 852.92 | 856.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 860.00 | 852.92 | 856.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 856.00 | 853.96 | 856.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 858.80 | 853.96 | 856.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 847.70 | 852.71 | 855.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:30:00 | 847.45 | 851.52 | 854.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 863.30 | 850.62 | 851.65 | SL hit (close>static) qty=1.00 sl=862.45 alert=retest2 |

### Cycle 44 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 867.30 | 853.96 | 853.08 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 850.05 | 857.78 | 858.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 840.00 | 854.22 | 857.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 849.00 | 848.73 | 853.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:45:00 | 847.00 | 848.73 | 853.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 847.80 | 848.43 | 852.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 840.35 | 846.23 | 850.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 798.33 | 816.83 | 829.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 817.30 | 802.56 | 811.93 | SL hit (close>ema200) qty=0.50 sl=802.56 alert=retest2 |

### Cycle 46 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 829.10 | 817.72 | 816.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 837.80 | 821.73 | 818.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 856.90 | 859.00 | 847.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 856.90 | 859.00 | 847.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 848.00 | 855.52 | 848.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 843.00 | 855.52 | 848.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 843.00 | 853.02 | 847.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 843.00 | 853.02 | 847.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 837.05 | 849.82 | 846.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:00:00 | 837.05 | 849.82 | 846.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 826.20 | 841.75 | 843.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 820.60 | 831.26 | 837.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 796.75 | 787.23 | 801.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:45:00 | 797.05 | 787.23 | 801.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 782.20 | 786.22 | 799.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 794.45 | 786.22 | 799.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 784.90 | 748.10 | 757.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 784.90 | 748.10 | 757.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 802.70 | 759.02 | 761.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 802.70 | 759.02 | 761.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 800.60 | 767.33 | 765.21 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 747.75 | 767.08 | 768.32 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 781.45 | 761.21 | 760.82 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 725.00 | 754.23 | 757.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 722.85 | 747.95 | 754.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 736.50 | 735.21 | 743.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 736.10 | 735.21 | 743.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 745.10 | 737.19 | 743.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 745.10 | 737.19 | 743.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 748.70 | 739.49 | 744.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 775.00 | 739.49 | 744.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 773.55 | 746.30 | 747.06 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 790.30 | 755.10 | 750.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 794.35 | 762.95 | 754.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 780.80 | 785.10 | 777.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 780.80 | 785.10 | 777.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 780.80 | 785.10 | 777.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 780.80 | 785.10 | 777.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 773.80 | 782.84 | 776.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 773.80 | 782.84 | 776.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 768.75 | 780.02 | 776.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 768.75 | 780.02 | 776.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 772.00 | 773.79 | 773.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 758.00 | 770.63 | 772.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 772.95 | 770.66 | 772.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 772.95 | 770.66 | 772.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 772.95 | 770.66 | 772.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:45:00 | 773.15 | 770.66 | 772.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 773.00 | 771.12 | 772.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 773.85 | 771.12 | 772.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 775.65 | 772.03 | 772.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 775.65 | 772.03 | 772.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 779.20 | 773.46 | 773.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 800.80 | 779.98 | 776.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 811.35 | 811.39 | 800.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:30:00 | 809.00 | 811.39 | 800.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 798.90 | 808.42 | 804.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 800.65 | 808.42 | 804.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 803.10 | 807.36 | 804.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 805.70 | 804.28 | 803.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 15:00:00 | 804.20 | 804.22 | 803.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 786.75 | 800.85 | 802.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 786.75 | 800.85 | 802.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 12:15:00 | 781.45 | 787.42 | 792.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 786.80 | 785.72 | 790.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 787.70 | 785.85 | 789.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 781.65 | 782.95 | 786.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 775.65 | 781.46 | 783.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 771.50 | 771.68 | 776.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 736.87 | 751.76 | 760.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:15:00 | 732.92 | 741.88 | 751.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 12:15:00 | 733.00 | 732.64 | 741.26 | SL hit (close>ema200) qty=0.50 sl=732.64 alert=retest2 |

### Cycle 56 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 727.25 | 715.50 | 714.59 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 690.85 | 713.27 | 714.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 12:15:00 | 687.65 | 701.16 | 708.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 706.90 | 699.11 | 704.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 706.90 | 699.11 | 704.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 706.90 | 699.11 | 704.69 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 717.85 | 707.38 | 707.25 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 702.95 | 708.10 | 708.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 699.30 | 705.55 | 706.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 711.10 | 703.25 | 705.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 711.10 | 703.25 | 705.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 711.10 | 703.25 | 705.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 711.10 | 703.25 | 705.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 706.70 | 703.94 | 705.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 13:45:00 | 701.45 | 703.63 | 705.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 666.38 | 683.04 | 691.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 679.60 | 674.92 | 681.95 | SL hit (close>ema200) qty=0.50 sl=674.92 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 694.00 | 683.92 | 683.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 702.35 | 687.61 | 685.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 681.20 | 691.81 | 688.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 681.20 | 691.81 | 688.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 681.20 | 691.81 | 688.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 678.95 | 691.81 | 688.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 678.80 | 689.21 | 687.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 682.50 | 689.21 | 687.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 676.40 | 685.13 | 686.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 674.50 | 683.01 | 685.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 680.30 | 679.21 | 682.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 680.30 | 679.21 | 682.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 680.30 | 679.21 | 682.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:00:00 | 672.95 | 678.12 | 681.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 639.30 | 658.94 | 669.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 639.75 | 638.87 | 651.71 | SL hit (close>ema200) qty=0.50 sl=638.87 alert=retest2 |

### Cycle 62 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 682.30 | 656.80 | 654.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 689.05 | 663.25 | 658.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 672.15 | 675.22 | 667.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 672.15 | 675.22 | 667.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 672.15 | 675.22 | 667.08 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 644.25 | 662.48 | 664.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 642.60 | 656.16 | 661.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 676.70 | 652.65 | 657.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 676.70 | 652.65 | 657.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 676.70 | 652.65 | 657.26 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 672.90 | 660.21 | 660.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 680.05 | 668.73 | 664.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 658.05 | 668.34 | 665.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 658.05 | 668.34 | 665.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 658.05 | 668.34 | 665.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 658.05 | 668.34 | 665.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 667.95 | 668.26 | 665.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 669.70 | 668.26 | 665.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 736.67 | 706.44 | 695.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 798.70 | 810.81 | 811.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 789.00 | 806.45 | 809.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 790.65 | 785.65 | 792.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 790.65 | 785.65 | 792.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 790.65 | 785.65 | 792.51 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 799.00 | 794.26 | 794.06 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 783.00 | 793.28 | 793.89 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 798.80 | 791.46 | 791.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 11:15:00 | 806.40 | 794.45 | 792.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 798.50 | 800.31 | 796.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 798.50 | 800.31 | 796.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 798.50 | 800.31 | 796.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 798.50 | 800.31 | 796.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 807.00 | 801.65 | 797.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 823.35 | 801.65 | 797.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-30 11:15:00 | 812.00 | 2025-06-03 15:15:00 | 802.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-05-30 14:15:00 | 810.00 | 2025-06-03 15:15:00 | 802.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-06-03 11:45:00 | 810.15 | 2025-06-03 15:15:00 | 802.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-10 15:00:00 | 879.85 | 2025-06-11 09:15:00 | 868.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-18 15:15:00 | 815.65 | 2025-06-20 09:15:00 | 830.80 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-06-19 09:30:00 | 816.80 | 2025-06-20 09:15:00 | 830.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-06-19 10:15:00 | 815.90 | 2025-06-20 09:15:00 | 830.80 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-06-19 11:00:00 | 815.95 | 2025-06-20 09:15:00 | 830.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-06-27 09:15:00 | 868.05 | 2025-06-27 14:15:00 | 855.30 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-06-30 09:15:00 | 868.90 | 2025-07-02 09:15:00 | 854.85 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-09 09:15:00 | 927.50 | 2025-07-14 11:15:00 | 922.45 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-09 12:00:00 | 923.00 | 2025-07-14 12:15:00 | 922.25 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-07-09 13:30:00 | 923.50 | 2025-07-14 12:15:00 | 922.25 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-07-10 10:00:00 | 922.60 | 2025-07-14 12:15:00 | 922.25 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-07-11 14:45:00 | 938.00 | 2025-07-14 12:15:00 | 922.25 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-16 14:30:00 | 939.20 | 2025-07-17 12:15:00 | 932.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-17 09:15:00 | 939.00 | 2025-07-17 12:15:00 | 932.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-08 14:45:00 | 909.45 | 2025-08-11 14:15:00 | 919.50 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-08-11 13:15:00 | 907.75 | 2025-08-11 14:15:00 | 919.50 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-08-14 11:30:00 | 930.00 | 2025-08-21 13:15:00 | 941.00 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-08-14 13:30:00 | 930.35 | 2025-08-21 13:15:00 | 941.00 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-08-14 14:30:00 | 930.00 | 2025-08-21 13:15:00 | 941.00 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-08-18 09:15:00 | 937.50 | 2025-08-21 13:15:00 | 941.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-09-04 10:30:00 | 878.90 | 2025-09-08 10:15:00 | 886.05 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-05 10:45:00 | 873.50 | 2025-09-08 10:15:00 | 886.05 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-09 14:00:00 | 885.35 | 2025-09-18 12:15:00 | 920.50 | STOP_HIT | 1.00 | 3.97% |
| BUY | retest2 | 2025-09-10 09:15:00 | 895.55 | 2025-09-18 12:15:00 | 920.50 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest2 | 2025-09-23 12:15:00 | 952.10 | 2025-09-24 14:15:00 | 944.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-30 09:15:00 | 912.40 | 2025-10-03 15:15:00 | 911.00 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-09-30 10:00:00 | 912.55 | 2025-10-03 15:15:00 | 911.00 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-10-09 09:30:00 | 932.10 | 2025-10-21 13:15:00 | 1025.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-04 11:15:00 | 1001.00 | 2025-11-07 12:15:00 | 1005.55 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-02 13:45:00 | 958.60 | 2025-12-03 09:15:00 | 936.05 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-12-19 14:30:00 | 858.15 | 2025-12-26 14:15:00 | 861.30 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-12-22 09:15:00 | 861.80 | 2025-12-26 14:15:00 | 861.30 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2026-01-01 10:30:00 | 847.45 | 2026-01-02 10:15:00 | 863.30 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-01-08 10:45:00 | 840.35 | 2026-01-09 14:15:00 | 798.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 840.35 | 2026-01-13 09:15:00 | 817.30 | STOP_HIT | 0.50 | 2.74% |
| BUY | retest2 | 2026-02-12 13:15:00 | 805.70 | 2026-02-13 09:15:00 | 786.75 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-02-12 15:00:00 | 804.20 | 2026-02-13 09:15:00 | 786.75 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-02-19 09:15:00 | 775.65 | 2026-02-24 14:15:00 | 736.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:00:00 | 771.50 | 2026-02-25 12:15:00 | 732.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 775.65 | 2026-02-26 12:15:00 | 733.00 | STOP_HIT | 0.50 | 5.50% |
| SELL | retest2 | 2026-02-20 11:00:00 | 771.50 | 2026-02-26 12:15:00 | 733.00 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2026-03-12 13:45:00 | 701.45 | 2026-03-16 09:15:00 | 666.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 13:45:00 | 701.45 | 2026-03-17 09:15:00 | 679.60 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2026-03-20 12:00:00 | 672.95 | 2026-03-23 10:15:00 | 639.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:00:00 | 672.95 | 2026-03-24 10:15:00 | 639.75 | STOP_HIT | 0.50 | 4.93% |
| BUY | retest2 | 2026-04-02 11:15:00 | 669.70 | 2026-04-08 09:15:00 | 736.67 | TARGET_HIT | 1.00 | 10.00% |
