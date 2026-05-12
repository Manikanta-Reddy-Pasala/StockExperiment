# Indian Railway Catering And Tourism Corporation Ltd. (IRCTC)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 565.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 137 |
| ALERT1 | 106 |
| ALERT2 | 106 |
| ALERT2_SKIP | 49 |
| ALERT3 | 275 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 112 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 115 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 124 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 85
- **Target hits / Stop hits / Partials:** 2 / 114 / 8
- **Avg / median % per leg:** 0.13% / -0.63%
- **Sum % (uncompounded):** 15.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 11 | 21.6% | 2 | 49 | 0 | -0.15% | -7.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 51 | 11 | 21.6% | 2 | 49 | 0 | -0.15% | -7.7% |
| SELL (all) | 73 | 28 | 38.4% | 0 | 65 | 8 | 0.32% | 23.5% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.54% | -6.2% |
| SELL @ 3rd Alert (retest2) | 69 | 28 | 40.6% | 0 | 61 | 8 | 0.43% | 29.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.54% | -6.2% |
| retest2 (combined) | 120 | 39 | 32.5% | 2 | 110 | 8 | 0.18% | 21.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1019.10 | 998.10 | 996.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 1022.25 | 1002.93 | 999.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 1029.45 | 1032.03 | 1024.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:00:00 | 1029.45 | 1032.03 | 1024.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1109.65 | 1120.29 | 1113.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1112.75 | 1120.29 | 1113.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1115.75 | 1119.38 | 1113.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:45:00 | 1113.50 | 1119.38 | 1113.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 1114.05 | 1117.97 | 1113.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:45:00 | 1116.00 | 1117.97 | 1113.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 1115.95 | 1117.56 | 1113.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:15:00 | 1112.75 | 1117.56 | 1113.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1110.00 | 1116.05 | 1113.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1110.00 | 1116.05 | 1113.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1107.90 | 1114.42 | 1113.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 1116.00 | 1114.42 | 1113.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:00:00 | 1113.65 | 1114.27 | 1113.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:45:00 | 1110.50 | 1112.96 | 1112.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 11:15:00 | 1106.10 | 1111.59 | 1112.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 11:15:00 | 1106.10 | 1111.59 | 1112.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 12:15:00 | 1100.65 | 1109.40 | 1110.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 1019.10 | 1017.83 | 1033.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 15:00:00 | 1019.10 | 1017.83 | 1033.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1044.40 | 1023.97 | 1034.01 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 1057.70 | 1039.14 | 1038.72 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 951.00 | 1021.71 | 1031.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 898.15 | 996.99 | 1019.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 976.85 | 931.05 | 952.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 976.85 | 931.05 | 952.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 976.85 | 931.05 | 952.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 976.85 | 931.05 | 952.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 975.40 | 939.92 | 954.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:00:00 | 968.15 | 945.56 | 955.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:00:00 | 970.25 | 950.50 | 956.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:15:00 | 970.00 | 955.17 | 958.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 15:15:00 | 975.00 | 962.00 | 961.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 975.00 | 962.00 | 961.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 978.50 | 970.83 | 966.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 978.55 | 979.11 | 974.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 14:15:00 | 978.55 | 979.11 | 974.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 978.55 | 979.11 | 974.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 978.45 | 979.11 | 974.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 1017.55 | 1020.97 | 1015.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 1023.90 | 1020.96 | 1016.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 12:45:00 | 1020.45 | 1020.58 | 1017.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 1024.05 | 1019.21 | 1017.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:45:00 | 1020.95 | 1026.83 | 1023.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1020.50 | 1025.56 | 1023.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 1015.20 | 1025.56 | 1023.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 1017.00 | 1023.57 | 1022.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:00:00 | 1017.00 | 1023.57 | 1022.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 1017.80 | 1022.41 | 1022.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 1017.80 | 1022.41 | 1022.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 1014.80 | 1020.89 | 1021.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 1020.15 | 1019.79 | 1021.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1020.15 | 1019.79 | 1021.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1020.15 | 1019.79 | 1021.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:30:00 | 1024.70 | 1019.79 | 1021.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1022.50 | 1020.33 | 1021.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 1022.50 | 1020.33 | 1021.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1024.15 | 1021.10 | 1021.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 1024.15 | 1021.10 | 1021.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 1017.50 | 1020.31 | 1021.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 14:15:00 | 1013.75 | 1020.31 | 1021.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1037.70 | 1022.72 | 1021.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1037.70 | 1022.72 | 1021.84 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 1010.30 | 1021.86 | 1022.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 15:15:00 | 1010.00 | 1014.50 | 1017.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 998.75 | 996.88 | 1003.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 12:00:00 | 998.75 | 996.88 | 1003.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 999.90 | 994.80 | 999.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 999.90 | 994.80 | 999.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 997.60 | 995.36 | 999.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:30:00 | 992.55 | 994.18 | 998.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1002.75 | 995.14 | 997.46 | SL hit (close>static) qty=1.00 sl=1000.45 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 1008.25 | 996.41 | 995.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 1011.15 | 1006.46 | 1003.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 1006.00 | 1006.62 | 1004.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 15:00:00 | 1006.00 | 1006.62 | 1004.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1019.35 | 1009.05 | 1005.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:30:00 | 1021.35 | 1011.10 | 1006.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:15:00 | 1021.00 | 1011.10 | 1006.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:45:00 | 1023.65 | 1014.71 | 1009.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 1012.50 | 1026.52 | 1027.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 1012.50 | 1026.52 | 1027.17 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 1035.95 | 1026.84 | 1026.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 13:15:00 | 1053.65 | 1035.67 | 1031.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 1038.00 | 1039.14 | 1036.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 15:00:00 | 1038.00 | 1039.14 | 1036.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1037.35 | 1038.53 | 1036.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 1037.35 | 1038.53 | 1036.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1037.90 | 1038.40 | 1036.47 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 1027.60 | 1034.89 | 1035.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 1019.15 | 1030.12 | 1032.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 12:15:00 | 997.50 | 995.54 | 1003.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 13:00:00 | 997.50 | 995.54 | 1003.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1003.45 | 997.12 | 1003.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 1007.00 | 997.12 | 1003.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 1005.80 | 998.86 | 1003.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 1006.90 | 998.86 | 1003.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 1009.00 | 1000.89 | 1004.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 1011.15 | 1000.89 | 1004.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 1013.00 | 1004.30 | 1005.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 1013.00 | 1004.30 | 1005.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 977.80 | 971.51 | 976.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:30:00 | 975.95 | 971.51 | 976.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 977.00 | 972.61 | 976.71 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 985.00 | 979.17 | 978.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 09:15:00 | 995.00 | 982.77 | 980.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 991.45 | 991.46 | 987.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:00:00 | 991.45 | 991.46 | 987.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 986.20 | 989.94 | 989.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 986.20 | 989.94 | 989.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 987.85 | 989.52 | 988.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:30:00 | 986.50 | 989.52 | 988.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 989.90 | 989.60 | 988.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 990.00 | 989.60 | 988.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:45:00 | 992.90 | 990.11 | 989.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 981.65 | 987.84 | 988.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 11:15:00 | 981.65 | 987.84 | 988.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 979.40 | 985.17 | 986.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 942.20 | 939.76 | 954.41 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:30:00 | 925.30 | 933.20 | 945.58 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 15:00:00 | 920.45 | 933.20 | 945.58 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 09:45:00 | 923.50 | 929.33 | 941.57 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 12:45:00 | 924.80 | 928.48 | 938.11 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 934.40 | 931.53 | 936.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:45:00 | 934.90 | 931.53 | 936.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 937.75 | 932.78 | 936.18 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 937.75 | 932.78 | 936.18 | SL hit (close>ema400) qty=1.00 sl=936.18 alert=retest1 |

### Cycle 15 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 923.00 | 919.45 | 919.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 939.70 | 923.50 | 920.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 929.40 | 933.02 | 928.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 10:15:00 | 929.40 | 933.02 | 928.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 929.40 | 933.02 | 928.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 928.85 | 933.02 | 928.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 929.90 | 932.40 | 928.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 12:45:00 | 930.95 | 932.06 | 929.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:00:00 | 930.95 | 931.59 | 929.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 12:15:00 | 929.40 | 933.85 | 934.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 929.40 | 933.85 | 934.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 927.25 | 932.53 | 933.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 14:15:00 | 931.25 | 929.03 | 930.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 14:15:00 | 931.25 | 929.03 | 930.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 931.25 | 929.03 | 930.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 931.25 | 929.03 | 930.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 931.00 | 929.42 | 930.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 930.80 | 929.42 | 930.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 928.55 | 929.25 | 930.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 10:15:00 | 925.80 | 929.25 | 930.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 934.05 | 930.45 | 930.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 934.05 | 930.45 | 930.31 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 926.55 | 930.18 | 930.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 926.00 | 929.35 | 929.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 924.50 | 922.28 | 925.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 924.50 | 922.28 | 925.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 925.75 | 923.09 | 925.28 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 941.50 | 928.53 | 927.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 948.35 | 940.84 | 938.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 933.30 | 941.91 | 940.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 933.30 | 941.91 | 940.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 933.30 | 941.91 | 940.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 933.30 | 941.91 | 940.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 930.45 | 939.62 | 939.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 924.65 | 935.29 | 937.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 12:15:00 | 932.80 | 927.72 | 930.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 12:15:00 | 932.80 | 927.72 | 930.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 932.80 | 927.72 | 930.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:30:00 | 932.55 | 927.72 | 930.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 931.55 | 928.49 | 930.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 14:15:00 | 932.20 | 928.49 | 930.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 930.50 | 928.89 | 930.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 927.50 | 929.12 | 930.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:45:00 | 925.50 | 928.63 | 929.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:30:00 | 926.95 | 925.98 | 927.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 945.85 | 931.27 | 929.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 945.85 | 931.27 | 929.58 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 929.65 | 932.60 | 932.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 09:15:00 | 928.90 | 932.02 | 932.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 889.25 | 888.02 | 900.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:00:00 | 889.25 | 888.02 | 900.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 898.15 | 893.61 | 897.97 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 906.35 | 900.68 | 900.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 909.15 | 902.37 | 901.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 901.80 | 908.85 | 906.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 901.80 | 908.85 | 906.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 901.80 | 908.85 | 906.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 901.80 | 908.85 | 906.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 905.00 | 908.08 | 906.47 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 899.05 | 905.15 | 905.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 897.00 | 902.14 | 903.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 906.30 | 902.55 | 903.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 906.30 | 902.55 | 903.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 906.30 | 902.55 | 903.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 906.30 | 902.55 | 903.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 905.00 | 903.04 | 903.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 909.25 | 903.04 | 903.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 909.95 | 904.42 | 904.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 916.65 | 906.87 | 905.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 913.75 | 926.26 | 923.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 913.75 | 926.26 | 923.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 913.75 | 926.26 | 923.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:15:00 | 911.95 | 926.26 | 923.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 903.40 | 921.69 | 921.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 894.45 | 916.24 | 919.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 861.50 | 861.25 | 874.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 864.40 | 861.25 | 874.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 866.10 | 864.00 | 873.47 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 883.50 | 875.59 | 875.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 888.90 | 880.86 | 878.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 884.00 | 884.42 | 881.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 15:00:00 | 884.00 | 884.42 | 881.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 883.00 | 884.13 | 881.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 882.15 | 884.13 | 881.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 893.35 | 885.98 | 882.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 893.90 | 887.22 | 885.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 11:45:00 | 894.20 | 889.53 | 887.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 12:45:00 | 894.05 | 890.14 | 887.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 13:30:00 | 894.10 | 891.41 | 888.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 894.45 | 893.45 | 890.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 895.10 | 893.45 | 890.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 894.60 | 893.68 | 890.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 894.45 | 893.68 | 890.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 891.00 | 892.96 | 891.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:30:00 | 889.00 | 892.96 | 891.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 892.65 | 892.90 | 891.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 14:45:00 | 889.50 | 892.90 | 891.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 894.00 | 893.12 | 891.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 892.50 | 893.12 | 891.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 884.90 | 891.48 | 890.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 884.90 | 891.48 | 890.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 885.95 | 890.37 | 890.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 885.95 | 890.37 | 890.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 880.85 | 887.63 | 889.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 879.55 | 877.41 | 881.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:45:00 | 878.45 | 877.41 | 881.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 881.00 | 878.13 | 881.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 881.00 | 878.13 | 881.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 880.00 | 878.51 | 881.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 877.00 | 878.51 | 881.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 874.00 | 877.60 | 880.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:45:00 | 867.10 | 874.22 | 878.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:30:00 | 867.35 | 872.87 | 877.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 867.45 | 872.87 | 877.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:30:00 | 867.60 | 871.68 | 876.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 823.75 | 842.14 | 855.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 823.98 | 842.14 | 855.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 824.08 | 842.14 | 855.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 824.22 | 842.14 | 855.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 09:15:00 | 839.30 | 835.84 | 844.96 | SL hit (close>ema200) qty=0.50 sl=835.84 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 839.10 | 824.75 | 823.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 842.30 | 828.26 | 825.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 835.30 | 837.61 | 832.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 835.30 | 837.61 | 832.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 835.30 | 837.61 | 832.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 836.85 | 837.61 | 832.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 831.00 | 836.29 | 832.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:45:00 | 830.95 | 836.29 | 832.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 823.05 | 833.64 | 831.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 823.05 | 833.64 | 831.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 818.80 | 830.67 | 830.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 818.75 | 830.67 | 830.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 820.50 | 828.64 | 829.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 809.35 | 824.83 | 827.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 824.10 | 818.16 | 821.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 824.10 | 818.16 | 821.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 824.10 | 818.16 | 821.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 824.10 | 818.16 | 821.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 816.70 | 817.87 | 821.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 12:15:00 | 814.35 | 818.67 | 821.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 14:15:00 | 828.85 | 821.99 | 822.12 | SL hit (close>static) qty=1.00 sl=828.40 alert=retest2 |

### Cycle 31 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 831.40 | 823.87 | 822.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 853.00 | 829.70 | 825.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 849.25 | 850.27 | 840.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 849.25 | 850.27 | 840.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 844.20 | 846.94 | 843.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 839.60 | 846.94 | 843.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 842.40 | 846.03 | 842.97 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 835.15 | 840.87 | 841.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 833.35 | 839.36 | 840.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 838.45 | 838.00 | 839.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 838.45 | 838.00 | 839.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 838.45 | 838.00 | 839.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:00:00 | 838.45 | 838.00 | 839.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 842.50 | 838.90 | 839.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 842.50 | 838.90 | 839.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 838.70 | 838.86 | 839.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:15:00 | 837.80 | 838.97 | 839.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 795.91 | 807.52 | 816.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 802.35 | 799.79 | 805.57 | SL hit (close>ema200) qty=0.50 sl=799.79 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 812.20 | 806.91 | 806.54 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 799.00 | 805.70 | 806.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 797.90 | 804.14 | 805.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 12:15:00 | 798.10 | 797.99 | 801.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-21 13:00:00 | 798.10 | 797.99 | 801.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 800.05 | 796.62 | 799.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:30:00 | 797.45 | 796.62 | 799.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 803.90 | 798.08 | 799.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 803.90 | 798.08 | 799.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 806.35 | 799.73 | 800.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:45:00 | 806.30 | 799.73 | 800.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 808.05 | 802.46 | 801.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 821.95 | 808.00 | 804.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 812.00 | 814.19 | 809.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 812.00 | 814.19 | 809.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 812.90 | 816.74 | 813.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 815.75 | 816.74 | 813.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 827.50 | 818.89 | 815.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 10:30:00 | 829.80 | 820.61 | 816.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 15:15:00 | 815.00 | 817.85 | 818.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 15:15:00 | 815.00 | 817.85 | 818.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 812.00 | 816.68 | 817.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 816.90 | 815.03 | 816.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 816.90 | 815.03 | 816.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 816.90 | 815.03 | 816.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 816.90 | 815.03 | 816.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 815.35 | 815.10 | 816.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:45:00 | 817.55 | 815.10 | 816.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 816.00 | 815.28 | 816.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 811.90 | 815.28 | 816.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 10:30:00 | 813.60 | 814.73 | 815.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 13:15:00 | 818.00 | 815.18 | 815.74 | SL hit (close>static) qty=1.00 sl=816.60 alert=retest2 |

### Cycle 37 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 822.95 | 817.18 | 816.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 14:15:00 | 830.70 | 824.47 | 820.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 11:15:00 | 827.35 | 828.27 | 824.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:45:00 | 827.40 | 828.27 | 824.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 833.95 | 835.49 | 833.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:30:00 | 833.70 | 835.49 | 833.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 830.10 | 834.41 | 832.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 830.10 | 834.41 | 832.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 830.00 | 833.53 | 832.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 836.75 | 833.53 | 832.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 828.40 | 840.04 | 841.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 828.40 | 840.04 | 841.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 10:15:00 | 826.05 | 837.24 | 839.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 836.35 | 835.58 | 837.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 836.35 | 835.58 | 837.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 837.10 | 835.79 | 837.64 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 843.15 | 838.39 | 838.23 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 834.60 | 838.00 | 838.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 832.05 | 836.81 | 837.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 13:15:00 | 789.55 | 789.02 | 797.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 13:45:00 | 790.30 | 789.02 | 797.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 782.80 | 787.97 | 791.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 781.55 | 787.97 | 791.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:45:00 | 781.80 | 787.16 | 791.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:30:00 | 781.55 | 785.52 | 789.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:45:00 | 781.15 | 784.14 | 788.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 778.85 | 774.61 | 778.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 778.85 | 774.61 | 778.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 778.65 | 775.42 | 778.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 781.35 | 775.42 | 778.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 782.30 | 776.80 | 778.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:00:00 | 782.30 | 776.80 | 778.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 788.95 | 779.23 | 779.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 788.95 | 779.23 | 779.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-31 14:15:00 | 787.65 | 780.91 | 780.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 787.65 | 780.91 | 780.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 790.70 | 784.89 | 782.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 786.75 | 787.52 | 785.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 786.75 | 787.52 | 785.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 786.75 | 787.52 | 785.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:00:00 | 790.75 | 787.92 | 785.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:30:00 | 791.15 | 795.32 | 792.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 778.50 | 791.96 | 791.29 | SL hit (close<static) qty=1.00 sl=785.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 780.30 | 789.63 | 790.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 772.20 | 786.14 | 788.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 779.30 | 778.85 | 783.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 783.00 | 778.85 | 783.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 762.50 | 765.81 | 769.63 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 11:15:00 | 787.20 | 774.36 | 773.13 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 764.65 | 773.96 | 774.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 11:15:00 | 758.00 | 770.77 | 772.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 763.85 | 758.67 | 764.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 763.85 | 758.67 | 764.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 763.85 | 758.67 | 764.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 763.85 | 758.67 | 764.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 757.00 | 758.34 | 763.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 754.80 | 758.34 | 763.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:15:00 | 756.30 | 760.03 | 761.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 766.45 | 761.43 | 761.82 | SL hit (close>static) qty=1.00 sl=764.40 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 767.40 | 762.62 | 762.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 11:15:00 | 774.60 | 767.27 | 765.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 784.70 | 786.24 | 779.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:30:00 | 784.50 | 786.24 | 779.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 783.80 | 786.89 | 782.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 783.50 | 786.89 | 782.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 782.45 | 786.00 | 782.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 769.80 | 786.00 | 782.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 769.35 | 782.67 | 781.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 769.80 | 782.67 | 781.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 768.30 | 779.80 | 779.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 763.90 | 776.62 | 778.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 784.35 | 774.00 | 775.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 784.35 | 774.00 | 775.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 784.35 | 774.00 | 775.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 784.35 | 774.00 | 775.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 790.20 | 777.24 | 777.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 15:15:00 | 795.25 | 787.72 | 782.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 789.60 | 791.57 | 787.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 14:00:00 | 789.60 | 791.57 | 787.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 787.40 | 790.74 | 787.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 787.00 | 790.74 | 787.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 787.30 | 790.05 | 787.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 771.20 | 790.05 | 787.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 760.10 | 784.06 | 784.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 757.00 | 778.65 | 782.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 753.45 | 752.42 | 762.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 753.45 | 752.42 | 762.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 760.90 | 753.84 | 760.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 760.90 | 753.84 | 760.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 761.40 | 755.35 | 760.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 761.40 | 755.35 | 760.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 760.50 | 756.38 | 760.21 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 779.50 | 764.01 | 762.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 795.60 | 779.79 | 772.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 808.45 | 813.30 | 798.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 11:45:00 | 811.40 | 813.30 | 798.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 793.30 | 809.30 | 798.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 793.30 | 809.30 | 798.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 800.15 | 807.47 | 798.37 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 773.30 | 792.03 | 793.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 765.50 | 786.72 | 790.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 780.55 | 777.92 | 783.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 780.55 | 777.92 | 783.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 780.55 | 777.92 | 783.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:30:00 | 782.20 | 777.92 | 783.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 783.15 | 779.44 | 782.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:00:00 | 783.15 | 779.44 | 782.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 781.00 | 779.76 | 782.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 787.15 | 779.76 | 782.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 785.60 | 780.92 | 782.60 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 795.00 | 785.44 | 784.47 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 782.85 | 786.37 | 786.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 779.95 | 784.55 | 785.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 10:15:00 | 785.15 | 778.43 | 780.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 10:15:00 | 785.15 | 778.43 | 780.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 785.15 | 778.43 | 780.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 785.15 | 778.43 | 780.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 777.60 | 778.26 | 780.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:00:00 | 772.90 | 777.12 | 779.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 734.25 | 749.76 | 761.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 11:15:00 | 760.15 | 751.80 | 760.45 | SL hit (close>ema200) qty=0.50 sl=751.80 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 730.75 | 727.02 | 726.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 734.20 | 729.10 | 728.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 728.85 | 730.95 | 729.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 728.85 | 730.95 | 729.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 728.85 | 730.95 | 729.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 728.85 | 730.95 | 729.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 728.45 | 730.45 | 729.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 728.45 | 730.45 | 729.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 728.40 | 730.04 | 729.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 728.40 | 730.04 | 729.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 729.55 | 729.94 | 729.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 732.00 | 730.01 | 729.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 722.50 | 728.32 | 728.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 722.50 | 728.32 | 728.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 714.65 | 719.82 | 723.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 673.00 | 669.55 | 681.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 673.00 | 669.55 | 681.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 675.50 | 671.47 | 680.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 678.70 | 671.47 | 680.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 675.55 | 673.30 | 679.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 675.55 | 673.30 | 679.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 681.55 | 674.92 | 677.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 685.25 | 674.92 | 677.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 686.65 | 677.27 | 678.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 686.65 | 677.27 | 678.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 687.55 | 679.33 | 679.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 695.40 | 685.02 | 681.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 702.55 | 703.15 | 697.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:45:00 | 703.70 | 703.15 | 697.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 700.20 | 701.39 | 698.05 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 690.20 | 696.32 | 696.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 688.00 | 694.65 | 695.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 693.75 | 690.47 | 692.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 15:15:00 | 693.75 | 690.47 | 692.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 693.75 | 690.47 | 692.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 692.45 | 690.47 | 692.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 694.00 | 691.18 | 692.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:30:00 | 689.75 | 691.57 | 692.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 14:15:00 | 698.35 | 693.26 | 693.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 698.35 | 693.26 | 693.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 703.00 | 695.90 | 694.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 10:15:00 | 695.65 | 695.85 | 694.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 10:15:00 | 695.65 | 695.85 | 694.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 695.65 | 695.85 | 694.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:00:00 | 695.65 | 695.85 | 694.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 693.45 | 695.37 | 694.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:45:00 | 691.40 | 695.37 | 694.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 693.35 | 694.96 | 694.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:15:00 | 692.55 | 694.96 | 694.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 690.50 | 694.07 | 693.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 690.50 | 694.07 | 693.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 689.55 | 693.17 | 693.55 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 700.75 | 693.69 | 693.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 702.00 | 696.56 | 694.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 11:15:00 | 715.25 | 715.46 | 709.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 11:30:00 | 714.95 | 715.46 | 709.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 725.30 | 727.11 | 723.20 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 716.80 | 721.24 | 721.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 715.00 | 719.99 | 720.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 710.35 | 710.11 | 714.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:45:00 | 710.25 | 710.11 | 714.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 717.90 | 710.39 | 712.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 717.90 | 710.39 | 712.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 722.55 | 712.82 | 713.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 728.40 | 712.82 | 713.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 728.30 | 715.92 | 715.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 736.20 | 728.71 | 725.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 715.80 | 729.88 | 727.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 715.80 | 729.88 | 727.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 715.80 | 729.88 | 727.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 715.80 | 729.88 | 727.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 722.90 | 728.48 | 727.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 724.05 | 728.48 | 727.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 716.00 | 724.86 | 725.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 716.00 | 724.86 | 725.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 712.95 | 722.48 | 724.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 701.60 | 699.56 | 708.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 715.00 | 699.56 | 708.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 710.05 | 701.66 | 708.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 705.60 | 703.93 | 708.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-08 15:15:00 | 715.05 | 711.87 | 711.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 715.05 | 711.87 | 711.58 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 708.75 | 711.24 | 711.33 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 716.00 | 711.73 | 711.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 12:15:00 | 720.05 | 713.40 | 712.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 15:15:00 | 773.35 | 773.36 | 766.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 09:15:00 | 775.90 | 773.36 | 766.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 764.85 | 773.33 | 770.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 764.85 | 773.33 | 770.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 768.60 | 772.38 | 770.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 765.30 | 772.38 | 770.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 773.60 | 776.62 | 774.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:00:00 | 773.60 | 776.62 | 774.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 777.30 | 776.75 | 774.93 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 762.10 | 773.86 | 773.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 753.35 | 769.76 | 772.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 761.30 | 758.89 | 764.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 761.30 | 758.89 | 764.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 763.45 | 760.15 | 764.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:45:00 | 763.75 | 760.15 | 764.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 763.80 | 760.88 | 764.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 766.05 | 760.88 | 764.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 763.55 | 761.41 | 764.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 13:30:00 | 764.45 | 761.41 | 764.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 763.40 | 761.81 | 763.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 763.40 | 761.81 | 763.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 765.00 | 762.45 | 764.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 769.70 | 762.45 | 764.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 768.20 | 763.60 | 764.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 766.60 | 763.60 | 764.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 760.75 | 763.03 | 764.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:45:00 | 752.45 | 758.14 | 760.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 753.65 | 756.29 | 758.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 765.35 | 757.63 | 757.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 765.35 | 757.63 | 757.07 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 750.00 | 756.28 | 756.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 747.50 | 754.52 | 756.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 741.05 | 740.58 | 745.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 14:45:00 | 743.25 | 740.58 | 745.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 744.40 | 741.65 | 745.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 736.80 | 741.29 | 744.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 750.95 | 733.12 | 733.76 | SL hit (close>static) qty=1.00 sl=750.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 750.20 | 736.54 | 735.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 759.20 | 743.50 | 738.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 805.20 | 806.99 | 798.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 805.20 | 806.99 | 798.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 798.35 | 804.74 | 799.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 796.95 | 804.74 | 799.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 798.50 | 803.49 | 799.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:15:00 | 796.00 | 803.49 | 799.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 795.80 | 801.95 | 798.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 795.80 | 801.95 | 798.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 789.65 | 798.22 | 797.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 789.65 | 798.22 | 797.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 786.65 | 795.91 | 796.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 784.75 | 792.19 | 794.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 783.00 | 781.86 | 785.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 11:00:00 | 783.00 | 781.86 | 785.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 785.55 | 782.60 | 785.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 785.55 | 782.60 | 785.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 785.60 | 783.20 | 785.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 785.50 | 783.20 | 785.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 783.80 | 783.32 | 785.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:30:00 | 783.65 | 783.24 | 785.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 788.55 | 784.33 | 785.52 | SL hit (close>static) qty=1.00 sl=785.85 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 789.45 | 786.16 | 786.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 793.20 | 788.18 | 787.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 789.65 | 791.16 | 789.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 14:00:00 | 789.65 | 791.16 | 789.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 792.70 | 791.46 | 789.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 793.80 | 791.46 | 789.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 783.70 | 790.29 | 789.49 | SL hit (close<static) qty=1.00 sl=788.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 783.30 | 788.89 | 788.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 777.65 | 784.53 | 786.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 763.75 | 760.99 | 766.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:00:00 | 763.75 | 760.99 | 766.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 769.55 | 762.70 | 767.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 769.55 | 762.70 | 767.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 766.00 | 763.36 | 766.95 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 12:15:00 | 772.00 | 768.63 | 768.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 13:15:00 | 776.00 | 770.11 | 769.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 766.15 | 769.57 | 769.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 766.15 | 769.57 | 769.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 766.15 | 769.57 | 769.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 767.85 | 769.57 | 769.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 768.15 | 769.28 | 769.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:15:00 | 776.05 | 769.28 | 769.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 782.40 | 786.56 | 786.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 782.40 | 786.56 | 786.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 780.85 | 785.22 | 786.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 768.25 | 767.31 | 771.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 768.25 | 767.31 | 771.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 770.05 | 767.86 | 771.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 770.85 | 767.86 | 771.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 768.80 | 768.08 | 771.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 767.30 | 767.92 | 770.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 773.45 | 769.36 | 770.94 | SL hit (close>static) qty=1.00 sl=772.10 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 767.45 | 758.97 | 757.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 768.70 | 763.65 | 761.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 764.40 | 766.36 | 764.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 10:15:00 | 764.40 | 766.36 | 764.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 764.40 | 766.36 | 764.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 764.40 | 766.36 | 764.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 766.45 | 766.38 | 764.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:30:00 | 768.50 | 766.90 | 765.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 774.80 | 776.84 | 776.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 774.80 | 776.84 | 776.98 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 778.45 | 777.05 | 776.99 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 772.55 | 776.35 | 776.69 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 779.55 | 777.18 | 776.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 784.10 | 780.21 | 779.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 780.50 | 780.88 | 779.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 12:00:00 | 780.50 | 780.88 | 779.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 782.45 | 781.19 | 779.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:15:00 | 783.30 | 781.19 | 779.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:45:00 | 783.05 | 781.59 | 780.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:00:00 | 784.30 | 782.98 | 781.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:45:00 | 783.10 | 783.22 | 781.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 786.95 | 788.03 | 785.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 786.20 | 788.03 | 785.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 786.00 | 787.62 | 785.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 788.45 | 787.62 | 785.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 787.15 | 787.10 | 785.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 784.00 | 785.83 | 785.51 | SL hit (close<static) qty=1.00 sl=784.65 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 784.20 | 785.31 | 785.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 778.25 | 783.90 | 784.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 11:15:00 | 772.80 | 772.15 | 774.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:00:00 | 772.80 | 772.15 | 774.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 774.20 | 772.89 | 774.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 774.00 | 772.89 | 774.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 772.20 | 772.75 | 774.24 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 776.75 | 774.86 | 774.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 777.85 | 775.72 | 775.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 775.60 | 776.44 | 775.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 775.60 | 776.44 | 775.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 775.60 | 776.44 | 775.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 775.60 | 776.44 | 775.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 776.10 | 776.37 | 775.78 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 768.20 | 774.41 | 775.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 766.15 | 772.76 | 774.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 769.00 | 767.75 | 770.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 11:00:00 | 769.00 | 767.75 | 770.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 770.90 | 768.69 | 770.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 770.90 | 768.69 | 770.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 771.35 | 769.23 | 770.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 769.50 | 769.23 | 770.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 767.50 | 768.88 | 769.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 765.50 | 768.20 | 769.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 727.23 | 736.33 | 743.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 735.25 | 734.50 | 740.44 | SL hit (close>ema200) qty=0.50 sl=734.50 alert=retest2 |

### Cycle 83 — BUY (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 09:15:00 | 732.95 | 726.86 | 726.47 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 721.70 | 726.36 | 726.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 718.90 | 722.36 | 724.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 722.55 | 721.52 | 723.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 722.55 | 721.52 | 723.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 722.55 | 721.52 | 723.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 722.55 | 721.52 | 723.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 721.50 | 721.52 | 723.00 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 725.50 | 723.77 | 723.58 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 721.20 | 723.32 | 723.54 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 724.80 | 723.59 | 723.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 727.35 | 724.34 | 723.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 724.25 | 726.66 | 725.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 14:15:00 | 724.25 | 726.66 | 725.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 724.25 | 726.66 | 725.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 724.25 | 726.66 | 725.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 724.80 | 726.28 | 725.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 725.70 | 726.28 | 725.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 721.10 | 725.78 | 725.57 | SL hit (close<static) qty=1.00 sl=723.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 724.90 | 725.39 | 725.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 14:15:00 | 721.15 | 724.54 | 725.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 724.25 | 724.10 | 724.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 724.25 | 724.10 | 724.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 724.25 | 724.10 | 724.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 724.25 | 724.10 | 724.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 726.35 | 724.55 | 724.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 725.00 | 724.55 | 724.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 724.60 | 724.56 | 724.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 725.75 | 724.56 | 724.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 725.20 | 724.69 | 724.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 725.70 | 724.69 | 724.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 724.00 | 724.55 | 724.80 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 727.00 | 725.31 | 725.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 731.65 | 726.58 | 725.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 730.05 | 730.91 | 729.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:30:00 | 730.35 | 730.91 | 729.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 728.20 | 730.37 | 729.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 727.95 | 730.37 | 729.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 726.10 | 729.51 | 728.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 726.10 | 729.51 | 728.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 725.30 | 728.67 | 728.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 722.75 | 728.67 | 728.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 722.00 | 727.34 | 727.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 717.15 | 721.08 | 722.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 699.00 | 697.64 | 704.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 699.00 | 697.64 | 704.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 703.35 | 700.54 | 703.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 703.35 | 700.54 | 703.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 705.75 | 701.58 | 703.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 701.50 | 701.58 | 703.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 704.55 | 702.17 | 703.65 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 710.05 | 704.81 | 704.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 712.10 | 706.27 | 705.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 709.85 | 710.26 | 708.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 709.85 | 710.26 | 708.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 711.45 | 712.16 | 710.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 711.90 | 712.16 | 710.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 710.50 | 711.91 | 710.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 710.50 | 711.91 | 710.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 711.80 | 711.88 | 710.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 714.40 | 711.88 | 710.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 711.20 | 711.75 | 710.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:15:00 | 712.30 | 711.75 | 710.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 711.20 | 711.64 | 710.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:30:00 | 712.90 | 711.27 | 710.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 713.00 | 714.22 | 713.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 713.50 | 713.72 | 712.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:30:00 | 713.05 | 712.80 | 712.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 734.40 | 735.49 | 733.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 727.90 | 732.47 | 732.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 727.90 | 732.47 | 732.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 723.80 | 730.01 | 731.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 719.75 | 719.21 | 722.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:00:00 | 719.75 | 719.21 | 722.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 702.60 | 701.08 | 704.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 700.05 | 700.72 | 703.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 708.10 | 704.32 | 704.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 708.10 | 704.32 | 704.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 711.40 | 708.97 | 707.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 708.95 | 713.09 | 710.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 708.95 | 713.09 | 710.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 708.95 | 713.09 | 710.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 708.95 | 713.09 | 710.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 704.35 | 711.34 | 710.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 704.35 | 711.34 | 710.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 706.35 | 708.94 | 709.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 701.65 | 707.48 | 708.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 706.20 | 706.01 | 707.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 706.20 | 706.01 | 707.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 709.00 | 706.61 | 707.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 709.00 | 706.61 | 707.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 708.80 | 707.05 | 707.75 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 710.50 | 708.20 | 708.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 714.00 | 709.36 | 708.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 707.75 | 712.41 | 711.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 707.75 | 712.41 | 711.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 707.75 | 712.41 | 711.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 709.35 | 712.41 | 711.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 707.35 | 711.40 | 710.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 707.35 | 711.40 | 710.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 708.85 | 710.32 | 710.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 708.05 | 709.80 | 710.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 710.90 | 706.36 | 707.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 710.90 | 706.36 | 707.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 710.90 | 706.36 | 707.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 710.90 | 706.36 | 707.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 713.65 | 707.82 | 708.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 713.65 | 707.82 | 708.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 713.20 | 708.89 | 708.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 715.60 | 710.24 | 709.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 717.50 | 717.69 | 715.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:45:00 | 717.60 | 717.69 | 715.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 720.95 | 722.69 | 720.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 720.95 | 722.69 | 720.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 722.05 | 722.37 | 720.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 722.00 | 722.37 | 720.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 716.50 | 721.20 | 720.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 716.50 | 721.20 | 720.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 720.60 | 721.08 | 720.13 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 717.55 | 719.76 | 719.76 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 723.00 | 719.89 | 719.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 10:15:00 | 726.30 | 721.17 | 720.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 11:15:00 | 720.75 | 721.09 | 720.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 11:15:00 | 720.75 | 721.09 | 720.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 720.75 | 721.09 | 720.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 720.75 | 721.09 | 720.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 722.30 | 721.33 | 720.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:30:00 | 720.30 | 721.33 | 720.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 717.60 | 720.58 | 720.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 717.60 | 720.58 | 720.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 716.05 | 719.68 | 719.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 15:15:00 | 714.60 | 718.66 | 719.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 12:15:00 | 721.25 | 718.87 | 719.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 12:15:00 | 721.25 | 718.87 | 719.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 721.25 | 718.87 | 719.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 721.25 | 718.87 | 719.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 722.85 | 719.66 | 719.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 724.20 | 720.57 | 719.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 14:15:00 | 722.00 | 722.77 | 721.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 722.00 | 722.77 | 721.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 722.00 | 722.77 | 721.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 721.65 | 722.77 | 721.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 724.50 | 723.12 | 721.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 726.05 | 723.12 | 721.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 728.70 | 724.23 | 722.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 730.30 | 724.23 | 722.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:00:00 | 729.35 | 729.44 | 726.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:45:00 | 728.85 | 728.93 | 726.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 729.00 | 728.87 | 726.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 728.00 | 728.79 | 727.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 728.00 | 728.79 | 727.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 728.20 | 728.93 | 727.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 723.45 | 726.99 | 727.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 723.45 | 726.99 | 727.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 717.30 | 725.05 | 726.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 723.80 | 722.20 | 723.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 13:15:00 | 723.80 | 722.20 | 723.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 723.80 | 722.20 | 723.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 723.80 | 722.20 | 723.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 723.50 | 722.46 | 723.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:30:00 | 724.70 | 722.46 | 723.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 724.50 | 722.87 | 723.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 722.75 | 722.87 | 723.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 722.35 | 722.77 | 723.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 720.70 | 722.30 | 723.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 711.85 | 707.40 | 707.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 711.85 | 707.40 | 707.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 714.80 | 708.88 | 707.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 709.40 | 713.43 | 711.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 709.40 | 713.43 | 711.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 709.40 | 713.43 | 711.40 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 703.65 | 709.48 | 710.18 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 714.00 | 709.11 | 709.00 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 704.25 | 709.18 | 709.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 695.80 | 703.06 | 704.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 683.70 | 682.05 | 686.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 683.70 | 682.05 | 686.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 687.70 | 683.73 | 686.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:30:00 | 687.70 | 683.73 | 686.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 687.70 | 684.53 | 686.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 687.70 | 684.53 | 686.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 692.35 | 687.97 | 687.80 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 687.40 | 688.04 | 688.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 685.00 | 687.07 | 687.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 686.90 | 686.81 | 687.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 686.90 | 686.81 | 687.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 689.65 | 687.34 | 687.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:30:00 | 689.00 | 687.57 | 687.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 672.65 | 669.87 | 669.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 672.65 | 669.87 | 669.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 675.00 | 671.95 | 670.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 671.25 | 672.07 | 671.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 671.25 | 672.07 | 671.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 671.25 | 672.07 | 671.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 671.35 | 672.07 | 671.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 672.85 | 672.51 | 671.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:30:00 | 672.10 | 672.51 | 671.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 672.65 | 672.54 | 671.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 671.60 | 672.54 | 671.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 671.60 | 672.35 | 671.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 671.60 | 672.35 | 671.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 671.35 | 672.15 | 671.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 671.35 | 672.15 | 671.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 670.30 | 671.78 | 671.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 670.30 | 671.78 | 671.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 670.00 | 671.43 | 671.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 669.00 | 670.87 | 671.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 666.80 | 666.76 | 668.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 666.50 | 666.76 | 668.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 668.65 | 665.89 | 667.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 669.05 | 665.89 | 667.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 668.25 | 666.36 | 667.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 668.60 | 666.36 | 667.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 667.95 | 666.78 | 667.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 668.60 | 666.78 | 667.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 671.95 | 667.82 | 667.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 674.30 | 669.11 | 668.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 13:15:00 | 680.90 | 681.53 | 677.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:30:00 | 681.15 | 681.53 | 677.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 681.00 | 681.43 | 678.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 680.00 | 681.43 | 678.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 681.25 | 680.84 | 678.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 682.10 | 681.09 | 679.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 685.20 | 680.45 | 679.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 680.10 | 691.20 | 692.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 680.10 | 691.20 | 692.40 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 694.50 | 687.61 | 686.82 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 680.95 | 687.39 | 687.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 677.00 | 685.31 | 686.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 673.10 | 672.02 | 675.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 673.10 | 672.02 | 675.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 629.70 | 629.78 | 633.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 631.60 | 629.78 | 633.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 630.00 | 630.20 | 632.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 628.90 | 630.20 | 632.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 627.25 | 629.27 | 631.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 629.10 | 629.38 | 631.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 624.25 | 629.69 | 630.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 626.95 | 629.14 | 630.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 621.20 | 626.84 | 629.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 620.70 | 625.61 | 628.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 621.25 | 617.27 | 619.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 620.90 | 619.96 | 620.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 623.25 | 620.62 | 620.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 623.25 | 620.62 | 620.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 628.90 | 622.28 | 621.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 628.35 | 628.52 | 625.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 13:00:00 | 628.35 | 628.52 | 625.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 618.65 | 626.55 | 624.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 618.65 | 626.55 | 624.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 618.55 | 624.95 | 624.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:15:00 | 616.60 | 624.95 | 624.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 616.60 | 623.28 | 623.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 613.75 | 621.37 | 622.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 611.00 | 610.95 | 615.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 617.90 | 610.95 | 615.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 612.15 | 611.19 | 615.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:30:00 | 610.65 | 611.76 | 615.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 625.40 | 616.98 | 616.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 625.40 | 616.98 | 616.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 627.00 | 618.98 | 617.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 617.10 | 618.61 | 617.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 617.10 | 618.61 | 617.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 617.10 | 618.61 | 617.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 617.10 | 618.61 | 617.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 615.55 | 617.99 | 617.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:00:00 | 615.55 | 617.99 | 617.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 615.05 | 617.41 | 617.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 614.65 | 617.41 | 617.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 615.20 | 616.96 | 617.11 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 620.75 | 617.39 | 617.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 623.90 | 618.92 | 617.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 617.10 | 622.40 | 620.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 617.10 | 622.40 | 620.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 617.10 | 622.40 | 620.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 617.10 | 622.40 | 620.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 615.10 | 620.94 | 619.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 615.80 | 620.94 | 619.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 609.60 | 617.73 | 618.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 603.80 | 614.94 | 617.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 611.45 | 606.84 | 611.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 611.45 | 606.84 | 611.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 611.45 | 606.84 | 611.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 611.45 | 606.84 | 611.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 612.45 | 607.96 | 611.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 616.45 | 607.96 | 611.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 612.70 | 608.91 | 611.40 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 614.35 | 612.83 | 612.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 619.00 | 614.25 | 613.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 618.75 | 619.10 | 616.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 618.75 | 619.10 | 616.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 618.75 | 619.10 | 616.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 618.75 | 619.10 | 616.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 620.40 | 619.31 | 617.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:15:00 | 621.25 | 619.31 | 617.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 621.80 | 620.23 | 618.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 614.70 | 619.25 | 618.09 | SL hit (close<static) qty=1.00 sl=617.25 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 616.25 | 617.35 | 617.44 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 620.20 | 617.92 | 617.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 623.20 | 619.31 | 618.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 627.00 | 631.71 | 627.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 627.00 | 631.71 | 627.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 627.00 | 631.71 | 627.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 627.00 | 631.71 | 627.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 627.95 | 630.96 | 627.97 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 623.25 | 627.28 | 627.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 622.00 | 626.23 | 626.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 627.60 | 624.86 | 625.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 10:15:00 | 627.60 | 624.86 | 625.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 627.60 | 624.86 | 625.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:30:00 | 631.40 | 624.86 | 625.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 626.60 | 625.21 | 625.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 12:45:00 | 622.95 | 624.97 | 625.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 623.65 | 616.70 | 618.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 624.25 | 620.09 | 619.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 624.25 | 620.09 | 619.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 632.05 | 623.61 | 621.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 628.00 | 629.27 | 626.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 628.00 | 629.27 | 626.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 628.00 | 629.27 | 626.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 633.20 | 629.27 | 626.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 639.00 | 631.21 | 627.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 643.85 | 633.11 | 628.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:15:00 | 646.65 | 633.11 | 628.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 641.15 | 640.38 | 635.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 626.95 | 635.62 | 636.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 626.95 | 635.62 | 636.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 621.05 | 632.71 | 634.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 556.70 | 556.68 | 566.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 556.70 | 556.68 | 566.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 560.50 | 557.24 | 563.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 560.50 | 557.24 | 563.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 567.10 | 559.83 | 563.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 567.75 | 559.83 | 563.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 560.50 | 559.97 | 563.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:30:00 | 559.45 | 559.39 | 562.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 531.48 | 536.12 | 541.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 538.50 | 535.95 | 540.68 | SL hit (close>ema200) qty=0.50 sl=535.95 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 535.75 | 529.46 | 529.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 539.30 | 531.43 | 529.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 533.00 | 535.56 | 532.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 533.00 | 535.56 | 532.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 533.00 | 535.56 | 532.81 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 525.50 | 531.04 | 531.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 523.45 | 526.94 | 529.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 515.35 | 512.03 | 516.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 514.30 | 512.03 | 516.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 516.50 | 512.93 | 516.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 517.45 | 512.93 | 516.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 514.70 | 513.28 | 516.46 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 528.80 | 518.93 | 518.42 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 510.60 | 518.95 | 519.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 503.25 | 511.48 | 515.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 521.00 | 504.82 | 508.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 521.00 | 504.82 | 508.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 521.00 | 504.82 | 508.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 521.00 | 504.82 | 508.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 517.30 | 507.31 | 509.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 514.85 | 508.68 | 509.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 15:15:00 | 513.00 | 510.85 | 510.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 513.00 | 510.85 | 510.62 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 500.55 | 508.79 | 509.70 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 513.45 | 509.47 | 509.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 514.85 | 511.08 | 509.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 510.95 | 511.95 | 510.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 10:15:00 | 510.95 | 511.95 | 510.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 510.95 | 511.95 | 510.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 510.95 | 511.95 | 510.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 513.55 | 512.27 | 510.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:30:00 | 515.20 | 512.96 | 511.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:45:00 | 516.00 | 513.63 | 511.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 10:15:00 | 566.72 | 557.36 | 551.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 557.75 | 560.07 | 560.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 554.75 | 558.28 | 559.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 544.10 | 543.78 | 547.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:00:00 | 544.10 | 543.78 | 547.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 552.10 | 546.08 | 547.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 551.95 | 546.08 | 547.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 546.95 | 546.26 | 547.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 545.20 | 546.26 | 547.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 551.05 | 546.81 | 546.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 551.05 | 546.81 | 546.65 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 539.40 | 546.62 | 546.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 538.45 | 544.98 | 546.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 550.40 | 543.85 | 544.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 550.40 | 543.85 | 544.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 550.40 | 543.85 | 544.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 551.50 | 543.85 | 544.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 555.20 | 546.12 | 545.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 555.90 | 550.52 | 548.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 566.65 | 566.71 | 561.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:45:00 | 566.80 | 566.71 | 561.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 567.25 | 569.34 | 566.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 567.50 | 569.34 | 566.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 566.45 | 568.76 | 566.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 566.45 | 568.76 | 566.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 565.90 | 568.19 | 566.82 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 09:15:00 | 1116.00 | 2024-05-27 11:15:00 | 1106.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-05-27 10:00:00 | 1113.65 | 2024-05-27 11:15:00 | 1106.10 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-05-27 10:45:00 | 1110.50 | 2024-05-27 11:15:00 | 1106.10 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-06-06 12:00:00 | 968.15 | 2024-06-06 15:15:00 | 975.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-06-06 13:00:00 | 970.25 | 2024-06-06 15:15:00 | 975.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-06-06 14:15:00 | 970.00 | 2024-06-06 15:15:00 | 975.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-06-14 09:45:00 | 1023.90 | 2024-06-19 13:15:00 | 1017.80 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-06-14 12:45:00 | 1020.45 | 2024-06-19 13:15:00 | 1017.80 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-06-18 09:15:00 | 1024.05 | 2024-06-19 13:15:00 | 1017.80 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-06-19 09:45:00 | 1020.95 | 2024-06-19 13:15:00 | 1017.80 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-06-20 14:15:00 | 1013.75 | 2024-06-21 09:15:00 | 1037.70 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-06-27 13:30:00 | 992.55 | 2024-06-28 09:15:00 | 1002.75 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-06-28 12:00:00 | 991.30 | 2024-07-02 09:15:00 | 1008.25 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-07-01 12:15:00 | 992.75 | 2024-07-02 09:15:00 | 1008.25 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-07-01 14:15:00 | 993.20 | 2024-07-02 09:15:00 | 1008.25 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-07-05 10:30:00 | 1021.35 | 2024-07-10 10:15:00 | 1012.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-07-05 11:15:00 | 1021.00 | 2024-07-10 10:15:00 | 1012.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-05 12:45:00 | 1023.65 | 2024-07-10 10:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-08-01 09:15:00 | 990.00 | 2024-08-01 11:15:00 | 981.65 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-08-01 09:45:00 | 992.90 | 2024-08-01 11:15:00 | 981.65 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest1 | 2024-08-06 14:30:00 | 925.30 | 2024-08-08 11:15:00 | 937.75 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2024-08-06 15:00:00 | 920.45 | 2024-08-08 11:15:00 | 937.75 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest1 | 2024-08-07 09:45:00 | 923.50 | 2024-08-08 11:15:00 | 937.75 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2024-08-07 12:45:00 | 924.80 | 2024-08-08 11:15:00 | 937.75 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-08-09 10:15:00 | 928.20 | 2024-08-16 15:15:00 | 923.00 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-08-09 11:00:00 | 929.50 | 2024-08-16 15:15:00 | 923.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2024-08-09 12:00:00 | 929.10 | 2024-08-16 15:15:00 | 923.00 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2024-08-09 14:15:00 | 928.20 | 2024-08-16 15:15:00 | 923.00 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-08-12 13:45:00 | 926.90 | 2024-08-16 15:15:00 | 923.00 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2024-08-20 12:45:00 | 930.95 | 2024-08-23 12:15:00 | 929.40 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-08-21 10:00:00 | 930.95 | 2024-08-23 12:15:00 | 929.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-08-27 10:15:00 | 925.80 | 2024-08-28 09:15:00 | 934.05 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-09-11 09:15:00 | 927.50 | 2024-09-13 09:15:00 | 945.85 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-09-11 13:45:00 | 925.50 | 2024-09-13 09:15:00 | 945.85 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-09-12 13:30:00 | 926.95 | 2024-09-13 09:15:00 | 945.85 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-10-15 10:00:00 | 893.90 | 2024-10-17 10:15:00 | 885.95 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-10-15 11:45:00 | 894.20 | 2024-10-17 10:15:00 | 885.95 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-10-15 12:45:00 | 894.05 | 2024-10-17 10:15:00 | 885.95 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-10-15 13:30:00 | 894.10 | 2024-10-17 10:15:00 | 885.95 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-10-21 12:45:00 | 867.10 | 2024-10-23 09:15:00 | 823.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 13:30:00 | 867.35 | 2024-10-23 09:15:00 | 823.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 867.45 | 2024-10-23 09:15:00 | 824.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:30:00 | 867.60 | 2024-10-23 09:15:00 | 824.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:45:00 | 867.10 | 2024-10-24 09:15:00 | 839.30 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2024-10-21 13:30:00 | 867.35 | 2024-10-24 09:15:00 | 839.30 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2024-10-21 14:00:00 | 867.45 | 2024-10-24 09:15:00 | 839.30 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2024-10-21 14:30:00 | 867.60 | 2024-10-24 09:15:00 | 839.30 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2024-10-28 15:00:00 | 821.40 | 2024-10-30 09:15:00 | 839.10 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-10-29 09:30:00 | 818.95 | 2024-10-30 09:15:00 | 839.10 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-11-05 12:15:00 | 814.35 | 2024-11-05 14:15:00 | 828.85 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-11-11 13:15:00 | 837.80 | 2024-11-14 09:15:00 | 795.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:15:00 | 837.80 | 2024-11-18 12:15:00 | 802.35 | STOP_HIT | 0.50 | 4.23% |
| BUY | retest2 | 2024-11-27 10:30:00 | 829.80 | 2024-11-28 15:15:00 | 815.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-12-02 09:15:00 | 811.90 | 2024-12-02 13:15:00 | 818.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-12-02 10:30:00 | 813.60 | 2024-12-02 13:15:00 | 818.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-12-09 09:15:00 | 836.75 | 2024-12-13 09:15:00 | 828.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-26 10:15:00 | 781.55 | 2024-12-31 14:15:00 | 787.65 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-12-26 10:45:00 | 781.80 | 2024-12-31 14:15:00 | 787.65 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-12-26 11:30:00 | 781.55 | 2024-12-31 14:15:00 | 787.65 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-12-26 13:45:00 | 781.15 | 2024-12-31 14:15:00 | 787.65 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-01-02 13:00:00 | 790.75 | 2025-01-06 10:15:00 | 778.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-01-06 09:30:00 | 791.15 | 2025-01-06 10:15:00 | 778.50 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-01-14 12:15:00 | 754.80 | 2025-01-16 09:15:00 | 766.45 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-01-15 14:15:00 | 756.30 | 2025-01-16 09:15:00 | 766.45 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-02-10 14:00:00 | 772.90 | 2025-02-12 09:15:00 | 734.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 14:00:00 | 772.90 | 2025-02-12 11:15:00 | 760.15 | STOP_HIT | 0.50 | 1.65% |
| BUY | retest2 | 2025-02-21 14:45:00 | 732.00 | 2025-02-24 09:15:00 | 722.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-03-12 11:30:00 | 689.75 | 2025-03-12 14:15:00 | 698.35 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-04-04 11:15:00 | 724.05 | 2025-04-04 12:15:00 | 716.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-04-08 10:30:00 | 705.60 | 2025-04-08 15:15:00 | 715.05 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-04-30 13:45:00 | 752.45 | 2025-05-05 13:15:00 | 765.35 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-05-02 11:15:00 | 753.65 | 2025-05-05 13:15:00 | 765.35 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-05-08 13:15:00 | 736.80 | 2025-05-12 10:15:00 | 750.95 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-23 14:30:00 | 783.65 | 2025-05-26 09:15:00 | 788.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-05-27 15:15:00 | 793.80 | 2025-05-28 09:15:00 | 783.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-06-04 11:15:00 | 776.05 | 2025-06-11 15:15:00 | 782.40 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-06-17 10:45:00 | 767.30 | 2025-06-17 12:15:00 | 773.45 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-17 15:15:00 | 767.00 | 2025-06-24 09:15:00 | 767.45 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-06-18 10:15:00 | 767.20 | 2025-06-24 09:15:00 | 767.45 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-06-26 12:30:00 | 768.50 | 2025-07-01 13:15:00 | 774.80 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-07-07 13:15:00 | 783.30 | 2025-07-10 13:15:00 | 784.00 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-07-07 13:45:00 | 783.05 | 2025-07-10 13:15:00 | 784.00 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-07-08 10:00:00 | 784.30 | 2025-07-10 15:15:00 | 784.20 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-07-08 11:45:00 | 783.10 | 2025-07-10 15:15:00 | 784.20 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-07-10 09:15:00 | 788.45 | 2025-07-10 15:15:00 | 784.20 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-10 10:30:00 | 787.15 | 2025-07-10 15:15:00 | 784.20 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-07-22 11:00:00 | 765.50 | 2025-07-29 09:15:00 | 727.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:00:00 | 765.50 | 2025-07-29 13:15:00 | 735.25 | STOP_HIT | 0.50 | 3.95% |
| BUY | retest2 | 2025-08-18 09:15:00 | 725.70 | 2025-08-18 11:15:00 | 721.10 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-09-05 13:30:00 | 712.90 | 2025-09-22 14:15:00 | 727.90 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2025-09-08 15:15:00 | 713.00 | 2025-09-22 14:15:00 | 727.90 | STOP_HIT | 1.00 | 2.09% |
| BUY | retest2 | 2025-09-09 09:30:00 | 713.50 | 2025-09-22 14:15:00 | 727.90 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2025-09-09 14:30:00 | 713.05 | 2025-09-22 14:15:00 | 727.90 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2025-10-01 10:45:00 | 700.05 | 2025-10-03 09:15:00 | 708.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-29 10:15:00 | 730.30 | 2025-10-31 13:15:00 | 723.45 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-10-30 10:00:00 | 729.35 | 2025-10-31 13:15:00 | 723.45 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-30 10:45:00 | 728.85 | 2025-10-31 13:15:00 | 723.45 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-30 11:30:00 | 729.00 | 2025-10-31 13:15:00 | 723.45 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-04 10:30:00 | 720.70 | 2025-11-11 15:15:00 | 711.85 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2025-12-01 10:30:00 | 689.00 | 2025-12-12 09:15:00 | 672.65 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2025-12-24 12:00:00 | 682.10 | 2025-12-30 11:15:00 | 680.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-12-26 09:15:00 | 685.20 | 2025-12-30 11:15:00 | 680.10 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-16 13:15:00 | 628.90 | 2026-01-22 13:15:00 | 623.25 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2026-01-16 15:00:00 | 627.25 | 2026-01-22 13:15:00 | 623.25 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2026-01-19 10:15:00 | 629.10 | 2026-01-22 13:15:00 | 623.25 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2026-01-20 09:15:00 | 624.25 | 2026-01-22 13:15:00 | 623.25 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2026-01-20 12:00:00 | 621.20 | 2026-01-22 13:15:00 | 623.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-01-20 13:00:00 | 620.70 | 2026-01-22 13:15:00 | 623.25 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-01-22 10:45:00 | 621.25 | 2026-01-22 13:15:00 | 623.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-01-22 13:15:00 | 620.90 | 2026-01-22 13:15:00 | 623.25 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-01-28 10:30:00 | 610.65 | 2026-01-28 14:15:00 | 625.40 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-02-05 13:15:00 | 621.25 | 2026-02-06 09:15:00 | 614.70 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-05 14:45:00 | 621.80 | 2026-02-06 09:15:00 | 614.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-13 12:45:00 | 622.95 | 2026-02-18 11:15:00 | 624.25 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2026-02-17 10:15:00 | 623.65 | 2026-02-18 11:15:00 | 624.25 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-02-20 10:30:00 | 643.85 | 2026-02-24 12:15:00 | 626.95 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2026-02-20 11:15:00 | 646.65 | 2026-02-24 12:15:00 | 626.95 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-02-23 10:45:00 | 641.15 | 2026-02-24 12:15:00 | 626.95 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-03-06 11:30:00 | 559.45 | 2026-03-12 09:15:00 | 531.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 11:30:00 | 559.45 | 2026-03-12 11:15:00 | 538.50 | STOP_HIT | 0.50 | 3.74% |
| SELL | retest2 | 2026-04-01 11:30:00 | 514.85 | 2026-04-01 15:15:00 | 513.00 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2026-04-07 13:30:00 | 515.20 | 2026-04-17 10:15:00 | 566.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 14:45:00 | 516.00 | 2026-04-17 10:15:00 | 567.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 11:15:00 | 545.20 | 2026-04-29 12:15:00 | 551.05 | STOP_HIT | 1.00 | -1.07% |
