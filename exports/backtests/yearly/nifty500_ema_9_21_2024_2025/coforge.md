# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1365.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 124 |
| ALERT1 | 89 |
| ALERT2 | 86 |
| ALERT2_SKIP | 43 |
| ALERT3 | 216 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 123 |
| PARTIAL | 16 |
| TARGET_HIT | 13 |
| STOP_HIT | 112 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 82
- **Target hits / Stop hits / Partials:** 13 / 112 / 16
- **Avg / median % per leg:** 0.88% / -0.93%
- **Sum % (uncompounded):** 124.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 24 | 36.4% | 6 | 60 | 0 | 0.52% | 34.6% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.56% | 1.6% |
| BUY @ 3rd Alert (retest2) | 65 | 23 | 35.4% | 6 | 59 | 0 | 0.51% | 33.1% |
| SELL (all) | 75 | 35 | 46.7% | 7 | 52 | 16 | 1.19% | 89.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.52% | -1.5% |
| SELL @ 3rd Alert (retest2) | 74 | 35 | 47.3% | 7 | 51 | 16 | 1.23% | 91.1% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.02% | 0.0% |
| retest2 (combined) | 139 | 58 | 41.7% | 13 | 110 | 16 | 0.89% | 124.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 15:15:00 | 1028.05 | 1030.84 | 1030.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 1004.52 | 1025.58 | 1028.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 14:15:00 | 1002.70 | 996.67 | 1001.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 14:15:00 | 1002.70 | 996.67 | 1001.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 1002.70 | 996.67 | 1001.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 15:00:00 | 1002.70 | 996.67 | 1001.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 1015.09 | 1000.35 | 1002.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 994.74 | 1000.35 | 1002.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:45:00 | 1000.00 | 999.42 | 1002.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 945.00 | 995.97 | 999.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 12:15:00 | 999.20 | 995.97 | 999.84 | SL hit (close>static) qty=0.50 sl=995.97 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1025.90 | 1003.22 | 1001.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 11:15:00 | 1033.22 | 1009.22 | 1004.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 1045.82 | 1069.69 | 1054.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 09:15:00 | 1045.82 | 1069.69 | 1054.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 1045.82 | 1069.69 | 1054.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 1045.82 | 1069.69 | 1054.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 1043.65 | 1064.48 | 1053.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:45:00 | 1045.00 | 1064.48 | 1053.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 10:15:00 | 1045.20 | 1048.79 | 1049.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 12:15:00 | 1043.00 | 1047.14 | 1048.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 09:15:00 | 1044.53 | 1042.84 | 1045.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 1044.53 | 1042.84 | 1045.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 1044.53 | 1042.84 | 1045.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 13:45:00 | 1037.21 | 1040.37 | 1043.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 15:00:00 | 1036.60 | 1039.62 | 1042.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 1054.80 | 1042.08 | 1043.33 | SL hit (close>static) qty=1.00 sl=1053.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 10:15:00 | 1060.80 | 1045.82 | 1044.92 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 1041.20 | 1046.68 | 1047.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 1038.58 | 1043.28 | 1045.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 1051.38 | 1042.71 | 1044.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 11:15:00 | 1051.38 | 1042.71 | 1044.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 1051.38 | 1042.71 | 1044.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 1051.38 | 1042.71 | 1044.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 1050.99 | 1044.37 | 1044.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:30:00 | 1050.23 | 1044.37 | 1044.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 13:15:00 | 1050.25 | 1045.54 | 1045.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 1054.52 | 1047.34 | 1046.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1046.23 | 1048.34 | 1047.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 1046.23 | 1048.34 | 1047.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1046.23 | 1048.34 | 1047.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1046.00 | 1048.34 | 1047.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1049.80 | 1048.63 | 1047.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:45:00 | 1055.41 | 1051.13 | 1048.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 15:15:00 | 1070.97 | 1071.63 | 1071.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 1070.97 | 1071.63 | 1071.69 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 1072.82 | 1071.87 | 1071.79 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 13:15:00 | 1066.68 | 1071.31 | 1071.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 1065.82 | 1070.22 | 1071.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 1072.20 | 1069.81 | 1070.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 1072.20 | 1069.81 | 1070.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1072.20 | 1069.81 | 1070.77 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 1082.21 | 1072.55 | 1071.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 1087.80 | 1078.07 | 1074.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 15:15:00 | 1082.99 | 1089.89 | 1084.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 15:15:00 | 1082.99 | 1089.89 | 1084.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1082.99 | 1089.89 | 1084.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:30:00 | 1103.80 | 1093.29 | 1086.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 11:15:00 | 1146.31 | 1162.01 | 1162.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 11:15:00 | 1146.31 | 1162.01 | 1162.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 1134.75 | 1151.54 | 1156.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 1140.52 | 1124.42 | 1132.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 1140.52 | 1124.42 | 1132.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1140.52 | 1124.42 | 1132.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:15:00 | 1151.57 | 1124.42 | 1132.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 1173.34 | 1134.20 | 1136.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 1173.34 | 1134.20 | 1136.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 1189.36 | 1145.23 | 1141.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 13:15:00 | 1195.00 | 1161.62 | 1149.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 12:15:00 | 1176.79 | 1177.13 | 1164.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 13:00:00 | 1176.79 | 1177.13 | 1164.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1179.60 | 1180.00 | 1170.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 1175.00 | 1180.00 | 1170.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 1164.07 | 1176.68 | 1170.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:00:00 | 1164.07 | 1176.68 | 1170.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 1168.74 | 1175.09 | 1170.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:30:00 | 1179.59 | 1175.22 | 1171.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:30:00 | 1178.70 | 1177.44 | 1173.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 1253.80 | 1261.97 | 1262.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 09:15:00 | 1253.80 | 1261.97 | 1262.11 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 1265.08 | 1262.47 | 1262.26 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 1258.81 | 1261.69 | 1261.94 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 1268.76 | 1263.11 | 1262.56 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 1259.94 | 1263.05 | 1263.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 1251.48 | 1260.03 | 1261.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 14:15:00 | 1258.10 | 1257.12 | 1259.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 14:15:00 | 1258.10 | 1257.12 | 1259.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1258.10 | 1257.12 | 1259.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 1258.10 | 1257.12 | 1259.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1211.21 | 1188.94 | 1206.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 1211.21 | 1188.94 | 1206.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1201.06 | 1191.36 | 1206.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:45:00 | 1190.43 | 1191.48 | 1203.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 1187.57 | 1189.62 | 1201.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:30:00 | 1189.43 | 1196.09 | 1197.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:15:00 | 1190.13 | 1196.09 | 1197.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1177.90 | 1178.91 | 1186.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 1173.13 | 1178.91 | 1186.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:15:00 | 1173.51 | 1178.59 | 1185.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 1159.36 | 1176.88 | 1182.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 13:30:00 | 1175.94 | 1171.34 | 1176.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 1175.05 | 1172.08 | 1176.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 13:00:00 | 1173.38 | 1176.69 | 1177.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 12:15:00 | 1178.43 | 1173.11 | 1174.37 | SL hit (close>static) qty=1.00 sl=1177.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 1203.61 | 1179.66 | 1177.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 1217.00 | 1201.64 | 1190.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 13:15:00 | 1208.38 | 1209.66 | 1200.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 13:45:00 | 1208.80 | 1209.66 | 1200.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 1216.36 | 1218.63 | 1213.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:30:00 | 1211.91 | 1218.63 | 1213.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 1213.14 | 1217.53 | 1213.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:45:00 | 1213.50 | 1217.53 | 1213.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1214.72 | 1216.97 | 1213.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:45:00 | 1211.60 | 1216.97 | 1213.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1217.81 | 1217.14 | 1214.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 15:15:00 | 1220.40 | 1217.14 | 1214.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 14:30:00 | 1219.49 | 1221.15 | 1218.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 1203.62 | 1216.75 | 1216.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 1203.62 | 1216.75 | 1216.84 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 10:15:00 | 1222.81 | 1215.04 | 1214.69 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 1207.01 | 1213.13 | 1213.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 14:15:00 | 1205.19 | 1211.55 | 1213.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1219.17 | 1211.93 | 1212.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1219.17 | 1211.93 | 1212.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1219.17 | 1211.93 | 1212.94 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 1216.99 | 1213.90 | 1213.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 1237.72 | 1218.84 | 1215.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 14:15:00 | 1269.39 | 1271.51 | 1263.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 14:45:00 | 1267.00 | 1271.51 | 1263.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1274.42 | 1271.66 | 1265.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:45:00 | 1278.80 | 1273.68 | 1267.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 13:15:00 | 1278.00 | 1274.01 | 1267.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 1257.00 | 1273.40 | 1269.94 | SL hit (close<static) qty=1.00 sl=1262.77 alert=retest2 |

### Cycle 23 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 1259.90 | 1267.58 | 1267.70 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 10:15:00 | 1278.60 | 1269.27 | 1268.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 11:15:00 | 1287.33 | 1272.89 | 1269.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 10:15:00 | 1320.38 | 1321.11 | 1307.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-09 11:00:00 | 1320.38 | 1321.11 | 1307.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 1305.80 | 1317.79 | 1307.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 1305.80 | 1317.79 | 1307.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1298.74 | 1313.98 | 1307.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:00:00 | 1298.74 | 1313.98 | 1307.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1298.72 | 1310.93 | 1306.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 1298.72 | 1310.93 | 1306.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1397.40 | 1397.87 | 1388.65 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 1362.32 | 1389.10 | 1390.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 1357.34 | 1374.98 | 1382.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1381.11 | 1375.16 | 1380.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1381.11 | 1375.16 | 1380.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1381.11 | 1375.16 | 1380.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 1404.21 | 1375.16 | 1380.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1379.24 | 1375.98 | 1380.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 1386.49 | 1375.98 | 1380.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1382.14 | 1377.21 | 1380.78 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 1384.26 | 1381.16 | 1381.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 1386.65 | 1382.25 | 1381.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 13:15:00 | 1377.94 | 1381.39 | 1381.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 13:15:00 | 1377.94 | 1381.39 | 1381.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1377.94 | 1381.39 | 1381.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 1376.37 | 1381.39 | 1381.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1389.96 | 1383.11 | 1382.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:30:00 | 1376.51 | 1383.11 | 1382.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 1384.24 | 1385.77 | 1383.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 1384.24 | 1385.77 | 1383.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 1385.36 | 1385.69 | 1383.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:00:00 | 1385.36 | 1385.69 | 1383.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 1388.92 | 1386.34 | 1384.33 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 15:15:00 | 1374.13 | 1381.89 | 1382.67 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 1391.60 | 1384.29 | 1383.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 1399.03 | 1387.23 | 1384.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 1381.54 | 1390.53 | 1387.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 11:15:00 | 1381.54 | 1390.53 | 1387.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 1381.54 | 1390.53 | 1387.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:45:00 | 1377.57 | 1390.53 | 1387.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1373.94 | 1387.21 | 1386.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 1373.94 | 1387.21 | 1386.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 1380.50 | 1385.87 | 1385.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 1373.34 | 1382.05 | 1384.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 1375.63 | 1373.68 | 1378.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 1375.63 | 1373.68 | 1378.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1406.90 | 1380.37 | 1380.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 1422.61 | 1380.37 | 1380.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 1407.39 | 1385.78 | 1382.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 13:15:00 | 1423.45 | 1408.43 | 1399.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 1417.42 | 1419.95 | 1409.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:45:00 | 1419.04 | 1419.95 | 1409.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1408.64 | 1416.81 | 1409.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:45:00 | 1407.37 | 1416.81 | 1409.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1412.39 | 1415.93 | 1409.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:15:00 | 1405.00 | 1415.93 | 1409.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1405.00 | 1413.74 | 1409.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 1425.63 | 1413.74 | 1409.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 10:15:00 | 1450.36 | 1477.42 | 1477.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 1450.36 | 1477.42 | 1477.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1436.66 | 1469.27 | 1473.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 11:15:00 | 1449.40 | 1447.70 | 1458.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 11:45:00 | 1445.80 | 1447.70 | 1458.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1431.56 | 1439.11 | 1445.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:15:00 | 1422.72 | 1439.11 | 1445.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 14:15:00 | 1351.58 | 1396.86 | 1420.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 1496.05 | 1393.56 | 1399.44 | SL hit (close>ema200) qty=0.50 sl=1393.56 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 10:15:00 | 1501.13 | 1415.07 | 1408.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 09:15:00 | 1534.59 | 1489.64 | 1454.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 1526.00 | 1528.62 | 1496.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-25 10:00:00 | 1526.00 | 1528.62 | 1496.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1531.68 | 1536.09 | 1520.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 14:00:00 | 1540.00 | 1534.99 | 1522.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 15:00:00 | 1540.83 | 1536.16 | 1524.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 14:15:00 | 1543.15 | 1531.18 | 1526.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 09:15:00 | 1508.20 | 1545.44 | 1542.27 | SL hit (close<static) qty=1.00 sl=1518.01 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 1495.59 | 1535.47 | 1538.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 1487.04 | 1525.78 | 1533.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 1527.98 | 1519.73 | 1528.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 14:15:00 | 1527.98 | 1519.73 | 1528.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1527.98 | 1519.73 | 1528.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 1527.98 | 1519.73 | 1528.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 1520.20 | 1519.82 | 1527.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-01 18:15:00 | 1512.38 | 1520.47 | 1526.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 1552.00 | 1515.58 | 1515.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1552.00 | 1515.58 | 1515.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 1557.08 | 1523.88 | 1519.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 1550.83 | 1553.02 | 1539.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:45:00 | 1554.80 | 1553.02 | 1539.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1603.65 | 1591.09 | 1575.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:45:00 | 1607.74 | 1593.88 | 1577.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:45:00 | 1610.80 | 1597.44 | 1581.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 1609.57 | 1607.19 | 1592.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 10:15:00 | 1607.25 | 1614.69 | 1605.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1619.13 | 1615.58 | 1606.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 13:45:00 | 1623.41 | 1616.10 | 1608.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 1626.44 | 1615.54 | 1609.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 1585.14 | 1608.51 | 1609.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1585.14 | 1608.51 | 1609.18 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 1640.07 | 1608.27 | 1606.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 1655.10 | 1640.62 | 1630.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 09:15:00 | 1726.80 | 1731.40 | 1718.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:30:00 | 1724.74 | 1731.40 | 1718.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1726.14 | 1729.09 | 1722.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 1716.21 | 1729.09 | 1722.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1721.28 | 1727.52 | 1722.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:45:00 | 1730.00 | 1727.81 | 1723.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:45:00 | 1731.04 | 1729.12 | 1724.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:45:00 | 1733.90 | 1738.50 | 1734.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-19 09:15:00 | 1903.00 | 1892.96 | 1879.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 1870.25 | 1897.26 | 1897.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 10:15:00 | 1867.83 | 1875.30 | 1881.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 13:15:00 | 1884.09 | 1874.99 | 1879.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 13:15:00 | 1884.09 | 1874.99 | 1879.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 1884.09 | 1874.99 | 1879.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:00:00 | 1884.09 | 1874.99 | 1879.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1896.11 | 1879.21 | 1880.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1896.11 | 1879.21 | 1880.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 1906.16 | 1884.60 | 1883.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 1976.39 | 1915.13 | 1900.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1896.81 | 1914.74 | 1902.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 1896.81 | 1914.74 | 1902.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1896.81 | 1914.74 | 1902.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 1896.81 | 1914.74 | 1902.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1918.02 | 1915.39 | 1904.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 11:45:00 | 1927.98 | 1917.70 | 1906.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 11:00:00 | 1921.60 | 1924.23 | 1915.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 14:00:00 | 1923.94 | 1921.88 | 1916.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 11:30:00 | 1926.83 | 1928.57 | 1925.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1944.51 | 1934.83 | 1930.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-07 13:15:00 | 1925.20 | 1930.06 | 1930.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 13:15:00 | 1925.20 | 1930.06 | 1930.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 15:15:00 | 1918.54 | 1926.69 | 1928.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 1916.35 | 1902.95 | 1912.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 14:15:00 | 1916.35 | 1902.95 | 1912.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1916.35 | 1902.95 | 1912.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1916.35 | 1902.95 | 1912.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1916.99 | 1905.76 | 1913.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 1909.54 | 1905.76 | 1913.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1900.00 | 1904.61 | 1912.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1892.23 | 1902.37 | 1910.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:15:00 | 1797.62 | 1852.23 | 1872.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-15 09:15:00 | 1703.01 | 1732.89 | 1773.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 40 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 1829.37 | 1688.95 | 1688.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 1842.84 | 1719.73 | 1702.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 1779.73 | 1828.69 | 1800.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 1779.73 | 1828.69 | 1800.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1779.73 | 1828.69 | 1800.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 1779.73 | 1828.69 | 1800.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 1799.44 | 1822.84 | 1800.46 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1751.06 | 1784.71 | 1788.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1731.39 | 1774.05 | 1783.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1744.49 | 1728.47 | 1750.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1744.49 | 1728.47 | 1750.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1744.49 | 1728.47 | 1750.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1744.49 | 1728.47 | 1750.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1745.24 | 1733.72 | 1749.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 1745.15 | 1733.72 | 1749.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1725.18 | 1732.01 | 1747.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:15:00 | 1722.58 | 1738.79 | 1746.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:15:00 | 1636.45 | 1649.65 | 1674.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 1642.40 | 1629.43 | 1650.56 | SL hit (close>ema200) qty=0.50 sl=1629.43 alert=retest2 |

### Cycle 42 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 1687.43 | 1658.48 | 1657.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1703.92 | 1680.98 | 1670.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 1699.84 | 1701.09 | 1686.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 14:45:00 | 1700.00 | 1701.09 | 1686.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1707.25 | 1705.70 | 1698.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 1718.23 | 1708.62 | 1700.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 1684.61 | 1698.87 | 1699.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1684.61 | 1698.87 | 1699.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 1680.03 | 1695.10 | 1697.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 1537.45 | 1536.47 | 1558.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 1537.45 | 1536.47 | 1558.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1531.48 | 1536.01 | 1554.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 1521.09 | 1536.01 | 1554.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 1514.81 | 1529.70 | 1541.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 1562.66 | 1536.29 | 1543.39 | SL hit (close>static) qty=1.00 sl=1558.70 alert=retest2 |

### Cycle 44 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 1559.95 | 1548.82 | 1547.55 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 09:15:00 | 1531.64 | 1545.58 | 1546.31 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 1557.99 | 1547.57 | 1546.17 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1500.99 | 1538.25 | 1542.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 1489.85 | 1528.57 | 1537.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 1520.85 | 1504.63 | 1516.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 1520.85 | 1504.63 | 1516.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1520.85 | 1504.63 | 1516.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 1515.84 | 1504.63 | 1516.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 1525.51 | 1508.80 | 1517.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 1525.51 | 1508.80 | 1517.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 1523.00 | 1511.64 | 1518.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 14:15:00 | 1509.86 | 1511.64 | 1518.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 14:15:00 | 1528.90 | 1515.09 | 1519.12 | SL hit (close>static) qty=1.00 sl=1528.55 alert=retest2 |

### Cycle 48 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1567.66 | 1473.40 | 1470.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 1592.20 | 1497.16 | 1481.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 1538.31 | 1547.79 | 1523.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 11:30:00 | 1536.90 | 1547.79 | 1523.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 1533.66 | 1541.91 | 1524.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:15:00 | 1538.27 | 1531.45 | 1525.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 1515.85 | 1531.02 | 1528.50 | SL hit (close<static) qty=1.00 sl=1519.40 alert=retest2 |

### Cycle 49 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 1516.40 | 1525.86 | 1526.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 1504.40 | 1521.56 | 1524.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 1526.01 | 1502.61 | 1509.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 1526.01 | 1502.61 | 1509.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 1526.01 | 1502.61 | 1509.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 1521.72 | 1502.61 | 1509.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 1526.18 | 1507.32 | 1511.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 1523.96 | 1507.32 | 1511.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 1476.73 | 1463.72 | 1471.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:00:00 | 1476.73 | 1463.72 | 1471.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 1469.00 | 1464.77 | 1471.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 1487.21 | 1464.77 | 1471.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1509.37 | 1473.69 | 1474.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1509.37 | 1473.69 | 1474.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1510.67 | 1481.09 | 1478.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 10:15:00 | 1530.60 | 1508.47 | 1498.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 1511.58 | 1514.86 | 1506.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 1511.22 | 1514.86 | 1506.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1534.10 | 1518.70 | 1508.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:30:00 | 1539.30 | 1524.16 | 1512.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:00:00 | 1540.63 | 1536.76 | 1521.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:45:00 | 1540.10 | 1541.64 | 1528.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:30:00 | 1541.94 | 1543.43 | 1530.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1597.20 | 1601.46 | 1587.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 13:45:00 | 1621.24 | 1608.82 | 1595.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 11:15:00 | 1573.80 | 1603.71 | 1606.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 1573.80 | 1603.71 | 1606.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 1566.30 | 1596.23 | 1602.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1280.00 | 1270.67 | 1334.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1306.87 | 1270.67 | 1334.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1288.13 | 1274.16 | 1330.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1276.21 | 1277.08 | 1326.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1256.07 | 1292.27 | 1316.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:45:00 | 1283.20 | 1267.38 | 1286.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 12:15:00 | 1280.88 | 1271.42 | 1284.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 1295.49 | 1276.24 | 1285.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 13:00:00 | 1295.49 | 1276.24 | 1285.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 1263.94 | 1273.78 | 1283.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 14:15:00 | 1259.15 | 1273.78 | 1283.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 15:15:00 | 1296.00 | 1286.37 | 1285.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 15:15:00 | 1296.00 | 1286.37 | 1285.18 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 12:15:00 | 1275.40 | 1283.25 | 1284.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 09:15:00 | 1256.00 | 1276.72 | 1280.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-17 12:15:00 | 1276.20 | 1273.65 | 1278.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 12:15:00 | 1276.20 | 1273.65 | 1278.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 1276.20 | 1273.65 | 1278.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:30:00 | 1279.00 | 1273.65 | 1278.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 1287.50 | 1276.42 | 1278.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:00:00 | 1287.50 | 1276.42 | 1278.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 14:15:00 | 1305.90 | 1282.31 | 1281.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 15:15:00 | 1331.00 | 1292.05 | 1285.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 15:15:00 | 1385.00 | 1386.03 | 1361.68 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:15:00 | 1452.80 | 1386.03 | 1361.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1448.60 | 1459.95 | 1444.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1448.60 | 1459.95 | 1444.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1471.70 | 1462.30 | 1446.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 13:00:00 | 1475.40 | 1464.92 | 1449.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 11:15:00 | 1472.90 | 1471.83 | 1459.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 10:30:00 | 1479.00 | 1477.83 | 1468.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 13:15:00 | 1475.50 | 1486.25 | 1481.29 | SL hit (close<ema400) qty=1.00 sl=1481.29 alert=retest1 |

### Cycle 55 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 1459.20 | 1476.75 | 1477.58 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 1502.10 | 1481.82 | 1479.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 1528.30 | 1502.11 | 1493.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 1503.60 | 1507.45 | 1498.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 12:15:00 | 1503.60 | 1507.45 | 1498.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 1503.60 | 1507.45 | 1498.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:30:00 | 1498.50 | 1507.45 | 1498.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 1487.20 | 1503.40 | 1497.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 1487.20 | 1503.40 | 1497.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 1488.70 | 1500.46 | 1496.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 1488.70 | 1500.46 | 1496.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 1504.10 | 1496.58 | 1495.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:30:00 | 1505.70 | 1498.12 | 1496.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 1506.20 | 1498.12 | 1496.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 13:45:00 | 1508.00 | 1501.43 | 1497.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-12 14:15:00 | 1656.27 | 1612.41 | 1576.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 12:15:00 | 1658.50 | 1672.76 | 1673.84 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1682.70 | 1673.09 | 1672.71 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1666.30 | 1671.73 | 1672.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 1663.70 | 1670.13 | 1671.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 1654.50 | 1653.97 | 1660.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1643.80 | 1653.59 | 1658.81 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1668.80 | 1656.63 | 1659.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1668.80 | 1656.63 | 1659.72 | SL hit (close>ema400) qty=1.00 sl=1659.72 alert=retest1 |

### Cycle 60 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1689.70 | 1659.78 | 1658.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 1700.50 | 1692.26 | 1685.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 1693.20 | 1693.83 | 1688.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:00:00 | 1693.20 | 1693.83 | 1688.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1700.00 | 1712.14 | 1703.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 1704.20 | 1712.14 | 1703.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1709.00 | 1711.51 | 1703.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:15:00 | 1712.10 | 1711.51 | 1703.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:45:00 | 1711.90 | 1710.79 | 1704.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:30:00 | 1711.80 | 1710.67 | 1705.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 1715.70 | 1710.39 | 1706.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1713.60 | 1712.33 | 1708.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:00:00 | 1713.60 | 1712.33 | 1708.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1720.00 | 1716.19 | 1711.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:45:00 | 1709.20 | 1714.65 | 1711.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1711.90 | 1714.10 | 1711.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:15:00 | 1707.80 | 1714.10 | 1711.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1706.70 | 1712.62 | 1710.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:15:00 | 1707.30 | 1712.62 | 1710.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1703.50 | 1710.80 | 1710.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-03 14:15:00 | 1700.60 | 1707.94 | 1708.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 1700.60 | 1707.94 | 1708.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 1698.70 | 1706.09 | 1707.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 1720.00 | 1708.87 | 1709.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1720.00 | 1708.87 | 1709.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1720.00 | 1708.87 | 1709.01 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1719.50 | 1711.00 | 1709.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1746.00 | 1724.29 | 1717.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 1820.00 | 1826.30 | 1809.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:30:00 | 1821.50 | 1826.30 | 1809.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1816.50 | 1822.11 | 1812.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 1814.50 | 1822.11 | 1812.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1779.50 | 1813.09 | 1810.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1774.50 | 1813.09 | 1810.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1792.50 | 1808.97 | 1808.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 1800.00 | 1808.98 | 1808.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 12:00:00 | 1809.00 | 1808.98 | 1808.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1788.50 | 1805.68 | 1807.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1788.50 | 1805.68 | 1807.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 1783.00 | 1798.08 | 1803.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 1801.50 | 1798.67 | 1802.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 10:15:00 | 1801.50 | 1798.67 | 1802.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 1801.50 | 1798.67 | 1802.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:30:00 | 1808.00 | 1798.67 | 1802.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 1805.00 | 1799.94 | 1802.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:30:00 | 1802.50 | 1799.94 | 1802.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 1791.50 | 1798.25 | 1801.86 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1836.50 | 1808.68 | 1805.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 1846.50 | 1816.24 | 1808.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 1834.50 | 1837.40 | 1827.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 09:15:00 | 1840.50 | 1837.40 | 1827.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1832.00 | 1836.32 | 1828.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 1826.00 | 1836.32 | 1828.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1815.50 | 1832.16 | 1826.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 1815.00 | 1832.16 | 1826.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1814.00 | 1828.53 | 1825.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 1819.50 | 1828.53 | 1825.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 1807.00 | 1825.27 | 1825.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1787.00 | 1817.62 | 1821.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1805.50 | 1800.53 | 1809.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1805.50 | 1800.53 | 1809.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1805.50 | 1800.53 | 1809.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 1807.50 | 1800.53 | 1809.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1818.50 | 1804.12 | 1809.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1818.50 | 1804.12 | 1809.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1815.00 | 1806.30 | 1810.42 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1831.50 | 1813.69 | 1813.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1835.00 | 1823.49 | 1818.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1865.50 | 1874.08 | 1861.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 1865.50 | 1874.08 | 1861.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1865.50 | 1874.08 | 1861.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 1863.00 | 1874.08 | 1861.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1903.00 | 1906.77 | 1893.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1890.00 | 1906.77 | 1893.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1901.00 | 1905.62 | 1894.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:30:00 | 1918.00 | 1912.36 | 1901.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:00:00 | 1923.10 | 1923.79 | 1914.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:45:00 | 1919.50 | 1924.51 | 1919.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 1936.40 | 1947.67 | 1947.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 12:15:00 | 1936.40 | 1947.67 | 1947.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 13:15:00 | 1926.90 | 1943.52 | 1945.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1877.90 | 1874.05 | 1891.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 1877.90 | 1874.05 | 1891.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1872.00 | 1873.64 | 1890.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 1868.30 | 1873.64 | 1890.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:30:00 | 1869.60 | 1870.17 | 1882.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1891.80 | 1875.11 | 1882.88 | SL hit (close>static) qty=1.00 sl=1890.80 alert=retest2 |

### Cycle 68 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1895.60 | 1887.35 | 1887.04 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 1875.50 | 1885.76 | 1886.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1872.00 | 1880.44 | 1883.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 11:15:00 | 1874.50 | 1864.47 | 1871.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 11:15:00 | 1874.50 | 1864.47 | 1871.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1874.50 | 1864.47 | 1871.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 1874.80 | 1864.47 | 1871.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 1861.80 | 1863.94 | 1870.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 1871.00 | 1863.94 | 1870.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1873.30 | 1864.31 | 1868.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 1866.80 | 1864.31 | 1868.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1877.90 | 1867.03 | 1868.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1877.90 | 1867.03 | 1868.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1869.10 | 1867.44 | 1868.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 1866.70 | 1867.26 | 1868.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 1861.60 | 1867.50 | 1868.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 1861.00 | 1867.36 | 1868.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 1842.50 | 1861.50 | 1864.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1850.30 | 1859.26 | 1863.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 1840.90 | 1855.40 | 1860.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1760.70 | 1851.91 | 1857.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1773.37 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1768.52 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1767.95 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1750.38 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:15:00 | 1748.86 | 1821.25 | 1842.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-24 14:15:00 | 1680.03 | 1732.99 | 1785.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 70 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 1740.20 | 1726.02 | 1724.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 1743.80 | 1729.58 | 1726.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1707.40 | 1730.63 | 1728.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1707.40 | 1730.63 | 1728.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1707.40 | 1730.63 | 1728.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 1707.40 | 1730.63 | 1728.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1715.60 | 1727.62 | 1726.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 1725.50 | 1727.64 | 1726.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 1702.70 | 1726.38 | 1729.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 1702.70 | 1726.38 | 1729.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 1701.10 | 1716.48 | 1723.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 1724.10 | 1717.69 | 1722.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 1724.10 | 1717.69 | 1722.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1724.10 | 1717.69 | 1722.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1724.10 | 1717.69 | 1722.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1742.00 | 1722.55 | 1724.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1742.00 | 1722.55 | 1724.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1747.80 | 1727.60 | 1726.73 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 11:15:00 | 1721.40 | 1726.33 | 1726.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 12:15:00 | 1704.00 | 1721.86 | 1724.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1673.80 | 1661.99 | 1683.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:00:00 | 1673.80 | 1661.99 | 1683.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1681.60 | 1665.58 | 1681.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 1681.60 | 1665.58 | 1681.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 1673.90 | 1667.24 | 1680.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 1669.60 | 1667.24 | 1680.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 1685.90 | 1670.97 | 1680.97 | SL hit (close>static) qty=1.00 sl=1684.70 alert=retest2 |

### Cycle 74 — BUY (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 11:15:00 | 1640.10 | 1628.07 | 1627.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1661.00 | 1639.85 | 1633.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1646.90 | 1649.21 | 1642.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1646.90 | 1649.21 | 1642.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1646.90 | 1649.21 | 1642.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 1647.20 | 1649.21 | 1642.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1651.40 | 1651.06 | 1646.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 1663.70 | 1654.37 | 1648.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 10:15:00 | 1730.10 | 1743.84 | 1744.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 1730.10 | 1743.84 | 1744.50 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1754.70 | 1743.73 | 1743.00 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 1734.10 | 1741.37 | 1742.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 1722.20 | 1737.53 | 1740.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1751.20 | 1738.42 | 1740.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1751.20 | 1738.42 | 1740.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1751.20 | 1738.42 | 1740.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 1763.20 | 1738.42 | 1740.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1759.00 | 1742.54 | 1741.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 1763.30 | 1746.69 | 1743.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1754.20 | 1763.31 | 1757.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1754.20 | 1763.31 | 1757.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1754.20 | 1763.31 | 1757.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1754.20 | 1763.31 | 1757.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1755.40 | 1761.73 | 1756.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 1757.80 | 1761.73 | 1756.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1754.00 | 1760.18 | 1756.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1753.40 | 1760.18 | 1756.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 1729.40 | 1751.12 | 1752.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 12:15:00 | 1725.10 | 1742.09 | 1748.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 10:15:00 | 1676.00 | 1669.12 | 1687.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 11:00:00 | 1676.00 | 1669.12 | 1687.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1677.50 | 1671.64 | 1685.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 1680.20 | 1671.64 | 1685.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1693.20 | 1672.17 | 1680.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 1694.80 | 1672.17 | 1680.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1696.30 | 1677.00 | 1682.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 1696.30 | 1677.00 | 1682.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 1700.00 | 1686.19 | 1685.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1754.20 | 1702.50 | 1693.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 1761.80 | 1767.86 | 1755.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 1761.80 | 1767.86 | 1755.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1743.80 | 1763.23 | 1755.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 1743.80 | 1763.23 | 1755.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1748.00 | 1760.18 | 1755.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 1746.20 | 1760.18 | 1755.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1752.50 | 1756.02 | 1754.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1750.00 | 1756.02 | 1754.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1759.00 | 1755.66 | 1754.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 1760.80 | 1755.66 | 1754.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 1765.20 | 1757.52 | 1755.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1732.60 | 1789.38 | 1793.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1732.60 | 1789.38 | 1793.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 1687.80 | 1729.17 | 1755.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 1553.00 | 1548.25 | 1572.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:45:00 | 1555.10 | 1548.25 | 1572.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1601.10 | 1558.62 | 1573.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 1601.10 | 1558.62 | 1573.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1606.30 | 1568.16 | 1576.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 1606.30 | 1568.16 | 1576.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 1591.60 | 1582.45 | 1581.61 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1563.30 | 1579.83 | 1580.64 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1586.10 | 1580.88 | 1580.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1600.10 | 1587.07 | 1583.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 12:15:00 | 1649.90 | 1650.35 | 1633.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 13:00:00 | 1649.90 | 1650.35 | 1633.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1713.00 | 1721.86 | 1712.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 1726.50 | 1714.18 | 1711.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 1690.20 | 1707.43 | 1708.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1690.20 | 1707.43 | 1708.82 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1750.00 | 1712.87 | 1710.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1758.50 | 1728.37 | 1718.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1728.20 | 1751.30 | 1743.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 1728.20 | 1751.30 | 1743.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1728.20 | 1751.30 | 1743.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 1725.00 | 1751.30 | 1743.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1734.10 | 1747.86 | 1742.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 1722.50 | 1747.86 | 1742.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1730.00 | 1740.36 | 1740.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 1726.20 | 1740.36 | 1740.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 1733.10 | 1738.91 | 1739.54 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1748.20 | 1740.59 | 1740.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1773.60 | 1749.42 | 1745.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1751.60 | 1759.03 | 1752.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1751.60 | 1759.03 | 1752.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1751.60 | 1759.03 | 1752.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1751.60 | 1759.03 | 1752.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1761.70 | 1759.57 | 1753.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1769.80 | 1759.57 | 1753.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 1764.50 | 1762.44 | 1756.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1832.50 | 1761.05 | 1757.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1793.50 | 1801.38 | 1801.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1793.50 | 1801.38 | 1801.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1775.30 | 1794.52 | 1798.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 1787.70 | 1780.78 | 1788.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 13:15:00 | 1787.70 | 1780.78 | 1788.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1787.70 | 1780.78 | 1788.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 1787.70 | 1780.78 | 1788.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1796.40 | 1783.90 | 1788.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 1796.40 | 1783.90 | 1788.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1794.90 | 1786.10 | 1789.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1775.00 | 1786.10 | 1789.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1770.80 | 1758.24 | 1758.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 09:15:00 | 1770.80 | 1758.24 | 1758.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 1779.80 | 1762.55 | 1760.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 1809.70 | 1818.33 | 1806.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 1809.70 | 1818.33 | 1806.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1810.30 | 1816.73 | 1807.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1791.30 | 1816.73 | 1807.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1778.90 | 1809.16 | 1804.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1778.90 | 1809.16 | 1804.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1782.10 | 1803.75 | 1802.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 1774.60 | 1803.75 | 1802.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1775.40 | 1798.08 | 1800.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1760.20 | 1790.50 | 1796.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 1802.60 | 1790.92 | 1795.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 1802.60 | 1790.92 | 1795.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1802.60 | 1790.92 | 1795.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1802.60 | 1790.92 | 1795.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1802.00 | 1793.14 | 1796.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1809.20 | 1793.14 | 1796.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1796.80 | 1796.48 | 1797.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 1797.40 | 1796.48 | 1797.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 1802.60 | 1797.70 | 1797.66 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1777.50 | 1795.48 | 1796.91 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 1823.00 | 1798.50 | 1795.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 1839.70 | 1806.74 | 1799.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 1849.00 | 1850.20 | 1834.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 15:00:00 | 1849.00 | 1850.20 | 1834.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1797.40 | 1839.45 | 1831.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 1797.40 | 1839.45 | 1831.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1800.60 | 1831.68 | 1829.08 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1803.30 | 1826.00 | 1826.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 1794.90 | 1808.94 | 1817.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1824.80 | 1812.11 | 1818.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1824.80 | 1812.11 | 1818.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1824.80 | 1812.11 | 1818.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 1825.50 | 1812.11 | 1818.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1818.60 | 1813.41 | 1818.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:45:00 | 1813.00 | 1812.45 | 1817.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 1832.10 | 1814.86 | 1814.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1832.10 | 1814.86 | 1814.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1855.00 | 1825.31 | 1819.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1908.40 | 1909.55 | 1890.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 1908.40 | 1909.55 | 1890.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1903.90 | 1912.07 | 1902.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 1899.30 | 1912.07 | 1902.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1902.00 | 1910.06 | 1902.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 1920.90 | 1910.06 | 1902.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 1910.90 | 1911.12 | 1905.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 1913.40 | 1911.57 | 1906.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 11:15:00 | 1923.60 | 1909.54 | 1906.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 1919.60 | 1911.55 | 1908.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1945.30 | 1913.42 | 1910.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1872.70 | 1938.07 | 1946.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1872.70 | 1938.07 | 1946.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 1857.90 | 1881.43 | 1906.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 1840.20 | 1839.70 | 1860.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 15:00:00 | 1840.20 | 1839.70 | 1860.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1852.10 | 1842.17 | 1851.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 1852.10 | 1842.17 | 1851.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 1852.20 | 1844.17 | 1851.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 1851.20 | 1844.17 | 1851.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1848.80 | 1845.10 | 1851.15 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1867.50 | 1855.65 | 1855.00 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 1850.70 | 1856.54 | 1856.78 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 14:15:00 | 1865.90 | 1858.41 | 1857.61 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 1850.20 | 1856.93 | 1857.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 1843.10 | 1854.16 | 1856.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1850.10 | 1848.87 | 1852.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 1850.00 | 1848.87 | 1852.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1844.70 | 1847.08 | 1851.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:30:00 | 1850.00 | 1847.08 | 1851.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1855.50 | 1848.77 | 1851.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 1854.00 | 1848.77 | 1851.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1853.00 | 1849.61 | 1851.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1848.00 | 1849.61 | 1851.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1844.60 | 1848.61 | 1851.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:00:00 | 1842.60 | 1848.05 | 1850.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 1842.10 | 1847.08 | 1849.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:30:00 | 1842.40 | 1847.00 | 1849.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 1840.00 | 1846.74 | 1849.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1865.70 | 1849.45 | 1849.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 1865.70 | 1849.45 | 1849.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1871.20 | 1853.80 | 1851.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1871.20 | 1853.80 | 1851.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 1873.10 | 1857.66 | 1853.75 | Break + close above crossover candle high |

### Cycle 103 — SELL (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 09:15:00 | 1797.70 | 1850.28 | 1852.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 1738.10 | 1761.99 | 1790.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 1685.00 | 1683.03 | 1713.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 1685.00 | 1683.03 | 1713.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1661.50 | 1661.31 | 1670.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 1657.50 | 1662.60 | 1668.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 1657.10 | 1662.12 | 1667.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 15:15:00 | 1655.80 | 1650.91 | 1650.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 15:15:00 | 1655.80 | 1650.91 | 1650.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 1693.90 | 1659.51 | 1654.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 1678.20 | 1685.44 | 1674.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 1678.20 | 1685.44 | 1674.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1672.00 | 1682.75 | 1674.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1672.00 | 1682.75 | 1674.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1671.20 | 1680.44 | 1673.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 1671.20 | 1680.44 | 1673.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1649.40 | 1674.23 | 1671.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 1649.40 | 1674.23 | 1671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 1648.50 | 1669.09 | 1669.49 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 1675.20 | 1669.29 | 1669.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 13:15:00 | 1682.10 | 1672.81 | 1670.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 1696.20 | 1700.36 | 1692.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:00:00 | 1696.20 | 1700.36 | 1692.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1699.80 | 1700.25 | 1693.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1723.20 | 1691.70 | 1691.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 1709.90 | 1720.35 | 1717.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 1699.80 | 1713.13 | 1714.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1699.80 | 1713.13 | 1714.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1689.10 | 1708.32 | 1712.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1668.80 | 1659.72 | 1679.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1668.80 | 1659.72 | 1679.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1714.90 | 1671.90 | 1681.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1723.10 | 1671.90 | 1681.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1702.00 | 1677.92 | 1683.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1683.60 | 1678.34 | 1683.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 1674.90 | 1665.64 | 1664.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 1674.90 | 1665.64 | 1664.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1688.00 | 1673.40 | 1668.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1664.50 | 1676.29 | 1671.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 1664.50 | 1676.29 | 1671.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1664.50 | 1676.29 | 1671.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 1664.50 | 1676.29 | 1671.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1668.50 | 1674.73 | 1671.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:45:00 | 1672.00 | 1672.82 | 1671.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 1664.30 | 1670.91 | 1670.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 1664.30 | 1670.91 | 1670.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 14:15:00 | 1653.20 | 1664.02 | 1667.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 11:15:00 | 1658.20 | 1655.35 | 1661.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1658.20 | 1655.35 | 1661.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1658.20 | 1655.35 | 1661.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1658.20 | 1655.35 | 1661.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1660.80 | 1656.44 | 1661.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 1660.80 | 1656.44 | 1661.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1682.60 | 1661.67 | 1663.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 1682.60 | 1661.67 | 1663.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1659.30 | 1661.20 | 1662.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 1648.00 | 1661.20 | 1662.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:00:00 | 1656.30 | 1658.11 | 1661.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1706.10 | 1668.58 | 1663.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1706.10 | 1668.58 | 1663.91 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 1590.40 | 1665.34 | 1671.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 11:15:00 | 1580.80 | 1648.43 | 1663.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1554.70 | 1553.44 | 1577.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 1554.70 | 1553.44 | 1577.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1557.00 | 1552.40 | 1566.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 1563.60 | 1552.40 | 1566.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1573.50 | 1557.21 | 1566.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:00:00 | 1573.50 | 1557.21 | 1566.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1573.60 | 1560.49 | 1566.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 1573.60 | 1560.49 | 1566.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1425.80 | 1388.96 | 1404.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1429.50 | 1388.96 | 1404.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1404.60 | 1392.09 | 1404.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 13:45:00 | 1395.70 | 1396.92 | 1403.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1325.91 | 1369.48 | 1376.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 1256.13 | 1286.85 | 1319.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 112 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 1167.90 | 1162.41 | 1161.89 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 1155.60 | 1161.05 | 1161.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 11:15:00 | 1140.70 | 1156.54 | 1159.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1121.80 | 1114.69 | 1126.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 1121.80 | 1114.69 | 1126.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1114.50 | 1114.65 | 1125.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:15:00 | 1108.60 | 1114.65 | 1125.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:15:00 | 1053.17 | 1073.35 | 1087.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 12:15:00 | 1074.90 | 1069.62 | 1081.91 | SL hit (close>ema200) qty=0.50 sl=1069.62 alert=retest2 |

### Cycle 114 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1137.30 | 1093.88 | 1089.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1140.00 | 1115.96 | 1101.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1112.00 | 1119.50 | 1107.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1112.00 | 1119.50 | 1107.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1112.00 | 1119.50 | 1107.16 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 1096.90 | 1101.51 | 1102.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 1084.60 | 1094.84 | 1098.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 1098.00 | 1095.47 | 1098.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 1098.00 | 1095.47 | 1098.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1098.00 | 1095.47 | 1098.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1077.90 | 1095.47 | 1098.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:45:00 | 1081.40 | 1092.58 | 1096.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 13:15:00 | 1101.80 | 1095.38 | 1096.82 | SL hit (close>static) qty=1.00 sl=1100.00 alert=retest2 |

### Cycle 116 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 1122.90 | 1102.16 | 1099.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1127.50 | 1111.62 | 1105.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 1148.10 | 1152.99 | 1138.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 1148.10 | 1152.99 | 1138.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1120.60 | 1144.34 | 1138.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 1120.60 | 1144.34 | 1138.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 1122.30 | 1132.69 | 1133.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1116.00 | 1129.35 | 1132.28 | Break + close below crossover candle low |

### Cycle 118 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 1182.20 | 1137.48 | 1135.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 1193.90 | 1169.70 | 1156.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1242.10 | 1252.96 | 1236.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 1242.10 | 1252.96 | 1236.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1227.00 | 1253.60 | 1245.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 1227.00 | 1253.60 | 1245.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1220.60 | 1247.00 | 1243.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:15:00 | 1215.80 | 1247.00 | 1243.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 12:15:00 | 1225.30 | 1238.02 | 1239.67 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1257.00 | 1235.47 | 1234.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 13:15:00 | 1285.50 | 1257.38 | 1246.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 1308.70 | 1309.36 | 1292.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:45:00 | 1304.70 | 1309.36 | 1292.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1295.70 | 1309.26 | 1297.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 1295.70 | 1309.26 | 1297.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 1303.20 | 1308.05 | 1298.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 11:15:00 | 1307.60 | 1308.05 | 1298.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:00:00 | 1307.80 | 1308.00 | 1299.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 13:15:00 | 1289.40 | 1302.76 | 1298.26 | SL hit (close<static) qty=1.00 sl=1291.00 alert=retest2 |

### Cycle 121 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 1292.20 | 1295.70 | 1296.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1240.30 | 1284.41 | 1290.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1186.90 | 1171.45 | 1199.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:45:00 | 1187.40 | 1171.45 | 1199.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1201.80 | 1183.70 | 1197.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 1201.80 | 1183.70 | 1197.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 1204.20 | 1187.80 | 1197.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 1204.20 | 1187.80 | 1197.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1202.30 | 1194.58 | 1198.71 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 15:15:00 | 1206.80 | 1201.03 | 1200.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1222.30 | 1205.28 | 1202.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1207.10 | 1208.54 | 1205.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1207.10 | 1208.54 | 1205.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1207.10 | 1208.54 | 1205.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1207.10 | 1208.54 | 1205.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1204.10 | 1207.65 | 1205.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1204.10 | 1207.65 | 1205.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1202.50 | 1206.62 | 1205.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1181.80 | 1206.62 | 1205.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1191.00 | 1201.77 | 1203.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 1172.00 | 1193.15 | 1198.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 1165.90 | 1163.58 | 1174.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 13:00:00 | 1165.90 | 1163.58 | 1174.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 124 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1264.00 | 1185.49 | 1181.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 1337.30 | 1291.04 | 1261.65 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-04 09:15:00 | 994.74 | 2024-06-04 12:15:00 | 945.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 994.74 | 2024-06-04 12:15:00 | 999.20 | STOP_HIT | 0.50 | -0.45% |
| SELL | retest2 | 2024-06-04 10:45:00 | 1000.00 | 2024-06-04 12:15:00 | 950.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:45:00 | 1000.00 | 2024-06-04 12:15:00 | 999.20 | STOP_HIT | 0.50 | 0.08% |
| SELL | retest2 | 2024-06-05 09:30:00 | 1001.03 | 2024-06-05 10:15:00 | 1025.90 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-06-12 13:45:00 | 1037.21 | 2024-06-13 09:15:00 | 1054.80 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-06-12 15:00:00 | 1036.60 | 2024-06-13 09:15:00 | 1054.80 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-06-19 11:45:00 | 1055.41 | 2024-06-25 15:15:00 | 1070.97 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2024-07-01 09:30:00 | 1103.80 | 2024-07-09 11:15:00 | 1146.31 | STOP_HIT | 1.00 | 3.85% |
| BUY | retest2 | 2024-07-16 14:30:00 | 1179.59 | 2024-07-30 09:15:00 | 1253.80 | STOP_HIT | 1.00 | 6.29% |
| BUY | retest2 | 2024-07-18 10:30:00 | 1178.70 | 2024-07-30 09:15:00 | 1253.80 | STOP_HIT | 1.00 | 6.37% |
| SELL | retest2 | 2024-08-06 12:45:00 | 1190.43 | 2024-08-14 12:15:00 | 1178.43 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2024-08-06 13:30:00 | 1187.57 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-08-08 09:30:00 | 1189.43 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-08-08 10:15:00 | 1190.13 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-08-09 10:15:00 | 1173.13 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-08-09 12:15:00 | 1173.51 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-08-12 09:15:00 | 1159.36 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2024-08-12 13:30:00 | 1175.94 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2024-08-13 13:00:00 | 1173.38 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-08-14 15:00:00 | 1173.26 | 2024-08-16 09:15:00 | 1203.61 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-08-21 15:15:00 | 1220.40 | 2024-08-23 09:15:00 | 1203.62 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-08-22 14:30:00 | 1219.49 | 2024-08-23 09:15:00 | 1203.62 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-09-03 11:45:00 | 1278.80 | 2024-09-04 09:15:00 | 1257.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-09-03 13:15:00 | 1278.00 | 2024-09-04 09:15:00 | 1257.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-10-04 09:15:00 | 1425.63 | 2024-10-16 10:15:00 | 1450.36 | STOP_HIT | 1.00 | 1.73% |
| SELL | retest2 | 2024-10-21 10:15:00 | 1422.72 | 2024-10-21 14:15:00 | 1351.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:15:00 | 1422.72 | 2024-10-23 09:15:00 | 1496.05 | STOP_HIT | 0.50 | -5.15% |
| BUY | retest2 | 2024-10-28 14:00:00 | 1540.00 | 2024-10-31 09:15:00 | 1508.20 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-10-28 15:00:00 | 1540.83 | 2024-10-31 09:15:00 | 1508.20 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-10-29 14:15:00 | 1543.15 | 2024-10-31 09:15:00 | 1508.20 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-11-01 18:15:00 | 1512.38 | 2024-11-06 09:15:00 | 1552.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-11-11 10:45:00 | 1607.74 | 2024-11-18 09:15:00 | 1585.14 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-11-11 11:45:00 | 1610.80 | 2024-11-18 09:15:00 | 1585.14 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-11-12 09:30:00 | 1609.57 | 2024-11-18 09:15:00 | 1585.14 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-11-13 10:15:00 | 1607.25 | 2024-11-18 09:15:00 | 1585.14 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-11-13 13:45:00 | 1623.41 | 2024-11-18 09:15:00 | 1585.14 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-11-14 09:15:00 | 1626.44 | 2024-11-18 09:15:00 | 1585.14 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2024-11-29 10:45:00 | 1730.00 | 2024-12-19 09:15:00 | 1903.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 11:45:00 | 1731.04 | 2024-12-19 09:15:00 | 1904.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-03 09:45:00 | 1733.90 | 2024-12-19 09:15:00 | 1907.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-31 11:45:00 | 1927.98 | 2025-01-07 13:15:00 | 1925.20 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-01-01 11:00:00 | 1921.60 | 2025-01-07 13:15:00 | 1925.20 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-01-01 14:00:00 | 1923.94 | 2025-01-07 13:15:00 | 1925.20 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-01-03 11:30:00 | 1926.83 | 2025-01-07 13:15:00 | 1925.20 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-01-09 10:45:00 | 1892.23 | 2025-01-13 10:15:00 | 1797.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 1892.23 | 2025-01-15 09:15:00 | 1703.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-30 10:15:00 | 1722.58 | 2025-02-01 11:15:00 | 1636.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 10:15:00 | 1722.58 | 2025-02-03 10:15:00 | 1642.40 | STOP_HIT | 0.50 | 4.65% |
| BUY | retest2 | 2025-02-07 10:30:00 | 1718.23 | 2025-02-10 11:15:00 | 1684.61 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1521.09 | 2025-02-19 09:15:00 | 1562.66 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1514.81 | 2025-02-19 09:15:00 | 1562.66 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-02-24 14:15:00 | 1509.86 | 2025-02-24 14:15:00 | 1528.90 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-02-25 09:15:00 | 1515.88 | 2025-02-25 14:15:00 | 1537.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-02-25 10:45:00 | 1508.13 | 2025-02-25 14:15:00 | 1537.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-02-25 15:15:00 | 1519.00 | 2025-02-28 09:15:00 | 1443.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 10:15:00 | 1503.00 | 2025-02-28 09:15:00 | 1427.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 10:45:00 | 1503.83 | 2025-02-28 09:15:00 | 1428.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 15:15:00 | 1519.00 | 2025-03-03 11:15:00 | 1464.09 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-02-27 10:15:00 | 1503.00 | 2025-03-03 11:15:00 | 1464.09 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-02-27 10:45:00 | 1503.83 | 2025-03-03 11:15:00 | 1464.09 | STOP_HIT | 0.50 | 2.64% |
| BUY | retest2 | 2025-03-07 13:15:00 | 1538.27 | 2025-03-10 10:15:00 | 1515.85 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-03-21 10:30:00 | 1539.30 | 2025-04-01 11:15:00 | 1573.80 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2025-03-21 14:00:00 | 1540.63 | 2025-04-01 11:15:00 | 1573.80 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-03-24 09:45:00 | 1540.10 | 2025-04-01 11:15:00 | 1573.80 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-03-24 10:30:00 | 1541.94 | 2025-04-01 11:15:00 | 1573.80 | STOP_HIT | 1.00 | 2.07% |
| BUY | retest2 | 2025-03-27 13:45:00 | 1621.24 | 2025-04-01 11:15:00 | 1573.80 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1276.21 | 2025-04-15 15:15:00 | 1296.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1256.07 | 2025-04-15 15:15:00 | 1296.00 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-04-11 09:45:00 | 1283.20 | 2025-04-15 15:15:00 | 1296.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-04-11 12:15:00 | 1280.88 | 2025-04-15 15:15:00 | 1296.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-04-11 14:15:00 | 1259.15 | 2025-04-15 15:15:00 | 1296.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest1 | 2025-04-23 09:15:00 | 1452.80 | 2025-04-30 13:15:00 | 1475.50 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-04-25 13:00:00 | 1475.40 | 2025-04-30 15:15:00 | 1459.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-04-28 11:15:00 | 1472.90 | 2025-04-30 15:15:00 | 1459.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-04-29 10:30:00 | 1479.00 | 2025-04-30 15:15:00 | 1459.20 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-05-07 11:30:00 | 1505.70 | 2025-05-12 14:15:00 | 1656.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 12:15:00 | 1506.20 | 2025-05-12 14:15:00 | 1656.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 13:45:00 | 1508.00 | 2025-05-12 14:15:00 | 1658.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-05-22 09:15:00 | 1643.80 | 2025-05-22 09:15:00 | 1668.80 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-05-22 12:15:00 | 1638.90 | 2025-05-23 09:15:00 | 1689.70 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-05-30 11:15:00 | 1712.10 | 2025-06-03 14:15:00 | 1700.60 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-05-30 12:45:00 | 1711.90 | 2025-06-03 14:15:00 | 1700.60 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-30 13:30:00 | 1711.80 | 2025-06-03 14:15:00 | 1700.60 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-02 10:45:00 | 1715.70 | 2025-06-03 14:15:00 | 1700.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-12 11:30:00 | 1800.00 | 2025-06-12 13:15:00 | 1788.50 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-12 12:00:00 | 1809.00 | 2025-06-12 13:15:00 | 1788.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-30 13:30:00 | 1918.00 | 2025-07-09 12:15:00 | 1936.40 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-07-01 14:00:00 | 1923.10 | 2025-07-09 12:15:00 | 1936.40 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-07-02 13:45:00 | 1919.50 | 2025-07-09 12:15:00 | 1936.40 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-07-14 11:15:00 | 1868.30 | 2025-07-15 09:15:00 | 1891.80 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-14 14:30:00 | 1869.60 | 2025-07-15 09:15:00 | 1891.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1866.70 | 2025-07-24 09:15:00 | 1773.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1861.60 | 2025-07-24 09:15:00 | 1768.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1861.00 | 2025-07-24 09:15:00 | 1767.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1842.50 | 2025-07-24 09:15:00 | 1750.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 13:15:00 | 1840.90 | 2025-07-24 09:15:00 | 1748.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1866.70 | 2025-07-24 14:15:00 | 1680.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 09:45:00 | 1861.60 | 2025-07-24 14:15:00 | 1675.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1861.00 | 2025-07-24 14:15:00 | 1674.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-24 09:15:00 | 1760.70 | 2025-07-24 14:15:00 | 1672.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1842.50 | 2025-07-25 09:15:00 | 1658.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-23 13:15:00 | 1840.90 | 2025-07-25 09:15:00 | 1656.81 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-24 09:15:00 | 1760.70 | 2025-07-28 09:15:00 | 1725.90 | STOP_HIT | 0.50 | 1.98% |
| BUY | retest2 | 2025-07-31 11:30:00 | 1725.50 | 2025-08-01 14:15:00 | 1702.70 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-07 13:15:00 | 1669.60 | 2025-08-07 13:15:00 | 1685.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1636.70 | 2025-08-14 11:15:00 | 1640.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-08-20 10:30:00 | 1663.70 | 2025-08-28 10:15:00 | 1730.10 | STOP_HIT | 1.00 | 3.99% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1760.80 | 2025-09-22 09:15:00 | 1732.60 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-16 09:45:00 | 1765.20 | 2025-09-22 09:15:00 | 1732.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-14 09:15:00 | 1726.50 | 2025-10-14 11:15:00 | 1690.20 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1769.80 | 2025-10-31 11:15:00 | 1793.50 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-10-24 11:30:00 | 1764.50 | 2025-10-31 11:15:00 | 1793.50 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1832.50 | 2025-10-31 11:15:00 | 1793.50 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1775.00 | 2025-11-11 09:15:00 | 1770.80 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1813.00 | 2025-11-25 14:15:00 | 1832.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-02 09:15:00 | 1920.90 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-12-02 13:45:00 | 1910.90 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-12-02 15:00:00 | 1913.40 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-12-03 11:15:00 | 1923.60 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-12-04 09:15:00 | 1945.30 | 2025-12-09 09:15:00 | 1872.70 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-12-19 12:00:00 | 1842.60 | 2025-12-22 10:15:00 | 1871.20 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-12-19 12:45:00 | 1842.10 | 2025-12-22 10:15:00 | 1871.20 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-12-19 13:30:00 | 1842.40 | 2025-12-22 10:15:00 | 1871.20 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-12-19 15:15:00 | 1840.00 | 2025-12-22 10:15:00 | 1871.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-01 14:15:00 | 1657.50 | 2026-01-06 15:15:00 | 1655.80 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2026-01-01 15:15:00 | 1657.10 | 2026-01-06 15:15:00 | 1655.80 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2026-01-16 09:15:00 | 1723.20 | 2026-01-20 12:15:00 | 1699.80 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-01-20 10:30:00 | 1709.90 | 2026-01-20 12:15:00 | 1699.80 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1683.60 | 2026-01-28 10:15:00 | 1674.90 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2026-01-29 13:45:00 | 1672.00 | 2026-01-30 10:15:00 | 1664.30 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-02-01 15:15:00 | 1648.00 | 2026-02-03 09:15:00 | 1706.10 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-02-02 10:00:00 | 1656.30 | 2026-02-03 09:15:00 | 1706.10 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2026-02-17 13:45:00 | 1395.70 | 2026-02-20 09:15:00 | 1325.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:45:00 | 1395.70 | 2026-02-24 09:15:00 | 1256.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 14:15:00 | 1108.60 | 2026-03-17 09:15:00 | 1053.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:15:00 | 1108.60 | 2026-03-17 12:15:00 | 1074.90 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1077.90 | 2026-03-23 13:15:00 | 1101.80 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-03-23 09:45:00 | 1081.40 | 2026-03-23 13:15:00 | 1101.80 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-04-20 11:15:00 | 1307.60 | 2026-04-20 13:15:00 | 1289.40 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-20 12:00:00 | 1307.80 | 2026-04-20 13:15:00 | 1289.40 | STOP_HIT | 1.00 | -1.41% |
